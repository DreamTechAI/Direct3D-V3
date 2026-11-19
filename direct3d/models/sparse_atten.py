import math

import flash_attn
import torch
from torch import nn


def sparse_scaled_dot_product_attention(q, kv, layout, layout_kv=None):
    device = q.device
    q_seqlen = [layout[i].stop - layout[i].start for i in range(len(layout))]
    cu_seqlens_q = (
        torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(q_seqlen), dim=0)]).int().to(device)
    )
    if layout_kv is None:
        kv_seqlen = [kv.shape[1]] * kv.shape[0]
        kv = kv.flatten(0, 1)
    else:
        kv_seqlen = [part.stop - part.start for part in layout_kv]
    cu_seqlens_kv = (
        torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(kv_seqlen), dim=0)]).int().to(device)
    )
    out = flash_attn.flash_attn_varlen_kvpacked_func(
        q, kv, cu_seqlens_q, cu_seqlens_kv, max(q_seqlen), max(kv_seqlen)
    )
    return out.reshape(q.shape[0], -1)


def calc_window_partition(
    coords: torch.Tensor, window_size: int | tuple[int, ...], shift_window: int | tuple[int, ...] = 0
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Sort sparse tokens into shifted spatial windows and record the inverse permutation."""
    DIM = coords.shape[1] - 1
    shift_window = (shift_window,) * DIM if isinstance(shift_window, int) else shift_window
    window_size = (window_size,) * DIM if isinstance(window_size, int) else window_size
    shifted_coords = coords.clone().detach()
    shifted_coords[:, 1:] += torch.tensor(shift_window, device=coords.device, dtype=torch.int32).unsqueeze(0)

    MAX_COORDS = shifted_coords[:, 1:].max(dim=0).values.tolist()
    NUM_WINDOWS = [math.ceil((mc + 1) / ws) for mc, ws in zip(MAX_COORDS, window_size)]
    OFFSET = torch.cumprod(torch.tensor([1] + NUM_WINDOWS[::-1]), dim=0).tolist()[::-1]

    shifted_coords[:, 1:] //= torch.tensor(window_size, device=coords.device, dtype=torch.int32).unsqueeze(0)
    shifted_indices = (
        shifted_coords * torch.tensor(OFFSET, device=coords.device, dtype=torch.int32).unsqueeze(0)
    ).sum(dim=1)
    fwd_indices = torch.argsort(shifted_indices)
    bwd_indices = torch.empty_like(fwd_indices)
    bwd_indices[fwd_indices] = torch.arange(fwd_indices.shape[0], device=coords.device)
    seq_lens = torch.bincount(shifted_indices)
    mask = seq_lens != 0
    seq_lens = seq_lens[mask].tolist()
    return fwd_indices, bwd_indices, seq_lens


def sparse_windowed_scaled_dot_product_self_attention(qkv, window_size, partition):
    fwd_indices, bwd_indices, seq_lens = partition
    H = qkv.shape[2]
    C = qkv.shape[3]
    qkv_feats = qkv[fwd_indices]  # [M, 3, H, C]

    if all(seq_len == window_size for seq_len in seq_lens):
        B = len(seq_lens)
        N = window_size
        qkv_feats = qkv_feats.reshape(B, N, 3, H, C)
        out = flash_attn.flash_attn_qkvpacked_func(qkv_feats)  # [B, N, H, C]
        out = out.reshape(B * N, H, C)  # [M, H, C]
    else:
        cu_seqlens = (
            torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(seq_lens), dim=0)], dim=0)
            .to(qkv.device)
            .int()
        )
        out = flash_attn.flash_attn_varlen_qkvpacked_func(qkv_feats, cu_seqlens, max(seq_lens))  # [M, H, C]

    out = out[bwd_indices]  # [T, H, C]

    return out


class SparseRotaryPositionEmbedder(nn.Module):
    def __init__(self, head_dim: int, dim: int = 3, rope_freq: tuple[float, float] = (1.0, 10000.0)):
        super().__init__()
        assert head_dim % 2 == 0, "Head dim must be divisible by 2"
        self.head_dim = head_dim
        self.dim = dim
        self.rope_freq = rope_freq
        self.freq_dim = head_dim // 2 // dim
        self.freqs = torch.arange(self.freq_dim, dtype=torch.float32, device="cpu") / self.freq_dim
        self.freqs = rope_freq[0] / (rope_freq[1] ** (self.freqs))

    def _get_phases(self, indices: torch.Tensor) -> torch.Tensor:
        self.freqs = self.freqs.to(indices.device)
        freqs = self.freqs
        phases = torch.outer(indices, freqs)
        phases = torch.polar(torch.ones_like(phases), phases)
        return phases

    def _rotary_embedding(self, x: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        x_rotated = x_complex * phases.unsqueeze(-2)
        x_embed = torch.view_as_real(x_rotated).reshape(*x_rotated.shape[:-1], -1).to(x.dtype)
        return x_embed

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q (SparseTensor): [..., N, H, D] tensor of queries
            k (SparseTensor): [..., N, H, D] tensor of keys
        """

        phases = self._get_phases(indices.reshape(-1)).reshape(*indices.shape[:-1], -1)
        if phases.shape[-1] < self.head_dim // 2:
            padn = self.head_dim // 2 - phases.shape[-1]
            phases = torch.cat(
                [
                    phases,
                    torch.polar(
                        torch.ones(*phases.shape[:-1], padn, device=phases.device),
                        torch.zeros(*phases.shape[:-1], padn, device=phases.device),
                    ),
                ],
                dim=-1,
            )

        q_embed = self._rotary_embedding(q, phases)
        k_embed = self._rotary_embedding(k, phases)
        return q_embed, k_embed


class CrossAttention(nn.Module):
    def __init__(
        self,
        qdim,
        kdim,
        num_heads,
        qkv_bias=True,
        qk_norm=False,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.qdim = qdim
        self.kdim = kdim
        self.num_heads = num_heads
        assert self.qdim % num_heads == 0, "self.qdim must be divisible by num_heads"
        self.head_dim = self.qdim // num_heads
        assert self.head_dim % 8 == 0 and self.head_dim <= 128, (
            "Only support head_dim <= 128 and divisible by 8"
        )

        self.to_q = nn.Linear(qdim, qdim, bias=qkv_bias)
        self.to_k = nn.Linear(kdim, qdim, bias=qkv_bias)
        self.to_v = nn.Linear(kdim, qdim, bias=qkv_bias)

        self.q_norm = (
            norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        )
        self.k_norm = (
            norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        )
        self.out_proj = nn.Linear(qdim, qdim, bias=True)

    def forward(self, x, y, layout, layout_y=None):
        q = self.to_q(x)
        q = q.reshape(q.shape[0], self.num_heads, self.head_dim)  # [b, s1, h, d]

        k = self.to_k(y).unflatten(-1, (self.num_heads, self.head_dim))
        v = self.to_v(y).unflatten(-1, (self.num_heads, self.head_dim))
        q = self.q_norm(q)
        k = self.k_norm(k)

        kv = torch.stack([k, v], dim=-3)
        context = sparse_scaled_dot_product_attention(q, kv, layout, layout_y)

        context = context.reshape(x.shape[0], -1)
        out = self.out_proj(context)

        return out


class Attention(nn.Module):
    """Self-attention over regular and shifted windows, using half the heads for each."""

    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=True,
        qk_norm=False,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        assert self.dim % num_heads == 0, "dim should be divisible by num_heads"
        self.head_dim = self.dim // num_heads
        # This assertion is aligned with flash attention
        assert self.head_dim % 8 == 0 and self.head_dim <= 128, (
            "Only support head_dim <= 128 and divisible by 8"
        )

        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_norm = (
            norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        )
        self.k_norm = (
            norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        )
        self.out_proj = nn.Linear(dim, dim)
        self.rope = SparseRotaryPositionEmbedder(head_dim=self.head_dim, dim=3)
        assert num_heads % 2 == 0, "Split-window attention requires an even number of heads"

    def forward(self, x, index, window_size, window_cache):
        N = len(x)

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        qkv = torch.cat((q, k, v), dim=-1)

        qkv = qkv.reshape(N, 3, self.num_heads, -1)

        q, k, v = qkv.unbind(dim=1)

        q, k = self.rope(q, k, index[..., 1:])

        q = self.q_norm(q)
        k = self.k_norm(k)

        qkv = torch.stack([q, k, v], dim=1)

        half_heads = self.num_heads // 2
        cache0, cache1 = window_cache
        x0 = sparse_windowed_scaled_dot_product_self_attention(
            qkv[:, :, :half_heads].contiguous(),
            window_size,
            cache0,
        )
        x1 = sparse_windowed_scaled_dot_product_self_attention(
            qkv[:, :, half_heads:].contiguous(),
            window_size,
            cache1,
        )
        x = torch.cat([x0, x1], dim=1)
        x = x.reshape(x.shape[0], -1)
        x = self.out_proj(x)
        return x


class MLP(nn.Module):
    def __init__(self, *, width: int):
        super().__init__()
        self.width = width
        self.fc1 = nn.Linear(width, width * 4)
        self.fc2 = nn.Linear(width * 4, width)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))
