import math

import torch
import torch.nn.functional as F
from torch import nn

from .decoder import DecoderFinalLayer, PixelAttentionDecoder
from .sparse_atten import MLP, Attention, CrossAttention, calc_window_partition


class LayerNorm(nn.LayerNorm):
    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        weight = self.weight.float() if self.weight is not None else None
        bias = self.bias.float() if self.bias is not None else None
        return F.layer_norm(x.float(), self.normalized_shape, weight, bias, self.eps).type_as(x)


class RMSNorm(nn.RMSNorm):
    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x):
        weight = self.weight.float() if self.weight is not None else None
        return F.rms_norm(x.float(), self.normalized_shape, weight, self.eps).type_as(x)


class Timesteps(nn.Module):
    def __init__(
        self, num_channels: int, downscale_freq_shift: float = 0.0, scale: int = 1, max_period: int = 10000
    ):
        super().__init__()
        self.num_channels = num_channels
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale
        self.max_period = max_period

    def forward(self, timesteps):
        assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"
        embedding_dim = self.num_channels
        half_dim = embedding_dim // 2
        exponent = -math.log(self.max_period) * torch.arange(
            start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
        )
        exponent = exponent / (half_dim - self.downscale_freq_shift)
        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = self.scale * emb
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if embedding_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, frequency_embedding_size, bias=True),
            nn.GELU(),
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
        )
        self.time_embed = Timesteps(hidden_size)

    def forward(self, t):
        t_freq = self.time_embed(t).type(self.mlp[0].weight.dtype)

        t = self.mlp(t_freq)
        t = t.unsqueeze(dim=1)
        return t


class SparseDiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        text_states_dim=1024,
        qk_norm=False,
        norm_layer=nn.LayerNorm,
        qk_norm_layer=nn.RMSNorm,
        qkv_bias=True,
        skip_connection=False,
    ):
        super().__init__()

        self.norm1 = norm_layer(hidden_size, elementwise_affine=True, eps=1e-6)
        self.attn1 = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm, norm_layer=qk_norm_layer
        )

        self.norm2 = norm_layer(hidden_size, elementwise_affine=True, eps=1e-6)

        self.attn2 = CrossAttention(
            hidden_size,
            text_states_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            norm_layer=qk_norm_layer,
        )
        self.norm3 = norm_layer(hidden_size, elementwise_affine=True, eps=1e-6)

        if skip_connection:
            self.skip_norm = norm_layer(hidden_size, elementwise_affine=True, eps=1e-6)
            self.skip_linear = nn.Linear(2 * hidden_size, hidden_size)
        else:
            self.skip_linear = None

        self.mlp = MLP(width=hidden_size)

        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(
        self,
        x,
        c=None,
        text_states=None,
        index=None,
        layout=None,
        skip_value=None,
        window_size=None,
        window_cache=None,
    ):
        if self.skip_linear is not None:
            cat = torch.cat([skip_value, x], dim=-1)
            x = self.skip_linear(cat)
            x = self.skip_norm(x)

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).squeeze(1).chunk(6, dim=1)
        )
        # Self-Attention
        x_norm = self.norm1(x)
        new_x_norm = []
        for i in range(len(layout)):
            new_x_norm.append(x_norm[layout[i]] * (1 + scale_msa[i : i + 1]) + shift_msa[i : i + 1])
        new_x_norm = torch.cat(new_x_norm, dim=0)

        attn_out = self.attn1(new_x_norm, index, window_size=window_size, window_cache=window_cache)
        new_attn_out = []
        for i in range(len(layout)):
            new_attn_out.append(attn_out[layout[i]] * gate_msa[i : i + 1])
        new_attn_out = torch.cat(new_attn_out, dim=0)
        x = x + new_attn_out

        # Cross-Attention
        x = x + self.attn2(self.norm2(x), text_states, layout)

        # FFN Layer
        mlp_inputs = self.norm3(x)

        new_mlp_inputs = []
        for i in range(len(layout)):
            new_mlp_inputs.append(mlp_inputs[layout[i]] * (1 + scale_mlp[i : i + 1]) + shift_mlp[i : i + 1])
        new_mlp_inputs = torch.cat(new_mlp_inputs, dim=0)

        mlp_out = self.mlp(new_mlp_inputs)

        new_mlp_out = []
        for i in range(len(layout)):
            new_mlp_out.append(mlp_out[layout[i]] * gate_mlp[i : i + 1])
        new_mlp_out = torch.cat(new_mlp_out, dim=0)

        x = x + new_mlp_out
        return x


class CamDecoder(nn.Module):
    """Predict camera parameters from the sparse geometry and reference image tokens."""

    def __init__(self, hidden_size, out_channels, context_dim, num_heads, qk_norm, qkv_bias):
        super().__init__()
        attn_options = dict(num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm, norm_layer=RMSNorm)
        self.cam_linear = nn.Linear(hidden_size, hidden_size)
        self.norm1 = LayerNorm(hidden_size, elementwise_affine=True)
        self.attn1 = CrossAttention(hidden_size, hidden_size, **attn_options)
        self.norm2 = LayerNorm(hidden_size, elementwise_affine=True)
        self.attn2 = CrossAttention(hidden_size, context_dim, **attn_options)
        self.norm3 = LayerNorm(hidden_size, elementwise_affine=True)
        self.out_linear = DecoderFinalLayer(hidden_size, out_channels, elementwise_affine=True)

    def forward(self, cam_token, geometry, cond, layout):
        batch_size = cond.shape[0]
        cam = self.cam_linear(cam_token.expand(batch_size, -1))
        cam_layout = [slice(i, i + 1) for i in range(batch_size)]
        cam = cam + self.attn1(self.norm1(cam), geometry, cam_layout, layout_y=layout)
        cam = cam + self.attn2(self.norm2(cam), cond, cam_layout)
        return self.out_linear(self.norm3(cam))


class SparseDiT(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        hidden_size,
        context_dim,
        depth,
        num_heads,
        patch_size,
        decoder_hidden_size,
        num_decoder_blocks,
        window_size,
        shift_window,
        time_factor,
        qk_norm,
        qkv_bias,
        cam_channels=0,
    ):
        super().__init__()
        self.depth = depth
        self.out_channels = in_channels
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.window_size = window_size
        self.shift_window = shift_window
        self.time_factor = time_factor
        self.cam_decoder = None
        if cam_channels:
            self.cam_token = nn.Parameter(torch.randn(1, hidden_size))
            self.cam_decoder = CamDecoder(hidden_size, cam_channels, context_dim, num_heads, qk_norm, qkv_bias)
        self.x_embedder = nn.Linear(in_channels * patch_size**3, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size, hidden_size * 4)
        self.blocks = nn.ModuleList(
            [
                SparseDiTBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    text_states_dim=context_dim,
                    qk_norm=qk_norm,
                    norm_layer=LayerNorm,
                    qk_norm_layer=RMSNorm,
                    qkv_bias=qkv_bias,
                    skip_connection=layer > depth // 2,
                )
                for layer in range(depth)
            ]
        )
        self.decoder = PixelAttentionDecoder(
            in_channels=in_channels,
            model_channels=decoder_hidden_size,
            out_channels=in_channels,
            z_channels=hidden_size,
            num_res_blocks=num_decoder_blocks,
            patch_size=patch_size,
            num_heads=num_heads,
            norm_layer=LayerNorm,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            qk_norm_layer=RMSNorm,
        )

    def forward(self, x, index, layout, t, cond: torch.Tensor, *, return_camera=False):
        if return_camera and self.cam_decoder is None:
            raise ValueError("Camera prediction requires a checkpoint with a configured camera head.")
        x, t = x.to(cond.dtype), t.to(cond.dtype)

        t = self.t_embedder(t * self.time_factor)
        s = self.x_embedder(x)
        c = t
        window0 = calc_window_partition(index, self.window_size, 0)
        window1 = calc_window_partition(index, self.window_size, self.shift_window)
        skip_value_list = []
        for layer, block in enumerate(self.blocks):
            skip_value = None if layer <= self.depth // 2 else skip_value_list.pop()
            s = block(
                s,
                c,
                cond,
                index,
                layout,
                skip_value=skip_value,
                window_size=self.window_size,
                window_cache=(window0, window1),
            )
            if layer < self.depth // 2:
                skip_value_list.append(s)
        x = x.reshape(-1, self.patch_size**3, self.in_channels)
        x = self.decoder(x, s, index, window_cache=(window0, window1), window_size=self.window_size).squeeze(
            -1
        )
        sdf = x.clamp(-1.0, 1.0)
        if return_camera:
            return sdf, self.cam_decoder(self.cam_token, s, cond, layout)
        return sdf
