from torch import nn

from .sparse_atten import MLP, Attention


class DecoderFinalLayer(nn.Module):
    """
    The final layer adopted from DiT.
    """

    def __init__(self, model_channels, out_channels, norm_layer=nn.LayerNorm, *, elementwise_affine=False):
        super().__init__()
        self.norm_final = norm_layer(model_channels, elementwise_affine=elementwise_affine, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)

    def forward(self, x):
        x = self.norm_final(x)
        x = self.linear(x)
        return x


class PixelAttentionBlock(nn.Module):
    """Mix patch features with compact window attention and a per-pixel MLP."""

    def __init__(
        self,
        channels,
        z_channels,
        patch_size,
        num_heads,
        norm_layer=nn.LayerNorm,
        qkv_bias=True,
        qk_norm=False,
        qk_norm_layer=nn.RMSNorm,
    ):
        super().__init__()
        self.channels = channels
        self.patch_size = patch_size

        self.norm1 = norm_layer(channels, elementwise_affine=True, eps=1e-6)

        self.compact_proj = nn.Linear(patch_size**3 * channels, z_channels)
        self.compact_unproj = nn.Linear(z_channels, patch_size**3 * channels)

        self.attn1 = Attention(
            z_channels, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm, norm_layer=qk_norm_layer
        )

        self.norm2 = norm_layer(channels, elementwise_affine=True, eps=1e-6)

        self.mlp = MLP(width=channels)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(z_channels, patch_size**3 * 6 * channels, bias=True)
        )

    def forward(self, x, y, index, *, window_size, window_cache):
        # Modulation and the pixel MLP are patch-local. Only the compact tokens
        # need to be resident together for attention across sparse windows.
        chunk_size = 1024
        patches = self.patch_size**3
        compact = x.new_empty(len(x), self.compact_proj.out_features)
        for start in range(0, len(x), chunk_size):
            part = slice(start, start + chunk_size)
            scale_shift = self.adaLN_modulation(y[part]).reshape(-1, patches, 6, self.channels)
            shift, scale = scale_shift[:, :, 0], scale_shift[:, :, 1]
            normalized = (1 + scale) * self.norm1(x[part]) + shift
            compact[part] = self.compact_proj(normalized.flatten(1))

        attended = self.attn1(compact, index, window_size=window_size, window_cache=window_cache)
        for start in range(0, len(x), chunk_size):
            part = slice(start, start + chunk_size)
            modulation = self.adaLN_modulation(y[part]).reshape(-1, patches, 6, self.channels)
            gate, shift, scale, mlp_gate = modulation[:, :, 2:].unbind(dim=2)
            residual = x[part] + gate * self.compact_unproj(attended[part]).reshape(
                -1, patches, self.channels
            )
            x[part] = residual + mlp_gate * self.mlp((1 + scale) * self.norm2(residual) + shift)
        return x


class PixelAttentionDecoder(nn.Module):
    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        z_channels,
        num_res_blocks,
        patch_size,
        num_heads,
        norm_layer=nn.LayerNorm,
        qkv_bias=True,
        qk_norm=False,
        qk_norm_layer=nn.RMSNorm,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.patch_size = patch_size

        self.input_proj = nn.Linear(in_channels, model_channels)

        self.res_blocks = nn.ModuleList(
            [
                PixelAttentionBlock(
                    model_channels,
                    z_channels,
                    patch_size,
                    num_heads=num_heads,
                    norm_layer=norm_layer,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    qk_norm_layer=qk_norm_layer,
                )
                for _ in range(num_res_blocks)
            ]
        )
        self.final_layer = DecoderFinalLayer(model_channels, out_channels, norm_layer=norm_layer)

    def forward(self, x, c, index, *, window_cache, window_size):
        """
        :param x: [L, patch_size**3, in_channels]
        :param c: [L, z_channels]
        :param index: [L, 4]
        """
        x = self.input_proj(x)  # [L, patch_size**3, model_channels]

        for block in self.res_blocks:
            x = block(x, c, index, window_cache=window_cache, window_size=window_size)
        x = self.final_layer(x)
        return x
