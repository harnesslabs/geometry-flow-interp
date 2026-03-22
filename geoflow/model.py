import math
from math import pi

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class VisionRotaryEmbeddingFast(nn.Module):
    def __init__(
        self,
        dim,
        pt_seq_len=16,
        ft_seq_len=None,
        custom_freqs=None,
        freqs_for="lang",
        theta=10000,
        max_freq=10,
        num_freqs=1,
        num_cls_token=0,
    ):
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == "lang":
            freqs = 1.0 / (
                theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
            )
        elif freqs_for == "pixel":
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * pi
        elif freqs_for == "constant":
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f"unknown modality {freqs_for}")

        if ft_seq_len is None:
            ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len

        freqs = torch.einsum("..., f -> ... f", t, freqs)
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)
        freqs = self._broadcat((freqs[:, None, :], freqs[None, :, :]), dim=-1)

        if num_cls_token > 0:
            freqs_flat = freqs.view(-1, freqs.shape[-1])  # [N_img, D]
            cos_img = freqs_flat.cos()
            sin_img = freqs_flat.sin()

            # prepend in-context cls token
            N_img, D = cos_img.shape
            cos_pad = torch.ones(
                num_cls_token, D, dtype=cos_img.dtype, device=cos_img.device
            )
            sin_pad = torch.zeros(
                num_cls_token, D, dtype=sin_img.dtype, device=sin_img.device
            )

            # [N_cls+N_img, D]
            freqs_cos = torch.cat([cos_pad, cos_img], dim=0)
            freqs_sin = torch.cat([sin_pad, sin_img], dim=0)
        else:
            freqs_cos = freqs.cos().view(-1, freqs.shape[-1])
            freqs_sin = freqs.sin().view(-1, freqs.shape[-1])

        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

    def _broadcat(self, tensors, dim=-1):
        num_tensors = len(tensors)
        shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
        assert len(shape_lens) == 1, (
            "tensors must all have the same number of dimensions"
        )
        shape_len = list(shape_lens)[0]
        dim = (dim + shape_len) if dim < 0 else dim
        dims = list(zip(*map(lambda t: list(t.shape), tensors)))
        expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
        assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), (
            "invalid dimensions for broadcastable concatentation"
        )
        max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
        expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
        expanded_dims.insert(dim, (dim, dims[dim]))
        expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
        tensors = list(
            map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes))
        )
        return torch.cat(tensors, dim=dim)

    def _rotate_half(self, x):
        x = rearrange(x, "... (d r) -> ... d r", r=2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        return rearrange(x, "... d r -> ... (d r)")

    def forward(self, t):
        return t * self.freqs_cos + self._rotate_half(t) * self.freqs_sin


class CastedLinear(nn.Linear):
    # Keep weights in fp32 for optimizer/state quality, cast at matmul time for bf16 compute.
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        bias = self.bias.to(input.dtype) if self.bias is not None else None
        return F.linear(input, self.weight.to(input.dtype), bias)


class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class BottleneckPatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        pca_dim=768,
        embed_dim=768,
        bias=True,
    ):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj1 = nn.Conv2d(
            in_chans, pca_dim, kernel_size=patch_size, stride=patch_size, bias=False
        )
        self.proj2 = nn.Conv2d(pca_dim, embed_dim, kernel_size=1, stride=1, bias=bias)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], (
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        )
        x = self.proj2(self.proj1(x)).flatten(2).transpose(1, 2)
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            CastedLinear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            CastedLinear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        self.num_classes = num_classes

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_heads // 2
        self.head_dim = dim // num_heads
        kv_dim = self.num_kv_heads * self.head_dim

        self.q_proj = CastedLinear(dim, dim, bias=qkv_bias)
        self.k_proj = CastedLinear(dim, kv_dim, bias=qkv_bias)
        self.v_proj = CastedLinear(dim, kv_dim, bias=qkv_bias)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.q_gain = nn.Parameter(torch.full((num_heads,), 1.5, dtype=torch.float32))

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope):
        B, N, C = x.shape

        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = (
            self.k_proj(x)
            .reshape(B, N, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .reshape(B, N, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )

        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        q, k = rope(q), rope(k)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]

        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            enable_gqa=True,
        )

        x = x.transpose(1, 2).contiguous().reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop=0.0) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = CastedLinear(dim, 2 * hidden_dim, bias=False)
        self.w3 = CastedLinear(hidden_dim, dim, bias=False)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(self.ffn_dropout(hidden))


class FinalLayer(nn.Module):
    """
    The final layer of JiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm()
        self.linear = CastedLinear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), CastedLinear(hidden_size, 2 * hidden_size, bias=True)
        )

    @torch.compile
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class JiTBlock(nn.Module):
    def __init__(
        self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0
    ):
        super().__init__()
        self.norm1 = RMSNorm(eps=1e-6)
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=False,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.norm2 = RMSNorm(eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), CastedLinear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.resid_mix = nn.Parameter(
            torch.stack((torch.ones(hidden_size), torch.zeros(hidden_size))).float()
        )

    @torch.compile
    def forward(self, x, c, emb, feat_rope=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * emb
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class JiT(nn.Module):
    """
    Just image Transformer.
    """

    def __init__(
        self,
        input_size,
        in_channels,
        num_classes,
        patch_size,
        hidden_size=256,
        depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        bottleneck_dim=128,
        in_context_len=4,
        in_context_start=2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.in_context_len = in_context_len
        self.in_context_start = in_context_start
        self.num_classes = num_classes

        # time and class embed
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)

        # linear embed
        self.x_embedder = BottleneckPatchEmbed(
            input_size, patch_size, in_channels, bottleneck_dim, hidden_size, bias=True
        )

        # use fixed sin-cos embedding
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size), requires_grad=False
        )

        # in-context cls token
        if self.in_context_len > 0:
            self.in_context_posemb = nn.Parameter(
                torch.zeros(1, self.in_context_len, hidden_size), requires_grad=True
            )
            nn.init.normal_(self.in_context_posemb, std=0.02)

        # rope
        half_head_dim = hidden_size // num_heads // 2
        hw_seq_len = input_size // patch_size
        self.feat_rope = VisionRotaryEmbeddingFast(
            dim=half_head_dim, pt_seq_len=hw_seq_len, num_cls_token=0
        )
        self.feat_rope_incontext = VisionRotaryEmbeddingFast(
            dim=half_head_dim, pt_seq_len=hw_seq_len, num_cls_token=self.in_context_len
        )

        # transformer
        self.blocks = nn.ModuleList(
            [
                JiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_drop=attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                    proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                )
                for i in range(depth)
            ]
        )

        # linear predict
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w1 = self.x_embedder.proj1.weight.data
        nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
        w2 = self.x_embedder.proj2.weight.data
        nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
        assert self.x_embedder.proj2.bias is not None
        nn.init.constant_(self.x_embedder.proj2.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        t_mlp_0 = self.t_embedder.mlp[0]
        t_mlp_2 = self.t_embedder.mlp[2]
        assert isinstance(t_mlp_0, nn.Linear) and isinstance(t_mlp_2, nn.Linear)
        nn.init.normal_(t_mlp_0.weight, std=0.02)
        nn.init.normal_(t_mlp_2.weight, std=0.02)

        # Zero-out adaLN modulation layers:
        for block in self.blocks:
            assert isinstance(block, JiTBlock)
            ada_ln = block.adaLN_modulation[-1]
            assert isinstance(ada_ln, nn.Linear)
            nn.init.constant_(ada_ln.weight, 0)
            nn.init.constant_(ada_ln.bias, 0)

        # Zero-out output layers:
        final_ada_ln = self.final_layer.adaLN_modulation[-1]
        assert isinstance(final_ada_ln, nn.Linear)
        nn.init.constant_(final_ada_ln.weight, 0)
        nn.init.constant_(final_ada_ln.bias, 0)

        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x, p):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y):
        """
        x: (N, C, H, W)
        t: (N,)
        y: (N,)
        """
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y)
        c = t_emb + y_emb

        x = emb = self.x_embedder(x) + self.pos_embed

        for i, block in enumerate(self.blocks):
            if self.in_context_len > 0 and i == self.in_context_start:
                ctx = (
                    y_emb.unsqueeze(1).expand(-1, self.in_context_len, -1)
                    + self.in_context_posemb
                )
                x = torch.cat([ctx, x], dim=1)
                emb = F.pad(emb, (0, 0, self.in_context_len, 0))
            rope = (
                self.feat_rope
                if i < self.in_context_start
                else self.feat_rope_incontext
            )
            x = block(x, c, emb, rope)

        x = x[:, self.in_context_len :]
        x = self.final_layer(x, c)
        return self.unpatchify(x, self.patch_size)


def JiT_S_7(**kwargs):
    return JiT(
        depth=4,
        hidden_size=192,
        num_heads=6,
        mlp_ratio=2.0,
        bottleneck_dim=48,
        in_context_len=4,
        in_context_start=2,
        patch_size=7,
        **kwargs,
    )


def JiT_S_8(**kwargs):
    return JiT(
        depth=4,
        hidden_size=192,
        num_heads=6,
        mlp_ratio=2.0,
        bottleneck_dim=48,
        in_context_len=4,
        in_context_start=2,
        patch_size=8,
        **kwargs,
    )


def JiT_M_7(**kwargs):
    return JiT(
        depth=4,
        hidden_size=256,
        num_heads=8,
        mlp_ratio=2.0,
        bottleneck_dim=48,
        in_context_len=4,
        in_context_start=2,
        patch_size=7,
        **kwargs,
    )


def JiT_B_4(**kwargs):
    return JiT(
        depth=8,
        hidden_size=512,
        num_heads=8,
        mlp_ratio=4.0,
        bottleneck_dim=128,
        in_context_len=4,
        in_context_start=4,
        patch_size=4,
        **kwargs,
    )


models = {
    "JiT-S/7": JiT_S_7,
    "JiT-S/8": JiT_S_8,
    "JiT-M/7": JiT_M_7,
    "JiT-B/4": JiT_B_4,
}
