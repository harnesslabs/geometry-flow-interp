import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=t.dtype)
            / half
        )
        args = t[:, None] * freqs[None, :]
        return torch.cat([args.cos(), args.sin()], dim=-1)


class SwiGLU(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class AdaLNBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.gates = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.ff = SwiGLU(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        scale, shift, gate = self.gates(c).chunk(3, dim=-1)
        h = (1.0 + scale) * self.norm(x) + shift
        h = self.drop(self.ff(h))
        return x + gate * h


class Model(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_classes: int,
        hidden_dim: int = 512,
        num_blocks: int = 6,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.out_features = out_features

        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cond_embed = nn.Embedding(n_classes + 1, hidden_dim)
        self.input_embed = nn.Linear(in_features, hidden_dim)

        self.blocks = nn.ModuleList(
            [AdaLNBlock(hidden_dim, dropout) for _ in range(num_blocks)]
        )

        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.gates = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.output = nn.Linear(hidden_dim, out_features)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        nn.init.normal_(self.cond_embed.weight, std=0.02)
        nn.init.normal_(self.input_embed.weight, std=0.02)

        for module in self.blocks:
            block: AdaLNBlock = module  # type: ignore[assignment]
            nn.init.zeros_(block.gates.weight)
            nn.init.zeros_(block.gates.bias)

        nn.init.zeros_(self.gates.weight)
        nn.init.zeros_(self.gates.bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self, z: torch.Tensor, t: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        # z: (B, N), t: (B,), cond: (B,)
        c = self.time_embed(t) + self.cond_embed(cond)
        x = self.input_embed(z)

        for block in self.blocks:
            x = block(x, c)

        scale, shift = self.gates(c).chunk(2, dim=-1)
        x = (1.0 + scale) * self.norm(x) + shift

        return self.output(x)
