import copy
from dataclasses import dataclass

import torch
import torch.nn as nn

from model import Model


@dataclass
class DenoiserConfig:
    in_features: int
    out_features: int
    cond_features: int
    hidden_dim: int = 256
    num_blocks: int = 4
    time_dim: int = 32
    dropout: float = 0.1
    #
    cond_drop_prob: float = 0.2
    P_mean: float = -0.8
    P_std: float = 0.8
    t_eps: float = 5e-2
    noise_scale: float = 2.0
    #
    ema_decay: tuple[float, ...] = (0.9980, 0.9996)
    #
    sampling_method: str = "heun"
    num_sampling_steps: int = 20
    cfg_scale: float = 2.0
    cfg_interval: tuple[float, float] = (0.1, 1.0)


class Denoiser(nn.Module):
    def __init__(self, config: DenoiserConfig, device: str):
        super().__init__()
        self.net = Model(
            in_features=config.in_features,
            out_features=config.out_features,
            cond_features=config.cond_features,
            hidden_dim=config.hidden_dim,
            num_blocks=config.num_blocks,
            time_dim=config.time_dim,
            dropout=config.dropout,
        )
        self.ema = {
            k: copy.deepcopy(self.net).to(device).eval().requires_grad_(False)
            for k in config.ema_decay
        }
        self.config = config

    def drop_cond(self, x):
        mask = (
            torch.rand(x.size(0), *[1] * (x.ndim - 1), device=x.device)
            >= self.config.cond_drop_prob
        ).float()
        return x * mask

    def sample_t(self, n: int, device=None):
        z = torch.randn(n, device=device) * self.config.P_std + self.config.P_mean
        return torch.sigmoid(z)

    def forward(self, x, cond):
        cond_dropped = self.drop_cond(cond) if self.training else cond

        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.config.noise_scale

        z = t * x + (1 - t) * e
        v = (x - z) / (1 - t).clamp_min(self.config.t_eps)

        x_pred = self.net(z, t.flatten(), cond_dropped)
        v_pred = (x_pred - z) / (1 - t).clamp_min(self.config.t_eps)

        loss = ((v - v_pred) ** 2).mean()
        return loss

    @torch.no_grad()
    def generate(self, cond):
        B = cond.size(0)
        z = self.config.noise_scale * torch.randn(
            B, self.config.out_features, device=cond.device, dtype=cond.dtype
        )
        timesteps = (
            torch.linspace(
                0.0, 1.0, self.config.num_sampling_steps + 1, device=cond.device
            )
            .view(-1, *([1] * z.ndim))
            .expand(-1, B, -1)
        )

        if self.config.sampling_method == "euler":
            stepper = self._euler_step
        elif self.config.sampling_method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError

        for i in range(self.config.num_sampling_steps - 1):
            t, s = timesteps[i], timesteps[i + 1]
            z = stepper(z, t, s, cond)
        z = self._euler_step(z, timesteps[-2], timesteps[-1], cond)
        return z

    @torch.no_grad()
    def _forward_sample(self, z, t, cond):
        # conditional
        x_cond = self.net(z, t.flatten(), cond)
        v_cond = (x_cond - z) / (1.0 - t).clamp_min(self.config.t_eps)
        if self.config.cfg_scale == 1:
            return v_cond

        # unconditional
        x_uncond = self.net(z, t.flatten(), torch.zeros_like(cond))
        v_uncond = (x_uncond - z) / (1.0 - t).clamp_min(self.config.t_eps)

        # cfg interval
        low, high = self.config.cfg_interval
        interval_mask = (t < high) & ((low == 0) | (t > low))
        cfg_scale_interval = torch.where(interval_mask, self.config.cfg_scale, 1.0)

        return v_uncond + cfg_scale_interval * (v_cond - v_uncond)

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, cond):
        v_pred = self._forward_sample(z, t, cond)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, cond):
        v_pred_t = self._forward_sample(z, t, cond)

        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, cond)

        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def update_ema(self):
        params = list(self.net.parameters())
        for decay, ema_params in self.ema.items():
            for targ, src in zip(ema_params.parameters(), params):
                targ.mul_(decay).add_(src, alpha=1 - decay)

    @torch.no_grad()
    def swap_ema(self, decay: float | None = None):
        params = copy.deepcopy(self.net.state_dict())
        ep = self.ema.get(decay, next(iter(self.ema.values())))
        # self.net._orig_mod.load_state_dict(ep.state_dict())  # not compiled
        self.net.load_state_dict(ep.state_dict())
        return params

    @torch.no_grad()
    def swap_params(self, params):
        self.net.load_state_dict(params)


if __name__ == "__main__":
    import time

    from torchinfo import summary

    import utils

    utils.setup_torch()
    device = utils.get_torch_device().type

    x = torch.randn(1, 64, dtype=torch.float32).to(device)
    cond = torch.randn(1, 32, dtype=torch.float32).to(device)

    print("Testing Denoiser...")
    model = Denoiser(
        DenoiserConfig(
            in_features=x.shape[1],
            out_features=x.shape[1],
            cond_features=cond.shape[1],
        ),
        device=device,
    ).to(device)
    summary(model, depth=3)

    print("Testing ema...")
    print("net:", next(model.net.parameters()).device)
    for k, v in model.ema.items():
        print("ema:", k, next(v.parameters()).device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with torch.autocast(device, dtype=torch.bfloat16):
        loss = model(x, cond)
    loss.backward()
    optimizer.step()
    model.update_ema()

    model.eval()
    with torch.inference_mode():
        with torch.autocast(device, dtype=torch.bfloat16):
            torch.manual_seed(42)
            loss = model(x, cond)
        print(f"original {loss=}")
        params = model.swap_ema()
        with torch.autocast(device, dtype=torch.bfloat16):
            torch.manual_seed(42)
            loss = model(x, cond)
        print(f"ema {loss=}")
        model.swap_params(params)
        with torch.autocast(device, dtype=torch.bfloat16):
            torch.manual_seed(42)
            loss = model(x, cond)
        print(f"reverted {loss=}")

    print("Testing generation speed...")
    model.eval()
    num_runs = 60
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        for _ in range(3):
            _ = model.generate(cond)
        torch.accelerator.synchronize()

        start_t = time.perf_counter()
        for _ in range(num_runs):
            _ = model.generate(cond)
        torch.accelerator.synchronize()
        total_t = time.perf_counter() - start_t

    avg_t = total_t / num_runs
    print(f"Total time over {num_runs} runs: {total_t:.6f} s")
    print(f"Avg time per sample: {avg_t:.6f} s")
    print(f"Time per step: {avg_t / model.config.num_sampling_steps:.6f} s")
    print(f"Steps per second: {model.config.num_sampling_steps / avg_t:.2f}")
