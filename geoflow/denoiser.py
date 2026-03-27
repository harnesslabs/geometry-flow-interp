import copy
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

from geoflow.model import models


@dataclass
class DenoiserConfig:
    model: str
    input_size: int
    in_channels: int
    num_classes: int
    #
    cond_drop_prob: float = 0.1
    t_eps: float = 5e-2
    noise_scale: float = 1.0
    #
    ema_decay: tuple[float, ...] = (0.9980, 0.9995)
    #
    sampling_method: Literal["euler", "heun"] = "heun"
    num_sampling_steps: int = 10
    cfg_scale: float = 2.5
    cfg_interval: tuple[float, float] = (0.1, 1.0)


class Denoiser(nn.Module):
    def __init__(self, config: DenoiserConfig, device: str):
        super().__init__()
        self.net = models[config.model](
            input_size=config.input_size,
            in_channels=config.in_channels,
            num_classes=config.num_classes,
        )
        self.ema = {
            k: copy.deepcopy(self.net).to(device).eval().requires_grad_(False)
            for k in config.ema_decay
        }
        self.config = config

    def drop_cond(self, x):
        drop = torch.rand(x.shape[0], device=x.device) < self.config.cond_drop_prob
        return torch.where(drop, self.config.num_classes, x)

    def sample_t(self, n: int, device=None):
        return torch.rand(n, device=device)

    def _to_velocity(self, x, z, t):
        return (x - z) / (1 - t).clamp_min(self.config.t_eps)

    def forward(self, x, cond):
        cond_dropped = self.drop_cond(cond) if self.training else cond

        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.config.noise_scale

        z = t * x + (1 - t) * e
        v = self._to_velocity(x, z, t)

        x_pred = self.net(z, t.flatten(), cond_dropped)
        v_pred = self._to_velocity(x_pred, z, t)
        loss = ((v - v_pred) ** 2).mean()

        return loss

    @torch.no_grad()
    def generate(self, cond, T: int | None = None):
        T = self.config.num_sampling_steps if T is None else T
        B = cond.size(0)
        C = self.net.in_channels
        H = W = self.net.input_size
        z = self.config.noise_scale * torch.randn(B, C, H, W, device=cond.device)
        timesteps = (
            torch.linspace(0.0, 1.0, T + 1, device=cond.device)
            .view(-1, *([1] * z.ndim))
            .expand(-1, B, *[-1] * (z.ndim - 1))
        )

        if self.config.sampling_method == "euler":
            stepper = self._euler_step
        elif self.config.sampling_method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError

        for i in range(T - 1):
            t, s = timesteps[i], timesteps[i + 1]
            z = stepper(z, t, s, cond)
        z = self._euler_step(z, timesteps[-2], timesteps[-1], cond)
        return z

    @torch.no_grad()
    def _forward_sample(self, z, t, cond):
        t_flat = t.flatten()
        # conditional
        x_cond = self.net(z, t_flat, cond)
        v_cond = self._to_velocity(x_cond, z, t)
        if self.config.cfg_scale == 1:
            return v_cond

        # unconditional
        x_uncond = self.net(z, t_flat, torch.full_like(cond, self.config.num_classes))
        v_uncond = self._to_velocity(x_uncond, z, t)

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
        ep = self.ema[decay] if decay is not None else next(iter(self.ema.values()))
        self.net.load_state_dict(ep.state_dict())
        return params

    @torch.no_grad()
    def swap_params(self, params):
        self.net.load_state_dict(params)
