"""Phase 0: Trajectory Atlas — reusable data collection pipeline.

Records full ODE trajectories with conditional/unconditional velocities
and internal AdaLN activations at every timestep.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from gf import utils
from gf.denoiser import Denoiser
from gf.generate import load_checkpoint
from gf.model import AdaLNBlock


@dataclass
class TrajectoryAtlas:
    z: Tensor  # (N, T+1, D) state at each timestep
    v_cond: Tensor  # (N, T, D) conditional velocity
    v_uncond: Tensor  # (N, T, D) unconditional velocity
    x_pred_cond: Tensor  # (N, T, D) conditional x-prediction
    timesteps: Tensor  # (T+1,) time schedule
    labels: Tensor  # (N,) class labels
    modulations: dict[str, Tensor]  # per-block scale/shift/gate: (N, T, hidden_dim)


def _compute_cfg_velocity(
    model: Denoiser,
    z: Tensor,
    t_val: Tensor,
    labels: Tensor,
    uncond_labels: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute conditional, unconditional, and CFG-guided velocities.

    Returns:
        v_cond, v_uncond, v_guided, x_cond
    """
    N = z.shape[0]
    t_eps = model.config.t_eps
    one_minus_t = (1.0 - t_val).clamp_min(t_eps)
    t_batch = t_val.expand(N)

    x_cond = model.net(z, t_batch, labels)
    v_cond = (x_cond - z) / one_minus_t

    x_uncond = model.net(z, t_batch, uncond_labels)
    v_uncond = (x_uncond - z) / one_minus_t

    low, high = model.config.cfg_interval
    t_for_mask = t_val.unsqueeze(0)
    interval_mask = (t_for_mask < high) & ((low == 0) | (t_for_mask > low))
    cfg_scale = torch.where(
        interval_mask, model.config.cfg_scale, torch.ones_like(t_for_mask)
    )
    v_guided = v_uncond + cfg_scale * (v_cond - v_uncond)

    return v_cond, v_uncond, v_guided, x_cond


def collect_atlas(
    model: Denoiser,
    n_per_class: int = 100,
    n_classes: int = 10,
    num_steps: int = 50,
    device: str = "cpu",
    sampling_method: str = "euler",
) -> TrajectoryAtlas:
    """Collect trajectory atlas by running ODE integration.

    Args:
        model: Denoiser model with EMA weights already swapped in.
        n_per_class: Number of samples per class.
        n_classes: Number of classes.
        num_steps: Number of integration steps.
        device: Device.
        sampling_method: "euler" (1st-order) or "heun" (2nd-order, matches
            denoiser.generate() default). Heun uses 2x model evaluations
            per step but gives trajectories faithful to actual generation.
    """
    N = n_per_class * n_classes
    D = model.config.out_features
    T = num_steps
    hidden_dim = model.config.hidden_dim
    n_blocks = model.config.num_blocks

    labels = torch.arange(n_classes, device=device).repeat_interleave(n_per_class)
    uncond_labels = torch.full_like(labels, n_classes)

    timesteps = torch.linspace(0.0, 1.0, T + 1, device=device)

    z_traj = torch.zeros(N, T + 1, D, device=device)
    v_cond_traj = torch.zeros(N, T, D, device=device)
    v_uncond_traj = torch.zeros(N, T, D, device=device)
    x_pred_cond_traj = torch.zeros(N, T, D, device=device)

    mod_storage: dict[str, Tensor] = {}
    for block_idx in range(n_blocks):
        for name in ("scale", "shift", "gate"):
            key = f"block_{block_idx}_{name}"
            mod_storage[key] = torch.zeros(N, T, hidden_dim, device=device)

    captured_mods: dict[int, Tensor] = {}

    def make_hook(block_idx: int):
        def hook_fn(module, input, output):  # noqa: A002
            captured_mods[block_idx] = output.detach()

        return hook_fn

    hooks = []
    for idx, block in enumerate(model.net.blocks):
        assert isinstance(block, AdaLNBlock)
        h = block.gates.register_forward_hook(make_hook(idx))
        hooks.append(h)

    z = model.config.noise_scale * torch.randn(N, D, device=device)
    z_traj[:, 0] = z

    use_heun = sampling_method == "heun"

    for step_idx in range(T):
        t_val = timesteps[step_idx]
        t_next = timesteps[step_idx + 1]
        dt = t_next - t_val

        # Evaluate velocity at current state (hooks fire during cond pass)
        v_cond, v_uncond, v_guided, x_cond = _compute_cfg_velocity(
            model, z, t_val, labels, uncond_labels
        )

        # Save modulations from the conditional pass at t_val
        for block_idx in range(n_blocks):
            mods = captured_mods[block_idx]
            s, sh, g = mods.chunk(3, dim=-1)
            mod_storage[f"block_{block_idx}_scale"][:, step_idx] = s
            mod_storage[f"block_{block_idx}_shift"][:, step_idx] = sh
            mod_storage[f"block_{block_idx}_gate"][:, step_idx] = g

        # Store velocities at the current point
        v_cond_traj[:, step_idx] = v_cond
        v_uncond_traj[:, step_idx] = v_uncond
        x_pred_cond_traj[:, step_idx] = x_cond

        if use_heun and step_idx < T - 1:
            # Heun (2nd-order): evaluate at Euler-predicted next point,
            # then average the two velocities. Last step uses Euler
            # (matching denoiser.generate).
            z_euler = z + dt * v_guided
            _, _, v_guided_next, _ = _compute_cfg_velocity(
                model, z_euler, t_next, labels, uncond_labels
            )
            v_step = 0.5 * (v_guided + v_guided_next)
            z = z + dt * v_step
        else:
            # Euler step
            z = z + dt * v_guided

        z_traj[:, step_idx + 1] = z

    for h in hooks:
        h.remove()

    return TrajectoryAtlas(
        z=z_traj.cpu(),
        v_cond=v_cond_traj.cpu(),
        v_uncond=v_uncond_traj.cpu(),
        x_pred_cond=x_pred_cond_traj.cpu(),
        timesteps=timesteps.cpu(),
        labels=labels.cpu(),
        modulations={k: v.cpu() for k, v in mod_storage.items()},
    )


def plot_sanity_check(atlas: TrajectoryAtlas, out_path: Path) -> None:
    """PCA projection of trajectories colored by class and time."""
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    N, T_plus_1, D = atlas.z.shape

    # Subsample for PCA: take every 5th timestep, all trajectories
    step_indices = list(range(0, T_plus_1, 5))
    z_sub = atlas.z[:, step_indices].reshape(-1, D).numpy()  # (N*steps, D)

    pca = PCA(n_components=2)
    z_pca = pca.fit_transform(z_sub)
    z_pca = z_pca.reshape(N, len(step_indices), 2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    cmap_class = plt.get_cmap("tab10")
    cmap_time = plt.get_cmap("viridis")

    # Panel 1: colored by class
    ax = axes[0]
    for i in range(N):
        label = atlas.labels[i].item()
        ax.plot(
            z_pca[i, :, 0],
            z_pca[i, :, 1],
            color=cmap_class(label / 10),
            alpha=0.15,
            linewidth=0.5,
        )
    # Add class legend
    for c in range(10):
        ax.plot([], [], color=cmap_class(c / 10), label=str(c), linewidth=2)
    ax.legend(title="Class", fontsize=7, ncol=2, loc="best")
    ax.set_title("Trajectories colored by class")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")

    # Panel 2: colored by time
    ax = axes[1]
    t_sub = atlas.timesteps[step_indices].numpy()
    for i in range(min(N, 200)):  # limit for clarity
        for j in range(len(step_indices) - 1):
            ax.plot(
                z_pca[i, j : j + 2, 0],
                z_pca[i, j : j + 2, 1],
                color=cmap_time(t_sub[j]),
                alpha=0.2,
                linewidth=0.5,
            )
    sm = plt.cm.ScalarMappable(
        cmap=cmap_time,
        norm=plt.Normalize(vmin=0, vmax=1),
    )
    plt.colorbar(sm, ax=ax, label="Time $t$")
    ax.set_title("Trajectories colored by time")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved sanity check figure to {out_path}")


def main(args: argparse.Namespace) -> None:
    utils.setup_torch()
    device = utils.get_torch_device().type

    model = load_checkpoint(args.experiment, args.checkpoint_dir, device)

    # Swap to EMA weights (use the higher-decay EMA loaded by load_checkpoint)
    ema_key = list(model.ema.keys())[1]
    params = model.swap_ema(decay=ema_key)

    with torch.no_grad():
        atlas = collect_atlas(
            model,
            n_per_class=args.n_per_class,
            n_classes=10,
            num_steps=args.num_steps,
            device=device,
            sampling_method=args.sampling_method,
        )

    # Restore original weights
    model.swap_params(params)

    # Save atlas
    out_dir = Path(args.checkpoint_dir) / args.experiment / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    atlas_path = out_dir / "atlas.pt"
    torch.save(
        {
            "z": atlas.z,
            "v_cond": atlas.v_cond,
            "v_uncond": atlas.v_uncond,
            "x_pred_cond": atlas.x_pred_cond,
            "timesteps": atlas.timesteps,
            "labels": atlas.labels,
            "modulations": atlas.modulations,
        },
        atlas_path,
    )
    print(f"Saved atlas to {atlas_path}")
    print(f"  z: {atlas.z.shape}")
    print(f"  v_cond: {atlas.v_cond.shape}")
    print(f"  timesteps: {atlas.timesteps.shape}")
    print(f"  labels: {atlas.labels.shape}")
    print(f"  modulations: {list(atlas.modulations.keys())}")

    # Sanity check figure
    fig_path = out_dir / "atlas_sanity.png"
    plot_sanity_check(atlas, fig_path)


def load_atlas(experiment: str, checkpoint_dir: str = "checkpoints") -> TrajectoryAtlas:
    """Load a saved trajectory atlas."""
    path = Path(checkpoint_dir) / experiment / "analysis" / "atlas.pt"
    data = torch.load(path, map_location="cpu")
    return TrajectoryAtlas(
        z=data["z"],
        v_cond=data["v_cond"],
        v_uncond=data["v_uncond"],
        x_pred_cond=data["x_pred_cond"],
        timesteps=data["timesteps"],
        labels=data["labels"],
        modulations=data["modulations"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="default")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--n-per-class", type=int, default=100)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument(
        "--sampling-method",
        type=str,
        default="euler",
        choices=["euler", "heun"],
        help="ODE integration method. 'heun' matches denoiser.generate() default.",
    )
    args = parser.parse_args()
    main(args)
