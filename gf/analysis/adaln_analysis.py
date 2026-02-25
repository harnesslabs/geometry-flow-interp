"""Phase 2: AdaLN Modulation Analysis (Mechanistic Interpretability).

Analyzes AdaLN scale/shift/gate vectors to understand how the model implements
class conditioning, with causal validation via counterfactual patching.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from torch import Tensor

from gf import utils
from gf.analysis.atlas import TrajectoryAtlas, load_atlas
from gf.denoiser import Denoiser
from gf.generate import load_checkpoint
from gf.model import AdaLNBlock


# ---------------------------------------------------------------------------
# Experiment 2a: Modulation fingerprints (from atlas, no new forward passes)
# ---------------------------------------------------------------------------


def compute_modulation_fingerprints(
    atlas: TrajectoryAtlas,
    n_classes: int = 10,
) -> dict[str, Tensor]:
    """Compute class-mean modulations and pairwise cosine similarity.

    Returns:
        class_means: dict of (n_classes, T, hidden_dim) per block/component
        cosine_sim: dict of (T, n_classes, n_classes) per block/component
        gate_magnitudes: dict of (n_classes, T) per block
    """
    T = atlas.v_cond.shape[1]
    results: dict[str, Tensor] = {}

    # Identify blocks and components
    block_indices = set()
    for key in atlas.modulations:
        # key format: "block_{idx}_{component}"
        parts = key.split("_")
        block_indices.add(int(parts[1]))

    for block_idx in sorted(block_indices):
        for comp in ("scale", "shift", "gate"):
            key = f"block_{block_idx}_{comp}"
            mods = atlas.modulations[key]  # (N, T, hidden_dim)

            # Class means
            class_means = torch.zeros(n_classes, T, mods.shape[-1])
            for c in range(n_classes):
                mask = atlas.labels == c
                class_means[c] = mods[mask].mean(dim=0)
            results[f"{key}_mean"] = class_means

            # Pairwise cosine similarity at each timestep
            # Mean-center to remove shared time-dependent component
            # and reveal class-specific structure
            cos_sim = torch.zeros(T, n_classes, n_classes)
            for t in range(T):
                vecs = class_means[:, t]  # (n_classes, hidden_dim)
                vecs_centered = vecs - vecs.mean(dim=0, keepdim=True)
                norms = vecs_centered.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                vecs_normed = vecs_centered / norms
                cos_sim[t] = vecs_normed @ vecs_normed.T
            results[f"{key}_cosine_sim"] = cos_sim

        # Gate magnitude per class
        gate_key = f"block_{block_idx}_gate"
        gate_mods = atlas.modulations[gate_key]  # (N, T, hidden_dim)
        gate_mag = torch.zeros(n_classes, T)
        for c in range(n_classes):
            mask = atlas.labels == c
            gate_mag[c] = gate_mods[mask].norm(dim=-1).mean(dim=0)
        results[f"block_{block_idx}_gate_magnitude"] = gate_mag

    return results


# ---------------------------------------------------------------------------
# Experiment 2b: Counterfactual modulation patching
# ---------------------------------------------------------------------------


def counterfactual_patch(
    model: Denoiser,
    source_class: int,
    target_class: int,
    patch_block: int,
    patch_component: str,
    n_samples: int = 10,
    num_steps: int = 50,
    device: str = "cpu",
) -> dict[str, Tensor]:
    """Run ODE for source_class but inject target_class conditioning at patch_block.

    Args:
        model: Denoiser model (should already have EMA weights swapped in).
        source_class: Class to generate.
        target_class: Class whose conditioning to inject.
        patch_block: Which block to patch (0-indexed).
        patch_component: "all", "scale", "shift", or "gate".
        n_samples: Number of samples to generate.
        num_steps: Number of Euler steps.
        device: Device.

    Returns:
        samples_source: (n_samples, D) source class generation (no patching)
        samples_patched: (n_samples, D) patched generation
        mse: scalar MSE between source and patched
    """
    D = model.config.out_features
    t_eps = model.config.t_eps
    n_classes_total = model.config.n_classes

    cond_source = torch.full(
        (n_samples,), source_class, device=device, dtype=torch.long
    )
    cond_target = torch.full(
        (n_samples,), target_class, device=device, dtype=torch.long
    )

    timesteps = torch.linspace(0.0, 1.0, num_steps + 1, device=device)

    # Shared initial noise
    torch.manual_seed(42)
    z_init = model.config.noise_scale * torch.randn(n_samples, D, device=device)

    def run_ode(cond, patch_block_idx=-1, patch_comp="all"):
        z = z_init.clone()
        net = model.net

        for step in range(num_steps):
            t_val = timesteps[step]
            t_next = timesteps[step + 1]
            dt = t_next - t_val
            one_minus_t = (1.0 - t_val).clamp_min(t_eps)

            t_batch = t_val.expand(n_samples)

            # Compute conditioning vectors
            c_source = net.time_embed(t_batch) + net.cond_embed(cond_source)
            c_target = net.time_embed(t_batch) + net.cond_embed(cond_target)
            c_uncond = net.time_embed(t_batch) + net.cond_embed(
                torch.full_like(cond, n_classes_total)
            )

            # Custom forward through blocks
            x = net.input_embed(z)
            for block_i, block in enumerate(net.blocks):
                assert isinstance(block, AdaLNBlock)
                if block_i == patch_block_idx:
                    x = _patched_block_forward(block, x, c_source, c_target, patch_comp)
                else:
                    x = block(x, cond_to_use(cond, c_source, c_target, block_i, -1))

            x_cond = net.output(x)
            v_cond = (x_cond - z) / one_minus_t

            # Unconditional pass (no patching)
            x_u = net.input_embed(z)
            for block in net.blocks:
                assert isinstance(block, AdaLNBlock)
                x_u = block(x_u, c_uncond)
            x_uncond = net.output(x_u)
            v_uncond = (x_uncond - z) / one_minus_t

            # CFG
            low, high = model.config.cfg_interval
            t_for_mask = t_val.unsqueeze(0)
            interval_mask = (t_for_mask < high) & ((low == 0) | (t_for_mask > low))
            cfg_scale = torch.where(
                interval_mask, model.config.cfg_scale, torch.ones_like(t_for_mask)
            )
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
            z = z + dt * v

        return z

    def cond_to_use(cond, c_src, c_tgt, block_i, patch_i):
        """Use source conditioning for all blocks (no patch)."""
        return c_src

    # Source (no patching)
    with torch.no_grad():
        samples_source = run_ode(cond_source, patch_block_idx=-1)

    # Patched
    with torch.no_grad():
        samples_patched = run_ode(
            cond_source, patch_block_idx=patch_block, patch_comp=patch_component
        )

    mse = ((samples_source - samples_patched) ** 2).mean()

    return {
        "samples_source": samples_source.cpu(),
        "samples_patched": samples_patched.cpu(),
        "mse": mse.cpu(),
    }


def _patched_block_forward(
    block: AdaLNBlock,
    x: Tensor,
    c_source: Tensor,
    c_target: Tensor,
    patch_component: str,
) -> Tensor:
    """Forward through a block with selective component patching."""
    # Source modulations
    scale_s, shift_s, gate_s = block.gates(c_source).chunk(3, dim=-1)
    # Target modulations
    scale_t, shift_t, gate_t = block.gates(c_target).chunk(3, dim=-1)

    # Select which components to patch
    if patch_component == "all":
        scale, shift, gate = scale_t, shift_t, gate_t
    elif patch_component == "scale":
        scale, shift, gate = scale_t, shift_s, gate_s
    elif patch_component == "shift":
        scale, shift, gate = scale_s, shift_t, gate_s
    elif patch_component == "gate":
        scale, shift, gate = scale_s, shift_s, gate_t
    else:
        raise ValueError(f"Unknown patch_component: {patch_component}")

    h = (1.0 + scale) * block.norm(x) + shift
    h = block.drop(block.ff(h))
    return x + gate * h


def run_patching_experiment(
    model: Denoiser,
    n_blocks: int = 2,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run full counterfactual patching grid."""
    components = ["all", "scale", "shift", "gate"]
    # Test a few class pairs
    class_pairs = [(0, 1), (3, 8), (4, 9), (7, 2)]

    mse_results = torch.zeros(len(class_pairs), n_blocks, len(components))

    all_samples: dict[str, Tensor] = {}

    for pair_idx, (src, tgt) in enumerate(class_pairs):
        print(f"  Patching {src} -> {tgt}")
        for block_idx in range(n_blocks):
            for comp_idx, comp in enumerate(components):
                result = counterfactual_patch(
                    model,
                    source_class=src,
                    target_class=tgt,
                    patch_block=block_idx,
                    patch_component=comp,
                    device=device,
                )
                mse_results[pair_idx, block_idx, comp_idx] = result["mse"]

                # Save samples for the "all" component for visualization
                if comp == "all":
                    all_samples[f"{src}_{tgt}_block{block_idx}_source"] = result[
                        "samples_source"
                    ]
                    all_samples[f"{src}_{tgt}_block{block_idx}_patched"] = result[
                        "samples_patched"
                    ]

    return {
        "mse_results": mse_results,
        "class_pairs": class_pairs,
        "components": components,
        "samples": all_samples,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_adaln_results(
    fingerprints: dict[str, Tensor],
    patching: dict[str, Any],
    atlas: TrajectoryAtlas,
    n_blocks: int,
    out_dir: Path,
) -> None:
    """Plot AdaLN analysis results (Figures 3a-3d)."""
    timesteps = atlas.timesteps[:-1].numpy()
    n_classes = 10
    cmap = plt.get_cmap("tab10")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Fig 3a: Gate magnitude heatmaps
    ax = axes[0, 0]
    for block_idx in range(n_blocks):
        gate_mag = fingerprints[f"block_{block_idx}_gate_magnitude"]  # (n_classes, T)
        # Plot as lines per class
        for c in range(n_classes):
            linestyle = "-" if block_idx == 0 else "--"
            label = f"b{block_idx}/c{c}" if c < 3 else None
            ax.plot(
                timesteps,
                gate_mag[c].numpy(),
                color=cmap(c / 10),
                linestyle=linestyle,
                alpha=0.6,
                label=label,
            )
    ax.set_xlabel("Time $t$")
    ax.set_ylabel("$\\|\\mathrm{gate}\\|$")
    ax.set_title("(a) Gate magnitude (solid=block 0, dashed=block 1)")
    ax.legend(fontsize=6, ncol=3, loc="best")

    # Fig 3b: Cosine similarity matrices at t=0.2, 0.5, 0.8
    ax = axes[0, 1]
    t_indices = [
        int(0.2 * len(timesteps)),
        int(0.5 * len(timesteps)),
        int(0.8 * len(timesteps)),
    ]
    # Show block 0 gate cosine similarity at 3 timepoints
    cos_key = "block_0_gate_cosine_sim"
    cos_sim = fingerprints[cos_key]  # (T, n_classes, n_classes)

    # Create a 1x3 sub-grid inside this axis
    ax.set_visible(False)
    gs = axes[0, 1].get_gridspec()
    sub_axes = fig.add_subplot(gs[0, 1]).inset_axes((0, 0, 1, 1))
    sub_axes.set_visible(False)

    for i, (ti, t_label) in enumerate(
        zip(t_indices, ["$t=0.2$", "$t=0.5$", "$t=0.8$"])
    ):
        sub_ax = fig.add_axes((0.55 + i * 0.14, 0.58, 0.12, 0.3))
        im = sub_ax.imshow(cos_sim[ti].numpy(), vmin=-1, vmax=1, cmap="RdBu_r")
        sub_ax.set_title(t_label, fontsize=9)
        sub_ax.set_xticks(range(0, 10, 2))
        sub_ax.set_yticks(range(0, 10, 2))
        if i == 0:
            sub_ax.set_ylabel("Class")
        if i == 2:
            plt.colorbar(im, ax=sub_ax, fraction=0.046)

    # Fig 3c: Counterfactual patching grid (sample images)
    ax = axes[1, 0]
    samples = patching["samples"]
    class_pairs = patching["class_pairs"]

    n_show = min(4, len(class_pairs))
    n_per = 5  # samples per pair
    grid_h = n_show * 2  # source + patched rows
    grid_w = n_per

    img_grid = torch.zeros(grid_h, grid_w, 28, 28)
    row = 0
    for pair_idx in range(n_show):
        src, tgt = class_pairs[pair_idx]
        src_key = f"{src}_{tgt}_block0_source"
        pat_key = f"{src}_{tgt}_block0_patched"
        if src_key in samples and pat_key in samples:
            src_imgs = samples[src_key][:n_per].view(-1, 28, 28)
            pat_imgs = samples[pat_key][:n_per].view(-1, 28, 28)
            img_grid[row, :n_per] = (src_imgs + 1) / 2
            img_grid[row + 1, :n_per] = (pat_imgs + 1) / 2
        row += 2

    # Flatten grid to single image
    grid_flat = img_grid.permute(0, 2, 1, 3).reshape(grid_h * 28, grid_w * 28)
    ax.imshow(grid_flat.numpy(), cmap="gray", vmin=0, vmax=1)
    ax.set_title("(c) Counterfactual patching (rows: source/patched)")
    ax.axis("off")

    # Fig 3d: Causal effect bar chart
    ax = axes[1, 1]
    mse_results = patching["mse_results"]  # (n_pairs, n_blocks, n_components)
    components = patching["components"]
    n_blocks_actual = mse_results.shape[1]

    avg_mse = mse_results.mean(dim=0)  # (n_blocks, n_components)
    x_pos = range(len(components))
    width = 0.35

    for block_idx in range(n_blocks_actual):
        offset = (block_idx - 0.5) * width
        ax.bar(
            [p + offset for p in x_pos],
            avg_mse[block_idx].numpy(),
            width,
            label=f"Block {block_idx}",
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(components)
    ax.set_ylabel("MSE (causal effect)")
    ax.set_title("(d) Causal effect by block/component")
    ax.legend()

    plt.tight_layout()
    out_path = out_dir / "adaln_analysis.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved AdaLN analysis figure to {out_path}")


def main(args: argparse.Namespace) -> None:
    utils.setup_torch()
    device = utils.get_torch_device().type

    print("Loading atlas...")
    atlas = load_atlas(args.experiment, args.checkpoint_dir)

    out_dir = Path(args.checkpoint_dir) / args.experiment / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_blocks = len([k for k in atlas.modulations if k.endswith("_gate")])
    print(f"Found {n_blocks} blocks")

    # Experiment 2a: Modulation fingerprints
    print("\nComputing modulation fingerprints...")
    fingerprints = compute_modulation_fingerprints(atlas)

    # Experiment 2b: Counterfactual patching
    print("\nRunning counterfactual patching...")
    model = load_checkpoint(args.experiment, args.checkpoint_dir, device)
    ema_key = list(model.ema.keys())[1]
    params = model.swap_ema(decay=ema_key)

    with torch.no_grad():
        patching_results = run_patching_experiment(
            model, n_blocks=n_blocks, device=device
        )

    model.swap_params(params)

    # Save results
    torch.save(
        {
            "fingerprints": fingerprints,
            "patching": patching_results,
        },
        out_dir / "adaln_results.pt",
    )
    print(f"Saved results to {out_dir / 'adaln_results.pt'}")

    # Plot
    plot_adaln_results(fingerprints, patching_results, atlas, n_blocks, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="default")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    args = parser.parse_args()
    main(args)
