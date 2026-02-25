"""Phase 3: Trajectory Topology and Counterfactual Branching.

Understands global structure of how trajectories organize and separate by class,
using counterfactual label switching and persistent homology.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from gf import utils
from gf.analysis.atlas import TrajectoryAtlas, load_atlas
from gf.denoiser import Denoiser
from gf.generate import load_checkpoint


# ---------------------------------------------------------------------------
# Experiment 3a: Counterfactual label branching
# ---------------------------------------------------------------------------


def counterfactual_branching(
    model: Denoiser,
    atlas: TrajectoryAtlas,
    source_class: int = 0,
    target_class: int = 1,
    branch_times: list[float] | None = None,
    n_samples: int = 10,
    num_steps: int = 50,
    device: str = "cpu",
) -> dict[str, Any]:
    """At branch_time, switch class label and continue ODE.

    Returns:
        original: (n_samples, D) original class-A completion
        branched: dict of branch_time -> (n_samples, D) branched completions
        fresh_target: (n_samples, D) fresh class-B generation from same noise
    """
    if branch_times is None:
        branch_times = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    D = model.config.out_features
    t_eps = model.config.t_eps
    n_classes_total = model.config.n_classes

    timesteps = torch.linspace(0.0, 1.0, num_steps + 1, device=device)

    cond_source = torch.full(
        (n_samples,), source_class, device=device, dtype=torch.long
    )
    cond_target = torch.full(
        (n_samples,), target_class, device=device, dtype=torch.long
    )
    cond_uncond = torch.full(
        (n_samples,), n_classes_total, device=device, dtype=torch.long
    )

    # Shared initial noise
    torch.manual_seed(42)
    z_init = model.config.noise_scale * torch.randn(n_samples, D, device=device)

    def run_ode(z_start, cond, start_step=0):
        """Run Euler ODE from start_step to end."""
        z = z_start.clone()
        for step in range(start_step, num_steps):
            t_val = timesteps[step]
            t_next = timesteps[step + 1]
            dt = t_next - t_val
            one_minus_t = (1.0 - t_val).clamp_min(t_eps)

            t_batch = t_val.expand(n_samples)

            x_cond = model.net(z, t_batch, cond)
            v_cond = (x_cond - z) / one_minus_t

            x_uncond = model.net(z, t_batch, cond_uncond)
            v_uncond = (x_uncond - z) / one_minus_t

            low, high = model.config.cfg_interval
            t_for_mask = t_val.unsqueeze(0)
            interval_mask = (t_for_mask < high) & ((low == 0) | (t_for_mask > low))
            cfg_scale = torch.where(
                interval_mask, model.config.cfg_scale, torch.ones_like(t_for_mask)
            )
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
            z = z + dt * v
        return z

    # Original class-A completion
    original = run_ode(z_init, cond_source)

    # Fresh class-B from same noise
    fresh_target = run_ode(z_init, cond_target)

    # Branched completions
    branched = {}
    for t_branch in branch_times:
        # Find the step closest to t_branch
        branch_step = int(t_branch * num_steps)
        branch_step = max(0, min(branch_step, num_steps - 1))

        # Run source class up to branch point, storing intermediate state
        z = z_init.clone()
        for step in range(branch_step):
            t_val = timesteps[step]
            t_next = timesteps[step + 1]
            dt = t_next - t_val
            one_minus_t = (1.0 - t_val).clamp_min(t_eps)
            t_batch = t_val.expand(n_samples)

            x_cond = model.net(z, t_batch, cond_source)
            v_cond = (x_cond - z) / one_minus_t
            x_uncond = model.net(z, t_batch, cond_uncond)
            v_uncond = (x_uncond - z) / one_minus_t

            low, high = model.config.cfg_interval
            t_for_mask = t_val.unsqueeze(0)
            interval_mask = (t_for_mask < high) & ((low == 0) | (t_for_mask > low))
            cfg_scale = torch.where(
                interval_mask, model.config.cfg_scale, torch.ones_like(t_for_mask)
            )
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
            z = z + dt * v

        # Continue with target class from branch point
        z_branched = run_ode(z, cond_target, start_step=branch_step)
        branched[t_branch] = z_branched

    return {
        "original": original.cpu(),
        "fresh_target": fresh_target.cpu(),
        "branched": {k: v.cpu() for k, v in branched.items()},
    }


# ---------------------------------------------------------------------------
# Experiment 3b: Persistent homology of trajectory point clouds
# ---------------------------------------------------------------------------


def compute_persistence(
    atlas: TrajectoryAtlas,
    n_classes: int = 10,
    step_stride: int = 5,
    pca_dim: int = 50,
) -> dict[str, Tensor]:
    """Compute Rips persistent homology per class at each timestep.

    PCA is fitted once on pooled data across all timesteps for stability,
    ensuring persistence values are directly comparable across time slices.

    Returns:
        total_persistence_h0: (n_classes, n_timesteps) H0 total persistence
        total_persistence_h1: (n_classes, n_timesteps) H1 total persistence
        persistence_entropy_h0: (n_classes, n_timesteps) H0 persistence entropy
        analyzed_times: (n_timesteps,) time values analyzed
    """
    from ripser import ripser
    from sklearn.decomposition import PCA

    N, T_plus_1, D = atlas.z.shape
    step_indices = list(range(0, T_plus_1, step_stride))
    n_steps = len(step_indices)

    total_pers_h0 = torch.zeros(n_classes, n_steps)
    total_pers_h1 = torch.zeros(n_classes, n_steps)
    pers_entropy_h0 = torch.zeros(n_classes, n_steps)

    # Fit PCA once on pooled data across all analyzed timesteps for stability.
    # This ensures the embedding is consistent across time slices.
    need_pca = D > pca_dim
    pca = None
    if need_pca:
        print("  Fitting global PCA for persistence...")
        z_pooled = atlas.z[:, step_indices].reshape(-1, D).numpy()
        pca = PCA(n_components=pca_dim)
        pca.fit(z_pooled)
        print(
            f"  PCA: {D} -> {pca_dim} dims, "
            f"explained variance = {pca.explained_variance_ratio_.sum():.3f}"
        )

    for c in range(n_classes):
        mask = atlas.labels == c
        z_class = atlas.z[mask]  # (n_per_class, T+1, D)

        for si, step in enumerate(step_indices):
            points = z_class[:, step].numpy()  # (n_per_class, D)

            if pca is not None:
                points = pca.transform(points)

            result = ripser(points, maxdim=1, thresh=np.inf)
            diagrams = result["dgms"]

            # H0: connected components
            h0 = diagrams[0]
            finite_h0 = h0[np.isfinite(h0[:, 1])]
            if len(finite_h0) > 0:
                lifetimes = finite_h0[:, 1] - finite_h0[:, 0]
                total_pers_h0[c, si] = lifetimes.sum()
                probs = lifetimes / lifetimes.sum()
                probs = probs[probs > 0]
                pers_entropy_h0[c, si] = -(probs * np.log2(probs)).sum()

            # H1: loops
            if len(diagrams) > 1:
                h1 = diagrams[1]
                finite_h1 = h1[np.isfinite(h1[:, 1])]
                if len(finite_h1) > 0:
                    lifetimes_h1 = finite_h1[:, 1] - finite_h1[:, 0]
                    total_pers_h1[c, si] = lifetimes_h1.sum()

        print(f"  Class {c} persistence computed")

    return {
        "total_persistence_h0": total_pers_h0,
        "total_persistence_h1": total_pers_h1,
        "persistence_entropy_h0": pers_entropy_h0,
        "analyzed_times": atlas.timesteps[step_indices],
    }


# ---------------------------------------------------------------------------
# Experiment 3c: Cross-class distance evolution (MMD)
# ---------------------------------------------------------------------------


def _rbf_kernel(x: Tensor, y: Tensor, bandwidth: float) -> Tensor:
    """Compute RBF kernel matrix between x and y."""
    xx = (x * x).sum(dim=-1, keepdim=True)  # (n, 1)
    yy = (y * y).sum(dim=-1, keepdim=True)  # (m, 1)
    dists_sq = xx + yy.T - 2.0 * x @ y.T  # (n, m)
    return torch.exp(-dists_sq / (2.0 * bandwidth**2))


def _mmd_squared(x: Tensor, y: Tensor, bandwidth: float) -> Tensor:
    """Compute squared MMD between two sets of samples with RBF kernel."""
    k_xx = _rbf_kernel(x, x, bandwidth)
    k_yy = _rbf_kernel(y, y, bandwidth)
    k_xy = _rbf_kernel(x, y, bandwidth)

    n = x.shape[0]
    m = y.shape[0]

    # Unbiased estimate (exclude diagonal for k_xx and k_yy)
    mmd = (
        (k_xx.sum() - k_xx.diagonal().sum()) / (n * (n - 1))
        + (k_yy.sum() - k_yy.diagonal().sum()) / (m * (m - 1))
        - 2.0 * k_xy.mean()
    )
    return mmd.clamp_min(0.0)


def compute_cross_class_distances(
    atlas: TrajectoryAtlas,
    n_classes: int = 10,
    step_stride: int = 5,
) -> dict[str, Tensor]:
    """Compute pairwise MMD between class point clouds over time.

    Uses Maximum Mean Discrepancy with RBF kernel instead of mean-only
    distance, which captures both location and spread differences.

    Returns:
        distance_matrix: (n_timesteps, n_classes, n_classes) pairwise MMD
        analyzed_times: (n_timesteps,) time values
    """
    N, T_plus_1, D = atlas.z.shape
    step_indices = list(range(0, T_plus_1, step_stride))
    n_steps = len(step_indices)

    dist_matrix = torch.zeros(n_steps, n_classes, n_classes)

    for si, step in enumerate(step_indices):
        # Collect per-class point clouds at this timestep
        class_points: list[Tensor] = []
        for c in range(n_classes):
            mask = atlas.labels == c
            class_points.append(atlas.z[mask, step])

        # Estimate bandwidth using the median heuristic on a subsample
        all_pts = torch.cat(class_points, dim=0)
        subsample_idx = torch.randperm(len(all_pts))[: min(200, len(all_pts))]
        subsample = all_pts[subsample_idx]
        pairwise_sq = torch.cdist(subsample, subsample).pow(2)
        # Use upper triangle for median (exclude diagonal zeros)
        triu_mask = torch.triu(
            torch.ones_like(pairwise_sq, dtype=torch.bool), diagonal=1
        )
        median_dist = pairwise_sq[triu_mask].median().sqrt().item()
        bandwidth = max(median_dist, 1e-5)

        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                mmd = _mmd_squared(class_points[i], class_points[j], bandwidth).sqrt()
                dist_matrix[si, i, j] = mmd
                dist_matrix[si, j, i] = mmd

    return {
        "distance_matrix": dist_matrix,
        "analyzed_times": atlas.timesteps[step_indices],
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_topology_results(
    branching: dict[str, Any],
    persistence: dict[str, Tensor],
    cross_dist: dict[str, Tensor],
    out_dir: Path,
) -> None:
    """Plot topology results (Figures 4a-4d)."""
    n_classes = 10
    cmap = plt.get_cmap("tab10")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Fig 4a: Counterfactual branching visualization
    ax = axes[0, 0]
    branched = branching["branched"]
    branch_times = sorted(branched.keys())
    original = branching["original"]
    fresh = branching["fresh_target"]

    n_show = min(5, original.shape[0])
    n_cols = len(branch_times) + 2  # original + branch times + fresh target
    img_grid = torch.zeros(n_show, n_cols, 28, 28)

    for i in range(n_show):
        img_grid[i, 0] = ((original[i].view(28, 28) + 1) / 2).clamp(0, 1)
        for j, tb in enumerate(branch_times):
            img_grid[i, j + 1] = ((branched[tb][i].view(28, 28) + 1) / 2).clamp(0, 1)
        img_grid[i, -1] = ((fresh[i].view(28, 28) + 1) / 2).clamp(0, 1)

    grid_flat = img_grid.permute(0, 2, 1, 3).reshape(n_show * 28, n_cols * 28)
    ax.imshow(grid_flat.numpy(), cmap="gray", vmin=0, vmax=1)
    ax.set_title("(a) Counterfactual branching (left=source, right=fresh target)")
    ax.axis("off")

    # Fig 4b: Point of no return curves
    ax = axes[0, 1]
    mse_vs_branch = []
    for tb in branch_times:
        mse = ((branched[tb] - fresh) ** 2).mean().item()
        mse_vs_branch.append((tb, mse))
    times_b, mses = zip(*mse_vs_branch)
    ax.plot(times_b, mses, "b-o", linewidth=2)
    ax.set_xlabel("Branch time $t_{\\mathrm{branch}}$")
    ax.set_ylabel("MSE(branched, fresh target)")
    ax.set_title("(b) Point of no return")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # Fig 4c: H0 total persistence vs time per class
    ax = axes[1, 0]
    times_p = persistence["analyzed_times"].numpy()
    for c in range(n_classes):
        ax.plot(
            times_p,
            persistence["total_persistence_h0"][c].numpy(),
            color=cmap(c / 10),
            alpha=0.7,
            label=str(c),
        )
    ax.set_xlabel("Time $t$")
    ax.set_ylabel("Total $H_0$ persistence")
    ax.set_title("(c) $H_0$ convergence dynamics")
    ax.legend(fontsize=7, ncol=3)

    # Fig 4d: Cross-class distance heatmaps
    ax = axes[1, 1]
    dist_matrix = cross_dist["distance_matrix"]
    times_d = cross_dist["analyzed_times"].numpy()
    # Show at 5 selected timepoints
    show_indices = [0, len(times_d) // 4, len(times_d) // 2, 3 * len(times_d) // 4, -1]

    ax.set_visible(False)
    for k, si in enumerate(show_indices):
        sub_ax = fig.add_axes((0.55 + k * 0.085, 0.05, 0.07, 0.35))
        t_val = times_d[si]
        im = sub_ax.imshow(dist_matrix[si].numpy(), cmap="viridis")
        sub_ax.set_title(f"$t={t_val:.2f}$", fontsize=8)
        sub_ax.set_xticks(range(0, 10, 3))
        sub_ax.set_yticks(range(0, 10, 3))
        if k == 0:
            sub_ax.set_ylabel("Class")
        if k == len(show_indices) - 1:
            plt.colorbar(im, ax=sub_ax, fraction=0.046)

    plt.tight_layout()
    out_path = out_dir / "topology.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved topology figure to {out_path}")


def main(args: argparse.Namespace) -> None:
    utils.setup_torch()
    device = utils.get_torch_device().type

    print("Loading atlas...")
    atlas = load_atlas(args.experiment, args.checkpoint_dir)

    out_dir = Path(args.checkpoint_dir) / args.experiment / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Experiment 3a: Counterfactual branching
    print("\nRunning counterfactual branching...")
    model = load_checkpoint(args.experiment, args.checkpoint_dir, device)
    ema_key = list(model.ema.keys())[1]
    params = model.swap_ema(decay=ema_key)

    with torch.no_grad():
        branching = counterfactual_branching(
            model, atlas, source_class=0, target_class=1, device=device
        )

    model.swap_params(params)

    # Experiment 3b: Persistent homology
    print("\nComputing persistent homology...")
    persistence = compute_persistence(atlas, step_stride=args.step_stride)

    # Experiment 3c: Cross-class distances (MMD)
    print("\nComputing cross-class MMD distances...")
    cross_dist = compute_cross_class_distances(atlas, step_stride=args.step_stride)

    # Save results
    torch.save(
        {
            "branching": branching,
            "persistence": persistence,
            "cross_dist": cross_dist,
        },
        out_dir / "topology_results.pt",
    )
    print(f"Saved results to {out_dir / 'topology_results.pt'}")

    # Plot
    plot_topology_results(branching, persistence, cross_dist, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="default")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--step-stride", type=int, default=5)
    args = parser.parse_args()
    main(args)
