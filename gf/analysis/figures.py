"""Phase 4: Figure Assembly — paper-ready figures with consistent styling.

Orchestrates all paper figures from pre-computed results.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from sklearn.decomposition import PCA

from gf import utils
from gf.analysis.atlas import TrajectoryAtlas, load_atlas

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

STYLE = {
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
}

CMAP = plt.get_cmap("tab10")
CLASS_COLORS = [CMAP(i / 10) for i in range(10)]


def apply_style():
    matplotlib.rcParams.update(STYLE)


def save_fig(fig: plt.Figure, out_dir: Path, name: str) -> None:
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=300)
    print(f"Saved {name}.png to {out_dir}")


# ---------------------------------------------------------------------------
# Fig 1: Overview — PCA trajectory atlas
# ---------------------------------------------------------------------------


def fig1_trajectory_atlas(atlas: TrajectoryAtlas, out_dir: Path) -> None:
    """PCA projection of trajectories colored by class and time."""
    apply_style()

    N, T_plus_1, D = atlas.z.shape
    step_indices = list(range(0, T_plus_1, 5))
    z_sub = atlas.z[:, step_indices].reshape(-1, D).numpy()

    pca = PCA(n_components=2)
    z_pca = pca.fit_transform(z_sub)
    z_pca = z_pca.reshape(N, len(step_indices), 2)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel A: colored by class
    ax = axes[0]
    for i in range(N):
        c = int(atlas.labels[i].item())
        ax.plot(
            z_pca[i, :, 0],
            z_pca[i, :, 1],
            color=CLASS_COLORS[c],
            alpha=0.12,
            linewidth=0.4,
        )
    for c in range(10):
        ax.plot([], [], color=CLASS_COLORS[c], label=str(c), linewidth=2)
    ax.legend(title="Digit", fontsize=7, ncol=2, loc="best", framealpha=0.8)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title("(a) Colored by class")

    # Panel B: colored by time
    ax = axes[1]
    cmap_time = plt.get_cmap("viridis")
    t_sub = atlas.timesteps[step_indices].numpy()
    for i in range(min(N, 200)):
        for j in range(len(step_indices) - 1):
            ax.plot(
                z_pca[i, j : j + 2, 0],
                z_pca[i, j : j + 2, 1],
                color=cmap_time(t_sub[j]),
                alpha=0.15,
                linewidth=0.4,
            )
    sm = plt.cm.ScalarMappable(
        cmap=cmap_time,
        norm=plt.Normalize(vmin=0, vmax=1),
    )
    plt.colorbar(sm, ax=ax, label="Time $t$")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title("(b) Colored by time")

    fig.suptitle("Fig 1: Trajectory Atlas", fontsize=13, y=1.02)
    plt.tight_layout()
    save_fig(fig, out_dir, "fig1_trajectory_atlas")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 2: Hodge decomposition results
# ---------------------------------------------------------------------------


def fig2_hodge_decomposition(
    results_path: Path, atlas: TrajectoryAtlas, out_dir: Path
) -> None:
    """Plot Helmholtz-Hodge decomposition results (neural + graph Hodge)."""
    apply_style()

    data = torch.load(results_path, map_location="cpu")
    results_cond = data["results_cond"]
    results_delta = data["results_delta"]
    timesteps = atlas.timesteps[:-1].numpy()

    has_graph = "graph_hodge_cond" in data
    n_rows = 3 if has_graph else 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(10, 4 * n_rows))

    # 2a: R^2 for conditional velocity
    ax = axes[0, 0]
    for c in range(10):
        ax.plot(
            timesteps,
            results_cond["r_squared_per_class"][c].numpy(),
            color=CLASS_COLORS[c],
            alpha=0.6,
            label=str(c),
        )
    ax.plot(timesteps, results_cond["r_squared"].numpy(), "k--", lw=2, label="mean")
    ax.set_xlabel("Time $t$")
    ax.set_ylabel("$R^2(t)$")
    ax.set_title("(a) Neural gradient fraction — $v_{\\mathrm{cond}}$")
    ax.legend(fontsize=6, ncol=3)
    ax.set_ylim(-0.1, 1.1)

    # 2b: R^2 for delta_v
    ax = axes[0, 1]
    for c in range(10):
        ax.plot(
            timesteps,
            results_delta["r_squared_per_class"][c].numpy(),
            color=CLASS_COLORS[c],
            alpha=0.6,
            label=str(c),
        )
    ax.plot(timesteps, results_delta["r_squared"].numpy(), "k--", lw=2, label="mean")
    ax.set_xlabel("Time $t$")
    ax.set_ylabel("$R^2(t)$")
    ax.set_title("(b) Neural gradient fraction — $\\Delta v$")
    ax.legend(fontsize=6, ncol=3)
    ax.set_ylim(-0.1, 1.1)

    # 2c: Scalar potential along trajectories
    ax = axes[1, 0]
    energy = results_cond["energy_along_traj"]
    for c in range(10):
        mask = atlas.labels == c
        e_class = energy[mask].mean(dim=0).numpy()
        ax.plot(timesteps, e_class, color=CLASS_COLORS[c], label=str(c))
    ax.set_xlabel("Time $t$")
    ax.set_ylabel("Scalar potential $\\phi(z(t), t)$")
    ax.set_title("(c) Potential along trajectories")
    ax.legend(fontsize=6, ncol=3)

    # 2d: PCA quiver at midpoint
    ax = axes[1, 1]
    mid = len(timesteps) // 2
    z_mid = atlas.z[:, mid].numpy()
    v_grad_mid = results_cond["v_grad"][:, mid].numpy()
    v_sol_mid = results_cond["v_sol"][:, mid].numpy()

    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(z_mid)
    v_grad_2d = v_grad_mid @ pca.components_.T
    v_sol_2d = v_sol_mid @ pca.components_.T

    idx = list(range(0, len(z_2d), 10))
    ax.quiver(
        z_2d[idx, 0],
        z_2d[idx, 1],
        v_grad_2d[idx, 0],
        v_grad_2d[idx, 1],
        color="steelblue",
        alpha=0.5,
        label="$v_{\\mathrm{grad}}$",
    )
    ax.quiver(
        z_2d[idx, 0],
        z_2d[idx, 1],
        v_sol_2d[idx, 0],
        v_sol_2d[idx, 1],
        color="indianred",
        alpha=0.5,
        label="$v_{\\mathrm{sol}}$",
    )
    ax.legend()
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(f"(d) PCA quiver at $t={timesteps[mid]:.2f}$")

    # Row 3: Graph Hodge 3-way decomposition (if available)
    if has_graph:
        gh_cond = data["graph_hodge_cond"]
        gh_delta = data["graph_hodge_delta"]

        # 2e: Stacked area for v_cond
        ax = axes[2, 0]
        gh_t = gh_cond["analyzed_times"].numpy()
        g_m = gh_cond["grad_frac"].mean(dim=0).numpy()
        c_m = gh_cond["curl_frac"].mean(dim=0).numpy()
        h_m = gh_cond["harm_frac"].mean(dim=0).numpy()
        ax.stackplot(
            gh_t,
            g_m,
            c_m,
            h_m,
            labels=["Gradient", "Curl", "Harmonic"],
            colors=["steelblue", "indianred", "goldenrod"],
            alpha=0.8,
        )
        ax.set_xlabel("Time $t$")
        ax.set_ylabel("Energy fraction")
        ax.set_title("(e) Graph Hodge 3-way — $v_{\\mathrm{cond}}$")
        ax.legend(fontsize=7, loc="upper right")
        ax.set_ylim(0, 1.05)

        # 2f: Stacked area for delta_v
        ax = axes[2, 1]
        gh_t = gh_delta["analyzed_times"].numpy()
        g_m = gh_delta["grad_frac"].mean(dim=0).numpy()
        c_m = gh_delta["curl_frac"].mean(dim=0).numpy()
        h_m = gh_delta["harm_frac"].mean(dim=0).numpy()
        ax.stackplot(
            gh_t,
            g_m,
            c_m,
            h_m,
            labels=["Gradient", "Curl", "Harmonic"],
            colors=["steelblue", "indianred", "goldenrod"],
            alpha=0.8,
        )
        ax.set_xlabel("Time $t$")
        ax.set_ylabel("Energy fraction")
        ax.set_title("(f) Graph Hodge 3-way — $\\Delta v$")
        ax.legend(fontsize=7, loc="upper right")
        ax.set_ylim(0, 1.05)

    fig.suptitle("Fig 2: Helmholtz-Hodge Decomposition", fontsize=13, y=1.02)
    plt.tight_layout()
    save_fig(fig, out_dir, "fig2_hodge_decomposition")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 3: AdaLN mechanistic analysis
# ---------------------------------------------------------------------------


def fig3_adaln_analysis(
    results_path: Path, atlas: TrajectoryAtlas, out_dir: Path
) -> None:
    """Plot AdaLN analysis results."""
    apply_style()

    data = torch.load(results_path, map_location="cpu")
    fingerprints = data["fingerprints"]
    patching = data["patching"]
    timesteps = atlas.timesteps[:-1].numpy()

    n_blocks = len([k for k in fingerprints if k.endswith("_gate_magnitude")])

    fig = plt.figure(figsize=(12, 9))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # 3a: Gate magnitude
    ax = fig.add_subplot(gs[0, 0])
    for block_idx in range(n_blocks):
        gate_mag = fingerprints[f"block_{block_idx}_gate_magnitude"]
        for c in range(10):
            ls = "-" if block_idx == 0 else "--"
            ax.plot(
                timesteps,
                gate_mag[c].numpy(),
                color=CLASS_COLORS[c],
                linestyle=ls,
                alpha=0.5,
            )
    ax.plot([], [], "k-", label="Block 0")
    ax.plot([], [], "k--", label="Block 1")
    ax.legend(fontsize=8)
    ax.set_xlabel("Time $t$")
    ax.set_ylabel("$\\|\\mathrm{gate}\\|$")
    ax.set_title("(a) Gate magnitude over time")

    # 3b: Cosine similarity at 3 timepoints (using GridSpec subdivision)
    gs_cos = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0, 1], wspace=0.35)
    t_show = [0.2, 0.5, 0.8]
    cos_key = "block_0_gate_cosine_sim"
    for k, t_val in enumerate(t_show):
        ti = int(t_val * len(timesteps))
        ti = min(ti, len(timesteps) - 1)
        cos_sim = fingerprints[cos_key][ti].numpy()

        ax_cos = fig.add_subplot(gs_cos[0, k])
        im = ax_cos.imshow(cos_sim, vmin=-1, vmax=1, cmap="RdBu_r")
        ax_cos.set_title(f"$t={t_val}$", fontsize=9)
        ax_cos.set_xticks(range(0, 10, 3))
        ax_cos.set_yticks(range(0, 10, 3))
        if k == 0:
            ax_cos.set_ylabel("Class")
        if k == 2:
            plt.colorbar(im, ax=ax_cos, fraction=0.046, pad=0.04)

    # 3c: Patching images
    ax = fig.add_subplot(gs[1, 0])
    samples = patching["samples"]
    class_pairs = patching["class_pairs"]
    n_show = min(4, len(class_pairs))
    n_per = 5

    img_grid = torch.zeros(n_show * 2, n_per, 28, 28)
    row = 0
    for pair_idx in range(n_show):
        src, tgt = class_pairs[pair_idx]
        src_key = f"{src}_{tgt}_block0_source"
        pat_key = f"{src}_{tgt}_block0_patched"
        if src_key in samples and pat_key in samples:
            img_grid[row, :n_per] = (
                (samples[src_key][:n_per].view(-1, 28, 28) + 1) / 2
            ).clamp(0, 1)
            img_grid[row + 1, :n_per] = (
                (samples[pat_key][:n_per].view(-1, 28, 28) + 1) / 2
            ).clamp(0, 1)
        row += 2

    grid_flat = img_grid.permute(0, 2, 1, 3).reshape(n_show * 2 * 28, n_per * 28)
    ax.imshow(grid_flat.numpy(), cmap="gray", vmin=0, vmax=1)
    ax.set_title("(c) Counterfactual patching (source / patched)")
    ax.axis("off")
    # Add pair labels on the left
    for pair_idx in range(n_show):
        src, tgt = class_pairs[pair_idx]
        y_center = pair_idx * 2 * 28 + 28
        ax.text(
            -5,
            y_center,
            f"{src}$\\to${tgt}",
            ha="right",
            va="center",
            fontsize=8,
            fontweight="bold",
        )

    # 3d: Causal effect bar chart
    ax = fig.add_subplot(gs[1, 1])
    mse_results = patching["mse_results"]
    components = patching["components"]
    n_blocks_actual = mse_results.shape[1]
    avg_mse = mse_results.mean(dim=0)
    x_pos = np.arange(len(components))
    width = 0.35

    for bi in range(n_blocks_actual):
        offset = (bi - 0.5) * width
        ax.bar(x_pos + offset, avg_mse[bi].numpy(), width, label=f"Block {bi}")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(components)
    ax.set_ylabel("MSE (causal effect)")
    ax.set_title("(d) Causal effect by block/component")
    ax.legend()

    fig.suptitle("Fig 3: AdaLN Mechanistic Analysis", fontsize=13, y=1.0)
    save_fig(fig, out_dir, "fig3_adaln_analysis")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 4: Trajectory topology
# ---------------------------------------------------------------------------


def fig4_topology(results_path: Path, out_dir: Path) -> None:
    """Plot topology results."""
    apply_style()

    data = torch.load(results_path, map_location="cpu")
    branching = data["branching"]
    persistence = data["persistence"]
    cross_dist = data["cross_dist"]

    fig = plt.figure(figsize=(12, 9))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # 4a: Branching images with labels
    ax = fig.add_subplot(gs[0, 0])
    branched = branching["branched"]
    original = branching["original"]
    fresh = branching["fresh_target"]
    branch_times = sorted(branched.keys())

    n_show = min(5, original.shape[0])
    n_cols = len(branch_times) + 2

    img_grid = torch.zeros(n_show, n_cols, 28, 28)
    for i in range(n_show):
        img_grid[i, 0] = ((original[i].view(28, 28) + 1) / 2).clamp(0, 1)
        for j, tb in enumerate(branch_times):
            img_grid[i, j + 1] = ((branched[tb][i].view(28, 28) + 1) / 2).clamp(0, 1)
        img_grid[i, -1] = ((fresh[i].view(28, 28) + 1) / 2).clamp(0, 1)

    grid_flat = img_grid.permute(0, 2, 1, 3).reshape(n_show * 28, n_cols * 28)
    ax.imshow(grid_flat.numpy(), cmap="gray", vmin=0, vmax=1)
    ax.set_title("(a) Counterfactual branching ($0 \\to 1$)")
    ax.set_yticks([])
    # Column labels
    col_labels = ["src"] + [f"{tb:.1f}" for tb in branch_times] + ["tgt"]
    for j, label in enumerate(col_labels):
        ax.text(
            j * 28 + 14,
            -4,
            label,
            ha="center",
            va="bottom",
            fontsize=7,
        )
    ax.set_xticks([])
    ax.set_xlabel("$t_{\\mathrm{branch}}$", labelpad=10)

    # 4b: Point of no return
    ax = fig.add_subplot(gs[0, 1])
    mses = [((branched[tb] - fresh) ** 2).mean().item() for tb in branch_times]
    ax.plot(branch_times, mses, "o-", color="steelblue", linewidth=2, markersize=6)
    ax.set_xlabel("Branch time $t_{\\mathrm{branch}}$")
    ax.set_ylabel("MSE(branched, fresh)")
    ax.set_title("(b) Point of no return")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)

    # 4c: H0 persistence
    ax = fig.add_subplot(gs[1, 0])
    times_p = persistence["analyzed_times"].numpy()
    for c in range(10):
        ax.plot(
            times_p,
            persistence["total_persistence_h0"][c].numpy(),
            color=CLASS_COLORS[c],
            alpha=0.7,
            label=str(c),
        )
    ax.set_xlabel("Time $t$")
    ax.set_ylabel("Total $H_0$ persistence")
    ax.set_title("(c) $H_0$ convergence dynamics")
    ax.legend(fontsize=6, ncol=3)

    # 4d: Cross-class distance heatmaps (using GridSpec subdivision)
    gs_dist = GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[1, 1], wspace=0.4)
    dist_matrix = cross_dist["distance_matrix"]
    times_d = cross_dist["analyzed_times"].numpy()
    show_idxs = [0, len(times_d) // 4, len(times_d) // 2, 3 * len(times_d) // 4, -1]

    for k, si in enumerate(show_idxs):
        ax_d = fig.add_subplot(gs_dist[0, k])
        im = ax_d.imshow(dist_matrix[si].numpy(), cmap="viridis")
        ax_d.set_title(f"$t={times_d[si]:.1f}$", fontsize=8)
        if k == 0:
            ax_d.set_ylabel("Class", fontsize=8)
            ax_d.set_yticks(range(0, 10, 3))
        else:
            ax_d.set_yticks([])
        ax_d.set_xticks(range(0, 10, 3))
        ax_d.tick_params(labelsize=7)
        if k == len(show_idxs) - 1:
            plt.colorbar(im, ax=ax_d, fraction=0.046, pad=0.04)

    fig.suptitle("Fig 4: Trajectory Topology", fontsize=13, y=1.0)
    save_fig(fig, out_dir, "fig4_topology")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 5: Synthesis — single timestep, all lenses
# ---------------------------------------------------------------------------


def fig5_synthesis(
    atlas: TrajectoryAtlas,
    hodge_path: Path,
    adaln_path: Path,
    out_dir: Path,
) -> None:
    """Tie all lenses together at a single interesting timestep (t=0.5)."""
    apply_style()

    hodge_data = torch.load(hodge_path, map_location="cpu")
    adaln_data = torch.load(adaln_path, map_location="cpu")

    results_cond = hodge_data["results_cond"]
    fingerprints = adaln_data["fingerprints"]
    timesteps = atlas.timesteps[:-1].numpy()
    mid = len(timesteps) // 2

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel A: PCA positions at t=mid, colored by class
    ax = axes[0]
    z_mid = atlas.z[:, mid].numpy()
    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(z_mid)
    for c in range(10):
        mask = (atlas.labels == c).numpy()
        ax.scatter(
            z_2d[mask, 0],
            z_2d[mask, 1],
            c=[CLASS_COLORS[c]],
            alpha=0.5,
            s=15,
            label=str(c),
        )
    ax.legend(fontsize=6, ncol=2)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(f"(a) Trajectory positions at $t={timesteps[mid]:.2f}$")

    # Panel B: R^2 snapshot
    ax = axes[1]
    r2_cond = results_cond["r_squared_per_class"][:, mid].numpy()
    r2_delta = hodge_data["results_delta"]["r_squared_per_class"][:, mid].numpy()
    x = np.arange(10)
    w = 0.35
    ax.bar(x - w / 2, r2_cond, w, label="$v_{\\mathrm{cond}}$", color="steelblue")
    ax.bar(x + w / 2, r2_delta, w, label="$\\Delta v$", color="indianred")
    ax.set_xticks(x)
    ax.set_xlabel("Class")
    ax.set_ylabel("$R^2$")
    ax.set_title(f"(b) Gradient fraction at $t={timesteps[mid]:.2f}$")
    ax.legend()

    # Panel C: Graph Hodge 3-way at t=mid (or gate magnitudes if unavailable)
    ax = axes[2]
    has_graph_hodge = "graph_hodge_cond" in hodge_data
    if has_graph_hodge:
        gh_cond = hodge_data["graph_hodge_cond"]
        gh_times = gh_cond["analyzed_times"].numpy()
        # Find closest graph Hodge timestep to mid
        gh_mid = int(np.argmin(np.abs(gh_times - timesteps[mid])))
        x = np.arange(10)
        w = 0.25
        ax.bar(
            x - w,
            gh_cond["grad_frac"][:, gh_mid].numpy(),
            w,
            label="Gradient",
            color="steelblue",
        )
        ax.bar(
            x,
            gh_cond["curl_frac"][:, gh_mid].numpy(),
            w,
            label="Curl",
            color="indianred",
        )
        ax.bar(
            x + w,
            gh_cond["harm_frac"][:, gh_mid].numpy(),
            w,
            label="Harmonic",
            color="goldenrod",
        )
        ax.set_xticks(x)
        ax.set_xlabel("Class")
        ax.set_ylabel("Energy fraction")
        ax.set_title(f"(c) Graph Hodge at $t={timesteps[mid]:.2f}$")
        ax.legend(fontsize=7)
    else:
        n_blocks = len([k for k in fingerprints if k.endswith("_gate_magnitude")])
        gate_data = {}
        for bi in range(n_blocks):
            gate_data[bi] = fingerprints[f"block_{bi}_gate_magnitude"][:, mid].numpy()

        x = np.arange(10)
        w = 0.35
        for bi in range(n_blocks):
            offset = (bi - 0.5) * w
            ax.bar(x + offset, gate_data[bi], w, label=f"Block {bi}")
        ax.set_xticks(x)
        ax.set_xlabel("Class")
        ax.set_ylabel("$\\|\\mathrm{gate}\\|$")
        ax.set_title(f"(c) Gate magnitude at $t={timesteps[mid]:.2f}$")
        ax.legend()

    fig.suptitle("Fig 5: Multi-lens Synthesis", fontsize=13, y=1.02)
    plt.tight_layout()
    save_fig(fig, out_dir, "fig5_synthesis")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    utils.setup_torch()

    base = Path(args.checkpoint_dir) / args.experiment / "analysis"
    out_dir = base / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading atlas...")
    atlas = load_atlas(args.experiment, args.checkpoint_dir)

    # Fig 1
    print("Generating Fig 1...")
    fig1_trajectory_atlas(atlas, out_dir)

    # Fig 2
    hodge_path = base / "hodge_results.pt"
    if hodge_path.exists():
        print("Generating Fig 2...")
        fig2_hodge_decomposition(hodge_path, atlas, out_dir)
    else:
        print(f"Skipping Fig 2 ({hodge_path} not found)")

    # Fig 3
    adaln_path = base / "adaln_results.pt"
    if adaln_path.exists():
        print("Generating Fig 3...")
        fig3_adaln_analysis(adaln_path, atlas, out_dir)
    else:
        print(f"Skipping Fig 3 ({adaln_path} not found)")

    # Fig 4
    topo_path = base / "topology_results.pt"
    if topo_path.exists():
        print("Generating Fig 4...")
        fig4_topology(topo_path, out_dir)
    else:
        print(f"Skipping Fig 4 ({topo_path} not found)")

    # Fig 5
    if hodge_path.exists() and adaln_path.exists():
        print("Generating Fig 5...")
        fig5_synthesis(atlas, hodge_path, adaln_path, out_dir)
    else:
        print("Skipping Fig 5 (missing Hodge or AdaLN results)")

    print(f"\nAll figures saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="default")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    args = parser.parse_args()
    main(args)
