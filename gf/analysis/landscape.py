"""The Generative Ascent — full trajectory visualization.

Time as the vertical axis: each trajectory becomes a 3D curve (PC1(t), PC2(t), t)
rising from the noise floor (t=0) to structured class basins (t=1).  At the
bottom all trajectories are intertwined in an isotropic cloud; as they ascend
they separate into distinct digit bundles — the funneling from entanglement to
separation is the visual story.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import median_filter
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA

from gf import utils
from gf.analysis.atlas import TrajectoryAtlas, load_atlas
from gf.analysis.figures import CLASS_COLORS, apply_style, save_fig

# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------


def transform_density(
    density: NDArray[np.floating], gamma: float = 0.35
) -> NDArray[np.floating]:
    """Power-transform density to compress dynamic range.

    Plots density^gamma — tall narrow spikes become shorter while low broad
    features get relatively amplified, producing readable rolling hills.
    """
    return np.power(density, gamma)


def fit_pca_on_endpoints(atlas: TrajectoryAtlas) -> PCA:
    """Fit PCA(n=2) on t=1.0 endpoints where class separation is maximal."""
    z_end = atlas.z[:, -1].numpy()
    pca = PCA(n_components=2)
    pca.fit(z_end)
    return pca


def compute_landscape_grid(
    atlas: TrajectoryAtlas,
    pca: PCA,
    time_fractions: list[float] | None = None,
    grid_resolution: int = 150,
    n_classes: int = 10,
    bw_factor: float = 1.5,
) -> dict[str, object]:
    """Compute KDE density grids at selected time slices.

    Args:
        atlas: Trajectory atlas with shape (N, T+1, D).
        pca: Fitted PCA(n=2) for projection.
        time_fractions: Fractional times in [0, 1] to evaluate.
        grid_resolution: Number of grid points per axis.
        n_classes: Number of classes.
        bw_factor: Multiplier for Scott's rule bandwidth (>1 = smoother).

    Returns:
        Dictionary with meshgrid, per-slice densities, dominant classes, and metadata.
    """
    if time_fractions is None:
        time_fractions = [0.0, 0.25, 0.5, 0.75, 1.0]

    timesteps = atlas.timesteps.numpy()

    # Map fractional times to nearest timestep indices
    time_indices = []
    for tf in time_fractions:
        idx = int(np.argmin(np.abs(timesteps - tf)))
        time_indices.append(idx)

    # Project all positions at all selected times to find global extent
    all_projected = []
    for ti in time_indices:
        z_ti = atlas.z[:, ti].numpy()
        z_2d = pca.transform(z_ti)
        all_projected.append(z_2d)
    all_projected_arr = np.concatenate(all_projected, axis=0)

    # Shared meshgrid with some padding
    pad = 0.1
    x_min, x_max = all_projected_arr[:, 0].min(), all_projected_arr[:, 0].max()
    y_min, y_max = all_projected_arr[:, 1].min(), all_projected_arr[:, 1].max()
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= pad * x_range
    x_max += pad * x_range
    y_min -= pad * y_range
    y_max += pad * y_range

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_resolution),
        np.linspace(y_min, y_max, grid_resolution),
    )
    grid_points = np.vstack([xx.ravel(), yy.ravel()])

    slices: dict[float, dict[str, NDArray[np.floating]]] = {}

    for tf, ti in zip(time_fractions, time_indices):
        z_ti = atlas.z[:, ti].numpy()
        z_2d = pca.transform(z_ti)

        # Per-class KDE
        per_class_density = np.zeros((n_classes, grid_resolution, grid_resolution))

        for c in range(n_classes):
            mask = (atlas.labels == c).numpy()
            pts = z_2d[mask].T  # (2, n_per_class)

            # Check for degenerate case (all points identical)
            if pts.shape[1] < 3 or np.std(pts, axis=1).min() < 1e-10:
                continue

            kde = gaussian_kde(pts)
            kde.set_bandwidth(bw_method=kde.factor * bw_factor)
            density_flat = kde(grid_points)
            per_class_density[c] = density_flat.reshape(
                grid_resolution, grid_resolution
            )

        total_density = per_class_density.sum(axis=0)
        dominant_class = per_class_density.argmax(axis=0)

        slices[tf] = {
            "total_density": total_density,
            "dominant_class": dominant_class,
            "per_class_density": per_class_density,
            "z_2d": z_2d,
        }

    return {
        "xx": xx,
        "yy": yy,
        "slices": slices,
        "time_fractions": time_fractions,
        "time_indices": time_indices,
    }


def project_trajectories_to_surface(
    atlas: TrajectoryAtlas,
    pca: PCA,
    landscape: dict[str, object],
    n_per_class: int = 3,
    n_classes: int = 10,
) -> list[dict[str, NDArray[np.floating] | int]]:
    """Project a subset of trajectories into 2D and compute density height.

    Selects n_per_class trajectories per class and projects all timesteps.
    Height at each point is computed via nearest-grid lookup in the t=1.0 density.

    Returns:
        List of dicts with keys: x, y, z (height), class_label.
    """
    xx: NDArray[np.floating] = landscape["xx"]  # type: ignore[assignment]
    yy: NDArray[np.floating] = landscape["yy"]  # type: ignore[assignment]
    slices: dict[float, dict[str, NDArray[np.floating]]] = landscape["slices"]  # type: ignore[assignment]

    N, T_plus_1, D = atlas.z.shape

    trajectories: list[dict[str, NDArray[np.floating] | int]] = []

    for c in range(n_classes):
        mask = (atlas.labels == c).numpy()
        indices = np.where(mask)[0]
        selected = indices[:n_per_class]

        for idx in selected:
            # Project full trajectory to 2D
            z_traj = atlas.z[idx].numpy()  # (T+1, D)
            z_2d = pca.transform(z_traj)  # (T+1, 2)

            # Compute height at each timestep using corresponding time slice density
            # Fall back to t=1.0 slice for height
            t1_density = slices[1.0]["total_density"]
            heights = np.zeros(T_plus_1)
            for t_idx in range(T_plus_1):
                # Find nearest grid point
                xi = np.argmin(np.abs(xx[0, :] - z_2d[t_idx, 0]))
                yi = np.argmin(np.abs(yy[:, 0] - z_2d[t_idx, 1]))
                heights[t_idx] = t1_density[yi, xi]

            trajectories.append(
                {
                    "x": z_2d[:, 0],
                    "y": z_2d[:, 1],
                    "z": heights,
                    "class_label": int(c),
                }
            )

    return trajectories


def build_posterior_colors(
    per_class_density: NDArray[np.floating],
    total_density: NDArray[np.floating],
    height: NDArray[np.floating],
    n_classes: int = 10,
    floor_frac: float = 0.02,
    saturation_boost: float = 1.3,
    shade: bool = True,
) -> NDArray[np.floating]:
    """Soft posterior class blending for terrain coloring.

    Computes p(c|x,y) = per_class_density[c] / total_density at each grid
    point, then blends class RGB values weighted by posteriors. Peaks get
    pure class color; boundaries show smooth gradients.

    Returns (rows, cols, 4) RGBA array.
    """
    from matplotlib.colors import LightSource

    rows, cols = total_density.shape

    # Posterior: p(c|x,y)
    safe_total = np.where(total_density > 0, total_density, 1.0)
    posteriors = per_class_density / safe_total[np.newaxis, :, :]  # (C, rows, cols)

    # Class RGB lookup (n_classes, 3)
    class_rgb = np.array([CLASS_COLORS[c % n_classes][:3] for c in range(n_classes)])

    # Blend: rgb[i,j,d] = sum_c posteriors[c,i,j] * class_rgb[c,d]
    blended = np.einsum("cij,cd->ijd", posteriors, class_rgb)

    # Saturation boost — push away from gray toward dominant hue
    gray = blended.mean(axis=-1, keepdims=True)
    blended = gray + saturation_boost * (blended - gray)
    blended = np.clip(blended, 0, 1)

    # Terrain shading via LightSource
    if shade:
        ls = LightSource(azdeg=315, altdeg=45)
        blended = ls.shade_rgb(blended, height, blend_mode="soft")

    # Build RGBA
    colors = np.zeros((rows, cols, 4))
    colors[..., :3] = blended

    # Density-modulated alpha
    d_max = total_density.max()
    d_norm = total_density / d_max if d_max > 0 else np.zeros_like(total_density)
    colors[..., 3] = 0.2 + 0.8 * d_norm

    # Density floor: neutral gray at low alpha
    low_mask = d_norm < floor_frac
    colors[low_mask] = (0.92, 0.92, 0.92, 0.15)

    return colors


def extract_ridgelines(
    xx: NDArray[np.floating],
    yy: NDArray[np.floating],
    dominant_class: NDArray[np.floating],
    height: NDArray[np.floating],
    n_classes: int = 10,
    z_offset_frac: float = 0.003,
) -> list[NDArray[np.floating]]:
    """Detect territorial boundaries and project onto the 3D surface.

    Finds where the dominant class changes (decision boundaries) and
    returns them as 3D line segments sitting just above the surface.
    """
    # Smooth dominant class to remove single-pixel noise
    dom_smooth = median_filter(dominant_class.astype(float), size=3)

    # Extract boundary contours at half-integer levels
    levels = [c + 0.5 for c in range(n_classes - 1)]
    tmp_fig, tmp_ax = plt.subplots()
    cs = tmp_ax.contour(xx, yy, dom_smooth, levels=levels)
    plt.close(tmp_fig)

    # Interpolator for surface height
    x_1d = xx[0, :]
    y_1d = yy[:, 0]
    interp = RegularGridInterpolator(
        (y_1d, x_1d), height, method="linear", bounds_error=False, fill_value=0.0
    )

    z_off = z_offset_frac * (height.max() - height.min())

    ridgelines: list[NDArray[np.floating]] = []
    for level_segs in cs.allsegs:
        for verts in level_segs:
            if len(verts) < 2:
                continue
            # Interpolate height at boundary points
            pts_yx = np.column_stack([verts[:, 1], verts[:, 0]])
            z_vals = interp(pts_yx) + z_off
            line_3d = np.column_stack([verts[:, 0], verts[:, 1], z_vals])
            ridgelines.append(line_3d)

    return ridgelines


def find_saddle_points(
    xx: NDArray[np.floating],
    yy: NDArray[np.floating],
    dominant_class: NDArray[np.floating],
    total_density: NDArray[np.floating],
    min_boundary_pixels: int = 5,
) -> list[dict]:
    """Identify mountain passes between adjacent class territories.

    For each pair of neighboring classes, finds the boundary pixel with
    maximum total density — the saddle point / col / pass.
    """
    rows, cols = dominant_class.shape
    dom = dominant_class.astype(int)

    # Build boundary mask via diff along both axes
    diff_x = np.diff(dom, axis=1) != 0  # (rows, cols-1)
    diff_y = np.diff(dom, axis=0) != 0  # (rows-1, cols)

    # Collect boundary pixels with their class-pair info
    pair_pixels: dict[tuple[int, int], list[tuple[int, int]]] = {}

    # Horizontal boundaries
    for i in range(rows):
        for j in range(cols - 1):
            if diff_x[i, j]:
                c1, c2 = sorted([dom[i, j], dom[i, j + 1]])
                pair_pixels.setdefault((c1, c2), []).append((i, j))

    # Vertical boundaries
    for i in range(rows - 1):
        for j in range(cols):
            if diff_y[i, j]:
                c1, c2 = sorted([dom[i, j], dom[i + 1, j]])
                pair_pixels.setdefault((c1, c2), []).append((i, j))

    saddles: list[dict] = []
    for (c1, c2), pixels in pair_pixels.items():
        if len(pixels) < min_boundary_pixels:
            continue
        # Find pixel with max total density among boundary pixels
        best_density = -1.0
        best_ij = pixels[0]
        for i, j in pixels:
            if total_density[i, j] > best_density:
                best_density = total_density[i, j]
                best_ij = (i, j)
        saddles.append(
            {
                "classes": (c1, c2),
                "x": float(xx[best_ij]),
                "y": float(yy[best_ij]),
                "density": float(best_density),
            }
        )

    return saddles


def find_peak_summits(
    xx: NDArray[np.floating],
    yy: NDArray[np.floating],
    per_class_density: NDArray[np.floating],
    n_classes: int = 10,
) -> list[dict]:
    """Extract per-class density peak locations."""
    summits: list[dict] = []
    for c in range(n_classes):
        class_density = per_class_density[c]
        if class_density.max() < 1e-10:
            continue
        peak_idx = np.unravel_index(class_density.argmax(), class_density.shape)
        summits.append(
            {
                "class_label": c,
                "x": float(xx[peak_idx]),
                "y": float(yy[peak_idx]),
            }
        )
    return summits


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_landscape(
    landscape: dict[str, object],
    trajectories: list[dict[str, NDArray[np.floating] | int]],
    out_dir: Path,
    n_classes: int = 10,
    gamma: float = 0.35,
) -> None:
    """Plot the generative ascent — noise to structure in one figure.

    Panel A (left, 60%): 3D trajectory fiber bundle with time as the vertical
    axis.  Trajectories rise from t=0 (noise) to t=1 (structure) with
    time-gradient rendering showing the funneling from entanglement to
    class-resolved separation.

    Panel B (right, 40%): 2D territorial flow map with time-gradient trajectory
    paths over the posterior density landscape.
    """
    from matplotlib.collections import LineCollection
    from matplotlib.gridspec import GridSpec
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    apply_style()

    xx: NDArray[np.floating] = landscape["xx"]  # type: ignore[assignment]
    yy: NDArray[np.floating] = landscape["yy"]  # type: ignore[assignment]
    slices: dict[float, dict[str, NDArray[np.floating]]] = landscape["slices"]  # type: ignore[assignment]

    # Use t=1.0 slice for background features
    s_final = slices[1.0]
    total_density = s_final["total_density"]
    per_class = s_final["per_class_density"]
    dominant = s_final["dominant_class"]

    height = transform_density(total_density, gamma)

    # Terrain features for Panel B
    posterior_colors_flat = build_posterior_colors(
        per_class, total_density, height, n_classes, shade=False
    )
    saddles = find_saddle_points(xx, yy, dominant, total_density)
    summits = find_peak_summits(xx, yy, per_class, n_classes)

    # -------------------------------------------------------------------
    # Figure layout
    # -------------------------------------------------------------------
    fig = plt.figure(figsize=(20, 9))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[3, 2], wspace=0.08)

    # ===================================================================
    # Panel A — Generative Ascent (3D)
    # ===================================================================
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")

    for traj in trajectories:
        c = int(traj["class_label"])
        rgb = CLASS_COLORS[c][:3]
        x_arr = np.asarray(traj["x"])
        y_arr = np.asarray(traj["y"])
        n_pts = len(x_arr)
        t_arr = np.linspace(0.0, 1.0, n_pts)
        n_seg = n_pts - 1

        # Build 3D segments: (x(t), y(t), t)
        points = np.column_stack([x_arr, y_arr, t_arr])
        segments = np.stack([points[:-1], points[1:]], axis=1)  # (n_seg, 2, 3)

        # Per-segment alpha and linewidth ramp
        seg_frac = np.linspace(0.0, 1.0, n_seg)
        alphas = 0.05 + 0.65 * seg_frac
        lws = 0.3 + 1.1 * seg_frac
        colors = np.zeros((n_seg, 4))
        colors[:, :3] = rgb
        colors[:, 3] = alphas

        lc = Line3DCollection(segments, colors=colors, linewidths=lws)
        ax3d.add_collection3d(lc)

    # Start markers at z=0: small gray dots
    starts_x = np.array([np.asarray(t["x"])[0] for t in trajectories])
    starts_y = np.array([np.asarray(t["y"])[0] for t in trajectories])
    ax3d.scatter(
        starts_x,
        starts_y,
        np.zeros(len(trajectories)),
        c=[(0.6, 0.6, 0.6)],
        s=4,
        alpha=0.3,
        zorder=1,
    )

    # End markers at z=1: colored dots with white edge
    for c_idx in range(n_classes):
        class_trajs = [t for t in trajectories if int(t["class_label"]) == c_idx]
        if not class_trajs:
            continue
        ends_x = np.array([np.asarray(t["x"])[-1] for t in class_trajs])
        ends_y = np.array([np.asarray(t["y"])[-1] for t in class_trajs])
        ax3d.scatter(
            ends_x,
            ends_y,
            np.ones(len(class_trajs)),
            c=[CLASS_COLORS[c_idx]],
            s=15,
            edgecolors="white",
            linewidths=0.4,
            alpha=0.9,
            zorder=5,
        )

    # Summit labels at z=1.04 above class centroid
    for s in summits:
        c = s["class_label"]
        ax3d.text(
            s["x"],
            s["y"],
            1.04,
            str(c),
            fontsize=9,
            fontweight="bold",
            color=CLASS_COLORS[c],
            ha="center",
            va="bottom",
            bbox={
                "boxstyle": "round,pad=0.15",
                "fc": "white",
                "alpha": 0.85,
                "lw": 0,
            },
        )

    # Axis limits and camera
    ax3d.set_xlim(float(xx.min()), float(xx.max()))
    ax3d.set_ylim(float(yy.min()), float(yy.max()))
    ax3d.set_zlim(0, 1.08)
    ax3d.view_init(elev=18, azim=-55)

    # Subtle pane styling
    for pane in [ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane]:
        pane.fill = True
        pane.set_facecolor((0.85, 0.85, 0.85, 0.3))
        pane.set_edgecolor((0.85, 0.85, 0.85, 0.3))

    ax3d.grid(True, alpha=0.1)
    ax3d.set_xlabel("PC 1")
    ax3d.set_ylabel("PC 2")
    ax3d.set_zlabel("Time $t$")
    ax3d.set_xticks([])
    ax3d.set_yticks([])
    ax3d.set_title("(a) Generative Ascent", fontsize=12, pad=10)

    # ===================================================================
    # Panel B — Territorial Flow Map (2D)
    # ===================================================================
    ax2d = fig.add_subplot(gs[0, 1])

    # Background: posterior color map (unshaded)
    extent = (float(xx.min()), float(xx.max()), float(yy.min()), float(yy.max()))
    ax2d.imshow(
        posterior_colors_flat,
        origin="lower",
        extent=extent,
        aspect="auto",
        interpolation="bilinear",
    )

    # Topographic contour lines
    ax2d.contour(xx, yy, height, levels=10, colors="k", linewidths=0.4, alpha=0.3)

    # Ridgeline boundaries
    dom_smooth = median_filter(dominant.astype(float), size=3)
    ridge_levels = [c + 0.5 for c in range(n_classes - 1)]
    ax2d.contour(
        xx,
        yy,
        dom_smooth,
        levels=ridge_levels,
        colors=[(0.2, 0.2, 0.2)],
        linewidths=0.8,
        alpha=0.6,
    )

    # Trajectory flow paths with time-gradient
    for traj in trajectories:
        c = int(traj["class_label"])
        rgb = CLASS_COLORS[c][:3]
        x_arr = np.asarray(traj["x"])
        y_arr = np.asarray(traj["y"])
        n_pts = len(x_arr)
        n_seg = n_pts - 1

        points = np.column_stack([x_arr, y_arr])
        segments = np.stack([points[:-1], points[1:]], axis=1)  # (n_seg, 2, 2)

        seg_frac = np.linspace(0.0, 1.0, n_seg)
        alphas = 0.03 + 0.47 * seg_frac
        lws = 0.15 + 0.55 * seg_frac
        colors = np.zeros((n_seg, 4))
        colors[:, :3] = rgb
        colors[:, 3] = alphas

        lc = LineCollection(list(segments), colors=colors, linewidths=lws)
        ax2d.add_collection(lc)

    # Saddle markers
    for sd in saddles:
        ax2d.plot(
            sd["x"],
            sd["y"],
            marker="^",
            markersize=4,
            color=(0.3, 0.3, 0.3),
            markeredgecolor="white",
            markeredgewidth=0.4,
            zorder=5,
        )

    # Summit labels
    for s in summits:
        c = s["class_label"]
        ax2d.text(
            s["x"],
            s["y"],
            str(c),
            fontsize=9,
            fontweight="bold",
            color=CLASS_COLORS[c],
            ha="center",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.15",
                "fc": "white",
                "alpha": 0.85,
                "lw": 0,
            },
            zorder=6,
        )

    ax2d.set_xlabel("PC 1")
    ax2d.set_ylabel("PC 2")
    ax2d.set_title("(b) Territorial Flow Map", fontsize=12)

    fig.suptitle("The Generative Ascent", fontsize=14, y=0.97)
    save_fig(fig, out_dir / "figures", "landscape")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    utils.setup_torch()

    print("Loading atlas...")
    atlas = load_atlas(args.experiment, args.checkpoint_dir)
    print(f"  z: {atlas.z.shape}, labels: {atlas.labels.shape}")

    out_dir = Path(args.checkpoint_dir) / args.experiment / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\nFitting PCA on t=1.0 endpoints...")
    pca = fit_pca_on_endpoints(atlas)
    print(f"  Explained variance: {pca.explained_variance_ratio_}")

    time_fractions = [0.0, 0.25, 0.5, 0.75, 1.0]

    print("\nComputing landscape grids...")
    landscape = compute_landscape_grid(
        atlas,
        pca,
        time_fractions=time_fractions,
        grid_resolution=args.grid_resolution,
    )

    print("Projecting trajectories to surface...")
    trajectories = project_trajectories_to_surface(atlas, pca, landscape, n_per_class=8)
    print(f"  {len(trajectories)} trajectories projected")

    # Save results
    results = {
        "xx": torch.from_numpy(landscape["xx"]),
        "yy": torch.from_numpy(landscape["yy"]),
        "time_fractions": time_fractions,
        "pca_components": torch.from_numpy(pca.components_),
        "pca_mean": torch.from_numpy(pca.mean_),
        "trajectories": [
            {
                "x": torch.from_numpy(t["x"]),
                "y": torch.from_numpy(t["y"]),
                "z": torch.from_numpy(t["z"]),
                "class_label": t["class_label"],
            }
            for t in trajectories
        ],
    }

    # Save per-slice data
    slice_data: dict[str, dict[str, torch.Tensor]] = {}
    slices = landscape["slices"]
    for tf in time_fractions:
        s = slices[tf]  # type: ignore[index]
        slice_data[str(tf)] = {
            "total_density": torch.from_numpy(s["total_density"]),
            "dominant_class": torch.from_numpy(s["dominant_class"].astype(np.int64)),
            "per_class_density": torch.from_numpy(s["per_class_density"]),
        }
    results["slice_data"] = slice_data

    results_path = out_dir / "landscape_results.pt"
    torch.save(results, results_path)
    print(f"\nSaved results to {results_path}")

    print("Plotting landscape...")
    plot_landscape(landscape, trajectories, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="default")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--grid-resolution", type=int, default=150)
    args = parser.parse_args()
    main(args)
