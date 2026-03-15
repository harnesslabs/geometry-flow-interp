"""Geometric mechanistic interpretability analysis.

Three experiments probing how a flow-matching JiT encodes SO(2) rotation.

Exp 1: Rotation decodability along the ODE trajectory.
Exp 2: Controlled S¹ orbit geometry from explicit rotations of generated trajectories.
Exp 3: Velocity-field Jacobian modes vs. the image-space angular derivative.
"""

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from geoflow import utils
from geoflow.denoiser import Denoiser, DenoiserConfig
from geoflow.mnist import MNIST

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Geometric mechanistic interpretability")
parser.add_argument("--experiment", type=str, default="default")
parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
parser.add_argument("--run", default="all", choices=["all", "1", "2", "3", "4"])
parser.add_argument("--n-samples", type=int, default=100, help="Samples per class")
parser.add_argument("--n-classes", type=int, default=10)
parser.add_argument(
    "--orbit-samples-per-class",
    type=int,
    default=3,
    help="Base generated trajectories per class for the controlled rotation-orbit probe",
)
parser.add_argument(
    "--orbit-angles",
    type=int,
    default=12,
    help="Number of explicit rotations used to trace each S¹ orbit in experiment 2",
)
parser.add_argument(
    "--steer-iterations",
    type=int,
    default=3,
    help="Closed-loop steering iterations for experiment 4",
)
parser.add_argument(
    "--steer-gain-angle-deg",
    type=float,
    default=12.0,
    help="Finite-difference rotation used to measure steering gain in experiment 4",
)
parser.add_argument(
    "--steer-max-step-angle-deg",
    type=float,
    default=35.0,
    help="Maximum control rotation applied at any single ODE step in experiment 4",
)
parser.add_argument(
    "--orbit-grid-rows",
    type=int,
    default=10,
    help="Number of relative orbit positions shown in the experiment 4 grid",
)

# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

OUT_DIR = Path("media/analysis")


@dataclass
class RotationDecoder:
    mu: np.ndarray
    std: np.ndarray
    W: np.ndarray


@dataclass
class RotationGeometryScan:
    angle_values: np.ndarray
    circle_scores: np.ndarray
    decode_scores: np.ndarray
    rank2_scores: np.ndarray
    geometry_score: np.ndarray
    layer_names: list[str]
    selected_indices: np.ndarray
    peak_layer: str
    peak_step: int
    t_peak: float


@dataclass
class RelativeRotationBank:
    angle_values: np.ndarray
    templates: np.ndarray


def load_checkpoint(experiment: str, checkpoint_dir: str, device: str) -> Denoiser:
    """Load EMA weights into model.net (same pattern as probe.py)."""
    path = Path(checkpoint_dir) / experiment / "last.pt"
    ckpt = torch.load(path, map_location=device)

    config = DenoiserConfig(**ckpt["config"])
    config.sampling_method = "euler"
    config.noise_scale = 0.8
    config.cfg_scale = 2.5

    model = Denoiser(config, device).to(device)

    decay = sorted(model.ema.keys())[0]
    model.ema[decay].load_state_dict(ckpt["ema"][decay])
    model.swap_ema(decay=decay)

    model.eval()
    print(f"Loaded checkpoint from {path} (step={ckpt.get('global_step', '?')})")
    return model


def pca(X: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PCA via SVD. Returns (projected, components, explained_variance_ratio)."""
    mean = X.mean(axis=0)
    Xc = X - mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    components = Vt[:n_components]
    projected = Xc @ components.T
    var = (S**2) / (X.shape[0] - 1)
    explained = var[:n_components] / var.sum()
    return projected, components, explained


def measure_rotation_pca(
    images: np.ndarray, labels: np.ndarray, n_classes: int
) -> np.ndarray:
    """Measure rotation angle proxy via per-class PCA on flattened images.

    For each class, PC1/PC2 capture the dominant within-class variation (rotation).
    Returns atan2(PC2, PC1) as angle proxy, shape (N,).
    """
    N = images.shape[0]
    angles = np.zeros(N)
    flat = images.reshape(N, -1)  # (N, 784)

    for cls in range(n_classes):
        mask = labels == cls
        proj, _, _ = pca(flat[mask], 2)
        angles[mask] = np.arctan2(proj[:, 1], proj[:, 0])

    return angles


def _normalize_to_circle(proj: np.ndarray) -> np.ndarray:
    """Project 2D points onto the unit circle (radial normalization to S¹)."""
    r = np.linalg.norm(proj, axis=1, keepdims=True)
    return proj / (r + 1e-12)


def generate_with_trajectory(
    model: Denoiser,
    cond: torch.Tensor,
    z0: torch.Tensor | None = None,
    steer_fn: Callable[[torch.Tensor, int, float, torch.Tensor], torch.Tensor]
    | None = None,
    start_step: int = 0,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """ODE generation saving z_t at each step. Returns (final_z, trajectory)."""
    B = cond.size(0)
    C = model.net.in_channels
    H = W = model.net.input_size
    device = cond.device

    if start_step < 0 or start_step > model.config.num_sampling_steps:
        raise ValueError("start_step out of range")
    if start_step > 0 and z0 is None:
        raise ValueError("z0 must be provided when start_step > 0")

    z = (
        z0.clone()
        if z0 is not None
        else model.config.noise_scale * torch.randn(B, C, H, W, device=device)
    )
    t_values = torch.linspace(
        0.0, 1.0, model.config.num_sampling_steps + 1, device=device
    )

    trajectory = [z.detach().cpu()]

    for i in range(start_step, model.config.num_sampling_steps):
        if steer_fn is not None:
            z = steer_fn(z, i, float(t_values[i].item()), cond)
            trajectory[-1] = z.detach().cpu()

        t = t_values[i].view(1, *([1] * (z.ndim - 1))).expand(B, *([-1] * (z.ndim - 1)))
        s = (
            t_values[i + 1]
            .view(1, *([1] * (z.ndim - 1)))
            .expand(B, *([-1] * (z.ndim - 1)))
        )
        z = model._euler_step(z, t, s, cond)
        trajectory.append(z.detach().cpu())

    return z, trajectory


def ridge_regression(X: np.ndarray, Y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Closed-form ridge regression, using the dual form when D >> N."""
    N, D = X.shape
    if D <= N:
        return np.linalg.solve(X.T @ X + alpha * np.eye(D), X.T @ Y)

    dual = np.linalg.solve(X @ X.T + alpha * np.eye(N), Y)
    return X.T @ dual


# ---------------------------------------------------------------------------
# Experiment 1: Rotation decodability along ODE trajectory
# ---------------------------------------------------------------------------


def exp1_rotation_decodability(
    trajectory: list[torch.Tensor],
    labels: np.ndarray,
    angles: np.ndarray,
    n_classes: int,
    n_pca: int = 50,
) -> None:
    """Ridge-regress (sin θ, cos θ) from PCA-reduced z_t at each ODE step."""
    print("=== Experiment 1: Rotation decodability ===")

    Y = np.stack([np.sin(angles), np.cos(angles)], axis=1)  # (N, 2)
    N = Y.shape[0]
    n_steps = len(trajectory)

    # 80/20 split
    perm = np.random.RandomState(42).permutation(N)
    split = int(0.8 * N)
    train_idx, test_idx = perm[:split], perm[split:]

    # Per-class and average R²
    r2_all = np.zeros(n_steps)  # average
    r2_per_class = np.zeros((n_classes, n_steps))

    for step_i in range(n_steps):
        z_flat = trajectory[step_i].numpy().reshape(N, -1)  # (N, 784)

        # Standardize features
        mu = z_flat.mean(axis=0)
        std = z_flat.std(axis=0) + 1e-8
        z_flat = (z_flat - mu) / std

        # PCA reduce to n_pca dims
        proj, _, _ = pca(z_flat, n_pca)  # (N, n_pca)

        X_train, X_test = proj[train_idx], proj[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]

        W = ridge_regression(X_train, Y_train, alpha=1.0)
        Y_pred = X_test @ W

        # Overall R²
        ss_res = ((Y_test - Y_pred) ** 2).sum()
        ss_tot = ((Y_test - Y_test.mean(axis=0)) ** 2).sum()
        r2_all[step_i] = 1.0 - ss_res / ss_tot

        # Per-class R²
        for cls in range(n_classes):
            test_mask = labels[test_idx] == cls
            if test_mask.sum() < 2:
                continue
            y_c = Y_test[test_mask]
            yp_c = Y_pred[test_mask]
            ss_r = ((y_c - yp_c) ** 2).sum()
            ss_t = ((y_c - y_c.mean(axis=0)) ** 2).sum()
            r2_per_class[cls, step_i] = 1.0 - ss_r / max(ss_t, 1e-12)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    steps = np.arange(n_steps)
    for cls in range(n_classes):
        ax.plot(steps, r2_per_class[cls], alpha=0.4, linewidth=1, label=f"class {cls}")
    ax.plot(steps, r2_all, color="black", linewidth=2.5, label="average")
    ax.set_xlabel("ODE step")
    ax.set_ylabel("R² (sin θ, cos θ)")
    ax.set_title("Rotation decodability along ODE trajectory")
    ax.legend(fontsize=7, ncol=3)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "exp1_rotation_decodability.png", dpi=300)
    plt.close(fig)
    print(f"  Final R² (avg): {r2_all[-1]:.4f}")
    print("  Saved exp1_rotation_decodability.png")


# ---------------------------------------------------------------------------
# Experiment 2: Controlled rotation orbits in activation space
# ---------------------------------------------------------------------------


class BlockActivationCollector:
    """Hook x_embedder and each JiTBlock, storing output for a single forward pass."""

    def __init__(self, net: torch.nn.Module):
        self.activations: dict[str, torch.Tensor] = {}
        self.hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._register(net)

    def _register(self, net: torch.nn.Module) -> None:
        self._add_hook("x_embedder", net.x_embedder)  # type: ignore[arg-type]
        for i, block in enumerate(net.blocks):  # type: ignore[union-attr]
            self._add_hook(f"block_{i}", block)

    def _add_hook(self, name: str, module: torch.nn.Module) -> None:
        def hook_fn(
            _mod: torch.nn.Module,
            _inp: object,
            output: torch.Tensor,
            _name: str = name,
        ) -> None:
            self.activations[_name] = output.detach().float().cpu()

        self.hooks.append(module.register_forward_hook(hook_fn))

    def clear(self) -> None:
        self.activations.clear()

    def remove(self) -> None:
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


def circular_corr(alpha: np.ndarray, beta: np.ndarray) -> float:
    """Circular correlation coefficient (Fisher & Lee, 1983)."""
    mean_a = np.arctan2(np.mean(np.sin(alpha)), np.mean(np.cos(alpha)))
    mean_b = np.arctan2(np.mean(np.sin(beta)), np.mean(np.cos(beta)))
    sin_a = np.sin(alpha - mean_a)
    sin_b = np.sin(beta - mean_b)
    return float(
        np.abs(
            np.sum(sin_a * sin_b) / np.sqrt(np.sum(sin_a**2) * np.sum(sin_b**2) + 1e-12)
        )
    )


def rotate_batch(images: torch.Tensor, angles_rad: torch.Tensor) -> torch.Tensor:
    """Rotate a batch of image-like states with bilinear interpolation."""
    if images.shape[0] != angles_rad.shape[0]:
        raise ValueError("Batch size and number of rotation angles must match")

    theta = torch.zeros(images.shape[0], 2, 3, dtype=images.dtype, device=images.device)
    cos_a = torch.cos(angles_rad)
    sin_a = torch.sin(angles_rad)
    theta[:, 0, 0] = cos_a
    theta[:, 0, 1] = -sin_a
    theta[:, 1, 0] = sin_a
    theta[:, 1, 1] = cos_a

    grid = F.affine_grid(theta, images.shape, align_corners=False)
    return F.grid_sample(
        images,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )


def _image_tokens(
    acts: torch.Tensor, layer_name: str, in_context_start: int, in_context_len: int
) -> np.ndarray:
    """Return image tokens only, stripping in-context tokens when present."""
    block_idx = -1
    if layer_name.startswith("block_"):
        block_idx = int(layer_name.split("_")[1])

    if block_idx >= in_context_start and in_context_len > 0:
        acts = acts[:, in_context_len:, :]

    return acts.numpy()


def _fit_standardized_decoder(
    X: np.ndarray, angles: np.ndarray, train_idx: np.ndarray, alpha: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a linear decoder from activations to (cos θ, sin θ)."""
    Y = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    X_train = X[train_idx]
    mu = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-6
    X_train_std = (X_train - mu) / std
    W = ridge_regression(X_train_std, Y[train_idx], alpha=alpha)
    return mu, std, W


def _predict_standardized_decoder(
    X: np.ndarray, mu: np.ndarray, std: np.ndarray, W: np.ndarray
) -> np.ndarray:
    X_std = (X - mu) / std
    return X_std @ W


def _orbit_rank2_score(X: np.ndarray) -> float:
    """Fraction of orbit variance captured by the best 2D linear subspace."""
    n_components = min(3, X.shape[0], X.shape[1])
    if n_components < 2:
        return 0.0
    _, _, explained = pca(X, n_components)
    return float(explained[:2].sum())


def _orbit_distance_correlation(X: np.ndarray, circle_chords: np.ndarray) -> float:
    """Correlation between activation distances and circle chord distances."""
    Xc = X - X.mean(axis=0, keepdims=True)
    dist = np.linalg.norm(Xc[:, None, :] - Xc[None, :, :], axis=-1)
    upper = np.triu_indices(dist.shape[0], k=1)
    rho = spearmanr(dist[upper], circle_chords[upper]).statistic
    if rho is None or np.isnan(rho):
        return 0.0
    return float(rho)


def _orbit_decode_correlation(X: np.ndarray, angles: np.ndarray) -> float:
    """Held-out circular correlation from an alternating-angle split."""
    if X.shape[0] < 4:
        return 0.0

    train_idx = np.arange(0, X.shape[0], 2)
    test_idx = np.arange(1, X.shape[0], 2)
    mu, std, W = _fit_standardized_decoder(X, angles, train_idx)
    pred = _predict_standardized_decoder(X[test_idx], mu, std, W)
    pred_angles = np.arctan2(pred[:, 1], pred[:, 0])
    return circular_corr(pred_angles, angles[test_idx])


def _select_orbit_indices(
    labels: np.ndarray, n_classes: int, per_class: int
) -> np.ndarray:
    """Pick a fixed subset of base trajectories per class."""
    rng = np.random.RandomState(42)
    selected: list[int] = []
    for cls in range(n_classes):
        cls_idx = np.where(labels == cls)[0]
        n_use = min(per_class, len(cls_idx))
        if n_use == 0:
            continue
        take = rng.choice(cls_idx, size=n_use, replace=False)
        selected.extend(sorted(int(i) for i in take))
    return np.array(selected, dtype=np.int64)


def _compute_orbit_cell_details(
    model: Denoiser,
    trajectory: list[torch.Tensor],
    cond: torch.Tensor,
    labels: np.ndarray,
    selected_indices: np.ndarray,
    angle_values: np.ndarray,
    step_i: int,
    layer_name: str,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Average tangent-energy patch map and representative orbit projections."""
    net = model.net
    in_context_start = net.in_context_start
    in_context_len = net.in_context_len
    device = cond.device
    collector = BlockActivationCollector(net)
    t_values = torch.linspace(0.0, 1.0, model.config.num_sampling_steps + 1)
    t_scalar = float(t_values[min(step_i, len(t_values) - 1)].item())

    angle_torch = torch.tensor(angle_values, device=device, dtype=torch.float32)
    delta = float(angle_values[1] - angle_values[0])
    patch_maps: list[np.ndarray] = []
    scatter_by_class: dict[int, np.ndarray] = {}

    show_classes = [
        cls for cls in [2, 3, 6, 7] if np.any(labels[selected_indices] == cls)
    ]
    if not show_classes:
        show_classes = sorted({int(labels[i]) for i in selected_indices})[:4]

    for base_idx in selected_indices:
        base = trajectory[step_i][base_idx : base_idx + 1].to(device).float()
        orbit = rotate_batch(base.repeat(len(angle_values), 1, 1, 1), angle_torch)
        t_batch = torch.full((len(angle_values),), t_scalar, device=device)
        y_batch = cond[base_idx : base_idx + 1].repeat(len(angle_values))

        collector.clear()
        with torch.no_grad(), utils.maybe_autocast(device):
            net(orbit, t_batch, y_batch)

        tokens = _image_tokens(
            collector.activations[layer_name],
            layer_name,
            in_context_start,
            in_context_len,
        )  # (A, T, H)

        deriv = (np.roll(tokens, -1, axis=0) - np.roll(tokens, 1, axis=0)) / (
            2.0 * delta
        )
        patch_maps.append(np.linalg.norm(deriv, axis=2).mean(axis=0))

        cls = int(labels[base_idx])
        if cls not in show_classes or cls in scatter_by_class:
            continue
        X = tokens.reshape(tokens.shape[0], -1)
        train_idx = np.arange(len(angle_values))
        mu, std, W = _fit_standardized_decoder(X, angle_values, train_idx)
        scatter_by_class[cls] = _predict_standardized_decoder(X, mu, std, W)

    collector.remove()

    avg_patch = np.mean(patch_maps, axis=0)
    grid_size = int(np.sqrt(avg_patch.shape[0]))
    return avg_patch.reshape(grid_size, grid_size), scatter_by_class


def _wrap_to_pi(angles: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angles), np.cos(angles))


def _image_tokens_tensor(
    acts: torch.Tensor, layer_name: str, in_context_start: int, in_context_len: int
) -> torch.Tensor:
    """Torch version of _image_tokens for differentiable steering."""
    block_idx = -1
    if layer_name.startswith("block_"):
        block_idx = int(layer_name.split("_")[1])

    if block_idx >= in_context_start and in_context_len > 0:
        acts = acts[:, in_context_len:, :]

    return acts


def forward_to_layer(
    net: torch.nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    y: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    """Forward pass that returns the requested internal activation with gradients intact."""
    t_emb = net.t_embedder(t)
    y_emb = net.y_embedder(y)
    c = t_emb + y_emb

    x = net.x_embedder(x)
    if layer_name == "x_embedder":
        return x

    x = x + net.pos_embed

    for i, block in enumerate(net.blocks):
        if net.in_context_len > 0 and i == net.in_context_start:
            in_context_tokens = y_emb.unsqueeze(1).repeat(1, net.in_context_len, 1)
            in_context_tokens = in_context_tokens + net.in_context_posemb
            x = torch.cat([in_context_tokens, x], dim=1)

        x = block(
            x,
            c,
            net.feat_rope if i < net.in_context_start else net.feat_rope_incontext,
        )
        if layer_name == f"block_{i}":
            return x

    raise ValueError(f"Unknown layer name: {layer_name}")


def _scan_rotation_geometry(
    model: Denoiser,
    trajectory: list[torch.Tensor],
    cond: torch.Tensor,
    labels: np.ndarray,
    n_classes: int,
    orbit_samples_per_class: int,
    orbit_angles: int,
) -> RotationGeometryScan:
    """Measure where explicit rotations become circular and decodable."""
    if orbit_angles < 6:
        raise ValueError("Experiment 2 needs at least 6 orbit angles")

    net = model.net
    in_context_start = net.in_context_start
    in_context_len = net.in_context_len
    device = cond.device
    timestep_values = torch.linspace(0.0, 1.0, model.config.num_sampling_steps + 1)
    angle_values = np.linspace(0.0, 2.0 * np.pi, orbit_angles, endpoint=False)
    angle_torch = torch.tensor(angle_values, device=device, dtype=torch.float32)
    delta = angle_values[:, None] - angle_values[None, :]
    wrapped = np.arctan2(np.sin(delta), np.cos(delta))
    circle_chords = 2.0 * np.sin(0.5 * np.abs(wrapped))

    selected_indices = _select_orbit_indices(labels, n_classes, orbit_samples_per_class)
    if selected_indices.size == 0:
        raise ValueError("No trajectories available for experiment 2")

    collector = BlockActivationCollector(net)
    layer_names = ["x_embedder", *[f"block_{i}" for i in range(len(net.blocks))]]
    n_layers = len(layer_names)
    n_steps = len(trajectory)

    rank2_scores = np.zeros((n_layers, n_steps))
    circle_scores = np.zeros((n_layers, n_steps))
    decode_scores = np.zeros((n_layers, n_steps))

    print(
        f"  Probing {selected_indices.size} generated trajectories "
        f"({orbit_samples_per_class}/class, {orbit_angles} rotations each)"
    )

    for step_i in range(n_steps):
        t_scalar = float(timestep_values[min(step_i, len(timestep_values) - 1)].item())
        step_rank2 = [[] for _ in range(n_layers)]
        step_circle = [[] for _ in range(n_layers)]
        step_decode = [[] for _ in range(n_layers)]

        for base_idx in selected_indices:
            base = trajectory[step_i][base_idx : base_idx + 1].to(device).float()
            orbit = rotate_batch(base.repeat(orbit_angles, 1, 1, 1), angle_torch)
            t_batch = torch.full((orbit_angles,), t_scalar, device=device)
            y_batch = cond[base_idx : base_idx + 1].repeat(orbit_angles)

            collector.clear()
            with torch.no_grad(), utils.maybe_autocast(device):
                net(orbit, t_batch, y_batch)

            for li, layer_name in enumerate(layer_names):
                tokens = _image_tokens(
                    collector.activations[layer_name],
                    layer_name,
                    in_context_start,
                    in_context_len,
                )
                X = tokens.reshape(tokens.shape[0], -1)
                step_rank2[li].append(_orbit_rank2_score(X))
                step_circle[li].append(_orbit_distance_correlation(X, circle_chords))
                step_decode[li].append(_orbit_decode_correlation(X, angle_values))

        for li in range(n_layers):
            rank2_scores[li, step_i] = float(np.mean(step_rank2[li]))
            circle_scores[li, step_i] = float(np.mean(step_circle[li]))
            decode_scores[li, step_i] = float(np.mean(step_decode[li]))

        print(
            f"  Step {step_i:02d}/{n_steps - 1}: "
            f"circle={circle_scores[:, step_i].max():.3f} "
            f"decode={decode_scores[:, step_i].max():.3f}"
        )

    collector.remove()

    geometry_score = 0.5 * (circle_scores + decode_scores)
    peak_li, peak_step = np.unravel_index(
        np.argmax(geometry_score), geometry_score.shape
    )
    peak_layer = layer_names[int(peak_li)]
    t_peak = float(
        timestep_values[min(int(peak_step), len(timestep_values) - 1)].item()
    )

    return RotationGeometryScan(
        angle_values=angle_values,
        circle_scores=circle_scores,
        decode_scores=decode_scores,
        rank2_scores=rank2_scores,
        geometry_score=geometry_score,
        layer_names=layer_names,
        selected_indices=selected_indices,
        peak_layer=peak_layer,
        peak_step=int(peak_step),
        t_peak=t_peak,
    )


def fit_absolute_image_rotation_decoders(
    n_classes: int,
    n_bases_per_class: int = 32,
    n_angles: int = 16,
) -> dict[int, RotationDecoder]:
    """Fit class-conditional absolute angle decoders from upright MNIST images."""
    ds = MNIST(train=False, rotate=False)
    labels = ds.c.numpy()
    angle_values = np.linspace(-np.pi, np.pi, n_angles, endpoint=False)
    angle_torch = torch.tensor(angle_values, dtype=torch.float32)
    rng = np.random.RandomState(0)
    decoders: dict[int, RotationDecoder] = {}

    for cls in range(n_classes):
        cls_idx = np.where(labels == cls)[0]
        n_use = min(n_bases_per_class, len(cls_idx))
        base_idx = rng.choice(cls_idx, size=n_use, replace=False)
        base_imgs = ds.x[base_idx].float()
        orbit_imgs = []
        orbit_angles = []

        for base in base_imgs:
            orbit = rotate_batch(
                base.unsqueeze(0).repeat(n_angles, 1, 1, 1), angle_torch
            )
            orbit_imgs.append(orbit)
            orbit_angles.append(angle_values)

        X = torch.cat(orbit_imgs, dim=0).reshape(-1, 28 * 28).numpy()
        angles = np.concatenate(orbit_angles, axis=0)
        mu, std, W = _fit_standardized_decoder(
            X, angles, np.arange(len(angles)), alpha=5.0
        )
        decoders[cls] = RotationDecoder(mu=mu, std=std, W=W)

    return decoders


def estimate_absolute_image_angles(
    images: torch.Tensor, labels: np.ndarray, decoders: dict[int, RotationDecoder]
) -> tuple[np.ndarray, np.ndarray]:
    """Predict absolute angles and decoder confidence from final images."""
    flat = images.detach().cpu().float().reshape(images.shape[0], -1).numpy()
    angles = np.zeros(images.shape[0], dtype=np.float64)
    confidence = np.zeros(images.shape[0], dtype=np.float64)

    for i, cls in enumerate(labels):
        decoder = decoders[int(cls)]
        pred = _predict_standardized_decoder(
            flat[i : i + 1], decoder.mu, decoder.std, decoder.W
        )[0]
        angles[i] = float(np.arctan2(pred[1], pred[0]))
        confidence[i] = float(np.linalg.norm(pred))

    return angles, confidence


def build_relative_rotation_bank(
    base_images: torch.Tensor, n_angles: int = 72
) -> RelativeRotationBank:
    """Precompute rotated templates of each base image for relative-angle matching."""
    angle_values = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
    angle_torch = torch.tensor(angle_values, dtype=base_images.dtype)
    templates = []

    for i in range(base_images.shape[0]):
        orbit = rotate_batch(
            base_images[i : i + 1].repeat(n_angles, 1, 1, 1), angle_torch
        )
        flat = orbit.reshape(n_angles, -1).numpy()
        flat /= np.linalg.norm(flat, axis=1, keepdims=True) + 1e-8
        templates.append(flat)

    return RelativeRotationBank(
        angle_values=angle_values,
        templates=np.stack(templates, axis=0),
    )


def estimate_relative_angles(
    images: torch.Tensor, bank: RelativeRotationBank
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate rotation relative to each image's own base template bank."""
    flat = images.detach().cpu().float().reshape(images.shape[0], -1).numpy()
    flat /= np.linalg.norm(flat, axis=1, keepdims=True) + 1e-8
    sims = np.einsum("bd,bad->ba", flat, bank.templates)
    best = sims.argmax(axis=1)
    angles = bank.angle_values[best]
    conf = sims[np.arange(images.shape[0]), best]
    return angles, conf


def apply_rotation_schedule(
    schedule: np.ndarray,
) -> Callable[[torch.Tensor, int, float, torch.Tensor], torch.Tensor]:
    """Create a steering callback that rotates each sample by a scheduled angle."""

    def steer_fn(
        z: torch.Tensor, step_i: int, _t_scalar: float, _cond_batch: torch.Tensor
    ) -> torch.Tensor:
        if step_i >= schedule.shape[1]:
            return z
        offsets = torch.tensor(schedule[:, step_i], device=z.device, dtype=z.dtype)
        if torch.all(offsets.abs() < 1e-8):
            return z
        return rotate_batch(z, offsets)

    return steer_fn


def render_with_schedule(
    model: Denoiser,
    cond: torch.Tensor,
    z0: torch.Tensor,
    schedule: np.ndarray,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Generate samples under a fixed per-sample rotation-control schedule."""
    return generate_with_trajectory(
        model,
        cond,
        z0=z0,
        steer_fn=apply_rotation_schedule(schedule),
    )


def continue_from_state(
    model: Denoiser,
    cond: torch.Tensor,
    z_t: torch.Tensor,
    schedule: np.ndarray,
    start_step: int,
    skip_current_step: bool = True,
) -> torch.Tensor:
    """Roll out from an intermediate state with the remaining control schedule."""
    steer_fn = apply_rotation_schedule(schedule)

    def future_steer_fn(
        z: torch.Tensor, step_i: int, t_scalar: float, cond_batch: torch.Tensor
    ) -> torch.Tensor:
        if skip_current_step and step_i == start_step:
            return z
        return steer_fn(z, step_i, t_scalar, cond_batch)

    final, _ = generate_with_trajectory(
        model,
        cond,
        z0=z_t,
        steer_fn=future_steer_fn,
        start_step=start_step,
    )
    return final


def correct_final_images_to_target(
    images: torch.Tensor, current_angles: np.ndarray, target_angles: np.ndarray
) -> torch.Tensor:
    """Exact output-space correction by rotating the final image residual."""
    residual = _wrap_to_pi(target_angles - current_angles)
    offsets = torch.tensor(residual, dtype=images.dtype)
    corrected = rotate_batch(images.float(), offsets)
    return corrected


def _fit_local_rotation_decoder(
    model: Denoiser,
    z_t: torch.Tensor,
    t_scalar: float,
    cond: torch.Tensor,
    layer_name: str,
    angle_values: np.ndarray,
) -> RotationDecoder:
    """Fit a relative angle decoder around a single state using explicit rotations."""
    net = model.net
    device = z_t.device
    angle_torch = torch.tensor(angle_values, device=device, dtype=torch.float32)
    orbit = rotate_batch(z_t.repeat(len(angle_values), 1, 1, 1), angle_torch)
    t_batch = torch.full((len(angle_values),), t_scalar, device=device)
    y_batch = cond.repeat(len(angle_values))

    with torch.no_grad():
        acts = forward_to_layer(net, orbit, t_batch, y_batch, layer_name)
        tokens = _image_tokens_tensor(
            acts, layer_name, net.in_context_start, net.in_context_len
        )
        X = tokens.reshape(len(angle_values), -1).detach().cpu().numpy()

    mu, std, W = _fit_standardized_decoder(
        X, angle_values, np.arange(len(angle_values)), alpha=1.0
    )
    return RotationDecoder(mu=mu, std=std, W=W)


def _steer_single_state(
    model: Denoiser,
    z_t: torch.Tensor,
    t_scalar: float,
    cond: torch.Tensor,
    layer_name: str,
    decoder: RotationDecoder,
    target_relative_angle: float,
    grad_steps: int,
    step_size: float,
    regularization: float,
) -> torch.Tensor:
    """Move a single latent state along its local rotation orbit toward the target angle."""
    device = z_t.device
    net = model.net
    z_ref = z_t.detach()
    z_work = z_ref
    target_vec = torch.tensor(
        [np.cos(target_relative_angle), np.sin(target_relative_angle)],
        device=device,
        dtype=z_work.dtype,
    )
    mu = torch.from_numpy(decoder.mu).to(device=device, dtype=z_work.dtype)
    std = torch.from_numpy(decoder.std).to(device=device, dtype=z_work.dtype)
    W = torch.from_numpy(decoder.W).to(device=device, dtype=z_work.dtype)
    t_batch = torch.full((1,), t_scalar, device=device, dtype=z_work.dtype)

    for _ in range(grad_steps):
        with torch.enable_grad():
            z_var = z_work.detach().requires_grad_(True)
            acts = forward_to_layer(net, z_var, t_batch, cond, layer_name)
            tokens = _image_tokens_tensor(
                acts, layer_name, net.in_context_start, net.in_context_len
            )
            feat = tokens.reshape(1, -1)
            pred = ((feat - mu) / std) @ W
            pred = pred[0]
            pred = pred / pred.norm().clamp_min(1e-6)
            loss = 1.0 - torch.dot(pred, target_vec)
            loss = loss + regularization * (z_var - z_ref).square().mean()
            grad = torch.autograd.grad(loss, z_var)[0]
            grad_norm = grad.flatten().norm().clamp_min(1e-6)
            z_work = (z_var - step_size * grad / grad_norm).detach()

    return z_work


def optimize_rotation_schedule(
    model: Denoiser,
    cond: torch.Tensor,
    z0: torch.Tensor,
    measure_angles: Callable[[torch.Tensor], tuple[np.ndarray, np.ndarray]],
    target_angles: np.ndarray,
    candidate_steps: list[int],
    steer_iterations: int,
    gain_angle_rad: float,
    max_step_angle_rad: float,
    initial_schedule: np.ndarray | None = None,
) -> tuple[np.ndarray, torch.Tensor, np.ndarray, np.ndarray]:
    """Closed-loop control using measured final-angle gains for latent rotations."""
    batch_size = cond.shape[0]
    num_steps = model.config.num_sampling_steps
    schedule = (
        initial_schedule.copy()
        if initial_schedule is not None
        else np.zeros((batch_size, num_steps), dtype=np.float32)
    )
    final_images = torch.empty(0)
    final_angles = np.zeros(batch_size, dtype=np.float64)
    final_conf = np.zeros(batch_size, dtype=np.float64)

    for iteration in range(steer_iterations):
        final, trajectory = render_with_schedule(model, cond, z0, schedule)
        images = final.float().cpu().view(-1, 1, 28, 28).clamp(-1, 1)
        angles, conf = measure_angles(images)
        residual = _wrap_to_pi(target_angles - angles)

        mean_err_deg = np.degrees(np.abs(residual)).mean()
        print(f"  Control iter {iteration}: mean |error| = {mean_err_deg:.1f}°")

        final_images = images
        final_angles = angles
        final_conf = conf
        if iteration == steer_iterations - 1:
            break

        gain_matrix = np.zeros((batch_size, len(candidate_steps)), dtype=np.float64)
        for si, step_i in enumerate(candidate_steps):
            base_state = trajectory[step_i].to(cond.device).float()
            plus_offsets = torch.full((batch_size,), gain_angle_rad, device=cond.device)
            minus_offsets = torch.full(
                (batch_size,), -gain_angle_rad, device=cond.device
            )
            plus = continue_from_state(
                model,
                cond,
                rotate_batch(base_state, plus_offsets),
                schedule,
                start_step=step_i,
            )
            minus = continue_from_state(
                model,
                cond,
                rotate_batch(base_state, minus_offsets),
                schedule,
                start_step=step_i,
            )

            plus_angles, _ = measure_angles(
                plus.float().cpu().view(-1, 1, 28, 28).clamp(-1, 1),
            )
            minus_angles, _ = measure_angles(
                minus.float().cpu().view(-1, 1, 28, 28).clamp(-1, 1)
            )
            gain_matrix[:, si] = _wrap_to_pi(plus_angles - minus_angles) / (
                2.0 * gain_angle_rad
            )

        chosen = np.argmax(np.abs(gain_matrix), axis=1)
        for batch_idx in range(batch_size):
            gain = gain_matrix[batch_idx, chosen[batch_idx]]
            if abs(gain) < 0.05:
                continue
            step_i = candidate_steps[int(chosen[batch_idx])]
            delta = residual[batch_idx] / gain
            delta = float(np.clip(delta, -max_step_angle_rad, max_step_angle_rad))
            schedule[batch_idx, step_i] += delta

    return schedule, final_images, final_angles, final_conf


def exp2_rotation_orbits(
    model: Denoiser,
    trajectory: list[torch.Tensor],
    cond: torch.Tensor,
    labels: np.ndarray,
    n_classes: int,
    orbit_samples_per_class: int,
    orbit_angles: int,
) -> None:
    """Back-trace explicit S¹ rotations through the network and locate the rotation plane."""
    print("=== Experiment 2: Controlled rotation orbit geometry ===")
    scan = _scan_rotation_geometry(
        model,
        trajectory,
        cond,
        labels,
        n_classes,
        orbit_samples_per_class,
        orbit_angles,
    )
    peak_li = scan.layer_names.index(scan.peak_layer)
    peak_step = scan.peak_step
    peak_layer = scan.peak_layer
    t_peak = scan.t_peak
    print(
        f"  Peak rotation geometry at {peak_layer}, step={peak_step}, t={t_peak:.2f}, "
        f"score={scan.geometry_score[peak_li, peak_step]:.3f}"
    )

    n_layers = len(scan.layer_names)
    n_steps = scan.circle_scores.shape[1]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    heatmaps = [
        (scan.circle_scores, "Circle-Distance Correlation"),
        (scan.decode_scores, "Held-Out Angle Decode"),
        (scan.rank2_scores, "Rank-2 Orbit Energy"),
    ]
    xticks = list(range(n_steps))
    timestep_values = torch.linspace(0.0, 1.0, model.config.num_sampling_steps + 1)
    xticklabels = [f"{timestep_values[i].item():.2f}" for i in range(n_steps)]

    for ax, (values, title) in zip(axes, heatmaps, strict=True):
        im = ax.imshow(values, aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
        ax.scatter(
            [peak_step], [peak_li], marker="x", color="white", s=80, linewidths=2
        )
        ax.set_title(title)
        ax.set_xlabel("ODE time t")
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, rotation=45, ha="right", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    axes[0].set_yticks(range(n_layers))
    axes[0].set_yticklabels(scan.layer_names, fontsize=8)
    axes[0].set_ylabel("Network layer")
    fig.suptitle(
        "Where the explicit rotation orbit becomes circular and linearly steerable",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "exp2_rotation_geometry.png", dpi=300)
    plt.close(fig)
    print("  Saved exp2_rotation_geometry.png")

    patch_map, scatter_by_class = _compute_orbit_cell_details(
        model,
        trajectory,
        cond,
        labels,
        scan.selected_indices,
        scan.angle_values,
        int(peak_step),
        peak_layer,
    )

    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(patch_map, cmap="viridis")
    ax.set_title(f"Rotation tangent energy\n{peak_layer}, t={t_peak:.2f}")
    ax.set_xlabel("Patch column")
    ax.set_ylabel("Patch row")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "exp2_rotation_where.png", dpi=300)
    plt.close(fig)
    print("  Saved exp2_rotation_where.png")

    show_classes = sorted(scatter_by_class.keys())
    n_show = max(len(show_classes), 1)
    fig, axes = plt.subplots(1, n_show, figsize=(3.6 * n_show, 3.4), squeeze=False)
    scatter = None
    for ax, cls in zip(axes[0], show_classes, strict=False):
        proj = scatter_by_class[cls]
        scatter = ax.scatter(
            proj[:, 0],
            proj[:, 1],
            c=scan.angle_values,
            cmap="hsv",
            s=30,
            vmin=0.0,
            vmax=2.0 * np.pi,
        )
        ax.plot(proj[:, 0], proj[:, 1], color="0.7", linewidth=1, alpha=0.8)
        ax.set_title(f"class {cls}")
        ax.set_xlabel("decoded cos θ")
        ax.set_ylabel("decoded sin θ")
        ax.set_aspect("equal")

    if scatter is not None:
        fig.colorbar(
            scatter,
            ax=axes[0].tolist(),
            fraction=0.03,
            pad=0.03,
            label="rotation angle (rad)",
        )
    fig.suptitle(
        f"Representative rotation orbits at {peak_layer}, t={t_peak:.2f}", fontsize=11
    )
    fig.subplots_adjust(top=0.80, wspace=0.35)
    fig.savefig(OUT_DIR / "exp2_rotation_orbits.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved exp2_rotation_orbits.png")


def exp4_rotation_circle_grid(
    model: Denoiser,
    n_classes: int,
    orbit_samples_per_class: int,
    orbit_angles: int,
    orbit_grid_rows: int,
    steer_iterations: int,
    steer_gain_angle_deg: float,
    steer_max_step_angle_deg: float,
) -> None:
    """Take one base sample per class and sweep it around a relative rotation orbit."""
    print("=== Experiment 4: Relative rotation circle grid ===")

    if orbit_grid_rows < 2:
        raise ValueError("orbit_grid_rows must be at least 2")

    device = next(model.parameters()).device

    calib_cond = torch.arange(n_classes, device=device).repeat_interleave(
        orbit_samples_per_class
    )
    calib_labels = calib_cond.cpu().numpy()
    print("  Calibrating the internal rotation plane...")
    with torch.inference_mode():
        _, calib_trajectory = generate_with_trajectory(model, calib_cond)

    scan = _scan_rotation_geometry(
        model,
        calib_trajectory,
        calib_cond,
        calib_labels,
        n_classes,
        orbit_samples_per_class,
        orbit_angles,
    )
    first_control_step = min(
        max(scan.peak_step, model.config.num_sampling_steps // 2),
        model.config.num_sampling_steps - 1,
    )
    candidate_steps = list(range(first_control_step, model.config.num_sampling_steps))
    print(
        f"  Steering from steps {candidate_steps} "
        f"(peak geometry at {scan.peak_layer}, t={scan.t_peak:.2f})"
    )

    cond = torch.arange(n_classes, device=device)
    z0 = model.config.noise_scale * torch.randn(
        n_classes,
        model.net.in_channels,
        model.net.input_size,
        model.net.input_size,
        device=device,
    )
    zero_schedule = np.zeros(
        (n_classes, model.config.num_sampling_steps), dtype=np.float32
    )
    with torch.inference_mode():
        base_final, _ = render_with_schedule(model, cond, z0, zero_schedule)
    base_images = base_final.float().cpu().view(-1, 1, 28, 28).clamp(-1, 1)
    bank = build_relative_rotation_bank(base_images)

    def measure_relative(images: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        return estimate_relative_angles(images, bank)

    base_angles, base_conf = measure_relative(base_images)
    print(f"  Base relative confidence mean: {base_conf.mean():.2f}")

    offsets = np.linspace(0.0, 2.0 * np.pi, orbit_grid_rows, endpoint=False)
    all_images: list[torch.Tensor] = [base_images]
    all_angles: list[np.ndarray] = [base_angles]
    all_errors: list[np.ndarray] = [np.zeros(n_classes)]
    schedule = zero_schedule.copy()

    for row_i, offset in enumerate(offsets[1:], start=1):
        target_angles = np.full(n_classes, offset, dtype=np.float64)
        print(
            f"  Orbit row {row_i}/{orbit_grid_rows - 1}: "
            f"target relative offset {np.degrees(offset):.1f}°"
        )
        schedule, row_images, row_angles, _ = optimize_rotation_schedule(
            model,
            cond,
            z0,
            measure_relative,
            target_angles,
            candidate_steps,
            steer_iterations=max(steer_iterations, 1),
            gain_angle_rad=np.deg2rad(steer_gain_angle_deg),
            max_step_angle_rad=np.deg2rad(steer_max_step_angle_deg),
            initial_schedule=schedule,
        )
        row_err = _wrap_to_pi(row_angles - target_angles)
        print(f"    Mean |relative error| = {np.degrees(np.abs(row_err)).mean():.1f}°")
        all_images.append(row_images)
        all_angles.append(row_angles)
        all_errors.append(row_err)

    fig, axes = plt.subplots(
        orbit_grid_rows,
        n_classes,
        figsize=(1.35 * n_classes, 1.35 * orbit_grid_rows),
        squeeze=False,
    )
    for row_i in range(orbit_grid_rows):
        for col in range(n_classes):
            img = ((all_images[row_i][col, 0].numpy() + 1.0) / 2.0).clip(0.0, 1.0)
            ax = axes[row_i, col]
            ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
            ax.axis("off")
            if row_i == 0:
                ax.set_title(f"c{col}", fontsize=9)
            if col == 0:
                offset_deg = np.degrees(offsets[row_i])
                ax.set_ylabel(f"{offset_deg:.0f}°", fontsize=8, rotation=0, labelpad=18)

    fig.suptitle(
        "One base sample per class, steered around a relative rotation circle",
        fontsize=11,
    )
    fig.subplots_adjust(top=0.92, wspace=0.02, hspace=0.02)
    fig.savefig(OUT_DIR / "exp4_rotation_circle_grid.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved exp4_rotation_circle_grid.png")

    err_mat = np.degrees(np.abs(np.stack(all_errors, axis=0)))
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(
        err_mat,
        aspect="auto",
        cmap="magma",
        vmin=0.0,
        vmax=max(45.0, float(err_mat.max())),
    )
    ax.set_xticks(range(n_classes))
    ax.set_xticklabels([str(c) for c in range(n_classes)])
    ax.set_yticks(range(orbit_grid_rows))
    ax.set_yticklabels([f"{np.degrees(off):.0f}°" for off in offsets], fontsize=8)
    ax.set_xlabel("class")
    ax.set_ylabel("target relative offset")
    ax.set_title("Relative angle error of the steered orbit grid")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="|error| (deg)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "exp4_rotation_circle_error.png", dpi=300)
    plt.close(fig)
    print("  Saved exp4_rotation_circle_error.png")


# ---------------------------------------------------------------------------
# Experiment 3: Velocity field Jacobian analysis
# ---------------------------------------------------------------------------


def _angular_derivative(image: np.ndarray, delta_deg: float = 5.0) -> np.ndarray:
    """Numerical angular derivative via finite-difference rotation ±delta_deg.

    image: (1, 28, 28) numpy array. Returns flattened (784,).
    Uses bilinear interpolation (required for sub-pixel accuracy at 28x28).
    """
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as TF

    img_t = torch.from_numpy(image).float()  # (1, 28, 28)
    plus = TF.rotate(img_t, angle=-delta_deg, interpolation=InterpolationMode.BILINEAR)
    minus = TF.rotate(img_t, angle=delta_deg, interpolation=InterpolationMode.BILINEAR)
    deriv = (plus - minus) / (2.0 * np.radians(delta_deg))
    return deriv.numpy().flatten()


def _compute_jacobian_svd(
    model: Denoiser,
    z_t: torch.Tensor,
    t_val: float,
    y_batch: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute Jacobian of x_pred w.r.t. z and return (U, S, Vt, x_pred_np).

    z_t: (1, 1, 28, 28), returns U (784, 784), S (784,), Vt (784, 784),
    x_pred_np (1, 28, 28).
    """
    device = z_t.device
    t_batch = torch.full((1,), t_val, device=device)
    z_flat = z_t.view(-1).requires_grad_(True)

    @torch.compiler.disable
    def forward_fn(z_in: torch.Tensor) -> torch.Tensor:
        z_img = z_in.view(1, 1, 28, 28)
        x_pred = model.net(z_img, t_batch, y_batch)
        return x_pred.view(-1)

    J = torch.autograd.functional.jacobian(forward_fn, z_flat)
    J_np = J.detach().cpu().float().numpy()

    U, S, Vt = np.linalg.svd(J_np, full_matrices=False)

    with torch.no_grad():
        x_pred_img = model.net(z_t, t_batch, y_batch)
    x_pred_np = x_pred_img.cpu().numpy()[0]  # (1, 28, 28)

    return U, S, Vt, x_pred_np


def exp3_jacobian_analysis(
    model: Denoiser,
    trajectory: list[torch.Tensor],
    cond: torch.Tensor,
    labels: np.ndarray,
    n_classes: int,
    top_k: int = 5,
    n_jac: int = 5,
    classes_to_show: list[int] | None = None,
) -> None:
    """SVD of ∂x_pred/∂z Jacobian, compare U modes to rotation derivative."""
    print("=== Experiment 3: Jacobian analysis ===")

    if classes_to_show is None:
        classes_to_show = [2, 3, 6, 7]  # digits with clear rotation structure

    device = cond.device
    n_steps = len(trajectory)
    mid_step = n_steps // 2  # t ≈ 0.5
    timestep_values = torch.linspace(0.0, 1.0, model.config.num_sampling_steps + 1)
    t_mid = timestep_values[mid_step].item()

    # Disable torch.compile for Jacobian computation
    torch._dynamo.reset()

    results: dict[int, dict] = {}

    for cls in classes_to_show:
        mask = labels == cls
        indices = np.where(mask)[0]
        n_use = min(n_jac, len(indices))
        sample_indices = indices[:n_use]

        all_cos_sims: list[list[float]] = []
        best_alignment = -1.0
        best_result: dict | None = None

        for idx in sample_indices:
            z_t = trajectory[mid_step][idx : idx + 1].to(device)
            y_batch = cond[idx : idx + 1]

            print(f"  Computing Jacobian for class {cls}, sample {idx}...")
            U, S, Vt, x_pred_np = _compute_jacobian_svd(model, z_t, t_mid, y_batch)

            ang_deriv = _angular_derivative(x_pred_np)
            ang_deriv_norm = ang_deriv / (np.linalg.norm(ang_deriv) + 1e-12)

            # Cosine similarity with top-k left singular vectors (output-space modes)
            cos_sims = []
            for k in range(min(top_k, U.shape[1])):
                cos_sim = float(np.dot(U[:, k], ang_deriv_norm))
                cos_sims.append(cos_sim)

            all_cos_sims.append(cos_sims)

            # Track sample with highest rotation alignment for visualization
            max_align = max(abs(c) for c in cos_sims)
            if max_align > best_alignment:
                best_alignment = max_align
                best_result = {
                    "z_img": x_pred_np,
                    "S": S,
                    "U": U,
                    "ang_deriv": ang_deriv,
                    "cos_sims": cos_sims,
                }

        # Average cosine similarities across samples
        avg_cos_sims = np.mean(np.abs(all_cos_sims), axis=0).tolist()

        assert best_result is not None
        results[cls] = {
            **best_result,
            "avg_cos_sims": avg_cos_sims,
        }
        print(
            f"    Avg |cos sims| ({n_use} samples): {[f'{c:.3f}' for c in avg_cos_sims]}"
        )

    n_show = len(classes_to_show)

    # Plot 1: Grid of image, top-k U modes (output-space), angular derivative
    n_cols = top_k + 2  # image + top_k modes + angular derivative
    fig, axes = plt.subplots(n_show, n_cols, figsize=(2.2 * n_cols, 2.2 * n_show))
    if n_show == 1:
        axes = axes[np.newaxis, :]

    for row, cls in enumerate(classes_to_show):
        r = results[cls]
        # Original image
        ax = axes[row, 0]
        img = r["z_img"][0]  # (28, 28)
        ax.imshow(img, cmap="gray")
        ax.set_title(f"class {cls}", fontsize=8)
        ax.axis("off")

        # Top-k left singular vectors (output-space modes)
        for k in range(top_k):
            ax = axes[row, k + 1]
            mode = r["U"][:, k].reshape(28, 28)
            ax.imshow(mode, cmap="RdBu_r", vmin=-mode.max(), vmax=mode.max())
            ax.set_title(f"U{k + 1} (σ={r['S'][k]:.1f})", fontsize=7)
            ax.axis("off")

        # Angular derivative
        ax = axes[row, -1]
        ad = r["ang_deriv"].reshape(28, 28)
        ax.imshow(ad, cmap="RdBu_r", vmin=-abs(ad).max(), vmax=abs(ad).max())
        ax.set_title("∂/∂θ", fontsize=8)
        ax.axis("off")

    fig.suptitle(
        "Jacobian SVD output modes vs. angular derivative (t≈0.5)", fontsize=11
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "exp3_jacobian_modes.png", dpi=300)
    plt.close(fig)
    print("  Saved exp3_jacobian_modes.png")

    # Plot 2: Singular value spectrum
    fig, ax = plt.subplots(figsize=(8, 5))
    for cls in classes_to_show:
        S = results[cls]["S"]
        ax.plot(
            range(min(20, len(S))),
            S[:20],
            marker="o",
            markersize=3,
            label=f"class {cls}",
        )
    ax.set_xlabel("Singular value index")
    ax.set_ylabel("Singular value")
    ax.set_title("Top-20 singular value spectrum of ∂x_pred/∂z")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "exp3_singular_values.png", dpi=300)
    plt.close(fig)
    print("  Saved exp3_singular_values.png")

    # Plot 3: Cosine similarity bar chart (averaged over samples)
    fig, axes_bar = plt.subplots(1, n_show, figsize=(3.5 * n_show, 4), squeeze=False)
    for i, cls in enumerate(classes_to_show):
        ax = axes_bar[0, i]
        cs = results[cls]["avg_cos_sims"]
        colors = ["#d62728" if c > 0.3 else "#1f77b4" for c in cs]
        ax.bar(range(len(cs)), cs, color=colors)
        ax.set_xlabel("Singular vector index")
        ax.set_ylabel("avg |cos sim|")
        ax.set_title(f"class {cls}", fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_xticks(range(len(cs)))
        ax.set_xticklabels([f"U{k + 1}" for k in range(len(cs))], fontsize=7)
    fig.suptitle(
        f"Alignment of U modes with angular derivative (n={n_jac} samples)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "exp3_rotation_alignment.png", dpi=300)
    plt.close(fig)
    print("  Saved exp3_rotation_alignment.png")

    # Plot 4: Alignment vs. timestep (sweep across ODE steps)
    print("  Computing alignment vs. timestep sweep...")
    n_jac_sweep = min(3, n_jac)  # fewer samples for expensive sweep
    top_k_sweep = 10
    probe_steps = list(range(0, n_steps, max(1, n_steps // 6)))  # ~6 steps
    if probe_steps[-1] != n_steps - 1:
        probe_steps.append(n_steps - 1)

    # cls -> (n_probe_steps,) max alignment
    alignment_vs_t: dict[int, list[float]] = {cls: [] for cls in classes_to_show}

    for step_i in probe_steps:
        t_val = timestep_values[min(step_i, len(timestep_values) - 1)].item()
        for cls in classes_to_show:
            mask = labels == cls
            indices = np.where(mask)[0][:n_jac_sweep]
            step_alignments = []
            for idx in indices:
                z_t = trajectory[step_i][idx : idx + 1].to(device)
                y_batch = cond[idx : idx + 1]
                U, S, Vt, x_pred_np = _compute_jacobian_svd(model, z_t, t_val, y_batch)
                ang_deriv = _angular_derivative(x_pred_np)
                ang_deriv_norm = ang_deriv / (np.linalg.norm(ang_deriv) + 1e-12)
                # Max alignment across top-k U modes
                max_align = max(
                    abs(float(np.dot(U[:, k], ang_deriv_norm)))
                    for k in range(min(top_k_sweep, U.shape[1]))
                )
                step_alignments.append(max_align)
            alignment_vs_t[cls].append(float(np.mean(step_alignments)))
        print(f"    Step {step_i}/{n_steps - 1} done")

    t_values = [
        timestep_values[min(s, len(timestep_values) - 1)].item() for s in probe_steps
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    for cls in classes_to_show:
        ax.plot(
            t_values,
            alignment_vs_t[cls],
            marker="o",
            markersize=4,
            label=f"class {cls}",
        )
    ax.set_xlabel("ODE time t")
    ax.set_ylabel("max |cos sim| (top-10 U modes)")
    ax.set_title("Rotation alignment in Jacobian U modes along ODE trajectory")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "exp3_alignment_vs_t.png", dpi=300)
    plt.close(fig)
    print("  Saved exp3_alignment_vs_t.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    utils.setup_torch()
    device = utils.get_torch_device().type

    model = load_checkpoint(args.experiment, args.checkpoint_dir, device)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.n_samples < 50 and args.run in ("all", "1"):
        print(
            f"WARNING: n_samples={args.n_samples} is low. "
            "Use --n-samples 100+ for reliable results."
        )

    if args.run == "4":
        exp4_rotation_circle_grid(
            model,
            n_classes=args.n_classes,
            orbit_samples_per_class=args.orbit_samples_per_class,
            orbit_angles=args.orbit_angles,
            orbit_grid_rows=args.orbit_grid_rows,
            steer_iterations=args.steer_iterations,
            steer_gain_angle_deg=args.steer_gain_angle_deg,
            steer_max_step_angle_deg=args.steer_max_step_angle_deg,
        )
        print("Done.")
        return

    samples_per_class = args.n_samples
    if args.run == "2":
        samples_per_class = args.orbit_samples_per_class

    cond = torch.arange(args.n_classes, device=device).repeat_interleave(
        samples_per_class
    )
    labels = cond.cpu().numpy()

    run = args.run

    print(f"Generating {cond.shape[0]} samples with trajectory...")
    with torch.inference_mode():
        final_z, trajectory = generate_with_trajectory(model, cond)

    angles = None
    if run in ("all", "1"):
        images = final_z.float().cpu().view(-1, 1, 28, 28).clamp(-1, 1) * 0.5 + 0.5
        angles = measure_rotation_pca(images.numpy(), labels, args.n_classes)

    if run in ("all", "1"):
        assert angles is not None
        exp1_rotation_decodability(trajectory, labels, angles, args.n_classes)

    if run in ("all", "2"):
        exp2_rotation_orbits(
            model,
            trajectory,
            cond,
            labels,
            args.n_classes,
            orbit_samples_per_class=args.orbit_samples_per_class,
            orbit_angles=args.orbit_angles,
        )

    if run in ("all", "3"):
        exp3_jacobian_analysis(model, trajectory, cond, labels, args.n_classes)

    print("Done.")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
