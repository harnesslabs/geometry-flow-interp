"""Geometric holonomy in flow-matching activations.

Measures rotation tracking fidelity and geometric structure in the representation
bundle of a flow-matching JiT vision transformer.  We rotate an image through
θ ∈ [0, 2π) and track how the internal activations respond — measuring both
the quality of rotation encoding (tracking score) and the curvature of the
learned representation manifold (Wilson-loop holonomy).

Figures produced
----------------
1. Tracking heatmap       – rotation tracking score + PCA quality across (layer × step)
2. Phase velocity profile – dα/dθ at peak cell, per-class fingerprint
3. PCA orbits             – multi-class 2-D orbits at peak cell coloured by θ
4. Berry curvature field  – Wilson-loop curvature F(θ, t) at peak layer
5. Parallel transport     – cumulative frame rotation around the orbit
6. Class comparison       – tracking score per digit class at peak cell
7. Controls               – rotation vs scrambled vs reversed vs untrained
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch
from scipy.linalg import orthogonal_procrustes

from geoflow import utils
from scripts.analysis import (
    BlockActivationCollector,
    _image_tokens,
    _select_orbit_indices,
    circular_corr,
    generate_with_trajectory,
    load_checkpoint,
    pca,
    rotate_batch,
)
from geoflow.denoiser import Denoiser

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Berry phase / holonomy analysis")
parser.add_argument("--experiment", type=str, required=True)
parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
parser.add_argument("--n-classes", type=int, default=10)
parser.add_argument(
    "--orbit-samples-per-class",
    type=int,
    default=5,
    help="Base trajectories per class",
)
parser.add_argument(
    "--orbit-angles",
    type=int,
    default=72,
    help="Orbit resolution N (points on the circle)",
)
parser.add_argument(
    "--convergence-values",
    type=str,
    default="12,24,36,48,72,120",
    help="Comma-separated N values for convergence study",
)
parser.add_argument(
    "--berry-theta-grid",
    type=int,
    default=36,
    help="Angular resolution for curvature field",
)
parser.add_argument(
    "--run",
    default="all",
    choices=["all", "deficit", "convergence", "berry", "controls", "paper"],
)
parser.add_argument(
    "--untrained-control",
    action="store_true",
    help="Include untrained model control (slow)",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUT_DIR = Path("media/holonomy")
PCA_QUALITY_THRESHOLD = 0.50  # min top-2 PCA explained variance to trust orbit


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class HolonomyScan:
    """Rotation tracking measurements across all (layer, step) cells."""

    tracking_score: np.ndarray  # (n_layers, n_steps) mean PCA tracking ρ
    structure_score: np.ndarray  # (n_layers, n_steps) mean RSA structure score
    decode_score: np.ndarray  # (n_layers, n_steps) mean linear decode score
    winding_consistency: np.ndarray  # (n_layers, n_steps) fraction with |n|=1
    pca_quality: np.ndarray  # (n_layers, n_steps) top-2 PCA explained variance
    holonomy_angle: np.ndarray  # (n_layers, n_steps) parallel-transport holonomy (rad)
    layer_names: list[str]
    peak_layer: str
    peak_step: int
    t_peak: float
    per_class_tracking: np.ndarray  # (n_classes, n_layers, n_steps)
    per_class_structure: np.ndarray  # (n_classes, n_layers, n_steps)
    per_class_decode: np.ndarray  # (n_classes, n_layers, n_steps)
    per_class_quality: np.ndarray  # (n_classes, n_layers, n_steps)
    per_class_spectra: np.ndarray  # (n_classes, n_layers, n_steps, n_modes)


@dataclass
class BerryCurvatureField:
    """Discrete Berry curvature on the (θ, t) parameter space."""

    theta_centers: np.ndarray
    t_centers: np.ndarray
    curvature: np.ndarray  # (n_theta, n_t)
    total_curvature: float
    layer_name: str


@dataclass
class ParallelTransportResult:
    """Parallel transport of a frame around a closed orbit."""

    cumulative_angle: np.ndarray  # (N+1,) angle at each step
    holonomy_angle: float  # total rotation at closure
    frame_dim: int


# ---------------------------------------------------------------------------
# Geometry functions (pure numpy)
# ---------------------------------------------------------------------------


@dataclass
class OrbitMetrics:
    """Metrics for a single rotation orbit through activation space."""

    tracking_score: float  # circular correlation ρ ∈ [0, 1], higher = better
    structure_score: float  # RSA: corr(Gram matrix, cos(Δθ)), uses full D dims
    decode_score: float  # linear probe: held-out circular correlation
    spectrum: np.ndarray  # (n_modes,) Fourier power spectrum of orbit
    winding_number: int  # integer turns in PCA space
    phase_velocity: np.ndarray  # (N,) local dα/dθ
    pca_proj: np.ndarray  # (N, 2) orbit in PCA space
    pca_explained: np.ndarray  # (2,) PCA explained variance
    pca_quality: float  # sum of top-2 explained variance
    eigenratio: float  # λ₁/λ₂, circularity measure (1 = circle)


def fourier_spectrum(h: np.ndarray, n_modes: int = 6) -> np.ndarray:
    """Fourier power spectrum of the activation orbit h(θ).

    Decomposes h(θ) = Σ_n c_n · e^{inθ} and returns the fraction of total
    variance in each mode n = 0, 1, ..., n_modes-1.

    Parameters
    ----------
    h : (N, D) activations at N equally-spaced angles θ_k = 2πk/N.
    n_modes : number of Fourier modes to return.

    Returns
    -------
    power : (n_modes,) fraction of total power in each harmonic.
        power[0] = DC (rotation-invariant), power[1] = fundamental (SO(2)),
        power[2] = 2-fold symmetry, etc.
    """
    # Mean-center to remove DC component — we want the variance decomposition
    # i.e. what fraction of rotation-dependent signal lives in each harmonic.
    h_c = h - h.mean(axis=0, keepdims=True)
    # DFT along the angle axis (axis=0).
    H = np.fft.fft(h_c, axis=0)  # (N, D) complex
    # Power per mode: ||c_n||^2 summed over all D dimensions
    P = np.real(H * np.conj(H)).sum(axis=1)  # (N,)
    # Normalize to fractions
    total = P.sum()
    if total < 1e-12:
        return np.zeros(n_modes)
    P = P / total
    return P[:n_modes]


def _rsa_structure_score(h: np.ndarray, angles: np.ndarray) -> float:
    """Representational Similarity Analysis: correlation between the cosine-
    similarity Gram matrix and the expected cos(θ_i − θ_j) structure.

    Works in the full D-dimensional space — no PCA needed.
    """
    h_norm = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-12)
    G = h_norm @ h_norm.T  # (N, N) cosine similarities
    E = np.cos(angles[:, None] - angles[None, :])  # expected structure
    mask = np.triu_indices(len(angles), k=1)
    g = G[mask]
    e = E[mask]
    if g.std() < 1e-12 or e.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(g, e)[0, 1])


def _linear_decode_score(
    h: np.ndarray, angles: np.ndarray, alpha: float = 10.0
) -> float:
    """Held-out linear probe: ridge regression from activations to (cos θ, sin θ).

    Uses the kernel (dual) form for efficiency when D >> N.
    """
    N = h.shape[0]
    if N < 6:
        return 0.0

    Y = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    train = np.arange(0, N, 2)
    test = np.arange(1, N, 2)

    # Standardize
    mu = h[train].mean(axis=0)
    std = h[train].std(axis=0) + 1e-6
    X = (h - mu) / std

    # Dual ridge: solve (K + αI) v = Y, then predict via kernel
    K = X[train] @ X[train].T  # (N_train, N_train)
    v = np.linalg.solve(K + alpha * np.eye(len(train)), Y[train])
    pred = (X[test] @ X[train].T) @ v

    pred_angles = np.arctan2(pred[:, 1], pred[:, 0])
    return circular_corr(pred_angles, angles[test])


def compute_orbit_metrics(h: np.ndarray, input_angles: np.ndarray) -> OrbitMetrics:
    """Measure rotation tracking and orbit geometry.

    Parameters
    ----------
    h : (N, D)  activations at N equally-spaced input angles (endpoint=False).
    input_angles : (N,)  the input rotation angles θ_k = 2πk/N.

    Returns
    -------
    OrbitMetrics with tracking score, structure score, decode score, etc.
    """
    proj, _, explained = pca(h, 2)
    pca_quality = float(explained[:2].sum())

    centroid = proj.mean(axis=0)
    centered = proj - centroid

    # Activation angle in PCA plane
    alpha = np.arctan2(centered[:, 1], centered[:, 0])

    # PCA-based tracking: max correlation between α and ±θ (phase-shift invariant)
    z_cw = np.mean(np.exp(1j * (alpha - input_angles)))
    z_ccw = np.mean(np.exp(1j * (alpha + input_angles)))
    tracking_score = float(max(abs(z_cw), abs(z_ccw)))

    # RSA structure score (full-D, no PCA needed)
    structure_score = _rsa_structure_score(h, input_angles)

    # Linear decode score (held-out ridge regression)
    decode_score = _linear_decode_score(h, input_angles)

    # Fourier power spectrum
    spectrum = fourier_spectrum(h)

    # Winding number from incremental angles
    dalpha = np.diff(alpha, append=alpha[0])
    dalpha = np.arctan2(np.sin(dalpha), np.cos(dalpha))
    total_winding = float(np.sum(dalpha))
    winding_number = round(total_winding / (2 * np.pi))

    # Phase velocity
    dtheta = np.diff(input_angles, append=input_angles[0] + 2 * np.pi)
    dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
    phase_velocity = dalpha / (np.abs(dtheta) + 1e-12)

    # Eigenvalue ratio (circularity): λ₁/λ₂, closer to 1 = more circular
    eigenratio = float(explained[0] / max(explained[1], 1e-12))

    return OrbitMetrics(
        tracking_score=tracking_score,
        structure_score=structure_score,
        decode_score=decode_score,
        spectrum=spectrum,
        winding_number=winding_number,
        phase_velocity=phase_velocity,
        pca_proj=proj,
        pca_explained=explained,
        pca_quality=pca_quality,
        eigenratio=eigenratio,
    )


def parallel_transport_holonomy(
    h: np.ndarray, pca_dim: int = 8, frame_dim: int = 2
) -> ParallelTransportResult:
    """Parallel transport in PCA-reduced subspace for stability.

    1. Project the full orbit to ``pca_dim`` dimensions (denoising).
    2. At each point, fit a local ``frame_dim``-D frame from circular neighbours.
    3. Align consecutive frames via Procrustes; accumulate rotation.
    """
    n = h.shape[0]
    effective_pca_dim = min(pca_dim, h.shape[1], n - 1)
    effective_frame_dim = min(frame_dim, effective_pca_dim)

    # Denoise: project orbit to PCA subspace
    h_proj, _, _ = pca(h, effective_pca_dim)  # (N, pca_dim)

    # Window size: ~1/6 of orbit, at least 3 points each side
    window = max(3, n // 6)

    # Fit local frames in PCA subspace
    frames: list[np.ndarray] = []
    for k in range(n):
        indices = [(k + j) % n for j in range(-window, window + 1)]
        local = h_proj[indices]
        _, components, _ = pca(local, effective_frame_dim)
        frames.append(components.T)  # (pca_dim, frame_dim)

    # Transport around the closed loop
    R_total = np.eye(effective_frame_dim)
    cumulative_angle = np.zeros(n + 1)
    for k in range(n):
        A = frames[k]
        B = frames[(k + 1) % n]
        R, _ = orthogonal_procrustes(A, B)
        R_total = R_total @ R
        cumulative_angle[k + 1] = float(np.arctan2(R_total[1, 0], R_total[0, 0]))

    holonomy = float(np.arctan2(R_total[1, 0], R_total[0, 0]))
    return ParallelTransportResult(
        cumulative_angle=cumulative_angle,
        holonomy_angle=holonomy,
        frame_dim=effective_frame_dim,
    )


def berry_curvature_plaquette(h_grid: np.ndarray) -> float:
    """Discrete Berry curvature for a plaquette via the Wilson-loop log-overlap.

    Parameters
    ----------
    h_grid : (2, 2, D)  corners ordered (θ_i, t_j) → (θ+dθ, t_j) → etc.

    Returns
    -------
    F ≈ −log ∏|⟨ψ_i|ψ_{i+1}⟩|,  positive where the bundle is curved.
    """
    corners = []
    for i in range(2):
        for j in range(2):
            v = h_grid[i, j]
            nrm = np.linalg.norm(v)
            if nrm < 1e-12:
                return 0.0
            corners.append(v / nrm)

    # Walk: (0,0)→(1,0)→(1,1)→(0,1)→(0,0)
    order = [corners[0], corners[1], corners[3], corners[2]]
    log_overlap = 0.0
    for idx in range(4):
        overlap = float(np.dot(order[idx], order[(idx + 1) % 4]))
        log_overlap += np.log(max(abs(overlap), 1e-12))
    return -log_overlap


# ---------------------------------------------------------------------------
# Orbit collection helpers
# ---------------------------------------------------------------------------


def _collect_single_orbit(
    net: torch.nn.Module,
    base: torch.Tensor,
    t_scalar: float,
    y: torch.Tensor,
    angle_torch: torch.Tensor,
    collector: BlockActivationCollector,
    layer_names: list[str],
    in_context_start: int,
    in_context_len: int,
) -> dict[str, np.ndarray]:
    """Forward one orbit and return {layer: (N, D)} activations."""
    n_angles = len(angle_torch)
    device = base.device
    orbit = rotate_batch(base.repeat(n_angles, 1, 1, 1), angle_torch)
    t_batch = torch.full((n_angles,), t_scalar, device=device)
    y_batch = y.repeat(n_angles)

    collector.clear()
    with torch.no_grad(), utils.maybe_autocast(str(device)):
        net(orbit, t_batch, y_batch)

    result: dict[str, np.ndarray] = {}
    for layer_name in layer_names:
        tokens = _image_tokens(
            collector.activations[layer_name],
            layer_name,
            in_context_start,
            in_context_len,
        )
        result[layer_name] = tokens.reshape(n_angles, -1)
    return result


# ---------------------------------------------------------------------------
# Main scan
# ---------------------------------------------------------------------------


def scan_holonomy(
    model: Denoiser,
    trajectory: list[torch.Tensor],
    cond: torch.Tensor,
    labels: np.ndarray,
    n_classes: int,
    orbit_samples_per_class: int,
    n_angles: int,
) -> HolonomyScan:
    """Measure Berry phase and PCA quality across all (layer, step) cells."""
    net = model.net
    device = cond.device
    in_context_start = net.in_context_start
    in_context_len = net.in_context_len
    t_values = torch.linspace(0.0, 1.0, model.config.num_sampling_steps + 1)

    selected = _select_orbit_indices(labels, n_classes, orbit_samples_per_class)
    if selected.size == 0:
        raise ValueError("No trajectories available")

    layer_names = ["x_embedder", *[f"block_{i}" for i in range(len(net.blocks))]]
    n_layers = len(layer_names)
    n_steps = len(trajectory)

    tracking_all = np.zeros((n_layers, n_steps))
    structure_all = np.zeros((n_layers, n_steps))
    decode_all = np.zeros((n_layers, n_steps))
    winding_consistency_all = np.zeros((n_layers, n_steps))
    quality_all = np.zeros((n_layers, n_steps))
    holonomy_all = np.zeros((n_layers, n_steps))
    per_class_tracking = np.zeros((n_classes, n_layers, n_steps))
    per_class_structure = np.zeros((n_classes, n_layers, n_steps))
    per_class_decode = np.zeros((n_classes, n_layers, n_steps))
    per_class_quality = np.zeros((n_classes, n_layers, n_steps))
    per_class_counts = np.zeros((n_classes, n_layers, n_steps))
    n_modes = 6
    per_class_spectra = np.zeros((n_classes, n_layers, n_steps, n_modes))

    angle_values = np.linspace(0.0, 2 * np.pi, n_angles, endpoint=False)
    angle_torch = torch.tensor(angle_values, device=device, dtype=torch.float32)

    print(
        f"  Scanning {selected.size} orbits × {n_steps} steps × {n_layers} layers "
        f"(N={n_angles})"
    )

    collector = BlockActivationCollector(net)

    for step_i in range(n_steps):
        t_scalar = float(t_values[min(step_i, len(t_values) - 1)].item())
        step_tracking: list[list[float]] = [[] for _ in range(n_layers)]
        step_structure: list[list[float]] = [[] for _ in range(n_layers)]
        step_decode: list[list[float]] = [[] for _ in range(n_layers)]
        step_winding: list[list[int]] = [[] for _ in range(n_layers)]
        step_quality: list[list[float]] = [[] for _ in range(n_layers)]
        step_holonomy: list[list[float]] = [[] for _ in range(n_layers)]

        for base_idx in selected:
            cls = int(labels[base_idx])
            base = trajectory[step_i][base_idx : base_idx + 1].to(device).float()
            acts = _collect_single_orbit(
                net,
                base,
                t_scalar,
                cond[base_idx : base_idx + 1],
                angle_torch,
                collector,
                layer_names,
                in_context_start,
                in_context_len,
            )

            for li, layer_name in enumerate(layer_names):
                h = acts[layer_name]
                m = compute_orbit_metrics(h, angle_values)
                step_tracking[li].append(m.tracking_score)
                step_structure[li].append(m.structure_score)
                step_decode[li].append(m.decode_score)
                step_winding[li].append(m.winding_number)
                step_quality[li].append(m.pca_quality)

                # Parallel transport (only for blocks, skip x_embedder for speed)
                if layer_name != "x_embedder":
                    pt = parallel_transport_holonomy(h)
                    step_holonomy[li].append(pt.holonomy_angle)
                else:
                    step_holonomy[li].append(0.0)

                per_class_tracking[cls, li, step_i] += m.tracking_score
                per_class_structure[cls, li, step_i] += m.structure_score
                per_class_decode[cls, li, step_i] += m.decode_score
                per_class_quality[cls, li, step_i] += m.pca_quality
                per_class_spectra[cls, li, step_i] += m.spectrum[:n_modes]
                per_class_counts[cls, li, step_i] += 1

        for li in range(n_layers):
            tracking_all[li, step_i] = float(np.mean(step_tracking[li]))
            structure_all[li, step_i] = float(np.mean(step_structure[li]))
            decode_all[li, step_i] = float(np.mean(step_decode[li]))
            windings = np.array(step_winding[li])
            winding_consistency_all[li, step_i] = float(np.mean(np.abs(windings) == 1))
            quality_all[li, step_i] = float(np.mean(step_quality[li]))
            holonomy_all[li, step_i] = float(np.mean(step_holonomy[li]))

        best_li = int(np.argmax(structure_all[1:, step_i])) + 1  # skip x_embedder
        print(
            f"  Step {step_i:02d}/{n_steps - 1}: "
            f"RSA={structure_all[best_li, step_i]:.3f} "
            f"decode={decode_all[best_li, step_i]:.3f} "
            f"PCA-track={tracking_all[best_li, step_i]:.3f} "
            f"at {layer_names[best_li]}"
        )

    collector.remove()

    # Average per-class
    mask = per_class_counts > 0
    per_class_tracking[mask] /= per_class_counts[mask]
    per_class_structure[mask] /= per_class_counts[mask]
    per_class_decode[mask] /= per_class_counts[mask]
    per_class_quality[mask] /= per_class_counts[mask]
    # Average spectra: expand counts to broadcast over n_modes dim
    counts_4d = per_class_counts[..., np.newaxis]
    counts_4d = np.where(counts_4d > 0, counts_4d, 1.0)
    per_class_spectra /= counts_4d

    # Find peak: highest structure score among blocks (exclude x_embedder)
    score = structure_all.copy()
    score[0, :] = 0  # exclude x_embedder
    if score.max() == 0:
        score = decode_all.copy()
        score[0, :] = 0
    peak_li, peak_step = np.unravel_index(np.argmax(score), score.shape)
    peak_layer = layer_names[int(peak_li)]
    t_peak = float(t_values[min(int(peak_step), len(t_values) - 1)].item())

    return HolonomyScan(
        tracking_score=tracking_all,
        structure_score=structure_all,
        decode_score=decode_all,
        winding_consistency=winding_consistency_all,
        pca_quality=quality_all,
        holonomy_angle=holonomy_all,
        layer_names=layer_names,
        peak_layer=peak_layer,
        peak_step=int(peak_step),
        t_peak=t_peak,
        per_class_tracking=per_class_tracking,
        per_class_structure=per_class_structure,
        per_class_decode=per_class_decode,
        per_class_quality=per_class_quality,
        per_class_spectra=per_class_spectra,
    )


def compute_berry_curvature_field(
    model: Denoiser,
    trajectory: list[torch.Tensor],
    cond: torch.Tensor,
    base_idx: int,
    layer_name: str,
    n_theta: int = 36,
) -> BerryCurvatureField:
    """Compute Berry curvature on the (θ, t) parameter space at a given layer."""
    net = model.net
    device = cond.device
    in_context_start = net.in_context_start
    in_context_len = net.in_context_len
    t_values = torch.linspace(0.0, 1.0, model.config.num_sampling_steps + 1)
    n_steps = len(trajectory)

    theta_edges = np.linspace(0.0, 2 * np.pi, n_theta + 1)
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    t_centers = np.array(
        [float(t_values[min(s, len(t_values) - 1)].item()) for s in range(n_steps)]
    )

    collector = BlockActivationCollector(net)
    h_grid: dict[tuple[int, int], np.ndarray] = {}

    for step_i in range(n_steps):
        t_scalar = float(t_values[min(step_i, len(t_values) - 1)].item())
        angle_torch = torch.tensor(theta_edges, device=device, dtype=torch.float32)
        base = trajectory[step_i][base_idx : base_idx + 1].to(device).float()
        orbit = rotate_batch(base.repeat(len(theta_edges), 1, 1, 1), angle_torch)
        t_batch = torch.full((len(theta_edges),), t_scalar, device=device)
        y_batch = cond[base_idx : base_idx + 1].repeat(len(theta_edges))

        collector.clear()
        with torch.no_grad(), utils.maybe_autocast(str(device)):
            net(orbit, t_batch, y_batch)

        tokens = _image_tokens(
            collector.activations[layer_name],
            layer_name,
            in_context_start,
            in_context_len,
        )
        h_flat = tokens.reshape(len(theta_edges), -1)
        for ti in range(len(theta_edges)):
            h_grid[(ti, step_i)] = h_flat[ti]

    collector.remove()

    curvature = np.zeros((n_theta, max(n_steps - 1, 1)))
    for ti in range(n_theta):
        for si in range(n_steps - 1):
            corners = np.stack(
                [
                    h_grid[(ti, si)],
                    h_grid[(ti + 1, si)],
                    h_grid[(ti, si + 1)],
                    h_grid[(ti + 1, si + 1)],
                ]
            ).reshape(2, 2, -1)
            curvature[ti, si] = berry_curvature_plaquette(corners)

    return BerryCurvatureField(
        theta_centers=theta_centers,
        t_centers=t_centers[: curvature.shape[1]],
        curvature=curvature,
        total_curvature=float(np.sum(curvature)),
        layer_name=layer_name,
    )


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------


@dataclass
class ControlResult:
    """Results from a control condition."""

    tracking_scores: list[float]  # tracking score per orbit
    windings: list[int]  # winding numbers per orbit
    qualities: list[float]  # PCA qualities per orbit


def _run_control_orbit(
    model: Denoiser,
    trajectory: list[torch.Tensor],
    cond: torch.Tensor,
    selected: np.ndarray,
    labels: np.ndarray,
    step_index: int,
    n_angles: int,
    layer_name: str,
    mode: str = "rotation",
) -> ControlResult:
    """Run a single control condition and return per-orbit results.

    mode:
        "rotation"  — standard rotation orbit θ: 0→2π (default)
        "noise"     — Gaussian noise perturbation (same pixel magnitude as rotation)
    """
    net = model.net
    device = cond.device
    in_context_start = net.in_context_start
    in_context_len = net.in_context_len
    t_values = torch.linspace(0.0, 1.0, model.config.num_sampling_steps + 1)
    t_scalar = float(t_values[min(step_index, len(t_values) - 1)].item())

    angle_values = np.linspace(0.0, 2 * np.pi, n_angles, endpoint=False)
    angle_torch = torch.tensor(angle_values, device=device, dtype=torch.float32)

    collector = BlockActivationCollector(net)
    tracking_scores: list[float] = []
    windings: list[int] = []
    qualities: list[float] = []
    for base_idx in selected:
        base = trajectory[step_index][base_idx : base_idx + 1].to(device).float()

        if mode == "noise":
            # Replace rotation with random noise of same pixel magnitude
            rotated = rotate_batch(base.repeat(n_angles, 1, 1, 1), angle_torch)
            rotation_magnitude = (rotated - base).std()
            rng = torch.Generator(device=device).manual_seed(42 + int(base_idx))
            noise = torch.randn(n_angles, *base.shape[1:], device=device, generator=rng)
            orbit = base + noise * rotation_magnitude
            t_batch = torch.full((n_angles,), t_scalar, device=device)
            y_batch = cond[base_idx : base_idx + 1].repeat(n_angles)
            collector.clear()
            with torch.no_grad(), utils.maybe_autocast(str(device)):
                net(orbit, t_batch, y_batch)
            tokens = _image_tokens(
                collector.activations[layer_name],
                layer_name,
                in_context_start,
                in_context_len,
            )
            h = tokens.reshape(n_angles, -1)
        else:
            acts = _collect_single_orbit(
                net,
                base,
                t_scalar,
                cond[base_idx : base_idx + 1],
                angle_torch,
                collector,
                [layer_name],
                in_context_start,
                in_context_len,
            )
            h = acts[layer_name]

        m = compute_orbit_metrics(h, angle_values)
        tracking_scores.append(m.tracking_score)
        windings.append(m.winding_number)
        qualities.append(m.pca_quality)

    collector.remove()
    return ControlResult(
        tracking_scores=tracking_scores, windings=windings, qualities=qualities
    )


def _run_untrained_control(
    experiment: str,
    checkpoint_dir: str,
    trajectory: list[torch.Tensor],
    cond: torch.Tensor,
    selected: np.ndarray,
    labels: np.ndarray,
    step_index: int,
    n_angles: int,
    layer_name: str,
    device_str: str,
) -> ControlResult:
    """Same rotation orbits through an untrained (random weights) model."""
    path = Path(checkpoint_dir) / experiment / "last.pt"
    ckpt = torch.load(path, map_location=device_str)
    from geoflow.denoiser import DenoiserConfig

    config = DenoiserConfig(**ckpt["config"])
    fresh = Denoiser(config, device_str).to(device_str)
    fresh.eval()

    net = fresh.net
    in_context_start = net.in_context_start
    in_context_len = net.in_context_len
    t_values = torch.linspace(0.0, 1.0, fresh.config.num_sampling_steps + 1)
    t_scalar = float(t_values[min(step_index, len(t_values) - 1)].item())
    device = torch.device(device_str)

    angle_values = np.linspace(0.0, 2 * np.pi, n_angles, endpoint=False)
    angle_torch = torch.tensor(angle_values, device=device, dtype=torch.float32)

    collector = BlockActivationCollector(net)
    tracking_scores: list[float] = []
    windings: list[int] = []
    qualities: list[float] = []
    for base_idx in selected:
        base = trajectory[step_index][base_idx : base_idx + 1].to(device).float()
        acts = _collect_single_orbit(
            net,
            base,
            t_scalar,
            cond[base_idx : base_idx + 1],
            angle_torch,
            collector,
            [layer_name],
            in_context_start,
            in_context_len,
        )
        h = acts[layer_name]
        m = compute_orbit_metrics(h, angle_values)
        tracking_scores.append(m.tracking_score)
        windings.append(m.winding_number)
        qualities.append(m.pca_quality)

    collector.remove()
    return ControlResult(
        tracking_scores=tracking_scores, windings=windings, qualities=qualities
    )


# ---------------------------------------------------------------------------
# Figure functions
# ---------------------------------------------------------------------------


def plot_holonomy_heatmap(scan: HolonomyScan, model: Denoiser) -> None:
    """Figure 1: Tracking score, winding consistency, and PCA quality heatmaps."""
    n_layers = len(scan.layer_names)
    n_steps = scan.tracking_score.shape[1]
    t_values = torch.linspace(0.0, 1.0, model.config.num_sampling_steps + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    xticks = list(range(n_steps))
    xticklabels = [f"{t_values[i].item():.2f}" for i in range(n_steps)]
    peak_li = scan.layer_names.index(scan.peak_layer)

    # Panel 1: Tracking score
    ax = axes[0]
    im = ax.imshow(
        scan.tracking_score,
        aspect="auto",
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
    )
    ax.scatter([scan.peak_step], [peak_li], marker="x", color="black", s=100, lw=2)
    ax.set_title("Rotation Tracking Score ρ")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="ρ")

    # Panel 2: Winding consistency
    ax = axes[1]
    im = ax.imshow(
        scan.winding_consistency,
        aspect="auto",
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
    )
    ax.scatter([scan.peak_step], [peak_li], marker="x", color="red", s=100, lw=2)
    ax.set_title("Winding Consistency (frac |n|=1)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="fraction")

    # Panel 3: PCA quality
    ax = axes[2]
    im = ax.imshow(scan.pca_quality, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.scatter([scan.peak_step], [peak_li], marker="x", color="red", s=100, lw=2)
    ax.set_title("PCA Quality (top-2 explained var)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="explained var")

    for ax in axes:
        ax.set_xlabel("ODE time t")
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, rotation=45, ha="right", fontsize=7)
    axes[0].set_yticks(range(n_layers))
    axes[0].set_yticklabels(scan.layer_names, fontsize=8)
    axes[0].set_ylabel("Layer")

    fig.suptitle("Rotation Geometry in Flow-Matching Activations", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig1_holonomy_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig1_holonomy_heatmap.png")


def plot_phase_velocity(
    phase_velocities: dict[int, np.ndarray],
    angles: np.ndarray,
    layer_name: str,
    t_value: float,
) -> None:
    """Figure 2: Phase velocity dα/dθ at the peak cell, per class."""
    fig, ax = plt.subplots(figsize=(8, 5))
    theta_deg = np.degrees(angles)
    for cls, pv in sorted(phase_velocities.items()):
        ax.plot(theta_deg, pv, linewidth=1.2, alpha=0.7, label=f"class {cls}")
    ax.axhline(-1.0, color="black", linestyle="--", linewidth=1, label="ideal (±1)")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Input rotation θ (°)")
    ax.set_ylabel("Phase velocity dα/dθ")
    ax.set_title(
        f"Berry connection: phase velocity profile\n{layer_name}, t={t_value:.2f}"
    )
    ax.legend(fontsize=7, ncol=5, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 360)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2_phase_velocity.png", dpi=300)
    plt.close(fig)
    print("  Saved fig2_phase_velocity.png")


def plot_pca_orbits(
    orbits_by_class: dict[int, np.ndarray],
    angles: np.ndarray,
    explained_by_class: dict[int, np.ndarray],
    layer_name: str,
    t_value: float,
) -> None:
    """Figure 3: Multi-class PCA orbits at peak cell."""
    classes = sorted(orbits_by_class.keys())
    n_show = min(len(classes), 5)
    show_classes = classes[:n_show]

    fig, axes = plt.subplots(1, n_show, figsize=(3.5 * n_show, 3.5), squeeze=False)
    scatter = None
    for ax, cls in zip(axes[0], show_classes, strict=False):
        proj = orbits_by_class[cls]
        expl = explained_by_class[cls]
        scatter = ax.scatter(
            proj[:, 0],
            proj[:, 1],
            c=np.degrees(angles),
            cmap="hsv",
            s=20,
            vmin=0,
            vmax=360,
            zorder=3,
        )
        ax.plot(proj[:, 0], proj[:, 1], color="0.7", lw=0.8, alpha=0.6, zorder=2)
        # Connect last to first (close the loop visually)
        ax.plot(
            [proj[-1, 0], proj[0, 0]],
            [proj[-1, 1], proj[0, 1]],
            color="0.7",
            lw=0.8,
            alpha=0.6,
            zorder=2,
        )
        ax.scatter([proj[0, 0]], [proj[0, 1]], c="red", s=60, marker="*", zorder=5)
        ax.set_title(f"class {cls}\nvar={expl.sum():.0%}", fontsize=10)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)

    if scatter is not None:
        fig.colorbar(
            scatter, ax=axes[0].tolist(), fraction=0.02, pad=0.03, label="θ (°)"
        )
    fig.suptitle(f"Rotation orbits at {layer_name}, t={t_value:.2f}", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig3_pca_orbits.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig3_pca_orbits.png")


def plot_berry_curvature(field: BerryCurvatureField) -> None:
    """Figure 4: Berry curvature pcolormesh on (θ, t)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    vmax = max(abs(field.curvature.min()), abs(field.curvature.max()), 1e-6)
    im = ax.pcolormesh(
        field.t_centers,
        np.degrees(field.theta_centers),
        field.curvature,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        shading="auto",
    )
    fig.colorbar(im, ax=ax, label="Berry curvature F(θ,t)")
    ax.set_xlabel("ODE time t")
    ax.set_ylabel("Rotation angle θ (°)")
    ax.set_title(
        f"Berry curvature field at {field.layer_name}\n"
        f"∫∫ F dθ dt = {field.total_curvature:.2f}"
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig4_berry_curvature.png", dpi=300)
    plt.close(fig)
    print("  Saved fig4_berry_curvature.png")


def plot_frame_rotation(
    result: ParallelTransportResult, layer_name: str, t_value: float
) -> None:
    """Figure 5: Cumulative parallel-transport angle vs θ."""
    n = len(result.cumulative_angle)
    theta = np.linspace(0.0, 360.0, n, endpoint=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(theta, np.degrees(result.cumulative_angle), linewidth=2, color="tab:blue")
    ax.axhline(0.0, color="0.5", linestyle="--", linewidth=1)
    ax.scatter(
        [theta[-1]],
        [np.degrees(result.cumulative_angle[-1])],
        color="red",
        s=80,
        zorder=5,
        label=f"Holonomy = {np.degrees(result.holonomy_angle):+.1f}°",
    )
    ax.set_xlabel("Input rotation θ (°)")
    ax.set_ylabel("Cumulative frame rotation (°)")
    ax.set_title(
        f"Parallel transport at {layer_name}, t={t_value:.2f}\n"
        f"(PCA-{result.frame_dim} subspace)"
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig5_frame_rotation.png", dpi=300)
    plt.close(fig)
    print("  Saved fig5_frame_rotation.png")


def plot_class_comparison(scan: HolonomyScan, n_classes: int) -> None:
    """Figure 6: Tracking score per digit class at peak cell."""
    peak_li = scan.layer_names.index(scan.peak_layer)
    class_tracking = scan.per_class_tracking[:, peak_li, scan.peak_step]
    class_quality = scan.per_class_quality[:, peak_li, scan.peak_step]
    classes = np.arange(n_classes)

    symmetric = {0, 1, 8}
    colors = ["tab:blue" if c in symmetric else "tab:orange" for c in classes]

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, height_ratios=[3, 1])

    # Top: Tracking score
    ax = axes[0]
    ax.bar(classes, class_tracking, color=colors)
    ax.set_ylabel("Tracking score ρ")
    ax.set_ylim(0, 1)
    ax.set_title(
        f"Per-class rotation tracking at {scan.peak_layer}, step {scan.peak_step}"
    )
    ax.legend(
        handles=[
            Patch(facecolor="tab:blue", label="Symmetric (0, 1, 8)"),
            Patch(facecolor="tab:orange", label="Asymmetric"),
        ],
        fontsize=9,
    )
    ax.grid(True, alpha=0.3, axis="y")

    # Bottom: PCA quality
    ax = axes[1]
    ax.bar(classes, class_quality, color=colors, alpha=0.6)
    ax.set_ylabel("PCA quality")
    ax.set_xlabel("Digit class")
    ax.set_xticks(classes)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig6_class_comparison.png", dpi=300)
    plt.close(fig)
    print("  Saved fig6_class_comparison.png")


def plot_controls(
    results: dict[str, ControlResult],
    layer_name: str,
    t_value: float,
) -> None:
    """Figure 7: Tracking score comparison across control conditions."""
    conditions = list(results.keys())
    palette = {
        "rotation": "tab:blue",
        "noise": "tab:orange",
        "untrained": "tab:red",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Tracking score box plot
    ax = axes[0]
    data = [results[c].tracking_scores for c in conditions]
    bp = ax.boxplot(
        data,
        tick_labels=conditions,
        patch_artist=True,
        widths=0.5,
    )
    for patch, cond in zip(bp["boxes"], conditions):
        patch.set_facecolor(palette.get(cond, "tab:gray"))
        patch.set_alpha(0.7)
    ax.set_ylabel("Tracking score ρ")
    ax.set_title("Rotation tracking quality")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")

    # Right: Mean tracking score bar chart
    ax = axes[1]
    means = [float(np.mean(results[c].tracking_scores)) for c in conditions]
    stds = [float(np.std(results[c].tracking_scores)) for c in conditions]
    bars = ax.bar(
        conditions,
        means,
        yerr=stds,
        capsize=4,
        color=[palette.get(c, "tab:gray") for c in conditions],
        alpha=0.7,
        edgecolor="black",
    )
    for bar, val in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_ylabel("Mean tracking score ρ")
    ax.set_title("Mean ± std")
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="y")

    n_orbits = len(results.get("rotation", ControlResult([], [], [])).tracking_scores)
    fig.suptitle(
        f"Controls at {layer_name}, t={t_value:.2f} (N={n_orbits} orbits)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig7_controls.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig7_controls.png")


# ---------------------------------------------------------------------------
# Paper figures (2 publication-quality composites)
# ---------------------------------------------------------------------------


def _collect_orbits_at_steps(
    model: Denoiser,
    trajectory: list[torch.Tensor],
    cond: torch.Tensor,
    base_idx: int,
    layer_name: str,
    n_angles: int,
    step_indices: list[int],
) -> dict[int, OrbitMetrics]:
    """Collect orbit metrics at multiple ODE timesteps for one sample."""
    net = model.net
    device = cond.device
    in_context_start = net.in_context_start
    in_context_len = net.in_context_len
    t_values = torch.linspace(0.0, 1.0, model.config.num_sampling_steps + 1)

    angle_values = np.linspace(0.0, 2 * np.pi, n_angles, endpoint=False)
    angle_torch = torch.tensor(angle_values, device=device, dtype=torch.float32)

    collector = BlockActivationCollector(net)
    results: dict[int, OrbitMetrics] = {}

    for step_i in step_indices:
        t_scalar = float(t_values[min(step_i, len(t_values) - 1)].item())
        base = trajectory[step_i][base_idx : base_idx + 1].to(device).float()
        acts = _collect_single_orbit(
            net,
            base,
            t_scalar,
            cond[base_idx : base_idx + 1],
            angle_torch,
            collector,
            [layer_name],
            in_context_start,
            in_context_len,
        )
        h = acts[layer_name]
        results[step_i] = compute_orbit_metrics(h, angle_values)

    collector.remove()
    return results


def plot_paper_figure1(
    scan: HolonomyScan,
    model: Denoiser,
    trajectory: list[torch.Tensor],
    cond: torch.Tensor,
    base_idx: int,
    n_angles: int,
) -> None:
    """Paper Figure 1: Emergence of rotation geometry along the generative ODE.

    Layout (3 rows):
      Row 1: Generated images at selected ODE timesteps (noise → data)
      Row 2: PCA orbits at matching timesteps — circles emerge as data forms
      Row 3: (a) Tracking heatmap  (b) Berry curvature field
    """
    import matplotlib.gridspec as gridspec

    n_steps = scan.tracking_score.shape[1]
    t_values = torch.linspace(0.0, 1.0, model.config.num_sampling_steps + 1)

    # Select ~5 timesteps: early (noise), early-mid, mid, late-mid, final (data)
    if n_steps <= 5:
        step_indices = list(range(n_steps))
    else:
        step_indices = [
            0,
            max(1, n_steps // 5),
            n_steps // 2,
            max(n_steps // 2 + 1, n_steps * 4 // 5),
            n_steps - 1,
        ]
    # Ensure no duplicates and sorted
    step_indices = sorted(set(step_indices))
    n_cols = len(step_indices)

    # Collect orbits across time
    orbits = _collect_orbits_at_steps(
        model, trajectory, cond, base_idx, scan.peak_layer, n_angles, step_indices
    )

    # --- Build figure ---
    fig = plt.figure(figsize=(2.8 * n_cols + 1.2, 9.5))
    gs = gridspec.GridSpec(
        3,
        n_cols + 1,
        height_ratios=[1, 1.3, 1.5],
        width_ratios=[1] * n_cols + [0.06],
        hspace=0.35,
        wspace=0.3,
    )

    # Row 1: Denoising images
    for ci, step_i in enumerate(step_indices):
        ax = fig.add_subplot(gs[0, ci])
        img = trajectory[step_i][base_idx].squeeze().cpu().numpy()
        ax.imshow(img, cmap="gray", vmin=-1, vmax=1)
        t_val = float(t_values[min(step_i, len(t_values) - 1)].item())
        ax.set_title(f"t = {t_val:.2f}", fontsize=9, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        if ci == 0:
            ax.set_ylabel("ODE state $z_t$", fontsize=9)

    # Row 2: PCA orbits at each timestep
    angle_values = np.linspace(0.0, 2 * np.pi, n_angles, endpoint=False)
    # Use consistent axis limits across all orbit panels
    all_proj = [orbits[s].pca_proj for s in step_indices if s in orbits]
    if all_proj:
        global_lim = max(np.abs(np.concatenate(all_proj)).max(), 1e-3) * 1.15

    for ci, step_i in enumerate(step_indices):
        ax = fig.add_subplot(gs[1, ci])
        if step_i in orbits:
            m = orbits[step_i]
            scatter = ax.scatter(
                m.pca_proj[:, 0],
                m.pca_proj[:, 1],
                c=np.degrees(angle_values),
                cmap="hsv",
                s=18,
                vmin=0,
                vmax=360,
                zorder=3,
                edgecolors="none",
            )
            ax.plot(
                m.pca_proj[:, 0],
                m.pca_proj[:, 1],
                color="0.75",
                lw=0.6,
                alpha=0.5,
                zorder=2,
            )
            # Close the loop
            ax.plot(
                [m.pca_proj[-1, 0], m.pca_proj[0, 0]],
                [m.pca_proj[-1, 1], m.pca_proj[0, 1]],
                color="0.75",
                lw=0.6,
                alpha=0.5,
                zorder=2,
            )
            ax.set_xlim(-global_lim, global_lim)
            ax.set_ylim(-global_lim, global_lim)
            # Annotate with RSA structure score and decode score
            ax.text(
                0.05,
                0.95,
                f"RSA = {m.structure_score:.2f}\ndec = {m.decode_score:.2f}",
                transform=ax.transAxes,
                fontsize=7,
                va="top",
                ha="left",
                linespacing=1.4,
                bbox={
                    "facecolor": "white",
                    "alpha": 0.7,
                    "edgecolor": "none",
                    "pad": 1,
                },
            )
        ax.set_aspect("equal")
        ax.tick_params(labelsize=6)
        if ci == 0:
            ax.set_ylabel("PCA orbit", fontsize=9)

    # Colorbar for orbits
    cax = fig.add_subplot(gs[1, -1])
    if "scatter" in dir():
        fig.colorbar(scatter, cax=cax, label="$\\theta$ (°)")
    cax.tick_params(labelsize=7)

    # Row 3: Two heatmaps side by side — RSA structure + Linear decode
    n_half = max(n_cols // 2, 2)
    peak_li = scan.layer_names.index(scan.peak_layer)
    n_layers = len(scan.layer_names)
    xticks = list(range(n_steps))
    xticklabels = [f"{t_values[i].item():.2f}" for i in range(n_steps)]

    # (a) RSA structure heatmap
    ax_rsa = fig.add_subplot(gs[2, :n_half])
    im_rsa = ax_rsa.imshow(
        scan.structure_score,
        aspect="auto",
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    ax_rsa.scatter(
        [scan.peak_step], [peak_li], marker="x", color="black", s=80, lw=2, zorder=5
    )
    ax_rsa.set_xlabel("ODE time $t$", fontsize=9)
    ax_rsa.set_ylabel("Layer", fontsize=9)
    ax_rsa.set_xticks(xticks)
    ax_rsa.set_xticklabels(xticklabels, rotation=45, ha="right", fontsize=6)
    ax_rsa.set_yticks(range(n_layers))
    ax_rsa.set_yticklabels(scan.layer_names, fontsize=7)
    ax_rsa.set_title("(a) Geometric similarity (RSA)", fontsize=10, loc="left")
    fig.colorbar(im_rsa, ax=ax_rsa, fraction=0.03, pad=0.02, label="$r$")

    # (b) Linear decode heatmap
    ax_dec = fig.add_subplot(gs[2, n_half:-1])
    # Find decode peak
    dec_score = scan.decode_score.copy()
    dec_score[0, :] = 0  # exclude x_embedder
    dec_peak_li, dec_peak_step = np.unravel_index(np.argmax(dec_score), dec_score.shape)
    im_dec = ax_dec.imshow(
        scan.decode_score,
        aspect="auto",
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    ax_dec.scatter(
        [dec_peak_step],
        [dec_peak_li],
        marker="x",
        color="red",
        s=80,
        lw=2,
        zorder=5,
    )
    ax_dec.set_xlabel("ODE time $t$", fontsize=9)
    ax_dec.set_xticks(xticks)
    ax_dec.set_xticklabels(xticklabels, rotation=45, ha="right", fontsize=6)
    ax_dec.set_yticks(range(n_layers))
    ax_dec.set_yticklabels(scan.layer_names, fontsize=7)
    ax_dec.set_title("(b) Linear decodability", fontsize=10, loc="left")
    fig.colorbar(im_dec, ax=ax_dec, fraction=0.03, pad=0.02, label="$\\rho$")

    fig.savefig(
        OUT_DIR / "paper_fig1_geometry_emergence.png", dpi=300, bbox_inches="tight"
    )
    fig.savefig(OUT_DIR / "paper_fig1_geometry_emergence.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved paper_fig1_geometry_emergence.{png,pdf}")


def plot_paper_figure2(
    orbits_by_class: dict[int, np.ndarray],
    angles: np.ndarray,
    explained_by_class: dict[int, np.ndarray],
    structure_by_class: np.ndarray,
    decode_by_class: np.ndarray,
    trajectory: list[torch.Tensor],
    step_idx: int,
    labels: np.ndarray,
    selected: np.ndarray,
    layer_name: str,
    t_value: float,
) -> None:
    """Paper Figure 2: Class-conditional orbit geometry.

    Layout: 2×5 grid of PCA orbits (all 10 digit classes) with digit thumbnails,
    plus a summary bar chart of tracking score by class.
    """
    import matplotlib.gridspec as gridspec
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    n_classes = 10
    classes = list(range(n_classes))
    symmetric = {0, 1, 8}

    # Use FINAL trajectory step for clean digit thumbnails
    final_step = len(trajectory) - 1

    fig = plt.figure(figsize=(14, 7.5))
    gs = gridspec.GridSpec(
        3,
        5,
        height_ratios=[1.4, 1.4, 0.8],
        hspace=0.3,
        wspace=0.35,
    )

    # Consistent axis limits across all orbits
    all_proj = [orbits_by_class[c] for c in classes if c in orbits_by_class]
    global_lim = (
        max(np.abs(np.concatenate(all_proj)).max(), 1e-3) * 1.15 if all_proj else 1.0
    )

    theta_deg = np.degrees(angles)
    scatter = None

    for idx, cls in enumerate(classes):
        row = idx // 5
        col = idx % 5
        ax = fig.add_subplot(gs[row, col])

        if cls in orbits_by_class:
            proj = orbits_by_class[cls]
            rsa = (
                float(structure_by_class[cls]) if cls < len(structure_by_class) else 0.0
            )
            dec = float(decode_by_class[cls]) if cls < len(decode_by_class) else 0.0

            scatter = ax.scatter(
                proj[:, 0],
                proj[:, 1],
                c=theta_deg,
                cmap="hsv",
                s=16,
                vmin=0,
                vmax=360,
                zorder=3,
                edgecolors="none",
            )
            ax.plot(proj[:, 0], proj[:, 1], color="0.75", lw=0.5, alpha=0.5, zorder=2)
            ax.plot(
                [proj[-1, 0], proj[0, 0]],
                [proj[-1, 1], proj[0, 1]],
                color="0.75",
                lw=0.5,
                alpha=0.5,
                zorder=2,
            )

            ax.set_xlim(-global_lim, global_lim)
            ax.set_ylim(-global_lim, global_lim)

            # Score annotation
            sym_label = "*" if cls in symmetric else ""
            ax.set_title(
                f"digit {cls}{sym_label}   RSA={rsa:.2f}  dec={dec:.2f}",
                fontsize=8,
                fontweight="bold" if cls in symmetric else "normal",
                color="tab:blue" if cls in symmetric else "tab:orange",
            )

            # Inset: generated digit thumbnail (use CLEAN final image)
            class_indices = [i for i in selected if labels[i] == cls]
            if class_indices:
                img = trajectory[final_step][class_indices[0]].squeeze().cpu().numpy()
                ax_inset = inset_axes(ax, width="28%", height="28%", loc="upper right")
                ax_inset.imshow(img, cmap="gray", vmin=-1, vmax=1)
                ax_inset.set_xticks([])
                ax_inset.set_yticks([])
                for spine in ax_inset.spines.values():
                    spine.set_edgecolor("0.3")
                    spine.set_linewidth(0.8)

        ax.set_aspect("equal")
        ax.tick_params(labelsize=5)
        if col > 0:
            ax.set_yticklabels([])

    # Row 3: Summary bar chart across all 5 columns
    ax_bar = fig.add_subplot(gs[2, :])
    bar_colors = ["tab:blue" if c in symmetric else "tab:orange" for c in classes]
    rsa_values = [
        float(structure_by_class[c]) if c < len(structure_by_class) else 0.0
        for c in classes
    ]
    dec_values = [
        float(decode_by_class[c]) if c < len(decode_by_class) else 0.0 for c in classes
    ]
    x = np.arange(n_classes)
    w = 0.35
    bars_rsa = ax_bar.bar(
        x - w / 2,
        rsa_values,
        w,
        color=bar_colors,
        edgecolor="white",
        lw=0.5,
        alpha=0.85,
    )
    ax_bar.bar(
        x + w / 2,
        dec_values,
        w,
        color=bar_colors,
        edgecolor="white",
        lw=0.5,
        alpha=0.45,
    )
    for bar, val in zip(bars_rsa, rsa_values):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{val:.2f}",
            ha="center",
            fontsize=7,
            fontweight="bold",
        )
    ax_bar.set_xlabel("Digit class", fontsize=10)
    ax_bar.set_ylabel("Score", fontsize=10)
    ax_bar.set_xticks(x)
    all_vals = rsa_values + dec_values
    ax_bar.set_ylim(0, max(all_vals) * 1.2 + 0.05)
    ax_bar.axhline(0, color="0.5", lw=0.5)
    ax_bar.grid(True, alpha=0.2, axis="y")
    ax_bar.legend(
        handles=[
            Patch(facecolor="tab:blue", label="Symmetric (0, 1, 8)"),
            Patch(facecolor="tab:orange", label="Asymmetric"),
            Patch(facecolor="0.3", alpha=0.85, label="RSA (solid)"),
            Patch(facecolor="0.3", alpha=0.45, label="Decode (light)"),
        ],
        fontsize=7,
        loc="upper right",
        ncol=2,
    )

    # Suptitle
    fig.suptitle(
        f"Class-conditional rotation orbits at {layer_name}, $t$ = {t_value:.2f}",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )

    # Add colorbar for θ
    if scatter is not None:
        cax = fig.add_axes([0.92, 0.42, 0.012, 0.45])
        fig.colorbar(scatter, cax=cax, label="Input rotation $\\theta$ (°)")
        cax.tick_params(labelsize=7)

    fig.savefig(OUT_DIR / "paper_fig2_class_orbits.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "paper_fig2_class_orbits.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved paper_fig2_class_orbits.{png,pdf}")


def plot_paper_figure3(
    spectra_by_class: dict[int, np.ndarray],
    orbits_by_class: dict[int, np.ndarray],
    angles: np.ndarray,
    trajectory: list[torch.Tensor],
    step_idx: int,
    labels: np.ndarray,
    selected: np.ndarray,
    layer_name: str,
    t_value: float,
) -> None:
    """Paper Figure 3: Fourier decomposition of rotation orbits.

    Left: Domain-coloured winding maps for each digit class — the complex
    function W(e^{iθ}) = PC1(θ) + i·PC2(θ) plotted as a coloured curve
    on the complex plane, with the input circle shown for reference.

    Right: Stacked bar chart of Fourier power spectrum per digit class,
    showing how much orbit variance lives in each harmonic (n=0 invariant,
    n=1 rotation, n=2 two-fold, ...).
    """
    import matplotlib.gridspec as gridspec

    n_classes = 10
    classes = list(range(n_classes))
    symmetric = {0, 1, 8}
    n_modes = 6
    # After mean-centering, n=0 is ~0.  Display modes 1..5 only.
    display_modes = list(range(1, n_modes))
    mode_labels = [
        "$n{=}1$\nrotation",
        "$n{=}2$\n2-fold",
        "$n{=}3$\n3-fold",
        "$n{=}4$",
        "$n{=}5$",
    ]
    mode_colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]

    fig = plt.figure(figsize=(15, 7))
    gs = gridspec.GridSpec(2, 6, height_ratios=[1, 1], hspace=0.4, wspace=0.35)

    # --- Top: 10 domain-coloured winding maps (2 rows × 5 cols) ---
    all_proj = [orbits_by_class[c] for c in classes if c in orbits_by_class]
    global_lim = (
        max(np.abs(np.concatenate(all_proj)).max(), 1e-3) * 1.15 if all_proj else 1.0
    )

    # Create a 2×5 sub-grid in the left 5 columns of the top row
    gs_orbits = gridspec.GridSpecFromSubplotSpec(
        2, 5, subplot_spec=gs[0, :5], hspace=0.35, wspace=0.3
    )

    for idx, cls in enumerate(classes):
        row = idx // 5
        col = idx % 5
        ax = fig.add_subplot(gs_orbits[row, col])

        if cls in orbits_by_class:
            proj = orbits_by_class[cls]
            # Treat as complex: z = PC1 + i*PC2
            z = proj[:, 0] + 1j * proj[:, 1]

            # Draw the orbit coloured by phase of z (activation angle)
            phase = np.angle(z)  # arg(z) in [-π, π]
            scatter = ax.scatter(
                proj[:, 0],
                proj[:, 1],
                c=phase,
                cmap="hsv",
                s=14,
                vmin=-np.pi,
                vmax=np.pi,
                zorder=3,
                edgecolors="none",
            )
            ax.plot(proj[:, 0], proj[:, 1], color="0.8", lw=0.5, alpha=0.4, zorder=2)
            ax.plot(
                [proj[-1, 0], proj[0, 0]],
                [proj[-1, 1], proj[0, 1]],
                color="0.8",
                lw=0.5,
                alpha=0.4,
                zorder=2,
            )

            # Reference unit circle scaled to orbit radius
            r_mean = float(np.mean(np.abs(z)))
            circle_t = np.linspace(0, 2 * np.pi, 100)
            ax.plot(
                r_mean * np.cos(circle_t),
                r_mean * np.sin(circle_t),
                "k--",
                lw=0.5,
                alpha=0.25,
                zorder=1,
            )

            ax.set_xlim(-global_lim, global_lim)
            ax.set_ylim(-global_lim, global_lim)

            # Spectrum annotation
            if cls in spectra_by_class:
                sp = spectra_by_class[cls]
                dominant = int(np.argmax(sp[1:]) + 1)  # skip n=0
                ax.set_title(
                    f"{cls}{'*' if cls in symmetric else ''}  $n$={dominant}",
                    fontsize=8,
                    fontweight="bold" if cls in symmetric else "normal",
                    color="tab:blue" if cls in symmetric else "tab:orange",
                )

        ax.set_aspect("equal")
        ax.tick_params(labelsize=5)
        if col > 0:
            ax.set_yticklabels([])
        if row == 0:
            ax.set_xticklabels([])

    # Legend/colorbar in top-right cell
    ax_cb = fig.add_subplot(gs[0, 5])
    ax_cb.axis("off")
    if scatter is not None:
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        sm = ScalarMappable(cmap="hsv", norm=Normalize(-180, 180))
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax_cb, fraction=0.8, pad=0.05, label="arg($z$) (°)")
        cb.ax.tick_params(labelsize=7)

    # --- Bottom: Grouped bar chart of Fourier power spectrum ---
    ax_bar = fig.add_subplot(gs[1, :])

    x = np.arange(n_classes)
    n_disp = len(display_modes)
    bar_width = 0.8 / n_disp

    for i, mode_idx in enumerate(display_modes):
        values = []
        for cls in classes:
            if cls in spectra_by_class and mode_idx < len(spectra_by_class[cls]):
                values.append(float(spectra_by_class[cls][mode_idx]))
            else:
                values.append(0.0)
        values = np.array(values)
        offset = (i - n_disp / 2 + 0.5) * bar_width
        ax_bar.bar(
            x + offset,
            values,
            bar_width * 0.9,
            color=mode_colors[i],
            edgecolor="white",
            lw=0.3,
            label=mode_labels[i],
        )

    ax_bar.set_xlabel("Digit class", fontsize=11)
    ax_bar.set_ylabel("Fraction of orbit variance", fontsize=11)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(
        [f"{c}*" if c in symmetric else str(c) for c in classes], fontsize=10
    )
    ax_bar.legend(
        fontsize=8,
        ncol=n_disp,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
    )
    ax_bar.grid(True, alpha=0.15, axis="y")

    # Highlight asymmetric vs symmetric via x-tick colors
    for label in ax_bar.get_xticklabels():
        txt = label.get_text().rstrip("*")
        if txt.isdigit() and int(txt) in symmetric:
            label.set_color("tab:blue")
            label.set_fontweight("bold")
        else:
            label.set_color("tab:orange")

    fig.suptitle(
        f"Fourier decomposition of rotation orbits — {layer_name}, $t$ = {t_value:.2f}",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    fig.savefig(OUT_DIR / "paper_fig3_fourier.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "paper_fig3_fourier.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved paper_fig3_fourier.{png,pdf}")


def plot_hero_figure(
    scan: HolonomyScan,
    model: Denoiser,
    trajectory: list[torch.Tensor],
    cond: torch.Tensor,
    labels: np.ndarray,
    selected: np.ndarray,
    n_angles: int,
) -> None:
    """Hero figure: Spectral cascade in flow-matching representations.

    Three panels:
      (A) Orbit morphogenesis — PCA snapshots at 4 ODE times with spectrum bars
      (B) Spectral cascade — Fourier mode power vs ODE time at peak layer
      (C) Symmetry-dependent onset — critical time t* per digit class
    """
    import matplotlib.gridspec as gridspec
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    n_steps = scan.tracking_score.shape[1]
    n_classes = scan.per_class_spectra.shape[0]
    t_values = torch.linspace(0.0, 1.0, model.config.num_sampling_steps + 1)

    peak_li = scan.layer_names.index(scan.peak_layer)

    # Mode colors and labels (skip n=0 DC)
    mode_colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
    mode_labels = ["$n{=}1$ rotation", "$n{=}2$ 2-fold", "$n{=}3$ 3-fold", "$n{=}4$"]
    n_display = 4  # modes 1..4

    # Symmetry groups
    symmetric = {0, 8}
    asymmetric = {2, 4, 6, 7}
    group_colors = {
        "symmetric": "#3498db",
        "asymmetric": "#e74c3c",
        "other": "#95a5a6",
    }

    def class_group(c: int) -> str:
        if c in symmetric:
            return "symmetric"
        if c in asymmetric:
            return "asymmetric"
        return "other"

    # =================================================================
    # Panel A data: orbit morphogenesis for representative digit (cls 6)
    # =================================================================
    repr_cls = 6
    if n_steps <= 4:
        panel_a_steps = list(range(n_steps))
    else:
        panel_a_steps = [
            0,
            max(1, int(n_steps * 0.3)),
            max(2, int(n_steps * 0.6)),
            n_steps - 1,
        ]
    panel_a_steps = sorted(set(panel_a_steps))

    # Find a class-6 sample
    repr_idx = None
    for idx in selected:
        if int(labels[idx]) == repr_cls:
            repr_idx = int(idx)
            break
    if repr_idx is None:
        repr_idx = int(selected[0])
        repr_cls = int(labels[repr_idx])

    orbits_a = _collect_orbits_at_steps(
        model, trajectory, cond, repr_idx, scan.peak_layer, n_angles, panel_a_steps
    )

    # =================================================================
    # Panel B data: spectral cascade at peak layer
    # =================================================================
    # Average across classes → (n_steps, n_modes)
    mean_spectrum = scan.per_class_spectra[:, peak_li, :, :].mean(axis=0)

    # =================================================================
    # Panel C data: per-class peak n=1 time and peak n=1 power
    # =================================================================
    # For each class, find ODE step where n=1 fraction is maximized
    t_star = np.full(n_classes, np.nan)
    peak_n1_power = np.full(n_classes, np.nan)
    for cls in range(n_classes):
        n1_vs_t = scan.per_class_spectra[cls, peak_li, :, 1]
        if n1_vs_t.max() > 0:
            peak_step = int(np.argmax(n1_vs_t))
            t_star[cls] = float(t_values[min(peak_step, len(t_values) - 1)].item())
            peak_n1_power[cls] = float(n1_vs_t.max())

    # =================================================================
    # Build the 3-panel figure
    # =================================================================
    fig = plt.figure(figsize=(7, 4))
    gs = gridspec.GridSpec(1, 3, width_ratios=[2.2, 2.8, 2.0], wspace=0.4)

    # ----- Panel A: Orbit morphogenesis -----
    n_snapshots = len(panel_a_steps)
    gs_a = gridspec.GridSpecFromSubplotSpec(
        n_snapshots,
        2,
        subplot_spec=gs[0, 0],
        width_ratios=[0.3, 1],
        hspace=0.4,
        wspace=0.1,
    )

    all_proj_a = [orbits_a[s].pca_proj for s in panel_a_steps if s in orbits_a]
    global_lim_a = (
        max(np.abs(np.concatenate(all_proj_a)).max(), 1e-3) * 1.15
        if all_proj_a
        else 1.0
    )
    angle_values = np.linspace(0.0, 2 * np.pi, n_angles, endpoint=False)

    for row_i, step_i in enumerate(panel_a_steps):
        t_val = float(t_values[min(step_i, len(t_values) - 1)].item())

        # Left: image thumbnail
        ax_img = fig.add_subplot(gs_a[row_i, 0])
        img = trajectory[step_i][repr_idx].squeeze().cpu().numpy()
        ax_img.imshow(img, cmap="gray", vmin=-1, vmax=1)
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        ax_img.set_ylabel(f"$t$={t_val:.1f}", fontsize=7, rotation=0, labelpad=18)

        # Right: PCA orbit
        ax_orb = fig.add_subplot(gs_a[row_i, 1])
        if step_i in orbits_a:
            m = orbits_a[step_i]
            ax_orb.scatter(
                m.pca_proj[:, 0],
                m.pca_proj[:, 1],
                c=np.degrees(angle_values),
                cmap="hsv",
                s=10,
                vmin=0,
                vmax=360,
                zorder=3,
                edgecolors="none",
            )
            ax_orb.plot(
                m.pca_proj[:, 0],
                m.pca_proj[:, 1],
                color="0.75",
                lw=0.4,
                alpha=0.5,
                zorder=2,
            )
            ax_orb.plot(
                [m.pca_proj[-1, 0], m.pca_proj[0, 0]],
                [m.pca_proj[-1, 1], m.pca_proj[0, 1]],
                color="0.75",
                lw=0.4,
                alpha=0.5,
                zorder=2,
            )
            ax_orb.set_xlim(-global_lim_a, global_lim_a)
            ax_orb.set_ylim(-global_lim_a, global_lim_a)

            # Spectrum bar below orbit
            sp = m.spectrum
            ax_bar_inset = inset_axes(
                ax_orb,
                width="80%",
                height="8%",
                loc="lower center",
                borderpad=0.3,
            )
            left = 0.0
            for mi in range(n_display):
                mode_idx = mi + 1
                val = float(sp[mode_idx]) if mode_idx < len(sp) else 0.0
                ax_bar_inset.barh(0, val, left=left, color=mode_colors[mi], height=1)
                left += val
            ax_bar_inset.set_xlim(0, max(left, 0.01))
            ax_bar_inset.set_yticks([])
            ax_bar_inset.set_xticks([])
            ax_bar_inset.patch.set_alpha(0.0)
            for spine in ax_bar_inset.spines.values():
                spine.set_visible(False)

        ax_orb.set_aspect("equal")
        ax_orb.tick_params(labelsize=4)
        if row_i == 0:
            ax_orb.set_title(f"digit {repr_cls}", fontsize=8, fontweight="bold")

    fig.text(0.01, 0.97, "(A)", fontsize=10, fontweight="bold", va="top")

    # ----- Panel B: Spectral cascade -----
    ax_b = fig.add_subplot(gs[0, 1])
    t_axis = np.array(
        [float(t_values[min(i, len(t_values) - 1)].item()) for i in range(n_steps)]
    )

    for mi in range(n_display):
        mode_idx = mi + 1
        ax_b.plot(
            t_axis,
            mean_spectrum[:, mode_idx],
            color=mode_colors[mi],
            lw=2,
            label=mode_labels[mi],
            marker="o",
            markersize=3,
        )

    ax_b.set_xlabel("ODE time $t$", fontsize=9)
    ax_b.set_ylabel("Fraction of orbit variance", fontsize=9)
    ax_b.set_title("Spectral cascade", fontsize=10, fontweight="bold")
    ax_b.legend(fontsize=6, loc="upper left", framealpha=0.8)
    ax_b.set_xlim(0, 1)
    ax_b.set_ylim(bottom=0)
    ax_b.grid(True, alpha=0.2)
    ax_b.tick_params(labelsize=7)
    ax_b.text(
        0.95,
        0.05,
        f"layer: {scan.peak_layer}",
        transform=ax_b.transAxes,
        fontsize=6,
        ha="right",
        va="bottom",
        color="0.4",
    )

    fig.text(0.35, 0.97, "(B)", fontsize=10, fontweight="bold", va="top")

    # ----- Panel C: Symmetry-dependent peak n=1 -----
    ax_c = fig.add_subplot(gs[0, 2])

    for cls in range(n_classes):
        if np.isnan(t_star[cls]):
            continue
        grp = class_group(cls)
        # Marker size proportional to peak n=1 power
        ms = 4 + 12 * (peak_n1_power[cls] / np.nanmax(peak_n1_power))
        ax_c.plot(
            cls,
            t_star[cls],
            "o",
            color=group_colors[grp],
            markersize=ms,
            zorder=3,
        )
        # Annotate with peak power
        ax_c.annotate(
            f"{peak_n1_power[cls]:.2f}",
            (cls, t_star[cls]),
            textcoords="offset points",
            xytext=(0, -10),
            fontsize=5,
            ha="center",
            color="0.4",
        )

    # Group mean lines
    sym_vals = [t_star[c] for c in symmetric if not np.isnan(t_star[c])]
    if sym_vals:
        ax_c.axhline(
            np.nanmean(sym_vals),
            color=group_colors["symmetric"],
            ls="--",
            lw=1,
            alpha=0.5,
        )
    asym_vals = [t_star[c] for c in asymmetric if not np.isnan(t_star[c])]
    if asym_vals:
        ax_c.axhline(
            np.nanmean(asym_vals),
            color=group_colors["asymmetric"],
            ls="--",
            lw=1,
            alpha=0.5,
        )

    ax_c.set_xlabel("Digit class", fontsize=9)
    ax_c.set_ylabel("Peak $n{=}1$ time $t^*$", fontsize=9)
    ax_c.set_title("Symmetry-dependent peak", fontsize=10, fontweight="bold")
    ax_c.set_xticks(np.arange(n_classes))
    ax_c.set_xlim(-0.5, 9.5)
    ax_c.grid(True, alpha=0.2)
    ax_c.tick_params(labelsize=7)

    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=group_colors["symmetric"],
            lw=0,
            markersize=5,
            label="Symmetric (0, 8)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=group_colors["asymmetric"],
            lw=0,
            markersize=5,
            label="Asymmetric (2, 4, 6, 7)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=group_colors["other"],
            lw=0,
            markersize=5,
            label="Other",
        ),
    ]
    ax_c.legend(handles=legend_handles, fontsize=5.5, loc="upper right", framealpha=0.8)

    fig.text(0.73, 0.97, "(C)", fontsize=10, fontweight="bold", va="top")

    fig.savefig(OUT_DIR / "hero_figure.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "hero_figure.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved hero_figure.{png,pdf}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    utils.setup_torch()
    device = utils.get_torch_device()
    device_str = str(device)

    print(f"Device: {device}")
    model = load_checkpoint(args.experiment, args.checkpoint_dir, device_str)

    # Generate base trajectories
    n_gen = args.n_classes * args.orbit_samples_per_class
    cond = torch.arange(args.n_classes, device=device).repeat_interleave(
        args.orbit_samples_per_class
    )
    labels = cond.cpu().numpy()

    print(f"Generating {n_gen} base trajectories...")
    with torch.inference_mode():
        _, trajectory = generate_with_trajectory(model, cond)
    n_steps = len(trajectory)
    print(f"  {n_steps} ODE steps collected")

    selected = _select_orbit_indices(
        labels, args.n_classes, args.orbit_samples_per_class
    )

    run_all = args.run == "all"

    # ------------------------------------------------------------------
    # Scan (Figures 1, 2, 3, 5, 6)
    # ------------------------------------------------------------------
    scan: HolonomyScan | None = None
    if run_all or args.run == "deficit":
        print("\n=== Berry phase scan ===")
        scan = scan_holonomy(
            model,
            trajectory,
            cond,
            labels,
            args.n_classes,
            args.orbit_samples_per_class,
            args.orbit_angles,
        )
        peak_li = scan.layer_names.index(scan.peak_layer)
        print(
            f"  Peak: {scan.peak_layer}, step={scan.peak_step}, t={scan.t_peak:.2f}\n"
            f"    Tracking score = {scan.tracking_score[peak_li, scan.peak_step]:.3f}\n"
            f"    Winding consistency = {scan.winding_consistency[peak_li, scan.peak_step]:.0%}\n"
            f"    PCA quality = {scan.pca_quality[peak_li, scan.peak_step]:.2f}"
        )

        # Figure 1: Heatmap
        plot_holonomy_heatmap(scan, model)

        # Collect detailed per-class orbits at peak cell
        net = model.net
        in_context_start = net.in_context_start
        in_context_len = net.in_context_len
        t_values = torch.linspace(0.0, 1.0, model.config.num_sampling_steps + 1)
        t_scalar = float(t_values[min(scan.peak_step, len(t_values) - 1)].item())
        angle_values = np.linspace(0.0, 2 * np.pi, args.orbit_angles, endpoint=False)
        angle_torch = torch.tensor(angle_values, device=device, dtype=torch.float32)

        collector = BlockActivationCollector(net)
        phase_velocities: dict[int, np.ndarray] = {}
        pca_orbits: dict[int, np.ndarray] = {}
        pca_explained: dict[int, np.ndarray] = {}
        pt_result: ParallelTransportResult | None = None

        for base_idx in selected:
            cls = int(labels[base_idx])
            if cls in phase_velocities:
                continue  # one per class
            base = (
                trajectory[scan.peak_step][base_idx : base_idx + 1].to(device).float()
            )
            acts = _collect_single_orbit(
                net,
                base,
                t_scalar,
                cond[base_idx : base_idx + 1],
                angle_torch,
                collector,
                [scan.peak_layer],
                in_context_start,
                in_context_len,
            )
            h = acts[scan.peak_layer]
            m = compute_orbit_metrics(h, angle_values)
            phase_velocities[cls] = m.phase_velocity
            pca_orbits[cls] = m.pca_proj
            pca_explained[cls] = m.pca_explained

            if pt_result is None:
                pt_result = parallel_transport_holonomy(h)

        collector.remove()

        # Figure 2: Phase velocity
        plot_phase_velocity(
            phase_velocities, angle_values, scan.peak_layer, scan.t_peak
        )

        # Figure 3: PCA orbits
        plot_pca_orbits(
            pca_orbits, angle_values, pca_explained, scan.peak_layer, scan.t_peak
        )

        # Figure 5: Parallel transport
        if pt_result is not None:
            plot_frame_rotation(pt_result, scan.peak_layer, scan.t_peak)

        # Figure 6: Class comparison
        plot_class_comparison(scan, args.n_classes)

    # ------------------------------------------------------------------
    # Convergence study (within controls or standalone)
    # ------------------------------------------------------------------
    if run_all or args.run == "convergence":
        print("\n=== Convergence study ===")
        conv_values = [int(x) for x in args.convergence_values.split(",")]

        if scan is not None:
            conv_layer = scan.peak_layer
            conv_step = scan.peak_step
        else:
            layer_names = [
                "x_embedder",
                *[f"block_{i}" for i in range(len(model.net.blocks))],
            ]
            conv_layer = layer_names[len(layer_names) // 2]
            conv_step = n_steps // 2

        net = model.net
        in_context_start = net.in_context_start
        in_context_len = net.in_context_len
        t_values_t = torch.linspace(0.0, 1.0, model.config.num_sampling_steps + 1)
        t_scalar = float(t_values_t[min(conv_step, len(t_values_t) - 1)].item())

        tracking_by_n: dict[int, list[float]] = {n: [] for n in conv_values}

        for n_ang in conv_values:
            angle_vals = np.linspace(0.0, 2 * np.pi, n_ang, endpoint=False)
            angle_t = torch.tensor(angle_vals, device=device, dtype=torch.float32)

            collector = BlockActivationCollector(net)
            for base_idx in selected[:15]:
                base = trajectory[conv_step][base_idx : base_idx + 1].to(device).float()
                acts = _collect_single_orbit(
                    net,
                    base,
                    t_scalar,
                    cond[base_idx : base_idx + 1],
                    angle_t,
                    collector,
                    [conv_layer],
                    in_context_start,
                    in_context_len,
                )
                h = acts[conv_layer]
                m = compute_orbit_metrics(h, angle_vals)
                tracking_by_n[n_ang].append(m.tracking_score)
            collector.remove()
            print(
                f"  N={n_ang:3d}: tracking = "
                f"{np.mean(tracking_by_n[n_ang]):.3f} "
                f"± {np.std(tracking_by_n[n_ang]):.3f}"
            )

        # Plot convergence
        ns = sorted(tracking_by_n.keys())
        means = [float(np.mean(tracking_by_n[n])) for n in ns]
        stds = [float(np.std(tracking_by_n[n])) for n in ns]

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.errorbar(ns, means, yerr=stds, fmt="o-", capsize=4, lw=2, markersize=6)
        ax.set_xlabel("Orbit resolution N")
        ax.set_ylabel("Tracking score ρ")
        ax.set_title(f"Tracking convergence at {conv_layer}, step {conv_step}")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "fig3_convergence.png", dpi=300)
        plt.close(fig)
        print("  Saved fig3_convergence.png")

    # ------------------------------------------------------------------
    # Berry curvature field (Figure 4)
    # ------------------------------------------------------------------
    if run_all or args.run == "berry":
        print("\n=== Berry curvature field ===")
        if scan is not None:
            berry_layer = scan.peak_layer
        else:
            layer_names = [
                "x_embedder",
                *[f"block_{i}" for i in range(len(model.net.blocks))],
            ]
            berry_layer = layer_names[len(layer_names) // 2]

        berry_base = int(selected[0])
        field = compute_berry_curvature_field(
            model,
            trajectory,
            cond,
            berry_base,
            berry_layer,
            n_theta=args.berry_theta_grid,
        )
        print(f"  Total Berry curvature: {field.total_curvature:.4f}")
        plot_berry_curvature(field)

    # ------------------------------------------------------------------
    # Controls (Figure 7)
    # ------------------------------------------------------------------
    if run_all or args.run == "controls":
        print("\n=== Controls ===")
        if scan is not None:
            ctrl_layer = scan.peak_layer
            ctrl_step = scan.peak_step
            ctrl_t = scan.t_peak
        else:
            layer_names = [
                "x_embedder",
                *[f"block_{i}" for i in range(len(model.net.blocks))],
            ]
            ctrl_layer = layer_names[len(layer_names) // 2]
            ctrl_step = n_steps // 2
            ctrl_t = 0.5

        ctrl_results: dict[str, ControlResult] = {}
        for mode in ["rotation", "noise"]:
            result = _run_control_orbit(
                model,
                trajectory,
                cond,
                selected,
                labels,
                ctrl_step,
                args.orbit_angles,
                ctrl_layer,
                mode=mode,
            )
            ctrl_results[mode] = result
            print(
                f"  {mode:12s}: tracking={np.mean(result.tracking_scores):.3f} "
                f"± {np.std(result.tracking_scores):.3f}"
            )

        if args.untrained_control:
            result = _run_untrained_control(
                args.experiment,
                args.checkpoint_dir,
                trajectory,
                cond,
                selected,
                labels,
                ctrl_step,
                args.orbit_angles,
                ctrl_layer,
                device_str,
            )
            ctrl_results["untrained"] = result
            print(
                f"  {'untrained':12s}: tracking={np.mean(result.tracking_scores):.3f} "
                f"± {np.std(result.tracking_scores):.3f}"
            )

        plot_controls(ctrl_results, ctrl_layer, ctrl_t)

    # ------------------------------------------------------------------
    # Paper figures (combined composites)
    # ------------------------------------------------------------------
    if args.run == "paper":
        print("\n=== Paper figures ===")

        # --- Full scan for heatmap data ---
        print("  Running full scan...")
        scan = scan_holonomy(
            model,
            trajectory,
            cond,
            labels,
            args.n_classes,
            args.orbit_samples_per_class,
            args.orbit_angles,
        )
        peak_li = scan.layer_names.index(scan.peak_layer)
        print(
            f"  Peak (RSA): {scan.peak_layer}, step={scan.peak_step}, t={scan.t_peak:.2f}\n"
            f"    RSA = {scan.structure_score[peak_li, scan.peak_step]:.3f}\n"
            f"    Decode = {scan.decode_score[peak_li, scan.peak_step]:.3f}\n"
            f"    PCA track = {scan.tracking_score[peak_li, scan.peak_step]:.3f}"
        )

        # --- Hero figure (spectral cascade) ---
        print("  Plotting hero figure...")
        plot_hero_figure(
            scan, model, trajectory, cond, labels, selected, args.orbit_angles
        )

        # --- Paper Figure 1 ---
        print("  Plotting paper figure 1...")
        berry_base = int(selected[0])
        plot_paper_figure1(scan, model, trajectory, cond, berry_base, args.orbit_angles)

        # --- Collect per-class orbits ---
        # Use mid-ODE step for visualization (cleaner orbits where images
        # have structure), but show population scores from the RSA peak.
        orbit_step = n_steps // 2  # mid-trajectory
        orbit_layer = scan.peak_layer
        t_values_t = torch.linspace(0.0, 1.0, model.config.num_sampling_steps + 1)
        orbit_t = float(t_values_t[min(orbit_step, len(t_values_t) - 1)].item())
        print(
            f"  Collecting per-class orbits at {orbit_layer}, "
            f"step={orbit_step} (t={orbit_t:.2f})..."
        )

        net = model.net
        in_context_start = net.in_context_start
        in_context_len = net.in_context_len
        t_scalar = orbit_t
        angle_values = np.linspace(0.0, 2 * np.pi, args.orbit_angles, endpoint=False)
        angle_torch = torch.tensor(angle_values, device=device, dtype=torch.float32)

        collector = BlockActivationCollector(net)
        # Collect ALL orbits per class, keep the best (highest structure score)
        best_score: dict[int, float] = {}
        pca_orbits: dict[int, np.ndarray] = {}
        pca_explained_map: dict[int, np.ndarray] = {}
        # Accumulate spectra for population average
        spectra_accum: dict[int, list[np.ndarray]] = {
            c: [] for c in range(args.n_classes)
        }

        for base_idx in selected:
            cls = int(labels[base_idx])
            base = trajectory[orbit_step][base_idx : base_idx + 1].to(device).float()
            acts = _collect_single_orbit(
                net,
                base,
                t_scalar,
                cond[base_idx : base_idx + 1],
                angle_torch,
                collector,
                [orbit_layer],
                in_context_start,
                in_context_len,
            )
            h = acts[orbit_layer]
            m = compute_orbit_metrics(h, angle_values)
            spectra_accum[cls].append(m.spectrum)
            if cls not in best_score or m.structure_score > best_score[cls]:
                best_score[cls] = m.structure_score
                pca_orbits[cls] = m.pca_proj
                pca_explained_map[cls] = m.pca_explained

        collector.remove()

        # Average spectra per class
        spectra_by_class: dict[int, np.ndarray] = {}
        for cls, specs in spectra_accum.items():
            if specs:
                spectra_by_class[cls] = np.mean(specs, axis=0)

        # Per-class scores: use orbit step for visualization consistency
        orbit_li = scan.layer_names.index(orbit_layer)
        structure_at_orbit = scan.per_class_structure[:, orbit_li, orbit_step]
        decode_at_orbit = scan.per_class_decode[:, orbit_li, orbit_step]

        # --- Paper Figure 2 ---
        print("  Plotting paper figure 2...")
        plot_paper_figure2(
            pca_orbits,
            angle_values,
            pca_explained_map,
            structure_at_orbit,
            decode_at_orbit,
            trajectory,
            orbit_step,
            labels,
            selected,
            orbit_layer,
            orbit_t,
        )

        # --- Paper Figure 3: Fourier decomposition ---
        print("  Plotting paper figure 3 (Fourier decomposition)...")
        plot_paper_figure3(
            spectra_by_class,
            pca_orbits,
            angle_values,
            trajectory,
            orbit_step,
            labels,
            selected,
            orbit_layer,
            orbit_t,
        )

    print("\nDone. Figures saved to", OUT_DIR)


if __name__ == "__main__":
    main(parser.parse_args())
