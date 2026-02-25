"""Phase 1: Helmholtz-Hodge Decomposition of the Velocity Field.

Two complementary approaches:
1. Neural scalar potential: fits phi so -grad(phi) ~ v, giving a 2-way split
   (gradient + solenoidal residual). Smooth and continuous.
2. Graph-based Hodge: builds a k-NN simplicial complex and decomposes the
   edge-projected velocity into gradient + curl + harmonic via the discrete
   Hodge Laplacian. Gives the true 3-way orthogonal split needed to test
   whether class-conditioning injects harmonic/circulation modes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy import sparse
from scipy.sparse.linalg import lsqr
from sklearn.neighbors import NearestNeighbors
from torch import Tensor

from gf import utils
from gf.analysis.atlas import TrajectoryAtlas, load_atlas


# ---------------------------------------------------------------------------
# Neural scalar potential (existing approach — 2-way split)
# ---------------------------------------------------------------------------


class ScalarPotential(nn.Module):
    """MLP that maps (z, t, y_onehot) -> scalar phi."""

    def __init__(self, z_dim: int, n_classes: int, hidden_dim: int = 512):
        super().__init__()
        # Input: z_dim + 1 (time) + n_classes (one-hot)
        in_dim = z_dim + 1 + n_classes
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: Tensor, t: Tensor, y_onehot: Tensor) -> Tensor:
        inp = torch.cat([z, t, y_onehot], dim=-1)
        return self.net(inp)


def train_potential(
    atlas: TrajectoryAtlas,
    target: str = "v_cond",
    n_classes: int = 10,
    hidden_dim: int = 512,
    lr: float = 1e-3,
    n_epochs: int = 200,
    batch_size: int = 4096,
    device: str = "cpu",
) -> tuple[ScalarPotential, dict[str, list[float]]]:
    """Train a scalar potential phi so that -grad_z(phi) approximates target velocity."""
    N, T, D = atlas.v_cond.shape

    if target == "v_cond":
        v_target = atlas.v_cond
    elif target == "v_uncond":
        v_target = atlas.v_uncond
    elif target == "delta_v":
        v_target = atlas.v_cond - atlas.v_uncond
    else:
        raise ValueError(f"Unknown target: {target}")

    z_flat = atlas.z[:, :T].reshape(-1, D)
    t_flat = atlas.timesteps[:T].unsqueeze(0).expand(N, -1).reshape(-1, 1)
    labels_flat = atlas.labels.unsqueeze(1).expand(-1, T).reshape(-1)
    v_flat = v_target.reshape(-1, D)

    y_onehot = torch.zeros(labels_flat.shape[0], n_classes)
    y_onehot.scatter_(1, labels_flat.unsqueeze(1), 1.0)

    dataset = torch.utils.data.TensorDataset(z_flat, t_flat, y_onehot, v_flat)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    phi_net = ScalarPotential(D, n_classes, hidden_dim).to(device)
    optimizer = torch.optim.Adam(phi_net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    history: dict[str, list[float]] = {"loss": []}

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0

        for z_b, t_b, y_b, v_b in loader:
            z_b = z_b.to(device).requires_grad_(True)
            t_b = t_b.to(device)
            y_b = y_b.to(device)
            v_b = v_b.to(device)

            phi = phi_net(z_b, t_b, y_b)
            grad_phi = torch.autograd.grad(phi.sum(), z_b, create_graph=True)[0]
            loss = ((v_b + grad_phi) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / n_batches
        history["loss"].append(avg_loss)

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1}/{n_epochs}: loss={avg_loss:.6f}")

    return phi_net, history


@torch.no_grad()
def compute_decomposition(
    phi_net: ScalarPotential,
    atlas: TrajectoryAtlas,
    target: str = "v_cond",
    n_classes: int = 10,
    device: str = "cpu",
    batch_size: int = 4096,
) -> dict[str, Tensor]:
    """Compute the neural Helmholtz-Hodge decomposition metrics (2-way split)."""
    N, T, D = atlas.v_cond.shape

    if target == "v_cond":
        v_target = atlas.v_cond
    elif target == "v_uncond":
        v_target = atlas.v_uncond
    elif target == "delta_v":
        v_target = atlas.v_cond - atlas.v_uncond
    else:
        raise ValueError(f"Unknown target: {target}")

    v_grad_all = torch.zeros(N, T, D)
    energy_all = torch.zeros(N, T)

    phi_net.eval()

    z_flat = atlas.z[:, :T].reshape(-1, D)
    t_flat = atlas.timesteps[:T].unsqueeze(0).expand(N, -1).reshape(-1, 1)
    labels_flat = atlas.labels.unsqueeze(1).expand(-1, T).reshape(-1)
    y_onehot = torch.zeros(labels_flat.shape[0], n_classes)
    y_onehot.scatter_(1, labels_flat.unsqueeze(1), 1.0)

    total = z_flat.shape[0]
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        z_b = z_flat[start:end].to(device).requires_grad_(True)
        t_b = t_flat[start:end].to(device)
        y_b = y_onehot[start:end].to(device)

        with torch.enable_grad():
            phi = phi_net(z_b, t_b, y_b)
            grad_phi = torch.autograd.grad(phi.sum(), z_b)[0]

        v_grad_all.view(-1, D)[start:end] = -grad_phi.cpu()
        energy_all.view(-1)[start:end] = phi.squeeze(-1).cpu()

    v_sol_all = v_target - v_grad_all

    v_norm_sq = (v_target**2).sum(dim=-1)
    sol_norm_sq = (v_sol_all**2).sum(dim=-1)

    r_squared = 1.0 - sol_norm_sq.sum(dim=0) / v_norm_sq.sum(dim=0).clamp_min(1e-10)

    r_squared_per_class = torch.zeros(n_classes, T)
    for c in range(n_classes):
        mask = atlas.labels == c
        v_c = v_norm_sq[mask]
        s_c = sol_norm_sq[mask]
        r_squared_per_class[c] = 1.0 - s_c.sum(dim=0) / v_c.sum(dim=0).clamp_min(1e-10)

    return {
        "r_squared": r_squared,
        "r_squared_per_class": r_squared_per_class,
        "energy_along_traj": energy_all,
        "v_grad": v_grad_all,
        "v_sol": v_sol_all,
    }


# ---------------------------------------------------------------------------
# Graph-based Hodge decomposition (3-way split: gradient/curl/harmonic)
# ---------------------------------------------------------------------------


def _build_simplicial_complex(
    z: np.ndarray, k: int = 15
) -> tuple[
    list[tuple[int, int]],
    dict[tuple[int, int], int],
    list[tuple[int, int, int]],
]:
    """Build k-NN graph and extract edges and triangles.

    Args:
        z: (n, D) point cloud positions.
        k: Number of nearest neighbors.

    Returns:
        edges: List of oriented edges (i, j) with i < j.
        edge_to_idx: Mapping from edge tuple to index.
        triangles: List of oriented triangles (i, j, k) with i < j < k.
    """
    n = len(z)
    k_actual = min(k, n - 1)

    nn_model = NearestNeighbors(n_neighbors=k_actual, algorithm="auto")
    nn_model.fit(z)
    knn_graph = nn_model.kneighbors_graph(mode="connectivity")

    # Symmetrize to undirected
    adj = (knn_graph + knn_graph.T) > 0
    rows, cols = adj.nonzero()

    # Build adjacency sets and oriented edge list
    adj_sets: list[set[int]] = [set() for _ in range(n)]
    edges: list[tuple[int, int]] = []
    edge_set: set[tuple[int, int]] = set()

    for i, j in zip(rows, cols):
        if i < j and (i, j) not in edge_set:
            edges.append((i, j))
            edge_set.add((i, j))
        adj_sets[i].add(j)

    edge_to_idx = {e: idx for idx, e in enumerate(edges)}

    # Find triangles: for each edge (i,j), find common neighbors k > j
    tri_set: set[tuple[int, int, int]] = set()
    for i, j in edges:
        common = adj_sets[i] & adj_sets[j]
        for k_node in common:
            tri = tuple(sorted([i, j, k_node]))
            tri_set.add(tri)  # type: ignore[arg-type]

    triangles = sorted(tri_set)
    return edges, edge_to_idx, triangles


def _build_incidence_matrices(
    n_vertices: int,
    edges: list[tuple[int, int]],
    edge_to_idx: dict[tuple[int, int], int],
    triangles: list[tuple[int, int, int]],
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Build the gradient (G) and curl (C) incidence matrices.

    G ∈ R^{m x n}: gradient operator, maps vertex potentials to edge flows.
        For edge e = (i->j): G[e, i] = -1, G[e, j] = +1

    C ∈ R^{p x m}: curl operator, maps edge flows to face circulations.
        For face f = (i,j,k) with boundary i->j->k->i:
        C[f, e(i,j)] = +1, C[f, e(j,k)] = +1, C[f, e(i,k)] = -1
    """
    m = len(edges)
    p = len(triangles)

    # Gradient operator G (m x n)
    row_g, col_g, data_g = [], [], []
    for idx, (i, j) in enumerate(edges):
        row_g.extend([idx, idx])
        col_g.extend([i, j])
        data_g.extend([-1.0, 1.0])
    G = sparse.csr_matrix(
        (data_g, (row_g, col_g)), shape=(m, n_vertices), dtype=np.float64
    )

    # Curl operator C (p x m)
    if p > 0:
        row_c, col_c, data_c = [], [], []
        for fidx, (a, b, c_node) in enumerate(triangles):
            # Boundary of (a, b, c): edge(a,b) + edge(b,c) - edge(a,c)
            e_ab = edge_to_idx.get((a, b))
            e_bc = edge_to_idx.get((b, c_node))
            e_ac = edge_to_idx.get((a, c_node))
            if e_ab is not None and e_bc is not None and e_ac is not None:
                row_c.extend([fidx, fidx, fidx])
                col_c.extend([e_ab, e_bc, e_ac])
                data_c.extend([1.0, 1.0, -1.0])
        C = sparse.csr_matrix((data_c, (row_c, col_c)), shape=(p, m), dtype=np.float64)
    else:
        C = sparse.csr_matrix((p, m), dtype=np.float64)

    return G, C


def _project_velocity_to_edges(
    z: np.ndarray,
    v: np.ndarray,
    edges: list[tuple[int, int]],
) -> np.ndarray:
    """Project vertex velocities onto edge signals.

    For each edge e = (i->j), the edge signal is the average velocity
    projected onto the unit edge direction:
        s[e] = ((v_i + v_j) / 2) . (z_j - z_i) / ||z_j - z_i||
    """
    m = len(edges)
    s = np.zeros(m, dtype=np.float64)
    for idx, (i, j) in enumerate(edges):
        edge_vec = z[j] - z[i]
        edge_len = np.linalg.norm(edge_vec)
        if edge_len > 1e-10:
            edge_dir = edge_vec / edge_len
            v_avg = (v[i] + v[j]) / 2.0
            s[idx] = np.dot(v_avg, edge_dir)
    return s


def graph_hodge_decomposition(
    z: np.ndarray,
    v: np.ndarray,
    k: int = 15,
) -> dict[str, float]:
    """Compute the 3-way Hodge decomposition on a k-NN simplicial complex.

    Decomposes the edge-projected velocity field into three orthogonal components:
        s = s_grad + s_curl + s_harm

    where:
        s_grad in im(G^T): gradient (exact) — conservative flow
        s_curl in im(C^T): curl (coexact) — rotational flow
        s_harm in ker(L_1): harmonic — nontrivial cohomology

    Args:
        z: (n, D) point cloud positions.
        v: (n, D) velocity vectors at each point.
        k: Number of nearest neighbors for graph construction.

    Returns:
        Dictionary with energy fractions and diagnostics.
    """
    n = len(z)
    if n < 4:
        return {
            "grad_frac": 0.0,
            "curl_frac": 0.0,
            "harm_frac": 0.0,
            "n_edges": 0,
            "n_triangles": 0,
        }

    edges, edge_to_idx, triangles = _build_simplicial_complex(z, k=k)
    m = len(edges)
    p = len(triangles)

    if m == 0:
        return {
            "grad_frac": 0.0,
            "curl_frac": 0.0,
            "harm_frac": 0.0,
            "n_edges": 0,
            "n_triangles": p,
        }

    G, C = _build_incidence_matrices(n, edges, edge_to_idx, triangles)
    s = _project_velocity_to_edges(z, v, edges)

    s_norm_sq = np.dot(s, s)
    if s_norm_sq < 1e-12:
        return {
            "grad_frac": 0.0,
            "curl_frac": 0.0,
            "harm_frac": 0.0,
            "n_edges": m,
            "n_triangles": p,
        }

    # Gradient component: s_grad = G^T f where G^T G f = G^T s
    # G^T G is the graph Laplacian (n x n), singular with nullspace = constants.
    # Use least-squares via lsqr (handles rank deficiency).
    GtG = G.T @ G
    Gts = G.T @ s
    f_sol, *_ = lsqr(GtG, Gts)
    s_grad = G @ f_sol

    # Curl component: s_curl = C^T g where C C^T g = C s
    if p > 0:
        CCt = C @ C.T
        Cs = C @ s
        g_sol, *_ = lsqr(CCt, Cs)
        s_curl = C.T @ g_sol
    else:
        s_curl = np.zeros(m, dtype=np.float64)

    # Harmonic component: orthogonal residual
    s_harm = s - s_grad - s_curl

    grad_frac = float(np.dot(s_grad, s_grad) / s_norm_sq)
    curl_frac = float(np.dot(s_curl, s_curl) / s_norm_sq)
    harm_frac = float(np.dot(s_harm, s_harm) / s_norm_sq)

    return {
        "grad_frac": grad_frac,
        "curl_frac": curl_frac,
        "harm_frac": harm_frac,
        "n_edges": m,
        "n_triangles": p,
    }


def compute_graph_hodge_timeseries(
    atlas: TrajectoryAtlas,
    target: str = "v_cond",
    n_classes: int = 10,
    k: int = 15,
    step_stride: int = 1,
) -> dict[str, Tensor]:
    """Compute graph Hodge 3-way decomposition per class per timestep.

    Returns:
        grad_frac: (n_classes, n_steps) gradient energy fraction
        curl_frac: (n_classes, n_steps) curl energy fraction
        harm_frac: (n_classes, n_steps) harmonic energy fraction
        analyzed_times: (n_steps,) time values
    """
    N, T, D = atlas.v_cond.shape

    if target == "v_cond":
        v_all = atlas.v_cond
    elif target == "v_uncond":
        v_all = atlas.v_uncond
    elif target == "delta_v":
        v_all = atlas.v_cond - atlas.v_uncond
    else:
        raise ValueError(f"Unknown target: {target}")

    step_indices = list(range(0, T, step_stride))
    n_steps = len(step_indices)

    grad_frac = torch.zeros(n_classes, n_steps)
    curl_frac = torch.zeros(n_classes, n_steps)
    harm_frac = torch.zeros(n_classes, n_steps)

    for c in range(n_classes):
        mask = atlas.labels == c
        z_class = atlas.z[mask]  # (n_per_class, T+1, D)
        v_class = v_all[mask]  # (n_per_class, T, D)

        for si, step in enumerate(step_indices):
            z_np = z_class[:, step].numpy()
            v_np = v_class[:, step].numpy()

            result = graph_hodge_decomposition(z_np, v_np, k=k)
            grad_frac[c, si] = result["grad_frac"]
            curl_frac[c, si] = result["curl_frac"]
            harm_frac[c, si] = result["harm_frac"]

        print(f"  Class {c}: graph Hodge computed ({target})")

    return {
        "grad_frac": grad_frac,
        "curl_frac": curl_frac,
        "harm_frac": harm_frac,
        "analyzed_times": atlas.timesteps[step_indices],
    }


# ---------------------------------------------------------------------------
# Divergence diagnostic for neural decomposition
# ---------------------------------------------------------------------------


def compute_divergence_diagnostic(
    atlas: TrajectoryAtlas,
    v_sol: Tensor,
    n_classes: int = 10,
    k: int = 15,
    step_stride: int = 5,
) -> dict[str, Tensor]:
    """Validate the neural decomposition by checking if v_sol is divergence-free.

    Projects v_sol onto a k-NN graph and computes the discrete divergence
    (G^T s_sol). For a truly solenoidal field, this should be near zero.

    Returns:
        div_ratio: (n_classes, n_steps) ratio ||div(v_sol)|| / ||div(v)||
        analyzed_times: (n_steps,) time values
    """
    N, T, D = atlas.v_cond.shape
    step_indices = list(range(0, T, step_stride))
    n_steps = len(step_indices)

    div_ratio = torch.zeros(n_classes, n_steps)

    for c in range(n_classes):
        mask = atlas.labels == c
        z_class = atlas.z[mask]
        v_sol_class = v_sol[mask]
        v_cond_class = atlas.v_cond[mask]

        for si, step in enumerate(step_indices):
            z_np = z_class[:, step].numpy()
            v_sol_np = v_sol_class[:, step].numpy()
            v_cond_np = v_cond_class[:, step].numpy()

            n = len(z_np)
            k_actual = min(k, n - 1)
            if k_actual < 2:
                continue

            edges, edge_to_idx, _ = _build_simplicial_complex(z_np, k=k_actual)
            if len(edges) == 0:
                continue

            n_verts = len(z_np)
            m = len(edges)
            row_g, col_g, data_g = [], [], []
            for idx, (i, j) in enumerate(edges):
                row_g.extend([idx, idx])
                col_g.extend([i, j])
                data_g.extend([-1.0, 1.0])
            G = sparse.csr_matrix(
                (data_g, (row_g, col_g)), shape=(m, n_verts), dtype=np.float64
            )

            s_sol = _project_velocity_to_edges(z_np, v_sol_np, edges)
            s_full = _project_velocity_to_edges(z_np, v_cond_np, edges)

            div_sol = G.T @ s_sol
            div_full = G.T @ s_full

            div_full_norm = np.linalg.norm(div_full)
            if div_full_norm > 1e-10:
                div_ratio[c, si] = float(np.linalg.norm(div_sol) / div_full_norm)

    return {
        "div_ratio": div_ratio,
        "analyzed_times": atlas.timesteps[step_indices],
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_hodge_results(
    results_cond: dict[str, Tensor],
    results_delta: dict[str, Tensor],
    atlas: TrajectoryAtlas,
    out_dir: Path,
    graph_hodge_cond: dict[str, Tensor] | None = None,
    graph_hodge_delta: dict[str, Tensor] | None = None,
) -> None:
    """Plot Helmholtz-Hodge decomposition results."""
    timesteps = atlas.timesteps[:-1].numpy()
    n_classes = 10
    cmap = plt.get_cmap("tab10")

    has_graph = graph_hodge_cond is not None
    n_rows = 3 if has_graph else 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 5 * n_rows))

    # Fig 2a: R^2(t) for each class — conditional velocity
    ax = axes[0, 0]
    for c in range(n_classes):
        ax.plot(
            timesteps,
            results_cond["r_squared_per_class"][c].numpy(),
            color=cmap(c / 10),
            alpha=0.7,
            label=str(c),
        )
    ax.plot(
        timesteps,
        results_cond["r_squared"].numpy(),
        "k--",
        linewidth=2,
        label="mean",
    )
    ax.set_xlabel("Time $t$")
    ax.set_ylabel("$R^2(t)$")
    ax.set_title("(a) Neural gradient fraction $R^2$ — $v_{\\mathrm{cond}}$")
    ax.legend(fontsize=7, ncol=3)
    ax.set_ylim(-0.1, 1.1)

    # Fig 2b: R^2(t) for delta_v
    ax = axes[0, 1]
    for c in range(n_classes):
        ax.plot(
            timesteps,
            results_delta["r_squared_per_class"][c].numpy(),
            color=cmap(c / 10),
            alpha=0.7,
            label=str(c),
        )
    ax.plot(
        timesteps,
        results_delta["r_squared"].numpy(),
        "k--",
        linewidth=2,
        label="mean",
    )
    ax.set_xlabel("Time $t$")
    ax.set_ylabel("$R^2(t)$")
    ax.set_title("(b) Neural gradient fraction $R^2$ — $\\Delta v$")
    ax.legend(fontsize=7, ncol=3)
    ax.set_ylim(-0.1, 1.1)

    # Fig 2c: Energy along trajectories
    ax = axes[1, 0]
    energy = results_cond["energy_along_traj"]
    for c in range(n_classes):
        mask = atlas.labels == c
        e_class = energy[mask].mean(dim=0).numpy()
        ax.plot(timesteps, e_class, color=cmap(c / 10), label=str(c))
    ax.set_xlabel("Time $t$")
    ax.set_ylabel("Scalar potential $\\phi(z(t), t)$")
    ax.set_title("(c) Potential along trajectories")
    ax.legend(fontsize=7, ncol=3)

    # Fig 2d: PCA quiver
    ax = axes[1, 1]
    from sklearn.decomposition import PCA

    mid_step = len(timesteps) // 2
    z_mid = atlas.z[:, mid_step].numpy()
    v_grad_mid = results_cond["v_grad"][:, mid_step].numpy()
    v_sol_mid = results_cond["v_sol"][:, mid_step].numpy()

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
        color="blue",
        alpha=0.5,
        scale=None,
        label="$v_{\\mathrm{grad}}$",
    )
    ax.quiver(
        z_2d[idx, 0],
        z_2d[idx, 1],
        v_sol_2d[idx, 0],
        v_sol_2d[idx, 1],
        color="red",
        alpha=0.5,
        scale=None,
        label="$v_{\\mathrm{sol}}$",
    )
    ax.legend()
    ax.set_title(f"(d) PCA quiver at $t={timesteps[mid_step]:.2f}$")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")

    # Row 3: Graph Hodge 3-way decomposition (if available)
    if has_graph and graph_hodge_cond is not None and graph_hodge_delta is not None:
        gh_times_cond = graph_hodge_cond["analyzed_times"].numpy()
        gh_times_delta = graph_hodge_delta["analyzed_times"].numpy()

        # Fig 2e: Stacked fractions for v_cond (class-averaged)
        ax = axes[2, 0]
        g_mean = graph_hodge_cond["grad_frac"].mean(dim=0).numpy()
        c_mean = graph_hodge_cond["curl_frac"].mean(dim=0).numpy()
        h_mean = graph_hodge_cond["harm_frac"].mean(dim=0).numpy()
        ax.stackplot(
            gh_times_cond,
            g_mean,
            c_mean,
            h_mean,
            labels=["Gradient", "Curl", "Harmonic"],
            colors=["steelblue", "indianred", "goldenrod"],
            alpha=0.8,
        )
        ax.set_xlabel("Time $t$")
        ax.set_ylabel("Energy fraction")
        ax.set_title("(e) Graph Hodge 3-way — $v_{\\mathrm{cond}}$")
        ax.legend(fontsize=8, loc="upper right")
        ax.set_ylim(0, 1.05)

        # Fig 2f: Stacked fractions for delta_v (class-averaged)
        ax = axes[2, 1]
        g_mean = graph_hodge_delta["grad_frac"].mean(dim=0).numpy()
        c_mean = graph_hodge_delta["curl_frac"].mean(dim=0).numpy()
        h_mean = graph_hodge_delta["harm_frac"].mean(dim=0).numpy()
        ax.stackplot(
            gh_times_delta,
            g_mean,
            c_mean,
            h_mean,
            labels=["Gradient", "Curl", "Harmonic"],
            colors=["steelblue", "indianred", "goldenrod"],
            alpha=0.8,
        )
        ax.set_xlabel("Time $t$")
        ax.set_ylabel("Energy fraction")
        ax.set_title("(f) Graph Hodge 3-way — $\\Delta v$")
        ax.legend(fontsize=8, loc="upper right")
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    out_path = out_dir / "hodge_decomposition.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Hodge decomposition figure to {out_path}")


def main(args: argparse.Namespace) -> None:
    utils.setup_torch()
    device = utils.get_torch_device().type

    print("Loading atlas...")
    atlas = load_atlas(args.experiment, args.checkpoint_dir)
    print(f"  z: {atlas.z.shape}, v_cond: {atlas.v_cond.shape}")

    out_dir = Path(args.checkpoint_dir) / args.experiment / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Neural potential decomposition (2-way) ---
    print("\nTraining scalar potential for v_cond...")
    phi_cond, hist_cond = train_potential(
        atlas,
        target="v_cond",
        device=device,
        n_epochs=args.n_epochs,
        hidden_dim=args.hidden_dim,
    )
    print("Computing decomposition for v_cond...")
    results_cond = compute_decomposition(
        phi_cond, atlas, target="v_cond", device=device
    )

    print("\nTraining scalar potential for delta_v...")
    phi_delta, hist_delta = train_potential(
        atlas,
        target="delta_v",
        device=device,
        n_epochs=args.n_epochs,
        hidden_dim=args.hidden_dim,
    )
    print("Computing decomposition for delta_v...")
    results_delta = compute_decomposition(
        phi_delta, atlas, target="delta_v", device=device
    )

    # Divergence diagnostic for neural v_sol
    print("\nComputing divergence diagnostic...")
    div_diag = compute_divergence_diagnostic(atlas, results_cond["v_sol"])
    mean_div = div_diag["div_ratio"].mean().item()
    print(f"  Mean ||div(v_sol)||/||div(v)|| = {mean_div:.4f}")
    print("  (closer to 0 = better solenoidal quality)")

    # --- Graph Hodge decomposition (3-way) ---
    print("\nComputing graph Hodge decomposition for v_cond...")
    gh_cond = compute_graph_hodge_timeseries(
        atlas, target="v_cond", k=args.knn_k, step_stride=args.step_stride
    )

    print("\nComputing graph Hodge decomposition for delta_v...")
    gh_delta = compute_graph_hodge_timeseries(
        atlas, target="delta_v", k=args.knn_k, step_stride=args.step_stride
    )

    # Summary
    print("\n--- Graph Hodge Summary (class-averaged) ---")
    for name, gh in [("v_cond", gh_cond), ("delta_v", gh_delta)]:
        g = gh["grad_frac"].mean().item()
        c = gh["curl_frac"].mean().item()
        h = gh["harm_frac"].mean().item()
        print(f"  {name}: grad={g:.3f}, curl={c:.3f}, harm={h:.3f}")

    # Save results
    torch.save(
        {
            "results_cond": results_cond,
            "results_delta": results_delta,
            "history_cond": hist_cond,
            "history_delta": hist_delta,
            "graph_hodge_cond": gh_cond,
            "graph_hodge_delta": gh_delta,
            "divergence_diagnostic": div_diag,
        },
        out_dir / "hodge_results.pt",
    )
    print(f"Saved results to {out_dir / 'hodge_results.pt'}")

    # Plot
    plot_hodge_results(
        results_cond,
        results_delta,
        atlas,
        out_dir,
        graph_hodge_cond=gh_cond,
        graph_hodge_delta=gh_delta,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="default")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--n-epochs", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--knn-k", type=int, default=15)
    parser.add_argument("--step-stride", type=int, default=2)
    args = parser.parse_args()
    main(args)
