"""Activation probing for flow-matching denoiser internals.

Collects per-layer, per-step activations across denoising trajectories
and analyzes them with PCA to reveal internal representation structure.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from geoflow import utils
from geoflow.denoiser import Denoiser, DenoiserConfig

parser = argparse.ArgumentParser(description="Probe denoiser activations with PCA")
parser.add_argument("--experiment", type=str, default="default")
parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
parser.add_argument("--n-samples", type=int, default=100, help="Samples per class")
parser.add_argument("--n-classes", type=int, default=10)
parser.add_argument("--n-components", type=int, default=3, help="PCA components")
parser.add_argument(
    "--cfg-scale",
    type=float,
    default=None,
    help="CFG scale (default: use checkpoint value)",
)
parser.add_argument("--skip-pca", action="store_true", help="Only save raw activations")


def load_checkpoint(
    experiment: str,
    checkpoint_dir: str,
    device: str,
    cfg_scale: float | None = None,
) -> Denoiser:
    path = Path(checkpoint_dir) / experiment / "last.pt"
    ckpt = torch.load(path, map_location=device)

    config = DenoiserConfig(**ckpt["config"])
    config.sampling_method = "euler"
    config.noise_scale = 0.8
    if cfg_scale is not None:
        config.cfg_scale = cfg_scale

    model = Denoiser(config, device).to(device)

    decay = sorted(model.ema.keys())[0]
    model.ema[decay].load_state_dict(ckpt["ema"][decay])
    model.swap_ema(decay=decay)

    model.eval()
    print(f"Loaded checkpoint from {path} (step={ckpt.get('global_step', '?')})")
    print(
        f"  cfg_scale={config.cfg_scale}, sampling_method=euler, steps={config.num_sampling_steps}"
    )
    return model


# ---------------------------------------------------------------------------
# Hook collection
# ---------------------------------------------------------------------------


class ActivationCollector:
    """Registers forward hooks and accumulates per-step activations.

    When ``cfg_mode=True``, hooks fire twice per step (conditional then
    unconditional). The collector separates them via a per-layer counter
    and ``gather_cfg`` applies the CFG formula to the activations.
    """

    def __init__(self, model: Denoiser, cfg_mode: bool = False):
        self.model = model
        self.cfg_mode = cfg_mode
        self.hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._buffers: dict[str, list[torch.Tensor]] = {}
        if cfg_mode:
            self._buffers_uncond: dict[str, list[torch.Tensor]] = {}
            self._counters: dict[str, int] = {}
        self._register(model)

    def _register(self, model: Denoiser) -> None:
        net = model.net
        self._add_hook("input_embed", net.input_embed)
        for i, block in enumerate(net.blocks):
            self._add_hook(f"block_{i}", block)

    def _add_hook(self, name: str, module: torch.nn.Module) -> None:
        self._buffers[name] = []
        if self.cfg_mode:
            self._buffers_uncond[name] = []
            self._counters[name] = 0

        def hook_fn(
            _mod: torch.nn.Module,
            _inp: object,
            output: torch.Tensor,
            _name: str = name,
        ) -> None:
            tensor = output.detach().float().cpu()
            if not self.cfg_mode:
                self._buffers[_name].append(tensor)
            else:
                if self._counters[_name] % 2 == 0:
                    self._buffers[_name].append(tensor)  # conditional
                else:
                    self._buffers_uncond[_name].append(tensor)  # unconditional
                self._counters[_name] += 1

        self.hooks.append(module.register_forward_hook(hook_fn))

    @property
    def layer_names(self) -> list[str]:
        return list(self._buffers.keys())

    def _stack(self, buffers: dict[str, list[torch.Tensor]]) -> dict[str, np.ndarray]:
        """Stack per-step tensors → (N, num_steps, hidden_dim) numpy arrays."""
        return {
            name: torch.stack(tensors, dim=1).numpy()
            for name, tensors in buffers.items()
        }

    def gather(self) -> dict[str, np.ndarray]:
        """Return collected activations as (N, num_steps, hidden_dim) arrays."""
        return self._stack(self._buffers)

    def gather_cfg(self, cfg_scale: float) -> dict[str, np.ndarray]:
        """Apply CFG formula to activations: uncond + scale * (cond - uncond)."""
        cond = self._stack(self._buffers)
        uncond = self._stack(self._buffers_uncond)
        return {
            name: uncond[name] + cfg_scale * (cond[name] - uncond[name])
            for name in cond
        }

    def remove(self) -> None:
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


# ---------------------------------------------------------------------------
# PCA (numpy SVD, no sklearn)
# ---------------------------------------------------------------------------


def pca(X: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PCA via SVD. X: (N, D). Returns projected, components, explained_variance_ratio."""
    mean = X.mean(axis=0)
    Xc = X - mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    components = Vt[:n_components]
    projected = Xc @ components.T
    var = (S**2) / (X.shape[0] - 1)
    explained = var[:n_components] / var.sum()
    return projected, components, explained


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_single_class_pca(
    acts: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
    n_components: int,
    out_dir: Path,
) -> None:
    """Per-class PCA scatter (PC1 vs PC2), colored by angular position in PC space.

    If orientation is encoded in the leading PCs, points form a ring/arc
    with smooth color gradients under the circular hsv colormap.
    """
    rows, cols = 2, 5
    fig, axes = plt.subplots(
        rows, cols, figsize=(3.5 * cols, 3.5 * rows), squeeze=False
    )

    for cls in range(n_classes):
        ax = axes[cls // cols][cls % cols]
        mask = labels == cls
        proj, _, explained = pca(acts[mask], n_components)
        theta = np.arctan2(proj[:, 1], proj[:, 0])  # [-pi, pi]
        sc = ax.scatter(
            proj[:, 0],
            proj[:, 1],
            c=theta,
            cmap="hsv",
            s=12,
            alpha=0.8,
            vmin=-np.pi,
            vmax=np.pi,
        )
        ax.set_title(f"class {cls}", fontsize=9)
        ax.set_xlabel(f"PC1 ({explained[0]:.0%})", fontsize=7)
        ax.set_ylabel(f"PC2 ({explained[1]:.0%})", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        "Single-class PCA — block_0, final step (colored by angle)", fontsize=11
    )
    fig.colorbar(sc, ax=axes, label="angle (rad)", shrink=0.6)
    fig.subplots_adjust(
        left=0.05, right=0.88, top=0.92, bottom=0.05, wspace=0.3, hspace=0.35
    )
    fig.savefig(out_dir / "single_class_pca.png", dpi=300)
    plt.close(fig)


def plot_image_strips(
    images: np.ndarray,
    acts: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
    n_components: int,
    out_dir: Path,
    n_show: int = 20,
) -> None:
    """For each class, sort generated images by PC-angle and display as a filmstrip.

    If orientation is encoded, each row shows a smooth rotation sweep.
    """
    fig, axes = plt.subplots(
        n_classes, n_show, figsize=(n_show * 0.4, n_classes * 0.415)
    )

    for cls in range(n_classes):
        mask = labels == cls
        cls_acts = acts[mask]
        cls_imgs = images[mask]  # (n_samples, 1, 28, 28)

        proj, _, _ = pca(cls_acts, n_components)
        theta = np.arctan2(proj[:, 1], proj[:, 0])
        order = np.argsort(theta)

        # Evenly-spaced samples from the sorted order
        indices = np.linspace(0, len(order) - 1, n_show, dtype=int)
        selected = order[indices]

        for j, idx in enumerate(selected):
            ax = axes[cls][j]
            ax.imshow(cls_imgs[idx, 0], cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(str(cls), fontsize=9, rotation=0, labelpad=8)
                ax.yaxis.set_visible(True)

    fig.suptitle("Images sorted by PC angle — block_0, final step", fontsize=10, y=0.99)
    fig.subplots_adjust(
        wspace=0.02, hspace=0.02, left=0.03, right=1, top=0.95, bottom=0.01
    )
    fig.savefig(out_dir / "image_strips.png", dpi=300)
    plt.close(fig)


def plot_explained_variance(
    all_acts: dict[str, np.ndarray],
    n_components: int,
    out_dir: Path,
) -> None:
    """Line plot: top-k explained variance ratio per layer across denoising steps."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for layer_name, acts in all_acts.items():
        n_steps = acts.shape[1]
        variances = np.zeros((n_steps, n_components))
        for step in range(n_steps):
            _, _, explained = pca(acts[:, step, :], n_components)
            variances[step] = explained

        for k in range(n_components):
            ax.plot(
                range(n_steps),
                variances[:, k],
                marker="o",
                markersize=3,
                label=f"{layer_name} PC{k + 1}",
            )

    ax.set_xlabel("Denoising step")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title("Explained variance across denoising steps")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "explained_variance.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    utils.setup_torch()
    device = utils.get_torch_device().type

    model = load_checkpoint(
        args.experiment, args.checkpoint_dir, device, args.cfg_scale
    )
    cfg_scale = model.config.cfg_scale
    cfg_mode = cfg_scale != 1.0
    num_steps = model.config.num_sampling_steps

    # Build conditioning: each class repeated n_samples times
    cond = torch.arange(args.n_classes, device=device).repeat_interleave(args.n_samples)
    labels = cond.cpu().numpy()
    total = cond.shape[0]
    print(
        f"Generating {total} samples ({args.n_classes} classes x {args.n_samples} each)"
    )

    # Collect activations
    collector = ActivationCollector(model, cfg_mode=cfg_mode)

    with torch.inference_mode(), utils.maybe_autocast(device):
        samples = model.generate(cond)

    # Convert to images: (N, 1, 28, 28) in [0, 1]
    images = samples.float().cpu().view(-1, 1, 28, 28).clamp(-1, 1) * 0.5 + 0.5
    images = images.numpy()

    if cfg_mode:
        activations = collector.gather_cfg(cfg_scale)
    else:
        activations = collector.gather()
    collector.remove()

    # Output directory
    out_dir = Path(args.checkpoint_dir) / args.experiment / "activations"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # Save raw activations
    np.save(out_dir / "labels.npy", labels)
    for name, acts in activations.items():
        np.save(out_dir / f"activations_{name}.npy", acts)
        print(f"  {name}: {acts.shape}")

    # Save metadata
    metadata = {
        "experiment": args.experiment,
        "n_samples": args.n_samples,
        "n_classes": args.n_classes,
        "num_steps": num_steps,
        "total_samples": total,
        "layers": collector.layer_names,
        "hidden_dim": int(activations[collector.layer_names[0]].shape[2]),
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved activations to {out_dir}")

    if args.skip_pca:
        print("Skipping PCA (--skip-pca)")
        return

    # PCA analysis
    print("Running PCA analysis...")

    # Explained variance across layers and steps
    plot_explained_variance(activations, args.n_components, fig_dir)

    # Within-class orientation analysis on block_0, final denoising step
    b_act = activations["block_0"][:, -1, :]  # (N, hidden_dim)

    plot_single_class_pca(b_act, labels, args.n_classes, args.n_components, fig_dir)
    plot_image_strips(images, b_act, labels, args.n_classes, args.n_components, fig_dir)

    print(f"Saved figures to {fig_dir}")
    print("Done.")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
