import argparse
import csv
import itertools
import warnings
from pathlib import Path

import numpy as np
import torch

from geoflow import utils
from geoflow.datasets import setup_dataloaders
from geoflow.fid import compute_fid_is
from scripts.generate import load_checkpoint

warnings.filterwarnings(
    "ignore",
    message=".*dtype.*align.*",
    category=np.exceptions.VisibleDeprecationWarning,
)

parser = argparse.ArgumentParser(
    description="Sweep cfg_scale and noise_scale for FID/IS"
)
parser.add_argument("--experiment", type=str, default="default")
parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
parser.add_argument(
    "--dataset", type=str, default="cifar10", choices=["mnist", "cifar10"]
)
parser.add_argument("--fid-samples", type=int, default=10000)
parser.add_argument("--cfg-scale", type=float, nargs="+", default=[3.5])
parser.add_argument("--noise-scale", type=float, nargs="+", default=[1.0])
parser.add_argument("--n-iterations", type=int, nargs="+", default=[1, 4, 8, 16, 32])
parser.add_argument("--num-sampling-steps", type=int, nargs="+", default=[10])


def main(args: argparse.Namespace) -> None:
    utils.setup_torch()
    device = utils.get_torch_device().type

    model = load_checkpoint(args.experiment, args.checkpoint_dir, device)
    n_classes = model.config.num_classes

    stats_path = f"fid_stats/{args.dataset}_train.npz"
    train_loader = setup_dataloaders(256, args.dataset)[0]

    combos = list(
        itertools.product(
            args.cfg_scale, args.noise_scale, args.n_iterations, args.num_sampling_steps
        )
    )
    total = len(combos)
    results: list[dict[str, float]] = []

    print(f"\nSweeping {total} combinations...")

    for idx, (cfg, noise, n_iter, n_steps) in enumerate(combos, 1):
        model.config.cfg_scale = cfg
        model.config.noise_scale = noise
        model.net.n_iterations = n_iter
        model.config.num_sampling_steps = n_steps

        # Generate samples
        all_samples = []
        remaining = args.fid_samples
        with torch.inference_mode(), utils.maybe_autocast(device):
            while remaining > 0:
                B = min(256, remaining)
                labels = torch.randint(0, n_classes, (B,), device=device)
                imgs = model.generate(labels)  # [-1, 1]
                imgs = ((imgs + 1) / 2).clamp(0, 1)
                all_samples.append((imgs * 255).to(torch.uint8).cpu())
                remaining -= B
                done = args.fid_samples - remaining
                print(
                    f"  [{idx}/{total}] cfg={cfg} noise={noise} n_iter={n_iter} steps={n_steps} — generating samples: {done}/{args.fid_samples}",
                    end="\r",
                )
        print()
        samples = torch.cat(all_samples, dim=0)

        metrics = compute_fid_is(samples, stats_path, device, train_loader)
        fid = metrics["val/fid"]
        is_mean = metrics["val/is_mean"]
        is_std = metrics["val/is_std"]

        row = {
            "n_iterations": n_iter,
            "num_sampling_steps": n_steps,
            "cfg_scale": cfg,
            "noise_scale": noise,
            "fid": fid,
            "is_mean": is_mean,
            "is_std": is_std,
        }
        results.append(row)
        print(
            f"  [{idx}/{total}] cfg={cfg} noise={noise} n_iter={n_iter} steps={n_steps} -> FID={fid:.2f} IS={is_mean:.2f}+/-{is_std:.2f}"
        )

    # Save CSV
    csv_path = Path(args.checkpoint_dir) / args.experiment / "sweep_results.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "n_iterations",
                "num_sampling_steps",
                "cfg_scale",
                "noise_scale",
                "fid",
                "is_mean",
                "is_std",
            ],
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {csv_path}")

    # Print summary sorted by FID
    results.sort(key=lambda r: r["fid"])
    print(
        f"\n{'n_iterations':>12} {'steps':>6} {'cfg_scale':>10} {'noise_scale':>12} {'FID':>8} {'IS':>14}"
    )
    print("-" * 68)
    for r in results:
        print(
            f"{r['n_iterations']:>12} {r['num_sampling_steps']:>6} {r['cfg_scale']:>10.1f} {r['noise_scale']:>12.1f} {r['fid']:>8.2f} "
            f"{r['is_mean']:>6.2f}+/-{r['is_std']:.2f}"
        )


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
