import argparse
import warnings
from pathlib import Path

import numpy as np
import torch
from torchinfo import summary
from torchvision.utils import save_image

from geoflow import utils
from geoflow.datasets import setup_dataloaders
from geoflow.denoiser import Denoiser, DenoiserConfig
from geoflow.fid import compute_fid_is

warnings.filterwarnings(
    "ignore",
    message=".*dtype.*align.*",
    category=np.exceptions.VisibleDeprecationWarning,
)

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", type=str, default="default")
parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
parser.add_argument(
    "--dataset", type=str, default="cifar10", choices=["mnist", "cifar10"]
)
parser.add_argument(
    "--fid-samples", type=int, default=10000, help="number of samples for FID/IS"
)
parser.add_argument(
    "--preview", action="store_true", help="preview output, skip metrics"
)


def load_checkpoint(experiment: str, checkpoint_dir: str, device: str) -> Denoiser:
    path = Path(checkpoint_dir) / experiment / "last.pt"
    ckpt = torch.load(path, map_location=device)

    config = DenoiserConfig(**ckpt["config"])
    model = Denoiser(config, device).to(device)
    summary(model, depth=3)

    decay = sorted(model.ema.keys())[0]
    model.ema[decay].load_state_dict(ckpt["ema"][decay])
    model.swap_ema(decay=decay)
    print(f"Using EMA with {decay=} for sampling")

    model.eval()

    print(f"Loaded checkpoint from {path} (step={ckpt.get('global_step', '?')})")
    return model


def main(args: argparse.Namespace) -> None:
    utils.setup_torch()
    device = utils.get_torch_device().type

    model = load_checkpoint(args.experiment, args.checkpoint_dir, device)
    # best args
    model.config.cfg_scale = 2.5
    model.config.noise_scale = 0.9

    n_classes = model.config.num_classes

    if args.preview:
        n_samples = 10
        cond = torch.arange(n_classes, device=device).repeat_interleave(n_samples)
        with torch.inference_mode(), utils.maybe_autocast(device):
            samples = model.generate(cond)
        samples = ((samples.float() + 1) / 2).clamp(0, 1)
        out_dir = Path("media")
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / "generated_grid.png"
        save_image(samples, out_path, nrow=n_samples)
        print(f"Saved {n_classes}x{n_samples} grid to {out_path}")
        return

    # --- FID / IS ---
    print(f"Generating {args.fid_samples} samples for FID/IS...")
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
            print(
                f"  generating samples: {args.fid_samples - remaining}/{args.fid_samples}",
                end="\r",
            )
    print()
    fid_samples = torch.cat(all_samples, dim=0)

    stats_path = f"fid_stats/{args.dataset}_train.npz"
    train_loader = setup_dataloaders(256, args.dataset)[0]
    metrics = compute_fid_is(fid_samples, stats_path, device, train_loader)
    print(f"FID: {metrics['val/fid']:.2f}")
    print(f"IS:  {metrics['val/is_mean']:.2f} ± {metrics['val/is_std']:.2f}")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
