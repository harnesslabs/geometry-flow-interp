import argparse
from pathlib import Path

import torch
from torchinfo import summary
from torchvision.utils import save_image

from gf import utils
from gf.denoiser import Denoiser, DenoiserConfig

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", type=str, default="default")
parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")


def load_checkpoint(experiment: str, checkpoint_dir: str, device: str) -> Denoiser:
    path = Path(checkpoint_dir) / experiment / "last.pt"
    ckpt = torch.load(path, map_location=device)

    config = DenoiserConfig(**ckpt["config"])
    config.cfg_scale = 2.0
    config.noise_scale = 0.5
    model = Denoiser(config, device).to(device)
    model.load_state_dict(ckpt["model"])
    for i, (k, v) in enumerate(model.ema.items()):
        if i == 1:
            v.load_state_dict(ckpt["ema"][k])

    model.eval()
    summary(model, depth=3)

    print(f"Loaded checkpoint from {path} (step={ckpt.get('global_step', '?')})")
    return model


def main(args: argparse.Namespace) -> None:
    utils.setup_torch()
    device = utils.get_torch_device().type

    model = load_checkpoint(args.experiment, args.checkpoint_dir, device)

    n_classes = 10
    n_samples = 10
    cond = torch.arange(n_classes, device=device).repeat_interleave(n_samples)  # (100,)

    with torch.inference_mode(), utils.maybe_autocast(device):
        samples = model.generate(cond)

    print(samples.shape)

    # (100, 1024) -> (100, 1, 28, 28), denormalize [-1,1] -> [0,1]
    samples = samples.float().view(-1, 1, 28, 28)
    samples = ((samples + 1) / 2).clamp(0, 1)

    out_dir = Path("media")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "generated_grid.png"
    save_image(samples, out_path, nrow=n_samples)
    print(f"Saved {n_classes}x{n_samples} grid to {out_path}")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
