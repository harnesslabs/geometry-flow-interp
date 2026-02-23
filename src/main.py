import argparse
import os
import time
from contextlib import nullcontext

import torch
from torch.amp import autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchinfo import summary

import utils
import wandb
from checkpoint import CheckpointManager
from denoiser import Denoiser, DenoiserConfig
from mnist import setup_dataloaders

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--learning-rate", type=float, default=1e-3)
parser.add_argument("--grad-norm", type=float, default=1.5)
parser.add_argument("--cosine", action="store_true")

# experiment
parser.add_argument("--experiment", type=str, default="default")
parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
parser.add_argument("--checkpoint-interval", type=int, default=10)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--offline", action="store_true")


def train(args):
    train_loader, val_loader = setup_dataloaders(args.batch_size)
    device = utils.get_torch_device().type

    ds = train_loader.dataset
    features = ds.img_resolution[0] * ds.img_resolution[1] * ds.n_channels
    model = Denoiser(
        DenoiserConfig(
            in_features=features,
            out_features=features,
            cond_features=1,
        ),
        device,
    ).to(device)
    summary(model, depth=3)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999)
    )

    warmup_steps = len(train_loader) * (args.epochs / 10)
    scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
    if args.cosine:
        total_steps = len(train_loader) * args.epochs
        cosine = CosineAnnealingLR(
            optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6
        )
        scheduler = SequentialLR(
            optimizer, schedulers=[scheduler, cosine], milestones=[warmup_steps]
        )

    ckpt = CheckpointManager(args, model, optimizer, scheduler, device)
    start_epoch, global_step = ckpt.load_if_available()

    for epoch in range(start_epoch, args.epochs):
        # training
        model.train()
        data_start = time.time()
        for x, y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            x = x.to(device)  # , non_blocking=True)
            y = y.to(device)  # , non_blocking=True)

            data_dt = time.time() - data_start
            iter_start = time.time()

            with (
                autocast(device, dtype=torch.bfloat16)
                if device == "cuda"
                else nullcontext()
            ):
                loss = model(x, y)

            loss.backward()
            gnorm = utils.grad_norm(model)
            lr = optimizer.param_groups[0]["lr"]
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
            optimizer.step()
            scheduler.step()

            model.update_ema()

            dt = time.time() - iter_start

            global_step += 1
            global_kimg = global_step * (args.batch_size / 1000)

            metrics = {
                "train/loss": loss.item(),
                "train/gnorm": gnorm,
                "train/lr": lr,
                "train/dt": dt,
                "train/data_dt": data_dt,
                "kimg": global_kimg,
                "epoch": epoch,
            }
            wandb.log(metrics, step=global_step)

            print(
                f"epoch={epoch} step={global_step} kimg={global_kimg:.1f} "
                f"train/loss={loss.item():.4f} train/gnorm={gnorm:.3f} lr={lr:.2e} "
                f"train/dt={dt:.3f}s train/data={data_dt:.3f}s"
            )
            data_start = time.time()

        # validation
        if epoch % args.checkpoint_interval != 0 and epoch != args.epochs - 1:
            continue

        model.eval()
        losses: list[float] = []
        mses: list[float] = []

        with torch.no_grad():
            params = model.swap_ema()
            for x, y in val_loader:
                # non_blocking=True makes evals non-deterministic for some reason
                x = x.to(device)
                y = y.to(device)

                with (
                    autocast(device, dtype=torch.bfloat16)
                    if device == "cuda"
                    else nullcontext()
                ):
                    loss = model(x, y)
                    pred = model.generate(y)

                losses.append(loss.item())
                mses.append((pred - x).pow(2).mean().item())

            model.swap_params(params)

        avg_val_loss = sum(losses) / len(losses)
        avg_mse = sum(mses) / len(mses)

        val_metrics = {"val/loss": avg_val_loss, "val/mse": avg_mse}
        wandb.log(val_metrics, step=global_step)

        print(f"val/loss={avg_val_loss:.4f} val/mse={avg_mse:.6f}")

        ckpt.save(epoch + 1, global_step, tag="last")


def main(args):
    utils.setup_torch()

    ckpt_dir = os.path.join(args.checkpoint_dir, args.experiment)
    os.makedirs(ckpt_dir, exist_ok=True)

    resume_path = None
    if args.resume:
        candidate = os.path.join(ckpt_dir, "last.pt")
        if os.path.isfile(candidate):
            resume_path = candidate
            print(f"Resuming from checkpoint: {resume_path}")
        else:
            print(f"Warning: --resume set but no checkpoint found at {candidate}")

    wandb.init(
        project="geometry-flow-interp",
        name=args.experiment,
        config=vars(args),
        mode="offline" if args.offline else "online",
    )

    print(f"Checkpoint directory: {ckpt_dir}")

    args.checkpoint_dir = ckpt_dir
    args.resume_path = resume_path

    train(args)
    wandb.finish()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
