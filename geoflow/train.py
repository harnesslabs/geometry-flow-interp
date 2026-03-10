import argparse
import dataclasses
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchinfo import summary

import wandb
from geoflow import utils
from geoflow.checkpoint import CheckpointManager
from geoflow.denoiser import Denoiser, DenoiserConfig
from geoflow.mnist import setup_dataloaders
from geoflow.model import models

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="JiT-S/7", choices=models.keys())
parser.add_argument("--bs", "--batch-size", type=int, default=256)
parser.add_argument("--epochs", type=int, default=100)

# optimizer
parser.add_argument("--lr", "--learning-rate", type=float, default=3e-4)
parser.add_argument("--wd", "--weight-decay", type=float, default=0.0)
parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
parser.add_argument("--eps", type=float, default=1e-10)

parser.add_argument("--muon-lr", type=float, default=0.01)
parser.add_argument("--muon-wd", type=float, default=0.1)
parser.add_argument("--muon-beta2", type=float, default=0.9)
parser.add_argument("--muon-momentum", type=float, default=0.95)

parser.add_argument("--warmup", type=float, default=0.05, help="lr warmup")
parser.add_argument("--adamw", action="store_true", help="only AdamW")
parser.add_argument("--cosine", action="store_true", help="lr anneal")
parser.add_argument("--grad-norm", type=float, default=1.0)

# experiment
parser.add_argument("--experiment", type=str, default="default")
parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
parser.add_argument("--checkpoint-interval", type=int, default=2)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--offline", action="store_true", help="disable wandb")


def _train_eval_classifier(train_loader, val_loader, device):
    ds, t0 = train_loader.dataset, time.time()
    clf = nn.Sequential(
        nn.Flatten(),
        nn.Linear(np.prod(ds.shape), 1024), nn.BatchNorm1d(1024), nn.ReLU(),
        nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(),
        nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
        nn.Linear(256, ds.n_classes),
    ).to(device)  # fmt: skip
    with torch.random.fork_rng(devices=[0], device_type=device):
        torch.manual_seed(0)
        opt, epochs = torch.optim.AdamW(clf.parameters(), lr=1e-3), 2
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=8e-2, steps_per_epoch=len(train_loader), epochs=epochs
        )
        for _ in range(epochs):
            for x, y in train_loader:
                opt.zero_grad(set_to_none=True)
                loss = F.cross_entropy(clf(x.to(device)), y.to(device))
                loss.backward()
                opt.step()
                sched.step()
    clf.eval().requires_grad_(False)
    correct = sum(
        (clf(x.to(device)).argmax(1) == y.to(device)).sum().item()
        for x, y in val_loader
    )
    print(
        f"  eval classifier — val acc={correct / len(val_loader.dataset):.4f} "
        f"({time.time() - t0:.1f}s)"
    )
    return clf


def _setup_optimizer(model, args):
    if args.adamw:
        return torch.optim.AdamW(model.parameters(), lr=args.lr)

    from geoflow.optim import MuonAdamW

    muon_params = []
    adamw_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (  # blocks.*.attn.*.weight & blocks.*.mlp.*.weight
            ".blocks." in name
            and name.endswith(".weight")
            and param.ndim == 2
            and ".adaLN_modulation." not in name
        ):
            muon_params.append(param)
        else:  # everything else
            adamw_params.append(param)

    param_groups = [
        {
            "params": adamw_params,
            "kind": "adamw",
            "lr": args.lr,
            "betas": args.betas,
            "eps": args.eps,
            "weight_decay": args.wd,
        },
    ]

    # Group muon params by shape for efficient stacking
    for shape in sorted({p.shape for p in muon_params}):
        group_params = [p for p in muon_params if p.shape == shape]
        param_groups.append(
            {
                "params": group_params,
                "kind": "muon",
                "lr": args.muon_lr,
                "momentum": args.muon_momentum,
                "ns_steps": 5,
                "beta2": args.muon_beta2,
                "weight_decay": args.muon_wd,
            }
        )

    # Print parameter group summary
    muon_ids = {id(p) for p in muon_params}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        kind = "muon" if id(param) in muon_ids else "adamw"
        print(f"  {kind:5s}  {str(list(param.shape)):>14s}  {name}")

    return MuonAdamW(param_groups)


def train(args):
    train_loader, val_loader = setup_dataloaders(args.bs)
    device = utils.get_torch_device().type

    ds = train_loader.dataset
    model = Denoiser(
        DenoiserConfig(
            model=args.model,
            input_size=ds.shape[1],
            in_channels=ds.shape[0],
            num_classes=ds.n_classes,
        ),
        device,
    ).to(device)
    summary(model, depth=3)
    wandb.config.update({"DenoiserConfig": dataclasses.asdict(model.config)})

    optimizer = _setup_optimizer(model, args)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup)
    scheduler = LinearLR(optimizer, start_factor=1e-6, total_iters=warmup_steps)
    if args.cosine:
        cosine = CosineAnnealingLR(
            optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6
        )
        scheduler = SequentialLR(
            optimizer, schedulers=[scheduler, cosine], milestones=[warmup_steps]
        )

    ckpt = CheckpointManager(args, model, optimizer, scheduler, device)
    start_epoch, global_step = ckpt.load_if_available()

    classifier = _train_eval_classifier(train_loader, val_loader, device)

    for epoch in range(start_epoch, args.epochs):
        # training
        model.train()
        data_start = time.time()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            data_dt = time.time() - data_start

            iter_start = time.time()
            optimizer.zero_grad(set_to_none=True)

            with utils.maybe_autocast(device):
                loss = model(x, y)

            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_norm
            ).item()
            if not args.adamw:  # warmup muon momentum & decay weight decay
                frac = min(global_step / warmup_steps, 1) if warmup_steps > 0 else 1
                muon_momentum = max(args.muon_momentum - 0.10, 0.0) + frac * 0.10
                muon_wd = args.muon_wd * (1 - global_step / total_steps)
                for g in optimizer.param_groups:
                    if g["kind"] == "muon":
                        g["momentum"] = muon_momentum
                        g["weight_decay"] = muon_wd
            lr = optimizer.param_groups[0]["lr"]
            optimizer.step()
            scheduler.step()

            model.update_ema()

            torch.accelerator.synchronize()
            dt = time.time() - iter_start

            global_step += 1
            global_kimg = global_step * (args.bs / 1000)

            metrics = {
                "train/loss": loss.item(),
                "train/gnorm": gnorm,
                "train/lr": lr,
                "train/dt": dt,
                "train/data_dt": data_dt,
                "kimg": global_kimg,
                "epoch": epoch,
            }
            if not args.adamw:
                metrics["train/muon_lr"] = optimizer.param_groups[1]["lr"]
            wandb.log(metrics, step=global_step)

            print(
                f"epoch={epoch} step={global_step} kimg={global_kimg:.1f} "
                f"train/loss={metrics['train/loss']:.4f} train/gnorm={gnorm:.3f} {lr=:.2e} "
                f"train/dt={dt:.3f}s train/data={data_dt:.3f}s"
            )
            data_start = time.time()

        # validation
        if epoch % args.checkpoint_interval != 0 and epoch != args.epochs - 1:
            continue

        model.eval()
        with torch.inference_mode():
            params = model.swap_ema()

            # val/loss
            losses: list[float] = []
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                with utils.maybe_autocast(device):
                    loss = model(x, y)
                losses.append(loss.item())

            # val/acc
            num_per_class = 10
            labels = torch.arange(10, device=device).repeat_interleave(num_per_class)
            with utils.maybe_autocast(device):
                samples = model.generate(labels)
            preds = classifier(samples).argmax(dim=1)
            acc = (preds == labels).float().mean().item()

            # val/samples
            grid = ((samples + 1) / 2).clamp(0, 1)  # [-1,1] → [0,1]
            rows = [
                torch.cat(
                    [grid[i * num_per_class + j] for j in range(num_per_class)], dim=2
                )
                for i in range(10)
            ]
            grid_img = torch.cat(rows, dim=1)  # (1, H*10, W*10)

            model.swap_params(params)

        avg_val_loss = sum(losses) / len(losses)

        val_metrics = {
            "val/loss": avg_val_loss,
            "val/acc": acc,
            "val/samples": wandb.Image(grid_img.cpu().float()),
        }
        wandb.log(val_metrics, step=global_step)

        print(f"val/loss={avg_val_loss:.4f} val/acc={acc:.4f}")

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
