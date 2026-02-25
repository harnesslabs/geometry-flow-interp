from pathlib import Path

import torch

from gf.denoiser import Denoiser


class CheckpointManager:
    def __init__(
        self,
        args,
        model: Denoiser | torch.nn.parallel.DistributedDataParallel,
        optimizer,
        scheduler,
        device,
    ):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        self.ckpt_dir = Path(args.checkpoint_dir)
        self.interval = getattr(args, "checkpoint_interval", 0)

    @staticmethod
    def _unwrap(
        module: Denoiser | torch.nn.parallel.DistributedDataParallel,
    ) -> Denoiser:
        if isinstance(module, torch.nn.parallel.DistributedDataParallel):
            return module.module
        return module

    def _optimizer_to_device(self):
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device, non_blocking=True)

    def save(self, epoch: int, global_step: int, tag: list[str] | str):
        base = self._unwrap(self.model)
        state = {
            "epoch": epoch,  # convention: next epoch to run
            "global_step": global_step,  # canonical progress
            "model": base.state_dict(),
            "ema": {k: v.state_dict() for k, v in base.ema.items()},
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "args": vars(self.args),
            "config": vars(base.config),
        }
        if isinstance(tag, str):
            tag = [tag]
        for t in tag:
            path = self.ckpt_dir / f"{t}.pt"
            torch.save(state, path)
            print(f"Saved checkpoint to {path} ({epoch=}, {global_step=})")

    def maybe_save(self, epoch: int, global_step: int):
        if self.interval <= 0:
            return
        if global_step % self.interval == 0:
            self.save(epoch, global_step, tag=[f"step_{global_step:010d}", "last"])

    def load_if_available(self):
        resume_path = getattr(self.args, "resume_path", "")
        if not resume_path:
            return 0, 0  # epoch, global_step

        print(f"Loading from {resume_path}...")
        ckpt = torch.load(resume_path, map_location=self.device)

        base = self._unwrap(self.model)
        base.load_state_dict(ckpt["model"])
        for k, v in base.ema.items():
            v.load_state_dict(ckpt["ema"][k])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self._optimizer_to_device()
        self.scheduler.load_state_dict(ckpt["scheduler"])

        epoch = ckpt.get("epoch", 0)
        global_step = ckpt.get("global_step", 0)

        print(f"Resumed checkpoint ({epoch=}, {global_step=})")
        return epoch, global_step
