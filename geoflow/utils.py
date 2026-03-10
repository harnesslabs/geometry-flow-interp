from contextlib import contextmanager, nullcontext

import torch
from torch.amp import autocast


def setup_torch() -> None:
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def get_torch_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


@contextmanager
def maybe_autocast(device: str):
    with autocast(device, dtype=torch.bfloat16) if device == "cuda" else nullcontext():
        yield
