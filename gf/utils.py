import torch


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


def grad_norm(model: torch.nn.Module) -> float:
    device = next(model.parameters()).device
    total_sq = torch.zeros(1, device=device)

    for p in model.parameters():
        if p.grad is None:
            continue
        total_sq += p.grad.detach().float().pow(2).sum()

    return total_sq.sqrt().item()
