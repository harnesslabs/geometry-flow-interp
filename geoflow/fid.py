"""Device-agnostic FID and Inception Score using TF-compatible InceptionV3 weights.

The InceptionV3 architecture replicates TensorFlow's pretrained model exactly,
including its bugs (max pool in Mixed_7c, count_include_pad=False, 1008 classes).
This ensures FID/IS numbers are directly comparable with published papers.

Architecture adapted from torch-fidelity (MIT License).
Weights: https://github.com/toshas/torch-fidelity/releases/download/v0.2.0/weights-inception-2015-12-05-6726825d.pth
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_WEIGHTS_URL = "https://github.com/toshas/torch-fidelity/releases/download/v0.2.0/weights-inception-2015-12-05-6726825d.pth"


# ---------------------------------------------------------------------------
# TF-compatible bilinear interpolation
# ---------------------------------------------------------------------------


def _tf_bilinear_resize(x: torch.Tensor, size: int) -> torch.Tensor:
    """Resize using TF 1.x coordinate mapping via manual bilinear interpolation.

    TF 1.x maps output pixel i to input coordinate i * (in_size / out_size).
    Manual interpolation avoids F.grid_sample which lacks MPS border padding support.
    """
    _, _, in_h, in_w = x.shape

    # TF 1.x coordinate mapping: out_pixel * (in_size / out_size)
    h = torch.arange(size, dtype=x.dtype, device=x.device) * (in_h / size)
    w = torch.arange(size, dtype=x.dtype, device=x.device) * (in_w / size)

    h0 = h.long().clamp(0, in_h - 1)
    h1 = (h0 + 1).clamp(0, in_h - 1)
    hf = (h - h0.float()).reshape(1, 1, size, 1)

    w0 = w.long().clamp(0, in_w - 1)
    w1 = (w0 + 1).clamp(0, in_w - 1)
    wf = (w - w0.float()).reshape(1, 1, 1, size)

    top = x[:, :, h0][:, :, :, w0] * (1 - wf) + x[:, :, h0][:, :, :, w1] * wf
    bot = x[:, :, h1][:, :, :, w0] * (1 - wf) + x[:, :, h1][:, :, :, w1] * wf
    return top * (1 - hf) + bot * hf


# ---------------------------------------------------------------------------
# InceptionV3 building blocks (TF-compatible)
# ---------------------------------------------------------------------------


class BasicConv2d(nn.Module):
    def __init__(self, in_c: int, out_c: int, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_c, eps=0.001)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)), inplace=True)


class InceptionA(nn.Module):
    def __init__(self, in_c: int, pool_features: int):
        super().__init__()
        self.branch1x1 = BasicConv2d(in_c, 64, kernel_size=1)
        self.branch5x5_1 = BasicConv2d(in_c, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)
        self.branch3x3dbl_1 = BasicConv2d(in_c, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)
        self.branch_pool = BasicConv2d(in_c, pool_features, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5_2(self.branch5x5_1(x))
        branch3x3dbl = self.branch3x3dbl_3(self.branch3x3dbl_2(self.branch3x3dbl_1(x)))
        branch_pool = self.branch_pool(
            F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        )
        return torch.cat([branch1x1, branch5x5, branch3x3dbl, branch_pool], 1)


class InceptionB(nn.Module):
    def __init__(self, in_c: int):
        super().__init__()
        self.branch3x3 = BasicConv2d(in_c, 384, kernel_size=3, stride=2)
        self.branch3x3dbl_1 = BasicConv2d(in_c, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch3x3 = self.branch3x3(x)
        branch3x3dbl = self.branch3x3dbl_3(self.branch3x3dbl_2(self.branch3x3dbl_1(x)))
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        return torch.cat([branch3x3, branch3x3dbl, branch_pool], 1)


class InceptionC(nn.Module):
    def __init__(self, in_c: int, c7: int):
        super().__init__()
        self.branch1x1 = BasicConv2d(in_c, 192, kernel_size=1)
        self.branch7x7_1 = BasicConv2d(in_c, c7, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_1 = BasicConv2d(in_c, c7, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch_pool = BasicConv2d(in_c, 192, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch1x1 = self.branch1x1(x)
        branch7x7 = self.branch7x7_3(self.branch7x7_2(self.branch7x7_1(x)))
        branch7x7dbl = self.branch7x7dbl_5(
            self.branch7x7dbl_4(
                self.branch7x7dbl_3(self.branch7x7dbl_2(self.branch7x7dbl_1(x)))
            )
        )
        branch_pool = self.branch_pool(
            F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        )
        return torch.cat([branch1x1, branch7x7, branch7x7dbl, branch_pool], 1)


class InceptionD(nn.Module):
    def __init__(self, in_c: int):
        super().__init__()
        self.branch3x3_1 = BasicConv2d(in_c, 192, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=2)
        self.branch7x7x3_1 = BasicConv2d(in_c, 192, kernel_size=1)
        self.branch7x7x3_2 = BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=3, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch3x3 = self.branch3x3_2(self.branch3x3_1(x))
        branch7x7x3 = self.branch7x7x3_4(
            self.branch7x7x3_3(self.branch7x7x3_2(self.branch7x7x3_1(x)))
        )
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        return torch.cat([branch3x3, branch7x7x3, branch_pool], 1)


class InceptionE_1(nn.Module):
    """First Mixed_7 block — uses avg_pool (count_include_pad=False)."""

    def __init__(self, in_c: int):
        super().__init__()
        self.branch1x1 = BasicConv2d(in_c, 320, kernel_size=1)
        self.branch3x3_1 = BasicConv2d(in_c, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))
        self.branch3x3dbl_1 = BasicConv2d(in_c, 448, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))
        self.branch_pool = BasicConv2d(in_c, 192, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = torch.cat(
            [self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)], 1
        )
        branch3x3dbl = self.branch3x3dbl_2(self.branch3x3dbl_1(x))
        branch3x3dbl = torch.cat(
            [self.branch3x3dbl_3a(branch3x3dbl), self.branch3x3dbl_3b(branch3x3dbl)], 1
        )
        branch_pool = self.branch_pool(
            F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        )
        return torch.cat([branch1x1, branch3x3, branch3x3dbl, branch_pool], 1)


class InceptionE_2(nn.Module):
    """Second Mixed_7 block — uses max_pool (replicates TF bug)."""

    def __init__(self, in_c: int):
        super().__init__()
        self.branch1x1 = BasicConv2d(in_c, 320, kernel_size=1)
        self.branch3x3_1 = BasicConv2d(in_c, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))
        self.branch3x3dbl_1 = BasicConv2d(in_c, 448, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))
        self.branch_pool = BasicConv2d(in_c, 192, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = torch.cat(
            [self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)], 1
        )
        branch3x3dbl = self.branch3x3dbl_2(self.branch3x3dbl_1(x))
        branch3x3dbl = torch.cat(
            [self.branch3x3dbl_3a(branch3x3dbl), self.branch3x3dbl_3b(branch3x3dbl)], 1
        )
        # TF bug: max_pool instead of avg_pool in the last InceptionE block
        branch_pool = self.branch_pool(
            F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        )
        return torch.cat([branch1x1, branch3x3, branch3x3dbl, branch_pool], 1)


# ---------------------------------------------------------------------------
# TF-compatible InceptionV3 model
# ---------------------------------------------------------------------------


class InceptionV3(nn.Module):
    """InceptionV3 with TF-pretrained weights. Returns (pool_2048, logits_1008)."""

    def __init__(self):
        super().__init__()
        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = InceptionC(768, c7=128)
        self.Mixed_6c = InceptionC(768, c7=160)
        self.Mixed_6d = InceptionC(768, c7=160)
        self.Mixed_6e = InceptionC(768, c7=192)
        self.Mixed_7a = InceptionD(768)
        self.Mixed_7b = InceptionE_1(1280)
        self.Mixed_7c = InceptionE_2(2048)
        self.fc = nn.Linear(2048, 1008)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        # N x 2048 x 8 x 8
        pool = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)  # N x 2048
        logits = self.fc(pool)  # N x 1008
        # Return logits without bias to match torch-fidelity convention
        logits_unbiased = logits - self.fc.bias.unsqueeze(0)
        return pool, logits_unbiased


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _load_inception(device):
    model = InceptionV3()
    state_dict = torch.hub.load_state_dict_from_url(_WEIGHTS_URL, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    model.requires_grad_(False)
    return model


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def _extract(model, images, device, batch_size=256):
    """Extract 2048-d pool features and 1008-d logits from uint8 images."""
    all_feats, all_logits = [], []
    n = len(images)
    for i in range(0, n, batch_size):
        batch = images[i : i + batch_size].to(device).float()
        if batch.shape[1] == 1:
            batch = batch.expand(-1, 3, -1, -1)
        batch = _tf_bilinear_resize(batch, 299)
        batch = (batch - 128) / 128  # normalize to [-1, 1]
        pool, logits = model(batch)
        all_feats.append(pool.cpu())
        all_logits.append(logits.cpu())
        print(f"  extracting features: {min(i + batch_size, n)}/{n}", end="\r")
    if n > 0:
        print()
    return torch.cat(all_feats).numpy(), torch.cat(all_logits)


# ---------------------------------------------------------------------------
# Statistics & metrics (unchanged)
# ---------------------------------------------------------------------------


def _compute_stats(feats):
    mu = np.mean(feats, axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu.astype(np.float32), sigma.astype(np.float64)


def _fid(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    eigvals = np.linalg.eigvals(sigma1 @ sigma2).astype("complex128")
    tr_covmean = np.real(np.sum(np.sqrt(eigvals)))
    return float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def _inception_score(logits, splits=10, rng_seed=2020):
    N = logits.shape[0]
    rng = np.random.RandomState(rng_seed)
    logits = logits[rng.permutation(N)]
    logits = logits.double()
    p = logits.softmax(dim=1)
    log_p = logits.log_softmax(dim=1)
    scores = []
    for k in range(splits):
        p_chunk = p[k * N // splits : (k + 1) * N // splits]
        log_p_chunk = log_p[k * N // splits : (k + 1) * N // splits]
        q = p_chunk.mean(dim=0, keepdim=True)
        kl = (p_chunk * (log_p_chunk - q.log())).sum(dim=1).mean().exp().item()
        scores.append(kl)
    return float(np.mean(scores)), float(np.std(scores))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def precompute_stats(loader, device):
    """Extract InceptionV3 features from a DataLoader and return (mu, sigma)."""
    model = _load_inception(device)
    all_feats = []
    n_batches = len(loader)
    for batch_idx, (x, _y) in enumerate(loader, 1):
        imgs = ((x + 1) / 2).clamp(0, 1)
        imgs = (imgs * 255).to(torch.uint8)
        feats, _ = _extract(model, imgs, device)
        all_feats.append(feats)
        print(f"  reference stats: {batch_idx}/{n_batches} batches", end="\r")
    print()
    return _compute_stats(np.concatenate(all_feats))


def compute_fid_is(samples, stats_path, device, train_loader=None, batch_size=256):
    """Compute FID and IS for uint8 sample tensor against cached reference stats.

    If stats_path doesn't exist and train_loader is provided, computes and caches
    reference statistics automatically.
    """
    if not os.path.exists(stats_path):
        if train_loader is None:
            raise FileNotFoundError(
                f"Reference stats not found at {stats_path} and no train_loader provided."
            )
        print(f"  computing reference stats → {stats_path}")
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)
        mu, sigma = precompute_stats(train_loader, device)
        np.savez(stats_path, mu=mu, sigma=sigma)
        print(f"  cached reference stats to {stats_path}")
    else:
        print(f"  loading cached reference stats from {stats_path}")

    ref = np.load(stats_path)
    ref_mu, ref_sigma = ref["mu"], ref["sigma"]

    model = _load_inception(device)
    feats, logits = _extract(model, samples, device, batch_size=batch_size)
    gen_mu, gen_sigma = _compute_stats(feats)

    fid = _fid(gen_mu, gen_sigma, ref_mu, ref_sigma)
    is_mean, is_std = _inception_score(logits)

    return {"val/fid": fid, "val/is_mean": is_mean, "val/is_std": is_std}
