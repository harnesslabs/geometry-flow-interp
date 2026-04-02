import random

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import functional as F


class MNIST(Dataset):
    def __init__(
        self,
        root: str = "~/.cache/data",
        train: bool = True,
        rotate: bool = True,
    ):
        raw = datasets.MNIST(root=root, train=train, download=True)
        self.x = raw.data.unsqueeze(1).float() / 255.0 * 2.0 - 1.0  # (N, 1, 28, 28)
        self.c = raw.targets  # (N,) digit class 0-9
        self.rotate = rotate

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.x.shape[1:]  # (C, H, W)

    @property
    def n_classes(self) -> int:
        return np.unique(self.c).shape[0]

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:  # ty: ignore[invalid-method-override]
        x = self.x[idx]
        if self.rotate:
            angle = random.uniform(0, 360)
            x = F.rotate(x, angle=-angle, fill=[-1])  # normalized fill value
        return x, self.c[idx]


class CIFAR10(Dataset):
    def __init__(
        self,
        root: str = "~/.cache/data",
        train: bool = True,
        N: int | None = None,
    ):
        raw = datasets.CIFAR10(root=root, train=train, download=True)
        self.x = (
            torch.from_numpy(raw.data).permute(0, 3, 1, 2).float() / 255.0 * 2.0 - 1.0
        )  # (N, 3, 32, 32)
        self.c = torch.tensor(raw.targets)  # (N,) class 0-9
        if N is not None:
            rng = np.random.RandomState(42)
            n_classes = len(torch.unique(self.c))
            per_class = N // n_classes
            idxs = []
            for cls in range(n_classes):
                cls_idxs = (self.c == cls).nonzero(as_tuple=True)[0].numpy()
                rng.shuffle(cls_idxs)
                idxs.append(torch.from_numpy(cls_idxs[:per_class]))
            idxs = torch.cat(idxs)
            self.x = self.x[idxs]
            self.c = self.c[idxs]

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.x.shape[1:]  # (C, H, W)

    @property
    def n_classes(self) -> int:
        return np.unique(self.c).shape[0]

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:  # ty: ignore[invalid-method-override]
        return self.x[idx], self.c[idx]


_DATASETS = {
    "mnist": MNIST,
    "cifar10": CIFAR10,
}


def setup_dataloaders(batch_size, dataset="mnist"):
    ds_cls = _DATASETS[dataset]
    train_loader = torch.utils.data.DataLoader(
        ds_cls(train=True),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=2,
        persistent_workers=True,
    )
    test_loader = torch.utils.data.DataLoader(
        ds_cls(train=False),
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        num_workers=2,
        persistent_workers=True,
    )
    return train_loader, test_loader


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    ds = MNIST(train=False)
    print(f"Dataset size: {len(ds)}")

    fig, axes = plt.subplots(1, 6, figsize=(12, 2))
    for i, ax in enumerate(axes):
        x, c = ds[i]
        ax.imshow(x.reshape(28, 28), cmap="gray")
        ax.set_title(f"class={c.item()}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()
