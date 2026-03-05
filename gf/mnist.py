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
    def shape(self) -> int:
        return np.prod(self.x.shape[1:]).item()

    @property
    def n_classes(self) -> int:
        return np.unique(self.c).shape[0]

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        x = self.x[idx]
        if self.rotate:
            angle = random.uniform(0, 360)
            x = F.rotate(x, angle=-angle, fill=[-1])  # normalized fill value
        return x, self.c[idx]


def setup_dataloaders(batch_size):
    train_loader = torch.utils.data.DataLoader(
        MNIST(train=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        MNIST(train=False), batch_size=batch_size, shuffle=False, num_workers=0
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
