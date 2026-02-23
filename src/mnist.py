import random

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import functional as F


class MNIST(Dataset):
    def __init__(self, root: str = "~/.cache/data", train: bool = True):
        raw = datasets.MNIST(root=root, train=train, download=True)
        x = raw.data.unsqueeze(1).float() / 255.0 * 2.0 - 1.0  # (N, 1, 28, 28)
        self.x = F.pad(x, [2, 2, 2, 2])  # (N, 1, 32, 32)
        self.c = raw.targets.reshape(-1, 1).float()  # (N,1) digit class 0-9

    @property
    def n_channels(self) -> int:
        return 1

    @property
    def img_resolution(self) -> tuple[int, int]:
        return (32, 32)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        angle = random.uniform(0, 360)
        x = F.rotate(self.x[idx], angle=-angle)  # negative = clockwise
        return x.flatten(), self.c[idx]


def setup_dataloaders(batch_size):
    train_ds = MNIST(train=True)
    test_ds = MNIST(train=False)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )
    return train_loader, test_loader


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    ds = MNIST(train=False)
    print(f"Dataset size: {len(ds)}")

    fig, axes = plt.subplots(1, 6, figsize=(12, 2))
    for i, ax in enumerate(axes):
        x, c = ds[i]
        ax.imshow(x.squeeze(0), cmap="gray")
        ax.set_title(f"class={c.item()}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()
