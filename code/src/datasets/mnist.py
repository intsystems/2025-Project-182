# datasets/mnist.py
import os
import torch
import random
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader


def _build_transforms(input_dim, augment, mean, std):
    t = []
    if input_dim != 28:
        t.append(transforms.Resize((input_dim, input_dim), antialias=True))
    if augment:
        t.extend(
            [
                transforms.RandomCrop(
                    input_dim, padding=2 if input_dim >= 28 else max(1, input_dim // 16)
                ),
                transforms.RandomRotation(degrees=10, fill=0),
            ]
        )
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(t)


def _default_norm(mode):
    mode = (mode or "mnist").lower()
    if mode == "mnist":
        mean = (0.1307,)
        std = (0.3081,)
    elif mode == "zeroone":
        mean = (0.0,)
        std = (1.0,)
    else:
        mean = (0.5,)
        std = (0.5,)
    return mean, std


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class MNISTDataset(Dataset):
    def __init__(self, root: str, train: bool, transform):
        super().__init__()
        self._ds = datasets.MNIST(
            root=root, train=train, transform=transform, download=True
        )

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        img, label = self._ds[idx]
        return img, label


def get_mnist_dataloader(conf, train: bool = True) -> DataLoader:
    mean, std = _default_norm(conf.normalization)

    transform = _build_transforms(
        input_dim=conf.input_dim,
        augment=train,
        mean=mean,
        std=std,
    )

    dataset = MNISTDataset(root=conf.root, train=train, transform=transform)

    generator = None
    worker_init_fn = None
    if conf.seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(conf.seed))
        worker_init_fn = _seed_worker

    dataloader = DataLoader(
        dataset,
        batch_size=conf.batch_size,
        shuffle=train,
        num_workers=0,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )

    return dataloader
