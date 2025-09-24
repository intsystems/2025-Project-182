import os
import torch
import random
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader


def _build_transforms(input_dim, augment, mean, std):
    t = []
    if input_dim != 32:
        t.append(transforms.Resize((input_dim, input_dim), antialias=True))
    if augment:
        t.extend(
            [
                transforms.RandomCrop(input_dim, padding=4 if input_dim >= 36 else 2),
                transforms.RandomHorizontalFlip(),
            ]
        )
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(t)


def _default_norm(mode):
    mode = (mode or "imagenet").lower()
    if mode == "cifar":
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif mode == "zeroone":
        mean = (0.0, 0.0, 0.0)
        std = (1.0, 1.0, 1.0)
    else:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    return mean, std


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class CIFAR100Dataset(Dataset):
    def __init__(self, root: str, train: bool, transform):
        super().__init__()
        self._ds = datasets.CIFAR100(
            root=root, train=train, transform=transform, download=True
        )

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        img, label = self._ds[idx]
        return img, label


def get_cifar100_dataloader(conf, train=True):

    mean, std = _default_norm(conf.normalization)

    transform = _build_transforms(
        input_dim=conf.input_dim,
        augment=train,
        mean=mean,
        std=std,
    )

    dataset = CIFAR100Dataset(root=conf.root, train=train, transform=transform)

    generator = None
    if conf.seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(conf.seed))

    dataloader = DataLoader(
        dataset,
        batch_size=conf.batch_size,
        shuffle=train,
        num_workers=0,
        worker_init_fn=_seed_worker if conf.seed is not None else None,
        generator=generator,
    )

    return dataloader
