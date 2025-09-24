from .mnist import get_mnist_dataloader
from .cifar_100 import get_cifar100_dataloader


def get_dataloader(conf, train=True):
    if conf.name == "CIFAR-100":
        get_dataloader_fn = get_cifar100_dataloader
    elif conf.name == "MNIST":
        get_dataloader_fn = get_mnist_dataloader
    else:
        raise NotImplementedError
    dataloader = get_dataloader_fn(conf, train)
    return dataloader
