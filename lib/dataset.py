import torch.utils
import random
from torchvision import datasets, transforms
from .preproc import NoiseTransform

# workaround for MNIST download problem: https://github.com/pytorch/vision/issues/1938#issuecomment-789986996
from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)
# end workaround

def make_datasets(bs=50, test_bs=4096, noise=0):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data',
            train=True,
            download=True,
            transform=transforms.Compose([NoiseTransform(noise),
                                     transforms.ToTensor()])),
        batch_size=bs,
        shuffle=False)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data',
            train=False,
            transform=transforms.Compose([NoiseTransform(noise),
                                        transforms.ToTensor()])),
        batch_size=test_bs)

    return train_loader, test_loader

def make_datasets_cifar10(bs=128, test_bs=100):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data', train=True, download=True, transform=transform_train),
        batch_size=bs, shuffle=True, num_workers=2)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data', train=False, transform=transform_test),
        batch_size=test_bs, shuffle=False, num_workers=2)

    
    # classes = ('plane', 'car', 'bird', 'cat', 'deer',
    #        'dog', 'frog', 'horse', 'ship', 'truck')

    return train_loader, test_loader

def _get_simultan_subsets_loader(trainset, batch_size):
    n_per_class = 6000
    skip = 4000
    if isinstance(trainset, datasets.CIFAR10):
        n_per_class = 5000

    excluded = [] 
    for i in range(10):
        choice = list(j for j in range(i*n_per_class+skip,(i+1)*n_per_class))
        excluded.append(random.choice(choice))

    excluded2 = []
    for i in range(10):
        choice = list(j for j in range(i*n_per_class+skip,(i+1)*n_per_class))
        while True:
            candidate = random.choice(choice)
            if candidate not in excluded:
                excluded2.append(candidate)
                break

    S = list(x for x in range(0,10*n_per_class) if x not in excluded)
    S2 = list(x for x in range(0,10*n_per_class) if x not in excluded2)

    trainset_s = torch.utils.data.Subset(trainset, S)
    trainset_s2 = torch.utils.data.Subset(trainset, S2)

    train_loader_s = torch.utils.data.DataLoader(trainset_s, batch_size=batch_size, shuffle=False)
    train_loader_s2 = torch.utils.data.DataLoader(trainset_s2, batch_size=batch_size, shuffle=False)

    return train_loader_s, train_loader_s2

def make_simultan(bs=50, test_bs=4096, noise=0):
    trainset = datasets.MNIST('mnist_data', train=True, download=True,
                            transform=transforms.Compose([NoiseTransform(0), transforms.ToTensor()])
                        )

    train_loader_s, train_loader_s2 = _get_simultan_subsets_loader(trainset, bs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data',
            train=False,
            transform=transforms.Compose([NoiseTransform(noise),
                                        transforms.ToTensor()])),
        batch_size=test_bs)

    return train_loader_s, train_loader_s2, test_loader

def make_simultan_cifar10(bs=128, test_bs=100):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR10('data', train=True, download=True, transform=transform_train)
    train_loader_s, train_loader_s2 = _get_simultan_subsets_loader(trainset, bs)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data', train=False, transform=transform_test),
        batch_size=test_bs, shuffle=False, num_workers=2)

    return train_loader_s, train_loader_s2, test_loader
