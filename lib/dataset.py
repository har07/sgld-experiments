import torch.utils
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