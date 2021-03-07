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