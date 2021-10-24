import torch.utils
import random
from torchvision import datasets, transforms
from .preproc import NoiseTransform
import pickle
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
import PIL
from torch.utils.data import Dataset
import glob
import os

# workaround for MNIST download problem: https://github.com/pytorch/vision/issues/1938#issuecomment-789986996
from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)
# end workaround

class NotMnist(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.filelist = glob.glob(os.path.join(self.root_dir, '**', '*.png'))
        new_filelist = []
        for file in self.filelist:
            try:
                io.imread(file)
                new_filelist.append(file)
            except:
                pass
                
        self.filelist = new_filelist
        
    def __len__(self):
        return len(self.filelist)
        
    def __getitem__(self, idx):
        image = io.imread(self.filelist[idx])
        image = PIL.Image.fromarray(image)
        if self.transform:
            return (self.transform(image), [])
        return (image, [])

class LimitDataset(Dataset):
    def __init__(self, dataset, n):
        self.dataset = dataset
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.dataset[i]


def make_datasets(bs=50, test_bs=4096, noise=0):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('mnist_data',
            train=True,
            download=True,
            transform=transforms.Compose([NoiseTransform(noise),
                                     transforms.ToTensor()])),
        batch_size=bs,
        shuffle=False)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('mnist_data',
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

def make_datasets_cifar100(bs=128, test_bs=100, shuffle=True):
    CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
    ])
    cifar100_training = datasets.CIFAR100(root='cifar100_data', train=True, download=True, transform=transform_train)
    cifar100_training_loader = torch.utils.data.DataLoader(
        cifar100_training, shuffle=shuffle, batch_size=bs)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
    ])
    cifar100_test = datasets.CIFAR100(root='cifar100_data', train=False, download=True, transform=transform_test)
    cifar100_test_loader = torch.utils.data.DataLoader(
        cifar100_test, shuffle=shuffle, batch_size=test_bs)

    return cifar100_training_loader, cifar100_test_loader

def make_datasets_notmnist(bs=128, test_bs=100, shuffle=True):
    ds = NotMnist('notmnist_data', transform=transforms.Compose([transforms.ToTensor()]))
    limited_ds = LimitDataset(ds, 10000)
    notmnist_test_loader = torch.utils.data.DataLoader(limited_ds, batch_size=test_bs, shuffle=shuffle)

    return None, notmnist_test_loader

def normalize(data_tensor):
    '''re-scale image values to [-1, 1]'''
    return (data_tensor / 255.) * 2. - 1. 

def make_datasets_svhn(bs=128, test_bs=100, shuffle=True):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5)),
    ])

    ds = datasets.SVHN('svhn_data', split='test', download=True, transform=transform_test)
    limited_ds = LimitDataset(ds, 10000)
    test_loader = torch.utils.data.DataLoader(
        limited_ds,
        batch_size=test_bs, shuffle=shuffle, num_workers=2)

    return None, test_loader

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
        datasets.MNIST('mnist_data',
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

# this class is taken from: https://github.com/fregu856/evaluating_bdl/blob/master/toyClassification/SGLD-64/dataset.py
class ToyDataset(torch.utils.data.Dataset):
    def __init__(self, base_dir):
        self.examples = []

        with open(base_dir + "/x.pkl", "rb") as file: # (needed for python3)
            x = pickle.load(file) # (shape: (2000, 2))

        with open(base_dir + "/y.pkl", "rb") as file: # (needed for python3)
            y = pickle.load(file) #  (shape: (2000, ))

        x_1_train = []
        x_2_train = []
        y_train = []
        for i in range(x.shape[0]):
            if x[i, 0] > 0:
                x_1_train.append(x[i, 0])
                x_2_train.append(x[i, 1])
                y_train.append(y[i])

        y_train = np.array(y_train)
        x_train = np.zeros((len(y_train), 2), dtype=np.float32)
        x_train[:, 0] = np.array(x_1_train)
        x_train[:, 1] = np.array(x_2_train)

        x_train_false = x_train[y_train == 0] # (shape: (num_false, 2))
        x_train_true = x_train[y_train == 1] # (shape: (num_true, 2))
        # print ("num_false: %d" % x_train_false.shape[0])
        # print ("num_true: %d" % x_train_true.shape[0])
        # plt.figure(1)
        # plt.plot(x_train_false[:, 0], x_train_false[:, 1], "r.")
        # plt.plot(x_train_true[:, 0], x_train_true[:, 1], "b.")
        # plt.ylabel("x_2")
        # plt.xlabel("x_1")
        # plt.xlim([-3, 3])
        # plt.ylim([-3, 3])
        # plt.savefig("/content/sgld-experiments/training_data.png")
        # plt.close(1)

        for i in range(x_train.shape[0]):
            example = {}
            example["x"] = x_train[i]
            example["y"] = y_train[i]
            self.examples.append(example)

        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        x = example["x"]
        y = example["y"]

        return (x, y)

    def __len__(self):
        return self.num_examples