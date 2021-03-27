from collections import OrderedDict
import torch
import numpy as np
import matplotlib.pyplot as plt
import PIL
import lib.dataset as dataset
import lib.model
import seaborn as sns
import glob
import os
import argparse
from lib.model import NotMnist
from torchvision import transforms, utils

def make_statedict(idx, histo):
    usable_statedict = OrderedDict()
    for k, v in histo[idx].items():
        # print(type(v))
        usable_statedict[k] = v
    return usable_statedict

parser = argparse.ArgumentParser(
                    description="Probability density study using models pre-trained on "
                                "MNIST dataset.")
parser.add_argument("-d", "--dir",
                    help="pretrained model location")
parser.add_argument("-nmnist", "--not_mnist",
                    help="notMNIST data directory location")

args = parser.parse_args()
model_dir = str(args.dir)
notmnist_dir = str(args.not_mnist)

train_loader, test_loader = dataset.make_datasets()

bs = 256
notmnist_loader = torch.utils.data.DataLoader(NotMnist(notmnist_dir,
                        transform=transforms.ToTensor()), batch_size=bs, shuffle=True)

def plot_non_bayesian():
    statedict_nonoise_path = glob.glob(os.path.join(model_dir, 'SGD', '*.accum.pt'))[0]
    nonoise_histo = torch.load(statedict_nonoise_path)

    usable_statedict = OrderedDict()
    for k, v in nonoise_histo[-1].items():
        usable_statedict[k] = v
    model = lib.model.MnistModel()
    model.load_state_dict(usable_statedict)
    model = model.cuda()

    model.eval()

    # Plot MNIST
    # probas = []
    # acc = []

    # for data, target in test_loader:
    #     data = data.cuda()
    #     target = target.cuda()
    #     output = model(data)
    #     prediction = output.data.max(1)[1]
    #     proba = output.data.max(1)[0]
    #     probas.append(proba.cpu().numpy())
    #     acc.append(prediction.eq(target.data).cpu().numpy())

    # probas = np.hstack(probas)
    # acc = np.hstack(acc)

    # correct_probas = np.exp(probas[acc == 1])
    # incorrect_probas = np.exp(probas[acc == 0])

    # plt.hist(correct_probas, bins=20, density=True, alpha = .8, label='correct')
    # plt.hist(incorrect_probas, bins=20, density=True, alpha=.8, label='incorrect')
    # plt.xlabel('confidence in prediction')
    # plt.ylabel('normalized counts')
    # plt.legend()
    # plt.show()

    # Plot Not MNIST
    notmnist_probas = []
    for data in notmnist_loader:
        data = data.cuda()
        output = model(data)
        notmnist_probas.append(output.max(1)[0].cpu().data.numpy())

    notmnist_probas = np.hstack(notmnist_probas)
    notmnist_probas = np.exp(notmnist_probas)

    # plt.hist(notmnist_probas, bins=20, density=True, alpha = .8)
    # plt.xlabel('confidence in prediction')
    # plt.ylabel('normalized count')
    # plt.show()

    sns.kdeplot(data=notmnist_probas)
    plt.legend(labels=['SGD'])
    plt.show()

def plot_bayesian(path):
    statedict_noise_path = glob.glob(os.path.join(path, '*.accum.pt'))[0]
    withnoise_histo = torch.load(statedict_noise_path)

    bayes_probas_nm = []

    for i in np.arange(-1, -20, -1):
        model = lib.model.MnistModel()
        state_dict = make_statedict(i, withnoise_histo)
        model.load_state_dict(state_dict)
        model.cuda()
        model.eval()
        
        epoch_probas = np.zeros((18724, 10))

        for idx, data in enumerate(notmnist_loader):
            data = data.cuda()
            output = model(data)
            proba = output.data
            epoch_probas[idx * notmnist_loader.batch_size:(idx + 1) * notmnist_loader.batch_size, :] = proba.cpu().numpy()
            
        bayes_probas_nm.append(epoch_probas)

    bayes_probas_nm = np.stack(bayes_probas_nm)
    bayes_averaged_probas_nm = np.mean(bayes_probas_nm, axis=0)
    bayes_max_probas_nm = np.exp(bayes_averaged_probas_nm.max(axis = 1))
    sns.kdeplot(data=bayes_max_probas_nm)
    plt.legend(labels=[os.path.dirname(path)])
    plt.show()


plot_non_bayesian()
algors = []
for subdir, dirs, files in os.walk(model_dir):
    for dir in dirs:
        if not dir in ['SGD', '.ipynb_checkpoints']:
            algors.append(dir)

for algor in algors:
    plot_bayesian(os.path.join(model_dir, algor))

