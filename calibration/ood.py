import torch
import torchvision.transforms.functional as TF
import sys
import random
import numpy as np
from scipy.special import softmax
from torch import optim
from torch.nn import functional as F
from torch.distributions import Categorical
import argparse
import os
import glob
import math

import metrics
import visualization
import auc_mu

sys.path.insert(1, '../')
import lib.model
import lib.dataset
import model.resnet as resnet
import datetime

seed = 1
train_batch=200
test_batch=200

parser = argparse.ArgumentParser(
                    description="Evaluate performance of neural network on OOD")
parser.add_argument("-d", "--dir",
                    help="directory location containing pretrained models")
parser.add_argument("-m", "--model",
                    help="model architecture")
parser.add_argument("-o", "--optimizers", default="",
                    help="optimizer")
parser.add_argument("-n", "--nmodel", default=10,
                    help="number of models")
parser.add_argument("-ds", "--dataset",
                    help="dataset")

args = parser.parse_args()
dir_path = str(args.dir)
model_arch = str(args.model)
dataset = str(args.dataset)
optimizers = str(args.optimizers)
nmodel = int(args.nmodel)
nmodel_max = 10

optimizers = optimizers.split(",")

torch.cuda.set_device(0)
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

thresholds = [.0, .1,.2, .3, .4, .5, .6, .7, .8, .9, 1.]
samples = {} # key=optimizer, value=number of samples in threshold order
entropies = {} # key=optimizer, value=list of entropy

session_id_prefix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
stats_path = f'ood/stats_{session_id_prefix}.pt'
f = open(f'ood/stats_{session_id_prefix}.txt', 'w')
print(f'model: {model_arch}', file=f)
print(f'dataset: {dataset}', file=f)
print(f'path: {dir_path}', file=f)
print(f'statistics: {stats_path}', file=f)

for optimizer in optimizers:
    if dataset == "mnist":
        model = lib.model.MnistModel(output_logits=True)
        model = model.cuda()
        _, test_loader = lib.dataset.make_datasets(bs=train_batch, test_bs=test_batch)
    else:
        model = eval(model_arch)(output_logits=True)
        model = model.cuda()
        # _, test_loader = lib.dataset.make_datasets_cifar10(bs=train_batch, test_bs=test_batch)
        _, test_loader = eval(f'lib.dataset.make_datasets_{dataset}')(bs=train_batch, test_bs=test_batch)

    models = []
    path_idxs = [i for i in range(nmodel_max, nmodel_max-nmodel, -1)]

    # only use last checkpoint if SGD
    if optimizer == "SGD":
        path_idxs = [nmodel_max]
        nmodel = 1

    path_idxs = sorted(path_idxs)
    for path_idx in path_idxs:
        path_glob = dir_path + f"/*.{optimizer}_*_{path_idx}.pt"
        # print("pretrained path glob: ", path_glob)
        path = glob.glob(path_glob)[0]
        print("pretrained path: ", path)
        chk = torch.load(path)
        model.load_state_dict(chk['model_state_dict'])

        model.eval()
        models.append(model)

    pred_class_list = [] # list of class prediction
    pred_probs = [] # list of confidence value
    data_labels = [] # list of correct class
    loss_list = [] # list of per batch NLL loss
    entropies[optimizer] = [] # list of prediction entropy value
    correct = 0
    total = 0

    with torch.no_grad():
        for data, _ in test_loader:
            data = data.cuda()

            mean_pred_soft = torch.zeros(len(data), 10).cuda()
            for model in models:
                logits = model(data)
                prob_vecs = F.softmax(logits,dim=1) # (200, 10); (batch_size, num_class)
                mean_pred_soft += prob_vecs/float(nmodel) # (200, 10)

            entropies[optimizer].append(Categorical(probs = mean_pred_soft).entropy())
            pred_probs.append(mean_pred_soft.cpu())
            

            #total
            pred = mean_pred_soft.data.max(1)[1] 
            pred_class_list.append(pred.cpu())

        pred_probs_soft = torch.cat(pred_probs)

    ################
    #metrics on confidence thresholds

    samples[optimizer] = []

    for thres in thresholds:
        correct = 0
        total = 0
        thres_pred_probs = []
        thres_labels = []
        for (pred_prob, pred_class) in zip(pred_probs,pred_class_list):
            class_pred_prob = pred_prob.data.max(1)[0]
            # find tensor indices where class predictive confidence is above threshold
            # https://stackoverflow.com/a/57570139/2998271
            mask = class_pred_prob >= thres
            indices = torch.nonzero(mask)

            # update counter using valid indices only:
            total += pred_class[indices].size(0)

            thres_pred_probs.append(pred_prob[indices])

        # save number of samples:
        samples[optimizer].append(total)

torch.save({
            'thresholds': thresholds,
            'entropies': entropies,
            'samples': samples
        }, stats_path)

f.close()