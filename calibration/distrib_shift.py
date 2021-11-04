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
                    description="Evaluate confidence calibration of neural network for ")
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

rotations = [0, 30, 60, 90, 120, 150, 180]
saved_accuracies = {} # key=optimizer, value=array of accuracies in rotations order
saved_labels = {} # key=optimizer, value=array of labels in rotations order
saved_pred_probs = {} # key=optimizer, value=array of predictive probabilities in rotations order
saved_nll = {} # key=optimizer, value=array of NLL in rotations order

session_id_prefix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
stats_path = f'distrib_shift/stats_{session_id_prefix}.pt'
f = open(f'distrib_shift/stats_{session_id_prefix}.txt', 'w')
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
        _, test_loader = lib.dataset.make_datasets_cifar10(bs=train_batch, test_bs=test_batch)

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

    saved_accuracies[optimizer] = [] # list of accuracies
    saved_labels[optimizer] = []
    saved_pred_probs[optimizer] = []
    saved_nll[optimizer] = []

    for rotate in rotations:
        pred_class_list = [] # list of class prediction
        pred_probs = [] # list of confidence value
        data_labels = [] # list of correct class
        loss_list = [] # list of per batch NLL loss
        entropy_list = [] # list of prediction entropy value
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data = data.cuda()
                target = target.cuda()

                if rotate > 0:
                    # use torchvision transform instead, much simpler: 
                    # https://pytorch.org/vision/stable/transforms.html#functional-transforms
                    data = TF.rotate(data, rotate)

                mean_pred_soft = torch.zeros(len(data), 10).cuda()
                mean_log_soft = torch.zeros(len(data), 10).cuda()
                for model in models:
                    logits = model(data)
                    prob_vecs = F.softmax(logits,dim=1) # (200, 10); (batch_size, num_class)
                    mean_pred_soft += prob_vecs/float(nmodel) # (200, 10)
                    mean_log_soft += F.log_softmax(logits,dim=1)/float(nmodel)

                entropy_list.append(Categorical(probs = mean_pred_soft).entropy())
                pred_probs.append(mean_pred_soft.cpu())
                data_labels.append(target.cpu())
                loss = F.nll_loss(mean_log_soft, target)
                loss_list.append(loss.cpu().numpy())
                

                #total
                pred = mean_pred_soft.data.max(1)[1] 
                pred_class_list.append(pred.cpu())
                total += target.size(0)
                correct += (pred == target).sum().item()

            pred_probs_soft = torch.cat(pred_probs)
            target = torch.cat(data_labels)

        print(f"Calculate calibration for network trained using {optimizer} {nmodel} models")
        val_acc = 100 * correct / total
        print(f"Accuracy of the network on the test images rotated {rotate} degree: {val_acc:.2f}")
        print(total)
        saved_accuracies[optimizer].append(val_acc)

        saved_pred_probs[optimizer].append(pred_probs_soft.numpy())
        saved_labels[optimizer].append(target.numpy())
        saved_nll[optimizer].append(loss_list)

torch.save({
            'rotations': rotations,
            'accuracies': saved_accuracies,
            'labels': saved_labels,
            'pred_probs': saved_pred_probs,
            'nll': saved_nll
        }, stats_path)

f.close()