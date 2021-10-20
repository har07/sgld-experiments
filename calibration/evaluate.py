import torch
import torchvision.transforms.functional as TF
import sys
import random
import numpy as np
from scipy.special import softmax
from torch import optim
from torch.nn import functional as F
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

seed = 1
train_batch=200
test_batch=200

parser = argparse.ArgumentParser(
                    description="Evaluate confidence calibration of neural network for ")
parser.add_argument("-d", "--dir",
                    help="directory location containing pretrained models")
parser.add_argument("-m", "--model",
                    help="model architecture")
parser.add_argument("-o", "--optimizer", default="",
                    help="optimizer")
parser.add_argument("-n", "--nmodel", default=10,
                    help="number of models")
parser.add_argument("-ds", "--dataset",
                    help="dataset")
parser.add_argument("-r", "--rotate", default=0,
                    help="rotate data")

args = parser.parse_args()
dir_path = str(args.dir)
model_arch = str(args.model)
dataset = str(args.dataset)
optimizer = str(args.optimizer)
nmodel = int(args.nmodel)
rotate = int(args.rotate)
nmodel_max = 10

torch.cuda.set_device(0)
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

if dataset == "mnist":
    model = lib.model.MnistModel(output_logits=True)
    model = model.cuda()
    _, test_loader = lib.dataset.make_datasets(bs=train_batch, test_bs=test_batch)
else:
    # model = resnet.ResNet34(output_logits=True)
    # model = resnet.ResNet18(output_logits=True)
    model = eval(model_arch)(output_logits=True)
    model = model.cuda()
    _, test_loader = lib.dataset.make_datasets_cifar10(bs=train_batch, test_bs=test_batch)

models = []
path_idxs = [i for i in range(nmodel_max, nmodel_max-nmodel, -1)]
# path_idxs = sorted(path_idxs, key=str)
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

accuracies = []
pred_class_list = [] # list of class prediction
pred_probs = [] # list of confidence value
data_labels = [] # list of correct class
loss_list = [] # list of per batch NLL loss
correct = 0
total = 0

with torch.no_grad():
    for data, target in test_loader:
        data = data.cuda()
        target = target.cuda()

        if rotate > 0:
            # rotation_matrix = torch.Tensor([[[math.cos(rotate/360.0*2*math.pi), -math.sin(rotate/360.0*2*math.pi), 0],
            #                         [math.sin(rotate/360.0*2*math.pi), math.cos(rotate/360.0*2*math.pi), 0]]]).cuda()
            # grid = F.affine_grid(rotation_matrix, data.size())
            # data = F.grid_sample(data, grid)

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

        pred_probs.append(mean_pred_soft.cpu())
        data_labels.append(target.cpu())
        loss = F.nll_loss(mean_log_soft, target)
        loss_list.append(loss.cpu().numpy())
        

        #total
        pred = mean_pred_soft.data.max(1)[1] 
        pred_class_list.append(pred.cpu().numpy())
        total += target.size(0)
        correct += (pred == target).sum().item()

    pred_probs_soft = torch.cat(pred_probs)
    target = torch.cat(data_labels)

print(f"Calculate calibration for network trained using {optimizer} {nmodel} models")
val_acc = 100 * correct / total
print(f"Accuracy of the network on the test images rotated {rotate} degree: {val_acc:.2f}")
print(total)

################
#metrics

ece_criterion = metrics.ECELoss()

pred_probs_soft_np = pred_probs_soft.numpy()
labels_np = target.numpy()

auc_mu_score = auc_mu.auc_mu(labels_np, pred_probs_soft_np)
print('AUCmu: %f' % (auc_mu_score))

ece_score = ece_criterion.loss(pred_probs_soft_np,labels_np, 15, logits=False)
print('ECE Softmax: %f' % (ece_score))

mce_criterion = metrics.MCELoss()
mce_score = mce_criterion.loss(pred_probs_soft_np,labels_np, logits=False)
print('MCE Softmax: %f' % (mce_score))

nll = np.mean(loss_list)
print(f"NLL: {nll}")

with open(f"plots/{model_arch}_{optimizer}_metrics_{nmodel}models.txt", 'w') as f:
    f.write(f"{optimizer} {nmodel} models:\n")
    f.write(f"AUCmu: {auc_mu_score}\n")
    f.write(f"ECE Softmax: {ece_score}\n")
    f.write(f"MCE Softmax: {mce_score}\n")
    f.write(f"NLL: {nll}\n")
    f.write(f"Accuracy: {val_acc}\n")

############
#visualizations

conf_hist = visualization.ConfidenceHistogram()
plt_test_soft = conf_hist.plot(pred_probs_soft_np,labels_np,title=f"",logits=False)
plt_test_soft.savefig(f"plots/{model_arch}_{optimizer}_conf_histogram_{nmodel}models.png", bbox_inches='tight')
plt_test_soft.show()

rel_diagram = visualization.ReliabilityDiagram()
plt_test_2_soft = rel_diagram.plot(pred_probs_soft_np,labels_np,title=f"ECE={ece_score}",logits=False)
plt_test_2_soft.savefig(f"plots/{model_arch}_{optimizer}_rel_diagram_{nmodel}models.png", bbox_inches='tight')
plt_test_2_soft.show()
