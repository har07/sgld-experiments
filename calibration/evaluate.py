import torch
import sys
import random
import numpy as np
from scipy.special import softmax
from torch import optim
from torch.nn import functional as F
import argparse
import os
import glob

import metrics
import visualization

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

args = parser.parse_args()
dir_path = str(args.dir)
model_arch = str(args.model)
optimizer = str(args.optimizer)
nmodel = int(args.nmodel)
nmodel_max = 10

torch.cuda.set_device(0)
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

if model_arch == "mnist":
    model = lib.model.MnistModel(output_logits=True)
    model = model.cuda()
    _, test_loader = lib.dataset.make_datasets(bs=train_batch, test_bs=test_batch)
else:
    model = resnet.ResNet34(output_logits=True)
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
pred_class_list = []
pred_probs = []
data_labels = []
loss_list = []
correct = 0
total = 0

with torch.no_grad():
    for data, target in test_loader:
        data = data.cuda()
        target = target.cuda()

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
print(f"Accuracy of the network on the test images: {val_acc:.2f}")
print(total)

################
#metrics

ece_criterion = metrics.ECELoss()

pred_probs_soft_np = pred_probs_soft.numpy()
labels_np = target.numpy()

ece_score = ece_criterion.loss(pred_probs_soft_np,labels_np, 15, logits=False)
print('ECE Softmax: %f' % (ece_score))

mce_criterion = metrics.MCELoss()
mce_score = mce_criterion.loss(pred_probs_soft_np,labels_np, logits=False)
print('MCE Softmax: %f' % (mce_score))

nll = np.mean(loss_list)
print(f"NLL: {nll}")

with open(f"plots/{optimizer}_metrics_{nmodel}models.txt", 'w') as f:
    f.write(f"ECE Softmax: {ece_score}")
    f.write(f"MCE Softmax: {mce_score}")
    f.write(f"NLL: {nll}\n")
    f.write(f"Accuracy: {val_acc}\n")

############
#visualizations

conf_hist = visualization.ConfidenceHistogram()
plt_test_soft = conf_hist.plot(pred_probs_soft_np,labels_np,title="Confidence Histogram",logits=False)
plt_test_soft.savefig(f"plots/{optimizer}_conf_histogram_{nmodel}models.png", bbox_inches='tight')
plt_test_soft.show()

rel_diagram = visualization.ReliabilityDiagram()
plt_test_2_soft = rel_diagram.plot(pred_probs_soft_np,labels_np,title="Reliability Diagram",logits=False)
plt_test_2_soft.savefig(f"plots/{optimizer}_rel_diagram_{nmodel}models.png", bbox_inches='tight')
plt_test_2_soft.show()
