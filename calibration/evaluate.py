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

args = parser.parse_args()
dir_path = str(args.dir)
model_arch = str(args.model)
optimizer = str(args.optimizer)

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

sub_path = "/*"
if optimizer != "":
    sub_path = f"/*.{optimizer}_*"
if optimizer == "SGD1":
    sub_path = f"/*.SGD_*_10.pt" # only get the final model
paths = glob.glob(dir_path + sub_path)
print("path pattern: ", dir_path + sub_path)
print("paths: ", paths)

models = []
mean_pred = np.zeros((2,))
for path in paths:
    # print("pretrained path: ", path)
    chk = torch.load(path)
    model.load_state_dict(chk['model_state_dict'])

    model.eval()
    models.append(model)

accuracies = []
pred_probs_list = []
pred_probs_list_soft = []
labels_list = []
correct = 0
total = 0

with torch.no_grad():
    for data, target in test_loader:
        data = data.cuda()
        labels = target.cuda()

        mean_pred = torch.zeros(len(data), 10).cuda()
        mean_pred_soft = torch.zeros(len(data), 10).cuda()
        for model in models:
            logits = model(data)
            prob_vecs = F.softmax(logits,dim=1) # (200, 10); (batch_size, num_class)
            mean_pred += logits/float(len(paths)) # (200, 10)
            mean_pred_soft += prob_vecs/float(len(paths)) # (200, 10)

        pred_probs_list.append(mean_pred.cpu())
        pred_probs_list_soft.append(mean_pred_soft.cpu())
        labels_list.append(labels.cpu())
        
        # pred_probs = mean_pred.data.max(1)[0] # (200); prob score of predicted class
        # pred = mean_pred.data.max(1)[1] # (200); predicted class index

        # pred_probs_list.append(pred_probs.cpu())
        # labels_list.append(labels.cpu())
        # val_accuracy = np.mean(pred.eq(labels.data).cpu().numpy())*100
        
        # accuracies.append(val_accuracy)

        #total
        pred = mean_pred_soft.data.max(1)[1] 
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    pred_probs = torch.cat(pred_probs_list)
    pred_probs_soft = torch.cat(pred_probs_list_soft)
    labels = torch.cat(labels_list)

print("Calculate calibration for network trained using ", optimizer)
print(f"Accuracy of the network on the test images: {100 * correct / total:.2f}")
print(total)

################
#metrics

ece_criterion = metrics.ECELoss()

pred_probs_np = pred_probs.numpy()
pred_probs_soft_np = pred_probs_soft.numpy()
labels_np = labels.numpy()

# print('pred_probs_np len: ', len(pred_probs_np))
# print('labels_np len:', len(labels_np))
# print('ECE: %f' % (ece_criterion.loss(pred_probs_np,labels_np, 15)))
ece_score = ece_criterion.loss(pred_probs_soft_np,labels_np, 15, logits=False)
print('ECE Softmax: %f' % (ece_score))

mce_criterion = metrics.MCELoss()
# print('MCE: %f' % (mce_criterion.loss(pred_probs_np,labels_np)))
mce_score = mce_criterion.loss(pred_probs_soft_np,labels_np, logits=False)
print('MCE Softmax: %f' % (mce_score))

with open(f"plots/{optimizer}_metrics.txt", 'w') as f:
    f.write(f"ECE Softmax: {ece_score}")
    f.write(f"MCE Softmax: {mce_score}")

############
#visualizations

conf_hist = visualization.ConfidenceHistogram()
plt_test = conf_hist.plot(pred_probs_np,labels_np,title="Confidence Histogram")
plt_test.savefig('plots/%s_conf_histogram_test.png' % optimizer, bbox_inches='tight')
plt_test_soft = conf_hist.plot(pred_probs_soft_np,labels_np,title="Confidence Histogram Softmax",logits=False)
plt_test_soft.savefig('plots/%s_conf_histogram_test_soft.png' % optimizer, bbox_inches='tight')
#plt_test.show()

rel_diagram = visualization.ReliabilityDiagram()
plt_test_2 = rel_diagram.plot(pred_probs_np,labels_np,title="Reliability Diagram")
plt_test_2.savefig('plots/%s_rel_diagram_test.png' % optimizer, bbox_inches='tight')
plt_test_2_soft = rel_diagram.plot(pred_probs_soft_np,labels_np,title="Reliability Diagram Softmax",logits=False)
plt_test_2_soft.savefig('plots/%s_rel_diagram_test_soft.png' % optimizer, bbox_inches='tight')
#plt_test_2.show()

# #Torch version
# logits_np = logits.cpu().numpy()
# labels_np = labels.cpu().numpy()

# #Numpy Version
# print('ECE: %f' % (ece_criterion.loss(logits_np,labels_np, 15)))

# softmaxes = softmax(logits_np, axis=1)

# print('ECE with probabilties %f' % (ece_criterion.loss(softmaxes,labels_np,15,False)))

# mce_criterion = metrics.MCELoss()
# print('MCE: %f' % (mce_criterion.loss(logits_np,labels_np)))

# ############
# #visualizations

# conf_hist = visualization.ConfidenceHistogram()
# plt_test = conf_hist.plot(logits_np,labels_np,title="Confidence Histogram")
# plt_test.savefig('plots/%s_conf_histogram_test.png' % optimizer, bbox_inches='tight')
# #plt_test.show()

# rel_diagram = visualization.ReliabilityDiagram()
# plt_test_2 = rel_diagram.plot(logits_np,labels_np,title="Reliability Diagram")
# plt_test_2.savefig('plots/%s_rel_diagram_test.png' % optimizer, bbox_inches='tight')
# #plt_test_2.show()