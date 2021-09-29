import torch
import sys
import random
import numpy as np
from scipy.special import softmax
from torch.nn import functional as F

import metrics

sys.path.insert(1, '../lib/')
import lib.model
import lib.dataset

seed = 1
train_batch=200
test_batch=200
PATH = '/content/drive/MyDrive/Tesis/langevin/azureml_20210829/result_cifar10'

torch.cuda.set_device(0)
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

model = lib.model.MnistModel()
chk = torch.load(PATH)
model.load_state_dict(chk['model_state_dict'])

_, test_loader = lib.dataset.make_datasets(bs=train_batch, test_bs=test_batch)

model.eval()
accuracies = []
logits_list = []
labels_list = []
correct = 0
total = 0

with torch.no_grad():
    for data, target in test_loader:
        data = data.cuda()
        labels = target.cuda()
        logits = model(data)

        logits_list.append(logits)
        labels_list.append(labels)

        output_probs = F.softmax(logits,dim=1)
        prediction = output_probs.data.max(1)[1]   # first column has actual prob.
        val_accuracy = np.mean(prediction.eq(target.data).cpu().numpy())*100
        
        accuracies.append(val_accuracy)

        #total
        total += labels.size(0)
        correct += (prediction == labels).sum().item()

    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
print(total)

ece_criterion = metrics.ECELoss()
#Torch version
logits_np = logits.numpy()
labels_np = labels.numpy()

################
#metrics

ece_criterion = metrics.ECELoss()
#Torch version
logits_np = logits.numpy()
labels_np = labels.numpy()

#Numpy Version
print('ECE: %f' % (ece_criterion.loss(logits_np,labels_np, 15)))

softmaxes = softmax(logits_np, axis=1)

print('ECE with probabilties %f' % (ece_criterion.loss(softmaxes,labels_np,15,False)))

mce_criterion = metrics.MCELoss()
print('MCE: %f' % (mce_criterion.loss(logits_np,labels_np)))