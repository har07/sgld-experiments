import torch
import sys
import random
import numpy as np
from scipy.special import softmax
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
# trained='optim.SGD_20210829_071116.pt'
# PATH = '/content/drive/MyDrive/Tesis/langevin/azureml_20210829/result_cifar10/' + trained
# trained='optim.SGD_20210829_131734.pt'
# PATH = '/content/drive/MyDrive/Tesis/langevin/azureml_20210829/result_mnist/' + trained

parser = argparse.ArgumentParser(
                    description="Evaluate confidence calibration of neural network for ")
parser.add_argument("-d", "--dir",
                    help="directory location containing pretrained models")
parser.add_argument("-m", "--model",
                    help="model architecture")

args = parser.parse_args()
dir_path = str(args.dir)
model_arch = str(args.model)

# torch.cuda.set_device(0)
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

if model_arch == "mnist":
    model = lib.model.MnistModel(output_logits=True)
    _, test_loader = lib.dataset.make_datasets(bs=train_batch, test_bs=test_batch)
else:
    model = resnet.ResNet34(output_logits=True)
    _, test_loader = lib.dataset.make_datasets_cifar10(bs=train_batch, test_bs=test_batch)

paths = glob.glob(dir_path + "/*")

for path in paths:
    file_name = os.path.basename(path)
    # this assume pretrained file name follows this format: optim.SGD_20210829_071116.pt
    optimizer = file_name.split(".")[1].split("_")[0]

    print("pretrained path: ", path)
    chk = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(chk['model_state_dict'])

    model.eval()
    accuracies = []
    logits_list = []
    labels_list = []
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data = data
            labels = target
            logits = model(data)

            logits_list.append(logits)
            labels_list.append(labels)

            output_probs = F.softmax(logits,dim=1)
            prediction = output_probs.data.max(1)[1]   # first column has actual prob.
            val_accuracy = np.mean(prediction.eq(target.data).numpy())*100
            
            accuracies.append(val_accuracy)

            #total
            total += labels.size(0)
            correct += (prediction == labels).sum().item()

        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)

    print("Calculate calibration for network trained using ", optimizer)
    print('Accuracy of the network on the test images: %d %%' % (
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

    ############
    #visualizations

    conf_hist = visualization.ConfidenceHistogram()
    plt_test = conf_hist.plot(logits_np,labels_np,title="Confidence Histogram")
    plt_test.savefig('plots/%s_conf_histogram_test.png' % optimizer, bbox_inches='tight')
    #plt_test.show()

    rel_diagram = visualization.ReliabilityDiagram()
    plt_test_2 = rel_diagram.plot(logits_np,labels_np,title="Reliability Diagram")
    plt_test_2.savefig('plots/%s_rel_diagram_test.png' % optimizer, bbox_inches='tight')
    #plt_test_2.show()