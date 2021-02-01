import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import lib.dataset
import lib.model
import lib.evaluation
import lib.sgd as sgd
import lib.sgld as sgld
import lib.ekfac_precond
import lib.kfac_precond
import lib.asgld as asgld
import pysgmcmc.optimizers.sgld as pysgmcmc_sgld
import argparse
import numpy as np
import time
import yaml
import datetime
from torch.utils.tensorboard import SummaryWriter

default_yaml =  "config.yaml"
default_silent = False
save_model_path = "/content/kfac-backpack"
session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

parser = argparse.ArgumentParser(
                    description="Trains and saves neural network for "
                                "MNIST classification.")
parser.add_argument("-y", "--yaml",
                    help="yaml config file location",
                    default=default_yaml)
parser.add_argument("-s", "--silent",
                    help="if True, do not print per epoch accuracy",
                    default=default_silent)

args = parser.parse_args()
yaml_path = str(args.yaml)
silent = bool(args.silent)

with open('config.yaml') as f:
    config = yaml.load(f, Loader=yaml.Loader)

seed = config['seed']
epochs = config['epoch']
optimizer_name = config['optimizer']

dataset_params = config['dataset']
train_batch = dataset_params['train_batch']
test_batch = dataset_params['test_batch']

torch.cuda.set_device(0)
torch.manual_seed(seed)
random.seed(seed)
model = lib.model.MnistModel()
train_loader, test_loader = lib.dataset.make_datasets(bs=train_batch, test_bs=test_batch)
model = model.cuda()

optim_params = {}
if optimizer_name in config:
    optim_params2 = config[optimizer_name]
    for k in optim_params2:
        v = optim_params2[k]
        if v:
            optim_params[k] = v

print('optimizer params: ', optim_params)
optimizer = eval(optimizer_name)(model.parameters(), **optim_params)

writer = SummaryWriter()

step = 0
for epoch in range(epochs):
    t0 = time.time()
    model.train()
    for data, target in train_loader:
        step += 1
        data = data.cuda()
        target = target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        prediction = output.data.max(1)[1]   # first column has actual prob.
        accuracy = np.mean(prediction.eq(target.data).cpu().numpy())*100

    # measure training time
    elapsed = time.time() - t0

    # validate
    val_accuracy, _ = lib.evaluation.evaluate(model, test_loader)
    writer.add_scalar("Loss/train", np.mean(loss.item()), epoch)
    writer.add_scalar("Acc/train", val_accuracy, epoch)
    writer.add_scalar("TAcc/train", np.mean(accuracy), epoch)

    if not silent:
        print('Epoch: {}\tTrain Sec: {:0.3f}\tLoss: {:.3f}\tAcc: {:.3f}\tVal Acc: {:.3f}'
                .format(epoch, elapsed, np.mean(loss.item()), np.mean(accuracy), val_accuracy))

# Save the model weights.
torch.save(model.state_dict(), save_model_path + "/" + session_id+".pt")
writer.flush()