# adapted from: https://github.com/fregu856/evaluating_bdl/blob/master/toyClassification/Ensemble-MAP-Adam/train.py

import torch
import torch.nn.functional as F
from lib.dataset import ToyDataset
from lib.model import ToyNet
import lib.lr_setter as lr_setter
import lib.psgld2 as psgld2
import lib.sgld3 as sgld3
import lib.ekfac_precond as ekfac
import lib.kfac_precond as kfac
import lib.asgld as asgld
import lib.ksgld as ksgld
import lib.eksgld as eksgld
import argparse
import numpy as np
import datetime
import time
import yaml
import torch.optim as optim
import inspect
import shutil

from torch.utils.tensorboard import SummaryWriter

dataset_dir = "./dataset"
default_yaml =  "config/config_ensemble.yaml"
default_silent = False
default_none = "None"
save_model_path = "/content/sgld-experiments"
session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

parser = argparse.ArgumentParser(
                    description="Trains and saves neural model for "
                                "Toy dataset classification.")

parser.add_argument("-y", "--yaml",
                    help="yaml config file location",
                    default=default_yaml)
parser.add_argument("-s", "--silent",
                    help="if True, do not print per epoch accuracy",
                    default=default_silent)
parser.add_argument("-l", "--logdir",
                    help="Log directory",
                    default=default_none)
parser.add_argument("-c", "--checkpoint",
                    help="Checkpoint file",
                    default=default_none)
parser.add_argument("-p", "--project_dir",
                    help="Project directory path")

args = parser.parse_args()
yaml_path = str(args.yaml)
silent = bool(args.silent)
logdir = str(args.logdir)
checkpoint = str(args.checkpoint)
project_dir = str(args.project_dir)

with open(yaml_path) as f:
    config = yaml.load(f, Loader=yaml.Loader)

model_id = config['model_id']
num_epochs = config['epoch']
optimizer_name = config['optimizer']
M = config['M']
poly_decay = config['poly_decay']
block_size = config['block_size']
block_decay = config['block_decay']
optimizer_name = config['optimizer']

batch_size = 32
alpha = 1.0

train_dataset = ToyDataset(dataset_dir)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

N = float(len(train_dataset))
num_train_batches = int(len(train_dataset)/batch_size)

if logdir != default_none:
    writer = SummaryWriter(log_dir=logdir)
else:
    writer = SummaryWriter()

optim_params = {}
if optimizer_name in config:
    optim_params2 = config[optimizer_name]
    for k in optim_params2:
        v = optim_params2[k]
        if v or v == False:
            optim_params[k] = v

print('optimizer: ', optimizer_name)
print('optimizer params: ', optim_params)

def lr_poly(start_lr, end_lr, step, decay_steps, power):
  step = min(step, decay_steps)
  return ((start_lr - end_lr) *
            (1 - step / decay_steps) ** (power)
           ) + end_lr

for i in range(M):
    network = ToyNet(model_id + "_%d" % i, project_dir=project_dir).cuda()

    if config['accept_model']:
        optimizer = eval(optimizer_name)(network, **optim_params)
    else:
        optimizer = eval(optimizer_name)(network.parameters(), **optim_params)

    # check if optimizer.step has 'lr' param
    step_args = inspect.signature(optimizer.step)
    lr_param = 'lr' in step_args.parameters

    step = 0
    current_lr = optim_params["lr"]
    min_lr = 1.e-256

    epoch_losses_train = []
    for epoch in range(num_epochs):

        t0 = time.time()
        network.train() # (set in training mode, this affects BatchNorm and dropout)
        batch_losses = []
        for step, (x, y) in enumerate(train_loader):
            x = x.cuda() # (shape: (batch_size, 2))
            y = y.cuda() # (shape: (batch_size, ))

            logits = network(x) # (shape: (batch_size, num_classes)) (num_classes==2)

            ####################################################################
            # compute the loss:
            ####################################################################
            loss_likelihood = F.cross_entropy(logits, y)

            loss_prior = 0.0
            for param in network.parameters():
                if param.requires_grad:
                    loss_prior += (1.0/2.0)*(1.0/N)*(1.0/alpha)*torch.sum(torch.pow(param, 2))

            loss = loss_likelihood + loss_prior

            loss_value = loss.data.cpu().numpy()
            batch_losses.append(loss_value)

            ########################################################################
            # optimization step:
            ########################################################################
            optimizer.zero_grad() # (reset gradients)
            loss.backward() # (compute gradients)
            optimizer.step() # (perform optimization step)

        elapsed = time.time() - t0

        epoch_loss = np.mean(batch_losses)
        epoch_losses_train.append(epoch_loss)

        # update learning rate for next epoch
        should_update_lr = False
        if poly_decay > 0:
            should_update_lr = True
            current_lr = lr_poly(optim_params["lr"], min_lr, epoch, num_epochs, poly_decay)
        elif block_size > 0 and block_decay > 0 and ((epoch) % block_size) == 0:
            should_update_lr = True
            current_lr = current_lr * block_decay
        if should_update_lr and not lr_param:
            optimizer = lr_setter.update_lr(optimizer, current_lr)

        writer.add_scalar("Loss/train", epoch_loss, epoch*(i+1))
        writer.add_scalar("Duration", elapsed, epoch*(i+1))
        writer.add_scalar("Learning Rate", current_lr, epoch*(i+1))

        # save the model weights to disk:
        # only the last epoch will be used in evaluation:
        if epoch+1 == num_epochs:
            checkpoint_path = network.checkpoints_dir + "/model_" + model_id +"_epoch_" + str(epoch+1) + ".pth"
            torch.save(network.state_dict(), checkpoint_path)

    shutil.copy2(yaml_path, network.checkpoints_dir + "/..")

writer.flush()