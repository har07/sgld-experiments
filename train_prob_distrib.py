# adapted from: https://github.com/fregu856/evaluating_bdl/blob/master/toyClassification/SGLD-64/train.py

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
import yaml
import datetime
import time
import inspect

from torch.utils.tensorboard import SummaryWriter

# example command to run:
# !python train_prob_distrib.py -p /content/sgld-experiments

dataset_dir = "./dataset"
default_yaml =  "config_prob_distrib.yaml"
default_silent = False
default_save_burnin = True
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
parser.add_argument("-sb", "--save-burnin",
                    help="if False, only save models for the last 75% iterations. Default True",
                    default=default_save_burnin)

args = parser.parse_args()
yaml_path = str(args.yaml)
silent = bool(args.silent)
logdir = str(args.logdir)
checkpoint = str(args.checkpoint)
project_dir = str(args.project_dir)
save_burnin = bool(args.save_burnin)

with open(yaml_path) as f:
    config = yaml.load(f, Loader=yaml.Loader)

seed = config['seed']
M = config['M']
epochs = config['epoch']
block_size = config['block_size']
block_decay = config['block_decay']
optimizer_name = config['optimizer']
precond_name = config['preconditioner']

batch_size = 32
optimizer_id = optimizer_name.split('.')[1]

optim_params = {}
if optimizer_name in config:
    optim_params2 = config[optimizer_name]
    for k in optim_params2:
        v = optim_params2[k]
        if v or v == False:
            optim_params[k] = v

precond_params = {}
precond = None
if precond_name in config:
    precond_params2 = config[precond_name]
    for k in precond_params2:
        v = precond_params2[k]
        if v or v == False:
            precond_params[k] = v

if logdir != default_none:
    writer = SummaryWriter(log_dir=logdir)
else:
    writer = SummaryWriter()

for i in range(M):
    model_id = f"{optimizer_id}-{epochs}_{i+1}"
    model = ToyNet(model_id, project_dir=project_dir).cuda()
    train_dataset = ToyDataset(dataset_dir)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    num_train_batches = int(len(train_dataset)/batch_size)
    num_steps = epochs*num_train_batches + 1
    num_epochs_low = int(0.75*epochs)

    # print('optimizer: ', optimizer_name)
    # print('optimizer params: ', optim_params)
    if config['accept_model']:
        optimizer = eval(optimizer_name)(model, **optim_params)
    else:
        optimizer = eval(optimizer_name)(model.parameters(), **optim_params)

    # check if optimizer.step has 'lr' param
    step_args = inspect.signature(optimizer.step)
    lr_param = 'lr' in step_args.parameters


    if precond_name != '' and precond_name.lower() != 'none':
        precond = eval(precond_name)(model, **precond_params)
        # print('preconditioner: ', precond_name)
        # print('preconditioner params: ', precond_params)

    step = 0
    current_lr = optim_params["lr"]

    epoch_losses_train = []
    for epoch in range(epochs):
        model.train() # (set in training mode, this affects BatchNorm and dropout)
        batch_losses = []

        t0 = time.time()
        for data, target in train_loader:
            data = data.cuda()
            target = target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            
            loss_value = loss.data.cpu().numpy()
            batch_losses.append(loss_value)

            loss.backward()
            if precond:
                precond.step()
            if block_size > 0 and block_decay > 0 and lr_param:
                optimizer.step(lr=current_lr)
            else:
                optimizer.step()

        elapsed = time.time() - t0
        epoch_loss = np.mean(batch_losses)
        epoch_losses_train.append(epoch_loss)

        # update learning rate for next epoch
        if block_size > 0 and block_decay > 0 and ((epoch) % block_size) == 0:
            current_lr = current_lr * block_decay
            if not lr_param:
                optimizer = lr_setter.update_lr(optimizer, current_lr)

        writer.add_scalar("Loss/train", epoch_loss, epoch*(i+1))
        writer.add_scalar("Duration", elapsed, epoch*(i+1))

        # if `save_burnin=False`, always save model in every epoch.
        # More expensive but we can compare small vs large number of epochs 
        # after training once for large epochs.
        # Otherwise if `save_burnin=True`, only save models if current epoch
        # is >= 75% of total number of epochs (only save models from the last 25% epochs)
        if save_burnin or epoch+1 >= num_epochs_low:
            checkpoint_path = model.checkpoints_dir + "/model_" + model_id +"_epoch_" + str(epoch+1) + ".pth"
            torch.save(model.state_dict(), checkpoint_path)

    writer.flush()