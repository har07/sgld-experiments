# adapted from: https://github.com/fregu856/evaluating_bdl/blob/master/toyClassification/SGLD-64/train.py

import torch
import torch.optim as optim
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
import lib.ekfac_precond as ekfac
import lib.kfac_precond as kfac
import argparse
import numpy as np
import yaml
import datetime
import time
import shutil
import inspect
import math
import random

from torch.utils.tensorboard import SummaryWriter

# example command to run:
# !python train_prob_distrib.py -p /content/sgld-experiments

dataset_dir = "./dataset"
default_yaml =  "config/config_prob_distrib.yaml"
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
burnin = config['burnin']
optimizer_name = config['optimizer']
precond_name = config['preconditioner']
lr_schedule_name = config['lr_schedule']
use_prior = config['use_prior']
skip_noise = config['skip_noise']

torch.cuda.set_device(0)
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

batch_size = 32
optimizer_id = optimizer_name.split('.')[1]

optim_params = {}
if optimizer_name in config:
    optim_params2 = config[optimizer_name]
    for k in optim_params2:
        v = optim_params2[k]
        if v or v == False:
            optim_params[k] = v

lr_schedule_params = {}
if lr_schedule_name in config:
    lr_schedule_params2 = config[lr_schedule_name]
    for k in lr_schedule_params2:
        v = lr_schedule_params2[k]
        if v or v == False:
            lr_schedule_params[k] = v

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

def loss_prior(network, loss_likelihood, lr, N, with_noise=True, alpha=1.0):
    loss_prior = 0.0
    for param in network.parameters():
        if param.requires_grad:
            loss_prior += (1.0/2.0)*(1.0/N)*(1.0/alpha)*torch.sum(torch.pow(param, 2))

    if skip_noise != 'None' and skip_noise:
        with_noise = False
    if with_noise:
        loss_noise = 0.0
        for param in network.parameters():
            if param.requires_grad:
                loss_noise += (1.0/math.sqrt(N))*math.sqrt(2.0/lr)*torch.sum(param*torch.normal(torch.zeros(param.size()), std=1.0).cuda())

        loss = loss_likelihood + loss_prior + loss_noise
    else:
        loss = loss_likelihood + loss_prior

    return loss


for i in range(M):
    model_id = f"{optimizer_id}-{epochs}_{i+1}"
    model = ToyNet(model_id, project_dir=project_dir).cuda()
    train_dataset = ToyDataset(dataset_dir)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    num_train_batches = int(len(train_dataset)/batch_size)
    num_steps = epochs*num_train_batches + 1
    num_epochs_low = int(burnin*epochs)

    # print('optimizer: ', optimizer_name)
    # print('optimizer params: ', optim_params)
    if config['accept_model']:
        optimizer = eval(optimizer_name)(model, **optim_params)
    else:
        optimizer = eval(optimizer_name)(model.parameters(), **optim_params)

    if precond_name != '' and precond_name.lower() != 'none':
        precond = eval(precond_name)(model, **precond_params)

    # check if optimizer.step has 'lr' param
    step_args = inspect.signature(optimizer.step)
    lr_param = 'lr' in step_args.parameters

    step = 0
    current_lr = optim_params["lr"]
    min_lr = 1.e-256

    epoch_losses_train = []
    for epoch in range(epochs):
        model.train() # (set in training mode, this affects BatchNorm and dropout)
        batch_losses = []

        t0 = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.cuda()
            target = target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            
            loss_likelihood = loss.data.cpu().numpy()
            batch_losses.append(loss_likelihood)

            if use_prior:
                alpha = config['prior_alpha']
                add_noise = False
                if optimizer_name[:6] == "optim.":
                    add_noise = True
                loss = loss_prior(model, loss, current_lr, len(train_dataset), with_noise=add_noise, alpha=alpha)

            # update learning rate for next epoch
            current_lr = lr_setter.update_lr(lr_schedule_name, optimizer, lr_param, optim_params['lr'], \
                current_lr, epoch, epochs, num_train_batches, batch_idx, **lr_schedule_params)

            loss.backward()
            if precond:
                precond.step()
            if lr_param:
                optimizer.step(lr=current_lr)
            else:
                optimizer.step()

        elapsed = time.time() - t0
        epoch_loss = np.mean(batch_losses)
        epoch_losses_train.append(epoch_loss)

        writer.add_scalar("Loss/train", epoch_loss, epoch*(i+1))
        writer.add_scalar("Duration", elapsed, epoch*(i+1))
        writer.add_scalar("Learning Rate", current_lr, epoch*(i+1))

        # if `save_burnin=False`, always save model in every epoch.
        # More expensive but we can compare small vs large number of epochs 
        # after training once for large epochs.
        # Otherwise if `save_burnin=True`, only save models if current epoch
        # is >= 75% of total number of epochs (only save models from the last 25% epochs)
        if save_burnin or epoch+1 >= num_epochs_low:
            checkpoint_path = model.checkpoints_dir + "/model_" + model_id +"_epoch_" + str(epoch+1) + ".pth"
            torch.save({
                'model_state': model.state_dict(),
                'lr': current_lr
            }, checkpoint_path)
    
    shutil.copy2(yaml_path, model.checkpoints_dir + "/..")

    writer.flush()