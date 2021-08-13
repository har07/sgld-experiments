import torch
import torch.nn.functional as F
from lib.dataset import ToyDataset
from lib.model import ToyNet
import argparse
import numpy as np
import yaml
import datetime
import time
import inspect

from torch.utils.tensorboard import SummaryWriter

default_yaml =  "config_prob_distrib.yaml"
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

seed = config['seed']
epochs = config['epoch']
block_size = config['block_size']
block_decay = config['block_decay']
optimizer_name = config['optimizer']
precond_name = config['preconditioner']

batch_size = 32

# TODO: construct model_id based on config
model_id = ''
model = ToyNet(model_id, project_dir=project_dir).cuda()
train_dataset = ToyDataset()
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

num_train_batches = int(len(train_dataset)/batch_size)
num_steps = epochs*num_train_batches + 1

optim_params = {}
if optimizer_name in config:
    optim_params2 = config[optimizer_name]
    for k in optim_params2:
        v = optim_params2[k]
        if v or v == False:
            optim_params[k] = v

print('optimizer: ', optimizer_name)
print('optimizer params: ', optim_params)
if config['accept_model']:
    optimizer = eval(optimizer_name)(model, **optim_params)
else:
    optimizer = eval(optimizer_name)(model.parameters(), **optim_params)

# check if optimizer.step has 'lr' param
step_args = inspect.signature(optimizer.step)
lr_param = 'lr' in step_args.parameters

precond_params = {}
precond1 = None
precond2 = None
if precond_name in config:
    precond_params2 = config[precond_name]
    for k in precond_params2:
        v = precond_params2[k]
        if v or v == False:
            precond_params[k] = v
if precond_name != '' and precond_name.lower() != 'none':
    precond = eval(precond_name)(model, **precond_params)
    print('preconditioner: ', precond_name)
    print('preconditioner params: ', precond_params)

if logdir != default_none:
    writer = SummaryWriter(log_dir=logdir)
else:
    writer = SummaryWriter()

step = 0
current_lr = optim_params["lr"]

epoch_losses_train = []
# for epoch in trange(epochs, desc='epoch', position=1, leave=False):
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

    writer.add_scalar("Loss/train", epoch_loss, epoch)
    writer.add_scalar("Duration", elapsed, epoch)

    # save the model weights to disk:
    checkpoint_path = model.checkpoints_dir + "/model_" + model_id +"_epoch_" + str(epoch+1) + ".pth"
    torch.save(model.state_dict(), checkpoint_path)

writer.flush()