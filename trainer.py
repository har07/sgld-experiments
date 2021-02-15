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
import lib.psgld as psgld
import lib.psgld2 as psgld2
import lib.psgld3 as psgld3
import lib.sgld2 as sgld2
import lib.sgld3 as sgld3
import lib.ekfac_precond as ekfac
import lib.kfac_precond as kfac
import lib.asgld as asgld
import lib.ksgld as ksgld
import lib.lr_setter as lr_setter
import pysgmcmc.optimizers.sgld as pysgmcmc_sgld
import argparse
import numpy as np
import time
import yaml
import datetime
import inspect
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

with open(yaml_path) as f:
    config = yaml.load(f, Loader=yaml.Loader)

seed = config['seed']
epochs = config['epoch']
block_size = config['block_size']
block_decay = config['block_decay']
optimizer_name = config['optimizer']
precond_name = config['preconditioner']

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
        if v or v == False:
            optim_params[k] = v

print('optimizer: ', optimizer_name)
print('optimizer params: ', optim_params)
if config['accept_model']:
    optimizer = eval(optimizer_name)(model, **optim_params)
else:
    optimizer = eval(optimizer_name)(model.parameters(), **optim_params)

precond_params = {}
precond = None
if precond_name in config:
    precond_params2 = config[precond_name]
    for k in precond_params2:
        v = precond_params2[k]
        if v or v == False:
            precond_params[k] = v
if precond_name != '' and precond_name.lower() != 'none':
    precond = eval(precond_name)(model, **precond_params)

writer = SummaryWriter()

step = 0
current_lr = optim_params["lr"]

# check if optimizer.step has 'lr' param
step_args = inspect.getfullargspec(optimizer.step)
lr_param = 'lr' in step_args.args

for epoch in range(1, epochs+1):
    t0 = time.time()

    print('current_lr: ', current_lr)
    model.train()
    for data, target in train_loader:
        step += 1
        data = data.cuda()
        target = target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        if precond:
            precond.step()
        if block_size > 0 and block_decay > 0 and lr_param:
            optimizer.step(lr=current_lr)
        else:
            optimizer.step()

        prediction = output.data.max(1)[1]   # first column has actual prob.
        accuracy = np.mean(prediction.eq(target.data).cpu().numpy())*100
        # print('step: ', step, ', accuracy: ', accuracy)

    # measure training time
    elapsed = time.time() - t0

    # update learning rate for next epoch
    if block_size > 0 and block_decay > 0 and ((epoch) % block_size) == 0:
        current_lr = current_lr * block_decay
        if not lr_param:
            optimizer = lr_setter.update_lr(optimizer, current_lr)

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