import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import lib.dataset
import lib.model
import lib.evaluation
import lib.sgd as sgd
import lib.psgld2 as psgld2
import lib.sgld3 as sgld3
import lib.ekfac_precond as ekfac
import lib.kfac_precond as kfac
import lib.asgld as asgld
import lib.ksgld as ksgld
import lib.eksgld as eksgld
import lib.lr_setter as lr_setter
import lib.sampling as sampling
import lib.generalization as generalization
import model.resnet as resnet
import model.densenet as densenet
import argparse
import numpy as np
import time
import yaml
import datetime
import inspect
from torch.utils.tensorboard import SummaryWriter
from os import makedirs

default_yaml =  "config.yaml"
default_silent = False
default_none = "None"
save_model_path = "/content/sgld-experiments"
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
parser.add_argument("-l", "--logdir",
                    help="Log directory",
                    default=default_none)
parser.add_argument("-c", "--checkpoint",
                    help="Checkpoint file",
                    default=default_none)

args = parser.parse_args()
yaml_path = str(args.yaml)
silent = bool(args.silent)
logdir = str(args.logdir)
checkpoint = str(args.checkpoint)

with open(yaml_path) as f:
    config = yaml.load(f, Loader=yaml.Loader)

seed = config['seed']
epochs = config['epoch']
block_size = config['block_size']
block_decay = config['block_decay']
optimizer_name = config['optimizer']
precond_name = config['preconditioner']
percentage_tosample = config['percentage_tosample']
step_samples = config['step_samples']
step_save_state = config['step_save_state']

dataset_params = config['dataset']
dataset_name = dataset_params['name']
if dataset_name not in ['MNIST','CIFAR10']:
    raise NotImplementedError
train_batch = dataset_params['train_batch']
test_batch = dataset_params['test_batch']

torch.cuda.set_device(0)
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

print('decay size: ', block_size, ', decay rate: ', block_decay)
print('train batch size: ', train_batch, ', test batch size: ', test_batch)

if dataset_name == "MNIST":
    model1 = lib.model.MnistModel()
    model2 = lib.model.MnistModel()
    train_loader1, train_loader2, test_loader = lib.dataset.make_simultan(bs=train_batch, test_bs=test_batch)
else:
    model_name = config['model']
    model2 = eval(model_name)()
    model1 = eval(model_name)()
    train_loader1, train_loader2, test_loader = lib.dataset.make_simultan_cifar10()(bs=train_batch, test_bs=test_batch)
    
model1 = model1.cuda()
model2 = model2.cuda()

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
    optimizer1 = eval(optimizer_name)(model, **optim_params)
    optimizer2 = eval(optimizer_name)(model, **optim_params)
else:
    optimizer1 = eval(optimizer_name)(model.parameters(), **optim_params)
    optimizer2 = eval(optimizer_name)(model.parameters(), **optim_params)

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
    precond1 = eval(precond_name)(model, **precond_params)
    precond2 = eval(precond_name)(model, **precond_params)
    print('preconditioner: ', precond_name)
    print('preconditioner params: ', precond_params)

if logdir != default_none:
    writer = SummaryWriter(log_dir=logdir)
else:
    writer = SummaryWriter()


step = 0
current_lr = optim_params["lr"]

# check if optimizer.step has 'lr' param
step_args = inspect.signature(optimizer1.step)
lr_param = 'lr' in step_args.parameters

val_accuracy=0
stdev_acc = []
std_median = 0
std_max = 0
burn_in = 0
if 'num_burn_in_steps' in optim_params:
    burn_in = optim_params['num_burn_in_steps']
batch_evaluator = lib.evaluation.BatchEvaluator(test_loader, burn_in=burn_in, thinning=100)
# print('burn_in: ', burn_in)

start_epoch = 1
if checkpoint != default_none:
    chk = torch.load(checkpoint)
    start_epoch = chk['epoch'] + 1
    step = chk['steps']
    current_lr = chk['lr']
    optimizer1.load_state_dict(chk['optimizer1_state_dict'])
    model1.load_state_dict(chk['model1_state_dict'])
    optimizer2.load_state_dict(chk['optimizer2_state_dict'])
    model2.load_state_dict(chk['model2_state_dict'])

# state_accum = []

for epoch in range(start_epoch, epochs+1):

    print('current_lr: ', current_lr)
    model1.train()
    model2.train()
    
    t0 = time.time()
    for data, target in train_loader1:
        step += 1
        data = data.cuda()
        target = target.cuda()
        optimizer1.zero_grad()
        output = model1(data)
        loss1 = F.nll_loss(output, target)
        loss1.backward()
        if precond1:
            precond1.step()
        if block_size > 0 and block_decay > 0 and lr_param:
            optimizer1.step(lr=current_lr)
        else:
            optimizer1.step()

        prediction = output.data.max(1)[1]   # first column has actual prob.
        accuracy1 = np.mean(prediction.eq(target.data).cpu().numpy())*100
    elapsed1 = time.time() - t0

    t0 = time.time()
    for data, target in train_loader2:
        data = data.cuda()
        target = target.cuda()
        optimizer2.zero_grad()
        output = model2(data)
        loss1 = F.nll_loss(output, target)
        loss1.backward()
        if precond2:
            precond2.step()
        if block_size > 0 and block_decay > 0 and lr_param:
            optimizer2.step(lr=current_lr)
        else:
            optimizer2.step()

        prediction = output.data.max(1)[1]   # first column has actual prob.
        accuracy2 = np.mean(prediction.eq(target.data).cpu().numpy())*100
    elapsed2 = time.time() - t0

    # update learning rate for next epoch
    if block_size > 0 and block_decay > 0 and ((epoch) % block_size) == 0:
        current_lr = current_lr * block_decay
        if not lr_param:
            optimizer1 = lr_setter.update_lr(optimizer1, current_lr)
            optimizer2 = lr_setter.update_lr(optimizer2, current_lr)

    # validate
    writer.add_scalar("NED", generalization.measure_diff(model1, model2), epoch)
    val_accuracy1, _ = lib.evaluation.evaluate(model1, test_loader)
    writer.add_scalar("S: Loss/train", np.mean(loss1.item()), epoch)
    writer.add_scalar("S: Acc/train", val_accuracy1, epoch)
    writer.add_scalar("S: TAcc/train", np.mean(accuracy1), epoch)
    val_accuracy2, _ = lib.evaluation.evaluate(model2, test_loader)
    writer.add_scalar("S': Loss/train", np.mean(loss2.item()), epoch)
    writer.add_scalar("S': Acc/train", val_accuracy2, epoch)
    writer.add_scalar("S': TAcc/train", np.mean(accuracy2), epoch)

    if not silent:
        print('S: Epoch: {}\tTrain Sec: {:0.3f}\tLoss: {:.3f}\tAcc: {:.3f}\tVal Acc: {:.3f}'
                .format(epoch, elapsed1, np.mean(loss1.item()), np.mean(accuracy1), val_accuracy1))
        print("S': Epoch: {}\tTrain Sec: {:0.3f}\tLoss: {:.3f}\tAcc: {:.3f}\tVal Acc: {:.3f}"
                .format(epoch, elapsed2, np.mean(loss2.item()), np.mean(accuracy2), val_accuracy2))

# Save the model weights.
# torch.save(state_accum, save_model_path + "/" + session_id+".accum.pt")
torch.save({
        'model1_state_dict': model1.state_dict(),
        'optimizer1_state_dict': optimizer1.state_dict(),
        'model2_state_dict': model2.state_dict(),
        'optimizer2_state_dict': optimizer2.state_dict(),
        'epoch': epoch,
        'steps': step,
        'lr': current_lr
    }, save_model_path + "/" + session_id+".pt")
writer.flush()