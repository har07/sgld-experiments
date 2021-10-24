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

default_yaml =  "config/batch_cifar10.yaml"
default_silent = False
default_none = "None"

parser = argparse.ArgumentParser(
                    description="Trains and saves neural network for "
                                "MNIST/CIFAR10/CIFAR100 classification.")
parser.add_argument("-y", "--yaml",
                    help="yaml config file location",
                    default=default_yaml)
parser.add_argument("-s", "--silent",
                    help="if True, do not print per epoch accuracy",
                    default=default_silent)
parser.add_argument("-p", "--path",
                    help="directory path for saving trained model")

args = parser.parse_args()
yaml_path = str(args.yaml)
save_model_path = str(args.path)
silent = bool(args.silent)

with open(yaml_path) as f:
    config = yaml.load(f, Loader=yaml.Loader)

seed = config['seed']
epochs = config['epoch']
block_size = config['block_size']
block_decay = config['block_decay']

dataset_params = config['dataset']
dataset_name = dataset_params['name']
if dataset_name not in ['MNIST','CIFAR10','CIFAR100']:
    raise NotImplementedError
train_batch = dataset_params['train_batch']
test_batch = dataset_params['test_batch']

torch.cuda.set_device(0)
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

print('decay size: ', block_size, ', decay rate: ', block_decay)
print('train batch size: ', train_batch, ', test batch size: ', test_batch)

session_id_prefix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
f = open(f'{save_model_path}/train_logs_{session_id_prefix}.txt', 'w')
optimizers = config['optimizers']
for optimizer_name in optimizers:
    durations = []

    if dataset_name == "MNIST":
        # model = lib.model.MnistModel()
        model_name = config['model']
        model = eval(model_name)()
        train_loader, test_loader = lib.dataset.make_datasets(bs=train_batch, test_bs=test_batch)
    elif dataset_name == "CIFAR10":
        model_name = config['model']
        model = eval(model_name)()
        train_loader, test_loader = lib.dataset.make_datasets_cifar10(bs=train_batch, test_bs=test_batch)
    else:
        model_name = config['model']
        model = eval(model_name)()
        train_loader, test_loader = lib.dataset.make_datasets_cifar100(bs=train_batch, test_bs=test_batch)
        
    model = model.cuda()

    accept_model = False
    if '_accept_model' in config[optimizer_name]:
        accept_model = config[optimizer_name]['_accept_model']
    optim_params = {}
    if optimizer_name in config:
        optim_params2 = config[optimizer_name]
        for k in optim_params2:
            # skip parameter that start with "_"
            if k[0] == '_':
                continue
            v = optim_params2[k]
            if v or v == False:
                optim_params[k] = v

    session_id = f"{optimizer_name}_{session_id_prefix}"
    print('optimizer: ', optimizer_name)
    print('optimizer params: ', optim_params)
    print('optimizer: ', optimizer_name, file=f)
    print('optimizer params: ', optim_params, file=f)
    if accept_model:
        optimizer = eval(optimizer_name)(model, **optim_params)
    else:
        optimizer = eval(optimizer_name)(model.parameters(), **optim_params)

    writer = SummaryWriter(log_dir=f"runs/{session_id}")


    step = 0
    current_lr = optim_params["lr"]

    # check if optimizer.step has 'lr' param
    step_args = inspect.signature(optimizer.step)
    lr_param = 'lr' in step_args.parameters

    val_accuracy=0

    start_epoch = 1
    for epoch in range(start_epoch, epochs+1):
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

            # exception for SGD: do not perform lr decay
            if optimizer_name == 'optim.SGD':
                optimizer.step()
            elif block_size > 0 and block_decay > 0 and lr_param:
                optimizer.step(lr=current_lr)
            else:
                optimizer.step()

            prediction = output.data.max(1)[1]   # first column has actual prob.
            accuracy = np.mean(prediction.eq(target.data).cpu().numpy())*100


        # measure training time
        elapsed = time.time() - t0
        durations.append(elapsed)

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
        writer.add_scalar("Duration", elapsed, epoch)
        writer.add_scalar("Learning rate", current_lr, epoch)

        if not silent:
            entry = f'Epoch: {epoch}\tTrain Sec: {elapsed:0.3f}\tLoss: {np.mean(loss.item()):.3f}\tAcc: {np.mean(accuracy):.3f}\tVal Acc: {val_accuracy:.3f}'
            print(entry)
            print(entry, file=f)

        # Save the model weights max for the last 20 epochs
        if epochs - epoch < 20:
            torch.save({
                    'model_state_dict': model.state_dict(),
                    # 'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'steps': step,
                    'lr': current_lr
                }, f"{save_model_path}/{session_id}_{epoch}.pt")
            
    writer.flush()
    print(f"epoch duration (mean +/- std): {np.mean(durations):.2f} +/- {np.std(durations):.2f}")
    print(f"epoch duration (mean +/- std): {np.mean(durations):.2f} +/- {np.std(durations):.2f}", file=f)

f.close()