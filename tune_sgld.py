import logging
import random
import sys
import torch
import torch.optim as optim
import torch.nn.functional as F
import lib.dataset
import lib.model
import lib.evaluation
import lib.ksgld as ksgld
import lib.eksgld as eksgld
import lib.psgld as psgld
import lib.psgld2 as psgld2
import lib.psgld3 as psgld3
import lib.sgld as sgld
import lib.sgld2 as sgld2
import lib.sgld3 as sgld3
import lib.asgld as asgld
import lib.ekfac_precond as ekfac
import lib.lr_setter as lr_setter
import argparse
import datetime
import inspect

import optuna

default_trial = 50
default_epochs = 10
default_batch = False

parser = argparse.ArgumentParser(
                    description="Perform  hyperparameter tuning of SGLD optimizer"
                                "for MNIST classification.")
parser.add_argument("-s", "--study",
                    help="hyperparam optimization study name. determine database file "+
                         "to save to.")
parser.add_argument("-t", "--trials",
                    help="number of trials to perform",
                    default=default_trial)
parser.add_argument("-e", "--epochs",
                    help="number of epoch to perform",
                    default=default_epochs)
parser.add_argument("-o", "--optimizer",
                    help="optimizer name: sgld, sgld2, sgld3, psgld, psgld2, psgld3, "+
                         "asgld, ksgld")
parser.add_argument("-b", "--batch",
                    help="tune batch size",
                    default=default_batch)
parser.add_argument("-bs", "--blocksize",
                    help="block decay size",
                    default=0)
parser.add_argument("-bd", "--blockdecay",
                    help="block decay",
                    default=0)

args = parser.parse_args()
if args.study:
    study_name = str(args.study)
else:
    study_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
trials = int(args.trials)
epochs = int(args.epochs)
blocksize = int(args.blocksize)
blockdecay = float(args.blockdecay)
tune_batch_size = bool(args.batch)
optimizer_name = str(args.optimizer)
if not optimizer_name in ['sgld', 'sgld2', 'sgld3', 'psgld', 'psgld2', 'psgld3', 
                            'asgld', 'ksgld', 'ekfac', 'eksgld']:
    raise ValueError('optimizer is not supported yet: ' + optimizer_name)

def train(model, optimizer, train_loader, test_loader, epochs, lr, precond=None):
    current_lr = lr
    # check if optimizer.step has 'lr' param
    step_args = inspect.getfullargspec(optimizer.step)
    lr_param = 'lr' in step_args.args

    for epoch in range(1, epochs+1):
        # print('epoch: ', epoch, ', current_lr: ', current_lr)
        model.train()
        for data, target in train_loader:
            data = data.cuda()
            target = target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()

            if precond:
                precond.step()
            if blocksize > 0 and blockdecay > 0 and lr_param:
                optimizer.step(lr=current_lr)
            else:
                optimizer.step()

        # update learning rate
        if blocksize > 0 and blockdecay > 0 and ((epoch) % blocksize) == 0:
            current_lr = current_lr * blockdecay
            if not lr_param:
                optimizer = lr_setter.update_lr(optimizer, current_lr)

        # validate
        val_accuracy, _ = lib.evaluation.evaluate(model, test_loader)
    return val_accuracy

def objective(trial):
    seed = 1
    batch_size = 100
    if tune_batch_size:
        # batch_size = trial.suggest_int("batch_size", 50, 1000, step=50)
        batch_size = trial.suggest_categorical("batch_size", [200])

    torch.cuda.set_device(0)
    torch.manual_seed(seed)
    random.seed(seed)
    model = lib.model.MnistModel()
    train_loader, test_loader = lib.dataset.make_datasets(bs=batch_size, test_bs=batch_size)
    model = model.cuda()

    precond = None
    if optimizer_name == "sgld":
        optimizer, lr = sgld_optimizer(model.parameters(), trial)
    elif optimizer_name == "sgld2":
        optimizer, lr = sgld2_optimizer(model.parameters(), trial)
    elif optimizer_name == "psgld":
        optimizer, lr = psgld_optimizer(model.parameters(), trial)
    elif optimizer_name == "asgld":
        optimizer, lr = asgld_optimizer(model.parameters(), trial)
    elif optimizer_name == "psgld2":
        optimizer, lr = psgld2_optimizer(model.parameters(), trial)
    elif optimizer_name == "psgld3":
        optimizer, lr = psgld3_optimizer(model.parameters(), trial)
    elif optimizer_name == "asgld":
        optimizer, lr = asgld_optimizer(model.parameters(), trial)
    elif optimizer_name == "ksgld":
        optimizer, lr = ksgld_optimizer(model, trial)
    elif optimizer_name == "ekfac":
        optimizer, precond, lr = ekfac_preconditioner(model, trial)
    elif optimizer_name == "eksgld":
        optimizer, lr = eksgld_optimizer(model, trial)

    accuracy = train(model, optimizer, train_loader, test_loader, epochs, lr, precond=precond)
    return accuracy

def print_stats(study):
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

def sgld3_optimizer(params, trial):
    # hyperparams search space
    lr = trial.suggest_categorical("lr", [5e-5, 5e-4, 5e-3, 5e-2, .5])
    burn_in = trial.suggest_int("num_burn_in_steps", 50, 300, step=50)
    optimizer = sgld3.SGLD(params, lr=lr, num_burn_in_steps=burn_in)
    return optimizer, lr

def sgld2_optimizer(params, trial):
    # hyperparams search space
    lr = trial.suggest_categorical("lr", [5e-5, 5e-4, 5e-3, 5e-2, .5])
    burn_in = trial.suggest_int("num_burn_in_steps", 50, 300, step=50)
    optimizer = sgld2.SGLD(params, lr=lr, num_burn_in_steps=burn_in, addnoise=True)
    return optimizer, lr

def sgld_optimizer(params, trial):
    # hyperparams search space
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
    burn_in = trial.suggest_int("num_burn_in_steps", 10, 1000, step=30)
    optimizer = sgld.SGLD(params, lr=lr, num_burn_in_steps=burn_in, vanilla=True)
    return optimizer, lr

def psgld_optimizer(params, trial):
    lr = trial.suggest_categorical("lr", [1e-3, 2e-3, 5e-3, 1e-2, .99])
    diagonal_bias = trial.suggest_categorical("diagonal_bias", [1e-8, 1e-5, 5e-5, 1e-3])
    burn_in = trial.suggest_categorical("num_burn_in_steps", [300, 600])
    precondition_decay_rate = trial.suggest_categorical("precondition_decay_rate", [.95, .99])
    num_pseudo_batches = trial.suggest_categorical("num_pseudo_batches", [60000, 30000, 10000, 1])
    optimizer = psgld.pSGLD(params, lr=lr, num_burn_in_steps=burn_in, precondition_decay_rate=precondition_decay_rate,
                            diagonal_bias=diagonal_bias, num_pseudo_batches=num_pseudo_batches)
    return optimizer, lr

def asgld_optimizer(params, trial):
    lr = trial.suggest_categorical("lr", [1e-3, 2e-3, 1e-2, .1, .2, .99])
    momentum = trial.suggest_categorical("momentum", [.1, .9, .99])
    eps = trial.suggest_categorical("eps", [1e-8, 1e-5, 5e-5, 1e-3])
    noise = trial.suggest_categorical("noise", [1e-2, 2e-2, 1e-1, .99])
    optimizer = asgld.ASGLD(params, lr=lr, momentum=momentum, eps=eps, noise=noise)
    return optimizer, lr

def psgld3_optimizer(params, trial):
    lr = trial.suggest_categorical("lr", [1e-3, 2e-3, 5e-3, 1e-2, .99])
    eps = trial.suggest_categorical("eps", [1e-8, 1e-5, 5e-5, 1e-3])
    burn_in = trial.suggest_categorical("num_burn_in_steps", [300, 600])
    rmsprop_decay = trial.suggest_categorical("rmsprop_decay", [.95, .99])
    train_size = trial.suggest_categorical("train_size", [60000, 30000, 10000])
    optimizer = psgld3.pSGLD(params, lr=lr, train_size=train_size, rmsprop_decay=rmsprop_decay, eps=eps, num_burn_in_steps=burn_in)
    return optimizer, lr

def psgld2_optimizer(params, trial):
    lr = trial.suggest_categorical("lr", [1e-3, 2e-3, 5e-3, 1e-2, .99])
    alpha = trial.suggest_categorical("alpha", [.95, .99])
    eps = trial.suggest_categorical("eps", [1e-8, 1e-5, 5e-5, 1e-3])
    burn_in = trial.suggest_categorical("num_burn_in_steps", [300, 600])
    train_size = trial.suggest_categorical("train_size", [60000, 30000, 10000, 1])
    optimizer = psgld2.pSGLD(params, lr=lr, train_size=train_size, alpha=alpha, eps=eps, num_burn_in_steps=burn_in)
    return optimizer, lr

def ksgld_optimizer(model, trial):
    lr = trial.suggest_float("lr", 1.e-3, 3.e-3, step=5.e-4)
    num_burn_in_steps = trial.suggest_categorical("num_burn_in_steps", [300, 600])
    # eps = trial.suggest_categorical("eps", [1e-3])
    # alpha = trial.suggest_categorical("alpha", [1.])
    # sua = trial.suggest_categorical("sua", [True])
    # pi = trial.suggest_categorical("pi", [False])
    # optimizer = ksgld.KSGLD(model, eps=eps, lr=lr, sua=sua, pi=pi, alpha=alpha, update_freq=50, add_noise=True)
    optimizer = ksgld.KSGLD(model, eps=1.e-3, lr=lr, sua=True, pi=False, alpha=1., update_freq=50, 
                            num_burn_in_steps=num_burn_in_steps, add_noise=True)
    return optimizer, lr

def ekfac_preconditioner(model, trial):
    lr = trial.suggest_float("lr", 1.e-3, 1.e-2, step=5.e-4)
    eps = trial.suggest_categorical("eps", [1e-3, 5e-4, 1e-4])
    alpha = trial.suggest_categorical("alpha", [.25,.5,.75, .9])
    optimizer = optim.SGD(model.parameters(), lr=lr)
    precond = ekfac.EKFAC(model, eps=eps, alpha=alpha, sua=False, ra=True, update_freq=50)
    return optimizer, precond, lr

def eksgld_optimizer(model, trial):
    lr = trial.suggest_categorical("lr", [1.e-3, 2.e-3, 2.5e-3, 3e-3, 5e-3, 1e-2])
    num_burn_in_steps = trial.suggest_categorical("num_burn_in_steps", [300, 600])
    eps = trial.suggest_categorical("eps", [1e-3, 5e-4, 1e-4])
    alpha = trial.suggest_categorical("alpha", [.15, .25, .5, .75])
    update_style = trial.suggest_categorical("update_style", ['sgld', 'psgld', 'ksgld'])
    optimizer = eksgld.EKSGLD(model, eps=eps, lr=lr, train_size=60000, sua=False, ra=True, alpha=alpha, 
                                update_freq=50, num_burn_in_steps=num_burn_in_steps, update_style=update_style)
    return optimizer, lr

def main():
    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True, direction="maximize")
    study.optimize(objective, n_trials=trials)

    print_stats(study)

if __name__ == "__main__":
    main()