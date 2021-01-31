import logging
import random
import sys
import torch
import torch.nn.functional as F
import lib.dataset
import lib.model
import lib.evaluation
import lib.sgld as sgld
import argparse
import datetime

import optuna

default_trial = 50

parser = argparse.ArgumentParser(
                    description="Perform  hyperparameter tuning of SGLD optimizer"
                                "for MNIST classification.")
parser.add_argument("-s", "--study",
                    help="hyperparam optimization study name. determine database file "+
                         "to save to.")
parser.add_argument("-t", "--trials",
                    help="number of trials to perform",
                    default=default_trial)

args = parser.parse_args()
if args.study:
    study_name = str(args.study)
else:
    study_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
trials = int(args.trials)

def train(model, optimizer, train_loader, test_loader, epochs):
    for _ in range(epochs):
        model.train()
        for data, target in train_loader:
            data = data.cuda()
            target = target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

        # validate
        val_accuracy, _ = lib.evaluation.evaluate(model, test_loader)
    return val_accuracy

def objective(trial):
    seed = 1
    epochs = 5
    train_batch = 50
    test_batch = 50

    torch.cuda.set_device(0)
    torch.manual_seed(seed)
    random.seed(seed)
    model = lib.model.MnistModel()
    train_loader, test_loader = lib.dataset.make_datasets(bs=train_batch, test_bs=test_batch)
    model = model.cuda()

    # hyperparams search space
    # optimizer = sgld_optimizer(model.parameters(), trial)
    optimizer = psgld_optimizer(model.parameters(), trial)

    accuracy = train(model, optimizer, train_loader, test_loader, epochs)
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

def sgld_optimizer(params, trial):
    # hyperparams search space
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
    burn_in = trial.suggest_int("num_burn_in_steps", 10, 1000, step=30)
    optimizer = sgld.SGLD(params, lr=lr, num_burn_in_steps=burn_in, vanilla=True)
    return optimizer

def psgld_optimizer(params, trial):
    lr = trial.suggest_loguniform("lr", 1e-3, 1.)
    burn_in = trial.suggest_int("num_burn_in_steps", 10, 1000, step=30)
    decay_rate = trial.suggest_uniform("precondition_decay_rate", 5e-1, 1.)
    optimizer = sgld.SGLD(params, lr=lr, num_burn_in_steps=burn_in, precondition_decay_rate=decay_rate)
    return optimizer

def asgld_optimizer(params, trial):
    lr = trial.suggest_loguniform("lr", 1e-2, 1.)
    momentum = trial.suggest_uniform("momentum", .1, 1.)
    weight_decay = trial.suggest_loguniform("weight_decay", 5e-4, 1e-1)
    eps = trial.suggest_loguniform("eps", 1e-6, 1e-1)
    noise = trial.suggest_uniform("noise", 1e-2, 1.)
    optimizer = sgld.SGLD(params, lr=lr, momentum=momentum, weight_decay=weight_decay, eps=eps, noise=noise)
    return optimizer

def main():
    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True, direction="maximize")
    study.optimize(objective, n_trials=trials)

    print_stats(study)

if __name__ == "__main__":
    main()