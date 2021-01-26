import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import lib.dataset
import lib.model
import lib.evaluation
import lib.sgd
import lib.ekfac_precond
import lib.kfac_precond
import lib.asgld
import argparse
import numpy as np
import time

default_seed = 1
default_epochs = 10
default_lr = 0.9
default_decay = 0.01
default_epsilon = 0.01
default_noise = False

parser = argparse.ArgumentParser(
                    description="Trains and saves neural network for "
                                "MNIST classification.")
parser.add_argument("-s", "--seed",
                    help="manual seed",
                    default=default_seed)
parser.add_argument("-e", "--epochs",
                    help="number of epochs",
                    default=default_epochs)
parser.add_argument("-lr", "--learning_rate",
                    help="learning rate",
                    default=default_lr)
parser.add_argument("-d", "--decay",
                    help="weight decay",
                    default=default_decay)
parser.add_argument("-eps", "--epsilon",
                    help="Tikhonov regularization",
                    default=default_epsilon)
parser.add_argument("-n", "--noise",
                    help="inject langevin noise to gradien",
                    default=default_noise)
parser.add_argument("-p", "--precond",
                    help="preconditioner")

args = parser.parse_args()
seed = int(args.seed)
epochs = int(args.epochs)
learning_rate = float(args.learning_rate)
decay = float(args.decay)
epsilon = float(args.epsilon)
noise = bool(args.noise)
use_precond = None 
if args.precond:
    use_precond = str(args.precond)

torch.cuda.set_device(0)
torch.manual_seed(seed)
random.seed(seed)
model = lib.model.MnistModel()
train_loader, test_loader = lib.dataset.make_datasets()
model = model.cuda()

# optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=decay)
# optimizer = optim.SGD(model.parameters(), lr=learning_rate)
optimizer = lib.asgld.ASGLD(model.parameters(), lr=learning_rate)
print('addnoise: ', noise)
# optimizer = lib.sgd.SGD(model.parameters(), lr=learning_rate, addnoise=noise)
precond = None
if use_precond:
    if use_precond == "EKFAC":
        precond = lib.ekfac_precond.EKFAC(model, epsilon, alpha=1.0)
    else:
        precond = lib.kfac_precond.KFAC(model, epsilon)

def inject_noise(params, lr, debug=False):
    for p in params:
        if p.grad is None:
            continue
        d_p = p.grad

        size = d_p.size()
        langevin_noise = Normal(
            torch.zeros(size),
            torch.ones(size) / np.sqrt(lr)
        )
        if debug:
            print('inject noise from mean 0 and std {0:.3f}'.format(np.sqrt(lr)))
        p.grad.add_(d_p + langevin_noise.sample().cuda())

for epoch in range(epochs):
    t0 = time.time()
    model.train()
    batch = 0
    for data, target in train_loader:
        batch += 1
        data = data.cuda()
        target = target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        if noise:
            debug = (batch == 1)
            inject_noise(model.parameters(), learning_rate, debug=debug)
        if precond:
            precond.step()
        optimizer.step()

        prediction = output.data.max(1)[1]   # first column has actual prob.
        accuracy = np.mean(prediction.eq(target.data).cpu().numpy())*100

    # measure training time
    elapsed = time.time() - t0

    # validate
    val_accuracy, _ = lib.evaluation.evaluate(model, test_loader)

    print('Epoch: {}\tTrain Sec: {:0.3f}\tLoss: {:.3f}\tAcc: {:.3f}\tVal Acc: {:.3f}'
            .format(epoch, elapsed, np.mean(loss.item()), np.mean(accuracy), val_accuracy))