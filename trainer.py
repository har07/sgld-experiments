import random
import torch
import torch.optim as optim
import torch.nn.functional as F
import lib.dataset
import lib.model
import lib.evaluation
import lib.ekfac_precond
import lib.kfac_precond
import argparse
import numpy as np
import time

default_seed = 1
default_epochs = 10
default_lr = 0.9
default_decay = 0.01
default_epsilon = 0.01
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

args = parser.parse_args()
seed = int(args.seed)
epochs = int(args.epochs)
learning_rate = float(args.learning_rate)
decay = float(args.decay)
epsilon = float(args.epsilon)

torch.cuda.set_device(0)
torch.manual_seed(seed)
random.seed(seed)
model = lib.model.MnistModel()
train_loader, test_loader = lib.dataset.make_datasets()
model = model.cuda()

# optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=decay)
optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=decay)
# precond = lib.ekfac_precond.EKFAC(model, epsilon, alpha=1.0)
precond = lib.kfac_precond.KFAC(model, epsilon)

for epoch in range(epochs):
    t0 = time.time()
    model.train()
    for data, target in train_loader:
        data = data.cuda()
        target = target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
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