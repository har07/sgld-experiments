import random
import torch
import torch.optim as optim
import torch.nn.functional as F
import lib.dataset
import lib.model
import lib.evaluation
import argparse
import numpy as np

default_seed = 1
default_epochs = 10
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
                    default=default_epochs)
parser.add_argument("-decay", "--decay",
                    help="weight decay",
                    default=default_epochs)

args = parser.parse_args()
seed = int(args.seed)
epochs = int(args.epochs)
learning_rate = float(args.learning_rate)
decay = float(args.decay)

torch.cuda.set_device(0)
torch.manual_seed(seed)
random.seed(seed)
model = lib.model.MnistModel()
train_loader, test_loader = lib.dataset.make_datasets()
model = model.cuda()

optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=decay)

for epoch in range(epochs):
    model.train()
    for data, target in train_loader:
        data = data.cuda()
        target = target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        optimizer.step()
        loss.backward()

        prediction = output.data.max(1)[1]   # first column has actual prob.
        accuracy = np.mean(prediction.eq(target.data).cpu().numpy())*100

    # validate
    val_accuracy, _ = lib.evaluation.evaluate(model, test_loader)

    print('Epoch: {}\tLoss: {:.3f}\tAcc: {:.3f}\tVal Acc: {:.3f}'.format(epoch,
                                                                np.mean(loss.item()),
                                                                np.mean(accuracy),
                                                                val_accuracy))