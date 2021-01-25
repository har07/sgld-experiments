import random
import torch
import torch.optim as optim
import lib.dataset
import lib.model
import argparse
from backpack import extend
from backpack import backpack
from backpack.extensions import KFAC

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
model = extend(model)
criterion = torch.nn.NLLLoss()
criterion = extend(criterion)

optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=decay)

def print_params(model):
    for name, param in model.named_parameters():
        print(name)
        print(".grad.shape:             ", param.grad.shape)
        print(".kfac (shapes):          ", [kfac.shape for kfac in param.kfac])

for epoch in range(epochs):
    model.train()
    for data, target in train_loader:
        data = data.cuda()
        target = target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        with backpack(KFAC()):
            loss.backward()
        optimizer.step()
        print_params(model)