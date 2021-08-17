from lib.model import ToyNet

import torch
import torch.utils.data
import torch.nn.functional as F

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import argparse

# base_dir = "/content/sgld-experiments"
# optimizer_name = "psgld2.pSGLD"

# num_epochs = 50

parser = argparse.ArgumentParser(
                    description="Evaluate probability distribution plots "
                                "From model trained using SGLD.")
parser.add_argument("-b", "--base_dir",
                    help="base directory of the project")
parser.add_argument("-o", "--optimizer",
                    help="optimizer code name")
parser.add_argument("-e", "--epoch",
                    help="number of epochs done for training the model")

args = parser.parse_args()
base_dir = str(args.base_dir)
optimizer_name = str(args.optimizer_name)
num_epochs = int(args.epoch)

num_epochs_low = int(0.75*num_epochs)
print (num_epochs_low)

x_min = -6.0
x_max = 6.0
num_points = 60

M_values = [1, 4, 16, 64, 256]
# M_values = [1]
for M in M_values:
    print(M)

    if M > 1:
        step_size = float(num_epochs - num_epochs_low)/float(M-1)
    else:
        step_size = 0
    print(step_size)

    networks = []
    for i in range(M):
        print (int(num_epochs - i*step_size))

        network = ToyNet("eval_" + optimizer_name, project_dir=base_dir).cuda()
        step = str(int(num_epochs - i*step_size))
        checkpoint_path = base_dir + f"/training_logs/model_{optimizer_name}-1/checkpoints/model_{optimizer_name}-1_epoch_{step}.pth"
        network.load_state_dict(torch.load(checkpoint_path))
        networks.append(network)

    M_float = float(len(networks))
    print (M_float)

    for network in networks:
        network.eval()

    false_prob_values = np.zeros((num_points, num_points))
    x_values = np.linspace(x_min, x_max, num_points, dtype=np.float32)
    for x_1_i, x_1_value in enumerate(x_values):
        for x_2_i, x_2_value in enumerate(x_values):
            x = torch.from_numpy(np.array([x_1_value, x_2_value])).unsqueeze(0).cuda() # (shape: (1, 2))

            mean_prob_vector = np.zeros((2, ))
            for network in networks:
                logits = network(x) # (shape: (1, num_classes)) (num_classes==2)
                prob_vector = F.softmax(logits, dim=1) # (shape: (1, num_classes))

                prob_vector = prob_vector.data.cpu().numpy()[0] # (shape: (2, ))

                mean_prob_vector += prob_vector/M_float

            false_prob_values[x_2_i, x_1_i] = mean_prob_vector[0]

    plt.figure(1)
    x_1, x_2 = np.meshgrid(x_values, x_values)
    plt.pcolormesh(x_1, x_2, false_prob_values, cmap="RdBu", vmin=0, vmax=1)
    plt.colorbar()
    plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
    plt.savefig("%s/predictive_density_M=%d_%d.png" % (network.model_dir, M, 1))
    plt.close(1)