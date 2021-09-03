# adapted from: https://github.com/fregu856/evaluating_bdl/blob/master/toyClassification/SGLD-64/eval_plots.py

from lib.model import ToyNet

import torch
import torch.utils.data
import torch.nn.functional as F

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import argparse

# example command to run:
# !python eval_prob_distrib.py -b /content/sgld-experiments -id pSGLD -e 1600 -n 6

parser = argparse.ArgumentParser(
                    description="Evaluate probability distribution plots "
                                "From model trained using SGLD.")
parser.add_argument("-b", "--base_dir",
                    help="base directory of the project")
parser.add_argument("-id", "--model_id",
                    help="model id to identify checkpoint directory path")
parser.add_argument("-e", "--epoch",
                    help="number of epochs done for training the model")                    
parser.add_argument("-n", "--n_models",
                    help="number of models trained")
parser.add_argument("-bi", "--burnin",
                    help="burnin percentage",
                    default=0.75)

args = parser.parse_args()
base_dir = str(args.base_dir)
model_id = str(args.model_id)
num_epochs = int(args.epoch)
n_models = int(args.n_models)
burnin = float(args.burnin)

num_epochs_low = int(burnin*num_epochs)
if num_epochs_low == 0:
    num_epochs_low = 1
# print (num_epochs_low)

x_min = -6.0
x_max = 6.0
num_points = 60

# M_values = [1, 4, 16, 64, 256]
M_values = [1, 4, 16, 64]
for M in M_values:
    for iter in range(n_models):
        # print(M)

        if M > 1:
            step_size = float(num_epochs - num_epochs_low)/float(M-1)
        else:
            step_size = 0
        # print(step_size)

        networks = []
        for i in range(M):
            # print (int(num_epochs - i*step_size))

            network = ToyNet(f"eval_{model_id}-{num_epochs}_1-{n_models}", project_dir=base_dir).cuda()
            step = str(int(num_epochs - i*step_size))
            checkpoint_path = base_dir + f"/training_logs/model_{model_id}-{num_epochs}_{iter+1}/checkpoints/model_{model_id}-{num_epochs}_{iter+1}_epoch_{step}.pth"
            chk = torch.load(checkpoint_path)
            network.load_state_dict(chk['model_state'])
            networks.append((network, chk['lr']))

        M_float = float(len(networks))
        # print (M_float)

        for (network, lr) in networks:
            network.eval()

        false_prob_values = np.zeros((num_points, num_points))
        x_values = np.linspace(x_min, x_max, num_points, dtype=np.float32)
        for x_1_i, x_1_value in enumerate(x_values):
            for x_2_i, x_2_value in enumerate(x_values):
                x = torch.from_numpy(np.array([x_1_value, x_2_value])).unsqueeze(0).cuda() # (shape: (1, 2))

                mean_prob_vector = np.zeros((2, ))
                sum_lr = sum(lr for (_, lr) in networks)
                for (network, lr) in networks:
                    logits = network(x) # (shape: (1, num_classes)) (num_classes==2)
                    prob_vector = F.softmax(logits, dim=1) # (shape: (1, num_classes))

                    prob_vector = prob_vector.data.cpu().numpy()[0] # (shape: (2, ))

                    mean_prob_vector += lr*prob_vector/sum_lr

                false_prob_values[x_2_i, x_1_i] = mean_prob_vector[0]

        plt.figure(1)
        x_1, x_2 = np.meshgrid(x_values, x_values)
        plt.pcolormesh(x_1, x_2, false_prob_values, cmap="RdBu", vmin=0, vmax=1)
        plt.colorbar()
        plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
        plt.savefig("%s/predictive_density_M=%d_%d.png" % (network.model_dir, M, iter+1))
        plt.close(1)