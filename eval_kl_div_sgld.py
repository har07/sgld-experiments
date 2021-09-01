# adapted from: https://github.com/fregu856/evaluating_bdl/blob/master/toyClassification/SGLD-64/eval_kl_div.py

from lib.model import ToyNet

import torch
import torch.utils.data
import torch.nn.functional as F

import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")

from torch.utils.tensorboard import SummaryWriter
import argparse

x_min = -6.0
x_max = 6.0
num_points = 60

epsilon = 1.0e-30

# example command to run:
# !python eval_kl_div_sgld.py -b /content/sgld-experiments -id ASGLD-1600 -e 1600 -n 6

default_none = "None"

parser = argparse.ArgumentParser(
                    description="Evaluate probability distribution plots "
                                "From model trained using SGLD.")
parser.add_argument("-b", "--base_dir",
                    help="base directory of the project")
parser.add_argument("-id", "--model_id",
                    help="model id to identify checkpoint directory path")
parser.add_argument("-e", "--num_epochs",
                    help="number of epoch")
parser.add_argument("-n", "--n_models",
                    help="number of models trained")
parser.add_argument("-l", "--logdir",
                    help="Log directory",
                    default=default_none)
parser.add_argument("-bi", "--burnin",
                    help="burnin percentage",
                    default=0.75)

args = parser.parse_args()
base_dir = str(args.base_dir)
model_id = str(args.model_id)
num_epochs = int(args.num_epochs)
n_models = int(args.n_models)
logdir = str(args.logdir)
burnin = float(args.burnin)

with open(f"{base_dir}/dataset/HMC/false_prob_values.pkl", "rb") as file: # (needed for python3)
    false_prob_values_HMC = pickle.load(file) # (shape: (60, 60))
# print (false_prob_values_HMC.shape)
# print (np.max(false_prob_values_HMC))
# print (np.min(false_prob_values_HMC))

# L = 64
# num_epochs = L*150

num_epochs_low = int(burnin*num_epochs)
# print (num_epochs_low)

p_HMC = false_prob_values_HMC/np.sum(false_prob_values_HMC)

x_values = np.linspace(x_min, x_max, num_points, dtype=np.float32)

x_1_train_lower = 0 # (0)
x_1_train_upper = 0 # (3)
x_2_train_lower = 0 # (-3)
x_2_train_upper = 0 # (3)
for index, value in enumerate(x_values):
    if value < 0:
        x_1_train_lower = index+1

    if value < 3:
        x_1_train_upper = index
        x_2_train_upper = index

    if value < -3:
        x_2_train_lower = index+1

# print (x_1_train_lower)
# print (x_values[x_1_train_lower])
# print (x_1_train_upper)
# print (x_values[x_1_train_upper])
# print (x_2_train_lower)
# print (x_values[x_2_train_lower])
# print (x_2_train_upper)
# print (x_values[x_2_train_upper])

p_HMC_train = p_HMC[x_2_train_lower:x_2_train_upper, x_1_train_lower:x_1_train_upper] # (shape: (29, 14))
p_HMC_train = p_HMC_train/np.sum(p_HMC_train)

if logdir != default_none:
    writer = SummaryWriter(log_dir=logdir)
else:
    writer = SummaryWriter()

# M_values = [2, 4, 8, 16, 32, 64, 128, 256, 512]
M_values = [2, 4, 16, 64]
for M in M_values:
    # print (M)

    step_size = float(num_epochs - num_epochs_low)/float(M-1)
    # print (step_size)

    if (step_size < 1):
        break

    KL_p_HMC_q_total_values = []
    KL_p_HMC_q_train_values = []
    for j in range(n_models):
        networks = []
        for i in range(M):
            #print (int(num_epochs - i*step_size))

            network = ToyNet(f"eval_kldiv_{model_id}_1-{n_models}", project_dir=base_dir).cuda()
            checkpoint_path = base_dir + f"/training_logs/model_{model_id}_{j+1}/checkpoints/model_{model_id}_{j+1}_epoch_{int(num_epochs - i*step_size)}.pth"
            chk = torch.load(checkpoint_path)
            network.load_state_dict(chk['model_state'])
            networks.append((network, chk['lr']))

        M_float = float(len(networks))
        # print (M_float)

        for (network, lr) in networks:
            network.eval()

        false_prob_values = np.zeros((num_points, num_points))
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

        # print (false_prob_values.shape)
        # print (np.max(false_prob_values))
        # print (np.min(false_prob_values))

        q = false_prob_values/np.sum(false_prob_values)

        KL_p_HMC_q_total = np.sum(p_HMC*np.log(p_HMC/(q + epsilon) + epsilon))
        KL_p_HMC_q_total_values.append(KL_p_HMC_q_total)
        #print ("KL_p_HMC_q_total: %g" % KL_p_HMC_q_total)

        q_train = q[x_2_train_lower:x_2_train_upper, x_1_train_lower:x_1_train_upper]
        q_train = q_train/np.sum(q_train)

        KL_p_HMC_q_train = np.sum(p_HMC_train*np.log(p_HMC_train/(q_train + epsilon) + epsilon))
        KL_p_HMC_q_train_values.append(KL_p_HMC_q_train)
        #print ("KL_p_HMC_q_train: %g" % KL_p_HMC_q_train)

    # print ("mean_total: %g" % np.mean(np.array(KL_p_HMC_q_total_values)))
    # print ("std_total: %g" % np.std(np.array(KL_p_HMC_q_total_values)))
    # print ("max_total: %g" % np.max(np.array(KL_p_HMC_q_total_values)))
    # print ("min_total: %g" % np.min(np.array(KL_p_HMC_q_total_values)))
    # print ("###")

    # print ("mean_train: %g" % np.mean(np.array(KL_p_HMC_q_train_values)))
    # print ("std_train: %g" % np.std(np.array(KL_p_HMC_q_train_values)))
    # print ("max_train: %g" % np.max(np.array(KL_p_HMC_q_train_values)))
    # print ("min_train: %g" % np.min(np.array(KL_p_HMC_q_train_values)))

    # print (M)

    # print ("########################")

    writer.add_scalar("mean_total", np.mean(np.array(KL_p_HMC_q_total_values)), str(M))
    writer.add_scalar("std_total", np.std(np.array(KL_p_HMC_q_total_values)), str(M))
    # writer.add_scalar("max_total", np.max(np.array(KL_p_HMC_q_total_values)), M)
    # writer.add_scalar("min_total", np.min(np.array(KL_p_HMC_q_total_values)), M)

    writer.add_scalar("mean_train", np.mean(np.array(KL_p_HMC_q_train_values)), str(M))
    writer.add_scalar("std_train", np.std(np.array(KL_p_HMC_q_train_values)), str(M))
    # writer.add_scalar("max_train", np.max(np.array(KL_p_HMC_q_train_values)), M)
    # writer.add_scalar("min_train", np.min(np.array(KL_p_HMC_q_train_values)), M)

writer.flush()
