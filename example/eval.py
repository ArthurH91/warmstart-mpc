import os
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torch.nn as nn

from model import Net
from training import NumpyDataset

import matplotlib.pyplot as plt

# Paths to the model and data
model_path = "/home/arthur/Desktop/Code/warmstart-mpc/trained_model_5000.pth"

# Load data from .npy file
data = np.load(
    "/home/arthur/Desktop/Code/warmstart-mpc/example/results_ball_5000.npy",
    allow_pickle=True,
)
T = len(data[0, 2])
nq = len(data[0, 0])
net = Net(nq, T)

# Load the model state
if os.path.exists(model_path):
    net.load_state_dict(torch.load(model_path))
    print("Model loaded successfully.")
else:
    print("Model file does not exist.")
    exit()

# Create dataset and dataloader
dataset = NumpyDataset(data)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Set the model to evaluation mode
net.eval()

# Initialize lists to store predictions and actual values
predictions = []
actuals = []
diff = []

# Make predictions and compare with actual values
with torch.no_grad():
    for inputs, actual in dataloader:
        output = net(inputs)
        diff.append((output.numpy()[0] - actual.numpy()[0])**2)
        predictions.append(output)
        actuals.append(actual)
        

mean_diff = np.mean(diff, axis=0)
std_diff = np.std(diff, axis=0)

from collections import defaultdict 

q = defaultdict(list)
q_sup_std = defaultdict(list)
q_inf_std = defaultdict(list)
for config_t, std_config_t in zip(mean_diff, std_diff):
    for i, (q_i, std_q_i) in enumerate(zip(config_t, std_config_t)):
        q[f"q_{i}"].append(q_i)
        q_sup_std[f"q_{i}"].append(q_i + std_q_i)
        q_inf_std[f"q_{i}"].append(q_i - std_q_i)
        

fig, axes = plt.subplots(7, 1, figsize=(10, 20), sharex=True)
nodes = np.linspace(0,len(q["q_0"]), len(q["q_0"]))
for i, (((key, q_i),q_i_inf), q_i_sup) in enumerate(zip(zip(q.items(), q_inf_std.values()), q_sup_std.values())):
    print(q_i_inf)
    axes[i].plot(nodes,q_i, "-o")
    axes[i].fill_between(nodes, q_i_inf, q_i_sup, alpha=0.2, label=f'{key} ± std')
    axes[i].set_title(key)
    axes[i].set_title(key)

    axes[i].grid("on")
axes[3].set_ylabel("Squared error (in (rad/s)**2)")
axes[-1].set_xlabel("Nodes of the trajectory")
plt.suptitle("Difference between output of the NN and input of the NN")
plt.tight_layout()
plt.show()