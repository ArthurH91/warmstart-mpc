import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, nq, T):
        super().__init__()
        self.T, self.nq = T, nq
        self.fc1 = nn.Linear(nq*2, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, T*nq)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.reshape(-1, self.T, self.nq)
