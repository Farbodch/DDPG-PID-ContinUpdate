from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd



class Critic(nn.Module, ABC):
    def __init__(self, inputSize, nnSize, outputSize):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(inputSize, nnSize)
        self.linear2 = nn.Linear(nnSize, nnSize)
        self.linear3 = nn.Linear(nnSize, nnSize)
        self.linear4 = nn.Linear(nnSize, outputSize)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.softplus(self.linear1(x))
        x = F.softplus(self.linear2(x))
        x = F.softplus(self.linear3(x))
        x = self.linear4(x)

        return x


class Actor(nn.Module, ABC):
    def __init__(self, inputSize, nnSize, outputSize):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(inputSize, nnSize)
        self.linear2 = nn.Linear(nnSize, nnSize)
        self.linear3 = nn.Linear(nnSize, nnSize)
        self.linear4 = nn.Linear(nnSize, outputSize)

    def forward(self, state):
        x = F.softplus(self.linear1(state))
        x = F.softplus(self.linear2(x))
        x = F.softplus(self.linear3(x))
        x = F.softplus(self.linear4(x))

        return x