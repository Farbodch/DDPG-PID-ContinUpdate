import torch
import torch.autograd
import torch.optim as optim
import torch.nn as nn
from model import *
from utils import *


class DDPGagent:
    def __init__(self, env, numStates, numNNSize, numActions, actorLearningRate=1e-4, criticLearningRate=1e-3,
                 gamma=0.99, tau=0.01, maxMemSize=50000):
        # Params
        self.numStates = numStates
        self.numActions = numActions
        self.numNNSize = numNNSize
        self.gamma = gamma
        self.tau = tau

        self.agentUpdateLimCount = 0

        # Networks
        self.actor = Actor(self.numStates, self.numNNSize, self.numActions)
        self.actor_target = Actor(self.numStates, self.numNNSize, self.numActions)
        self.critic = Critic(self.numStates + self.numActions, self.numNNSize, self.numActions)
        self.criticTarget = Critic(self.numStates + self.numActions, self.numNNSize, self.numActions)

        for targetParam, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            targetParam.data.copy_(param.data)

        for targetParam, param in zip(self.criticTarget.parameters(), self.critic.parameters()):
            targetParam.data.copy_(param.data)

        # Training
        self.memory = Memory(maxMemSize)
        self.criticCriterion = nn.MSELoss()
        self.actorOptimizer = optim.Adam(self.actor.parameters(), lr=actorLearningRate)
        self.criticOptimizer = optim.Adam(self.critic.parameters(), lr=criticLearningRate)

    def get_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        action = self.actor.forward(state)
        action = action.data[0].tolist()
        return action

    def update(self, batchSize):
        self.agentUpdateLimCount += 1

        states, actions, rewards, nextStates = self.memory.sample(batchSize)
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        nextStates = torch.FloatTensor(nextStates)

        # Critic loss
        Qvals = self.critic.forward(states, actions)
        nextActions = self.actor_target.forward(nextStates)
        nextQ = self.criticTarget.forward(nextStates, nextActions.detach())
        Qprime = rewards + self.gamma * nextQ
        criticLoss = self.criticCriterion(Qvals, Qprime)

        # Actor loss

        policyLoss = -self.critic.forward(states, self.actor.forward(states)).mean()

        # update networks
        if self.agentUpdateLimCount % 2 == 0:

            self.actorOptimizer.zero_grad()
            policyLoss.backward()
            self.actorOptimizer.step()
            self.agentUpdateLimCount = 0

        self.criticOptimizer.zero_grad()
        criticLoss.backward()
        self.criticOptimizer.step()

        # update target networks

        for targetParam, param in zip(self.criticTarget.parameters(), self.critic.parameters()):
            targetParam.data.copy_(param.data * self.tau + targetParam.data * (1.0 - self.tau))

        for targetParam, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            targetParam.data.copy_(param.data * self.tau + targetParam.data * (1.0 - self.tau))