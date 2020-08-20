import numpy as np
from collections import deque
import random


# Ornstein-Ulhenbeck Process
# Taken from #https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
class OUNoise(object):
    def __init__(self, actionSpace, mu=0.0, theta=0.15, maxSigma=0.3, minSigma=0.3, decayT=100000):
        self.mu = mu
        self.theta = theta
        self.sigma = maxSigma
        self.maxSigma = maxSigma
        self.minSigma = minSigma
        self.decayT = decayT
        self.actionDim = actionSpace
        self.low = 0.01
        self.high = 0.99
        self.reset()

    def reset(self):
        self.state = np.ones(self.actionDim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.actionDim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.maxSigma - (self.maxSigma - self.minSigma) * min(1.0, t / self.decayT)
        return np.clip(action + ou_state, self.low, self.high)


class Memory:
    def __init__(self, maxSize):
        self.max_size = maxSize
        self.buffer = deque(maxlen=maxSize)

    def push(self, state, action, reward, nextState):
        experience = (state, action, np.array([reward]), nextState)
        self.buffer.append(experience)

    def sample(self, batchSize):
        stateBatch = []
        actionBatch = []
        rewardBatch = []
        nextStateBatch = []

        batch = random.sample(self.buffer, batchSize)

        for experience in batch:
            state, action, reward, next_state = experience
            stateBatch.append(state)
            actionBatch.append(action)
            rewardBatch.append(reward)
            nextStateBatch.append(next_state)

        return stateBatch, actionBatch, rewardBatch, nextStateBatch

    def __len__(self):
        return len(self.buffer)
