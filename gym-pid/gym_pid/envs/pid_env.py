import gym
import matplotlib.pyplot as plt
import numpy as np


class PidEnv(gym.Env):
    def __init__(self, sample_rate=1, setpoint=50):
        self.sample_rate = sample_rate
        self.setpoint = setpoint
        self.ref = self.setpoint
        self.prevError = self.setpoint
        self.ki = 0
        self.kp = 0
        self.x1 = 0
        self.x2 = 0
        self.currentErr = self.setpoint
        self.done = 0
        self.count = 0
        self.samplingLim = 250

    def step(self, action):
        self.kp = action[0]
        self.ki = action[1]

        while self.count < self.samplingLim or self.done != 1:
            dx2 = -(self.ki * self.x1) - ((1 + self.kp) * self.x2) + self.ref
            self.x2 += dx2
            dx1 = self.x2
            self.x1 += dx1
            y = (self.ki*self.x1) + (self.kp*self.x2)
            self.currentErr = self.ref - y
            self.count += 1

            if self.prevError == 0 and self.prevError == 0:
                self.done = 1

        self.prevError = self.currentErr

        reward = -abs(self.currentErr ** 2)

        return (self.currentErr, self.setpoint), reward, self.done

    def reset(self):
        self.currentErr = 0
        self.x1 = 0
        self.x2 = 0

    def render(self, mode='human'):
        print("Error = " + self.currentErr)