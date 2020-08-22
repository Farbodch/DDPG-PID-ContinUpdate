# A PID Environment that takes in deltaT (as tStep), and PID constants, and uses the values to calculate PID
# parameters to accelerate an ideal(*) point-object on a linear trajectory, attempting to reach
# a distance 'targetVal' away.
# Environment's point of action is the object's acceleration, changed only through tweaking PID parameters.
# Object(*) velocity (v) and object displacement (x) are calculated using Verlet's Algorithm.
# (*)Ideal refers to a point-object's massless-ness, and omitting any resistance to motion ie friction, air drag etc.
# (*)Object refers to the abstraction of a physical point-object, not a programming abstraction.
from abc import ABC

import gym
import numpy as np


class PidEnv(gym.Env, ABC):
    def __init__(self, targetVal=10):
        self.tStep = 0
        self.targetVal = targetVal
        self.err = np.array([targetVal, targetVal])
        self.kp = 0.5
        self.ki = 0.5
        self.kd = 0.5
        self.P = 0
        self.I = 0
        self.D = 0
        self.x = 0.0  # position
        self.v = 0.0  # velocity
        # self.xHist = np.array([0.0, 0.0, 0.0])
        # self.xHistCount = 0
        self.accel = np.array([0.0, 0.0])
        self.done = 0
        self.stepsTaken = 0

    def step(self, action):
        self.stepsTaken += 1

        self.kp = action[0]
        self.ki = action[1]
        self.kd = action[2]
        self.tStep = action[3]

        # if self.x < self.targetVal:
        self.moveForward()
        #  elif self.x > self.targetVal:
        #      self.moveReverse()

        #reward = -abs(self.err[1])

        reward = -(self.err[1] * self.err[1])

        if abs(self.err[1]) == 0:
            reward += self.targetVal**2
            self.done = 1

        state = np.array([self.kp, self.ki, self.kd, self.tStep])

        return state, reward, self.done

    def moveForward(self):
        self.nav(1)

    # def moveReverse(self):
    #     self.nav(-1)

    def nav(self, direction):
        done = 0
        timeCount = 0

        while done != 1:
            timeCount += 1
            if timeCount % 5 == 0:
                self.P = self.err[1] * self.kp
                self.I = (self.err[1] + self.err[0]) * self.ki * self.tStep
                self.D = ((self.err[1] - self.err[0]) / self.tStep) * self.kd
                self.err[0] = self.err[1]
                self.accel[0] = self.accel[1]
                if self.x + (direction * self.v *
                             self.tStep) + (0.5 * self.accel[0] * self.tStep * self.tStep) == self.targetVal:
                    self.accel[1] = 0
                else:
                    self.accel[1] = (direction * (self.P + self.I + self.D)) + self.accel[0]
                self.v = self.v + (direction * 0.5 * (self.accel[1] + self.accel[0]) * self.tStep)
                self.x = self.x + (direction * self.v * self.tStep) + (0.5 * self.accel[1] * self.tStep * self.tStep)
                self.err[1] = self.targetVal - self.x

                #       self.xHist[self.xHistCount] = self.x
                #       self.xHistCount += 1
                #       self.xHistCount = self.xHistCount % 3

                #     print('x: ', round(self.x, 2), ' | accel: ', round(self.accel[1], 2),
                #           ' | v: ', round(self.v, 2), ' | P: ', round(self.P, 2),
                #           ' | I: ', round(self.I, 10), ' | D: ', round(self.D, 2),
                #           " |E0: ", round(self.err[0], 2), ' |E1: ', round(self.err[1], 2))
                done = 1

    def reset(self):
        self.err = np.array([self.targetVal, self.targetVal])
        self.kp = 0.5
        self.ki = 0.5
        self.kd = 0.5
        self.P = 0
        self.I = 0
        self.D = 0
        self.x = 0.0  # position
        self.v = 0.0  # velocity
        self.accel[0] = 0.0
        self.accel[1] = 0.0
        # self.xHist = 0
        # self.xHistCount = 0
        self.done = 0
        self.stepsTaken = 0

        state = np.array([self.kp, self.ki, self.kd, self.tStep])

        return state
