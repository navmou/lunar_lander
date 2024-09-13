import random
import numpy as np

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done, valid_actions, valid_actions_next):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done, valid_actions, valid_actions_next)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        for ind in np.random.choice(len(self.buffer), batch_size, replace = False):
            yield self.buffer[ind] 

    def __len__(self):
        return len(self.buffer)
