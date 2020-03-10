# import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from fu_tron_env_v2 import ActionSpace, EnvTest, hard_coded_policy
from config import *
# from test import evaluate, plot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

env = EnvTest()

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class InputStack(object):
    def __init__(self, env):  # TODO
        self.input_stack = np.zeros((2*env.config.INPUT_FRAME_NUM, env.board_shape[0], env.board_shape[1]))
        observation, head_board, _ = env.init_board()
        for c in range(2 * env.config.INPUT_FRAME_NUM):
            if np.mod(c, 2) == 0:
                self.input_stack[c, :, :] = observation
            else:
                self.input_stack[c, :, :] = head_board

    def update(self, env):
        self.input_stack = np.append(np.expand_dims(env.head_board, axis=0), self.input_stack, axis=0)
        self.input_stack = np.append(np.expand_dims(env.observation, axis=0), self.input_stack, axis=0)
        self.input_stack = np.delete(self.input_stack, -1, axis=0)
        self.input_stack = np.delete(self.input_stack, -1, axis=0)
    
    def valid_actions(self, player_num):
        head = np.argwhere(env.head_board==player_num).squeeze()
        def valid(pos):
            if pos[0] >= env.board_shape[0] or pos[0] < 0:
                return False
            if pos[1]>= env.board_shape[1] or pos[1] < 0:
                return False
            if env.observation[pos[0], pos[1]] != 0:
                return False
            return True
        return [valid([head[0], head[1]+1]), valid([head[0]-1, head[1]]), valid([head[0], head[1]-1]), valid([head[0]+1, head[1]])]


see = InputStack(env)

temp_board = see.input_stack[0].copy()
temp_head  = see.input_stack[1].copy()

for i in range(2, 2*env.config.INPUT_FRAME_NUM, 2):
    # one_ind = np.squeeze(np.argwhere(temp_head == 1.))
    # two_ind = np.squeeze(np.argwhere(temp_head == 2.))
    # print(one_ind)
    # print(two_ind)
    # temp_head[one_ind[0], one_ind[1]] = 0.
    # temp_head[two_ind[0], two_ind[1]] = 0.
    # temp_head[one_ind[0]-1, one_ind[1]] = 1.
    # temp_head[two_ind[0]-1, two_ind[1]] = 2.

    # temp_board[one_ind[0], one_ind[1]] = 0.
    # temp_board[two_ind[0], two_ind[1]] = 0.

    # print(i)

    # see.input_stack[i]   = temp_board
    # see.input_stack[i+1] = temp_head

    for p in range(1, env.config.num_players+1):
        ind = np.squeeze(np.argwhere(temp_head == p))
        temp_head[ind[0], ind[1]] = 0.
        temp_head[ind[0]-1, ind[1]] = p
        temp_board[ind[0], ind[1]] = 0.
    see.input_stack[i]   = temp_board
    see.input_stack[i+1] = temp_head
    # print(see.input_stack[0])


print('hello')
for i in range(0, 2*env.config.INPUT_FRAME_NUM, 2):
    print(i)
    print(see.input_stack[i])
