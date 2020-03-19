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



# class InputStack(object):
#     def __init__(self):  # TODO
#         # self.input_stack = np.zeros((env.board_shape[0], env.board_shape[1], 2*env.config.INPUT_FRAME_NUM))
#         self.input_stack = np.zeros((2*2, 2, 2))
#         observation = np.ones((2, 2))
#         head_board = np.eye(2)
#         for c in range(2 * 2):
#             if np.mod(c, 2) == 0:
#                 # self.input_stack[:, :, c] = observation
#                 self.input_stack[c, :, :] = observation
#             else:
#                 # self.input_stack[:, :, c] = head_board
#                 self.input_stack[c, :, :] = head_board

#     def update(self):
#         # self.input_stack = np.append(np.expand_dims(env.head_board, axis=-1), self.input_stack, axis=-1)
#         self.input_stack = np.append(np.expand_dims(2*np.eye(2), axis=0), self.input_stack, axis=0)
#         # self.input_stack = np.append(np.expand_dims(env.observation, axis=-1), self.input_stack, axis=-1)
#         self.input_stack = np.append(np.expand_dims(2*np.ones((2, 2)), axis=0), self.input_stack, axis=0)
#         self.input_stack = np.delete(self.input_stack, -1, axis=0)
#         self.input_stack = np.delete(self.input_stack, -1, axis=0)


# a = InputStack()

# print(np.asarray(a.input_stack))
# print()
# a.update()
# print(np.asarray(a.input_stack))
# a.update()
# print(np.asarray(a.input_stack))
# print()


# valid_actions = np.array([0, 0, 0, 1])
# print(valid_actions)
# valid_ind = np.argwhere(valid_actions==1)
# print(valid_ind)
# print(valid_ind.shape[0] == 0)
# print(valid_ind.shape)
# ndex = np.random.choice(valid_ind.shape[0], 1, replace=False)


for e in range(1, 2 + 1):
    print(e)
