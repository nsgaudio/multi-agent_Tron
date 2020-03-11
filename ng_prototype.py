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

a = np.array([0, 0, 1, 0])
b = np.array([1, 0, 1, 1])


a = np.array([1, 1, 0, 1])
b = np.array([0, 1, 1, 1])

c = np.outer(a, b)

# print(a)
# print(b)
# print(c)

# d = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
# print('d shape', d.shape)

# e = np.reshape(d, (4, 4))
# c = np.reshape(c, (16))
# print(c)

# print('d2 shape', d2.shape)

# print(d)
# print(e)
# print(d2)

# a = True
# b = True
# c = False

# print(a * b)
# print(a * c)



a = 12

print('column', a % 4)
print('row', np.floor_divide(a, 4))