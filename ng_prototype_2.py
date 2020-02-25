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

from fu_tron_env_v2 import ActionSpace, EnvTest
from config import *





for i in range(100):

    print(random.randrange(4))


board = np.array([[0, 0, 0],
                  [0, 2, 1],
                  [0, 0, 0]])

x = np.argwhere(board==1)

print(x)

