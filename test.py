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

from ng_train import evaluate, Tron_DQN

if __name__ == '__main__':
    policy_net = torch.load('models/episode_203250.pth', map_location=torch.device('cpu')) # placeholder
    stats = evaluate(policy_net)
    print(stats)