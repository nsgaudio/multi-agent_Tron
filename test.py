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
    num = '116k'

    policy_net = torch.load('models/exp0_500k.pth'.format(num), map_location=torch.device('cpu')) # placeholder
    stats0 = evaluate(policy_net)

    print(stats0)

    # policy_net = torch.load('models/exp1_{}.pth'.format(num), map_location=torch.device('cpu')) # placeholder
    # stats1 = evaluate(policy_net)
    
    # policy_net = torch.load('models/exp2_{}.pth'.format(num), map_location=torch.device('cpu')) # placeholder
    # stats2 = evaluate(policy_net)

    # policy_net = torch.load('models/exp3_{}.pth'.format(num), map_location=torch.device('cpu')) # placeholder
    # stats3 = evaluate(policy_net)

    # print("exp1:{};\nexp2:{};\nexp3:{};".format(stats1,stats2, stats3))