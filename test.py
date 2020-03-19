# import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import sys
np.set_printoptions(threshold=sys.maxsize)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from config import *

######### To fix import problem ############
class Tron_DQN(nn.Module):
    def __init__(self, h, w, outputs, env):
        super(Tron_DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2*env.config.INPUT_FRAME_NUM, out_channels=16, kernel_size=env.config.KERNEL_SIZE, stride=env.config.STRIDE)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=env.config.KERNEL_SIZE, stride=env.config.STRIDE)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=env.config.KERNEL_SIZE, stride=env.config.STRIDE)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=env.config.KERNEL_SIZE, stride=env.config.STRIDE):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

######### To fix import problem ############
############################################

# from ng_train import evaluate
from double_DQN import evaluate
from ng_train import Tron_DQN

if __name__ == '__main__':
    num = 50000

    stats_list = []
    for i in range(1):

        print('{}th model'.format(i))

        policy_net = torch.load('models/single_agent_multi_player/episode_{}.pth'.format(num), map_location=torch.device('cpu')) # placeholder
        # policy_net = torch.load('models/neg_pretrained/models/episode_{}.pth'.format(num), map_location=torch.device('cpu')) # placeholder
        # policy_net = torch.load('models/zero_pretrained/models/episode_{}.pth'.format(num), map_location=torch.device('cpu')) # placeholder
        # policy_net = torch.load('models/pos_pretrained/models/episode_{}.pth'.format(num), map_location=torch.device('cpu')) # placeholde
        stats = evaluate(policy_net)
        stats_list.append(stats)
        num += 500
        print(stats)