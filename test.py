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
from ng_train import evaluate
# from double_DQN import evaluate

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

# from ng_train import Tron_DQN

if __name__ == '__main__':
    num = 50000

    stats_list = []
    for i in range(1):

        print('{}th model'.format(i))

        # policy_net = torch.load('models/doubleDQN/models/episode_{}.pth'.format(num), map_location=torch.device('cpu')) # placeholder
        policy_net = torch.load('models/neg_tr/models/episode_{}.pth'.format(num), map_location=torch.device('cpu')) # placeholder
        # policy_net = torch.load('models/zero_tr/models/episode_{}.pth'.format(num), map_location=torch.device('cpu')) # placeholder
        # policy_net = torch.load('models/pos_tr/models/episode_{}.pth'.format(num), map_location=torch.device('cpu')) # placeholder        stats = evaluate(policy_net)
        stats = evaluate(policy_net)
        stats_list.append(stats)
        num += 500
        print(stats)

    # plt.figure()
    # avg_reward = np.array([stats[0] for stats in stats_list])
    # std_reward = np.array([stats[1] for stats in stats_list])
    # num_wins = np.array([stats[2] for stats in stats_list])
    # num_loss = np.array([stats[3] for stats in stats_list])

    # episode = np.linspace(500, 50000, 100)

    # plt.plot(episode, avg_reward)
    # reward_upper = avg_reward + std_reward
    # reward_lower = avg_reward - std_reward
    # plt.fill_between(episode, reward_lower, reward_upper, color='grey', alpha=.2,
    #                  label=r'$\pm$ 1 std. dev.')
    # plt.show()

    # plt.plot(episode, num_wins)
    # plt.show()
    # plt.save('Average Reward')


    

    # print("neg:{};\nzero:{};\npos:{};".format(stats1,stats2, stats3))