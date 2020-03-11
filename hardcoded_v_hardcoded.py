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

from env import ActionSpace, EnvSolo, hard_coded_policy
from config import *


total_rewards = []
win_loss = []

env = EnvSolo()


for e in range(1000): #probably put number of episodes in conifg
    # Initialize the environment and state
    env.reset()
    prev_action = 1
    prev_hard_coded_a = 1  # players init to up
    print('Starting episode:', e)

    while True:
        # Select and perform an action
        action = hard_coded_policy(env.observation, np.argwhere(env.head_board==1)[0], prev_action, env.config.board_shape,  env.action_space, eps=env.config.hcp_eps)
        prev_action = action

        hard_coded_a = hard_coded_policy(env.observation, np.argwhere(env.head_board==2)[0], prev_hard_coded_a, env.config.board_shape,  env.action_space, eps=env.config.hcp_eps)
        prev_hard_coded_a = hard_coded_a

        next_observation, reward, done, dictionary = env.step([action, hard_coded_a])

        if env.config.show:
            env.render()

        if done:
            player_reward = reward[0]
            win_loss.append(player_reward > 0)
            break
    total_rewards.append(player_reward)


stats = [np.mean(total_rewards), np.std(total_rewards), np.sum(win_loss), len(win_loss)-np.sum(win_loss)]

print(stats)