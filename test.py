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

from ng_train import *

def evaluate(policy_net):
    total_rewards = []
    win_loss = []
    for e in range(5): #probably put number of episodes in conifg
        # Initialize the environment and state
        env.reset()
        input_stack.__init__(env)
        prev_hard_coded_a = 1  # players init to up
        print('Starting episode:', e)
        rewards = []

        while True:
            # Select and perform an action
            action = test_select_action(policy_net, input_stack, env)

            hard_coded_a = hard_coded_policy(env.observation, np.argwhere(env.head_board==2)[0], prev_hard_coded_a, env.config.board_shape,  env.action_space, eps=env.config.hcp_eps)
            prev_hard_coded_a = hard_coded_a
            next_observation, reward, done, dictionary = env.step([action.item(), hard_coded_a])
            rewards.append(reward)

            input_stack.update(env)

            if done:
                win_loss.append(reward)
                break

            env.render()
        total_rewards.append(np.sum(rewards))

    stats = [np.mean(total_rewards), np.std(total_rewards), np.sum(win_loss==1), np.sum(win_loss==-1), np.sum(win_loss==0)]

    return stats

def test_select_action(policy_net, input_stack, env):
    with torch.no_grad():
        # t.max(1) will return largest column value of each row.
        # second column on max result is index of where max element was
        # found, so we pick action with the larger expected reward.
        input_tensor = torch.tensor(input_stack.input_stack, device=device).unsqueeze(0)
        output = policy_net(input_tensor)
        valid_actions = np.array(input_stack.valid_actions(player_num=1))
        adjustement = 500000 * (valid_actions - 1)
        output += adjustement
        output = output.max(1)[1].view(1, 1)
        return output

def plot(stats_list):
    avg_reward = np.array([stats[0] for stats in stats_list])
    std_reward = np.array([stats[1] for stats in stats_list])
    num_wins = np.array([stats[2] for stats in stats_list])
    num_loss = np.array([stats[3] for stats in stats_list])
    num_ties = np.array([stats[4] for stats in stats_list])

    plt.plot(avg_reward)
    reward_upper = avg_reward + std_reward
    reward_lower = avg_reward - std_reward
    plt.fill_between(avg_reward, reward_lower, reward_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')
    plt.save('Average Reward')

if __name__ == '__main__':
    policy_net = load_model() # placeholder
    stats = evaluate(policy_net)
    print(stats)