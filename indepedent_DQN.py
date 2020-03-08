# 2 agents conrolling separate 2 separate snakes/cycles

import os
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

from env import ActionSpace, EnvTeam, hard_coded_policy
from config import *
import utils
from ng_train import ReplayMemory, InputStack, Tron_DQN

# from test import evaluate, plot, test_select_action

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

env = EnvTeam()
env.config.indep_Q = True

env.reset()
env.render()

input_stack = InputStack(env)
# input_stack.update(env)

policy_net_1 = Tron_DQN(h=env.board_shape[0], w=env.board_shape[1], outputs=env.action_space.n, env=env).to(device)
target_net_1 = Tron_DQN(h=env.board_shape[0], w=env.board_shape[1], outputs=env.action_space.n, env=env).to(device)

policy_net_2 = Tron_DQN(h=env.board_shape[0], w=env.board_shape[1], outputs=env.action_space.n, env=env).to(device)
target_net_2 = Tron_DQN(h=env.board_shape[0], w=env.board_shape[1], outputs=env.action_space.n, env=env).to(device)

policy_net_1 = policy_net_1.double()
target_net_1 = target_net_1.double()

policy_net_2 = policy_net_2.double()
target_net_2 = target_net_2.double()

target_net_1.load_state_dict(policy_net_1.state_dict())
target_net_1.eval()

target_net_2.load_state_dict(policy_net_2.state_dict())
target_net_2.eval()

optimizer = optim.RMSprop(list(policy_net_1.parameters()) + list(policy_net_1.parameters()))
memory_1 = ReplayMemory(env.config.REPLAY_MEMORY_CAP)
memory_2 = ReplayMemory(env.config.REPLAY_MEMORY_CAP)

def select_action(policy_net, input_stack, env, player_num, iterate=False):
    sample = random.random()
    eps_threshold = env.config.EPS_END + (env.config.EPS_START - env.config.EPS_END) * np.exp(
        -1. * env.num_iters / env.config.EPS_DECAY)
    if iterate:
        env.num_iters += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            input_tensor = torch.tensor(input_stack.input_stack, device=device).unsqueeze(0)
            output = policy_net(input_tensor)
            valid_actions = np.array(input_stack.valid_actions(player_num=player_num))
            adjustement = 500000 * (valid_actions - 1)
            output = output + torch.tensor(adjustement, device=device)
            output = output.max(1)[1].view(1, 1)
            return output
    else:
        valid_actions = np.array(input_stack.valid_actions(player_num=player_num))
        valid_ind = np.argwhere(valid_actions == 1)
        if valid_ind.shape[0] != 0:
            index = np.random.choice(valid_ind.shape[0], 1, replace=False)  # there is a valid index
            selected_action = valid_ind[index]
        else:
            index = np.random.choice(env.action_space.n, 1, replace=False)  # will die
            selected_action = np.expand_dims(index, axis=-1)
        return torch.tensor(selected_action, device=device, dtype=torch.long)


def optimize_model(input_stack, env):
    if len(memory_1) < env.config.BATCH_SIZE:
        return
    transitions_1 = memory_1.sample(env.config.BATCH_SIZE)
    transitions_2 = memory_2.sample(env.config.BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch_1 = Transition(*zip(*transitions_1))
    batch_2 = Transition(*zip(*transitions_2))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask_1 = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch_1.next_state)), device=device, dtype=torch.bool)
    non_final_mask_2 = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch_2.next_state)), device=device, dtype=torch.bool)

    # non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    # state_batch = torch.cat(batch.state)
    # action_batch = torch.cat(batch.action)
    # reward_batch = torch.cat(batch.reward)

    non_final_next_states_1 = torch.cat([torch.tensor(s, device=device) for s in batch_1.next_state if s is not None])
    non_final_next_states_2 = torch.cat([torch.tensor(s, device=device) for s in batch_2.next_state if s is not None])

    # torch.tensor(input_stack.input_stack, device=device)
    state_batch_1 = torch.cat(batch_1.state)
    action_batch_1 = torch.cat(batch_1.action)
    reward_batch_1 = torch.cat(batch_1.reward)

    state_batch_2 = torch.cat(batch_2.state)
    action_batch_2 = torch.cat(batch_2.action)
    reward_batch_2 = torch.cat(batch_2.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values_1 = policy_net_1(state_batch_1).gather(1, action_batch_1)
    state_action_values_2 = policy_net_2(state_batch_2).gather(2, action_batch_2)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    output_1 = target_net_1(non_final_next_states_1)
    output_2 = target_net_1(non_final_next_states_2)

    def batch_valid_actions(player_num, non_final_next_states, env):
        def valid(pos, obs):
            if pos[0] >= env.board_shape[0] or pos[0] < 0:
                return False
            if pos[1] >= env.board_shape[1] or pos[1] < 0:
                return False
            if obs[pos[0], pos[1]] != 0:
                return False
            return True

        bvs = np.zeros((env.config.BATCH_SIZE, env.action_space.n))
        for b in range(env.config.BATCH_SIZE):
            head = np.argwhere(non_final_next_states[b, 1, :, :].cpu().numpy() == player_num).squeeze()
            bvs[b, :] = [valid([head[0], head[1] + 1], non_final_next_states[b, 0, :, :]),
                         valid([head[0] - 1, head[1]], non_final_next_states[b, 0, :, :]),
                         valid([head[0], head[1] - 1], non_final_next_states[b, 0, :, :]),
                         valid([head[0] + 1, head[1]], non_final_next_states[b, 0, :, :])]
        return np.array(bvs)

    valid_actions_1 = batch_valid_actions(player_num=1, non_final_next_states=non_final_next_states_1, env=env)
    valid_actions_2 = batch_valid_actions(player_num=2, non_final_next_states=non_final_next_states_2, env=env)

    adjustement = 500000 * (valid_actions_1 - 1)
    output_1 = output_1 + torch.tensor(adjustement, device=device)
    next_state_values_1 = torch.zeros(env.config.BATCH_SIZE, device=device).double()
    next_state_values_1[non_final_mask_1] = output_1.max(1)[0].detach()

    adjustement = 500000 * (valid_actions_2 - 1)
    output_2 = output_2 + torch.tensor(adjustement, device=device)
    next_state_values_2 = torch.zeros(env.config.BATCH_SIZE, device=device).double()
    next_state_values_2[non_final_mask_2] = output_2.max(1)[0].detach()

    # next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    # Compute the expected Q values
    expected_state_action_values_1 = (next_state_values_1 * env.config.GAMMA) + reward_batch_1[:, 0]
    expected_state_action_values_2 = (next_state_values_2 * env.config.GAMMA) + reward_batch_2[:, 1]

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values_1, expected_state_action_values_1.unsqueeze(1)) + \
           F.smooth_l1_loss(state_action_values_2, expected_state_action_values_2.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net_1.parameters():
        param.grad.data.clamp_(-1, 1)
    for param in policy_net_2.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def evaluate(policy_net_1, policy_net_2):
    player_1_rewards = []
    player_2_rewards = []
    team_rewards = []
    player_1_win = []
    player_2_win = []
    team_win = []

    for e in range(env.config.EVAL_EPISODE):
        # Initialize the environment and state
        env.reset()
        input_stack.__init__(env)
        prev_hard_coded_a = 1  # players init to up
        prev_hard_coded_b = 1  # players init to up
        print('Starting episode:', e)
        while True:
            # Select and perform an action
            action_1 = select_action(policy_net_1, input_stack, env, player_num=1, iterate=False)
            action_2 = select_action(policy_net_2, input_stack, env, player_num=2, iterate=True)
            hard_coded_a = hard_coded_policy(env.observation, np.argwhere(env.head_board == 3)[0], prev_hard_coded_a,
                                             env.config.board_shape, env.action_space, eps=env.config.hcp_eps)
            hard_coded_b = hard_coded_policy(env.observation, np.argwhere(env.head_board == 4)[0], prev_hard_coded_b,
                                             env.config.board_shape, env.action_space, eps=env.config.hcp_eps)

            prev_hard_coded_a = hard_coded_a
            prev_hard_coded_b = hard_coded_b
            next_observation, reward, done, dictionary = env.step(
                [action_1.item(), action_2.item(), hard_coded_a, hard_coded_b])

            input_stack.update(env)

            if done:
                player_1_rewards.append(reward[0])
                player_2_rewards.append(reward[1])
                team_rewards.append(reward[0] + reward[1])
                player_1_win.append(reward[0] > 0)
                player_2_win.append(reward[1] > 0)
                team_win.append((reward[0] > 0) or (reward[1] > 0))
                break

    stats = [np.mean(player_1_rewards), np.std(player_1_rewards), np.mean(player_2_rewards), np.std(player_2_rewards),
             np.mean(team_rewards), np.std(team_rewards), np.sum(player_1_win), np.sum(player_2_win), np.sum(team_win)]

    return stats


def test_select_action(policy_net, input_stack, env, player_num):
    with torch.no_grad():
        # t.max(1) will return largest column value of each row.
        # second column on max result is index of where max element was
        # found, so we pick action with the larger expected reward.
        input_tensor = torch.tensor(input_stack.input_stack, device=device).unsqueeze(0)
        output = policy_net(input_tensor)
        valid_actions = np.array(input_stack.valid_actions(player_num=player_num))
        adjustement = 500000 * (valid_actions - 1)
        output = output + torch.tensor(adjustement, device=device)
        output = output.max(1)[1].view(1, 1)
        return output


def plot(stats_list):
    avg_reward_1 = np.array([stats[0] for stats in stats_list])
    std_reward_1 = np.array([stats[1] for stats in stats_list])
    avg_reward_2 = np.array([stats[2] for stats in stats_list])
    std_reward_2 = np.array([stats[3] for stats in stats_list])
    avg_reward_team = np.array([stats[4] for stats in stats_list])
    std_reward_team = np.array([stats[5] for stats in stats_list])

    num_wins_1 = np.array([stats[6] for stats in stats_list])
    num_wins_2 = np.array([stats[7] for stats in stats_list])
    num_wins_team = np.array([stats[8] for stats in stats_list])

    episode = np.arange(1, len(avg_reward_1) + 1)

    utils.cond_mkdir('./plots_2_agent/')

    plt.figure()
    plt.plot(episode, avg_reward_1)
    reward_upper = avg_reward_1 + std_reward_1
    reward_lower = avg_reward_1 - std_reward_1
    plt.fill_between(episode, reward_lower, reward_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')
    plt.xlabel('Evaluation #')
    plt.ylabel('Reward')
    plt.title('Average Reward of Player 1')
    plt.savefig('./plots_2_agent/reward_1')

    plt.figure()
    plt.plot(episode, avg_reward_2)
    reward_upper = avg_reward_2 + std_reward_2
    reward_lower = avg_reward_2 - std_reward_2
    plt.fill_between(episode, reward_lower, reward_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')
    plt.xlabel('Evaluation #')
    plt.ylabel('Reward')
    plt.title('Average Reward of Player 2')
    plt.savefig('./plots_2_agent/reward_2')

    plt.figure()
    plt.plot(episode, avg_reward_1)
    reward_upper = avg_reward_team + std_reward_team
    reward_lower = avg_reward_team - std_reward_team
    plt.fill_between(episode, reward_lower, reward_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')
    plt.xlabel('Evaluation #')
    plt.ylabel('Reward')
    plt.title('Average Reward of Team')
    plt.savefig('./plots_2_agent/reward_team')

    plt.figure()
    plt.plot(episode, 100 * num_wins_1 / env.config.EVAL_EPISODE)
    plt.xlabel('Evaluation #')
    plt.ylabel('Win (%)')
    plt.title('Win % of Player 1')
    plt.savefig('./plots_2_agent/wins_1')

    plt.figure()
    plt.plot(episode, 100 * num_wins_2 / env.config.EVAL_EPISODE)
    plt.xlabel('Evaluation #')
    plt.ylabel('Win (%)')
    plt.title('Win % of Player 2')
    plt.savefig('./plots_2_agent/wins_2')

    plt.figure()
    plt.plot(episode, 100 * num_wins_team / env.config.EVAL_EPISODE)
    plt.xlabel('Evaluation #')
    plt.ylabel('Win (%)')
    plt.title('Win % of Team')
    plt.savefig('./plots_2_agent/wins_team')


if __name__ == '__main__':

    stats_list = []
    for e in range(1, env.config.NUM_EPISODES + 1):
        # Initialize the environment and state
        env.reset()
        input_stack.__init__(env)
        prev_hard_coded_a = 1  # players init to up
        prev_hard_coded_b = 1  # players init to up
        print('Starting episode:', e)
        while True:
            # Select and perform an action
            action_1 = select_action(policy_net_1, input_stack, env, player_num=1, iterate=False)
            action_2 = select_action(policy_net_2, input_stack, env, player_num=2, iterate=True)
            hard_coded_a = hard_coded_policy(env.observation, np.argwhere(env.head_board == 3)[0], prev_hard_coded_a,
                                             env.config.board_shape, env.action_space, eps=env.config.hcp_eps)
            hard_coded_b = hard_coded_policy(env.observation, np.argwhere(env.head_board == 4)[0], prev_hard_coded_b,
                                             env.config.board_shape, env.action_space, eps=env.config.hcp_eps)

            prev_hard_coded_a = hard_coded_a
            prev_hard_coded_b = hard_coded_b
            next_observation, reward, done, dictionary = env.step([action_1.item(), action_2.item(), hard_coded_a, hard_coded_b])
            reward = torch.tensor([reward], device=device)

            # Observe new state
            if done:
                next_state = None

            state = input_stack.input_stack
            input_stack.update(env)
            next_state = input_stack.input_stack

            # Store the transition in memory
            # memory.push(torch.tensor(state, action, next_state, reward)
            memory_1.push(torch.tensor(state, device=device).unsqueeze(0), torch.tensor(action_1, device=device),
                        torch.tensor(next_state, device=device).unsqueeze(0), torch.tensor(reward, device=device))

            memory_2.push(torch.tensor(state, device=device).unsqueeze(0), torch.tensor(action_1, device=device),
                          torch.tensor(next_state, device=device).unsqueeze(0), torch.tensor(reward, device=device))

            # print('THIS HAPPENS')
            # Perform one step of the optimization (on the target network)
            optimize_model(input_stack, env)
            if done:
                break
            env.render()
        # Update the target network, copying all weights and biases in Tron_DQN
        if e % env.config.TARGET_UPDATE_FREQUENCY == 0:
            target_net_1.load_state_dict(policy_net_1.state_dict())
            target_net_2.load_state_dict(policy_net_2.state_dict())

        if e % env.config.MODEL_EVAL_FREQUENCY == 0:
            stats_list.append(evaluate(policy_net_1, policy_net_2))
            plot(stats_list)

        if e % env.config.MODEL_SAVE_FREQUENCY == 0:
            print('Saving model')
            utils.cond_mkdir('./models/')
            torch.save(policy_net_1, os.path.join('./models/', 'episode_%d_model_1.pth' % (e)))
            torch.save(policy_net_2, os.path.join('./models/', 'episode_%d_model_2.pth' % (e)))

    print('Complete')
    env.render()
    plot(stats_list)
    # env.close()
