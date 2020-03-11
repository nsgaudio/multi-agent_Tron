# import gym
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

env.reset()
env.render()

input_stack = InputStack(env)
# input_stack.update(env)

# print('lets see this', np.square(env.action_space.n))

if env.config.load_model is not None:
    policy_net = torch.load('pre_trained_models/{}'.format(env.config.load_model)).to(device)
    target_net = torch.load('pre_trained_models/{}'.format(env.config.load_model)).to(device)
    print('load model {} as pre-trained network'.format(env.config.load_model))
else:
    policy_net = Tron_DQN(h=env.board_shape[0], w=env.board_shape[1], outputs=np.square(env.action_space.n), env=env).to(device)
    target_net = Tron_DQN(h=env.board_shape[0], w=env.board_shape[1], outputs=np.square(env.action_space.n), env=env).to(device)

policy_net = policy_net.double()
target_net = target_net.double()

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(env.config.REPLAY_MEMORY_CAP)

if env.config.load_opponent is not None:
        opponent_net = torch.load('pre_trained_models/{}'.format(env.config.load_opponent))

def select_action(input_stack, env):
    sample = random.random()
    eps_threshold = env.config.EPS_END + (env.config.EPS_START - env.config.EPS_END) * np.exp(-1. * env.num_iters / env.config.EPS_DECAY)
    env.num_iters += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            input_tensor = torch.tensor(input_stack.input_stack, device=device).unsqueeze(0)
            output = policy_net(input_tensor)
            if env.config.with_adjustment:
                valid_actions_1 = np.array(input_stack.valid_actions(player_num=1))
                valid_actions_2 = np.array(input_stack.valid_actions(player_num=2))
                valid_actions = np.outer(valid_actions_1, valid_actions_2)
                valid_actions = np.reshape(valid_actions, (np.square(env.action_space.n)))
                adjustement = np.inf * (valid_actions - 1)
                output = output + torch.tensor(adjustement, device=device)
            output = output.max(1)[1].view(1, 1)
            return output
    else:
        valid_actions_1 = np.array(input_stack.valid_actions(player_num=1))
        valid_actions_2 = np.array(input_stack.valid_actions(player_num=2))
        valid_actions = np.outer(valid_actions_1, valid_actions_2)
        valid_actions = np.reshape(valid_actions, (np.square(env.action_space.n)))
        valid_ind = np.argwhere(valid_actions==1)
        if valid_ind.shape[0] != 0:
            index = np.random.choice(valid_ind.shape[0], 1, replace=False) # there is a valid index
            selected_action = valid_ind[index]
        else:
            index = np.random.choice(np.square(env.action_space.n), 1, replace=False)  # will die
            selected_action = np.expand_dims(index, axis=-1)
        return torch.tensor(selected_action, device=device, dtype=torch.long)

def optimize_model(input_stack, env):
    if len(memory) < env.config.BATCH_SIZE:
        return
    transitions = memory.sample(env.config.BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    # non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    # state_batch = torch.cat(batch.state)
    # action_batch = torch.cat(batch.action)
    # reward_batch = torch.cat(batch.reward)

    non_final_next_states = torch.cat([torch.tensor(s, device=device) for s in batch.next_state if s is not None])
    # torch.tensor(input_stack.input_stack, device=device)
    state_batch  = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(env.config.BATCH_SIZE, device=device).double()

    output = target_net(non_final_next_states)
    def batch_valid_actions(non_final_next_states, env):
        def valid(pos, obs):
            if pos[0] >= env.board_shape[0] or pos[0] < 0:
                return False
            if pos[1] >= env.board_shape[1] or pos[1] < 0:
                return False
            if obs[pos[0], pos[1]] != 0:
                return False
            return True
        bvs = np.zeros((env.config.BATCH_SIZE, np.square(env.action_space.n)))
        for b in range(env.config.BATCH_SIZE):
            head_1 = np.argwhere(non_final_next_states[b, 1, :, :].cpu().numpy()==1).squeeze()
            head_2 = np.argwhere(non_final_next_states[b, 1, :, :].cpu().numpy()==2).squeeze()
            bvs[b, :] = [valid([head_1[0], head_1[1]+1], non_final_next_states[b, 0, :, :]) * valid([head_2[0], head_2[1]+1], non_final_next_states[b, 0, :, :]), 
                         valid([head_1[0], head_1[1]+1], non_final_next_states[b, 0, :, :]) * valid([head_2[0]-1, head_2[1]], non_final_next_states[b, 0, :, :]), 
                         valid([head_1[0], head_1[1]+1], non_final_next_states[b, 0, :, :]) * valid([head_2[0], head_2[1]-1], non_final_next_states[b, 0, :, :]), 
                         valid([head_1[0], head_1[1]+1], non_final_next_states[b, 0, :, :]) * valid([head_2[0]+1, head_2[1]], non_final_next_states[b, 0, :, :]), 
                         valid([head_1[0]-1, head_1[1]], non_final_next_states[b, 0, :, :]) * valid([head_2[0], head_2[1]+1], non_final_next_states[b, 0, :, :]), 
                         valid([head_1[0]-1, head_1[1]], non_final_next_states[b, 0, :, :]) * valid([head_2[0]-1, head_2[1]], non_final_next_states[b, 0, :, :]), 
                         valid([head_1[0]-1, head_1[1]], non_final_next_states[b, 0, :, :]) * valid([head_2[0], head_2[1]-1], non_final_next_states[b, 0, :, :]), 
                         valid([head_1[0]-1, head_1[1]], non_final_next_states[b, 0, :, :]) * valid([head_2[0]+1, head_2[1]], non_final_next_states[b, 0, :, :]),
                         valid([head_1[0], head_1[1]-1], non_final_next_states[b, 0, :, :]) * valid([head_2[0], head_2[1]+1], non_final_next_states[b, 0, :, :]), 
                         valid([head_1[0], head_1[1]-1], non_final_next_states[b, 0, :, :]) * valid([head_2[0]-1, head_2[1]], non_final_next_states[b, 0, :, :]), 
                         valid([head_1[0], head_1[1]-1], non_final_next_states[b, 0, :, :]) * valid([head_2[0], head_2[1]-1], non_final_next_states[b, 0, :, :]), 
                         valid([head_1[0], head_1[1]-1], non_final_next_states[b, 0, :, :]) * valid([head_2[0]+1, head_2[1]], non_final_next_states[b, 0, :, :]),
                         valid([head_1[0]+1, head_1[1]], non_final_next_states[b, 0, :, :]) * valid([head_2[0], head_2[1]+1], non_final_next_states[b, 0, :, :]), 
                         valid([head_1[0]+1, head_1[1]], non_final_next_states[b, 0, :, :]) * valid([head_2[0]-1, head_2[1]], non_final_next_states[b, 0, :, :]), 
                         valid([head_1[0]+1, head_1[1]], non_final_next_states[b, 0, :, :]) * valid([head_2[0], head_2[1]-1], non_final_next_states[b, 0, :, :]), 
                         valid([head_1[0]+1, head_1[1]], non_final_next_states[b, 0, :, :]) * valid([head_2[0]+1, head_2[1]], non_final_next_states[b, 0, :, :])]
        return np.array(bvs)

    if env.config.with_adjustment:
        valid_actions = batch_valid_actions(non_final_next_states=non_final_next_states, env=env)
        adjustement = np.inf * (valid_actions - 1)
        output = output + torch.tensor(adjustement, device=device)

    next_state_values[non_final_mask] = output.max(1)[0].detach()
    # next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * env.config.GAMMA) + reward_batch[:, 0]

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def evaluate(policy_net):
    total_rewards = []
    win_loss = []
    for e in range(env.config.EVAL_EPISODE): #probably put number of episodes in conifg
        # Initialize the environment and state
        env.reset()
        input_stack.__init__(env)
        prev_a_3 = 1  # players init to up
        prev_a_4 = 1  # players init to up
        print('Starting episode:', e)

        while True:
            # Select and perform an action
            action = test_select_action(policy_net, input_stack, env, 1, 2)
            a_1 = np.floor_divide(action.item(), env.action_space.n)
            a_2 = action.item() % env.action_space.n

            # print('column', a % 4)
            # print('row', np.floor_divide(a, 4))
            # print(input_stack.input_stack[0:2,10:30,10:30])

            if env.config.load_opponent is not None:
                opponent_action = test_select_action(opponent_net, input_stack, env, 3, 4)
                a_3 = np.floor_divide(opponent_action.item(), env.action_space.n)
                a_4 = opponent_action.item() % env.action_space.n
            else:
                a_3 = hard_coded_policy(env.observation, np.argwhere(env.head_board==3)[0], prev_a_3, env.config.board_shape,  env.action_space, eps=env.config.hcp_eps)
                a_4 = hard_coded_policy(env.observation, np.argwhere(env.head_board==4)[0], prev_a_4, env.config.board_shape,  env.action_space, eps=env.config.hcp_eps)
                prev_a_3 = a_3
                prev_a_4 = a_4

            next_observation, reward, done, dictionary = env.step([a_1, a_2, a_3, a_4])

            if env.config.show:
                env.render()
            # print(next_observation)
            # print(a_2)

            input_stack.update(env)

            if done:
                player_reward = reward[0]
                win_loss.append(player_reward > 0)
                break
        total_rewards.append(player_reward)


    stats = [np.mean(total_rewards), np.std(total_rewards), np.sum(win_loss), len(win_loss)-np.sum(win_loss)]

    return stats

def test_select_action(policy_net, input_stack, env, player_num_a, player_num_b):
    with torch.no_grad():
        # t.max(1) will return largest column value of each row.
        # second column on max result is index of where max element was
        # found, so we pick action with the larger expected reward.
        input_tensor = torch.tensor(input_stack.input_stack, device=device).unsqueeze(0)
        output = policy_net(input_tensor)
        if env.config.with_adjustment:
            valid_actions_1 = np.array(input_stack.valid_actions(player_num=player_num_a))
            valid_actions_2 = np.array(input_stack.valid_actions(player_num=player_num_b))
            valid_actions = np.outer(valid_actions_1, valid_actions_2) 
            valid_actions = np.reshape(valid_actions, (np.square(env.action_space.n)))
            adjustement = np.inf * (valid_actions - 1)
            output = output + torch.tensor(adjustement, device=device)
            # print(valid_actions,output)
            # print('valid 1: {}\n valid 2: {}'.format(valid_actions_1, valid_actions_2))
            # print(valid_actions)
        output = output.max(1)[1].view(1, 1)
        return output

def plot(stats_list):
    plt.figure()
    avg_reward = np.array([stats[0] for stats in stats_list])
    std_reward = np.array([stats[1] for stats in stats_list])
    num_wins = np.array([stats[2] for stats in stats_list])
    num_loss = np.array([stats[3] for stats in stats_list])

    episode = np.arange(1, len(avg_reward)+1)

    plt.plot(episode, avg_reward)
    reward_upper = avg_reward + std_reward
    reward_lower = avg_reward - std_reward
    plt.fill_between(episode, reward_lower, reward_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')
    # plt.save('Average Reward')
    utils.cond_mkdir('./plots/')
    plt.savefig('./plots/plot')


if __name__ == '__main__':
    stats_list = []

    for e in range(1, env.config.NUM_EPISODES + 1):
        # Initialize the environment and state
        env.reset()
        input_stack.__init__(env)
        prev_a_3 = 1  # players init to up
        prev_a_4 = 1  # players init to up
        print('Starting episode:', e)
        while True:
            # Select and perform an action
            action = test_select_action(policy_net, input_stack, env, 1, 2)
            a_1 = np.floor_divide(action.item(), env.action_space.n)
            a_2 = action.item() % env.action_space.n

            if env.config.load_opponent is not None:
                opponent_action = test_select_action(opponent_net, input_stack, env, 3, 4)
                a_3 = np.floor_divide(opponent_action.item(), env.action_space.n)
                a_4 = opponent_action.item() % env.action_space.n
            else:
                a_3 = hard_coded_policy(env.observation, np.argwhere(env.head_board==3)[0], prev_a_3, env.config.board_shape,  env.action_space, eps=env.config.hcp_eps)
                a_4 = hard_coded_policy(env.observation, np.argwhere(env.head_board==4)[0], prev_a_4, env.config.board_shape,  env.action_space, eps=env.config.hcp_eps)
                prev_a_3 = a_3
                prev_a_4 = a_4

            next_observation, reward, done, dictionary = env.step([a_1, a_2, a_3, a_4])

            reward = torch.tensor([reward], device=device)

            # Observe new state
            if done:
                next_state = None

            state = input_stack.input_stack
            input_stack.update(env)
            next_state = input_stack.input_stack

            # Store the transition in memory
            # memory.push(torch.tensor(state, action, next_state, reward)
            memory.push(torch.tensor(state, device=device).unsqueeze(0), torch.tensor(action, device=device), torch.tensor(next_state, device=device).unsqueeze(0), torch.tensor(reward, device=device))

            # print('THIS HAPPENS')
            # Perform one step of the optimization (on the target network)
            optimize_model(input_stack, env)
            if done:
                break
            env.render()
        # Update the target network, copying all weights and biases in Tron_DQN
        if e % env.config.TARGET_UPDATE_FREQUENCY == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if e % env.config.MODEL_EVAL_FREQUENCY == 0:
            stats_list.append(evaluate(policy_net))
            plot(stats_list)

        if e % env.config.MODEL_SAVE_FREQUENCY == 0:
            print('Saving model')
            utils.cond_mkdir('./models/')
            torch.save(policy_net, os.path.join('./models/', 'episode_%d.pth' % (e)))

    print('Complete')
    env.render()
    plot(stats_list)
    # env.close()
