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

from env import ActionSpace, EnvSolo, hard_coded_policy
from config import *
import utils 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

env = EnvSolo()

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class InputStack(object):
    def __init__(self, env):  
        self.input_stack = np.zeros((2*env.config.INPUT_FRAME_NUM, env.board_shape[0], env.board_shape[1]))
        observation, head_board, _ = env.init_board()
        for c in range(2 * env.config.INPUT_FRAME_NUM):
            if np.mod(c, 2) == 0:
                self.input_stack[c, :, :] = observation
            else:
                self.input_stack[c, :, :] = head_board
            
        temp_board = self.input_stack[0].copy()
        temp_head  = self.input_stack[1].copy()
        for i in range(2, 2*env.config.INPUT_FRAME_NUM, 2):
            for p in range(1, env.config.num_players+1):
                ind = np.squeeze(np.argwhere(temp_head == p))
                temp_head[ind[0], ind[1]]   = 0.
                temp_head[ind[0]-1, ind[1]] = p
                temp_board[ind[0], ind[1]]  = 0.
            self.input_stack[i]   = temp_board
            self.input_stack[i+1] = temp_head
        self.env = env

    def update(self, env):
        self.input_stack = np.append(np.expand_dims(env.head_board, axis=0), self.input_stack, axis=0)
        self.input_stack = np.append(np.expand_dims(env.observation, axis=0), self.input_stack, axis=0)
        self.input_stack = np.delete(self.input_stack, -1, axis=0)
        self.input_stack = np.delete(self.input_stack, -1, axis=0)
    
    def valid_actions(self, player_num):
        head = np.argwhere(self.env.head_board==player_num).squeeze()
        def valid(pos):
            if pos[0] >= self.env.board_shape[0] or pos[0] < 0:
                return False
            if pos[1]>= self.env.board_shape[1] or pos[1] < 0:
                return False
            if self.env.observation[pos[0], pos[1]] != 0:
                return False
            return True
        return [valid([head[0], head[1]+1]), valid([head[0]-1, head[1]]), valid([head[0], head[1]-1]), valid([head[0]+1, head[1]])]

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

env.reset()
env.render()

input_stack = InputStack(env)

if env.config.load_model is not None:
    policy_net = torch.load('pre_trained_models/{}'.format(env.config.load_model)).to(device)
    target_net = torch.load('pre_trained_models/{}'.format(env.config.load_model)).to(device)
    print('load model {} as pre-trained network'.format(env.config.load_model))
else:
    policy_net = Tron_DQN(h=env.board_shape[0], w=env.board_shape[1], outputs=env.action_space.n, env=env).to(device)
    target_net = Tron_DQN(h=env.board_shape[0], w=env.board_shape[1], outputs=env.action_space.n, env=env).to(device)

policy_net = policy_net.double()
target_net = target_net.double()

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(env.config.REPLAY_MEMORY_CAP)


if env.config.load_opponent is not None:
    print('load opponent model')
    
    if ~torch.cuda.is_available():
        player2_net = torch.load('pre_trained_models/indpQ_episode_50000_model_1.pth'.format(env.config.load_opponent), map_location=torch.device('cpu'))
    else:
        player2_net = torch.load('pre_trained_models/{}'.format(env.config.load_opponent))
    

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
                valid_actions = np.array(input_stack.valid_actions(player_num=1))
                adjustement = np.finfo(float).max * (valid_actions - 1)
                output = output + torch.tensor(adjustement, device=device)
            output = output.max(1)[1].view(1, 1)
            return output
    else:
        valid_actions = np.array(input_stack.valid_actions(player_num=1))
        valid_ind = np.argwhere(valid_actions==1)
        if valid_ind.shape[0] != 0:
            index = np.random.choice(valid_ind.shape[0], 1, replace=False) # there is a valid index
            selected_action = valid_ind[index]
        else:
            index = np.random.choice(env.action_space.n, 1, replace=False)  # will die
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

    non_final_next_states = torch.cat([torch.tensor(s, device=device) for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
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
    def batch_valid_actions(player_num, non_final_next_states, env):
        def valid(pos, obs):
            if pos[0] >= env.board_shape[0] or pos[0] < 0:
                return False
            if pos[1]>= env.board_shape[1] or pos[1] < 0:
                return False
            if obs[pos[0], pos[1]] != 0:
                return False
            return True
        bvs = np.zeros((env.config.BATCH_SIZE, env.action_space.n))
        for b in range(env.config.BATCH_SIZE):
            head = np.argwhere(non_final_next_states[b, 1, :, :].cpu().numpy()==player_num).squeeze()
            bvs[b, :] = [valid([head[0], head[1]+1], non_final_next_states[b, 0, :, :]), 
                         valid([head[0]-1, head[1]], non_final_next_states[b, 0, :, :]), 
                         valid([head[0], head[1]-1], non_final_next_states[b, 0, :, :]), 
                         valid([head[0]+1, head[1]], non_final_next_states[b, 0, :, :])]
        return np.array(bvs)

    if env.config.with_adjustment:
        valid_actions = batch_valid_actions(player_num=1, non_final_next_states=non_final_next_states, env=env)
        adjustement = np.finfo(float).max * (valid_actions - 1)
        output = output + torch.tensor(adjustement, device=device)

    next_state_values[non_final_mask] = output.max(1)[0].detach()

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * env.config.GAMMA) + reward_batch[:, 0]
    print('reward_batch shape {}'.format(reward_batch.shape))

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
    for e in range(env.config.EVAL_EPISODE):
        # Initialize the environment and state
        env.reset()
        input_stack.__init__(env)
        prev_hard_coded_a = 1  # players init to up
        print('Starting episode:', e)

        while True:
            # Select and perform an action
            action = test_select_action(policy_net, input_stack, env)

            if env.config.load_opponent is not None:
                hard_coded_a = test_select_action(player2_net, input_stack, env, is_opponent=True).item()
            else:
                hard_coded_a = hard_coded_policy(env.observation, np.argwhere(env.head_board==2)[0], prev_hard_coded_a, env.config.board_shape,  env.action_space, eps=env.config.hcp_eps)
                prev_hard_coded_a = hard_coded_a

            next_observation, reward, done, dictionary = env.step([action.item(), hard_coded_a])

            env.render()

            input_stack.update(env)

            if done:
                # utils.show_board(next_observation, dictionary['head_board'], env.config.cmap, delay=env.config.delay, filename='tmp.png')
                player_reward = reward[0]
                win_loss.append(player_reward > 0)
                break
        total_rewards.append(player_reward)


    stats = [np.mean(total_rewards), np.std(total_rewards), np.sum(win_loss), len(win_loss)-np.sum(win_loss)]

    return stats

def test_select_action(policy_net, input_stack, env, is_opponent=False):
    with torch.no_grad():
        # t.max(1) will return largest column value of each row.
        # second column on max result is index of where max element was
        # found, so we pick action with the larger expected reward.
        if is_opponent:
            inp = input_stack.input_stack[:,:,::-1].copy()
            inp[inp == 1] = -1
            inp[inp == 2] = 1
            inp[inp == -1] = 2
            input_tensor = torch.tensor(inp, device=device).unsqueeze(0)
        else:
            input_tensor = torch.tensor(input_stack.input_stack, device=device).unsqueeze(0)

        output = policy_net(input_tensor)
        if is_opponent:
            tmp = output[0,0]
            output[0,0] = output[0,2]
            output[0,2] = tmp

        if env.config.with_adjustment:
            if is_opponent:
                valid_actions = np.array(input_stack.valid_actions(player_num=2))
            else:
                valid_actions = np.array(input_stack.valid_actions(player_num=1))
           
            adjustement = np.finfo(float).max * (valid_actions - 1)
            
            output = output + torch.tensor(adjustement, device=device)
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
    utils.cond_mkdir('./plots/')
    plt.savefig('./plots/plot')


if __name__ == '__main__':
    stats_list = []

    for e in range(1, env.config.NUM_EPISODES + 1):
        # Initialize the environment and state
        env.reset()
        input_stack.__init__(env)
        prev_hard_coded_a = 1  # players init to up
        print('Starting episode:', e)
        while True:
            # Select and perform an action
            action = select_action(input_stack, env)

            if env.config.load_opponent is not None:
                hard_coded_a = test_select_action(player2_net, input_stack, env, is_opponent=True).item()
            else:
                hard_coded_a = hard_coded_policy(env.observation, np.argwhere(env.head_board==2)[0], prev_hard_coded_a, env.config.board_shape,  env.action_space, eps=env.config.hcp_eps)
                prev_hard_coded_a = hard_coded_a

            next_observation, reward, done, dictionary = env.step([action.item(), hard_coded_a])

            reward = torch.tensor([reward], device=device)

            # Observe new state
            if done:
                next_state = None

            state = input_stack.input_stack
            input_stack.update(env)
            next_state = input_stack.input_stack

            # Store the transition in memory
            memory.push(torch.tensor(state, device=device).unsqueeze(0), torch.tensor(action, device=device), torch.tensor(next_state, device=device).unsqueeze(0), torch.tensor(reward, device=device))

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
