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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

env = EnvTest()

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
    def __init__(self, env):  # TODO
        self.input_stack = np.zeros((2*env.config.INPUT_FRAME_NUM, env.board_shape[0], env.board_shape[1]))
        observation, head_board, _ = env.init_board()
        for c in range(2 * env.config.INPUT_FRAME_NUM):
            if np.mod(c, 2) == 0:
                self.input_stack[c, :, :] = observation
            else:
                self.input_stack[c, :, :] = head_board

    def update(self, env):
        self.input_stack = np.append(np.expand_dims(env.head_board, axis=0), self.input_stack, axis=0)
        self.input_stack = np.append(np.expand_dims(env.observation, axis=0), self.input_stack, axis=0)
        self.input_stack = np.delete(self.input_stack, -1, axis=0)
        self.input_stack = np.delete(self.input_stack, -1, axis=0)
    
    def valid_actions(self, player_num):
        head = np.argwhere(env.head_board==player_num).squeeze()
        print('THIS IS THE HEAD', head)
        def valid(pos):
            if pos[0] >= env.board_shape[0] or pos[0] < 0:
                return False
            if pos[1]>= env.board_shape[1] or pos[1] < 0:
                return False
            if env.observation[pos[0], pos[1]] != 0:
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
# input_stack.update(env)

policy_net = Tron_DQN(h=env.board_shape[0], w=env.board_shape[1], outputs=env.action_space.n, env=env).to(device)
target_net = Tron_DQN(h=env.board_shape[0], w=env.board_shape[1], outputs=env.action_space.n, env=env).to(device)

policy_net = policy_net.double()
target_net = target_net.double()

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(env.config.REPLAY_MEMORY_CAP)

# steps_done = 0

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
            # return policy_net(input_tensor.permute(0, 3, 1, 2)).max(1)[1].view(1, 1)

            # print('THIS IS THE SHAPE', input_tensor.shape)

            output = policy_net(input_tensor)
            print('Output values', output)
            valid_actions = np.array(input_stack.valid_actions(player_num=1))
            print('Valid actions', valid_actions)
            adjustement = 50000 * (valid_actions - 1)
            print('Adjustment', adjustement)
            output += adjustement
            print('Adjusted output', output)
            # print('I WANNA SEE THIS', output)
            output = output.max(1)
            # print('I WANNA SEE THIS', output)
            output = output[1]
            # print('I WANNA SEE THIS', output)
            output = output.view(1, 1)
            print('I WANNA SEE THIS', output)
            print('**************THIS HAPPENED***************')
            return output
    else:
        print('*****************************')
        valid_actions = np.array(input_stack.valid_actions(player_num=1))
        print('Valid actions', valid_actions)
        valid_ind = np.argwhere(valid_actions==1)
        # valid_ind = list(valid_ind.squeeze())
        # valid_ind = valid_ind.squeeze()
        print('Args', valid_ind)

        # index = np.random.choice(valid_ind.shape[0], 1, replace=False)
        print('Length', len(valid_ind))
        index = np.random.choice(valid_ind.shape[0], 1, replace=False)
        valid_action = valid_ind[index]
        print('Selected valid action', valid_action)
        print('*****************************')
        # return torch.tensor([[random.randrange(env.action_space.n)]], device=device, dtype=torch.long)
        return torch.tensor(valid_action, device=device, dtype=torch.long)

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
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
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

for e in range(env.config.NUM_EPISODES):
    # Initialize the environment and state
    env.reset()
    input_stack.__init__(env)
    prev_hard_coded_a = 1  # players init to up
    print('Starting episode:', e)
    while True:
        # Select and perform an action
        action = select_action(input_stack, env)

        hard_coded_a = hard_coded_policy(env.observation, np.argwhere(env.head_board==2)[0], prev_hard_coded_a, env.config.board_shape,  env.action_space, eps=env.config.hcp_eps)
        prev_hard_coded_a = hard_coded_a
        next_observation, reward, done, dictionary = env.step([action.item(), hard_coded_a ])
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

        # Perform one step of the optimization (on the target network)
        optimize_model(input_stack, env)
        if done:
            break
        env.render()
    # Update the target network, copying all weights and biases in Tron_DQN
    if e % env.config.TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
env.render()
# env.close()
