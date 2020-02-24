import numpy as np
from utils import vector, show_board
from config import *
from collections import deque

from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import colors


class ActionSpace(object):
    def __init__(self, n):
        self.n = n

    def sample(self):
        return np.random.randint(0, high=self.n)

class EnvTest(object):
    def __init__(self):
        self.config = Config()
        self.num_iters = 0
        self.action_space = ActionSpace(self.config.num_players)
        self.board_shape = self.config.board_shape
        self.num_players = self.config.num_players
        self.init_len = self.config.init_len
        self.lengthen_every = self.config.lengthen_every
        
        """
            snakes: a list of deque with len = num_players
                    deque = [(tail_vector), (), ...., (head_vector)] : positions of body
        """
        self.observation, self.snakes = self.init_board()

        # visualization
        self.show = self.config.show
        if self.show:
            self.colors = self.config.colors
            self.cmap = self.config.cmap
            self.filename = self.config.filename
    
    def init_board(self):

        ob = np.zeros(self.board_shape, dtype=np.int16)
        snakes = []

        mid_height = int(self.board_shape[0] / 2)        
        for i in range(self.num_players):
            # init each snake with length init_len
            x = int( (i+1) * self.board_shape[1] / (self.num_players+1) )
            init_vecs = [vector(x, y) for y in range(mid_height, mid_height - self.init_len, -1)]
            snakes.append(deque(init_vecs))

            for vec in init_vecs:
                ob[vec.y, vec.x] = i+1

        return ob, snakes
    
    def inside(self, head):
        "Return True if head inside screen."
        return 0 <= head[1] < self.board_shape[1] and 0 <= head[0] < self.board_shape[0]

    def update_observation(self):
        """
            add head to body, do nothing if head exceeds the board
        """
        for i in range(self.num_players):
            try:
                self.observation[self.snakes[i][-1].y, self.snakes[i][-1].x] = i + 1
            except IndexError:
                pass

    def reset(self):
        self.num_iters = 0
        self.observation, self.snakes = self.init_board()
                
    def step(self, actions):
        """
            update observation, rewards base on actions
            actions: a list of action, len(actions) is num_players
            action == 0 : (x, y) += (1,  0) right
                   == 1 : (x, y) += (0, -1) down
                   == 2 : (x, y) += (-1, 0) left
                   == 3 : (x, y) += (0,  1) up
        """
        self.num_iters += 1.0
        new_heads = [] # (y, x)
        rewards = np.zeros(self.num_players)
        done = False

        # use action to update tmp head
        # check collision from tmp head to snakes
        for i in range(self.num_players):
            assert(actions[i] in {0,1,2,3})
            current_head = self.snakes[i][-1]

            if actions[i] == 0:
                tmp_head = current_head + vector(1,0)
            elif actions[i] == 1:
                tmp_head = current_head + vector(0,-1)
            elif actions[i] == 2:
                tmp_head = current_head + vector(-1,0)
            elif actions[i] == 3:
                tmp_head = current_head + vector(0,1)

            new_heads.append(tmp_head)

            if not self.inside(tmp_head):
                rewards[i] += self.config.loss
                done = True
                print("player {} outside of board".format(i+1))
            
            for j in range(self.num_players):
                if tmp_head in self.snakes[j]:
                    rewards[i] += self.config.loss
                    rewards[j] += self.config.win 
                    done = True
                    print("{} in {}'s body".format(i+1,j+1))
        
        # check collision within new_heads
        checked = np.zeros(self.num_players, dtype=bool)
        for i, head in enumerate(new_heads):
            checked[i] = True
            for j in range(i+1, self.num_players):
                if not checked[j] and head == new_heads[j]:
                    checked[j] = True
                    rewards[i] += self.config.loss
                    rewards[j] += self.config.loss
                    done = True
                    print("{} and {} bump into each other".format(i+1,j+1))

        # update tmp head to snakes
        # TODO: do we need to update deque if done? will deque be used?
        for i, head in enumerate(new_heads):
            self.snakes[i].append(head)
            if not (self.num_iters % self.lengthen_every == 0):
                self.snakes[i].popleft()

        # update observation
        if not done:
            self.update_observation()

        rewards += self.config.time_reward * self.num_iters

        return self.observation, rewards, done, {'num_iters':self.num_iters}


    def render(self):
        if self.show:
            show_board(self.observation, self.cmap, filename=self.filename)
        else:
            print(self.observation)

def hard_codes_policy(ob):
    return 0

if __name__ == '__main__':

    env = EnvTest()
    env.render()
    # a1 = hard_codes_policy(env.observation)
    # a2 = hard_codes_policy(env.observation)

    while(True):
        A = ActionSpace(4)
        ob, r, done, _ = env.step([0,0,0])
        
        # a1 = hard_codes_policy(ob)
        # a2 = hard_codes_policy(ob)

        env.render()
        if done:
            break

    