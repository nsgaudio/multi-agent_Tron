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

    def a_to_4dir(self, a):
        # print(self.n)
        assert(self.n == 4)
        v = None
        if a == 0:
            v = vector(1,0)
        elif a == 1:
            v = vector(0,-1)
        elif a == 2:
            v = vector(-1,0)
        elif a == 3:
            v = vector(0,1)
        return v
    
    def dir4_to_a(self, dirr):
        assert(self.n == 4)
        a = None
        if dirr == vector(1,0):
            a = 0
        elif dirr == vector(0,-1):
            a = 1
        elif dirr == vector(-1,0):
            a = 2
        elif dirr == vector(0,1):
            a = 3
        return int(a)

class EnvTest(object):
    def __init__(self):
        self.config = Config()
        self.num_iters = 0
        self.action_space = ActionSpace(4)
        self.board_shape = self.config.board_shape
        self.num_players = self.config.num_players
        self.init_len = self.config.init_len
        self.lengthen_every = self.config.lengthen_every
        
        """
            snakes: a list of deque with len = num_players
                    deque = [(tail_vector), (), ...., (head_vector)] : positions of body
        """
        self.observation, self.head_board, self.snakes = self.init_board()

        # visualization
        self.show = self.config.show
        if self.show:
            self.colors = self.config.colors
            self.cmap = self.config.cmap
            self.delay = self.config.delay
            self.filename = self.config.filename
    
    def init_board(self):

        ob = np.zeros(self.board_shape, dtype=np.int16)
        head_board = np.zeros(self.board_shape, dtype=np.int16)
        snakes = []

        mid_height = int(self.board_shape[0] / 2)        
        for i in range(self.num_players):
            # init each snake with length init_len
            x = int( (i+1) * self.board_shape[1] / (self.num_players+1) )
            init_vecs = [vector(x, y) for y in range(mid_height, mid_height - self.init_len, -1)]
            snakes.append(deque(init_vecs))

            for vec in init_vecs:
                ob[vec.y, vec.x] = i+1
            
            head_board[init_vecs[-1].y, init_vecs[-1].x] = i+1

        return ob, head_board, snakes
    
    def inside(self, head):
        "Return True if head inside screen."
        return 0 <= head[1] < self.board_shape[1] and 0 <= head[0] < self.board_shape[0]

    def update_observation(self):
        """
            add head to body, do nothing if head exceeds the board
        """
        self.head_board = np.zeros_like(self.observation)

        for i in range(self.num_players):
            try:
                self.observation[self.snakes[i][-1].y, self.snakes[i][-1].x] = i + 1
            except IndexError:
                print("IndexError: {}".format(self.snakes[i][-1]))

            try:
                self.head_board[self.snakes[i][-1].y, self.snakes[i][-1].x] = i + 1
            except IndexError:
                print("IndexError: {}".format(self.snakes[i][-1]))

    def reset(self):
        self.num_iters = 0
        self.observation, self.head_board, self.snakes = self.init_board()
                
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

            tmp_head = current_head + self.action_space.a_to_4dir(actions[i])
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

        return self.observation, rewards, done, {'num_iters':self.num_iters, 'head_board':self.head_board}


    def render(self):
        if self.show:
            show_board(self.observation, self.cmap, delay=self.delay, filename=self.filename)
        else:
            print(self.observation)

def hard_codes_policy(ob, head, a, board_shape,  A_space):
    """
        head = np.array [y, x]
    """
    def valid(pos):
        if pos.y >= board_shape[0] or pos.y < 0:
            return False
        if pos.x >= board_shape[1] or pos.x < 0:
            return False
        if ob[pos.y, pos.x] != 0:
            return False
        return True

    head = vector(head[1], head[0])
    forward = head + A_space.a_to_4dir(a)
    # print(head, a, forward)

    if valid(forward):
        selected = forward
    else:
        possible = []
        right = head + A_space.a_to_4dir(0)
        down  = head + A_space.a_to_4dir(1)
        left  = head + A_space.a_to_4dir(2)
        up    = head + A_space.a_to_4dir(3)

        if valid(up): possible.append(up)
        if valid(down): possible.append(down)
        if valid(left): possible.append(left)
        if valid(right): possible.append(right)
        possible = np.array(possible)

        try:
            selected = possible[np.random.randint(0, high=possible.shape[0]), :]
        except ValueError:
            selected = forward
        
    return A.dir4_to_a(vector(selected[0], selected[1]) - head)

if __name__ == '__main__':

    env = EnvTest()
    A = ActionSpace(4)
    a1, a2, a3 = (1, 1, 1)
    while(True):
        env.render()
        hb = env.head_board
        a1 = hard_codes_policy(env.observation, np.argwhere(hb == 1)[0], a1, env.board_shape, A)
        a2 = hard_codes_policy(env.observation, np.argwhere(hb == 2)[0], a2, env.board_shape, A)
        a3 = hard_codes_policy(env.observation, np.argwhere(hb == 3)[0], a3, env.board_shape, A)

        ob, r, done, _ = env.step([a1,a2,a3])
        
        if done:
            break

    