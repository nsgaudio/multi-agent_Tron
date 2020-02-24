import numpy as np
from utils import vector
from ng_tron import make_board, show_board
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
        self.config = config()
        self.num_iters = 0
        self.action_space = ActionSpace(self.config.num_players)
        self.board_shape = self.config.board_size
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
    
    def init_board(self):

        ob = np.zeros(self.board_shape, dtype=np.int16)
        snakes = []

        mid_height = int(self.board_shape[0] / 2)        
        for snake in range(self.num_players):
            # init each snake with length init_len
            pass

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
            action == 0 : (1,  0) right
                   == 1 : (0, -1) down
                   == 2 : (-1, 0) left
                   == 3 : (0,  1) up
        """
        self.num_iters += 1
        
        # use action to update head and body
        for i in range(self.num_players):
            assert(actions[i] in {0,1,2,3})

            if actions[i] == 1:
                self.players_dir[i].rotate(90)
            elif actions[i] == 2:
                self.players_dir[i].rotate(-90)

            self.players_head[i] = self.players_head[i] + self.players_dir[i]
            self.players_body[i].add(self.players_head[i])
        
        # check: outside of board / collision
        rewards = np.zeros(self.num_players)
        done = False
        for i in range(self.num_players):
            head = self.players_head[i]
            if not inside(head):
                rewards[i] -= 1
                done = True
                print("player {} outside of board".format(i+1))
            for j in range(self.num_players):
                if i != j and head in self.players_body[j]:
                    rewards[i] -= 1
                    rewards[j] += 1
                    done = True
                    print("{} in {}'s body".format(i+1,j+1))
        
        self.update_observation()

        return self.observation, rewards, done, {'num_iters':self.num_iters}


    def render(self):
        if self.show:
            # make_board and show board
            pass
        else:
            print(self.observation)


def vecs_to_cycles(vecs):
    """
        Handling list of vectors to cycles(np.array)
    """
    num_cycles = len(vecs)
    len_cycles = len(vecs[1])
    cycles = np.zeros((2 * num_cycles, len_cycles))
    for i, p in enumerate(vecs):
        for j, v in enumerate(p):
            cycles[2 * i + 1, j] = v.x
            cycles[2 * i, j] = v.y
    cycles = cycles.astype(int)
    return cycles

if __name__ == '__main__':
    width = 10
    env = EnvTest(board_shape=(width, width), num_players=2)

    while(True):
        _, r, done, _ = env.step([0,0])
        env.render()

        # visulize with ng's code
        # cycles = vecs_to_cycles(env.players_body)
        # board = make_board(size=width, cycles=cycles, num_cycle=2)
        # show_board(file_name=None, show=True, board=board, num_cycle=2)

        if done:
            break

    