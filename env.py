import numpy as np
from utils import vector, show_board
from config import *
from collections import deque

from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import colors


class ActionSpace(object):
    def __init__(self, n):
        """
        action == 0 : (x, y) += (1,  0) right in Cartisian coordinate
               == 1 : (x, y) += (0, -1) down  in Cartisian coordinate
               == 2 : (x, y) += (-1, 0) left  in Cartisian coordinate
               == 3 : (x, y) += (0,  1) up    in Cartisian coordinate
        """
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
        self.done = False
        
        """
            observation: board, shape = board_shape
                         Ex. 2 players
                         [ 0 0 0 0 0 
                           0 1 0 0 0
                           0 1 0 2 2
                           0 0 0 0 0 ]

            head_board : board, shape = board_shape
                        Ex. 2 players
                         [ 0 0 0 0 0 
                           0 1 0 0 0
                           0 0 0 0 2
                           0 0 0 0 0 ]

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
            init_vecs = [vector(x, y) for y in range(mid_height - self.init_len + 1, mid_height + 1)]
            snakes.append(deque(init_vecs))

            for vec in init_vecs:
                ob[vec.y, vec.x] = i+1
            
            head_board[init_vecs[-1].y, init_vecs[-1].x] = i+1

        return ob, head_board, snakes
    
    def inside(self, head):
        "Return True if head inside board."
        return 0 <= head[1] < self.board_shape[1] and 0 <= head[0] < self.board_shape[0]

    def update_observation(self):
        """
            1. add head to body
            2. update head_board
            do nothing if new head exceeds the board
        """
        self.head_board = np.zeros_like(self.observation)

        for i in range(self.num_players):
            try:
                self.observation[self.snakes[i][-1].y, self.snakes[i][-1].x] = i + 1
                self.head_board[self.snakes[i][-1].y, self.snakes[i][-1].x] = i + 1
            except IndexError:
                print("IndexError: {}".format(self.snakes[i][-1]))
    def compute_rewards(self, status):
        """
            lose : 0
            win  : 1
            tie  : 2
        """
        rewards = np.zeros(self.num_players, dtype=float)

        rewards[np.nonzero(status == 1)] = self.config.win
        rewards[np.nonzero(status == 0)] = self.config.lose 
        # rewards[np.nonzero(status == 2)] = self.config.tie

        rewards += self.config.time_reward * self.num_iters

        return rewards

    def reset(self):
        self.num_iters = 0
        self.observation, self.head_board, self.snakes = self.init_board()
        self.done = False
                
    def step(self, actions):
        """
            update observation, rewards base on actions
            actions: a list of action, len(actions) == num_players
            
        """
        self.num_iters += 1
        new_heads = [] # (y, x)
        
        status = np.ones(self.num_players, dtype=int) # initialize to all win
        rewards = np.zeros(self.num_players)

        # take away the tail        
        if not (self.num_iters % self.lengthen_every == 0):
            for i in range(self.num_players):
                pos = self.snakes[i].popleft()
                self.observation[pos.y, pos.x] = 0

        # use action to update tmp_head
        # check collision for tmp_head with snakes
        for i in range(self.num_players):

            assert(actions[i] in {0,1,2,3})

            current_head = self.snakes[i][-1]
            tmp_head = current_head + self.action_space.a_to_4dir(actions[i])
            new_heads.append(tmp_head)

            if not self.inside(tmp_head):
                status[i] = 0
                self.done = True
                print("player {} outside of board".format(i+1))
                continue

            j = self.observation[tmp_head.y, tmp_head.x]
            if j != 0:
                status[i] = 0
                # TODO: assign something else to the win array
                # if win_type is 'one'

                self.done = True
                print("{} in {}'s body".format(i+1,j))

        
        # check collision within new_heads
        checked = np.zeros(self.num_players, dtype=bool)
        for i, head in enumerate(new_heads):
            checked[i] = True
            for j in range(i+1, self.num_players):
                if not checked[j] and head == new_heads[j]:
                    checked[j] = True
  
                    status[i] = 0
                    status[j] = 0
                    self.done = True
                    print("{} and {} bump into each other".format(i+1,j+1))
        

        # update observation and tmp_head to snakes
        if not self.done:
            for i, head in enumerate(new_heads):
                self.snakes[i].append(head)
            self.update_observation()
        else:
            rewards = self.compute_rewards(status)
            
        return self.observation, rewards, self.done, {'num_iters':self.num_iters, 'head_board':self.head_board}


    def render(self):
        if self.show:
            show_board(self.observation, self.head_board, self.cmap, delay=self.delay, filename=self.filename)
        # else:
        # print(self.observation)

class EnvTeam(EnvTest):
    def __init__(self):
        super().__init__()
        self.teams  = self.config.teams
    
    def get_team_ids(self, i):
        """
            input : i = 0 --> player 1 on board
            output: team_ids = [0, 1] --> index in list
        """
        team_ids = None

        if (i+1) in self.teams[0]:
            team_ids = self.teams[0] - 1
        elif (i+1) in self.teams[1]:
            team_ids = self.teams[1] - 1
        else:
            raise ValueError("player id: {} not defined in config".format(i))

        return team_ids
    
    def switch_id(self, i, j):
        """
            input i = 1 --> player 1 on board
        """
        tmp_val = -1
        ob2 = self.observation.copy()
        head2 = self.head_board.copy()

        ob2[ob2 == i] = tmp_val
        ob2[ob2 == j] = i
        ob2[ob2 == tmp_val] = j

        head2[head2 == i] = tmp_val
        head2[head2 == j] = i
        head2[head2 == tmp_val] = j

        return ob2, head2

    def step(self, actions):
        """
            update observation, rewards base on actions
            actions: a list of action, len(actions) == num_players
            
        """
        self.num_iters += 1
        new_heads = [] # (y, x)
        
        status = np.ones(self.num_players, dtype=int) # initialize to all win
        rewards = np.zeros(self.num_players)

        # take away the tail        
        if not (self.num_iters % self.lengthen_every == 0):
            for i in range(self.num_players):
                pos = self.snakes[i].popleft()
                self.observation[pos.y, pos.x] = 0

        # use action to update tmp_head
        # check collision for tmp_head with snakes
        for i in range(self.num_players):

            assert(actions[i] in {0,1,2,3})

            current_head = self.snakes[i][-1]
            tmp_head = current_head + self.action_space.a_to_4dir(actions[i])
            new_heads.append(tmp_head)

            team_ids = self.get_team_ids(i)

            if not self.inside(tmp_head):
                status[team_ids] = 0
                self.done = True
                print("player {} outside of board".format(i+1))
                continue

            j = self.observation[tmp_head.y, tmp_head.x]
            if j != 0:
                status[team_ids] = 0
                # TODO: assign something else to the win array
                # if win_type is 'one'

                self.done = True
                print("{} in {}'s body".format(i+1,j))

        
        # check collision within new_heads
        checked = np.zeros(self.num_players, dtype=bool)
        for i, head in enumerate(new_heads):
            checked[i] = True
            for j in range(i+1, self.num_players):
                if not checked[j] and head == new_heads[j]:
                    checked[j] = True

                    status[self.get_team_ids(i)] = 0
                    status[self.get_team_ids(j)] = 0
                    self.done = True
                    print("{} and {} bump into each other".format(i+1,j+1))
        

        # update observation and tmp_head to snakes
        ob2 = None
        head2 = None
        if not self.done:
            for i, head in enumerate(new_heads):
                self.snakes[i].append(head)
            self.update_observation()

            if self.config.indep_Q:
                ob2, head2 = self.switch_id(1,2)
        else:
            rewards = self.compute_rewards(status)
            
        return self.observation, rewards, self.done, {'num_iters':self.num_iters, 'head_board':self.head_board, 
                                                 'ob2': ob2, 'head2':head2}


def hard_coded_policy(ob, head, a, board_shape,  A_space, eps=0.5):
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
    sample = np.random.random()

    if valid(forward) and sample > eps:
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
        
    return A_space.dir4_to_a(vector(selected[0], selected[1]) - head)


if __name__ == '__main__':

    env = EnvTeam()
    A = env.action_space
    a1, a2, a3, a4 = (3, 3, 3, 3)
    while(True):
        env.render()
        hb = env.head_board
        # print(hb)
        a1 = hard_coded_policy(env.observation, np.argwhere(hb == 1)[0], a1, env.board_shape, A, eps=env.config.hcp_eps)
        a2 = hard_coded_policy(env.observation, np.argwhere(hb == 2)[0], a2, env.board_shape, A, eps=env.config.hcp_eps)
        a3 = hard_coded_policy(env.observation, np.argwhere(hb == 3)[0], a3, env.board_shape, A, eps=env.config.hcp_eps)
        a4 = hard_coded_policy(env.observation, np.argwhere(hb == 4)[0], a4, env.board_shape, A, eps=env.config.hcp_eps)

        ob, r, done, info = env.step([a1,a2, a3, a4])
        
        if done:
            env.render()
            print("iter: {}, rewards: {}".format(info['num_iters'], r))
            break