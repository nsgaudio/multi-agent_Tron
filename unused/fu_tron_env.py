import numpy as np
from utils import vector
from ng_tron import make_board, show_board

class ActionSpace(object):
    def __init__(self, n):
        self.n = n

    def sample(self):
        return np.random.randint(0, high=self.n)

def inside(head):
    "Return True if head inside screen."
    return 0 <= head.x < 25 and 0 <= head.y < 25

class EnvTest(object):
    def __init__(self, board_shape=(25, 25), num_players=2):
        """
        self.num_iters: counting the horizon, if we want to use survival time as rewards later
        self.action_space: {0,1,2}
                           0: stay in current direction
                           1: turn 90 degree, counter-clockwise
                           2: turn -90 degree, counter-clockwise
        self.palyers_head: list of vectors, length = num_players
                           vector v = (v.x, v.y)
                           self.palyers_head[i]: head position of player i on the board
        self.palyers_dir:  list of vectors, length = num_players
                           self.palyers_dir[i] : direction of player i on the board
                           example: (0,1), (1,0), (0,-1), (-1,0)
        self.palyers_body: list of sets of vectors, length = num_players
                           self.palyers_body[i]: set of vectors occupied by player i
                           example: [{(v1.x, v1.y), (v2.x, v2.y)}, {...}, ...]
        self.observation:  board, the pixels in player i's body will be marked as "i"
                           input of our network?

        """
        self.num_iters = 0
        self.action_space = ActionSpace(3)
        self.board_shape = board_shape
        self.num_players = num_players
        
        self.players_head = [vector(2*i, 0) for i in range(num_players)] ### TODO: player init
        self.players_body = [{self.players_head[i]} for i in range(num_players)]
        self.players_dir = [vector(0, 1) for i in range(num_players)]
        self.observation = np.zeros(board_shape, dtype=np.int16)
        self.update_observation()
    
    def update_observation(self):
        """
            add head to body, do nothing if head exceeds the board
        """
        for i in range(self.num_players):
            try:
                self.observation[self.players_head[i].y, self.players_head[i].x] = i + 1
            except IndexError:
                pass

    def reset(self):
        self.num_iters = 0

        self.players_head = [vector(2*i, 0) for i in range(self.num_players)] ### TODO: player init
        self.players_body = [{self.players_head[i]} for i in range(self.num_players)]
        self.players_dir = [vector(0, 1) for i in range(self.num_players)]
        self.observation = np.zeros(self.board_shape, dtype=np.int16)
        self.update_observation()
        
    def step(self, actions):
        """
            update observation, rewards base on actions
            actions: a list of action, len(actions) is num_players
        """
        self.num_iters += 1
        
        # use action to update head and body
        for i in range(self.num_players):
            assert(actions[i] in {0,1,2})

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

    