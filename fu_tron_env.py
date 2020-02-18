import numpy as np
from utils import vector
from ng_tron import make_board, show_board

class ActionSpace(object):
    def __init__(self, n):
        self.n = n

    def sample(self):
        return np.random.randint(0, high=self.n)

# class ObservationSpace(object):
#     def __init__(self, shape):
#         self.shape = shape
#         self.ob = np.zeros(shape)
#     def update_observation(self, ob):
#         self.ob = ob

def inside(head):
    "Return True if head inside screen."
    return 0 <= head.x < 25 and 0 <= head.y < 25

class EnvTest(object):
    def __init__(self, board_shape=(25, 25), num_players=2):
        # self.rewards = 1.0
        # self.cur_state = 0
        self.num_iters = 0
        self.action_space = ActionSpace(3)
        self.observation = np.zeros(board_shape, dtype=np.int16)
        self.num_players = num_players
        self.players_body = [set() for i in range(num_players)]
        self.players_head = [vector(2*i, 0) for i in range(num_players)] ### TODO: player init
        self.players_dir = [vector(0, 1) for i in range(num_players)]
        self.board_shape = board_shape

    def reset(self):
        # self.cur_state = 0
        self.num_iters = 0
        self.observation = np.zeros(self.board_shape)
        self.players_head = [vector(2*i, 0) for i in range(self.num_players)] ### TODO: player init
        self.players_body = [set() for i in range(self.num_players)]
        self.players_dir = [vector(0, i) for i in range(self.num_players)]
        # return self.observation[self.cur_state]
        
    def step(self, actions):
        #assert()
        self.num_iters += 1
        
        # action -> next move
        # check status
        # update observation
        for i in range(self.num_players):
            if actions[i] == 1:
                self.players_dir[i].rotate(90)
            elif actions[i] == 2:
                self.players_dir[i].rotate(-90)

            # head = self.players_head[i].copy()
            # head.move(self.players_dir[i])
            self.players_head[i] = self.players_head[i] + self.players_dir[i]
            self.players_body[i].add(self.players_head[i])
            # print("player {}, head {}".format(i, self.players_body[i]))
        
        rewards = np.zeros(self.num_players)
        done = False
        for i in range(self.num_players):
            head = self.players_head[i]
            if not inside(head):
                rewards[i] -= 1
                done = True
                print("{} outside of board".format())
            for j in range(self.num_players):
                if i != j and head in self.players_body[j]:
                    rewards[i] -= 1
                    rewards[j] += 1
                    done = True
                    print("{} in {}'s body".format(i,j))

        return self.observation, rewards, done, {'num_iters':self.num_iters}


    def render(self):
        print(self.observation)


def vecs_to_cycles(vecs):
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
        cycles = vecs_to_cycles(env.players_body)
        board = make_board(size=width, cycles=cycles, num_cycle=2)
        show_board(file_name=None, show=True, board=board, num_cycle=2)
        if done:
            break

    