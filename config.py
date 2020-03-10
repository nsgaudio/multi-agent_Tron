import numpy as np

from matplotlib import colors
class Config():
    def __init__(self):
        ########## Env config #########

        # init
        self.board_shape     = [40, 40] # y, x
        self.num_players     = 4
        
        # snake dynamics
        self.init_len        = 5
        self.lengthen_every  = 1

        # Visualization
        self.show            = False
        self.colors          = ['white', 'red', 'blue', 'green', 'orange', 'purple']  # white is board color, others are cycles
        self.cmap            = colors.ListedColormap(self.colors[: self.num_players + 1])
        # self.delay           = 0.25
        self.delay           = 0.00025
        # other
        self.filename        = None
        self.verbal          = False

        # reward
        self.win             = 1.0
        self.lose            = -1.0
        self.time_reward     = -0.001
        self.win_type        = 'all'

        # team
        self.teams = np.array([[1,2], [3,4]]) 
            # Ex. input of env.step(actions), actions = [a1, a2, a3, a4], a in {0,1,2,3}
            # a1, a2 are actions in the same team; a3, a4 are actions in the other team
            # assume a1, a2 are from agent(s), a3, a4 are from computer
        self.indep_Q = False
            # False: 1 agent controls 2 players
            # True : 1 agent controls 1 players, a total of 2 agents

        ##############################
        # hard-coded policy
        self.hcp_eps         = 0.3
        ##############################

        ######## Training parameters #########
        self.REPLAY_MEMORY_CAP       = 10000
        self.BATCH_SIZE              = 16
        self.GAMMA                   = 1.0
        self.EPS_START               = 0.7
        self.EPS_END                 = 0.05
        self.EPS_DECAY               = 10000
        self.TARGET_UPDATE_FREQUENCY = 20
        self.MODEL_SAVE_FREQUENCY    = 100

        self.INPUT_FRAME_NUM         = 4

        self.KERNEL_SIZE             = 5
        self.STRIDE                  = 2

        self.NUM_EPISODES            = 100000

        self.MODEL_EVAL_FREQUENCY    = 100
        self.EVAL_EPISODE            = 100

        self.with_adjustment         = True

        # None or 'name.pth'
        # self.load_model              = 'exp3_116k.pth'  # start training with pre-trained model
        self.load_model              = None  # start training with pre-trained model
        # self.load_opponent            = 'exp3_116k.pth'  # use pre-trained model instead of a hard-coded policy
        self.load_opponent           = None  # use pre-trained model instead of a hard-coded policy