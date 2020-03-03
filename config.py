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
        self.show            = True
        self.colors          = ['white', 'red', 'blue', 'green', 'orange', 'purple']  # white is board color, others are cycles
        self.cmap            = colors.ListedColormap(self.colors[: self.num_players + 1])
        self.delay           = 0.25 
        # other
        self.filename        = None

        # reward
        self.win             = 1.0
        self.lose            = -1.0
        self.time_reward     = 0.001
        self.win_type        = 'all'

        # team
        self.teams = np.array([[1,2], [3,4]])
        self.indep_Q = False

        ##############################
        # hard-coded policy
        self.hcp_eps         = 0.5
        ##############################

        ########## Q learning config #########
        self.time_steps      = 1000

        ######## Training parameters #########
        self.REPLAY_MEMORY_CAP = 10000
        self.BATCH_SIZE        = 128
        self.GAMMA             = 0.999
        self.EPS_START         = 0.9
        self.EPS_END           = 0.05
        self.EPS_DECAY         = 200
        self.TARGET_UPDATE     = 10

        self.INPUT_FRAME_NUM   = 4

        self.KERNEL_SIZE       = 5
        self.STRIDE            = 2

        self.NUM_EPISODES      = 50
