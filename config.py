import numpy as np

from matplotlib import colors
class Config():
    def __init__(self):
        ########## Env config #########

        # init
        self.board_size     = [35, 35]
        self.num_players    = 3
        
        # snake dynamics
        self.init_len       = 5
        self.lengthen_every = 2

        # Visualization
        self.show           = True
        self.colors   = ['white', 'red', 'blue', 'green', 'orange', 'purple']  # white is board color, others are cycles
        self.cmap           = colors.ListedColormap(self.colors[: self.num_players + 1])

        # other
        self.file_name      = None
        ##############################

        ########## Q learning config #########
        self.time_steps     = 1000

        ######## Training parameters #########
        self.BATCH_SIZE = 128
        self.GAMMA = 0.999
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 200
        self.TARGET_UPDATE = 10