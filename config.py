import numpy as np

from matplotlib import colors
class config():
    def __init__(self):
        ########## Env config #########

        # init
        self.board_size     = (25, 25)
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

  
