import os
import random
import numpy as np
# import torch

from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import colors


def show_board(file_name, board):
    '''Plots the points on the plane (red) and off the plane.'''
    fig = plt.figure()

    cmap = colors.ListedColormap(['Blue','red'])

    plt.pcolor(board[::-1],cmap=cmap,edgecolors='k', linewidths=1)
    # plt.xticks(np.arange(0.5,10.5,step=1))
    # plt.yticks(np.arange(0.5,10.5,step=1))
    plt.show()
    if file_name is not None:
        fig.savefig(file_name)




if __name__ == '__main__':
    # board = np.zeros((10, 10))
    board = np.eye(25)

    show_board(None, board)
