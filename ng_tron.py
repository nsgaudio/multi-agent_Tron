import os
import random
import numpy as np
# import torch

from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import colors


def show_board(file_name, board, num_cycle):
    '''Makes a plot of the game board.'''
    fig = plt.figure()

    cycle_colors = ['white','red', 'blue', 'green', 'orange', 'purple']
    cycle_colors = cycle_colors[:num_cycle+1]
    cmap = colors.ListedColormap(cycle_colors)
    plt.pcolor(board[::-1],cmap=cmap,edgecolors='k', linewidths=1)
    # plt.xticks(np.arange(0.5,10.5,step=1))
    # plt.yticks(np.arange(0.5,10.5,step=1))
    plt.show()
    if file_name is not None:
        fig.savefig(file_name)




if __name__ == '__main__':
    board = np.zeros((25, 25))

    # print(np.floor(board.shape[1]/4).type)

    board[np.floor(board.shape[1]/2).astype(int), np.floor(board.shape[1]/4).astype(int)] = 1
    board[np.floor(board.shape[1]/2).astype(int), np.floor(3*board.shape[1]/4).astype(int)] = 2



    show_board(None, board, 2)
