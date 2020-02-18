import os
import random
import numpy as np
# import torch

from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import colors


def init_board(size):
    board = np.zeros((size, size))
    return board


def init_cycles(size, num_cycle, cycle_len):
    cycles = np.zeros((2*num_cycle, cycle_len))

    for c in range(1, num_cycle+1):
        y_2 = int(size/2)
        cycles[2*c-2, :] = np.array(range(y_2, y_2-cycle_len, -1))
        cycles[2*c-1, :] = int(c*size/(num_cycle+1))
        cycles = cycles.astype(int)
    
    print(cycles)
    return cycles


def show_board(file_name, cycles, board, num_cycle):
    '''Makes a plot of the game board.'''
    fig = plt.figure()

    cycle_colors = ['white','red', 'blue', 'green', 'orange', 'purple']  # white is board color, others are cycles
    cycle_colors = cycle_colors[:num_cycle+1]
    cmap = colors.ListedColormap(cycle_colors)

    for c in range(1, num_cycle+1):
        print(2*c-2)
        print(2*c-1)
        board[cycles[2*c-2, :], cycles[2*c-1, :]] = c
    plt.pcolor(board,cmap=cmap,edgecolors='k', linewidths=1)
    plt.show()
    if file_name is not None:
        fig.savefig(file_name)

if __name__ == '__main__':
    size      = 25
    num_cycle = 2
    cycle_len = 7

    board = init_board(size=size)
    cycles = init_cycles(size=size, num_cycle=num_cycle, cycle_len=cycle_len)
    show_board(file_name=None, cycles=cycles, board=board, num_cycle=num_cycle)

