import os
import random
import numpy as np
# import torch

from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import colors

class Config:
    def __init__(self):
        self.size           = 25
        self.num_cycle      = 4
        self.cycle_len      = 7
        self.show           = True
        self.time_steps     = 1000
        self.lengthen_every = 2
        self.cycle_colors   = ['white', 'red', 'blue', 'green', 'orange', 'purple']  # white is board color, others are cycles
        self.cmap           = colors.ListedColormap(self.cycle_colors[:self.num_cycle+1])
        self.file_name      = None

def init_board(config):
    board = np.zeros((config.size, config.size))
    return board

def init_cycles(config):
    '''Creates the cycles matrix of form:
        Head -> [[c1y, c1y, c1y, ...] <- Tail
                 [c1x, c1x, c1x, ...]
                 [c2y, c2y, c2y, ...]
                 [c2x, c2x, c2x, ...]]'''
    cycles = np.zeros((2*config.num_cycle, config.cycle_len))

    for c in range(1, config.num_cycle+1):
        y_2 = int(config.size/2)
        cycles[2*c-2, :] = np.array(range(y_2, y_2-config.cycle_len, -1))
        cycles[2*c-1, :] = int(c*config.size/(config.num_cycle+1))
        cycles = cycles.astype(int)
    print('Cycles matrix:\n', cycles)
    return cycles

def valid_move(config, coord, board):
    '''Checks if the move is valid iff: 
        (1) in the board bounds.
        (2) the space is empty.'''
    return np.all(np.greater(coord, -1)) and \
           np.all(np.less(coord, config.size)) and \
           board[coord[0], coord[1]] == 0  # 0 means empty state

def update_cycles(config, cycles, t):
    board = make_board(config, cycles=cycles)
    s_p = np.zeros((cycles.shape[0], 1))

    for c in range(1, config.num_cycle+1):
        head  = cycles[2*c-2:2*c, 0]
        behind = cycles[2*c-2:2*c, 1]
        direction = head - behind
        forward = head + direction

        if valid_move(config, coord=forward, board=board):
            selected = forward
        else:
            possible = []
            up    = [head[0]-1, head[1]]
            down  = [head[0]+1, head[1]]
            left  = [head[0], head[1]-1]
            right = [head[0], head[1]+1]
            if valid_move(config, coord=up, board=board): possible.append(up)
            if valid_move(config, coord=down, board=board): possible.append(down)
            if valid_move(config, coord=left, board=board): possible.append(left)
            if valid_move(config, coord=right, board=board): possible.append(right)
            possible = np.array(possible)
            print('possible shape:', possible.shape)
            print(possible)
            selected = possible[np.random.randint(0, high=possible.shape[0]), :]
        print('selected:', selected)
        s_p[2*c-2:2*c] = np.expand_dims(selected, axis=-1)

    cycles = np.append(s_p, cycles, axis=1).astype(int)
    if  not (t%config.lengthen_every == 0):
        cycles = np.delete(cycles, -1, axis=1)
    print('Cycles matrix:\n', cycles)
    return cycles

def make_board(config, cycles):
    board = init_board(config)
    for c in range(1, config.num_cycle+1):
        board[cycles[2*c-2, :], cycles[2*c-1, :]] = c
    return board

def show_board(config, board):
    '''Makes a plot of the game board.'''
    fig = plt.figure()
    plt.pcolor(board, cmap=config.cmap, edgecolors='k', linewidths=1)
    if config.show:
        plt.draw()
        plt.pause(0.025)
        plt.clf()
    if config.file_name is not None:
        fig.savefig(config.file_name)

if __name__ == '__main__':
    config = Config()
    cycles = init_cycles(config)
    board  = make_board(config, cycles=cycles)
    show_board(config, board=board)

    for t in range(config.time_steps):
        cycles = update_cycles(config, cycles, t=t)
        board  = make_board(config, cycles=cycles)
        show_board(config, board=board)
