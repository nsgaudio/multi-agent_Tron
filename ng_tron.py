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


def valid_move(size, coord, board):
    '''Checks if the move is valid: 
        (1) in the board bounds.
        (2) the space id empty.'''
    return np.all(np.greater(coord, -1)) and \
           np.all(np.less(coord, size)) and \
           board[coord[0], coord[1]] == 0  # 0 means empty state


def init_cycles(size, num_cycle, cycle_len):
    '''Creates the cycles matrix of form:
        Head -> [[c1y, c1y, c1y, ...] <- Tail
                 [c1x, c1x, c1x, ...]
                 [c2y, c2y, c2y, ...]
                 [c2x, c2x, c2x, ...]]'''
    cycles = np.zeros((2*num_cycle, cycle_len))

    for c in range(1, num_cycle+1):
        y_2 = int(size/2)
        cycles[2*c-2, :] = np.array(range(y_2, y_2-cycle_len, -1))
        cycles[2*c-1, :] = int(c*size/(num_cycle+1))
        cycles = cycles.astype(int)
    print('Cycles matrix:\n', cycles)
    return cycles


def update_cycles(size, cycles, num_cycle):
    board = make_board(size=size, cycles=cycles, num_cycle=num_cycle)
    s_p = np.zeros((cycles.shape[0], 1))

    for c in range(1, num_cycle+1):
        head  = cycles[2*c-2:2*c, 0]
        behind = cycles[2*c-2:2*c, 1]
        direction = head - behind
        forward = head + direction

        if valid_move(size=size, coord=forward, board=board):
            selected = forward
        else:
            possible = []
            up    = [head[0]-1, head[1]]
            down  = [head[0]+1, head[1]]
            left  = [head[0], head[1]-1]
            right = [head[0], head[1]+1]
            if valid_move(size=size, coord=up, board=board): possible.append(up)
            if valid_move(size=size, coord=down, board=board): possible.append(down)
            if valid_move(size=size, coord=left, board=board): possible.append(left)
            if valid_move(size=size, coord=right, board=board): possible.append(right)
            possible = np.array(possible)
            print('possible shape:', possible.shape)
            print(possible)
            selected = possible[np.random.randint(0, high=possible.shape[0]), :]
        print('selected:', selected)
        s_p[2*c-2:2*c] = np.expand_dims(selected, axis=-1)

    cycles = np.append(s_p, cycles, axis=1).astype(int)
    cycles = np.delete(cycles, -1, axis=1)
    print('Cycles matrix:\n', cycles)
    return cycles


def make_board(size, cycles, num_cycle):
    board = init_board(size=size)
    for c in range(1, num_cycle+1):
        board[cycles[2*c-2, :], cycles[2*c-1, :]] = c
    return board


def show_board(file_name, show, board, num_cycle):
    '''Makes a plot of the game board.'''
    fig = plt.figure()
    cycle_colors = ['white', 'red', 'blue', 'green', 'orange', 'purple']  # white is board color, others are cycles
    cycle_colors = cycle_colors[:num_cycle+1]
    cmap = colors.ListedColormap(cycle_colors)
    plt.pcolor(board, cmap=cmap, edgecolors='k', linewidths=1)
    if show:
        plt.draw()
        plt.pause(0.25)
        plt.clf()
    if file_name is not None:
        fig.savefig(file_name)


if __name__ == '__main__':
    size       = 25
    num_cycle  = 4
    cycle_len  = 7
    show       = True
    time_steps = 100

    cycles = init_cycles(size=size, num_cycle=num_cycle, cycle_len=cycle_len)
    board = make_board(size=size, cycles=cycles, num_cycle=num_cycle)
    show_board(file_name=None, show=show, board=board, num_cycle=num_cycle)

    for t in range(time_steps):
        cycles = update_cycles(size, cycles, num_cycle)
        board = make_board(size=size, cycles=cycles, num_cycle=num_cycle)
        show_board(file_name=None, show=True, board=board, num_cycle=num_cycle)
