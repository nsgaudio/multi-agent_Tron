import os
import random
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import colors

from ng_tron import *

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

    directions = np.arange(0, num_cycle)
    directions = np.mod(directions, 2)

    print(directions)
    
    return cycles, directions

def move(cycles, directions, board, cycle_num, action):
    new_point = [cycles[2 * (cycle_num-1)][0], cycles[2 * (cycle_num-1)+1][0]]
    if action == 0: #UP
        if directions[cycle_num-1] != 2: # Not going down
            new_point[0] -= 1
            directions[cycle_num-1] = action
        else:
            new_point[0] += 1                
    elif action == 1: #RIGHT
        if directions[cycle_num-1] != 3: # Not going left
            new_point[1] += 1
            directions[cycle_num-1] = action
        else:
            new_point[1] -= 1
    elif action == 2: #DOWN
        if directions[cycle_num-1] != 0: # Not going up
            new_point[0] += 1
            directions[cycle_num-1] = action            
        else:
            new_point[0] -= 1 
    elif action == 3: #LEFT
        if directions[cycle_num-1] != 1: # Not going right
            new_point[1] -= 1
            directions[cycle_num-1] = action
        else:
            new_point[1] += 1

    return new_point

def check_collsion(new_point, board):
    if new_point[0] == 0 or new_point[1] == 0 or new_point[0] = board.shape[0] or new_point[1] = board.shape[1]:
        return True
    else:
        return False

if __name__ == '__main__':
    size      = 25
    num_cycle = 4
    cycle_len = 7

    board = init_board(size=size)
    cycles, directions = init_cycles(size=size, num_cycle=num_cycle, cycle_len=cycle_len)
    show_board(file_name=None, cycles=cycles, board=board, num_cycle=num_cycle)
