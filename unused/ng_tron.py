import os
import random
import numpy as np
# import torch

from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import colors

class TronEnvironment:
    def __init__(self, config):
        self.config = config
        self.board  = self.clear_board()
        self.cycles = self.init_cycles()
        self.t      = 0
        self.fig = plt.figure('1')
        
    def init_cycles(self):
        '''Creates the cycles matrix of form:
            Head -> [[c1y, c1y, c1y, ...] <- Tail
                    [c1x, c1x, c1x, ...]
                    [c2y, c2y, c2y, ...]
                    [c2x, c2x, c2x, ...]]'''
        cycles = np.zeros((2*self.config.num_cycle, self.config.cycle_len))

        for c in range(1, self.config.num_cycle+1):
            y_2 = int(self.config.size/2)
            cycles[2*c-2, :] = np.array(range(y_2, y_2-self.config.cycle_len, -1))
            cycles[2*c-1, :] = int(c*self.config.size/(self.config.num_cycle+1))
            cycles = cycles.astype(int)
        print('Cycles matrix:\n', cycles)
        return cycles

    def clear_board(self):
        board = np.zeros((self.config.size, self.config.size))
        return board

    def valid_move(self, coord):
        '''Checks if the move is valid iff: 
            (1) in the board bounds.
            (2) the space is empty.'''
        return np.all(np.greater(coord, -1)) and \
            np.all(np.less(coord, self.config.size)) and \
            self.board[coord[0], coord[1]] == 0  # 0 means empty state

    def update_cycles(self):
        self.update_board()
        s_p = np.zeros((self.cycles.shape[0], 1))

        for c in range(1, self.config.num_cycle+1):
            head      = self.cycles[2*c-2:2*c, 0]
            behind    = self.cycles[2*c-2:2*c, 1]
            direction = head - behind
            forward   = head + direction

            if self.valid_move(coord=forward):
                selected = forward
            else:
                possible = []
                up    = [head[0]-1, head[1]]
                down  = [head[0]+1, head[1]]
                left  = [head[0], head[1]-1]
                right = [head[0], head[1]+1]
                if self.valid_move(coord=up): possible.append(up)
                if self.valid_move(coord=down): possible.append(down)
                if self.valid_move(coord=left): possible.append(left)
                if self.valid_move(coord=right): possible.append(right)
                possible = np.array(possible)
                print('possible shape:', possible.shape)
                print(possible)
                selected = possible[np.random.randint(0, high=possible.shape[0]), :]
            print('selected:', selected)
            s_p[2*c-2:2*c] = np.expand_dims(selected, axis=-1)

        self.cycles = np.append(s_p, self.cycles, axis=1).astype(int)
        if  not (self.t%self.config.lengthen_every == 0):
            self.cycles = np.delete(self.cycles, -1, axis=1)
        self.t += 1
        print('Cycles matrix:\n', self.cycles)

    def update_board(self):
        self.board = self.clear_board()
        for c in range(1, self.config.num_cycle+1):
            self.board[self.cycles[2*c-2, :], self.cycles[2*c-1, :]] = c

    def show_board(self):
        '''Makes a plot of the game board.'''
        plt.pcolor(self.board, cmap=self.config.cmap, edgecolors='k', linewidths=1)
        if config.show:
            plt.draw()
            plt.pause(0.025)
            plt.clf()
        if self.config.file_name is not None:
            self.fig.savefig(self.config.file_name)

class Config:
    def __init__(self):
        self.size           = 25
        self.num_cycle      = 5
        self.cycle_len      = 7
        self.show           = True
        self.time_steps     = 1000
        self.lengthen_every = 2
        self.cycle_colors   = ['white', 'red', 'blue', 'green', 'orange', 'purple']  # white is board color, others are cycles
        self.cmap           = colors.ListedColormap(self.cycle_colors[:self.num_cycle+1])
        self.file_name      = None

if __name__ == '__main__':
    config = Config()
    tron = TronEnvironment(config)
    
    tron.update_board()
    tron.show_board()
    for t in range(config.time_steps):
        tron.update_cycles()
        tron.update_board()
        tron.show_board()
