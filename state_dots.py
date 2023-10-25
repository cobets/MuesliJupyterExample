import time

from line_profiler_pycharm import profile

import numpy as np
from matplotlib.path import Path as PlotPath
from dotspath import find_paths

DISABLED = 1
ENABLED = 2
BLACK = 4
RED = 8


class State:
    def __init__(self, width=24, height=24):
        self.width = width
        self.height = height
        # board with dot states
        self.board = np.zeros((self.width, self.height), dtype=int)  # (x, y)
        # self.legal_actions_set = set([a for a in range(width * height)])
        self.trace = {BLACK: set(), RED: set()}
        self.player = BLACK
        self.opponent = RED

    def show_board(self):
        # output board
        result = np.zeros_like(self.board, dtype=object)
        for x in range(self.width):
            for y in range(self.height):
                s = ''
                v = self.board[x, y]
                if v & DISABLED:
                    s += 'D'
                if v & ENABLED:
                    s += 'E'
                if v & BLACK:
                    s += 'B'
                if v & RED:
                    s += 'R'
                result[x, y] = s
        return result

    def get_dot_neighbours(self, x, y):
        result = set()
        if x > 0:
            result.add((x - 1, y))
            if y > 0:
                result.add((x - 1, y - 1))
        if y > 0:
            result.add((x, y - 1))
            if x < self.width - 1:
                result.add((x + 1, y - 1))
        if x < self.width - 1:
            result.add((x + 1, y))
            if y < self.height - 1:
                result.add((x + 1, y + 1))
        if y < self.height - 1:
            result.add((x, y + 1))
            if x > 0:
                result.add((x - 1, y + 1))
        return result

    @profile
    def do_find_paths(self, paths, path, v, path_set):
        path.append(v)
        path_set.add(v)
        neighbours = self.get_dot_neighbours(*v) & self.trace[self.player]
        for nv in neighbours:
            if path[0] == nv:
                if sum(PlotPath(path).contains_points(list(self.trace[self.opponent]))) > 0:
                    paths.append(path)
            elif nv not in path_set:
                self.do_find_paths(paths, path.copy(), nv, path_set)

    def find_paths_py(self, v):
        result = list()
        self.do_find_paths(result, list(), v, set())
        return result

    def disable_area(self, area):
        for x, y in area:
            self.board[x, y] |= DISABLED
            self.board[x, y] &= ~ENABLED
            if (x, y) in self.trace[BLACK]:
                self.trace[BLACK].remove((x, y))
            if (x, y) in self.trace[RED]:
                self.trace[RED].remove((x, y))
            # v = x * self.width + y
            # if v in self.legal_actions_set:
            #     self.legal_actions_set.remove(v)

    def apply_to_board(self, paths):
        for path in paths:
            min_x, min_y = np.min(path, axis=0)
            max_x, max_y = np.max(path, axis=0)
            path_square_area = np.array([
                (x, y) for x in range(min_x, max_x+1) for y in range(min_y, max_y+1)
            ], dtype=int)
            point_in_path = PlotPath(path).contains_points(path_square_area)
            internal_area = path_square_area[point_in_path]
            internal_area = [(p[0], p[1]) for p in internal_area if (p[0], p[1]) not in path]
            self.disable_area(internal_area)

    def play(self, action):
        # print(f'empty={sum(sum(self.board == 0))}')
        # self.legal_actions_set.remove(action)
        x, y = action // self.width, action % self.height
        self.trace[self.player].add((x, y))
        self.board[x, y] = self.player | ENABLED
        # paths = self.find_paths_py((x, y))
        paths = find_paths(self.trace[self.player], self.trace[self.opponent], x, y)
        self.apply_to_board(paths)
        self.player, self.opponent = self.opponent, self.player
        return self

    def terminal(self):
        # terminal state check
        return not (0 in self.board)

    def terminal_reward(self):
        # terminal reward
        black_reward = 0
        red_reward = 0
        for bs in [self.board[a // self.width, a % self.height] for a in range(self.action_length())]:
            if bs & (BLACK | ENABLED) == (BLACK | ENABLED):
                black_reward += 1
            if bs & (RED | ENABLED) == (RED | ENABLED):
                red_reward += 1
        result = max(-1, min(1, black_reward - red_reward))
        print(result, time.strftime("%D %H:%M:%S", time.localtime()))
        return result

    def action_length(self):
        return self.width * self.height

    def legal_actions(self):
        # list of legal actions on each state
        return [a for a in range(self.action_length()) if self.board[a // self.width, a % self.height] == 0]
        # return list(self.legal_actions_set)

    def feature(self):
        # input tensor for neural net (state)
        return np.stack([self.board & self.player, self.board & self.opponent]).astype(np.float32)

    def action_feature(self, action):
        # input tensor for neural net (action)
        a = np.zeros((1, self.width, self.height), dtype=np.float32)
        a[0, action // self.width, action % self.height] = 1
        return a
