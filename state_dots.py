import numpy as np
from matplotlib.path import Path as PlotPath

DISABLED = 1
ENABLED = 2
BLACK = 4
RED = 8


class Path(list):
    def __init__(self, other=None):
        if other is not None:
            super().__init__(other)
        else:
            super().__init__()
        self.min_x = None
        self.max_x = None
        self.min_y = None
        self.max_y = None

    def set_bounds(self, item):
        x, y = item

        if len(self) == 0:
            self.min_x = x
            self.max_x = x
            self.min_y = y
            self.max_y = y
        else:
            if x < self.min_x:
                self.min_x = x
            if x > self.max_x:
                self.max_x = x
            if y < self.min_y:
                self.min_y = y
            if y > self.max_y:
                self.max_y = y

    def append(self, item):
        self.set_bounds(item)
        super().append(item)

    def copy_it(self):
        result = Path(self)
        result.min_x = self.min_x
        result.max_x = self.max_x
        result.min_y = self.min_y
        result.max_y = self.max_y
        return result


class State:
    def __init__(self, width=7, height=7):
        self.width = width
        self.height = height
        # board with dot states
        self.board = np.zeros((self.width, self.height), dtype=int)  # (x, y)
        self.trace = {BLACK: list(), RED: list()}
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
        result = list()
        if x > 0:
            result.append((x - 1, y))
            if y > 0:
                result.append((x - 1, y - 1))
        if y > 0:
            result.append((x, y - 1))
            if x < self.width - 1:
                result.append((x + 1, y - 1))
        if x < self.width - 1:
            result.append((x + 1, y))
            if y < self.height - 1:
                result.append((x + 1, y + 1))
        if y < self.height - 1:
            result.append((x, y + 1))
            if x > 0:
                result.append((x - 1, y + 1))
        return result

    def do_find_paths(self, paths, path, x, y):
        path.append((x, y))
        neighbours = [
            (nx, ny) for (nx, ny) in self.get_dot_neighbours(x, y)
            if self.board[nx, ny] & (self.player | ENABLED) == (self.player | ENABLED)
        ]
        for nx, ny in neighbours:
            if path[0] == (nx, ny):
                if path.max_x - path.min_x > 1 and path.max_y - path.min_y > 1:
                    for ot in self.trace[self.opponent]:
                        if path.min_x < ot[0] < path.max_x and path.min_y < ot[1] < path.max_y:
                            # path loops
                            paths.append(path)
                            break
            elif (nx, ny) not in path:
                # path optimization based on pre pre dot
                if len(path) < 2 or abs(nx - path[-2][0]) > 1 or abs(ny - path[-2][1]) > 1:
                    self.do_find_paths(paths, path.copy_it(), nx, ny)

    def find_paths(self, x, y):
        result = list()
        self.do_find_paths(result, Path(), x, y)
        return result

    def disable_area(self, area):
        for x, y in area:
            self.board[x, y] |= DISABLED
            self.board[x, y] &= ~ENABLED

    def apply_to_board(self, paths):
        for path in paths:
            path_square_area = np.array([
                (x, y) for x in range(path.min_x, path.max_x+1) for y in range(path.min_y, path.max_y+1)
            ], dtype=int)
            point_in_path = PlotPath(path).contains_points(path_square_area)
            internal_area = path_square_area[point_in_path]
            internal_area = [(p[0], p[1]) for p in internal_area if (p[0], p[1]) not in path]
            for x, y in internal_area:
                if self.board[x, y] & (self.opponent | ENABLED) == (self.opponent | ENABLED):
                    self.disable_area(internal_area)
                    break

    def play(self, action):
        x, y = action // self.width, action % self.height
        self.trace[self.player].append((x, y))
        self.board[x, y] = self.player | ENABLED
        paths = self.find_paths(x, y)
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
        # print(max(-1, min(1, black_reward - red_reward)))
        return max(-1, min(1, black_reward - red_reward))

    def action_length(self):
        return self.width * self.height

    def legal_actions(self):
        # list of legal actions on each state
        return [a for a in range(self.action_length()) if self.board[a // self.width, a % self.height] == 0]

    def feature(self):
        # input tensor for neural net (state)
        return np.stack([self.board & self.player, self.board & self.opponent]).astype(np.float32)

    def action_feature(self, action):
        # input tensor for neural net (action)
        a = np.zeros((1, self.width, self.height), dtype=np.float32)
        a[0, action // self.width, action % self.height] = 1
        return a
