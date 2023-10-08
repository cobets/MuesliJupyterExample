import numpy as np

BLACK, WHITE = 1, -1  # first turn or second turn player


class State:
    # Board implementation of Tic-Tac-Toe
    X, Y = 'ABC', '123'
    C = {0: '_', BLACK: 'O', WHITE: 'X'}

    def __init__(self):
        self.board = np.zeros((3, 3))  # (x, y)
        self.color = 1
        self.win_color = 0
        self.record = []

    def action2str(self, a):
        return self.X[a // 3] + self.Y[a % 3]

    def str2action(self, s):
        return self.X.find(s[0]) * 3 + self.Y.find(s[1])

    def record_string(self):
        return ' '.join([self.action2str(a) for a in self.record])

    def __str__(self):
        # output board.
        s = '   ' + ' '.join(self.Y) + '\n'
        for i in range(3):
            s += self.X[i] + ' ' + ' '.join([self.C[self.board[i, j]] for j in range(3)]) + '\n'
        s += 'record = ' + self.record_string()
        return s

    def play(self, action):
        # state transition function
        # action is position integer (0~8) or string representation of action sequence
        if isinstance(action, str):
            for astr in action.split():
                self.play(self.str2action(astr))
            return self

        x, y = action // 3, action % 3
        self.board[x, y] = self.color

        # check whether 3 stones are on the line
        if self.board[x, :].sum() == 3 * self.color \
                or self.board[:, y].sum() == 3 * self.color \
                or (x == y and np.diag(self.board, k=0).sum() == 3 * self.color) \
                or (x == 2 - y and np.diag(self.board[::-1, :], k=0).sum() == 3 * self.color):
            self.win_color = self.color

        self.color = -self.color
        self.record.append(action)
        return self

    def terminal(self):
        # terminal state check
        return self.win_color != 0 or len(self.record) == 3 * 3

    def terminal_reward(self):
        # terminal reward
        return self.win_color

    def action_length(self):
        return 3 * 3

    def legal_actions(self):
        # list of legal actions on each state
        return [a for a in range(3 * 3) if self.board[a // 3, a % 3] == 0]

    def feature(self):
        # input tensor for neural net (state)
        return np.stack([self.board == self.color, self.board == -self.color]).astype(np.float32)

    def action_feature(self, action):
        # input tensor for neural net (action)
        a = np.zeros((1, 3, 3), dtype=np.float32)
        a[0, action // 3, action % 3] = 1
        return a
