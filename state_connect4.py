import gym
import numpy as np
import gym_connect4


class State:
    def __init__(self, width=7, height=6):
        self.width = width
        self.height = height
        self.env = gym.make('Connect4-v0', height=self.height, width=7, connect=4)
        self.env.reset()

    def __str__(self):
        # output board
        self.env.render()

    def play(self, action):
        self.env.step(action)
        return self

    def terminal(self):
        # terminal state check
        return len(self.env.get_moves()) == 0

    def terminal_reward(self):
        # terminal reward
        if self.env.winner == 0:
            return -1
        elif self.env.winner == 1:
            return 1
        else:
            return 0

    def action_length(self):
        return self.env.width

    def legal_actions(self):
        # list of legal actions
        return self.env.get_moves()

    def feature(self):
        # input tensor for neural net (state)
        f = self.env.filter_observation_player_perspective(self.env.current_player)
        return f[1:3].astype(np.float32)

    def action_feature(self, action):
        # input tensor for neural net (action)
        a = np.zeros((1, self.env.width, self.env.height), dtype=np.float32)
        a[0, action, 0] = 1
        return a
