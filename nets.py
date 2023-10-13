# Small neural nets with PyTorch

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from state import State


class Conv(nn.Module):
    def __init__(self, filters0, filters1, kernel_size, bn=False):
        super().__init__()
        self.conv = nn.Conv2d(filters0, filters1, kernel_size, stride=1, padding=kernel_size//2, bias=False)
        self.bn = None
        if bn:
            self.bn = nn.BatchNorm2d(filters1)

    def forward(self, x):
        h = self.conv(x)
        if self.bn is not None:
            h = self.bn(h)
        return h


class ResidualBlock(nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.conv = Conv(filters, filters, 3, True)

    def forward(self, x):
        return F.relu(x + (self.conv(x)))


num_filters = 16
num_blocks = 4


class Representation(nn.Module):
    ''' Conversion from observation to inner abstract state '''
    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape
        self.board_size = self.input_shape[1] * self.input_shape[2]

        self.layer0 = Conv(self.input_shape[0], num_filters, 3, bn=True)
        self.blocks = nn.ModuleList([ResidualBlock(num_filters) for _ in range(num_blocks)])

    def forward(self, x):
        h = F.relu(self.layer0(x))
        for block in self.blocks:
            h = block(h)
        return h

    def inference(self, x):
        self.eval()
        with torch.no_grad():
            rp = self(torch.from_numpy(x).unsqueeze(0))
        return rp.cpu().numpy()[0]


class Prediction(nn.Module):
    ''' Policy and value prediction from inner abstract state '''
    def __init__(self, action_shape):
        super().__init__()
        self.board_size = np.prod(action_shape[1:])
        self.action_size = action_shape[0] * self.board_size

        self.conv_p1 = Conv(num_filters, 4, 1, bn=True)
        self.conv_p2 = Conv(4, 1, 1)

        self.conv_v = Conv(num_filters, 4, 1, bn=True)
        self.fc_v = nn.Linear(self.board_size * 4, 1, bias=False)

    def forward(self, rp):
        h_p = F.relu(self.conv_p1(rp))
        h_p = self.conv_p2(h_p).view(-1, self.action_size)

        h_v = F.relu(self.conv_v(rp))
        h_v = self.fc_v(h_v.view(-1, self.board_size * 4))

        # range of value is -1 ~ 1
        return F.softmax(h_p, dim=-1), torch.tanh(h_v)

    def inference(self, rp):
        self.eval()
        with torch.no_grad():
            p, v = self(torch.from_numpy(rp).unsqueeze(0))
        return p.cpu().numpy()[0], v.cpu().numpy()[0][0]

class Dynamics(nn.Module):
    '''Abstract state transition'''
    def __init__(self, rp_shape, act_shape):
        super().__init__()
        self.rp_shape = rp_shape
        self.layer0 = Conv(rp_shape[0] + act_shape[0], num_filters, 3, bn=True)
        self.blocks = nn.ModuleList([ResidualBlock(num_filters) for _ in range(num_blocks)])

    def forward(self, rp, a):
        h = torch.cat([rp, a], dim=1)
        h = self.layer0(h)
        for block in self.blocks:
            h = block(h)
        return h

    def inference(self, rp, a):
        self.eval()
        with torch.no_grad():
            rp = self(torch.from_numpy(rp).unsqueeze(0), torch.from_numpy(a).unsqueeze(0))
        return rp.cpu().numpy()[0]


class Net(nn.Module):
    # Whole net
    def __init__(self):
        super().__init__()
        state = State()
        input_shape = state.feature().shape
        action_shape = state.action_feature(0).shape
        rp_shape = (num_filters, *input_shape[1:])

        self.representation = Representation(input_shape)
        self.prediction = Prediction(action_shape)
        self.dynamics = Dynamics(rp_shape, action_shape)

    def predict(self, state0, path):
        # Predict p and v from original state and path
        outputs = []
        x = state0.feature()
        rp = self.representation.inference(x)
        outputs.append(self.prediction.inference(rp))
        for action in path:
            a = state0.action_feature(action)
            rp = self.dynamics.inference(rp, a)
            outputs.append(self.prediction.inference(rp))
        return outputs


def show_net(net, state):
    # Display policy (p) and value (v)
    print(state)
    p, v = net.predict(state, [])[-1]
    print('p = ')
    print((p * 1000).astype(int).reshape((-1, *net.representation.input_shape[1:3])))
    print('v = ', v)
    print()
