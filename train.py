# Training of neural net

import torch
import numpy as np

from state import State

batch_size = 32
num_steps = 100
K = 1


def gen_target(state, ep):
    # Generate inputs and targets for training
    # path, reward, observation, action, policy
    ep_length = len(ep['feature'])
    turn_idx = np.random.randint(ep_length)

    x = ep['feature'][turn_idx]
    ps, rs, acts, axs = [], [], [], []
    sas, seas, szs = [], [], []
    for t in range(turn_idx, turn_idx + K + 1):
        if t < ep_length:
            p = ep['policy'][t]
            a = ep['action'][t]
            ax = ep['action_feature'][t]
            sa = ep['sampled_info'][t]['a']
            sea = ep['sampled_info'][t]['exadv']
            sz = ep['sampled_info'][t]['z']
        else:  # state after finishing game
            p = np.zeros_like(ep['policy'][-1])
            # random action selection
            a = np.random.randint(state.action_length())
            ax = state.action_feature(a)
            sa = np.random.randint(state.action_length(), size=len(sa))
            sea = np.ones_like(sea)
            sz = np.ones_like(sz)

        rs.append([ep['reward'] if t % 2 == 0 else -ep['reward']])
        acts.append([a])
        axs.append(ax)
        ps.append(p)
        sas.append(sa)
        seas.append(sea)
        szs.append(sz)

    return x, rs, acts, axs, ps, sas, seas, szs


def train(episodes, net, optimizer):
    # Train neural net
    pg_loss_sum, cmpo_loss_sum, v_loss_sum = 0, 0, 0
    net.train()
    state = State()

    for _ in range(num_steps):
        targets = [gen_target(state, episodes[np.random.randint(len(episodes))]) for j in range(batch_size)]
        x, r, a, ax, p_prior, sa, sea, sz = zip(*targets)
        x = torch.from_numpy(np.array(x))
        r = torch.from_numpy(np.array(r))
        a = torch.from_numpy(np.array(a))
        ax = torch.from_numpy(np.array(ax))
        p_prior = torch.from_numpy(np.array(p_prior))
        sa = torch.from_numpy(np.array(sa))
        sea = torch.from_numpy(np.array(sea))
        sz = torch.from_numpy(np.array(sz))

        # Compute losses for k (+ current) steps
        ps, vs = [], []
        rp = net.representation(x)
        for t in range(K + 1):
            p, v = net.prediction(rp)
            ps.append(p)
            vs.append(v)
            rp = net.dynamics(rp, ax[:, t])

        cmpo_loss, v_loss = 0, 0
        for t in range(K, -1, -1):
            # sz[:, t] == z cmpo(s)
            cmpo_loss += -torch.mean(sea[:, t] / sz[:, t] * torch.log(ps[t].gather(1, sa[:, t].type(torch.int64))), dim=1).sum()
            v_loss += torch.sum(((vs[t] - r[:, t]) ** 2) / 2)

        p_selected = ps[0].gather(1, a[:, 0].type(torch.int64))
        p_selected_prior = p_prior[:, 0].gather(1, a[:, 0].type(torch.int64))
        clipped_rho = torch.clamp(p_selected.detach() / p_selected_prior, 0, 1)
        pg_loss = torch.sum(-clipped_rho * torch.log(p_selected) * (r[:, 0] - vs[0]))

        pg_loss_sum += pg_loss.item()
        cmpo_loss_sum += cmpo_loss.item() / (K + 1)
        v_loss_sum += v_loss.item() / (K + 1)

        optimizer.zero_grad()
        (pg_loss + cmpo_loss + v_loss).backward()
        optimizer.step()

    data_count = num_steps * batch_size
    print('pg_loss %f cmpo_loss %f v_loss %f' % (
    pg_loss_sum / data_count, cmpo_loss_sum / data_count, v_loss_sum / data_count))
    return net
