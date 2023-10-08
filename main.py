import torch.optim as optim
import numpy as np

from state import State
from nets import Net, show_net
from train import train

#  Battle against random agents


def vs_random(net, n=100):
    results = {}
    for i in range(n):
        first_turn = i % 2 == 0
        turn = first_turn
        state = State()
        while not state.terminal():
            if turn:
                p, _ = net.predict(state, [])[-1]
                action = sorted([(a, p[a]) for a in state.legal_actions()], key=lambda x:-x[1])[0][0]
            else:
                action = np.random.choice(state.legal_actions())
            state.play(action)
            turn = not turn
        r = state.terminal_reward() if first_turn else -state.terminal_reward()
        results[r] = results.get(r, 0) + 1
    return results


def main():
    # Main algorithm of Muesli

    num_games = 50000
    num_games_one_epoch = 40
    num_sampled_actions = 10
    simulation_depth = 1

    C = 1

    net = Net()
    optimizer = optim.SGD(net.parameters(), lr=3e-4, weight_decay=3e-5, momentum=0.8)

    # Display battle results
    vs_random_sum = vs_random(net)
    print('vs_random   win: %d  draw: %d  lose: %d' %
          (vs_random_sum.get(1, 0), vs_random_sum.get(0, 0), vs_random_sum.get(-1, 0)))

    episodes = []
    result_distribution = {1: 0, 0: 0, -1: 0}

    for g in range(num_games):
        # Generate one episode
        state = State()

        features, policies, selected_actions, selected_action_features = [], [], [], []
        sampled_infos = []
        while not state.terminal():
            feature = state.feature()
            rp_root = net.representation.inference(feature)
            p_root, v_root = net.prediction.inference(rp_root)
            p_mask = np.zeros_like(p_root)
            p_mask[state.legal_actions()] = 1
            p_root *= p_mask
            p_root /= p_root.sum()

            features.append(feature)
            policies.append(p_root)

            actions, exadvs = [], []
            for i in range(num_sampled_actions):
                action = np.random.choice(np.arange(len(p_root)), p=p_root)
                actions.append(action)

                rp = rp_root
                qs = []
                for t in range(simulation_depth):
                    action_feature = state.action_feature(action)
                    rp = net.dynamics.inference(rp, action_feature)
                    p, v = net.prediction.inference(rp)
                    qs.append(-v if t % 2 == 0 else v)
                    action = np.random.choice(np.arange(len(p)), p=p)

                q = np.mean(qs)
                exadvs.append(np.exp(np.clip(q - v_root, -C, C)))

            exadv_sum = np.sum(exadvs)
            zs = []
            for exadv in exadvs:
                z = (1 + exadv_sum - exadv) / num_sampled_actions
                zs.append(z)
            sampled_infos.append({'a': actions, 'q': qs, 'exadv': exadvs, 'z': zs})

            # Select action with generated distribution, and then make a transition by that action
            selected_action = np.random.choice(np.arange(len(p_root)), p=p_root)
            selected_actions.append(selected_action)
            selected_action_features.append(state.action_feature(selected_action))
            state.play(selected_action)

        # reward seen from the first turn player
        reward = state.terminal_reward()
        result_distribution[reward] += 1
        episodes.append({
            'feature': features, 'action': selected_actions,
            'action_feature': selected_action_features, 'policy': policies,
            'reward': reward,
            'sampled_info': sampled_infos})

        if g % num_games_one_epoch == 0:
            print('game ', end='')
        print(g, ' ', end='')

        # Training of neural net
        if (g + 1) % num_games_one_epoch == 0:
            # Show the result distributiuon of generated episodes
            print('generated = ', sorted(result_distribution.items()))
            net = train(episodes, net, optimizer)
            vs_random_once = vs_random(net)
            print('vs_random   win: %d  draw: %d  lose: %d' %
                  (vs_random_once.get(1, 0), vs_random_once.get(0, 0), vs_random_once.get(-1, 0)))
            for r, n in vs_random_once.items():
                vs_random_sum[r] += n
            print('(total)           win: %d  draw: %d  lose: %d ' %
                  (vs_random_sum.get(1, 0), vs_random_sum.get(0, 0), vs_random_sum.get(-1, 0)))
            # show_net(net, State())
            # show_net(net, State().play('A1 C1 A2 C2'))
            # show_net(net, State().play('A1 B2 C3 B3 C1'))
            # show_net(net, State().play('B2 A2 A3 C1 B3'))
            # show_net(net, State().play('B2 A2 A3 C1'))
    print('finished')
    return net


if __name__ == '__main__':
    net = main()
    print('initial state')
    show_net(net, State())

    print('WIN by put')
    show_net(net, State().play('A1 C1 A2 C2'))

    print('LOSE by opponent\'s double')
    show_net(net, State().play('B2 A2 A3 C1 B3'))

    print('WIN through double')
    show_net(net, State().play('B2 A2 A3 C1'))

    # hard case: putting on A1 will cause double
    print('strategic WIN by following double')
    show_net(net, State().play('B1 A3'))
