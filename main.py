from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch
import numpy as np

from nets import Net, show_net
from train import train

#  Battle against random agents


def vs_random(net, state_class, n=100):
    results = {}
    for i in range(n):
        first_turn = i % 2 == 0
        turn = first_turn
        state = state_class()
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


def main(state_class, checkpoint, state_width, state_height, n_vs_random, state_dict_saver):
    # Main algorithm of Muesli

    num_games = 50000
    num_games_one_epoch = 40
    num_sampled_actions = 10
    simulation_depth = 1

    C = 1

    writer = SummaryWriter()

    if checkpoint is not None:
        state_height = checkpoint['state_height']
        state_width = checkpoint['state_width']

    class StateClassSized(state_class):
        def __init__(self):
            super(StateClassSized, self).__init__(state_width, state_height)

    net = Net(StateClassSized)
    optimizer = optim.SGD(net.parameters(), lr=3e-4, weight_decay=3e-5, momentum=0.8)

    continue_from_game = 0
    if checkpoint is not None:
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        continue_from_game = checkpoint['game'] + 1

    # Display battle results
    vs_random_once = vs_random(net, StateClassSized, n_vs_random)

    writer.add_scalars(
        'train/battle',
        {
            'win': vs_random_once.get(1, 0),
            'draw': vs_random_once.get(0, 0),
            'lose': vs_random_once.get(-1, 0)
        },
        -1
    )

    episodes = []
    result_distribution = {1: 0, 0: 0, -1: 0}

    training_step = 0

    for g in range(continue_from_game, num_games):
        # Generate one episode
        state = StateClassSized()

        features, policies, selected_actions, selected_action_features = [], [], [], []
        sampled_infos = []
        while not state.terminal():
            feature = state.feature()
            rp_root = net.representation.inference(feature)  # rp_root == s
            p_root, v_root = net.prediction.inference(rp_root)  # v_root == v p prior(s)
            p_mask = np.zeros_like(p_root)
            p_mask[state.legal_actions()] = 1
            p_root *= p_mask
            p_root /= p_root.sum()

            features.append(feature)
            policies.append(p_root)

            actions, exadvs = [], []
            for i in range(num_sampled_actions):  # num_sampled_actions == N
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

                q = np.mean(qs)  # q == q p prior(s, a)
                exadvs.append(np.exp(np.clip(q - v_root, -C, C)))  # q - v_root == adv(s, a)

            exadv_sum = np.sum(exadvs)
            zs = []
            for exadv in exadvs:
                z = (1 + exadv_sum - exadv) / num_sampled_actions  # == z cmpo i (s)
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

        # Training of neural net
        if (g + 1) % num_games_one_epoch == 0:
            training_step += 1

            # Show the result distribution of generated episodes
            print(f'game: {g} generated: {sorted(result_distribution.items())}')

            net, pg_loss, cmpo_loss, v_loss = train(episodes, net, optimizer, StateClassSized)

            writer.add_scalars(
                'train/loss',
                {
                    'pg_loss': pg_loss,
                    'cmpo_loss': cmpo_loss,
                    'v_loss': v_loss
                },
                g
            )

            vs_random_once = vs_random(net, StateClassSized, n_vs_random)

            writer.add_scalars(
                'train/battle',
                {
                    'win': vs_random_once.get(1, 0),
                    'draw': vs_random_once.get(0, 0),
                    'lose': vs_random_once.get(-1, 0)
                },
                g
            )

            if state_dict_saver is not None:
                if training_step % state_dict_saver['interval'] == 0:
                    torch.save(
                        {
                            'game': g,
                            'epoch': training_step,
                            'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'state_width': state_width,
                            'state_height': state_height

                        },
                        state_dict_saver['path'] + f'-checkpoint-{state_width}-{state_height}-{g}.tar'
                    )

            # show_net(net, State())
            # show_net(net, State().play('A1 C1 A2 C2'))
            # show_net(net, State().play('A1 B2 C3 B3 C1'))
            # show_net(net, State().play('B2 A2 A3 C1 B3'))
            # show_net(net, State().play('B2 A2 A3 C1'))
    print('finished')
    return net


def test_tic_tac_toe(net, state_class):
    print('initial state')
    show_net(net, state_class())

    print('WIN by put')
    show_net(net, state_class().play('A1 C1 A2 C2'))

    print('LOSE by opponent\'s double')
    show_net(net, state_class().play('B2 A2 A3 C1 B3'))

    print('WIN through double')
    show_net(net, state_class().play('B2 A2 A3 C1'))

    # hard case: putting on A1 will cause double
    print('strategic WIN by following double')
    show_net(net, state_class().play('B1 A3'))


if __name__ == '__main__':
    from state_dots import State as StateClass
    # from state import State as StateClass
    # checkpoint = torch.load('d:/cobets/github/MuesliJupyterExample/models/muesli-dots-checkpoint-3-3-79.tar')
    main(
        state_class=StateClass,
        checkpoint=None,
        state_width=24,
        state_height=24,
        n_vs_random=25,
        state_dict_saver={
            'path': 'D:/cobets/github/MuesliJupyterExample/models/muesli-dots',
            'interval': 2
        }
    )
