import torch
from enum import Enum


class Config:
    class Optimizer(Enum):
        SGD = 'sgd'
        ADAM = 'adam'

    def __init__(self, state_class, state_width, state_height, device=None):
        if device is None:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.device = torch.device(self.device)

        self.num_filters = 16
        self.num_blocks = 4

        self.state_width = state_width
        self.state_height = state_height

        class StateClassSized(state_class):
            def __init__(self):
                super(StateClassSized, self).__init__(state_width, state_height)

        self.state_class = StateClassSized
        self.game = 0
        self.num_games = 50000
        self.num_games_one_epoch = 40
        self.num_sampled_actions = 10
        self.simulation_depth = 1
        self.C = 1
        self.n_vs_random = 100
        self.model_state_dict = None
        self.optimizer_state_dict = None
        self.lr = 3e-4
        self.weight_decay = 3e-5
        self.momentum = 0.8
        self.model_save_path = None
        self.model_save_interval = 10
        self.optimizer = Config.Optimizer.SGD
        self.episodes = []

    @classmethod
    def from_checkpoint(cls, state_class, checkpoint, device=None):
        result: Config = cls(
            state_class=state_class,
            state_width=checkpoint['state_width'],
            state_height=checkpoint['state_height'],
            device=device
        )

        for pg in checkpoint['optimizer_state_dict']['param_groups']:
            for key in ['lr', 'weight_decay', 'momentum']:
                if key in pg:
                    setattr(result, key, pg[key])

        result.model_state_dict = checkpoint['model_state_dict']
        result.optimizer_state_dict = checkpoint['optimizer_state_dict']
        result.game = checkpoint['game'] + 1

        if 'num_filters' in checkpoint:
            result.num_filters = checkpoint['num_filters']
        if 'num_blocks' in checkpoint:
            result.num_blocks = checkpoint['num_blocks']
        if 'optimizer' in checkpoint:
            result.optimizer = checkpoint['optimizer']
        if 'episodes' in checkpoint:
            result.episodes = checkpoint['episodes']

        return result

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        if key in ['lr', 'weight_decay', 'momentum']:
            if self.optimizer_state_dict is not None:
                for pg in self.optimizer_state_dict['param_groups']:
                    pg[key] = value
