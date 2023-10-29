class Config:
    def __init__(self, state_class, state_width, state_height):
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

    @classmethod
    def from_checkpoint(cls, state_class, checkpoint):
        result: Config = cls(
            state_class=state_class,
            state_width=checkpoint['state_width'],
            state_height=checkpoint['state_height']
        )

        for key in ['lr', 'weight_decay', 'momentum']:
            for pg in checkpoint['optimizer_state_dict']['param_groups']:
                setattr(result, key, pg[key])

        result.model_state_dict = checkpoint['model_state_dict']
        result.optimizer_state_dict = checkpoint['optimizer_state_dict']
        result.game = checkpoint['game'] + 1

        if 'num_filters' in checkpoint:
            result.num_filters = checkpoint['num_filters']
        if 'num_blocks' in checkpoint:
            result.num_blocks = checkpoint['num_blocks']

        return result

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        if key in ['lr', 'weight_decay', 'momentum']:
            if self.optimizer_state_dict is not None:
                for pg in self.optimizer_state_dict['param_groups']:
                    pg[key] = value
