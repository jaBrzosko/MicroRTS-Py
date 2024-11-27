import torch

class Parameters():
    def __init__(self):
        self.epochs = 100
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_steps = 256
        self.gamma = 0.99
        self.partial_obs = False
        self.num_selfplay_envs = 4
        self.num_bot_envs = 0
        self.batch_size = self.num_selfplay_envs + self.num_bot_envs
        self.maps = ["maps/8x8/basesWorkers8x8A.xml"]
        self.lr = 0.01
        self.eps = 1e-4
