import torch


class Parameters:
    def __init__(self):
        self.epochs = 100
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.max_steps = 256
        self.gamma = 0.99
        self.partial_obs = False
        self.num_selfplay_envs = 4
        self.num_bot_envs = 0
        self.batch_size = self.num_selfplay_envs + self.num_bot_envs
        self.maps = ["maps/8x8/basesWorkers8x8A.xml"]
        self.lr = 0.01
        self.eps = 1e-4
        self.save_freq = 10

        self.ai = False
        self.gym_id = "GYMID"
        self.exp_name = "test"
        self.seed = 0
        self.torch_deterministic = False
        self.num_selfplay_envs = 2
        self.num_envs = self.num_selfplay_envs
        self.agent_model_path = "models/model_90.pt"
        self.agent2_model_path = "models/model_90.pt"
        self.num_updates = 100
        self.num_steps = 256
        self.total_timesteps = 1000000
        self.batch_size = self.num_envs * self.num_steps
        self.num_updates = self.total_timesteps // self.batch_size
