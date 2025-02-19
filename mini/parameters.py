import torch
from gym_microrts import microrts_ai


class Parameters:
    def __init__(self):
        self.epochs = 501
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.max_steps = 512
        self.gamma = 0.99
        self.partial_obs = False
        self.num_selfplay_envs = 12
        self.num_bot_envs = 0
        self.batch_size = self.num_selfplay_envs + self.num_bot_envs
        self.maps = [f"maps/8x8/basesWorkers8x8{variant}.xml" for variant in "ABCDGFGHIJKL"]
        self.lr = 0.01
        self.eps = 1e-4
        self.save_freq = 25
        self.start_from_model = None # "models/model_2024-11-29_12-45-28_1580.pt"
        self.start_epoch = None
        self.ais = [microrts_ai.coacAI, microrts_ai.lightRushAI]#, microrts_ai.naiveMCTSAI, microrts_ai.workerRushAI]

        # PPO
        self.ppo = False
        self.value_loss_coeff = 0.5
        self.ppo_epochs = 8
        self.lambda_gae = 0.95
        self.epsilon = 0.1
        self.entropy_coeff = 0.01
        self.max_grad_norm = 0.5


class EvalParameters:
    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.partial_obs = False
        self.num_selfplay_envs = 8
        self.maps = [f"maps/8x8/basesWorkers8x8{variant}.xml" for variant in "A"]#BCDEFGHIJKL"]

        self.ai = False
        # self.ai = 'coacAI'
        self.gym_id = "GYMID"
        self.exp_name = "test"
        self.seed = 0
        self.torch_deterministic = False
        self.num_selfplay_envs = 2 if not self.ai else 0
        self.num_envs = 2
        self.agent_model_path = "models/model_2024-12-19_21-32-24_100.pt"
        self.agent2_model_path = "models/model_2024-12-20_12-23-02_100.pt"
        self.num_updates = 3
        self.num_steps = 512
        self.max_steps = 512
        self.total_timesteps = 1024

class TournamentParameters:
    def __init__(self):
        self.partial_obs = False
        self.update_db = False
        self.cuda = True
        self.maps = [f"maps/8x8/basesWorkers8x8{variant}.xml" for variant in "A"]
        self.evals = [
            "randomBiasedAI",
            "workerRushAI",
            "lightRushAI",  
            "coacAI", 
            "models/model_2024-11-29_12-45-28_1580.pt",
            "models/model_2024-12-19_21-32-24_100.pt", # selfplay only
            "models/model_2024-12-19_22-32-54_100.pt", # bot only
            "models/model_2024-12-20_12-23-02_100.pt", # mixed
            ]
        self.num_matches = 1
        self.highest_sigma = 1.4
        self.output_path = 'tournament.temp.csv'
