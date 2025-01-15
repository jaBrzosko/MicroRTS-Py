import torch
from gym_microrts import microrts_ai


class EvalParameters:
    def __init__(self):
        self.partial_obs = False
        self.update_db = False
        self.cuda = True
        self.maps = [f"maps/8x8/basesWorkers8x8{variant}.xml" for variant in "A"]
        self.evals = [
            "models/model_2024-12-19_21-32-24_100.pt", # selfplay only
            "models/model_2024-12-19_22-32-54_100.pt", # bot only
            "models/model_2024-12-20_12-23-02_100.pt", # mixed
            ]
        self.num_matches = 1
        self.highest_sigma = 1.4
        self.output_path = 'tournament.temp.csv'
