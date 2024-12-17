import torch
from gym_microrts import microrts_ai


class EvalParameters:
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
            "models/model_2024-11-29_12-45-28_1580.pt"]
        self.num_matches = 1
        self.highest_sigma = 1.4
        self.output_path = 'tournament.temp.csv'
