from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv
from train import train, train_ppo
from parameters import Parameters
from agents.premade_agent import PremadeAgent
from agents.ppo_agent import PpoAgent

import torch
import torch.optim as optim
import numpy as np
import sys

from gym_microrts import microrts_ai
import argparse


def train_from_parameters(params, reward_weight):
    env = MicroRTSGridModeVecEnv(
        num_selfplay_envs=params.num_selfplay_envs,
        num_bot_envs=params.num_bot_envs,
        partial_obs=params.partial_obs,
        max_steps=params.max_steps,
        render_theme=2,
        ai2s=params.ais,
        map_paths=params.maps,
        reward_weight=reward_weight,
        # cycle_maps=args.train_maps,
    )

    # agent = FirstAgent(env)
    if params.ppo:
        agent = PpoAgent(env).to(params.device)
    else:
        agent = PremadeAgent(env).to(params.device)
    
    if params.start_from_model is not None:
        agent.load_state_dict(torch.load(params.start_from_model))

    optimizer = optim.Adam(agent.parameters(), lr=params.lr, eps=params.eps)

    if params.ppo:
        train_ppo(env, params, agent, optimizer)
    else:
        train(env, params, agent, optimizer)

if __name__ == "__main__":

    def parse_args():
        parser = argparse.ArgumentParser(description="MicroRTS Training Script")
        parser.add_argument('--reward-weight', type=float, nargs='+', default=[10.0, 1.0, 1.0, 0.2, 1.0, 4.0],
                            help='Reward weights for the environment')
        return parser.parse_args()

    args = parse_args()
    reward_weight = np.array(args.reward_weight)

    params = Parameters()
    params.num_selfplay_envs = 4
    params.num_bot_envs = 8
    params.max_steps = 512
    params.ais = [microrts_ai.POLightRush, microrts_ai.coacAI, microrts_ai.naiveMCTSAI, microrts_ai.rojo,
                microrts_ai.izanagi, microrts_ai.tiamat, microrts_ai.mixedBot, microrts_ai.POWorkerRush]
    print("Training selfplay and bots")
    train_from_parameters(params, reward_weight)
