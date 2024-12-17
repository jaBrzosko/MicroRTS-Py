from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv
from train import train, train_ppo
from parameters import Parameters
from agents.premade_agent import PremadeAgent
from agents.ppo_agent import PpoAgent

import torch
import torch.optim as optim
import numpy as np

from gym_microrts import microrts_ai


def train_from_parameters(params):
    env = MicroRTSGridModeVecEnv(
        num_selfplay_envs=params.num_selfplay_envs,
        num_bot_envs=params.num_bot_envs,
        partial_obs=params.partial_obs,
        max_steps=params.max_steps,
        render_theme=2,
        ai2s=params.ais,
        map_paths=params.maps,
        reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
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

    params = Parameters()
    params.num_selfplay_envs = 6
    params.num_bot_envs = 0
    params.ais = []
    print("Training only selfplay")
    train_from_parameters(params)

    params = Parameters()
    params.num_selfplay_envs = 0
    params.num_bot_envs = 6
    params.ais = [microrts_ai.POLightRush, microrts_ai.POWorkerRush,
                  microrts_ai.coacAI, microrts_ai.izanagi,
                  microrts_ai.guidedRojoA3N, microrts_ai.naiveMCTSAI]
    print("Training only bots")
    train_from_parameters(params)

    params = Parameters()
    params.num_selfplay_envs = 3
    params.num_bot_envs = 3
    params.ais = [microrts_ai.POLightRush, microrts_ai.coacAI, microrts_ai.naiveMCTSAI]
    print("Training selfplay and bots")
    train_from_parameters(params)
    