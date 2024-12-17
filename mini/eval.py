import os
import random
import time
from distutils.util import strtobool

import numpy as np
import torch
from gym.spaces import MultiDiscrete
from mini.parameters import EvalParameters
from stable_baselines3.common.vec_env import VecMonitor, VecVideoRecorder
from torch.utils.tensorboard import SummaryWriter

from gym_microrts import microrts_ai  # noqa
from agents.premade_agent import PremadeAgent
from experiments.ppo_gridnet_large import MicroRTSStatsRecorder

from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv


if __name__ == "__main__":
    args = EvalParameters()

    # TRY NOT TO MODIFY: setup the environment
    experiment_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{experiment_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    ais = []
    if args.ai:
        ais = [eval(f"microrts_ai.{args.ai}")]

    # TRY NOT TO MODIFY: seeding
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    envs = MicroRTSGridModeVecEnv(
        num_bot_envs=len(ais),
        num_selfplay_envs=args.num_selfplay_envs,
        partial_obs=args.partial_obs,
        max_steps=args.max_steps,
        render_theme=2,
        ai2s=ais,
        map_paths=args.maps,
        reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
    )
    envs = MicroRTSStatsRecorder(envs)
    print(envs.num_envs)
    envs = VecMonitor(envs)
    envs = VecVideoRecorder(
        envs,
        f"videos/{experiment_name}",
        record_video_trigger=lambda x: x % 100000 == 0,
        video_length=2000,
    )

    agent = PremadeAgent(envs).to(device)

    # ALGO Logic: Storage for epoch data
    mapsize = 8 * 8
    invalid_action_shape = (mapsize, envs.action_plane_space.nvec.sum())

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    # Note how `next_obs` and `next_done` are used; their usage is equivalent to
    # https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/84a7582477fb0d5c82ad6d850fe476829dddd2e1/a2c_ppo_acktr/storage.py#L60
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    # CRASH AND RESUME LOGIC:
    starting_update = 1
    agent.load_state_dict(torch.load(
        args.agent_model_path, map_location=device))
    agent.eval()
    if not args.ai:
        agent2 = PremadeAgent(envs).to(device)
        agent2.load_state_dict(torch.load(
            args.agent2_model_path, map_location=device))
        agent2.eval()

    print("Model's state_dict:")
    for param_tensor in agent.state_dict():
        print(param_tensor, "\t", agent.state_dict()[param_tensor].size())
    total_params = sum([param.nelement() for param in agent.parameters()])
    print("Model's total parameters:", total_params)

    for update in range(starting_update, args.num_updates + 1):
        # TRY NOT TO MODIFY: prepare the execution of the game.
        for step in range(0, args.num_steps):
            envs.render()
            global_step += 1 * args.num_envs
            # ALGO LOGIC: put action logic here
            with torch.no_grad():
                invalid_action_masks = torch.tensor(
                    np.array(envs.get_action_mask())
                ).to(device)

                if args.ai:
                    action, logproba, _, _, vs = agent.get_action_and_value(
                        next_obs,
                        envs=envs,
                        invalid_action_masks=invalid_action_masks,
                        device=device,
                    )
                else:
                    p1_obs = next_obs[::2]
                    p2_obs = next_obs[1::2]
                    p1_mask = invalid_action_masks[::2]
                    p2_mask = invalid_action_masks[1::2]

                    p1_action, _ = agent.forward(p1_obs, p1_mask)
                    p2_action, _ = agent2.forward(p2_obs, p2_mask)
                    action = torch.zeros(
                        (args.num_envs, p2_action.shape[1], p2_action.shape[2])
                    )
                    action[::2] = p1_action
                    action[1::2] = p2_action

            try:
                next_obs, rs, ds, infos = envs.step(
                    action.cpu().numpy().reshape(envs.num_envs, -1)
                )
                next_obs = torch.Tensor(next_obs).to(device)
            except Exception as e:
                e.printStackTrace()
                raise

            for idx, info in enumerate(infos):
                if "episode" in info.keys():
                    if args.ai:
                        print(
                            "against",
                            args.ai,
                            info["microrts_stats"]["WinLossRewardFunction"],
                        )
                    else:
                        if idx % 2 == 0:
                            print(
                                f"player{idx % 2}",
                                info["microrts_stats"]["WinLossRewardFunction"],
                            )

    envs.close()
    writer.close()
