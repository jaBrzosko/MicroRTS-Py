import os
import numpy as np
from parameters import Parameters as Parameters
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv
from gym_microrts import microrts_ai

import torch
import torch.optim as optim

from agents.abstract_agent import AbstractAgent
from agents.first_agent import FirstAgent
from agents.premade_agent import PremadeAgent
from datetime import datetime


def train(
    env: MicroRTSGridModeVecEnv,
    params: Parameters,
    agent: AbstractAgent,
    optimizer: optim.Optimizer,
):
    current_time = datetime.now()
    version_string = current_time.strftime("%Y-%m-%d_%H-%M-%S")

    print(f"Training for {params.epochs} epochs")

    for epoch in range(params.epochs):
        print(f"Epoch {epoch+1}/{params.epochs}")

        trajectory = {"log_probs": [], "rewards": [], "dones": []}
        observations = env.reset()
        observations_tensor = torch.tensor(
            observations, device=params.device, dtype=torch.float32
        )

        for step in range(params.max_steps):
            invalid_action_masks = env.get_action_mask()
            invalid_action_masks_tensor = torch.tensor(
                invalid_action_masks, device=params.device
            )

            actions, log_probs = agent.forward(
                observations_tensor, invalid_action_masks_tensor
            )

            observations, rewards, dones, _ = env.step(actions.cpu().numpy())
            observations_tensor = torch.tensor(
                observations, device=params.device, dtype=torch.float32
            )

            # rewards_tensor = torch.tensor(rewards, device=params.device)
            trajectory["log_probs"].append(log_probs)
            trajectory["rewards"].append(
                torch.tensor(rewards, device=params.device))
            trajectory["dones"].append(
                torch.tensor(dones, device=params.device))

            if dones.all():
                break

        # Compute returns
        discounted_rewards = torch.zeros(
            (params.batch_size, step + 1), device=params.device
        )
        R = torch.zeros(params.batch_size, device=params.device)
        for t in reversed(range(step + 1)):
            R = trajectory["rewards"][t] + params.gamma * R * (
                ~trajectory["dones"][t]
            )  # Apply gamma and mask done games
            discounted_rewards[:, t] = R

        # Normalize rewards for stability (per batch)
        # discounted_rewards = (discounted_rewards - discounted_rewards.mean(dim=1, keepdim=True)) / \
        #                     (discounted_rewards.std(dim=1, keepdim=True) + 1e-8)

        # Compute loss
        log_probs = torch.stack(
            trajectory["log_probs"], dim=1
        )  # Shape: [batch_size, step+1]
        loss = -(log_probs * discounted_rewards).sum()

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % params.save_freq == 0:
            print(f"Saving model at epoch {epoch}")
            os.makedirs("models", exist_ok=True)
            torch.save(agent.state_dict(),
                       f"models/model_{version_string}_{epoch}.pt")

        # Print some statistics
        print(f"Loss: {loss.item()}")
        print(f"Rewards: {discounted_rewards.mean().item()}")
        print(f"Log probs: {log_probs.mean().item()}")


if __name__ == "__main__":
    params = Parameters()

    env = MicroRTSGridModeVecEnv(
        num_selfplay_envs=params.num_selfplay_envs,
        num_bot_envs=params.num_bot_envs,
        partial_obs=params.partial_obs,
        max_steps=2000,
        render_theme=2,
        ai2s=[],
        map_paths=params.maps,
        reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
        # cycle_maps=args.train_maps,
    )

    # agent = FirstAgent(env)
    agent = PremadeAgent(env).to(params.device)

    optimizer = optim.Adam(agent.parameters(), lr=params.lr, eps=params.eps)

    train(env, params, agent, optimizer)
