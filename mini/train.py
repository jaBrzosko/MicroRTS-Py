import os
import numpy as np
from parameters import Parameters as Parameters
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv
from gym_microrts import microrts_ai

import torch
import torch.optim as optim
import torch.nn.functional as F

from agents.abstract_agent import AbstractAgent
from agents.first_agent import FirstAgent
from agents.premade_agent import PremadeAgent
from agents.ppo_agent import PpoAgent
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

    start = 0 if params.start_epoch is None else params.start_epoch

    for epoch in range(start, params.epochs):
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

def train_ppo(
    env: MicroRTSGridModeVecEnv,
    params: Parameters,
    agent: AbstractAgent,
    optimizer: optim.Optimizer,
):
    current_time = datetime.now()
    version_string = current_time.strftime("%Y-%m-%d_%H-%M-%S")

    file_name = f"statistics_{version_string}.txt"

    print(f"Training for {params.epochs} epochs")

    start = 0 if params.start_epoch is None else params.start_epoch

    for epoch in range(start, params.epochs):
        print(f"Epoch {epoch+1}/{params.epochs}")

        observations = env.reset()
        observations_tensor = torch.tensor(
            observations, device=params.device, dtype=torch.float32
        )

        trajectory = {
            "observations": [],
            "actions": [],
            "log_probs": [],
            "rewards": [],
            "dones": [],
            "values": [],
            "invalid_action_masks": [],
        }

        for step in range(params.max_steps):
            invalid_action_masks = env.get_action_mask()
            invalid_action_masks_tensor = torch.tensor(
                invalid_action_masks, device=params.device
            )
            trajectory["invalid_action_masks"].append(invalid_action_masks_tensor)

            with torch.no_grad():
                actions, log_probs, _, values = agent.forward(
                    observations_tensor, invalid_action_masks_tensor
                )

            next_observations, rewards, dones, _ = env.step(actions.cpu().numpy())
            next_observations_tensor = torch.tensor(
                next_observations, device=params.device, dtype=torch.float32
            )

            trajectory["observations"].append(observations_tensor)
            trajectory["actions"].append(actions)
            trajectory["log_probs"].append(log_probs)
            trajectory["rewards"].append(
                torch.tensor(rewards, device=params.device, dtype=torch.float32)
            )
            trajectory["dones"].append(
                torch.tensor(dones, device=params.device, dtype=torch.float32)
            )
            trajectory["values"].append(values)

            observations_tensor = next_observations_tensor

            if dones.all():
                break

        # Compute returns and advantages
        trajectory["values"].append(torch.zeros_like(trajectory["values"][-1]))  # Bootstrap value
        returns = []
        advantages = []
        R = torch.zeros(params.batch_size, device=params.device)
        A = torch.zeros(params.batch_size, device=params.device)

        for t in reversed(range(len(trajectory["rewards"]))):
            mask = 1.0 - trajectory["dones"][t]
            R = trajectory["rewards"][t] + params.gamma * R * mask
            td_error = (
                trajectory["rewards"][t]
                + params.gamma * trajectory["values"][t + 1] * mask
                - trajectory["values"][t]
            )
            A = td_error + params.gamma * params.lambda_gae * A * mask

            returns.insert(0, R)
            advantages.insert(0, A)

        returns = torch.stack(returns)
        advantages = torch.stack(advantages)

        # PPO update
        observations = torch.cat(trajectory["observations"])
        actions = torch.cat(trajectory["actions"])
        old_log_probs = torch.cat(trajectory["log_probs"]).detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8 if advantages.std() > 1e-8 else 1.0)
        invalid_action_masks = torch.cat(trajectory["invalid_action_masks"])

        advantages = advantages.unsqueeze(-1)
        returns = returns.flatten().reshape(-1, 1)

        for _ in range(params.ppo_epochs):
            # Compute new log_probs and entropy
            new_log_probs, entropy = agent.evaluate_actions(observations, actions, invalid_action_masks)

            # Compute ratio (new probability / old probability)
            ratios = torch.exp(new_log_probs - old_log_probs)

            # Compute surrogate losses
            surrogate1 = ratios * advantages
            surrogate2 = torch.clamp(ratios, 1.0 - params.epsilon, 1.0 + params.epsilon) * advantages

            # Compute PPO loss
            policy_loss = -torch.min(surrogate1, surrogate2).mean()
            vals = agent.get_values(observations)
            value_loss = F.mse_loss(returns, vals)

            loss = policy_loss + params.value_loss_coeff * value_loss - params.entropy_coeff * entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), params.max_grad_norm)
            optimizer.step()

        if epoch % params.save_freq == 0:
            print(f"Saving model at epoch {epoch}")
            os.makedirs("models", exist_ok=True)
            torch.save(agent.state_dict(),
                       f"models/model_{version_string}_{epoch}.pt")

        # Print some statistics
        print(f"Loss: {loss.item()}")
        print(f"Policy loss: {policy_loss.item()}")
        print(f"Value loss: {value_loss.item()}")
        print(f"Entropy: {entropy.mean().item()}")
        print(f"Rewards: {returns.mean().item()}")
        print(f"Log probs: {new_log_probs.mean().item()}")
        print(f"Advantages: {advantages.mean().item()}")
        print(f"Values: {vals.mean().item()}")

        # Dump data to file
        data = [loss.item(), policy_loss.item(),
                value_loss.item(), entropy.mean().item(),
                returns.mean().item(), new_log_probs.mean().item(),
                advantages.mean().item(), vals.mean().item()]

        with open(f"statistics/{file_name}", "a") as statistics_file:
            statistics_file.write(";".join(map(str, data)) + "\n")

if __name__ == "__main__":
    params = Parameters()


    env = MicroRTSGridModeVecEnv(
        num_selfplay_envs=params.num_selfplay_envs,
        num_bot_envs=params.num_bot_envs,
        partial_obs=params.partial_obs,
        max_steps=2000,
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
    