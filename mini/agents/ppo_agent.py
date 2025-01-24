from .abstract_agent import AbstractAgent
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np

class Transpose(nn.Module):
    def __init__(self, permutation):
        super().__init__()
        self.permutation = permutation

    def forward(self, x):
        return x.permute(self.permutation)

class CategoricalMasked(Categorical):
    def __init__(
        self, probs=None, logits=None, validate_args=None, masks=[], mask_value=None
    ):
        logits = torch.where(masks.bool(), logits, mask_value)
        super(CategoricalMasked, self).__init__(probs, logits, validate_args)


class PpoAgent(AbstractAgent):
    def __init__(self, envs):
        super().__init__()
        h, w, c = envs.observation_space.shape
        self.mapsize = h * w
        self.action_nvec = envs.action_plane_space.nvec
        action_count = sum(self.action_nvec)

        # Policy and value network
        self.encoder = nn.Sequential(
            Transpose((0, 3, 1, 2)),
            nn.Conv2d(in_channels=c, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
        )

        self.actor = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=action_count, kernel_size=1),
            Transpose((0, 2, 3, 1)),
        )

        self.critic = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * h * w, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.register_buffer("mask_value", torch.tensor(-1e8))

    def get_action_and_value(self, obs, envs=None, invalid_action_masks=None, device=None):
        x, y = self.forward(obs, invalid_action_masks)
        return x, y, None, None, None


    def forward(self, x, invalid_action_masks=None):
        encoded = self.encoder(x)
        logits = self.actor(encoded)
        values = self.critic(encoded)

        grid_logits = logits.reshape(-1, self.action_nvec.sum())
        split_logits = torch.split(
            grid_logits, self.action_nvec.tolist(), dim=1)

        invalid_action_masks = invalid_action_masks.reshape(
            -1, invalid_action_masks.shape[-1]
        )
        split_invalid_action_masks = torch.split(
            invalid_action_masks, self.action_nvec.tolist(), dim=1
        )
        multi_categoricals = [
            CategoricalMasked(logits=logits, masks=iam,
                              mask_value=self.mask_value)
            for (logits, iam) in zip(split_logits, split_invalid_action_masks)
        ]
        action = torch.stack(
            [categorical.sample() for categorical in multi_categoricals]
        )

        logprob = torch.stack(
            [
                categorical.log_prob(a)
                for a, categorical in zip(action, multi_categoricals)
            ]
        )

        entropy = torch.stack(
            [
                categorical.entropy()
                for categorical in multi_categoricals]
        )
        
        num_predicted_parameters = len(self.action_nvec)
        logprob = logprob.T.view(-1, self.mapsize, num_predicted_parameters)
        entropy = entropy.T.view(-1, self.mapsize, num_predicted_parameters)
        action = action.T.view(-1, self.mapsize, num_predicted_parameters)
        return action, logprob.sum(1).sum(1), entropy.sum(1).sum(1), values

    def get_values(self, obs):
        encoded = self.encoder(obs)
        values = self.critic(encoded)
        return values
    
    def evaluate_actions(self, obs, actions, invalid_action_masks):
        _, logproba, entropy, _ = self.forward(obs, invalid_action_masks)
        return logproba, entropy
