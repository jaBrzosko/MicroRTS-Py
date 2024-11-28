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


# ALGO LOGIC: initialize agent here:
class CategoricalMasked(Categorical):
    def __init__(
        self, probs=None, logits=None, validate_args=None, masks=[], mask_value=None
    ):
        logits = torch.where(masks.bool(), logits, mask_value)
        super(CategoricalMasked, self).__init__(probs, logits, validate_args)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class PremadeAgent(AbstractAgent):
    def __init__(self, envs):
        super().__init__()
        h, w, c = envs.observation_space.shape
        self.mapsize = h * w
        self.action_nvec = envs.action_plane_space.nvec
        action_count = sum(self.action_nvec)

        self.network = nn.Sequential(
            Transpose((0, 3, 1, 2)),
            nn.Conv2d(in_channels=c, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),

            # Second convolutional block
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=action_count, kernel_size=1),
            Transpose((0, 2, 3, 1)),
        )

        self.register_buffer("mask_value", torch.tensor(-1e8))

    def forward(self, x, invalid_action_masks=None):
        logits = self.network(x)
        grid_logits = logits.reshape(-1, self.action_nvec.sum())
        split_logits = torch.split(
            grid_logits, self.action_nvec.tolist(), dim=1)

        # invalid_action_masks = invalid_action_masks.view(-1, invalid_action_masks.shape[-1])
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
        
        num_predicted_parameters = len(self.action_nvec)
        logprob = logprob.T.view(-1, self.mapsize, num_predicted_parameters)
        action = action.T.view(-1, self.mapsize, num_predicted_parameters)
        return action, logprob.sum(1).sum(1)
