import torch
import torch.nn as nn
import torch.nn.functional as F
from .abstract_agent import AbstractAgent
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv

class FirstAgent(AbstractAgent):
    def __init__(self, envs: MicroRTSGridModeVecEnv):
        super().__init__()
        height, width, feature_count = envs.observation_space.shape
        self.action_dims = envs.action_space_dims  # List of integers, e.g., [dim1, dim2, ...]

        # Define the neural network
        self.network = nn.Sequential(
            nn.Conv2d(feature_count, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(height * width * 64, 256),
            nn.ReLU()
        )

        # Separate output layers for each action dimension
        self.action_heads = nn.ModuleList([
            nn.Linear(256, dim) for dim in self.action_dims
        ])

    def forward(self, x, invalid_action_mask):
        """
        Args:
            x: torch.Tensor, batch of observations (batch_size, height, width, feature_count).
            invalid_action_mask: List[torch.Tensor], one mask per action dimension, 
                each of shape (batch_size, dim).
            envs: Environment object for action space dimensions.
        
        Returns:
            actions: List[torch.Tensor], sampled actions for each dimension (batch_size,).
            log_probs: torch.Tensor, summed log probabilities of sampled actions (batch_size,).
        """
        batch_size, height, width, feature_count = x.size()

        # Convert input to float and ensure it is on the same device as the model
        x = x.float()  # or x = x.to(torch.float32)
        x = x.to(next(self.network.parameters()).device)

        # Rearrange input for Conv2D: (batch_size, channels, height, width)
        x = x.permute(0, 3, 1, 2)

        # Shared network for feature extraction
        features = self.network(x)  # Shape: (batch_size, 256)

        actions = []
        log_probs = []

        # Split invalid_action_mask for each action head
        start_idx = 0
        for i, (head, action_dim) in enumerate(zip(self.action_heads, self.action_dims)):
            end_idx = start_idx + action_dim

            # Extract the mask for the current head
            head_mask = invalid_action_mask[:, :, start_idx:end_idx].float()  # Shape: (batch_size, height * width, action_dim)
            head_mask = head_mask.to(next(self.network.parameters()).device)
            start_idx = end_idx

            # Compute logits for this action head
            logits = head(features)  # Shape: (batch_size, action_dim)
            
            # Reduce logits across spatial dimensions (e.g., by summing or averaging if needed)
            logits = logits.view(batch_size, -1, action_dim).mean(dim=1)  # Reduce over spatial dimensions

            # Apply the invalid action mask
            logits = logits + (head_mask.mean(dim=1) * -1e9)

            # Compute probabilities and sample actions
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()  # Sample actions for this dimension
            log_prob = dist.log_prob(action)  # Log probabilities for sampled actions

            actions.append(action)  # Collect actions
            log_probs.append(log_prob)  # Collect log probabilities

        # Concatenate log probabilities for a single output
        total_log_probs = torch.stack(log_probs, dim=-1).sum(dim=-1)  # Sum log probs across dimensions

        return actions, total_log_probs
