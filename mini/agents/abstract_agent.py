import torch.nn as nn

# Abstract class for agents
class AbstractAgent(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, invalid_action_mask):
        pass
