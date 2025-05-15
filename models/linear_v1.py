import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetworkLinear(nn.Module):
    """Actor (Policy) Model."""

    def __init__(
            self,
            state_size: int,
            action_size: int,
            fc1_units: int = 64,
            fc2_units: int = 64
            ) -> None:
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super().__init__()

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Build a network that maps state -> action values."""
        h = F.relu(self.fc1(state))
        h = F.relu(self.fc2(h))
        action = self.fc3(h)
        return action
        