# Source code for the DQN agent with experience replay and fixed Q-targets
# This code is ispired from the Udacity Deep Reinforcement Learning Nanodegree program codes.
# The code is provided for educational purposes and is not intended for production use.
# Source : Udacity Deep Reinforcement Learning Nanodegree - Course "Value-based method" Material - Exercice 2.7

import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetworkLinear(nn.Module):
    """Agent Neural Network Model."""
    def __init__(
            self,
            state_size: int,
            action_size: int,
            fc1_units: int = 64,
            fc2_units: int = 64
            ) -> None:
        """Initialize parameters and build model.
        
        Args:
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            fc1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super().__init__()

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(fc2_units, action_size)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Build a network that maps state -> action values.
        Args:
            state (torch.Tensor): State input to the network
        Returns:
            torch.Tensor: Action values for the given state
        """
        h = self.relu1(self.fc1(state))
        h = self.relu2(self.fc2(h))
        action = self.fc3(h)
        return action
        