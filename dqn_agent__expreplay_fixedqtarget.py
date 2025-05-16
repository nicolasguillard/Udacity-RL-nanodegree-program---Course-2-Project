# Source code for the DQN agent with experience replay and fixed Q-targets
# This code is ispired from the Udacity Deep Reinforcement Learning Nanodegree program codes.
# The code is provided for educational purposes and is not intended for production use.
# Source : Udacity Deep Reinforcement Learning Nanodegree - Course "Value-based method" Material - Exercice 2.7

import random
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from models.linear_v1 import QNetworkLinear as QNetwork
from dqn_extensions.ext__exp_replay import ReplayBuffer

DEFAULT_BUFFER_SIZE = int(1e5)      # replay buffer size
DEFAULT_BATCH_SIZE = 64             # minibatch size
DEFAULT_GAMMA = 0.99                # discount factor
DEFAULT_TAU = 1e-3                  # for soft update of target parameters
DEFAULT_LEARNING_RATE = 5e-4        # learning rate 
DEFAULT_UPDATE_EVERY_N_STEPS = 4    # how often to update the network

class DQNAgentExpReplayFixedQTarget():
    def __init__(
            self,
            state_size: int,
            action_size: int, 
            model_parameters: dict,
            device: str,
            buffer_size: int = DEFAULT_BUFFER_SIZE,
            batch_size: int = DEFAULT_BATCH_SIZE,
            gamma: float = DEFAULT_GAMMA,
            tau:float = DEFAULT_TAU,
            lr:float = DEFAULT_LEARNING_RATE,
            update_every_n_steps:int = DEFAULT_UPDATE_EVERY_N_STEPS
            ) -> None:
        """Initialize an Agent object.
        
        Args:
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            model_parameters (dict): parameters for the QNetwork model
            device (str): device to use for training ('cpu' or 'cuda')
            buffer_size (int): replay buffer size
            batch_size (int): minibatch size
            gamma (float): discount factor
            tau (float): for soft update of target parameters
            lr (float): learning rate 
            update_every_n_steps (int): how often to update the network
        """
        self.state_size = state_size
        self.action_size = action_size
        self.device = device

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.update_every_n_steps = update_every_n_steps

        # Q-Network
        self.qnetwork_local = QNetwork(
            state_size, action_size, **model_parameters
            ).to(device)
        self.qnetwork_target = QNetwork(
            state_size, action_size, **model_parameters
            ).to(device)
        self.optimizer = optim.Adam(
            self.qnetwork_local.parameters(), lr=DEFAULT_LEARNING_RATE
            )

        # Replay memory
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, device)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.train_mode = True
    
    def step(
            self,
            state: list[float],
            action: int,
            reward: int,
            next_state: list[float],
            done: bool
            ) :
        """Save experience in replay memory, and use random sample from buffer to learn.
        
        Args:
            state (list[float]): current state
            action (int): action taken
            reward (int): reward received
            next_state (list[float]): next state
            done (bool): whether the episode has ended
        """
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every_n_steps
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, DEFAULT_GAMMA)

    def act(
            self,
            state: list[float],
            eps: float = 0.,
            train_mode: bool = False
            ):
        """Returns actions for given state as per current policy.

        Args:
            state (list[float]): current state
            eps (float): epsilon, for epsilon-greedy action selection
            train_mode (bool): whether the agent is in training mode or not
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        if self.train_mode or train_mode: 
            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local(state)
            self.qnetwork_local.train()

            # Epsilon-greedy action selection
            if random.random() > eps:
                return np.argmax(action_values.cpu().data.numpy())
            else:
                return random.choice(np.arange(self.action_size))
        else:
            with torch.no_grad():
                action_values = self.qnetwork_local(state)
            return np.argmax(action_values.cpu().data.numpy())

    def learn(self, experiences: Tuple[torch.Tensor], gamma: float):
        """Update value parameters using given batch of experience tuples.

        Args:
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

         # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, DEFAULT_TAU)                     

    def soft_update(self, local_model: QNetwork, target_model: QNetwork, tau: float):
        """Soft update target model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Args
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save(self, filename: str):
        """Save the model parameters to file.
        
        Args:
            filename (str): name of the file to save the model parameters
        """
        torch.save(self.qnetwork_local.state_dict(), filename)
    
    def load(self, filename: str):
        """Load the model parameters from file.
        
        Args:
            filename (str): name of the file to load the model parameters from
        """
        self.qnetwork_local.load_state_dict(torch.load(filename, weights_only=True))
        self.qnetwork_target.load_state_dict(torch.load(filename, weights_only=True))
        self.qnetwork_local.to(self.device)
        self.qnetwork_target.to(self.device)

    def train(self):
        """Set the model to training mode."""
        self.qnetwork_local.train()
        self.qnetwork_target.train()    
        self.train_mode = True
    
    def eval(self):
        """Set the model to evaluation mode."""
        self.qnetwork_local.eval()
        self.qnetwork_target.eval()
        self.train_mode = False
