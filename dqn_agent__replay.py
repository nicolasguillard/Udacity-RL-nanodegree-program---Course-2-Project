import numpy as np
import random
from collections import namedtuple, deque

from models.linear_v1 import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

from dqn_extensions.ext_replay import ReplayBuffer

DEFAULT_BUFFER_SIZE = int(1e5)  # replay buffer size
DEFAULT_BATCH_SIZE = 64         # minibatch size
DEFAULT_GAMMA = 0.99            # discount factor
DEFAULT_TAU = 1e-3              # for soft update of target parameters
DEFAULT_LR = 5e-4               # learning rate 
DEFAULT_UPDATE_EVERY = 4        # how often to update the network

class DQNAgentReplay():
    """Interacts with and learns from the environment."""

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
            lr:float = DEFAULT_LR,
            update_every:int = DEFAULT_UPDATE_EVERY,
            seed: int = None
            ) -> None:
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.device = device

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.update_every = update_every

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed, **model_parameters).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed, **model_parameters).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=DEFAULT_LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, device, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, DEFAULT_GAMMA)


    def act(self, state, eps=0., train_mode=True):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        if train_mode: 
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


    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
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


    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
