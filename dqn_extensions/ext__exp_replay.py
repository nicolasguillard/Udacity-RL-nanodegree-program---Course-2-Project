# Source code for the DQN agent with experience replay and fixed Q-targets
# This code is ispired from the Udacity Deep Reinforcement Learning Nanodegree program codes.
# The code is provided for educational purposes and is not intended for production use.
# Source : Udacity Deep Reinforcement Learning Nanodegree - Course "Value-based method" Material - Exercice 2.7

from collections import namedtuple, deque
import random
import numpy as np
import torch

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(
            self,
            action_size: int,
            buffer_size: int,
            batch_size: int,
            device: str|torch.device
            ) -> None:
        """Initialize a ReplayBuffer object.

        Args :
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.device = device
    
    def add(self,
            state: list[float],
            action: int,
            reward: int,
            next_state: list[float],
            done: bool
            ) -> None:
        """
        Add a new experience to memory.
        
        Args :
            state (list[float]): current state
            action (int): action taken
            reward (int): reward received
            next_state (list[float]): next state
            done (bool): whether the episode has ended  
        """
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        for e in experiences:
            if e is not None:
                states.append(e.state)
                actions.append(e.action)
                rewards.append(e.reward)
                next_states.append(e.next_state)
                dones.append(e.done)

        states = torch.from_numpy(np.vstack(states)).float().to(self.device)
        actions = torch.from_numpy(np.vstack(actions)).long().to(self.device)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(self.device)
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(self.device)
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(self.device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self) -> int:
        """Return the current size of internal memory."""
        return len(self.memory)