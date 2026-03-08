"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        
        self.layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            self.layers.append(nn.ReLU())
            input_dim = hidden_dim
            
        self.layers.append(nn.Linear(input_dim, action_dim * chunk_size))
        self.network = nn.Sequential(*self.layers)

    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        
        # action_chunk is (B, chunk_size, action_dim)
        # state is (B, state_dim)
        
        sampled_actions = self.sample_actions(state)
        error = (action_chunk - sampled_actions) ** 2
        loss = error.sum(dim = (1, 2)).mean()
        return loss

    def sample_actions(self, state: torch.Tensor, *, num_steps: int = 10) -> torch.Tensor:
        
        output = self.network(state) # (B, action_dim * chunk_size)
        output = output.view(-1, self.chunk_size, self.action_dim) # (B, chunk_size, action_dim)
        return output


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        
        self.layers = []
        input_dim = state_dim + (chunk_size * action_dim) + 1
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            self.layers.append(nn.ReLU())
            input_dim = hidden_dim
            
        self.layers.append(nn.Linear(input_dim, action_dim * chunk_size))
        self.network = nn.Sequential(*self.layers)

    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        
        batch_size = state.shape[0]

        A_0 = torch.randn_like(action_chunk) # (B, chunk_size, action_dim)
        t = torch.rand(batch_size, 1, 1, device=action_chunk.device)  # (B, 1, 1)
        A_t = t * action_chunk + (1 - t) * A_0
        
        A_t_flat = A_t.view(batch_size, -1) # (B, chunk_size * action_dim)
        t = t.view(batch_size, 1) # (B, 1)
        
        combined_input = torch.cat([state, A_t_flat, t], dim = 1) # (B, state_dim + (chunk_size * action_dim) + 1)
        velocity = self.network(combined_input)
        velocity = velocity.view(batch_size, self.chunk_size, self.action_dim)
        
        denoised_action = action_chunk - A_0
        error = (velocity - denoised_action) ** 2
        loss = error.sum(dim = (1, 2)).mean()
        return loss
        
    def sample_actions(self, state: torch.Tensor, num_steps: int = 10) -> torch.Tensor:
        
        batch_size = state.shape[0]
        A_0 = torch.randn(batch_size, self.chunk_size, self.action_dim, device=state.device)
        flow_steps = torch.linspace(0, 1, num_steps)
        
        A_t = A_0
        for t in flow_steps:
            
            A_t_flat = A_t.view(batch_size, -1)
            t_tensor = torch.full((batch_size, 1), t, device=state.device)
            combined_input = torch.cat([state, A_t_flat, t_tensor], dim = 1)
            velocity = self.network(combined_input)
            velocity = velocity.view(batch_size, self.chunk_size, self.action_dim)
            A_t = A_t + (1 / num_steps) * velocity
            
        return A_t 


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
