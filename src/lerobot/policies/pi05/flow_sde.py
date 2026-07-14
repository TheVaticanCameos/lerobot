#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hybrid Flow-SDE transition math adapted from RLinf's OpenPI implementation."""

import math
from dataclasses import dataclass, replace

import torch
from torch import Tensor


@dataclass(frozen=True)
class FlowSDETrace:
    """The single stochastic transition retained from a hybrid Flow-SDE rollout."""

    x_before: Tensor
    x_after: Tensor
    timestep: Tensor
    next_timestep: Tensor
    step_index: Tensor
    num_steps: int
    noise_level: float
    action_dim: int
    prefix_horizon: int
    old_log_prob: Tensor
    value_features: Tensor | None = None

    def detach(self, *, cpu: bool = False) -> "FlowSDETrace":
        """Return a graph-free trace, optionally moved to CPU for replay storage."""
        tensors = {
            field: getattr(self, field).detach().cpu() if cpu else getattr(self, field).detach()
            for field in (
                "x_before",
                "x_after",
                "timestep",
                "next_timestep",
                "step_index",
                "old_log_prob",
            )
        }
        if self.value_features is not None:
            tensors["value_features"] = (
                self.value_features.detach().cpu() if cpu else self.value_features.detach()
            )
        return replace(self, **tensors)

    def to(self, device: torch.device | str) -> "FlowSDETrace":
        """Return a copy whose tensor fields are on ``device``."""
        return replace(
            self,
            x_before=self.x_before.to(device),
            x_after=self.x_after.to(device),
            timestep=self.timestep.to(device),
            next_timestep=self.next_timestep.to(device),
            step_index=self.step_index.to(device),
            old_log_prob=self.old_log_prob.to(device),
            value_features=None if self.value_features is None else self.value_features.to(device),
        )

    def select(self, index: int) -> "FlowSDETrace":
        """Select one batch item while retaining a leading batch dimension."""
        return replace(
            self,
            x_before=self.x_before[index : index + 1],
            x_after=self.x_after[index : index + 1],
            timestep=self.timestep[index : index + 1],
            next_timestep=self.next_timestep[index : index + 1],
            step_index=self.step_index[index : index + 1],
            old_log_prob=self.old_log_prob[index : index + 1],
            value_features=None if self.value_features is None else self.value_features[index : index + 1],
        )

    @classmethod
    def stack(cls, traces: list["FlowSDETrace"]) -> "FlowSDETrace":
        """Combine compatible single-item traces into one training batch."""
        if not traces:
            raise ValueError("Cannot stack an empty Flow-SDE trace list")
        first = traces[0]
        scalar_fields = ("num_steps", "noise_level", "action_dim", "prefix_horizon")
        for trace in traces[1:]:
            for field in scalar_fields:
                if getattr(trace, field) != getattr(first, field):
                    raise ValueError(f"Flow-SDE traces have incompatible {field}")
            if (trace.value_features is None) != (first.value_features is None):
                raise ValueError("Flow-SDE traces must consistently include value_features")
        return cls(
            x_before=torch.cat([trace.x_before for trace in traces]),
            x_after=torch.cat([trace.x_after for trace in traces]),
            timestep=torch.cat([trace.timestep for trace in traces]),
            next_timestep=torch.cat([trace.next_timestep for trace in traces]),
            step_index=torch.cat([trace.step_index for trace in traces]),
            num_steps=first.num_steps,
            noise_level=first.noise_level,
            action_dim=first.action_dim,
            prefix_horizon=first.prefix_horizon,
            old_log_prob=torch.cat([trace.old_log_prob for trace in traces]),
            value_features=None
            if first.value_features is None
            else torch.cat([trace.value_features for trace in traces]),
        )


def make_flow_timesteps(num_steps: int, *, device: torch.device | str, dtype: torch.dtype) -> Tensor:
    """Return RLinf's denoising grid: ``[1, ..., 1 / num_steps, 0]``."""
    if num_steps < 1:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    timesteps = torch.linspace(1, 1 / num_steps, num_steps, device=device, dtype=dtype)
    return torch.cat([timesteps, torch.zeros(1, device=device, dtype=dtype)])


def _expand_time(value: Tensor, reference: Tensor) -> Tensor:
    if value.ndim == 0:
        value = value.expand(reference.shape[0])
    if value.ndim != 1 or value.shape[0] != reference.shape[0]:
        raise ValueError(f"Expected one timestep per batch item, got {tuple(value.shape)}")
    return value[:, None, None].expand_as(reference)


def flow_ode_transition_mean(
    x_before: Tensor,
    velocity: Tensor,
    timestep: Tensor,
    next_timestep: Tensor,
) -> Tensor:
    """Compute one deterministic flow Euler transition using RLinf's time direction."""
    timestep = _expand_time(timestep, x_before)
    next_timestep = _expand_time(next_timestep, x_before)
    delta = timestep - next_timestep
    x0_pred = x_before - velocity * timestep
    x1_pred = x_before + velocity * (1 - timestep)
    post_step_timestep = timestep - delta
    return x0_pred * (1 - post_step_timestep) + x1_pred * post_step_timestep


def flow_sde_transition_parameters(
    x_before: Tensor,
    velocity: Tensor,
    timestep: Tensor,
    next_timestep: Tensor,
    noise_level: float | Tensor,
) -> tuple[Tensor, Tensor]:
    """Compute the exact Gaussian mean and std of RLinf's Flow-SDE transition."""
    if torch.is_tensor(noise_level):
        if torch.any(noise_level <= 0):
            raise ValueError("noise_level must be positive")
        noise_level_tensor = noise_level.to(device=x_before.device, dtype=x_before.dtype)
    else:
        if noise_level <= 0:
            raise ValueError(f"noise_level must be positive, got {noise_level}")
        noise_level_tensor = torch.as_tensor(noise_level, device=x_before.device, dtype=x_before.dtype)

    timestep = _expand_time(timestep, x_before)
    next_timestep = _expand_time(next_timestep, x_before)
    delta = timestep - next_timestep

    # RLinf replaces the singular t=1 denominator with the following timestep.
    denominator_timestep = torch.where(timestep == 1, next_timestep, timestep)
    sigma = noise_level_tensor * torch.sqrt(timestep / (1 - denominator_timestep))

    x0_pred = x_before - velocity * timestep
    x1_pred = x_before + velocity * (1 - timestep)
    post_step_timestep = timestep - delta
    x0_weight = torch.ones_like(timestep) - post_step_timestep
    x1_weight = post_step_timestep - sigma.square() * delta / (2 * timestep)
    mean = x0_pred * x0_weight + x1_pred * x1_weight
    std = torch.sqrt(delta) * sigma
    return mean, std


def sample_flow_sde_transition(
    x_before: Tensor,
    velocity: Tensor,
    timestep: Tensor,
    next_timestep: Tensor,
    noise_level: float | Tensor,
    noise: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Sample one Flow-SDE transition and return ``(sample, mean, std)``."""
    mean, std = flow_sde_transition_parameters(x_before, velocity, timestep, next_timestep, noise_level)
    return mean + noise * std, mean, std


def gaussian_transition_log_prob(
    value: Tensor,
    mean: Tensor,
    std: Tensor,
    *,
    action_dim: int,
    prefix_horizon: int,
) -> Tensor:
    """Sum exact Gaussian log-probability over an action prefix and real action dimensions."""
    if not 0 < action_dim <= value.shape[-1]:
        raise ValueError(f"action_dim must be in [1, {value.shape[-1]}], got {action_dim}")
    if not 0 < prefix_horizon <= value.shape[-2]:
        raise ValueError(f"prefix_horizon must be in [1, {value.shape[-2]}], got {prefix_horizon}")
    if torch.any(std <= 0):
        raise ValueError("Gaussian transition std must be positive")

    log_prob = -torch.log(std) - 0.5 * math.log(2 * math.pi) - 0.5 * ((value - mean) / std).square()
    return log_prob[:, :prefix_horizon, :action_dim].sum(dim=(1, 2))
