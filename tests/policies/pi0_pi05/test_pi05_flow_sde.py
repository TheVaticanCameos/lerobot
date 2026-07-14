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

import math
import re
from types import MethodType, SimpleNamespace

import torch
from torch import nn

from lerobot.policies.pi05.flow_sde import (
    FlowSDETrace,
    flow_ode_transition_mean,
    flow_sde_transition_parameters,
    gaussian_transition_log_prob,
    make_flow_timesteps,
)
from lerobot.policies.pi05.modeling_pi05 import PI05Policy, PI05Pytorch
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS


def _make_tiny_model() -> PI05Pytorch:
    model = PI05Pytorch.__new__(PI05Pytorch)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        chunk_size=3,
        max_action_dim=4,
        num_inference_steps=4,
        flow_sde_noise_level=0.5,
        rtc_config=None,
    )
    model.velocity = nn.Parameter(torch.tensor(0.2))
    model.rtc_processor = None
    model.noise_calls = 0
    language_model = SimpleNamespace(config=SimpleNamespace())
    model.paligemma_with_expert = SimpleNamespace(
        paligemma=SimpleNamespace(model=SimpleNamespace(language_model=language_model)),
        forward=lambda **kwargs: ([None, None], None),
    )

    def embed_prefix(self, images, img_masks, tokens, masks):
        batch_size = tokens.shape[0]
        return (
            torch.zeros(batch_size, 1, 1),
            torch.ones(batch_size, 1, dtype=torch.bool),
            torch.zeros(batch_size, 1, dtype=torch.bool),
        )

    def prepare_attention_masks(self, masks):
        return masks[:, None]

    def build_prefix_cache(self, images, img_masks, tokens, masks):
        batch_size = tokens.shape[0]
        return (
            torch.ones(batch_size, 1, 2),
            torch.ones(batch_size, 1, dtype=torch.bool),
            None,
        )

    def denoise_step(self, prefix_pad_masks, past_key_values, x_t, timestep):
        return torch.ones_like(x_t) * self.velocity

    def sample_noise(self, shape, device):
        self.noise_calls += 1
        return torch.ones(shape, device=device)

    model._build_prefix_cache = MethodType(build_prefix_cache, model)
    model.embed_prefix = MethodType(embed_prefix, model)
    model._prepare_attention_masks_4d = MethodType(prepare_attention_masks, model)
    model.denoise_step = MethodType(denoise_step, model)
    model.sample_noise = MethodType(sample_noise, model)
    return model


def _dummy_inputs(batch_size: int = 2):
    return [], [], torch.zeros(batch_size, 1, dtype=torch.long), torch.ones(batch_size, 1, dtype=torch.bool)


def test_flow_sde_transition_math_matches_rlinf_and_handles_endpoints():
    timesteps = make_flow_timesteps(4, device="cpu", dtype=torch.float32)
    torch.testing.assert_close(timesteps, torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0]))

    x = torch.full((1, 1, 1), 2.0)
    velocity = torch.full_like(x, 0.4)
    mean, std = flow_sde_transition_parameters(x, velocity, timesteps[0:1], timesteps[1:2], noise_level=0.5)
    expected_mean = 0.25 * (x - velocity) + 0.625 * x
    torch.testing.assert_close(mean, expected_mean)
    torch.testing.assert_close(std, torch.full_like(std, 0.5))

    final_mean, final_std = flow_sde_transition_parameters(
        x, velocity, timesteps[3:4], timesteps[4:5], noise_level=0.5
    )
    sigma_squared = 0.25 / 3
    expected_final_mean = x - 0.25 * velocity - sigma_squared / 2 * (x + 0.75 * velocity)
    torch.testing.assert_close(final_mean, expected_final_mean)
    torch.testing.assert_close(final_std, torch.full_like(final_std, 0.5 * math.sqrt(sigma_squared)))


def test_flow_ode_uses_rlinf_time_direction():
    x = torch.tensor([[[1.0, 2.0]]])
    velocity = torch.tensor([[[0.4, -0.8]]])
    actual = flow_ode_transition_mean(x, velocity, torch.tensor([0.75]), torch.tensor([0.5]))
    torch.testing.assert_close(actual, x - 0.25 * velocity)


def test_gaussian_log_prob_sums_only_requested_prefix_and_action_dimensions():
    mean = torch.zeros(1, 3, 4)
    std = torch.full_like(mean, 0.5)
    value = torch.zeros_like(mean)
    value[:, 2:, :] = 100
    value[:, :, 2:] = 100

    actual = gaussian_transition_log_prob(value, mean, std, action_dim=2, prefix_horizon=2)
    expected = torch.distributions.Normal(mean, std).log_prob(value)[:, :2, :2].sum(dim=(1, 2))
    torch.testing.assert_close(actual, expected)


def test_legacy_deterministic_sampler_remains_euler_flow_ode():
    model = _make_tiny_model()
    inputs = _dummy_inputs()
    initial_noise = torch.zeros(2, 3, 4)

    actions = model.sample_actions(*inputs, noise=initial_noise, num_steps=4)

    torch.testing.assert_close(actions, torch.full_like(actions, -0.2))
    assert model.noise_calls == 0


def test_hybrid_sampler_has_exactly_one_stochastic_step_and_cpu_trace():
    model = _make_tiny_model()
    inputs = _dummy_inputs()
    initial_noise = torch.zeros(2, 3, 4)
    torch.manual_seed(7)

    actions, trace = model.sample_actions_with_flow_sde(
        *inputs,
        noise=initial_noise,
        num_steps=4,
        noise_level=0.5,
        action_dim=2,
        prefix_horizon=2,
    )

    assert actions.shape == initial_noise.shape
    assert model.noise_calls == 1
    assert trace.step_index.shape == (2,)
    assert torch.all((trace.step_index >= 0) & (trace.step_index < trace.num_steps))
    assert trace.num_steps == 4
    assert trace.action_dim == 2
    assert trace.prefix_horizon == 2
    assert trace.value_features.shape == (2, 2)
    for tensor in (
        trace.x_before,
        trace.x_after,
        trace.timestep,
        trace.next_timestep,
        trace.step_index,
        trace.old_log_prob,
    ):
        assert tensor.device.type == "cpu"
        assert not tensor.requires_grad


def test_frozen_log_prob_equality_and_current_policy_differentiability():
    model = _make_tiny_model()
    inputs = _dummy_inputs()
    _, trace = model.sample_actions_with_flow_sde(
        *inputs,
        noise=torch.zeros(2, 3, 4),
        num_steps=4,
        action_dim=2,
        prefix_horizon=2,
    )

    current_log_prob = model.compute_flow_sde_log_prob(*inputs, trace)
    torch.testing.assert_close(current_log_prob.detach().cpu(), trace.old_log_prob)
    assert current_log_prob.requires_grad

    current_log_prob.sum().backward()
    assert model.velocity.grad is not None
    assert model.velocity.grad.abs() > 0


def test_policy_flow_sde_apis_unpad_rollout_and_keep_evaluator_gradients():
    class TinyFlowModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))

        def sample_actions_with_flow_sde(self, *args, **kwargs):
            assert not torch.is_grad_enabled()
            self.rollout_kwargs = kwargs
            actions = torch.zeros(2, 3, 4)
            trace = FlowSDETrace(
                x_before=actions,
                x_after=actions,
                timestep=torch.ones(2),
                next_timestep=torch.full((2,), 0.75),
                step_index=torch.zeros(2, dtype=torch.long),
                num_steps=4,
                noise_level=0.5,
                action_dim=kwargs["action_dim"],
                prefix_horizon=kwargs["prefix_horizon"],
                old_log_prob=torch.zeros(2),
                value_features=torch.ones(2, 2),
            )
            return actions, trace

        def compute_flow_sde_log_prob(self, *args, **kwargs):
            assert torch.is_grad_enabled()
            return self.weight.expand(2)

    policy = PI05Policy.__new__(PI05Policy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        n_action_steps=2,
        output_features={ACTION: SimpleNamespace(shape=(2,))},
    )
    policy.model = TinyFlowModel()

    def preprocess_images(self, batch):
        return [], []

    policy._preprocess_images = MethodType(preprocess_images, policy)
    batch = {
        OBS_LANGUAGE_TOKENS: torch.zeros(2, 1, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(2, 1, dtype=torch.bool),
    }

    actions, trace = policy.predict_action_chunk_with_flow_sde(batch)
    assert actions.shape == (2, 3, 2)
    assert policy.model.rollout_kwargs["action_dim"] == 2
    assert policy.model.rollout_kwargs["prefix_horizon"] == 2

    current_log_prob = policy.compute_flow_sde_log_prob(batch, trace)
    current_log_prob.sum().backward()
    assert policy.model.weight.grad is not None


def test_flow_sde_trace_select_and_stack_round_trip():
    model = _make_tiny_model()
    _, trace = model.sample_actions_with_flow_sde(
        *_dummy_inputs(),
        noise=torch.zeros(2, 3, 4),
        num_steps=4,
        action_dim=2,
        prefix_horizon=2,
    )

    rebuilt = FlowSDETrace.stack([trace.select(0), trace.select(1)])

    torch.testing.assert_close(rebuilt.x_before, trace.x_before)
    torch.testing.assert_close(rebuilt.x_after, trace.x_after)
    torch.testing.assert_close(rebuilt.step_index, trace.step_index)
    torch.testing.assert_close(rebuilt.old_log_prob, trace.old_log_prob)
    torch.testing.assert_close(rebuilt.value_features, trace.value_features)


def test_pi05_lora_targets_match_real_projection_names_only():
    target_pattern = PI05Policy._get_default_peft_targets(None)["target_modules"]
    expected = [
        "model.paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj",
        "model.paligemma_with_expert.gemma_expert.model.layers.0.self_attn.v_proj",
        "model.action_in_proj",
        "model.action_out_proj",
        "model.time_mlp_in",
        "model.time_mlp_out",
    ]
    nonexistent = [
        "model.state_proj",
        "model.action_time_mlp_in",
        "model.action_time_mlp_out",
    ]

    assert all(re.fullmatch(target_pattern, name) for name in expected)
    assert not any(re.fullmatch(target_pattern, name) for name in nonexistent)
