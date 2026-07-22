#!/usr/bin/env python

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lerobot.scripts import lerobot_eval


def test_eval_config_applies_pretrained_revision(monkeypatch, tmp_path: Path) -> None:
    from types import SimpleNamespace

    from lerobot.configs import parser
    from lerobot.configs.eval import EvalPipelineConfig
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import Scene1LiberoEnv

    captured = {}

    def fake_from_pretrained(cls, policy_path, **kwargs):
        captured.update(policy_path=policy_path, **kwargs)
        return SimpleNamespace(type="pi05", pretrained_path=None)

    monkeypatch.setattr(parser, "get_path_arg", lambda field: "org/policy")
    monkeypatch.setattr(parser, "get_yaml_overrides", lambda field: [])
    monkeypatch.setattr(
        parser,
        "get_cli_overrides",
        lambda field: ["--pretrained_revision=commit-sha"],
    )
    monkeypatch.setattr(
        PreTrainedConfig,
        "from_pretrained",
        classmethod(fake_from_pretrained),
    )

    config = EvalPipelineConfig(env=Scene1LiberoEnv(), output_dir=tmp_path)

    assert captured["policy_path"] == "org/policy"
    assert captured["revision"] == "commit-sha"
    assert config.policy.pretrained_path == Path("org/policy")


class MetadataVectorEnv:
    num_envs = 2

    def call(self, name: str):
        if name != "get_episode_metadata":
            raise AttributeError(name)
        return (
            {"domain_randomization": {"seed": np.int64(4), "position": np.array([1.0, 2.0])}},
            {"domain_randomization": None},
        )


def test_get_episode_metadata_is_per_lane_and_json_serializable() -> None:
    metadata = lerobot_eval._get_episode_metadata(MetadataVectorEnv())
    assert metadata[0]["domain_randomization"]["seed"] == 4
    assert metadata[0]["domain_randomization"]["position"] == [1.0, 2.0]
    assert metadata[1] == {"domain_randomization": None}
    json.dumps(metadata)


def test_get_episode_metadata_tolerates_missing_hook() -> None:
    class NoMetadataVectorEnv:
        num_envs = 3

        def call(self, name: str):
            raise AttributeError(name)

    assert lerobot_eval._get_episode_metadata(NoMetadataVectorEnv()) == [None, None, None]


def test_eval_one_carries_metadata_into_task_metrics(monkeypatch) -> None:
    monkeypatch.setattr(
        lerobot_eval,
        "eval_policy",
        lambda **kwargs: {
            "per_episode": [
                {
                    "sum_reward": 1.0,
                    "max_reward": 1.0,
                    "success": True,
                    "episode_metadata": {"domain_randomization": {"profile": "pose_s1"}},
                }
            ]
        },
    )
    metrics = lerobot_eval.eval_one(
        object(),
        policy=object(),
        env_preprocessor=object(),
        env_postprocessor=object(),
        preprocessor=object(),
        postprocessor=object(),
        n_episodes=1,
        max_episodes_rendered=0,
        videos_dir=None,
        return_episode_data=False,
        start_seed=0,
    )
    assert metrics["episode_metadata"] == [{"domain_randomization": {"profile": "pose_s1"}}]
