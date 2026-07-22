#!/usr/bin/env python

from __future__ import annotations

import json
import math
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from lerobot.envs.configs import Scene1LiberoEnv
from lerobot.envs.scene1_randomization import (
    SCENE1_RANDOMIZATION_PROFILES,
    Scene1RandomizationConfig,
    apply_scene1_randomization,
    capture_scene1_randomization_baseline,
    capture_scene1_realized_metadata,
    is_valid_object_candidate,
    sample_scene1_randomization,
)


class FakeModel:
    def __init__(self) -> None:
        self.jnt_qposadr = np.array([0])
        self.light_pos = np.array([[1.0, 2.0, 3.0], [-1.0, 1.0, 2.0]])
        self.light_diffuse = np.array([[0.8, 0.7, 0.6], [0.4, 0.5, 0.6]])
        self.light_specular = np.array([[0.3, 0.3, 0.3], [0.2, 0.2, 0.2]])
        self.cam_pos = np.array([[0.5, 0.0, 1.5]])
        self.cam_quat = np.array([[1.0, 0.0, 0.0, 0.0]])
        self.cam_fovy = np.array([45.0])

    def joint_name2id(self, name: str) -> int:
        assert name.endswith("_joint0")
        return 0

    def light_name2id(self, name: str) -> int:
        return {"light1": 0, "light2": 1}[name]

    def camera_name2id(self, name: str) -> int:
        assert name == "agentview"
        return 0


class FakeData:
    def __init__(self) -> None:
        self.qpos = np.array([0.1, -0.2, 0.9, 1.0, 0.0, 0.0, 0.0])


class FakeSim:
    def __init__(self) -> None:
        self.model = FakeModel()
        self.data = FakeData()
        self.forward_calls = 0

    def forward(self) -> None:
        self.forward_calls += 1


def test_profiles_have_exact_requested_amplitudes() -> None:
    assert SCENE1_RANDOMIZATION_PROFILES["nominal"].object_pose is None
    assert SCENE1_RANDOMIZATION_PROFILES["nominal"].lighting is None
    assert SCENE1_RANDOMIZATION_PROFILES["nominal"].camera is None

    for severity, (xy, yaw, ev, light_pos, camera_pos, camera_rotation, fov) in enumerate(
        zip(
            (0.02, 0.05, 0.08),
            (10.0, 25.0, 45.0),
            (0.5, 1.0, 1.5),
            (0.10, 0.25, 0.50),
            (0.02, 0.05, 0.10),
            (3.0, 7.0, 12.0),
            (3.0, 7.0, 12.0),
            strict=True,
        ),
        start=1,
    ):
        profile = SCENE1_RANDOMIZATION_PROFILES[f"combined_s{severity}"]
        assert profile.object_pose.xy == xy
        assert profile.object_pose.yaw == pytest.approx(math.radians(yaw))
        assert profile.lighting.exposure_ev == ev
        assert profile.lighting.position == light_pos
        assert profile.camera.position == camera_pos
        assert profile.camera.rotation == pytest.approx(math.radians(camera_rotation))
        assert profile.camera.fov == fov


def test_sample_ranges_ev_axis_angle_and_json_serialization() -> None:
    sample = sample_scene1_randomization("combined_s3", 123)
    assert abs(sample.object_pose.position_delta[0]) <= 0.08
    assert abs(sample.object_pose.position_delta[1]) <= 0.08
    assert sample.object_pose.position_delta[2] == 0.0
    assert abs(sample.object_pose.yaw_delta) <= math.radians(45.0)
    for light in sample.lights:
        assert abs(light.exposure_ev) <= 1.5
        assert light.exposure_scale == pytest.approx(2.0**light.exposure_ev)
        assert all(abs(value) <= 0.50 for value in light.position_delta)
    assert all(abs(value) <= 0.10 for value in sample.camera.position_delta)
    assert np.linalg.norm(sample.camera.axis_angle_delta) <= math.radians(12.0)
    assert abs(sample.camera.fov_delta) <= 12.0
    json.dumps(sample.to_dict())
    with pytest.raises(FrozenInstanceError):
        sample.profile = "nominal"  # type: ignore[misc]


def test_sampling_is_deterministic_and_factor_streams_are_independent() -> None:
    combined = sample_scene1_randomization("combined_s2", 987)
    assert combined == sample_scene1_randomization("combined_s2", 987)
    assert combined.object_pose == sample_scene1_randomization("pose_s2", 987).object_pose
    assert combined.lights == sample_scene1_randomization("lighting_s2", 987).lights
    assert combined.camera == sample_scene1_randomization("camera_s2", 987).camera


def test_object_resampling_does_not_advance_other_factor_streams() -> None:
    initial = sample_scene1_randomization("combined_s2", 91)
    first_candidate = initial.object_pose.position_delta[:2]
    blockers = {"first_candidate": first_candidate}
    combined = sample_scene1_randomization(
        "combined_s2", 91, other_object_xy=blockers, minimum_clearance=0.001
    )
    pose_only = sample_scene1_randomization("pose_s2", 91, other_object_xy=blockers, minimum_clearance=0.001)

    assert combined.object_pose.sampling_attempt == 2
    assert combined.object_pose == pose_only.object_pose
    assert combined.lights == initial.lights
    assert combined.camera == initial.camera


def test_candidate_validation_and_deterministic_rejection() -> None:
    objects = {"other": (0.1, 0.1)}
    assert is_valid_object_candidate((0.0, 0.0), objects, minimum_clearance=0.08)
    assert not is_valid_object_candidate((0.06, 0.06), objects, minimum_clearance=0.08)
    assert not is_valid_object_candidate((0.46, 0.0), {}, table_margin=0.05)

    with pytest.raises(RuntimeError, match="after 2 attempts"):
        sample_scene1_randomization("pose_s1", 4, table_margin=0.5, max_pose_sampling_attempts=2)


def test_nominal_has_no_factors_and_disabled_config_is_default() -> None:
    assert Scene1RandomizationConfig() == Scene1RandomizationConfig(enabled=False, profile="nominal")
    sample = sample_scene1_randomization("nominal", 1)
    assert sample.object_pose is None
    assert sample.lights == ()
    assert sample.camera is None

    sim = FakeSim()
    baseline = capture_scene1_randomization_baseline(sim, "target")
    metadata = capture_scene1_realized_metadata(sim, "target", sample, baseline)
    assert set(metadata["realized"]) == {"target_object", "lights", "camera"}


def test_scene1_config_forwards_nested_domain_randomization() -> None:
    domain_randomization = Scene1RandomizationConfig(enabled=True, profile="camera_s1")
    config = Scene1LiberoEnv(domain_randomization=domain_randomization)
    assert config.gym_kwargs["domain_randomization"] is domain_randomization
    assert "randomization" not in config.gym_kwargs


def test_apply_and_realized_metadata_use_actual_post_settle_values() -> None:
    sim = FakeSim()
    sample = sample_scene1_randomization("combined_s3", 42)
    baseline = capture_scene1_randomization_baseline(sim, "target")
    apply_scene1_randomization(sim, baseline, sample)

    requested_position = np.add(baseline.object_pose.position, sample.object_pose.position_delta)
    np.testing.assert_allclose(sim.data.qpos[:3], requested_position)
    assert np.linalg.norm(sim.data.qpos[3:7]) == pytest.approx(1.0)
    assert np.linalg.norm(sim.model.cam_quat[0]) == pytest.approx(1.0)
    np.testing.assert_allclose(
        sim.model.light_specular[1],
        np.multiply(baseline.lights[1].specular, sample.lights[1].exposure_scale),
    )

    sim.data.qpos[0] += 0.003  # Physics settling changed the realized target pose.
    metadata = capture_scene1_realized_metadata(sim, "target", sample, baseline)
    assert metadata["requested"] == sample.to_dict()
    assert metadata["realized"]["target_object"]["qpos"][0] == pytest.approx(sim.data.qpos[0])
    assert metadata["realized"]["camera"]["fov"] == pytest.approx(sim.model.cam_fovy[0])
    assert len(metadata["realized"]["lights"]) == 2
    assert metadata["realized"]["lights"][0]["specular"] == pytest.approx(sim.model.light_specular[0])
    json.dumps(metadata)


def test_scene1_reset_applies_pose_before_settling_and_reports_realized_metadata() -> None:
    scene1_libero = pytest.importorskip("lerobot.envs.scene1_libero")
    sim = FakeSim()

    class Controller:
        use_delta = False

    class Robot:
        controller = Controller()

    class InnerEnv:
        def _get_observations(self):
            return {}

    class FakeLiberoEnv:
        def __init__(self) -> None:
            self.sim = sim
            self.env = InnerEnv()
            self.robots = [Robot()]
            self.steps = 0

        def seed(self, seed) -> None:
            pass

        def reset(self):
            return {}

        def step(self, action):
            self.steps += 1
            assert sim.data.qpos[0] != 0.1
            sim.data.qpos[0] += 0.001
            return {}, 0.0, False, {}

    env = object.__new__(scene1_libero.Scene1LiberoEnv)
    env._env = FakeLiberoEnv()
    env.scene_task = "pick_magnifying_glass"
    env.scene_variant = "full_scene"
    env.bddl_path = "/tmp/scene1_pick_magnifying_glass.bddl"
    env.target_object = "scene1_fying_glass_1"
    env.domain_randomization = Scene1RandomizationConfig(enabled=True, profile="pose_s1")
    env.num_steps_wait = 2
    env.variant_settle_steps = 2
    env.layout = "original"
    env.control_mode = "relative"
    env._ensure_env = lambda: None
    env._apply_initial_scene_layout = lambda: None
    env._apply_original_scene_layout = lambda: pytest.fail("randomized pose must not be restored")
    env._format_raw_obs = lambda observation: observation

    _, info = scene1_libero.Scene1LiberoEnv.reset(env, seed=11)
    metadata = info["domain_randomization"]
    assert env._env.steps == 2
    assert metadata["requested"]["object_pose"] is not None
    assert metadata["realized"]["target_object"]["qpos"][0] == pytest.approx(sim.data.qpos[0])
    episode_metadata = env.get_episode_metadata()
    assert episode_metadata["domain_randomization"] == metadata
    assert episode_metadata["scene1"] == {
        "task": "pick_magnifying_glass",
        "variant": "full_scene",
        "layout": "original",
        "bddl_file": "scene1_pick_magnifying_glass.bddl",
        "bddl_path": "/tmp/scene1_pick_magnifying_glass.bddl",
        "target_object": "scene1_fying_glass_1",
        "settle_steps": 2,
        "effective_settle_steps": 2,
    }


def test_scene1_render_reuses_latest_observation() -> None:
    scene1_libero = pytest.importorskip("lerobot.envs.scene1_libero")
    image = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)

    class InnerEnv:
        def _get_observations(self):
            pytest.fail("render must not request a second simulator observation")

    class FakeLiberoEnv:
        env = InnerEnv()

    env = object.__new__(scene1_libero.Scene1LiberoEnv)
    env._env = FakeLiberoEnv()
    env._last_raw_obs = {"agentview_image": image}
    env._ensure_env = lambda: None
    env._format_raw_obs = lambda observation: {"pixels": {"image": observation["agentview_image"]}}

    np.testing.assert_array_equal(scene1_libero.Scene1LiberoEnv.render(env), image[::-1, ::-1])


def test_scene1_vector_env_owns_same_step_autoreset() -> None:
    scene1_libero = pytest.importorskip("lerobot.envs.scene1_libero")

    class RecordingVectorEnv:
        def __init__(self, env_fns, **kwargs) -> None:
            self.env_fns = env_fns
            self.kwargs = kwargs

    envs = scene1_libero.create_scene1_libero_envs(
        n_envs=2,
        env_cls=RecordingVectorEnv,
        scene_task="pick_magnifying_glass",
        assets_root=Path("unused"),
    )
    vector_env = envs["scene1_libero"][0]
    assert vector_env.kwargs["autoreset_mode"].name == "SAME_STEP"
    assert len(vector_env.env_fns) == 2


def test_scene1_factory_resolves_runtime_identity_without_starting_simulator() -> None:
    scene1_libero = pytest.importorskip("lerobot.envs.scene1_libero")

    class RecordingVectorEnv:
        def __init__(self, env_fns, **kwargs) -> None:
            self.env_fns = env_fns

    envs = scene1_libero.create_scene1_libero_envs(
        n_envs=1,
        env_cls=RecordingVectorEnv,
        scene_task="pick_magnifying_glass",
        scene_variant="magnifying_glass_near_clutter",
        assets_root=Path("data/mujoco_scene1_libero/assets"),
    )
    env = envs["scene1_libero"][0].env_fns[0]()
    metadata = env.get_episode_metadata()
    assert metadata["scene1"]["task"] == "pick_magnifying_glass"
    assert metadata["scene1"]["variant"] == "near_clutter"
    assert metadata["scene1"]["layout"] == "nearby_red_cars_magnifying_glass"
    assert metadata["scene1"]["bddl_file"] == "scene1_pick_magnifying_glass.bddl"
    assert metadata["scene1"]["target_object"] == "scene1_fying_glass_1"
    assert metadata["scene1"]["settle_steps"] == 40
    assert metadata["domain_randomization"] is None


def test_factory_rejects_conflicting_custom_bddl_for_explicit_canonical_variant() -> None:
    scene1_libero = pytest.importorskip("lerobot.envs.scene1_libero")

    class UnusedVectorEnv:
        def __init__(self, env_fns, **kwargs) -> None:
            pytest.fail("factory must validate before constructing the vector environment")

    with pytest.raises(ValueError, match="conflicts with canonical variant 'target_only'"):
        scene1_libero.create_scene1_libero_envs(
            n_envs=1,
            env_cls=UnusedVectorEnv,
            scene_task="pick_magnifying_glass",
            scene_variant="target_only",
            assets_root=Path("unused"),
            bddl_path=Path("data/mujoco_scene1_libero/bddl/scene1_pick_magnifying_glass.bddl"),
        )


def test_factory_rejects_gym_kwargs_that_override_resolved_spec() -> None:
    scene1_libero = pytest.importorskip("lerobot.envs.scene1_libero")

    with pytest.raises(ValueError, match="cannot be overridden.*layout"):
        scene1_libero.create_scene1_libero_envs(
            n_envs=1,
            env_cls=lambda env_fns: None,
            scene_task="pick_magnifying_glass",
            scene_variant="near_clutter",
            assets_root=Path("unused"),
            gym_kwargs={"layout": "original"},
        )


def test_scene1_step_does_not_reset_terminated_environment() -> None:
    scene1_libero = pytest.importorskip("lerobot.envs.scene1_libero")

    class FakeLiberoEnv:
        def step(self, action):
            return {}, 0.0, True, {}

        def check_success(self) -> bool:
            return False

    env = object.__new__(scene1_libero.Scene1LiberoEnv)
    env._env = FakeLiberoEnv()
    env.task = "task"
    env.task_id = 0
    env._ensure_env = lambda: None
    env._format_raw_obs = lambda observation: observation
    env.reset = lambda: pytest.fail("step() must leave autoreset to the vector environment")

    _, _, terminated, _, _ = scene1_libero.Scene1LiberoEnv.step(env, np.zeros(7))
    assert terminated


def test_scene1_vector_env_autoreset_compatibility_fallback() -> None:
    scene1_libero = pytest.importorskip("lerobot.envs.scene1_libero")

    class LegacyVectorEnv:
        def __init__(self, env_fns, **kwargs) -> None:
            if "autoreset_mode" in kwargs:
                raise TypeError("unexpected keyword argument 'autoreset_mode'")
            self.kwargs = kwargs

    envs = scene1_libero.create_scene1_libero_envs(
        n_envs=1,
        env_cls=LegacyVectorEnv,
        scene_task="pick_magnifying_glass",
        assets_root=Path("unused"),
    )
    assert envs["scene1_libero"][0].kwargs == {}
