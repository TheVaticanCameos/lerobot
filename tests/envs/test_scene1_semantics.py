from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from lerobot.envs.scene1_specs import Scene1SuccessSemantics


def _semantic_env():
    scene1_libero = pytest.importorskip("lerobot.envs.scene1_libero")
    env = object.__new__(scene1_libero.Scene1LiberoEnv)
    env.target_object = "scene1_fying_glass_1"
    env.receptacle_object = "scene1_toolbox_1"
    env.success_semantics = Scene1SuccessSemantics(
        kind="pick_and_place",
        receptacle_object="scene1_toolbox_1",
        relation="In",
        require_grasp=True,
        require_release=True,
        minimum_lift_height_delta=0.03,
        max_linear_speed=0.05,
        max_angular_speed=0.5,
        stable_steps=3,
    )
    env._stable_success_steps = 0
    env._ever_grasped = False
    env._ever_lifted = False
    env._grasp_start_height = None
    return scene1_libero, env


def test_containment_uses_toolbox_local_site_bounds() -> None:
    _, env = _semantic_env()
    angle = np.pi / 2
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    class Model:
        site_size = np.array([[0.055, 0.105, 0.04]])

        @staticmethod
        def site_name2id(name: str) -> int:
            assert name == "scene1_toolbox_1_containment_site"
            return 0

    data = SimpleNamespace(site_xpos=np.array([[0.4, -0.2, 0.95]]), site_xmat=rotation.reshape(1, 9))
    env._env = SimpleNamespace(sim=SimpleNamespace(model=Model(), data=data))

    inside_world = data.site_xpos[0] + rotation @ np.array([0.05, 0.10, 0.03])
    outside_world = data.site_xpos[0] + rotation @ np.array([0.06, 0.0, 0.0])
    assert env._target_inside_receptacle(inside_world)[0]
    assert not env._target_inside_receptacle(outside_world)[0]


def test_success_requires_inside_release_low_motion_for_three_steps() -> None:
    _, env = _semantic_env()
    state = {
        "position": np.array([0.0, 0.0, 1.2]),
        "linear": np.zeros(3),
        "angular": np.zeros(3),
        "inside": False,
        "grasped": False,
    }
    env._free_joint_state = lambda name: (state["position"], state["linear"], state["angular"])
    env._site_position = lambda name, site: state["position"]
    env._target_inside_receptacle = lambda position: (state["inside"], 0.01)
    env._is_target_grasped = lambda: state["grasped"]
    raw_obs = {"robot0_eef_pos": np.zeros(3)}

    # Lifting alone is not success.
    progress = env._task_progress(raw_obs, update_stability=True)
    assert progress["target_height"] == pytest.approx(1.2)
    assert not progress["success"]
    assert progress["stable_steps"] == 0

    state["inside"] = True
    # Pushing an ungrasped target into the box is not pick-and-place success.
    for _ in range(3):
        assert env._task_progress(raw_obs, update_stability=True)["stable_steps"] == 0

    state["position"] = np.array([0.0, 0.0, 0.9])
    state["grasped"] = True
    assert env._task_progress(raw_obs, update_stability=True)["stable_steps"] == 0
    assert not env._ever_lifted
    state["position"] = np.array([0.0, 0.0, 0.94])
    assert env._task_progress(raw_obs, update_stability=True)["stable_steps"] == 0
    assert env._ever_lifted

    state["grasped"] = False
    state["linear"] = np.array([0.051, 0.0, 0.0])
    assert env._task_progress(raw_obs, update_stability=True)["stable_steps"] == 0

    state["linear"] = np.zeros(3)
    for expected_steps in (1, 2):
        progress = env._task_progress(raw_obs, update_stability=True)
        assert progress["stable_steps"] == expected_steps
        assert not progress["success"]
    progress = env._task_progress(raw_obs, update_stability=True)
    assert progress["release"]
    assert progress["ever_grasped"]
    assert progress["ever_lifted"]
    assert progress["stable_steps"] == 3
    assert progress["success"]


def test_wrapper_owns_sparse_reward_backend_done_and_horizon() -> None:
    scene1_libero, env = _semantic_env()

    class Backend:
        def step(self, action):
            return {}, 123.0, True, {"backend": "kept"}

    env._env = Backend()
    env.task = "pick up the magnifying glass and place it into the box"
    env.task_id = 0
    env._step_count = 0
    env._max_episode_steps = 2
    env._ensure_env = lambda: None
    env._format_raw_obs = lambda observation: observation
    env._task_progress = lambda observation, update_stability: {"success": False}

    _, reward, terminated, truncated, info = scene1_libero.Scene1LiberoEnv.step(env, np.zeros(7))
    assert (reward, terminated, truncated) == (0.0, False, False)
    assert info["backend_done"]

    _, reward, terminated, truncated, _ = scene1_libero.Scene1LiberoEnv.step(env, np.zeros(7))
    assert (reward, terminated, truncated) == (0.0, False, True)

    env._step_count = 0
    env._task_progress = lambda observation, update_stability: {"success": True}
    _, reward, terminated, truncated, _ = scene1_libero.Scene1LiberoEnv.step(env, np.zeros(7))
    assert (reward, terminated, truncated) == (1.0, True, False)


def test_toolbox_bddl_in_predicate_has_a_safe_fallback() -> None:
    scene1_libero = pytest.importorskip("lerobot.envs.scene1_libero")
    toolbox = object.__new__(scene1_libero.Scene1Toolbox)

    assert toolbox.in_box(np.zeros(3), np.array([0.03, 0.0, 0.04]))
    assert not toolbox.in_box(np.zeros(3), np.array([0.2, 0.0, 0.04]))
