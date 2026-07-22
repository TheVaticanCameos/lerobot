#!/usr/bin/env python

from __future__ import annotations

import math
import os
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import gymnasium as gym
import libero.libero.envs.problems.libero_tabletop_manipulation  # noqa: F401
import numpy as np
from gymnasium import spaces
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.envs.base_object import OBJECTS_DICT, register_object
from libero.libero.envs.bddl_base_domain import TASK_MAPPING, register_problem
from libero.libero.envs.regions import REGION_SAMPLERS
from robosuite.models.objects import MujocoXMLObject

from lerobot.types import RobotObservation

from .libero import ACTION_DIM, ACTION_HIGH, ACTION_LOW, get_libero_dummy_action
from .scene1_randomization import (
    Scene1RandomizationBaseline,
    Scene1RandomizationConfig,
    Scene1RandomizationSample,
    apply_scene1_randomization,
    capture_scene1_randomization_baseline,
    capture_scene1_realized_metadata,
    sample_scene1_randomization,
)
from .scene1_specs import (
    SCENE1_LAYOUTS,
    SCENE1_ORIGINAL_XY,
    SCENE1_VARIANTS,
    ResolvedScene1Spec,
    Scene1ObjectPose,
    read_scene1_bddl_inventory,
    resolve_scene1_specs,
    validate_scene1_bddl,
)
from .utils import parse_camera_names

DEFAULT_ASSETS_ROOT = Path(__file__).resolve().parents[3] / "data" / "mujoco_scene1_libero" / "assets"
DEFAULT_BDDL_ROOT = Path(__file__).resolve().parents[3] / "data" / "mujoco_scene1_libero" / "bddl"


def resolve_scene1_tasks(
    scene_task: str | Sequence[str], scene_variant: str | None = None
) -> list[ResolvedScene1Spec]:
    """Compatibility wrapper around the canonical Scene1 task/variant registry."""

    return resolve_scene1_specs(scene_task, scene_variant)


def _parse_bddl_obj_of_interest(bddl_file_name: str | Path) -> str | None:
    return read_scene1_bddl_inventory(bddl_file_name).target_object


def _assets_root() -> Path:
    return Path(os.environ.get("LEROBOT_SCENE1_LIBERO_ASSETS", DEFAULT_ASSETS_ROOT)).expanduser().resolve()


class Scene1Object(MujocoXMLObject):
    object_category: str = ""

    def __init__(self, name: str, joints: list[dict[str, str]] | None = None):
        if joints is None:
            joints = [{"type": "free", "damping": "0.0005"}]
        super().__init__(
            str(_assets_root() / "scene1_objects" / self.object_category / f"{self.object_category}.xml"),
            name=name,
            joints=joints,
            obj_type="all",
            duplicate_collision_geoms=False,
        )
        self.category_name = self.object_category
        self.rotation = (0, 0)
        self.rotation_axis = "z"
        self.object_properties = {"vis_site_names": {}}

    def _site_pos(self, site_name: str) -> np.ndarray:
        site = self.worldbody.find(f"./body/site[@name='{self.naming_prefix}{site_name}']")
        if site is None:
            site = self.worldbody.find(f".//site[@name='{self.naming_prefix}{site_name}']")
        if site is None:
            raise ValueError(f"Missing {site_name} in {self.name}")
        return np.fromstring(site.get("pos"), sep=" ")

    @property
    def bottom_offset(self):
        return self._site_pos("bottom_site")

    @property
    def top_offset(self):
        return self._site_pos("top_site")

    @property
    def horizontal_radius(self):
        return float(self._site_pos("horizontal_radius_site")[0])


@register_object
class Scene1BlueCar(Scene1Object):
    object_category = "scene1_blue_car"

    def __init__(self, name="scene1_blue_car", joints=None):
        super().__init__(name=name, joints=joints)


@register_object
class Scene1ToyCar8(Scene1Object):
    object_category = "scene1_toy_car_8"

    def __init__(self, name="scene1_toy_car_8", joints=None):
        super().__init__(name=name, joints=joints)


@register_object
class Scene1ToyCar2(Scene1Object):
    object_category = "scene1_toy_car_2"

    def __init__(self, name="scene1_toy_car_2", joints=None):
        super().__init__(name=name, joints=joints)


@register_object
class Scene1Toolbox(Scene1Object):
    object_category = "scene1_toolbox"

    def __init__(self, name="scene1_toolbox", joints=None):
        super().__init__(name=name, joints=joints)


@register_object
class Scene1FyingGlass(Scene1Object):
    object_category = "scene1_fying_glass"

    def __init__(self, name="scene1_fying_glass", joints=None):
        super().__init__(name=name, joints=joints)


@register_object
class Scene1Screwdriver(Scene1Object):
    object_category = "scene1_screwdriver"

    def __init__(self, name="scene1_screwdriver", joints=None):
        super().__init__(name=name, joints=joints)


@register_object
class Scene1Bottle4(Scene1Object):
    object_category = "scene1_bottle_4"

    def __init__(self, name="scene1_bottle_4", joints=None):
        super().__init__(name=name, joints=joints)


@register_object
class Scene1Bottle5(Scene1Object):
    object_category = "scene1_bottle_5"

    def __init__(self, name="scene1_bottle_5", joints=None):
        super().__init__(name=name, joints=joints)


@register_object
class Scene1Bottle6(Scene1Object):
    object_category = "scene1_bottle_6"

    def __init__(self, name="scene1_bottle_6", joints=None):
        super().__init__(name=name, joints=joints)


OBJECTS_DICT.update(
    {
        "scene1_blue_car": Scene1BlueCar,
        "scene1_toy_car_8": Scene1ToyCar8,
        "scene1_toy_car_2": Scene1ToyCar2,
        "scene1_toolbox": Scene1Toolbox,
        "scene1_fying_glass": Scene1FyingGlass,
        "scene1_screwdriver": Scene1Screwdriver,
        "scene1_bottle_4": Scene1Bottle4,
        "scene1_bottle_5": Scene1Bottle5,
        "scene1_bottle_6": Scene1Bottle6,
    }
)


class Scene1_Tabletop_Manipulation(TASK_MAPPING["libero_tabletop_manipulation"]):  # noqa: N801
    """LIBERO tabletop task with scene-1 movable objects.

    The BDDL goal is intentionally simple; for evaluation we use a physical
    lift condition on the magnifying glass so the task represents "pick up".
    """

    def __init__(self, bddl_file_name, *args, **kwargs):
        self._scene1_target_object = _parse_bddl_obj_of_interest(bddl_file_name)
        super().__init__(bddl_file_name, *args, **kwargs)

    def _check_success(self):
        height = self._object_height(self._scene1_target_object or "scene1_fying_glass_1")
        # return height is not None and height > 0.98
        return height is not None and height > 1.0

    def _object_height(self, name_fragment: str) -> float | None:
        try:
            body_names = getattr(self.sim.model, "body_names", [])
            heights = []
            for body_name in body_names:
                if name_fragment in body_name:
                    body_id = self.sim.model.body_name2id(body_name)
                    heights.append(float(self.sim.data.body_xpos[body_id][2]))
            if heights:
                return max(heights)
        except Exception:
            return None
        return None


register_problem(Scene1_Tabletop_Manipulation)
REGION_SAMPLERS["scene1_tabletop_manipulation"] = REGION_SAMPLERS["libero_tabletop_manipulation"]


class Scene1LiberoEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 80}

    def __init__(
        self,
        bddl_path: str | Path,
        assets_root: str | Path,
        prompt: str = "pick up the magnifying glass",
        episode_length: int = 280,
        camera_name: str | Sequence[str] = "agentview_image,robot0_eye_in_hand_image",
        obs_type: str = "pixels_agent_pos",
        render_mode: str = "rgb_array",
        observation_width: int = 360,
        observation_height: int = 360,
        camera_name_mapping: dict[str, str] | None = None,
        num_steps_wait: int = 10,
        control_mode: str = "relative",
        task_id: int = 0,
        scene_task: str = "pick_magnifying_glass",
        scene_variant: str = "full_scene",
        layout: str = "original",
        target_object: str = "scene1_fying_glass_1",
        variant_settle_steps: int = 10,
        unsupported_randomization_profiles: Sequence[str] = (),
        domain_randomization: Scene1RandomizationConfig | Mapping[str, Any] | None = None,
    ):
        super().__init__()
        os.environ["LEROBOT_SCENE1_LIBERO_ASSETS"] = str(Path(assets_root).expanduser().resolve())
        self.bddl_path = str(Path(bddl_path).expanduser().resolve())
        self.task = prompt
        self.task_description = prompt
        self.task_id = task_id
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.camera_name = parse_camera_names(camera_name)
        self.num_steps_wait = num_steps_wait
        self._max_episode_steps = episode_length
        self.control_mode = control_mode
        self.scene_task = scene_task
        self.scene_variant = scene_variant
        self.layout = layout
        self.target_object = target_object
        self.variant_settle_steps = variant_settle_steps
        if domain_randomization is None:
            domain_randomization = Scene1RandomizationConfig()
        elif isinstance(domain_randomization, Mapping):
            domain_randomization = Scene1RandomizationConfig(**domain_randomization)
        self.domain_randomization = domain_randomization
        if self.domain_randomization.profile in unsupported_randomization_profiles:
            raise ValueError(
                f"Randomization profile '{self.domain_randomization.profile}' is not compatible "
                f"with Scene1 variant '{scene_variant}'."
            )
        bddl_inventory = read_scene1_bddl_inventory(self.bddl_path)
        if bddl_inventory.target_object != self.target_object:
            raise ValueError(
                f"Scene1 BDDL target {bddl_inventory.target_object!r} does not match task target "
                f"{self.target_object!r}."
            )
        try:
            layout_spec = SCENE1_LAYOUTS[self.layout]
        except KeyError as error:
            available = ", ".join(sorted(SCENE1_LAYOUTS))
            raise ValueError(
                f"Unknown Scene1 layout '{self.layout}'. Available layouts: {available}"
            ) from error
        missing_layout_objects = layout_spec.required_objects - bddl_inventory.object_names
        if missing_layout_objects:
            missing = ", ".join(sorted(missing_layout_objects))
            raise ValueError(
                f"Scene1 layout '{self.layout}' requires objects absent from BDDL "
                f"'{self.bddl_path}': {missing}"
            )
        self._randomization_sample: Scene1RandomizationSample | None = None
        self._randomization_baseline: Scene1RandomizationBaseline | None = None
        self._episode_metadata: dict[str, Any] = self._make_episode_metadata(None)
        self._env: OffScreenRenderEnv | None = None
        self._last_raw_obs: RobotObservation | None = None

        if camera_name_mapping is None:
            camera_name_mapping = {
                "agentview_image": "image",
                "robot0_eye_in_hand_image": "image2",
            }
        self.camera_name_mapping = camera_name_mapping

        images = {}
        for cam in self.camera_name:
            images[self.camera_name_mapping[cam]] = spaces.Box(
                low=0,
                high=255,
                shape=(self.observation_height, self.observation_width, 3),
                dtype=np.uint8,
            )
        self.observation_space = spaces.Dict(
            {
                "pixels": spaces.Dict(images),
                "robot_state": spaces.Dict(
                    {
                        "eef": spaces.Dict(
                            {
                                "pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
                                "quat": spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64),
                                "mat": spaces.Box(low=-np.inf, high=np.inf, shape=(3, 3), dtype=np.float64),
                            }
                        ),
                        "gripper": spaces.Dict(
                            {
                                "qpos": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64),
                                "qvel": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64),
                            }
                        ),
                        "joints": spaces.Dict(
                            {
                                "pos": spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64),
                                "vel": spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64),
                            }
                        ),
                    }
                ),
            }
        )
        self.action_space = spaces.Box(
            low=ACTION_LOW, high=ACTION_HIGH, shape=(ACTION_DIM,), dtype=np.float32
        )

    def _ensure_env(self) -> None:
        if self._env is not None:
            return
        self._env = OffScreenRenderEnv(
            bddl_file_name=self.bddl_path,
            camera_heights=self.observation_height,
            camera_widths=self.observation_width,
        )
        self._env.reset()

    def _apply_original_scene_layout(self) -> None:
        assert self._env is not None
        sim = self._env.sim
        for object_name, (x, y) in SCENE1_ORIGINAL_XY.items():
            joint_name = f"{object_name}_joint0"
            try:
                joint_id = sim.model.joint_name2id(joint_name)
            except Exception:
                continue
            qpos_addr = sim.model.jnt_qposadr[joint_id]
            # Keep the settled tabletop height from robosuite, but restore the
            # original scene's XY layout. MuJoCo free-joint quaternions use
            # (w, x, y, z), so this is the identity orientation.
            sim.data.qpos[qpos_addr] = x
            sim.data.qpos[qpos_addr + 1] = y
            sim.data.qpos[qpos_addr + 3 : qpos_addr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
        sim.forward()

    @staticmethod
    def _pose_quaternion(pose: Scene1ObjectPose) -> np.ndarray:
        cr = math.cos(pose.roll / 2.0)
        sr = math.sin(pose.roll / 2.0)
        cp = math.cos(pose.pitch / 2.0)
        sp = math.sin(pose.pitch / 2.0)
        cy = math.cos(pose.yaw / 2.0)
        sy = math.sin(pose.yaw / 2.0)
        return np.array(
            [
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ]
        )

    def _apply_cluttered_scene_layout(self) -> None:
        assert self._env is not None
        self._apply_original_scene_layout()
        sim = self._env.sim
        try:
            layout = SCENE1_LAYOUTS[self.layout].object_poses
        except KeyError as exc:
            raise ValueError(f"Unknown Scene1 layout: {self.layout}") from exc

        for object_name, pose in layout.items():
            joint_name = f"{object_name}_joint0"
            joint_id = sim.model.joint_name2id(joint_name)
            qpos_addr = sim.model.jnt_qposadr[joint_id]
            base_z = float(sim.data.qpos[qpos_addr + 2])
            sim.data.qpos[qpos_addr : qpos_addr + 3] = np.array([pose.x, pose.y, base_z + pose.z_offset])
            sim.data.qpos[qpos_addr + 3 : qpos_addr + 7] = self._pose_quaternion(pose)
        sim.forward()

    def _apply_initial_scene_layout(self) -> None:
        if self.layout == "original":
            self._apply_original_scene_layout()
        else:
            self._apply_cluttered_scene_layout()

    def _format_raw_obs(self, raw_obs: RobotObservation) -> RobotObservation:
        assert self._env is not None
        images = {}
        for camera_name in self.camera_name:
            images[self.camera_name_mapping[camera_name]] = raw_obs[camera_name]

        eef_pos = raw_obs.get("robot0_eef_pos")
        eef_quat = raw_obs.get("robot0_eef_quat")
        gripper_qpos = raw_obs.get("robot0_gripper_qpos")
        if eef_pos is None or eef_quat is None or gripper_qpos is None:
            raise ValueError("Scene1 LIBERO observation is missing robot state fields.")

        return {
            "pixels": images,
            "robot_state": {
                "eef": {
                    "pos": eef_pos,
                    "quat": eef_quat,
                    "mat": self._env.robots[0].controller.ee_ori_mat,
                },
                "gripper": {
                    "qpos": gripper_qpos,
                    "qvel": raw_obs.get("robot0_gripper_qvel"),
                },
                "joints": {
                    "pos": raw_obs.get("robot0_joint_pos"),
                    "vel": raw_obs.get("robot0_joint_vel"),
                },
            },
        }

    def reset(self, seed=None, **kwargs):
        self._ensure_env()
        assert self._env is not None
        super().reset(seed=seed)
        self._env.seed(seed)
        raw_obs = self._env.reset()
        self._apply_initial_scene_layout()
        if self.domain_randomization.enabled:
            episode_seed = (
                int(seed)
                if seed is not None
                else int(self.np_random.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
            )
            joint_id = self._env.sim.model.joint_name2id(f"{self.target_object}_joint0")
            qpos_address = int(self._env.sim.model.jnt_qposadr[joint_id])
            target_xy = (
                float(self._env.sim.data.qpos[qpos_address]),
                float(self._env.sim.data.qpos[qpos_address + 1]),
            )
            object_xy = {}
            configured_xy = dict(SCENE1_ORIGINAL_XY)
            configured_xy.update(
                {name: (pose.x, pose.y) for name, pose in SCENE1_LAYOUTS[self.layout].object_poses.items()}
            )
            for name, xy in configured_xy.items():
                try:
                    self._env.sim.model.joint_name2id(f"{name}_joint0")
                except Exception:
                    continue
                object_xy[name] = xy
            other_object_xy = {name: xy for name, xy in object_xy.items() if name != self.target_object}
            self._randomization_sample = sample_scene1_randomization(
                self.domain_randomization.profile,
                episode_seed,
                target_xy=target_xy,
                other_object_xy=other_object_xy,
                table_margin=self.domain_randomization.table_margin,
                minimum_clearance=self.domain_randomization.minimum_object_clearance,
                max_pose_sampling_attempts=self.domain_randomization.max_pose_sampling_attempts,
            )
            self._randomization_baseline = capture_scene1_randomization_baseline(
                self._env.sim, self.target_object
            )
            apply_scene1_randomization(
                self._env.sim, self._randomization_baseline, self._randomization_sample
            )
        else:
            self._randomization_sample = None
            self._randomization_baseline = None
        settle_steps = max(
            self.num_steps_wait,
            getattr(self, "variant_settle_steps", self.num_steps_wait),
        )
        for _ in range(settle_steps):
            raw_obs, _, _, _ = self._env.step(get_libero_dummy_action())
        pose_randomized = (
            self._randomization_sample is not None and self._randomization_sample.object_pose is not None
        )
        if self.layout == "original" and not pose_randomized:
            self._apply_original_scene_layout()
        if self._randomization_sample is not None and self._randomization_baseline is not None:
            self._episode_metadata = self._make_episode_metadata(
                capture_scene1_realized_metadata(
                    self._env.sim,
                    self.target_object,
                    self._randomization_sample,
                    self._randomization_baseline,
                )
            )
        else:
            self._episode_metadata = self._make_episode_metadata(None)
        raw_obs = self._env.env._get_observations()
        self._last_raw_obs = raw_obs
        if self.control_mode == "absolute":
            for robot in self._env.robots:
                robot.controller.use_delta = False
        elif self.control_mode == "relative":
            for robot in self._env.robots:
                robot.controller.use_delta = True
        else:
            raise ValueError(f"Invalid control mode: {self.control_mode}")
        return self._format_raw_obs(raw_obs), {"is_success": False, **self.get_episode_metadata()}

    def get_episode_metadata(self) -> dict[str, Any]:
        return self._episode_metadata

    def _make_episode_metadata(self, randomization: dict[str, Any] | None) -> dict[str, Any]:
        return {
            "scene1": {
                "task": self.scene_task,
                "variant": self.scene_variant,
                "layout": self.layout,
                "bddl_file": Path(self.bddl_path).name,
                "bddl_path": self.bddl_path,
                "target_object": self.target_object,
                "settle_steps": self.variant_settle_steps,
                "effective_settle_steps": max(self.num_steps_wait, self.variant_settle_steps),
            },
            "domain_randomization": randomization,
        }

    def get_randomization_metadata(self) -> dict[str, Any] | None:
        return self._episode_metadata["domain_randomization"]

    def step(self, action: np.ndarray) -> tuple[RobotObservation, float, bool, bool, dict[str, Any]]:
        self._ensure_env()
        assert self._env is not None
        if action.ndim != 1:
            raise ValueError(f"Expected 1-D action, got {action.shape}")
        raw_obs, reward, done, info = self._env.step(action)
        self._last_raw_obs = raw_obs
        is_success = self._env.check_success()
        terminated = bool(done or is_success)
        info.update({"task": self.task, "task_id": self.task_id, "done": done, "is_success": is_success})
        observation = self._format_raw_obs(raw_obs)
        return observation, float(reward), terminated, False, info

    def render(self):
        self._ensure_env()
        assert self._env is not None
        raw_obs = self._last_raw_obs
        if raw_obs is None:
            raw_obs = self._env.env._get_observations()
        image = self._format_raw_obs(raw_obs)["pixels"]["image"]
        return image[::-1, ::-1]

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None
        self._last_raw_obs = None


def create_scene1_libero_envs(
    *,
    n_envs: int,
    env_cls: Callable[[Sequence[Callable[[], Any]]], Any],
    scene_task: str | Sequence[str],
    scene_variant: str | None = None,
    assets_root: str | Path,
    bddl_path: str | Path | None = None,
    prompt: str | None = None,
    gym_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, dict[int, Any]]:
    gym_kwargs = dict(gym_kwargs or {})
    resolved_fields = {
        "bddl_path",
        "layout",
        "scene_task",
        "scene_variant",
        "target_object",
        "task_id",
        "variant_settle_steps",
        "unsupported_randomization_profiles",
    }
    conflicting_fields = resolved_fields & gym_kwargs.keys()
    if conflicting_fields:
        fields = ", ".join(sorted(conflicting_fields))
        raise ValueError(f"Scene1 resolved spec fields cannot be overridden through gym_kwargs: {fields}")
    task_specs = resolve_scene1_tasks(scene_task, scene_variant)
    if len(task_specs) > 1 and (bddl_path is not None or prompt is not None):
        raise ValueError("Custom bddl_path/prompt overrides are only supported for a single scene1 task.")

    envs: dict[str, dict[int, Any]] = {"scene1_libero": {}}

    resolved_bddl_paths: dict[str, Path] = {}
    canonical_variant_requested = scene_variant in SCENE1_VARIANTS
    for spec in task_specs:
        selected_bddl_path = bddl_path if bddl_path is not None else DEFAULT_BDDL_ROOT / spec.bddl_file
        resolved_bddl_path = Path(selected_bddl_path).expanduser().resolve()
        validate_scene1_bddl(
            spec,
            resolved_bddl_path,
            require_canonical_inventory=bddl_path is None or canonical_variant_requested,
        )
        resolved_bddl_paths[spec.key] = resolved_bddl_path

    def _make_env(spec: ResolvedScene1Spec, task_id: int, **kwargs) -> Scene1LiberoEnv:
        return Scene1LiberoEnv(
            bddl_path=resolved_bddl_paths[spec.key],
            assets_root=assets_root,
            prompt=prompt or spec.prompt,
            task_id=task_id,
            scene_task=spec.task_key,
            scene_variant=spec.variant_key,
            layout=spec.layout.key,
            target_object=spec.target_object,
            variant_settle_steps=spec.settle_steps,
            unsupported_randomization_profiles=spec.unsupported_randomization_profiles,
            **kwargs,
        )

    for task_id, spec in enumerate(task_specs):
        fns = [partial(_make_env, spec, task_id, **gym_kwargs) for _ in range(n_envs)]
        vector_kwargs = {"context": "forkserver"} if env_cls is gym.vector.AsyncVectorEnv else {}
        try:
            from gymnasium.vector import AutoresetMode
        except ImportError:
            envs["scene1_libero"][task_id] = env_cls(fns, **vector_kwargs)
        else:
            try:
                envs["scene1_libero"][task_id] = env_cls(
                    fns, autoreset_mode=AutoresetMode.SAME_STEP, **vector_kwargs
                )
            except TypeError as exc:
                if "autoreset_mode" not in str(exc):
                    raise
                envs["scene1_libero"][task_id] = env_cls(fns, **vector_kwargs)
    return envs
