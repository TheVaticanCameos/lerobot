#!/usr/bin/env python

from __future__ import annotations

import math
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
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
from .utils import parse_camera_names

DEFAULT_ASSETS_ROOT = Path(__file__).resolve().parents[3] / "data" / "mujoco_scene1_libero" / "assets"
DEFAULT_BDDL_ROOT = Path(__file__).resolve().parents[3] / "data" / "mujoco_scene1_libero" / "bddl"


@dataclass(frozen=True)
class Scene1TaskSpec:
    key: str
    prompt: str
    bddl_file: str
    target_object: str
    layout: str = "original"


SCENE1_TASKS: dict[str, Scene1TaskSpec] = {
    "pick_red_car": Scene1TaskSpec(
        key="pick_red_car",
        prompt="pick up the red car",
        bddl_file="scene1_pick_red_car.bddl",
        target_object="scene1_toy_car_8_1",
    ),
    "pick_magnifying_glass": Scene1TaskSpec(
        key="pick_magnifying_glass",
        prompt="pick up the magnifying glass",
        bddl_file="scene1_pick_magnifying_glass.bddl",
        target_object="scene1_fying_glass_1",
    ),
    "pick_magnifying_glass_cluttered": Scene1TaskSpec(
        key="pick_magnifying_glass_cluttered",
        prompt="pick up the magnifying glass",
        bddl_file="scene1_pick_magnifying_glass_cluttered.bddl",
        target_object="scene1_fying_glass_1",
        layout="cluttered_red_cars_magnifying_glass",
    ),
    "pick_gray_box": Scene1TaskSpec(
        key="pick_gray_box",
        prompt="pick up the gray box",
        bddl_file="scene1_pick_gray_box.bddl",
        target_object="scene1_toolbox_1",
    ),
}

SCENE1_ORIGINAL_XY: dict[str, tuple[float, float]] = {
    "scene1_blue_car_1": (0.34996563199999997, -0.02677555945),
    "scene1_toy_car_8_1": (0.011990453899999996, 0.11388444525),
    "scene1_toy_car_2_1": (-0.13097472655, -0.14315710595),
    "scene1_toolbox_1": (-0.399399996, 0.09579999815),
    "scene1_fying_glass_1": (0.14760476515, -0.10315680495),
    "scene1_screwdriver_1": (-0.377967, -0.1730795205),
    "scene1_bottle_4_1": (0.38229599599999997, 0.18590948699999998),
    "scene1_bottle_5_1": (0.266199991, 0.164800003),
    "scene1_bottle_6_1": (0.29869999, 0.21510000499999998),
}


@dataclass(frozen=True)
class Scene1ObjectPose:
    x: float
    y: float
    z_offset: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0


SCENE1_CLUTTERED_RED_CARS_MAGNIFYING_GLASS: dict[str, Scene1ObjectPose] = {
    "scene1_fying_glass_1": Scene1ObjectPose(x=0.125, y=-0.085, yaw=math.radians(18.0)),
    "scene1_toy_car_8_1": Scene1ObjectPose(
        x=0.105,
        y=-0.085,
        z_offset=0.035,
        yaw=math.radians(28.0),
    ),
    "scene1_toy_car_2_1": Scene1ObjectPose(
        x=0.115,
        y=-0.085,
        z_offset=0.115,
        yaw=math.radians(-32.0),
    ),
}

SCENE1_LAYOUTS: dict[str, dict[str, Scene1ObjectPose]] = {
    "cluttered_red_cars_magnifying_glass": SCENE1_CLUTTERED_RED_CARS_MAGNIFYING_GLASS,
}


def resolve_scene1_tasks(scene_task: str | Sequence[str]) -> list[Scene1TaskSpec]:
    if isinstance(scene_task, str):
        task_keys = [key.strip() for key in scene_task.split(",") if key.strip()]
    else:
        task_keys = list(scene_task)
    if not task_keys:
        raise ValueError("At least one scene1 task must be specified.")

    specs = []
    for key in task_keys:
        try:
            specs.append(SCENE1_TASKS[key])
        except KeyError as exc:
            available = ", ".join(sorted(SCENE1_TASKS))
            raise ValueError(f"Unknown scene1 task '{key}'. Available tasks: {available}") from exc
    return specs


def _parse_bddl_obj_of_interest(bddl_file_name: str | Path) -> str | None:
    try:
        text = Path(bddl_file_name).expanduser().read_text()
    except OSError:
        return None
    match = re.search(r"\(:obj_of_interest\s+([^\s()]+)", text)
    return match.group(1) if match else None


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
        layout: str = "original",
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
        self.layout = layout
        self._env: OffScreenRenderEnv | None = None

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
            layout = SCENE1_LAYOUTS[self.layout]
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
        settle_steps = max(self.num_steps_wait, 40) if self.layout != "original" else self.num_steps_wait
        for _ in range(settle_steps):
            raw_obs, _, _, _ = self._env.step(get_libero_dummy_action())
        if self.layout == "original":
            self._apply_original_scene_layout()
        raw_obs = self._env.env._get_observations()
        if self.control_mode == "absolute":
            for robot in self._env.robots:
                robot.controller.use_delta = False
        elif self.control_mode == "relative":
            for robot in self._env.robots:
                robot.controller.use_delta = True
        else:
            raise ValueError(f"Invalid control mode: {self.control_mode}")
        return self._format_raw_obs(raw_obs), {"is_success": False}

    def step(self, action: np.ndarray) -> tuple[RobotObservation, float, bool, bool, dict[str, Any]]:
        self._ensure_env()
        assert self._env is not None
        if action.ndim != 1:
            raise ValueError(f"Expected 1-D action, got {action.shape}")
        raw_obs, reward, done, info = self._env.step(action)
        is_success = self._env.check_success()
        terminated = bool(done or is_success)
        info.update({"task": self.task, "task_id": self.task_id, "done": done, "is_success": is_success})
        observation = self._format_raw_obs(raw_obs)
        if terminated:
            self.reset()
        return observation, float(reward), terminated, False, info

    def render(self):
        self._ensure_env()
        assert self._env is not None
        raw_obs = self._env.env._get_observations()
        image = self._format_raw_obs(raw_obs)["pixels"]["image"]
        return image[::-1, ::-1]

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None


def create_scene1_libero_envs(
    *,
    n_envs: int,
    env_cls: Callable[[Sequence[Callable[[], Any]]], Any],
    scene_task: str | Sequence[str],
    assets_root: str | Path,
    bddl_path: str | Path | None = None,
    prompt: str | None = None,
    gym_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, dict[int, Any]]:
    gym_kwargs = dict(gym_kwargs or {})
    task_specs = resolve_scene1_tasks(scene_task)
    if len(task_specs) > 1 and (bddl_path is not None or prompt is not None):
        raise ValueError("Custom bddl_path/prompt overrides are only supported for a single scene1 task.")

    envs: dict[str, dict[int, Any]] = {"scene1_libero": {}}

    def _make_env(spec: Scene1TaskSpec, task_id: int, **kwargs) -> Scene1LiberoEnv:
        return Scene1LiberoEnv(
            bddl_path=bddl_path or DEFAULT_BDDL_ROOT / spec.bddl_file,
            assets_root=assets_root,
            prompt=prompt or spec.prompt,
            task_id=task_id,
            layout=spec.layout,
            **kwargs,
        )

    for task_id, spec in enumerate(task_specs):
        fns = [partial(_make_env, spec, task_id, **gym_kwargs) for _ in range(n_envs)]
        if env_cls is gym.vector.AsyncVectorEnv:
            envs["scene1_libero"][task_id] = env_cls(
                fns, context="forkserver"
            )
        else:
            envs["scene1_libero"][task_id] = env_cls(fns)
    return envs
