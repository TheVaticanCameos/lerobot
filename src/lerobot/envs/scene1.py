#!/usr/bin/env python

"""Scene1 task metadata and environment helpers.

This module intentionally mirrors the module-level organisation used by
``lerobot.envs.libero``.  The Gymnasium wrapper and vector-environment
factories are added separately; the code here contains only the task/suite
adapter and the small, side-effect-free helpers they will depend on.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from libero.libero.envs import OffScreenRenderEnv

from lerobot.lerobot_types import RobotObservation

from .utils import _LazyAsyncVectorEnv, parse_camera_names


# Keep the same action convention as the LIBERO adapter.  The policy-facing
# action is a 6D end-effector command plus one gripper command.
ACTION_DIM = 7
ACTION_LOW = -1.0
ACTION_HIGH = 1.0

DEFAULT_CONTROL_FREQ = 20
DEFAULT_EPISODE_LENGTH = 300
DEFAULT_NUM_STEPS_WAIT = 10
DEFAULT_SUCCESS_HOLD_STEPS = 3

DEFAULT_CAMERA_NAME = "agentview_image,robot0_eye_in_hand_image"
DEFAULT_CAMERA_NAME_MAPPING = {
    "agentview_image": "image",
    "robot0_eye_in_hand_image": "image2",
}

VALID_CONTROL_MODES = frozenset({"relative", "absolute"})

@dataclass(frozen=True)
class Scene1Task:
    """Task metadata exposed through the LIBERO-like suite interface."""

    name: str
    language: str
    bddl_file: str
    target_object: str
    success_type: str
    receptacle: str | None = None
    max_episode_steps: int | None = None


# Keep this order stable: it is the public mapping from task_id to task name.
# Scene1 currently exposes one task/one BDDL only.
SCENE1_TASKS: tuple[Scene1Task, ...] = (
    Scene1Task(
        name="pick_and_place_magnifying_glass",
        language="pick up the magnifying glass and place it into the box",
        bddl_file="scene1_pick_and_place_magnifying_glass.bddl",
        target_object="scene1_fying_glass_1",
        receptacle="scene1_toolbox_1",
        success_type="place",
        max_episode_steps=400,
    ),
)

SCENE1_TASK_ORDER = tuple(task.name for task in SCENE1_TASKS)
SCENE1_TASKS_BY_NAME = {task.name: task for task in SCENE1_TASKS}

_SCENE1_REGISTERED_ROOT: Path | None = None


class Scene1TaskSuite:
    """Small suite adapter with the methods used by ``libero.py``.

    It deliberately does not register anything in LIBERO's global benchmark
    registry.  ``Scene1Env`` can nevertheless use the same ``tasks`` /
    ``get_task(task_id)`` protocol as the existing LIBERO wrapper.
    """

    def __init__(self, scene_root: str | Path):
        self.scene_root = resolve_scene1_root(scene_root)
        self.tasks = SCENE1_TASKS

    def get_task(self, task_id: int) -> Scene1Task:
        if not isinstance(task_id, (int, np.integer)):
            raise TypeError(f"task_id must be an integer, got {type(task_id).__name__}")
        task_id = int(task_id)
        if task_id < 0 or task_id >= len(self.tasks):
            raise ValueError(f"task_id {task_id} out of range [0, {len(self.tasks) - 1}]")
        return self.tasks[task_id]


def get_scene1_task(task: str | int) -> Scene1Task:
    """Resolve either a stable task name or an integer task id."""
    if isinstance(task, (int, np.integer)):
        task_id = int(task)
        if task_id < 0 or task_id >= len(SCENE1_TASKS):
            raise ValueError(
                f"task_id {task_id} out of range [0, {len(SCENE1_TASKS) - 1}]"
            )
        return SCENE1_TASKS[task_id]
    try:
        return SCENE1_TASKS_BY_NAME[task]
    except KeyError as err:
        available = ", ".join(SCENE1_TASK_ORDER)
        raise ValueError(f"Unknown Scene1 task '{task}'. Available: {available}") from err


def get_scene1_suite(scene_root: str | Path) -> Scene1TaskSuite:
    """Create the local suite adapter used by the Scene1 vector factory."""
    return Scene1TaskSuite(scene_root)


def parse_scene1_task_names(task: str | Sequence[str]) -> list[str]:
    """Parse a comma-separated task argument and validate every name."""
    if isinstance(task, str):
        names = [item.strip() for item in task.split(",") if item.strip()]
    else:
        names = [str(item).strip() for item in task if str(item).strip()]

    if not names:
        raise ValueError("At least one Scene1 task is required")

    for name in names:
        get_scene1_task(name)
    return names


def select_scene1_task_ids(
    total_tasks: int,
    task_ids: Iterable[int] | None,
) -> list[int]:
    """Validate and normalize task ids, matching LIBERO semantics."""
    if total_tasks <= 0:
        raise ValueError("Scene1 suite has no tasks")
    if task_ids is None:
        return list(range(total_tasks))

    ids = sorted({int(task_id) for task_id in task_ids})
    for task_id in ids:
        if task_id < 0 or task_id >= total_tasks:
            raise ValueError(
                f"task_id {task_id} out of range [0, {total_tasks - 1}]"
            )
    return ids


def scene1_task_name_to_id(task_name: str) -> int:
    try:
        return SCENE1_TASK_ORDER.index(task_name)
    except ValueError as err:
        available = ", ".join(SCENE1_TASK_ORDER)
        raise ValueError(
            f"Unknown Scene1 task '{task_name}'. Available: {available}"
        ) from err


def scene1_task_id_to_name(task_id: int) -> str:
    task_id = int(task_id)
    if task_id < 0 or task_id >= len(SCENE1_TASK_ORDER):
        raise ValueError(f"task_id {task_id} out of range [0, {len(SCENE1_TASK_ORDER) - 1}]")
    return SCENE1_TASK_ORDER[task_id]


def resolve_scene1_root(scene_root: str | Path) -> Path:
    """Resolve and validate the repository-local Scene1 asset root."""
    root = Path(scene_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Scene1 root does not exist: {root}")

    for required in (root / "assets", root / "bddl"):
        if not required.is_dir():
            raise FileNotFoundError(
                f"Required Scene1 directory does not exist: {required}"
            )
    return root


def resolve_scene1_bddl(
    scene_root: str | Path,
    task: Scene1Task | str,
) -> Path:
    """Resolve the single supported Scene1 task's BDDL path."""
    root = resolve_scene1_root(scene_root)
    task_spec = get_scene1_task(task) if isinstance(task, str) else task

    bddl_path = root / "bddl" / task_spec.bddl_file
    if not bddl_path.is_file():
        raise FileNotFoundError(
            f"BDDL file for task '{task_spec.name}' does not exist: "
            f"{bddl_path}"
        )
    return bddl_path


def get_scene1_dummy_action() -> np.ndarray:
    """Return a no-op action used during post-reset settling."""
    return np.asarray(
        [0, 0, 0, 0, 0, 0, -1],
        dtype=np.float32,
    )


def validate_scene1_control_mode(control_mode: str) -> None:
    if control_mode not in VALID_CONTROL_MODES:
        raise ValueError(
            f"Invalid Scene1 control mode '{control_mode}'. "
            f"Expected one of {sorted(VALID_CONTROL_MODES)}"
        )


def validate_scene1_camera_names(camera_names: Sequence[str]) -> None:
    required = set(DEFAULT_CAMERA_NAME_MAPPING)
    missing = required.difference(camera_names)
    if missing:
        raise ValueError(
            "Scene1's LIBERO-compatible observation requires cameras "
            f"{sorted(required)}; missing {sorted(missing)}"
        )


def load_scene1_init_states(
    scene_root: str | Path,
    task: Scene1Task | str,
) -> Any:
    """Load optional Scene1 initial states from ``init_states/``.

    The current repository does not ship this directory yet.  Keeping the
    loader here lets ``Scene1Env`` provide a clear error when init_states is
    explicitly enabled, instead of silently falling back to random reset.
    """
    root = resolve_scene1_root(scene_root)
    task_spec = get_scene1_task(task) if isinstance(task, str) else task
    candidates = (
        root / "init_states" / f"{task_spec.name}.pt",
    )
    state_path = next((path for path in candidates if path.is_file()), None)
    if state_path is None:
        searched = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            f"No Scene1 init states found for task '{task_spec.name}'. "
            f"Searched: {searched}"
        )
    return torch.load(state_path, weights_only=False)  # nosec B614


def select_scene1_init_state(
    init_states: Sequence[Any],
    episode_index: int,
) -> Any:
    if len(init_states) == 0:
        raise ValueError("Scene1 init state collection is empty")
    return init_states[int(episode_index) % len(init_states)]


def _register_scene1_libero_assets(scene_root: Path) -> None:
    """Register Scene1 BDDL problem and XML object factories in LIBERO.

    LIBERO's ``OffScreenRenderEnv`` dispatches the BDDL ``problem_name``
    through a process-global ``TASK_MAPPING`` and resolves each BDDL object
    category through ``OBJECTS_DICT``.  Scene1 is intentionally not installed
    into the LIBERO package, so register these two small adapters lazily when
    a worker is about to create its own simulator.
    """
    global _SCENE1_REGISTERED_ROOT

    from libero.libero.envs.bddl_base_domain import register_problem
    from libero.libero.envs.base_object import OBJECTS_DICT
    from libero.libero.envs import TASK_MAPPING
    from libero.libero.envs.regions import REGION_SAMPLERS
    from robosuite.models.objects import MujocoXMLObject
    from robosuite.utils.mjcf_utils import string_to_array

    root = resolve_scene1_root(scene_root)
    if _SCENE1_REGISTERED_ROOT is not None and _SCENE1_REGISTERED_ROOT != root:
        raise RuntimeError(
            "Scene1 assets were already registered from a different root: "
            f"{_SCENE1_REGISTERED_ROOT}; cannot switch to {root} in one process"
        )

    object_root = root / "assets" / "scene1_objects"

    class Scene1XMLObject(MujocoXMLObject):
        """MujocoXMLObject adapter for Scene1's extra nested body layer."""

        def _scene1_site_position(self, site_name: str) -> np.ndarray:
            site = self.worldbody.find(
                f".//site[@name='{self.naming_prefix}{site_name}']"
            )
            if site is None:
                raise ValueError(
                    f"Scene1 object '{self.name}' has no site '{site_name}'"
                )
            return string_to_array(site.get("pos"))

        @property
        def bottom_offset(self) -> np.ndarray:
            return self._scene1_site_position("bottom_site")

        @property
        def top_offset(self) -> np.ndarray:
            return self._scene1_site_position("top_site")

        @property
        def horizontal_radius(self) -> float:
            return float(self._scene1_site_position("horizontal_radius_site")[0])

    object_names = {
        "scene1_blue_car",
        "scene1_toy_car_8",
        "scene1_toy_car_2",
        "scene1_fying_glass",
        "scene1_toolbox",
        "scene1_screwdriver",
        "scene1_bottle_4",
        "scene1_bottle_5",
        "scene1_bottle_6",
    }
    for object_name in object_names:
        xml_path = object_root / object_name / f"{object_name}.xml"
        if not xml_path.is_file():
            raise FileNotFoundError(
                f"Scene1 object XML does not exist: {xml_path}"
            )

        # Capture the path as a default argument so each factory keeps its own
        # XML file rather than observing the loop variable after the loop.
        def object_factory(
            *,
            name: str,
            joints: Any = "default",
            _xml_path=xml_path,
            _category_name=object_name,
            **kwargs,
        ):
            obj = Scene1XMLObject(
                fname=str(_xml_path),
                name=name,
                joints=joints,
                obj_type="all",
                duplicate_collision_geoms=False,
                **kwargs,
            )
            # LIBERO's BDDL domain expects every object to expose this small
            # metadata dictionary when it builds visualization/site state.
            obj.object_properties = {"vis_site_names": {}}
            obj.category_name = _category_name
            obj.rotation = (0.0, 0.0)
            obj.rotation_axis = "z"
            return obj

        existing = OBJECTS_DICT.get(object_name)
        if existing is None:
            OBJECTS_DICT[object_name] = object_factory
        elif existing is not object_factory:
            # A second registration is harmless only if it points to the same
            # root.  Keep an existing factory to avoid mutating LIBERO state.
            pass

    problem_name = "scene1_tabletop_manipulation"
    if problem_name not in TASK_MAPPING:
        # The register_problem decorator in this LIBERO version does not
        # return the decorated class, so import the registered base class from
        # TASK_MAPPING rather than from the problem module symbol.
        base_problem = TASK_MAPPING.get("libero_tabletop_manipulation")
        if base_problem is None:
            raise RuntimeError(
                "LIBERO tabletop problem is not registered; cannot create "
                "the Scene1 problem adapter"
            )

        @register_problem
        class Scene1_Tabletop_Manipulation(base_problem):
            pass

    # Scene1 uses the same ``main_table - table`` workspace declaration as
    # LIBERO tabletop tasks, so share the corresponding sampler class.
    REGION_SAMPLERS.setdefault(
        problem_name,
        dict(REGION_SAMPLERS["libero_tabletop_manipulation"]),
    )

    _SCENE1_REGISTERED_ROOT = root


class Scene1Env(gym.Env):
    """Gymnasium wrapper for the single Scene1 pick-and-place task.

    The wrapper intentionally follows ``lerobot.envs.libero.LiberoEnv``'s
    observation and lifecycle contract.  The underlying MuJoCo renderer is
    created lazily on first reset so AsyncVectorEnv workers create their own
    rendering context after process creation.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": DEFAULT_CONTROL_FREQ}

    def __init__(
        self,
        task_suite: Scene1TaskSuite,
        task_id: int,
        task_suite_name: str = "scene1",
        scene_root: str | Path = "data/test_scene1",
        episode_length: int | None = None,
        camera_name: str | Sequence[str] = DEFAULT_CAMERA_NAME,
        obs_type: str = "pixels_agent_pos",
        render_mode: str = "rgb_array",
        observation_width: int = 256,
        observation_height: int = 256,
        visualization_width: int = 640,
        visualization_height: int = 480,
        init_states: bool = False,
        episode_index: int = 0,
        n_envs: int = 1,
        camera_name_mapping: dict[str, str] | None = None,
        num_steps_wait: int = DEFAULT_NUM_STEPS_WAIT,
        control_freq: int = DEFAULT_CONTROL_FREQ,
        control_mode: str = "relative",
        hard_reset: bool = True,
        success_hold_steps: int = DEFAULT_SUCCESS_HOLD_STEPS,
    ):
        super().__init__()

        if not isinstance(task_suite, Scene1TaskSuite):
            raise TypeError(
                "task_suite must be a Scene1TaskSuite, "
                f"got {type(task_suite).__name__}"
            )
        if control_freq <= 0:
            raise ValueError(f"control_freq must be positive, got {control_freq}")
        if not hard_reset and not init_states:
            raise ValueError("hard_reset=False requires init_states=True")
        if observation_width <= 0 or observation_height <= 0:
            raise ValueError("observation dimensions must be positive")
        if episode_length is not None and episode_length <= 0:
            raise ValueError("episode_length must be positive")
        if num_steps_wait < 0:
            raise ValueError("num_steps_wait must be non-negative")
        if success_hold_steps <= 0:
            raise ValueError("success_hold_steps must be positive")
        validate_scene1_control_mode(control_mode)

        self.task_suite = task_suite
        self.task_id = int(task_id)
        self.task_suite_name = task_suite_name
        self._task_spec = task_suite.get_task(self.task_id)
        self.task = self._task_spec.name
        self.task_description = self._task_spec.language

        self.scene_root = resolve_scene1_root(scene_root)
        self._task_bddl_file = resolve_scene1_bddl(
            self.scene_root,
            self._task_spec,
        )

        self.obs_type = obs_type
        self.render_mode = render_mode
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.visualization_width = visualization_width
        self.visualization_height = visualization_height
        self.init_states = init_states
        self.camera_name = parse_camera_names(camera_name)
        if camera_name_mapping is None:
            camera_name_mapping = dict(DEFAULT_CAMERA_NAME_MAPPING)
        self.camera_name_mapping = camera_name_mapping
        validate_scene1_camera_names(self.camera_name)

        self.num_steps_wait = num_steps_wait
        self.control_freq = control_freq
        self.control_mode = control_mode
        self.hard_reset = hard_reset
        self.success_hold_steps = success_hold_steps
        self.episode_length = episode_length
        if episode_length is not None:
            self._max_episode_steps = episode_length
        elif self._task_spec.max_episode_steps is not None:
            self._max_episode_steps = self._task_spec.max_episode_steps
        else:
            self._max_episode_steps = DEFAULT_EPISODE_LENGTH

        self._reset_stride = int(n_envs)
        if self._reset_stride <= 0:
            raise ValueError(f"n_envs must be positive, got {n_envs}")
        self.init_state_id = int(episode_index)

        self._init_states = (
            load_scene1_init_states(self.scene_root, self._task_spec)
            if self.init_states
            else None
        )
        self._env: OffScreenRenderEnv | None = None
        self._episode_step = 0
        self._success_streak = 0
        self._target_initial_pos: np.ndarray | None = None

        images: dict[str, spaces.Box] = {}
        for camera in self.camera_name:
            mapped_name = self.camera_name_mapping.get(camera)
            if mapped_name is None:
                raise ValueError(
                    f"No camera mapping provided for raw camera '{camera}'"
                )
            images[mapped_name] = spaces.Box(
                low=0,
                high=255,
                shape=(self.observation_height, self.observation_width, 3),
                dtype=np.uint8,
            )

        if self.obs_type == "pixels":
            self.observation_space = spaces.Dict({"pixels": spaces.Dict(images)})
        elif self.obs_type == "pixels_agent_pos":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Dict(images),
                    "robot_state": spaces.Dict(
                        {
                            "eef": spaces.Dict(
                                {
                                    "pos": spaces.Box(
                                        low=-np.inf,
                                        high=np.inf,
                                        shape=(3,),
                                        dtype=np.float64,
                                    ),
                                    "quat": spaces.Box(
                                        low=-np.inf,
                                        high=np.inf,
                                        shape=(4,),
                                        dtype=np.float64,
                                    ),
                                    "mat": spaces.Box(
                                        low=-np.inf,
                                        high=np.inf,
                                        shape=(3, 3),
                                        dtype=np.float64,
                                    ),
                                }
                            ),
                            "gripper": spaces.Dict(
                                {
                                    "qpos": spaces.Box(
                                        low=-np.inf,
                                        high=np.inf,
                                        shape=(2,),
                                        dtype=np.float64,
                                    ),
                                    "qvel": spaces.Box(
                                        low=-np.inf,
                                        high=np.inf,
                                        shape=(2,),
                                        dtype=np.float64,
                                    ),
                                }
                            ),
                            "joints": spaces.Dict(
                                {
                                    "pos": spaces.Box(
                                        low=-np.inf,
                                        high=np.inf,
                                        shape=(7,),
                                        dtype=np.float64,
                                    ),
                                    "vel": spaces.Box(
                                        low=-np.inf,
                                        high=np.inf,
                                        shape=(7,),
                                        dtype=np.float64,
                                    ),
                                }
                            ),
                        }
                    ),
                }
            )
        else:
            raise NotImplementedError(
                "Scene1 supports only image observations: "
                "'pixels' or 'pixels_agent_pos'"
            )

        self.action_space = spaces.Box(
            low=ACTION_LOW,
            high=ACTION_HIGH,
            shape=(ACTION_DIM,),
            dtype=np.float32,
        )

    def _ensure_env(self) -> None:
        """Create the OffScreenRenderEnv inside the active worker process."""
        if self._env is not None:
            return

        _register_scene1_libero_assets(self.scene_root)
        env = OffScreenRenderEnv(
            bddl_file_name=str(self._task_bddl_file),
            camera_heights=self.observation_height,
            camera_widths=self.observation_width,
            control_freq=self.control_freq,
            hard_reset=self.hard_reset,
        )
        env.reset()
        self._env = env

    def render(self):
        self._ensure_env()
        assert self._env is not None
        raw_obs = self._env.env._get_observations()
        pixels = self._format_raw_obs(raw_obs)["pixels"]
        image = next(iter(pixels.values()))
        return image[::-1, ::-1]

    def _format_raw_obs(self, raw_obs: RobotObservation) -> RobotObservation:
        assert self._env is not None, "_format_raw_obs called before _ensure_env()"

        images = {}
        for camera_name in self.camera_name:
            if camera_name not in raw_obs:
                raise ValueError(
                    f"Raw Scene1 observation has no camera '{camera_name}'. "
                    f"Available keys: {sorted(raw_obs)}"
                )
            images[self.camera_name_mapping[camera_name]] = raw_obs[camera_name]

        eef_pos = raw_obs.get("robot0_eef_pos")
        eef_quat = raw_obs.get("robot0_eef_quat")
        gripper_qpos = raw_obs.get("robot0_gripper_qpos")
        gripper_qvel = raw_obs.get("robot0_gripper_qvel")
        joint_pos = raw_obs.get("robot0_joint_pos")
        joint_vel = raw_obs.get("robot0_joint_vel")

        eef_mat = None
        if eef_pos is not None:
            try:
                eef_mat = self._env.robots[0].controller.ee_ori_mat
            except (AttributeError, IndexError):
                eef_mat = None

        if self.obs_type == "pixels":
            return {"pixels": images.copy()}

        if self.obs_type == "pixels_agent_pos":
            missing = []
            if eef_pos is None:
                missing.append("robot0_eef_pos")
            if eef_quat is None:
                missing.append("robot0_eef_quat")
            if eef_mat is None:
                missing.append("eef rotation matrix")
            if gripper_qpos is None:
                missing.append("robot0_gripper_qpos")
            if missing:
                raise ValueError(
                    "Missing required Scene1 robot state fields: "
                    + ", ".join(missing)
                )

            return {
                "pixels": images,
                "robot_state": {
                    "eef": {
                        "pos": eef_pos,
                        "quat": eef_quat,
                        "mat": eef_mat,
                    },
                    "gripper": {
                        "qpos": gripper_qpos,
                        "qvel": gripper_qvel,
                    },
                    "joints": {
                        "pos": joint_pos,
                        "vel": joint_vel,
                    },
                },
            }

        raise NotImplementedError(
            f"Unsupported Scene1 observation type '{self.obs_type}'"
        )

    def _set_control_mode(self) -> None:
        assert self._env is not None
        use_delta = self.control_mode == "relative"
        for robot in self._env.robots:
            robot.controller.use_delta = use_delta

    def _get_sim(self):
        """Return the robosuite simulation object used by success checks."""
        assert self._env is not None
        inner = getattr(self._env, "env", self._env)
        sim = getattr(inner, "sim", None)
        if sim is None:
            raise RuntimeError(
                "Scene1 success checks require the underlying robosuite sim"
            )
        return sim

    def _body_position(self, body_name: str) -> np.ndarray:
        sim = self._get_sim()
        model = sim.model
        data = sim.data
        body_name = self._resolve_body_name(body_name)
        try:
            body_id = model.body_name2id(body_name)
        except Exception as exc:
            raise RuntimeError(
                f"Could not find Scene1 body '{body_name}'"
            ) from exc
        return np.asarray(data.body_xpos[body_id], dtype=np.float64).copy()

    def _body_speed(self, body_name: str) -> tuple[float, float]:
        sim = self._get_sim()
        model = sim.model
        data = sim.data
        body_name = self._resolve_body_name(body_name)
        try:
            body_id = model.body_name2id(body_name)
        except Exception as exc:
            raise RuntimeError(
                f"Could not find Scene1 body '{body_name}'"
            ) from exc

        # MuJoCo cvel is [angular velocity, linear velocity].
        cvel = np.asarray(data.cvel[body_id], dtype=np.float64)
        return float(np.linalg.norm(cvel[:3])), float(np.linalg.norm(cvel[3:]))

    def _resolve_body_name(self, object_or_body_name: str) -> str:
        """Translate a BDDL object instance name to its MuJoCo root body."""
        assert self._env is not None
        inner = getattr(self._env, "env", None)
        objects_dict = getattr(inner, "objects_dict", {})
        obj = objects_dict.get(object_or_body_name)
        return getattr(obj, "root_body", object_or_body_name)

    def _site_position(self, site_name: str) -> np.ndarray:
        sim = self._get_sim()
        try:
            site_id = sim.model.site_name2id(site_name)
        except Exception as exc:
            raise RuntimeError(
                f"Could not find Scene1 site '{site_name}'"
            ) from exc
        return np.asarray(sim.data.site_xpos[site_id], dtype=np.float64).copy()

    def _site_containment_bounds(
        self,
        receptacle_name: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return world position, world rotation and local half-size of a site."""
        sim = self._get_sim()
        candidates = (
            f"{receptacle_name}_containment_site",
            f"{receptacle_name}_0_containment_site",
            "containment_site",
        )
        site_id = None
        for candidate in candidates:
            try:
                site_id = sim.model.site_name2id(candidate)
                break
            except Exception:
                continue
        if site_id is None:
            raise RuntimeError(
                f"Could not find containment site for receptacle "
                f"'{receptacle_name}'. Tried {candidates}"
            )

        position = np.asarray(sim.data.site_xpos[site_id], dtype=np.float64).copy()
        rotation = np.asarray(sim.data.site_xmat[site_id], dtype=np.float64).reshape(3, 3).copy()
        size = np.asarray(sim.model.site_size[site_id], dtype=np.float64).copy()
        return position, rotation, size

    def _is_object_inside_containment_site(
        self,
        object_name: str,
        receptacle_name: str,
    ) -> bool:
        object_position = self._body_position(object_name)
        site_position, site_rotation, site_size = self._site_containment_bounds(
            receptacle_name
        )
        local_position = site_rotation.T @ (object_position - site_position)
        # For a box site, MuJoCo's site_size stores the half extents.  The
        # object's centre is intentionally used here; the subsequent speed
        # and release checks prevent a transient edge contact from counting.
        return bool(np.all(np.abs(local_position) <= site_size))

    def _is_gripper_released(self) -> bool:
        assert self._env is not None
        raw_obs = self._env.env._get_observations()
        qpos = raw_obs.get("robot0_gripper_qpos")
        if qpos is None:
            return False
        # Robosuite's parallel gripper qpos is positive when opening.  Keep
        # this threshold conservative; it can be tuned after observing the
        # actual Scene1 controller range.
        return bool(float(np.mean(qpos)) > 0.02)

    def _check_pick_success(self) -> bool:
        target_pos = self._body_position(self._task_spec.target_object)
        if self._target_initial_pos is None:
            return False
        # A pick is considered successful after the target is lifted clear of
        # the tabletop.  The object must also be reasonably stationary so a
        # transient collision is not counted as success.
        lifted = target_pos[2] > self._target_initial_pos[2] + 0.04
        _, linear_speed = self._body_speed(self._task_spec.target_object)
        return bool(lifted and linear_speed < 0.25)

    def _check_place_success(self) -> bool:
        if self._task_spec.receptacle is None:
            return False

        target_inside = self._is_object_inside_containment_site(
            self._task_spec.target_object,
            self._task_spec.receptacle,
        )
        _, linear_speed = self._body_speed(self._task_spec.target_object)
        angular_speed, _ = self._body_speed(self._task_spec.target_object)
        stable = linear_speed < 0.05 and angular_speed < 0.2
        valid = target_inside and self._is_gripper_released() and stable
        self._success_streak = self._success_streak + 1 if valid else 0
        return self._success_streak >= self.success_hold_steps

    def _check_scene1_success(self) -> bool:
        if self._task_spec.success_type == "pick":
            return self._check_pick_success()
        if self._task_spec.success_type == "place":
            return self._check_place_success()
        raise ValueError(
            f"Unknown Scene1 success type '{self._task_spec.success_type}'"
        )

    def reset(self, seed=None, **kwargs):
        self._ensure_env()
        super().reset(seed=seed)
        assert self._env is not None

        self._env.seed(seed)
        raw_obs = self._env.reset()

        if self.init_states and self._init_states is not None:
            init_state = select_scene1_init_state(
                self._init_states,
                self.init_state_id,
            )
            raw_obs = self._env.set_init_state(init_state)
            self.init_state_id += self._reset_stride

        for _ in range(self.num_steps_wait):
            raw_obs, _, _, _ = self._env.step(get_scene1_dummy_action())

        self._set_control_mode()
        self._episode_step = 0
        self._success_streak = 0
        self._target_initial_pos = self._body_position(
            self._task_spec.target_object
        )

        observation = self._format_raw_obs(raw_obs)
        info = {
            "task": self.task,
            "task_id": self.task_id,
            "task_description": self.task_description,
            "is_success": False,
        }
        return observation, info

    def step(self, action: np.ndarray):
        self._ensure_env()
        assert self._env is not None
        action = np.asarray(action, dtype=np.float32)
        if action.ndim != 1 or action.shape != (ACTION_DIM,):
            raise ValueError(
                f"Expected action shape ({ACTION_DIM},), got {action.shape}"
            )

        raw_obs, reward, simulator_done, info = self._env.step(action)
        self._episode_step += 1
        is_success = self._check_scene1_success()
        terminated = bool(simulator_done or is_success)
        truncated = bool(
            self._episode_step >= self._max_episode_steps and not terminated
        )

        info = dict(info or {})
        info.update(
            {
                "task": self.task,
                "task_id": self.task_id,
                "done": bool(simulator_done),
                "is_success": bool(is_success),
            }
        )
        observation = self._format_raw_obs(raw_obs)
        return observation, float(reward), terminated, truncated, info

    def close(self):
        if self._env is not None:
            try:
                self._env.close()
            finally:
                self._env = None


def _make_env_fns(
    *,
    task_suite: Scene1TaskSuite,
    scene_root: str | Path,
    task_id: int,
    n_envs: int,
    camera_names: list[str],
    episode_length: int | None,
    init_states: bool,
    gym_kwargs: Mapping[str, Any],
    control_mode: str,
    camera_name_mapping: dict[str, str] | None = None,
) -> list[Callable[[], Scene1Env]]:
    """Build ``n_envs`` factories for one Scene1 task.

    This follows LIBERO's factory shape closely.  Each callable captures only
    serializable/configuration data; ``Scene1Env`` creates MuJoCo lazily in
    the worker when reset() is first called.
    """
    if not isinstance(n_envs, int) or n_envs <= 0:
        raise ValueError(f"n_envs must be a positive int; got {n_envs}")

    local_gym_kwargs = dict(gym_kwargs)
    # task_ids is consumed by create_scene1_envs(), not by each individual
    # Scene1Env.  Popping it here also prevents duplicate/unexpected kwargs.
    local_gym_kwargs.pop("task_ids", None)

    def _make_env(episode_index: int, **kwargs) -> Scene1Env:
        local_kwargs = dict(kwargs)
        return Scene1Env(
            task_suite=task_suite,
            task_id=task_id,
            task_suite_name="scene1",
            scene_root=scene_root,
            camera_name=camera_names,
            init_states=init_states,
            episode_length=episode_length,
            episode_index=episode_index,
            n_envs=n_envs,
            control_mode=control_mode,
            camera_name_mapping=camera_name_mapping,
            **local_kwargs,
        )

    return [
        partial(_make_env, episode_index, **local_gym_kwargs)
        for episode_index in range(n_envs)
    ]


def create_scene1_envs(
    task: str = "scene1",
    n_envs: int = 1,
    gym_kwargs: dict[str, Any] | None = None,
    scene_root: str | Path = "data/test_scene1",
    camera_name: str | Sequence[str] = DEFAULT_CAMERA_NAME,
    init_states: bool = False,
    env_cls: Callable[[Sequence[Callable[[], Any]]], Any] | None = None,
    control_mode: str = "relative",
    episode_length: int | None = None,
    camera_name_mapping: dict[str, str] | None = None,
) -> dict[str, dict[int, Any]]:
    """Create Scene1 vector environments in LIBERO's return format.

    The ``task`` argument is suite-style and currently must be ``"scene1"``.
    ``task_ids`` can be supplied inside ``gym_kwargs``; if omitted, the sole
    Scene1 task (task_id 0) is selected.
    """
    if env_cls is None or not callable(env_cls):
        raise ValueError(
            "env_cls must be a callable that wraps a list of environment "
            "factory callables"
        )
    if not isinstance(n_envs, int) or n_envs <= 0:
        raise ValueError(f"n_envs must be a positive int; got {n_envs}")

    gym_kwargs = dict(gym_kwargs or {})
    task_ids_filter = gym_kwargs.pop("task_ids", None)

    suite_names = [name.strip() for name in str(task).split(",") if name.strip()]
    if not suite_names:
        raise ValueError("`task` must contain the Scene1 suite name 'scene1'")
    unknown_suites = [name for name in suite_names if name != "scene1"]
    if unknown_suites:
        raise ValueError(
            f"Unknown Scene1 suite(s) {unknown_suites}; only 'scene1' is supported"
        )

    camera_names = parse_camera_names(camera_name)
    validate_scene1_camera_names(camera_names)
    suite = get_scene1_suite(scene_root)
    selected = select_scene1_task_ids(len(suite.tasks), task_ids_filter)
    if not selected:
        raise ValueError("No Scene1 task_ids were selected")

    print(
        f"Creating Scene1 envs | task_ids={selected} | "
        f"n_envs(per task)={n_envs} | init_states={init_states}"
    )

    is_async = env_cls is gym.vector.AsyncVectorEnv
    is_sync = env_cls is gym.vector.SyncVectorEnv

    out: dict[str, dict[int, Any]] = defaultdict(dict)
    cached_obs_space: spaces.Space | None = None
    cached_act_space: spaces.Space | None = None
    cached_metadata: dict[str, Any] | None = None

    for task_id in selected:
        fns = _make_env_fns(
            task_suite=suite,
            scene_root=suite.scene_root,
            task_id=task_id,
            n_envs=n_envs,
            camera_names=camera_names,
            episode_length=episode_length,
            init_states=init_states,
            gym_kwargs=gym_kwargs,
            control_mode=control_mode,
            camera_name_mapping=camera_name_mapping,
        )

        if is_async:
            # As in libero.py, lazy construction avoids creating an EGL/MuJoCo
            # context in the parent process before worker creation.
            vec_env = _LazyAsyncVectorEnv(
                fns,
                cached_obs_space,
                cached_act_space,
                cached_metadata,
            )
            if cached_obs_space is None:
                cached_obs_space = vec_env.observation_space
                cached_act_space = vec_env.action_space
                cached_metadata = vec_env.metadata
        elif is_sync:
            vec_env = gym.vector.SyncVectorEnv(
                fns,
                autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP,
            )
        else:
            vec_env = env_cls(fns)

        out["scene1"][task_id] = vec_env
        print(
            f"Built Scene1 vec env | task_id={task_id} | n_envs={n_envs}"
        )

    return {suite_name: dict(task_map) for suite_name, task_map in out.items()}
