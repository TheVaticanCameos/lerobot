"""Canonical task, layout, and BDDL specifications for the Hybrid1 GLB scene."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True)
class Hybrid1SuccessSemantics:
    kind: str = "pick_and_place"
    receptacle_object: str = "hybrid1_toolbox_1"
    relation: str = "In"
    require_grasp: bool = True
    require_release: bool = True
    minimum_lift_height_delta: float = 0.03
    max_linear_speed: float = 0.05
    max_angular_speed: float = 0.5
    stable_steps: int = 3


@dataclass(frozen=True)
class ResolvedHybrid1Spec:
    task_key: str
    variant_key: str
    prompt: str
    semantics_id: str
    semantics_version: int
    target_object: str
    semantics: Hybrid1SuccessSemantics
    bddl_file: str
    object_names: frozenset[str]
    object_xy: dict[str, tuple[float, float]]
    settle_steps: int = 10


@dataclass(frozen=True)
class Hybrid1BddlInventory:
    language: str
    target_object: str
    object_names: frozenset[str]
    goal_relations: frozenset[tuple[str, str, str]]


@dataclass(frozen=True)
class Hybrid1ObjectPose:
    position: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        position = _finite_tuple(self.position, 3, "position")
        quaternion = _finite_tuple(self.quaternion_wxyz, 4, "quaternion_wxyz")
        if not math.isclose(math.sqrt(sum(value * value for value in quaternion)), 1.0, abs_tol=1e-6):
            raise ValueError("Hybrid1 object quaternion_wxyz must be unit length.")
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "quaternion_wxyz", quaternion)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> Hybrid1ObjectPose:
        if not isinstance(value, Mapping):
            raise ValueError("Hybrid1 object pose must be a mapping.")
        if set(value) != {"position", "quaternion_wxyz"}:
            raise ValueError("Hybrid1 object pose requires exactly position and quaternion_wxyz.")
        try:
            return cls(tuple(value["position"]), tuple(value["quaternion_wxyz"]))
        except TypeError as error:
            raise ValueError("Hybrid1 object pose fields must be numeric sequences.") from error

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "position": list(self.position),
            "quaternion_wxyz": list(self.quaternion_wxyz),
        }


@dataclass(frozen=True)
class Hybrid1LayoutProfile:
    key: str
    xy_jitter: float
    yaw_jitter_radians: float
    x_bounds: tuple[float, float]
    y_bounds: tuple[float, float]
    z_bounds: tuple[float, float]
    minimum_clearance: float
    minimum_bottle_toolbox_separation: float
    max_sampling_attempts: int = 256

    def __post_init__(self) -> None:
        for name in ("x_bounds", "y_bounds", "z_bounds"):
            bounds = _finite_tuple(getattr(self, name), 2, name)
            if bounds[0] > bounds[1]:
                raise ValueError(f"Hybrid1 {name} must be ordered.")
            object.__setattr__(self, name, bounds)
        if self.xy_jitter < 0 or self.yaw_jitter_radians < 0:
            raise ValueError("Hybrid1 layout jitter must be non-negative.")
        if self.minimum_clearance < 0 or self.minimum_bottle_toolbox_separation < 0:
            raise ValueError("Hybrid1 layout clearances must be non-negative.")
        if self.max_sampling_attempts < 1:
            raise ValueError("Hybrid1 max_sampling_attempts must be positive.")


@dataclass(frozen=True)
class Hybrid1ObjectLayout:
    schema: str
    profile: str
    seed: int | None
    poses: Mapping[str, Hybrid1ObjectPose]

    def __post_init__(self) -> None:
        if self.schema != HYBRID1_OBJECT_LAYOUT_SCHEMA:
            raise ValueError(
                f"Unsupported Hybrid1 object layout schema {self.schema!r}; "
                f"expected {HYBRID1_OBJECT_LAYOUT_SCHEMA!r}."
            )
        if not isinstance(self.profile, str):
            raise ValueError("Hybrid1 layout profile must be a string.")
        try:
            profile = HYBRID1_LAYOUT_PROFILES[self.profile]
        except KeyError as error:
            raise ValueError(f"Unknown Hybrid1 layout profile {self.profile!r}.") from error
        if self.seed is not None and (isinstance(self.seed, bool) or not isinstance(self.seed, int)):
            raise ValueError("Hybrid1 layout seed must be an integer or null.")
        if not isinstance(self.poses, Mapping):
            raise ValueError("Hybrid1 object layout poses must be a mapping.")
        if set(self.poses) != HYBRID1_OBJECT_NAMES:
            raise ValueError("Hybrid1 object layout must contain exactly all five dynamic objects.")
        poses = {
            name: pose if isinstance(pose, Hybrid1ObjectPose) else Hybrid1ObjectPose.from_mapping(pose)
            for name, pose in self.poses.items()
        }
        _validate_hybrid1_layout_poses(poses, profile)
        object.__setattr__(self, "poses", MappingProxyType(poses))

    @property
    def identity(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return f"hybrid1-layout-{hashlib.sha256(payload.encode()).hexdigest()}"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> Hybrid1ObjectLayout:
        if not isinstance(value, Mapping):
            raise ValueError("Hybrid1 object layout must be a mapping.")
        if set(value) != {"schema", "profile", "seed", "poses"}:
            raise ValueError("Hybrid1 object layout has unknown or missing fields.")
        poses = value["poses"]
        if not isinstance(poses, Mapping):
            raise ValueError("Hybrid1 object layout poses must be a mapping.")
        return cls(
            schema=value["schema"],
            profile=value["profile"],
            seed=value["seed"],
            poses=poses,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "profile": self.profile,
            "seed": self.seed,
            "poses": {name: self.poses[name].to_dict() for name in sorted(self.poses)},
        }


def _finite_tuple(value: Sequence[Real], length: int, name: str) -> tuple[float, ...]:
    try:
        valid_length = not isinstance(value, (str, bytes)) and len(value) == length
    except TypeError:
        valid_length = False
    if not valid_length:
        raise ValueError(f"Hybrid1 {name} must contain exactly {length} numbers.")
    if any(isinstance(item, bool) or not isinstance(item, Real) for item in value):
        raise ValueError(f"Hybrid1 {name} must contain only numbers.")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"Hybrid1 {name} must contain only finite numbers.")
    return result


HYBRID1_SCENE_ID = "hybrid1"
HYBRID1_SCENE_VERSION = 3
HYBRID1_VARIANT_ID = "full_scene"
HYBRID1_FRUIT_TASK_ID = "pick_and_place_fruit"
HYBRID1_FRUIT_PROMPT = "pick up the fruit and place it into the box"
HYBRID1_FRUIT_SEMANTICS_ID = "hybrid1.pick_and_place_fruit"
HYBRID1_FRUIT_SEMANTICS_VERSION = 2
HYBRID1_FRUIT_TARGET_OBJECT = "hybrid1_fruit_1"
HYBRID1_FRUIT_BDDL_FILE = "hybrid1_pick_and_place_fruit.bddl"
HYBRID1_BOTTLE_TASK_ID = "pick_and_place_bottle"
HYBRID1_BOTTLE_PROMPT = "pick up the bottle and place it into the box"
HYBRID1_BOTTLE_SEMANTICS_ID = "hybrid1.pick_and_place_bottle"
HYBRID1_BOTTLE_SEMANTICS_VERSION = 1
HYBRID1_BOTTLE_TARGET_OBJECT = "hybrid1_bottle_1"
HYBRID1_BOTTLE_BDDL_FILE = "hybrid1_pick_and_place_bottle.bddl"
HYBRID1_OBJECT_POSITIONS = {
    "hybrid1_fruit_1": (0.1272283383, -0.0852129199, 0.9118465921),
    "hybrid1_toolbox_1": (-0.1315192282, 0.1465310799, 0.9061421454),
    "hybrid1_plate_1": (0.1333000029, -0.1416999998, 0.8999999762),
    "hybrid1_bottle_1": (-0.0882067086, -0.1341110990, 0.9018836617),
    "hybrid1_toy_car_1": (0.1405245326, 0.1138844453, 0.9058270454),
}
HYBRID1_OBJECT_XY = {name: position[:2] for name, position in HYBRID1_OBJECT_POSITIONS.items()}
HYBRID1_OBJECT_NAMES = frozenset(HYBRID1_OBJECT_XY)
HYBRID1_OBJECT_LAYOUT_SCHEMA = "hybrid1-object-layout/v1"
HYBRID1_LAYOUT_PROFILES = MappingProxyType(
    {
        "nominal": Hybrid1LayoutProfile(
            key="nominal",
            xy_jitter=0.0,
            yaw_jitter_radians=0.0,
            x_bounds=(-0.18, 0.18),
            y_bounds=(-0.18, 0.18),
            z_bounds=(0.85, 0.98),
            minimum_clearance=0.05,
            minimum_bottle_toolbox_separation=0.20,
        ),
        "controlled-v1": Hybrid1LayoutProfile(
            key="controlled-v1",
            xy_jitter=0.015,
            yaw_jitter_radians=math.radians(15.0),
            x_bounds=(-0.18, 0.18),
            y_bounds=(-0.18, 0.18),
            z_bounds=(0.85, 0.98),
            minimum_clearance=0.045,
            minimum_bottle_toolbox_separation=0.20,
        ),
    }
)
HYBRID1_SUCCESS = Hybrid1SuccessSemantics()


def _validate_hybrid1_layout_poses(
    poses: Mapping[str, Hybrid1ObjectPose], profile: Hybrid1LayoutProfile
) -> None:
    for name, pose in poses.items():
        x, y, z = pose.position
        nominal_x, nominal_y, nominal_z = HYBRID1_OBJECT_POSITIONS[name]
        if not (
            profile.x_bounds[0] <= x <= profile.x_bounds[1]
            and profile.y_bounds[0] <= y <= profile.y_bounds[1]
            and profile.z_bounds[0] <= z <= profile.z_bounds[1]
        ):
            raise ValueError(f"Hybrid1 pose for {name!r} is outside profile bounds.")
        if abs(x - nominal_x) > profile.xy_jitter or abs(y - nominal_y) > profile.xy_jitter:
            raise ValueError(f"Hybrid1 pose for {name!r} exceeds profile XY jitter bounds.")
        if not math.isclose(z, nominal_z, abs_tol=1e-9):
            raise ValueError(f"Hybrid1 pose for {name!r} must preserve its nominal height.")
        w, qx, qy, qz = pose.quaternion_wxyz
        if not math.isclose(qx, 0.0, abs_tol=1e-9) or not math.isclose(qy, 0.0, abs_tol=1e-9):
            raise ValueError(f"Hybrid1 pose for {name!r} must use a yaw-only WXYZ quaternion.")
        yaw = 2.0 * math.atan2(qz, w)
        if abs(yaw) > profile.yaw_jitter_radians + 1e-9:
            raise ValueError(f"Hybrid1 pose for {name!r} exceeds profile yaw bounds.")
    for index, left_name in enumerate(sorted(poses)):
        for right_name in sorted(poses)[index + 1 :]:
            left = poses[left_name].position
            right = poses[right_name].position
            separation = math.dist(left[:2], right[:2])
            minimum = profile.minimum_clearance
            if {left_name, right_name} == {
                "hybrid1_bottle_1",
                "hybrid1_toolbox_1",
            }:
                minimum = max(minimum, profile.minimum_bottle_toolbox_separation)
            if separation < minimum:
                raise ValueError(
                    f"Hybrid1 objects {left_name!r} and {right_name!r} violate minimum clearance."
                )


def _sample_hybrid1_poses(profile: Hybrid1LayoutProfile, rng: random.Random) -> dict[str, Hybrid1ObjectPose]:
    poses = {}
    for name in sorted(HYBRID1_OBJECT_NAMES):
        x, y, z = HYBRID1_OBJECT_POSITIONS[name]
        yaw = rng.uniform(-profile.yaw_jitter_radians, profile.yaw_jitter_radians)
        poses[name] = Hybrid1ObjectPose(
            position=(
                x + rng.uniform(-profile.xy_jitter, profile.xy_jitter),
                y + rng.uniform(-profile.xy_jitter, profile.xy_jitter),
                z,
            ),
            quaternion_wxyz=(math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)),
        )
    return poses


def sample_hybrid1_controlled_layout(seed: int, profile: str = "controlled-v1") -> Hybrid1ObjectLayout:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Hybrid1 controlled layout seed must be an integer.")
    try:
        profile_spec = HYBRID1_LAYOUT_PROFILES[profile]
    except KeyError as error:
        raise ValueError(f"Unknown Hybrid1 layout profile {profile!r}.") from error
    rng = random.Random(seed)
    for _ in range(profile_spec.max_sampling_attempts):
        poses = _sample_hybrid1_poses(profile_spec, rng)
        try:
            return Hybrid1ObjectLayout(
                schema=HYBRID1_OBJECT_LAYOUT_SCHEMA,
                profile=profile,
                seed=seed,
                poses=poses,
            )
        except ValueError:
            continue
    raise ValueError(
        f"Could not sample valid Hybrid1 profile {profile!r} after "
        f"{profile_spec.max_sampling_attempts} attempts."
    )


HYBRID1_NOMINAL_LAYOUT = Hybrid1ObjectLayout(
    schema=HYBRID1_OBJECT_LAYOUT_SCHEMA,
    profile="nominal",
    seed=None,
    poses={
        name: Hybrid1ObjectPose(position=position, quaternion_wxyz=(1.0, 0.0, 0.0, 0.0))
        for name, position in HYBRID1_OBJECT_POSITIONS.items()
    },
)

HYBRID1_TASK_SPECS = {
    HYBRID1_FRUIT_TASK_ID: ResolvedHybrid1Spec(
        task_key=HYBRID1_FRUIT_TASK_ID,
        variant_key=HYBRID1_VARIANT_ID,
        prompt=HYBRID1_FRUIT_PROMPT,
        semantics_id=HYBRID1_FRUIT_SEMANTICS_ID,
        semantics_version=HYBRID1_FRUIT_SEMANTICS_VERSION,
        target_object=HYBRID1_FRUIT_TARGET_OBJECT,
        semantics=HYBRID1_SUCCESS,
        bddl_file=HYBRID1_FRUIT_BDDL_FILE,
        object_names=HYBRID1_OBJECT_NAMES,
        object_xy=dict(HYBRID1_OBJECT_XY),
    ),
    HYBRID1_BOTTLE_TASK_ID: ResolvedHybrid1Spec(
        task_key=HYBRID1_BOTTLE_TASK_ID,
        variant_key=HYBRID1_VARIANT_ID,
        prompt=HYBRID1_BOTTLE_PROMPT,
        semantics_id=HYBRID1_BOTTLE_SEMANTICS_ID,
        semantics_version=HYBRID1_BOTTLE_SEMANTICS_VERSION,
        target_object=HYBRID1_BOTTLE_TARGET_OBJECT,
        semantics=HYBRID1_SUCCESS,
        bddl_file=HYBRID1_BOTTLE_BDDL_FILE,
        object_names=HYBRID1_OBJECT_NAMES,
        object_xy=dict(HYBRID1_OBJECT_XY),
    ),
}


def read_hybrid1_bddl_inventory(bddl_path: str | Path) -> Hybrid1BddlInventory:
    path = Path(bddl_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Hybrid1 BDDL file does not exist: {path}")
    text = path.read_text(encoding="utf-8")
    language_match = re.search(r"\(:language\s+([^()\n]+?)\s*\)", text)
    objects_match = re.search(r"\(:objects\s+(.*?)\)\s*\(:obj_of_interest", text, re.DOTALL)
    target_match = re.search(r"\(:obj_of_interest\s+([^\s()]+)", text)
    if language_match is None or objects_match is None or target_match is None:
        raise ValueError(f"Hybrid1 BDDL {path!s} is missing required task fields.")
    object_names = frozenset(
        re.findall(
            r"^\s*([^\s();]+)\s+-\s+[^\s();]+",
            objects_match.group(1),
            re.MULTILINE,
        )
    )
    target_object = target_match.group(1)
    if target_object not in object_names:
        raise ValueError("Hybrid1 BDDL target is not declared in :objects.")
    goal_start = text.find("(:goal")
    if goal_start < 0:
        raise ValueError("Hybrid1 BDDL is missing :goal.")
    goal_relations = frozenset(
        (relation, subject, object_name)
        for relation, subject, object_name in re.findall(
            r"\((In|On)\s+([^\s()]+)\s+([^\s()]+)\)", text[goal_start:]
        )
    )
    return Hybrid1BddlInventory(
        language=language_match.group(1).strip(),
        target_object=target_object,
        object_names=object_names,
        goal_relations=goal_relations,
    )


def resolve_hybrid1_specs(
    scene_task: str | Sequence[str], scene_variant: str | None = None
) -> list[ResolvedHybrid1Spec]:
    tasks = (
        [item.strip() for item in scene_task.split(",") if item.strip()]
        if isinstance(scene_task, str)
        else list(scene_task)
    )
    if not tasks:
        raise ValueError("Hybrid1 requires at least one task.")
    if len(tasks) != len(set(tasks)):
        raise ValueError(f"Hybrid1 task list contains duplicates: {tasks!r}.")
    unknown = [task for task in tasks if task not in HYBRID1_TASK_SPECS]
    if unknown:
        raise ValueError(
            f"Hybrid1 supports tasks {sorted(HYBRID1_TASK_SPECS)!r}; requested unknown tasks {unknown!r}."
        )
    variant = scene_variant or HYBRID1_VARIANT_ID
    if variant != HYBRID1_VARIANT_ID:
        raise ValueError(f"Hybrid1 supports only variant {HYBRID1_VARIANT_ID!r}; requested {variant!r}.")
    return [HYBRID1_TASK_SPECS[task] for task in tasks]


def validate_hybrid1_bddl(spec: ResolvedHybrid1Spec, bddl_path: str | Path) -> Hybrid1BddlInventory:
    inventory = read_hybrid1_bddl_inventory(bddl_path)
    if inventory.language != spec.prompt:
        raise ValueError("Hybrid1 BDDL language does not match the task prompt.")
    if inventory.target_object != spec.target_object:
        raise ValueError("Hybrid1 BDDL target does not match the task target.")
    if inventory.object_names != spec.object_names:
        raise ValueError("Hybrid1 BDDL object inventory does not match the variant.")
    expected_goal = (
        spec.semantics.relation,
        spec.target_object,
        spec.semantics.receptacle_object,
    )
    if expected_goal not in inventory.goal_relations:
        raise ValueError(f"Hybrid1 BDDL goal must contain {expected_goal!r}.")
    return inventory


__all__ = [
    "HYBRID1_BOTTLE_BDDL_FILE",
    "HYBRID1_BOTTLE_PROMPT",
    "HYBRID1_BOTTLE_SEMANTICS_ID",
    "HYBRID1_BOTTLE_SEMANTICS_VERSION",
    "HYBRID1_BOTTLE_TARGET_OBJECT",
    "HYBRID1_BOTTLE_TASK_ID",
    "HYBRID1_FRUIT_BDDL_FILE",
    "HYBRID1_FRUIT_PROMPT",
    "HYBRID1_FRUIT_SEMANTICS_ID",
    "HYBRID1_FRUIT_SEMANTICS_VERSION",
    "HYBRID1_FRUIT_TARGET_OBJECT",
    "HYBRID1_FRUIT_TASK_ID",
    "HYBRID1_OBJECT_NAMES",
    "HYBRID1_NOMINAL_LAYOUT",
    "HYBRID1_OBJECT_LAYOUT_SCHEMA",
    "HYBRID1_OBJECT_POSITIONS",
    "HYBRID1_OBJECT_XY",
    "HYBRID1_LAYOUT_PROFILES",
    "HYBRID1_SCENE_ID",
    "HYBRID1_SCENE_VERSION",
    "HYBRID1_SUCCESS",
    "HYBRID1_TASK_SPECS",
    "HYBRID1_VARIANT_ID",
    "Hybrid1BddlInventory",
    "Hybrid1LayoutProfile",
    "Hybrid1ObjectLayout",
    "Hybrid1ObjectPose",
    "Hybrid1SuccessSemantics",
    "ResolvedHybrid1Spec",
    "read_hybrid1_bddl_inventory",
    "resolve_hybrid1_specs",
    "sample_hybrid1_controlled_layout",
    "validate_hybrid1_bddl",
]
