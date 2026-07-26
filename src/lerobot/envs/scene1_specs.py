"""Canonical task, variant, layout, and BDDL specifications for Scene1."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Scene1ObjectPose:
    x: float
    y: float
    z_offset: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0


@dataclass(frozen=True)
class Scene1LayoutSpec:
    key: str
    object_poses: Mapping[str, Scene1ObjectPose]
    required_objects: frozenset[str] = frozenset()
    pose_randomization_group: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Scene1SuccessSemantics:
    kind: str
    receptacle_object: str | None = None
    relation: str | None = None
    minimum_target_height: float | None = None
    require_grasp: bool = False
    require_release: bool = False
    minimum_lift_height_delta: float = 0.0
    max_linear_speed: float = math.inf
    max_angular_speed: float = math.inf
    stable_steps: int = 1


@dataclass(frozen=True)
class Scene1TaskSpec:
    key: str
    prompt: str
    target_object: str
    semantics: Scene1SuccessSemantics


@dataclass(frozen=True)
class Scene1VariantSpec:
    key: str
    task_key: str
    bddl_file: str
    object_names: frozenset[str]
    layout_key: str = "original"
    settle_steps: int = 10
    unsupported_randomization_profiles: frozenset[str] = frozenset()


@dataclass(frozen=True)
class ResolvedScene1Spec:
    key: str
    task_key: str
    variant_key: str
    prompt: str
    target_object: str
    semantics: Scene1SuccessSemantics
    bddl_file: str
    object_names: frozenset[str]
    layout: Scene1LayoutSpec
    settle_steps: int
    unsupported_randomization_profiles: frozenset[str]


@dataclass(frozen=True)
class Scene1BddlInventory:
    language: str
    target_object: str
    object_names: frozenset[str]
    goal_relations: frozenset[tuple[str, str, str]]


SCENE1_TASK_FAMILIES: dict[str, Scene1TaskSpec] = {
    "pick_red_car": Scene1TaskSpec(
        key="pick_red_car",
        prompt="pick up the red car",
        target_object="scene1_toy_car_8_1",
        semantics=Scene1SuccessSemantics(kind="lift", minimum_target_height=1.0),
    ),
    "pick_and_place_magnifying_glass": Scene1TaskSpec(
        key="pick_and_place_magnifying_glass",
        prompt="pick up the magnifying glass and place it into the box",
        target_object="scene1_fying_glass_1",
        semantics=Scene1SuccessSemantics(
            kind="pick_and_place",
            receptacle_object="scene1_toolbox_1",
            relation="In",
            require_grasp=True,
            require_release=True,
            minimum_lift_height_delta=0.03,
            max_linear_speed=0.05,
            max_angular_speed=0.5,
            stable_steps=3,
        ),
    ),
    "pick_gray_box": Scene1TaskSpec(
        key="pick_gray_box",
        prompt="pick up the gray box",
        target_object="scene1_toolbox_1",
        semantics=Scene1SuccessSemantics(kind="lift", minimum_target_height=1.0),
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

_ORIGINAL_POSES = {name: Scene1ObjectPose(x=x, y=y) for name, (x, y) in SCENE1_ORIGINAL_XY.items()}
_NEAR_CLUTTER_POSES = {
    "scene1_fying_glass_1": Scene1ObjectPose(x=0.125, y=-0.085),
    "scene1_toy_car_8_1": Scene1ObjectPose(
        x=0.025,
        y=-0.06,
        yaw=math.radians(20.0),
    ),
    "scene1_toy_car_2_1": Scene1ObjectPose(
        x=0.17,
        y=0.01,
        yaw=math.radians(-25.0),
    ),
}
_STACKED_CLUTTER_POSES = {
    "scene1_fying_glass_1": Scene1ObjectPose(
        x=0.125,
        y=-0.085,
        yaw=math.radians(18.0),
    ),
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
_TASK_OBJECTS_NEAR_POSES = {
    "scene1_fying_glass_1": _ORIGINAL_POSES["scene1_fying_glass_1"],
    "scene1_toolbox_1": Scene1ObjectPose(x=-0.15, y=-0.10),
}

SCENE1_LAYOUTS: dict[str, Scene1LayoutSpec] = {
    "original": Scene1LayoutSpec(key="original", object_poses=_ORIGINAL_POSES),
    "task_objects_near": Scene1LayoutSpec(
        key="task_objects_near",
        object_poses=_TASK_OBJECTS_NEAR_POSES,
        required_objects=frozenset(_TASK_OBJECTS_NEAR_POSES),
    ),
    "nearby_red_cars_magnifying_glass": Scene1LayoutSpec(
        key="nearby_red_cars_magnifying_glass",
        object_poses=_NEAR_CLUTTER_POSES,
        required_objects=frozenset(_NEAR_CLUTTER_POSES),
    ),
    "cluttered_red_cars_magnifying_glass": Scene1LayoutSpec(
        key="cluttered_red_cars_magnifying_glass",
        object_poses=_STACKED_CLUTTER_POSES,
        required_objects=frozenset(_STACKED_CLUTTER_POSES),
        pose_randomization_group=frozenset(_STACKED_CLUTTER_POSES),
    ),
}

_FULL_SCENE_OBJECTS = frozenset(SCENE1_ORIGINAL_XY)
_TASK_OBJECTS = frozenset({"scene1_fying_glass_1", "scene1_toolbox_1"})
_SPARSE_DISTRACTOR_OBJECTS = _TASK_OBJECTS | {
    "scene1_toy_car_8_1",
    "scene1_toy_car_2_1",
}
SCENE1_VARIANTS: dict[str, Scene1VariantSpec] = {
    "red_car_full": Scene1VariantSpec(
        key="red_car_full",
        task_key="pick_red_car",
        bddl_file="scene1_pick_red_car.bddl",
        object_names=_FULL_SCENE_OBJECTS,
    ),
    "task_objects_near": Scene1VariantSpec(
        key="task_objects_near",
        task_key="pick_and_place_magnifying_glass",
        bddl_file="scene1_pick_and_place_magnifying_glass_task_objects.bddl",
        object_names=_TASK_OBJECTS,
        layout_key="task_objects_near",
    ),
    "task_objects": Scene1VariantSpec(
        key="task_objects",
        task_key="pick_and_place_magnifying_glass",
        bddl_file="scene1_pick_and_place_magnifying_glass_task_objects.bddl",
        object_names=_TASK_OBJECTS,
    ),
    "sparse_distractors": Scene1VariantSpec(
        key="sparse_distractors",
        task_key="pick_and_place_magnifying_glass",
        bddl_file="scene1_pick_and_place_magnifying_glass_sparse.bddl",
        object_names=_SPARSE_DISTRACTOR_OBJECTS,
    ),
    "full_scene": Scene1VariantSpec(
        key="full_scene",
        task_key="pick_and_place_magnifying_glass",
        bddl_file="scene1_pick_and_place_magnifying_glass.bddl",
        object_names=_FULL_SCENE_OBJECTS,
    ),
    "near_clutter": Scene1VariantSpec(
        key="near_clutter",
        task_key="pick_and_place_magnifying_glass",
        bddl_file="scene1_pick_and_place_magnifying_glass.bddl",
        object_names=_FULL_SCENE_OBJECTS,
        layout_key="nearby_red_cars_magnifying_glass",
        settle_steps=40,
    ),
    "stacked_clutter": Scene1VariantSpec(
        key="stacked_clutter",
        task_key="pick_and_place_magnifying_glass",
        bddl_file="scene1_pick_and_place_magnifying_glass.bddl",
        object_names=_FULL_SCENE_OBJECTS,
        layout_key="cluttered_red_cars_magnifying_glass",
        settle_steps=40,
    ),
    "gray_box_full": Scene1VariantSpec(
        key="gray_box_full",
        task_key="pick_gray_box",
        bddl_file="scene1_pick_gray_box.bddl",
        object_names=_FULL_SCENE_OBJECTS,
    ),
}

SCENE1_TASK_DEFAULT_VARIANTS: dict[str, str] = {
    "pick_red_car": "red_car_full",
    "pick_and_place_magnifying_glass": "full_scene",
    "pick_gray_box": "gray_box_full",
}


def read_scene1_bddl_inventory(bddl_path: str | Path) -> Scene1BddlInventory:
    path = Path(bddl_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Scene1 BDDL file does not exist: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise OSError(f"Unable to read Scene1 BDDL file '{path}': {error}") from error

    language_match = re.search(r"\(:language\s+([^()\n]+?)\s*\)", text)
    if language_match is None:
        raise ValueError(f"Scene1 BDDL '{path}' is missing a parseable :language field.")
    language = language_match.group(1).strip()

    objects_match = re.search(r"\(:objects\s+(.*?)\)\s*\(:obj_of_interest", text, re.DOTALL)
    if objects_match is None:
        raise ValueError(f"Scene1 BDDL '{path}' is missing a parseable :objects section.")
    object_names = frozenset(
        re.findall(r"^\s*([^\s();]+)\s+-\s+[^\s();]+", objects_match.group(1), re.MULTILINE)
    )
    if not object_names:
        raise ValueError(f"Scene1 BDDL '{path}' declares no objects.")

    target_match = re.search(r"\(:obj_of_interest\s+([^\s()]+)", text)
    if target_match is None:
        raise ValueError(f"Scene1 BDDL '{path}' is missing a single :obj_of_interest target.")
    target_object = target_match.group(1)
    if target_object not in object_names:
        raise ValueError(
            f"Scene1 BDDL '{path}' target '{target_object}' is not declared in its :objects section."
        )
    goal_start = text.find("(:goal")
    if goal_start < 0:
        raise ValueError(f"Scene1 BDDL '{path}' is missing a :goal section.")
    goal_relations = frozenset(
        (relation, subject, object_name)
        for relation, subject, object_name in re.findall(
            r"\((In|On)\s+([^\s()]+)\s+([^\s()]+)\)", text[goal_start:]
        )
    )
    return Scene1BddlInventory(
        language=language,
        target_object=target_object,
        object_names=object_names,
        goal_relations=goal_relations,
    )


def validate_scene1_bddl(
    spec: ResolvedScene1Spec,
    bddl_path: str | Path,
    *,
    require_canonical_inventory: bool,
) -> Scene1BddlInventory:
    inventory = read_scene1_bddl_inventory(bddl_path)
    if inventory.language != spec.prompt:
        raise ValueError(
            f"Scene1 BDDL '{bddl_path}' language {inventory.language!r} does not match task "
            f"prompt {spec.prompt!r}."
        )
    if inventory.target_object != spec.target_object:
        raise ValueError(
            f"Scene1 BDDL '{bddl_path}' targets '{inventory.target_object}', but task "
            f"'{spec.task_key}' targets '{spec.target_object}'."
        )
    receptacle = spec.semantics.receptacle_object
    if receptacle is not None:
        if receptacle not in inventory.object_names:
            raise ValueError(f"Scene1 BDDL '{bddl_path}' is missing task receptacle '{receptacle}'.")
        expected_relation = (spec.semantics.relation or "In", spec.target_object, receptacle)
        if expected_relation not in inventory.goal_relations:
            relation, target, receptacle = expected_relation
            raise ValueError(
                f"Scene1 BDDL '{bddl_path}' goal must contain ({relation} {target} {receptacle})."
            )
    missing_layout_objects = spec.layout.required_objects - inventory.object_names
    if missing_layout_objects:
        missing = ", ".join(sorted(missing_layout_objects))
        raise ValueError(
            f"Scene1 layout '{spec.layout.key}' requires objects absent from BDDL '{bddl_path}': {missing}"
        )
    if require_canonical_inventory and inventory.object_names != spec.object_names:
        missing = ", ".join(sorted(spec.object_names - inventory.object_names)) or "none"
        unexpected = ", ".join(sorted(inventory.object_names - spec.object_names)) or "none"
        raise ValueError(
            f"Scene1 BDDL '{bddl_path}' conflicts with canonical variant '{spec.variant_key}' "
            f"(missing objects: {missing}; unexpected objects: {unexpected})."
        )
    return inventory


def resolve_scene1_specs(
    scene_task: str | Sequence[str],
    scene_variant: str | None = None,
) -> list[ResolvedScene1Spec]:
    task_keys = (
        [key.strip() for key in scene_task.split(",") if key.strip()]
        if isinstance(scene_task, str)
        else list(scene_task)
    )
    if not task_keys:
        raise ValueError("At least one scene1 task must be specified.")
    if scene_variant is not None and len(task_keys) != 1:
        raise ValueError("scene_variant can only be used with one scene1 task.")

    resolved = []
    for requested_key in task_keys:
        try:
            task = SCENE1_TASK_FAMILIES[requested_key]
            default_variant = SCENE1_TASK_DEFAULT_VARIANTS[requested_key]
        except KeyError as error:
            available = ", ".join(sorted(SCENE1_TASK_FAMILIES))
            raise ValueError(
                f"Unknown scene1 task '{requested_key}'. Available tasks: {available}"
            ) from error
        requested_variant = scene_variant or default_variant
        variant_key = requested_variant
        try:
            variant = SCENE1_VARIANTS[variant_key]
            layout = SCENE1_LAYOUTS[variant.layout_key]
        except KeyError as error:
            available = ", ".join(sorted(SCENE1_VARIANTS))
            raise ValueError(
                f"Unknown Scene1 variant '{requested_variant}'. Available variants: {available}"
            ) from error
        if variant.task_key != task.key:
            raise ValueError(
                f"Variant '{variant.key}' belongs to task '{variant.task_key}', not '{task.key}'."
            )
        if task.target_object not in variant.object_names:
            raise ValueError(
                f"Variant '{variant.key}' does not contain target object '{task.target_object}'."
            )
        missing_layout_objects = layout.required_objects - variant.object_names
        if missing_layout_objects:
            missing = ", ".join(sorted(missing_layout_objects))
            raise ValueError(f"Variant '{variant.key}' layout '{layout.key}' is missing objects: {missing}")
        resolved.append(
            ResolvedScene1Spec(
                key=requested_key,
                task_key=task.key,
                variant_key=variant.key,
                prompt=task.prompt,
                target_object=task.target_object,
                semantics=task.semantics,
                bddl_file=variant.bddl_file,
                object_names=variant.object_names,
                layout=layout,
                settle_steps=variant.settle_steps,
                unsupported_randomization_profiles=variant.unsupported_randomization_profiles,
            )
        )
    return resolved
