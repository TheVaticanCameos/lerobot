"""Canonical task, layout, and BDDL specifications for the Hybrid1 GLB scene."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


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


HYBRID1_SCENE_ID = "hybrid1"
HYBRID1_SCENE_VERSION = 2
HYBRID1_TASK_ID = "pick_and_place_fruit"
HYBRID1_VARIANT_ID = "full_scene"
HYBRID1_PROMPT = "pick up the fruit and place it into the box"
HYBRID1_SEMANTICS_ID = "hybrid1.pick_and_place_fruit"
HYBRID1_SEMANTICS_VERSION = 2
HYBRID1_TARGET_OBJECT = "hybrid1_fruit_1"
HYBRID1_OBJECT_POSITIONS = {
    "hybrid1_fruit_1": (0.1272283383, -0.0852129199, 0.9118465921),
    "hybrid1_toolbox_1": (-0.1315192282, 0.1465310799, 0.9061421454),
    "hybrid1_plate_1": (0.1333000029, -0.1416999998, 0.8999999762),
    "hybrid1_bottle_1": (-0.0882067086, -0.1341110990, 0.9018836617),
    "hybrid1_toy_car_1": (0.1405245326, 0.1138844453, 0.9058270454),
}
HYBRID1_OBJECT_XY = {
    name: position[:2] for name, position in HYBRID1_OBJECT_POSITIONS.items()
}
HYBRID1_OBJECT_NAMES = frozenset(HYBRID1_OBJECT_XY)
HYBRID1_SUCCESS = Hybrid1SuccessSemantics()
HYBRID1_BDDL_FILE = "hybrid1_pick_and_place_fruit.bddl"


def read_hybrid1_bddl_inventory(bddl_path: str | Path) -> Hybrid1BddlInventory:
    path = Path(bddl_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Hybrid1 BDDL file does not exist: {path}")
    text = path.read_text(encoding="utf-8")
    language_match = re.search(r"\(:language\s+([^()\n]+?)\s*\)", text)
    objects_match = re.search(
        r"\(:objects\s+(.*?)\)\s*\(:obj_of_interest", text, re.DOTALL
    )
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
    if tasks != [HYBRID1_TASK_ID]:
        raise ValueError(
            f"Hybrid1 supports only task {HYBRID1_TASK_ID!r}; requested {tasks!r}."
        )
    variant = scene_variant or HYBRID1_VARIANT_ID
    if variant != HYBRID1_VARIANT_ID:
        raise ValueError(
            f"Hybrid1 supports only variant {HYBRID1_VARIANT_ID!r}; requested {variant!r}."
        )
    return [
        ResolvedHybrid1Spec(
            task_key=HYBRID1_TASK_ID,
            variant_key=HYBRID1_VARIANT_ID,
            prompt=HYBRID1_PROMPT,
            target_object=HYBRID1_TARGET_OBJECT,
            semantics=HYBRID1_SUCCESS,
            bddl_file=HYBRID1_BDDL_FILE,
            object_names=HYBRID1_OBJECT_NAMES,
            object_xy=dict(HYBRID1_OBJECT_XY),
        )
    ]


def validate_hybrid1_bddl(
    spec: ResolvedHybrid1Spec, bddl_path: str | Path
) -> Hybrid1BddlInventory:
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
    "HYBRID1_BDDL_FILE",
    "HYBRID1_OBJECT_NAMES",
    "HYBRID1_OBJECT_POSITIONS",
    "HYBRID1_OBJECT_XY",
    "HYBRID1_PROMPT",
    "HYBRID1_SCENE_ID",
    "HYBRID1_SCENE_VERSION",
    "HYBRID1_SEMANTICS_ID",
    "HYBRID1_SEMANTICS_VERSION",
    "HYBRID1_SUCCESS",
    "HYBRID1_TARGET_OBJECT",
    "HYBRID1_TASK_ID",
    "HYBRID1_VARIANT_ID",
    "Hybrid1BddlInventory",
    "Hybrid1SuccessSemantics",
    "ResolvedHybrid1Spec",
    "read_hybrid1_bddl_inventory",
    "resolve_hybrid1_specs",
    "validate_hybrid1_bddl",
]
