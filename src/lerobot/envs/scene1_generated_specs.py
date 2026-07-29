"""Deterministic BDDL compiler for generated Scene1 object-pool variants."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Scene1GeneratedCatalogObject:
    object_id: str
    asset_id: str
    region_name: str
    canonical_region: tuple[float, float, float, float]
    half_extents_m: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class Scene1GeneratedObjectPose:
    position_m: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if len(self.position_m) != 3 or len(self.quaternion_wxyz) != 4:
            raise ValueError("Generated Scene1 poses require position and quaternion values.")
        values = (*self.position_m, *self.quaternion_wxyz)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("Generated Scene1 pose values must be finite.")
        norm = math.sqrt(sum(value * value for value in self.quaternion_wxyz))
        if not math.isclose(norm, 1.0, abs_tol=1e-6):
            raise ValueError("Generated Scene1 pose quaternion must be normalized.")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> Scene1GeneratedObjectPose:
        if set(value) != {"position_m", "quaternion_wxyz"}:
            raise ValueError("Generated Scene1 pose contains unsupported fields.")
        return cls(
            position_m=tuple(float(item) for item in value["position_m"]),
            quaternion_wxyz=tuple(float(item) for item in value["quaternion_wxyz"]),
        )


SCENE1_GENERATED_OBJECT_CATALOG: dict[str, Scene1GeneratedCatalogObject] = {
    item.object_id: item
    for item in (
        Scene1GeneratedCatalogObject(
            "scene1_blue_car_1",
            "scene1_blue_car",
            "blue_car_region",
            (0.2606, -0.0838, 0.4394, 0.0302),
            (0.08935, 0.05694, 0.03),
        ),
        Scene1GeneratedCatalogObject(
            "scene1_toy_car_8_1",
            "scene1_toy_car_8",
            "toy_car_8_region",
            (-0.0765, 0.0515, 0.1005, 0.1763),
            (0.08843, 0.06232, 0.03),
        ),
        Scene1GeneratedCatalogObject(
            "scene1_toy_car_2_1",
            "scene1_toy_car_2",
            "toy_car_2_region",
            (-0.2175, -0.2100, -0.0445, -0.0764),
            (0.08644, 0.06675, 0.03),
        ),
        Scene1GeneratedCatalogObject(
            "scene1_toolbox_1",
            "scene1_toolbox",
            "toolbox_region",
            (-0.4995, -0.0293, -0.2994, 0.2208),
            (0.1, 0.125, 0.1),
        ),
        Scene1GeneratedCatalogObject(
            "scene1_fying_glass_1",
            "scene1_fying_glass",
            "fying_glass_region",
            (0.0606, -0.1710, 0.2346, -0.0353),
            (0.08694, 0.06778, 0.0075),
        ),
        Scene1GeneratedCatalogObject(
            "scene1_screwdriver_1",
            "scene1_screwdriver",
            "screwdriver_region",
            (-0.4410, -0.2176, -0.3150, -0.1286),
            (0.06294, 0.04443, 0.005),
        ),
        Scene1GeneratedCatalogObject(
            "scene1_bottle_4_1",
            "scene1_bottle_4",
            "bottle_4_region",
            (0.3648, 0.1683, 0.3998, 0.2035),
            (0.01745, 0.01753, 0.035),
        ),
        Scene1GeneratedCatalogObject(
            "scene1_bottle_5_1",
            "scene1_bottle_5",
            "bottle_5_region",
            (0.2486, 0.1472, 0.2838, 0.1824),
            (0.0175, 0.0175, 0.035),
        ),
        Scene1GeneratedCatalogObject(
            "scene1_bottle_6_1",
            "scene1_bottle_6",
            "bottle_6_region",
            (0.2811, 0.1975, 0.3163, 0.2327),
            (0.0175, 0.0175, 0.035),
        ),
    )
}


def render_scene1_generated_bddl(enabled_object_ids: Sequence[str]) -> str:
    """Render an inventory-only patch; the generated runtime applies exact poses."""

    object_ids = tuple(sorted(enabled_object_ids))
    if not object_ids or len(set(object_ids)) != len(object_ids):
        raise ValueError("Generated Scene1 inventory must be non-empty and unique.")
    unknown = set(object_ids) - set(SCENE1_GENERATED_OBJECT_CATALOG)
    if unknown:
        raise ValueError(f"Unknown generated Scene1 objects: {sorted(unknown)}.")
    required = {"scene1_fying_glass_1", "scene1_toolbox_1"}
    if not required <= set(object_ids):
        raise ValueError("Generated Scene1 inventory must retain target and receptacle.")

    catalog = [SCENE1_GENERATED_OBJECT_CATALOG[object_id] for object_id in object_ids]
    region_lines = [
        "    "
        f"({item.region_name} (:target main_table) "
        f"(:ranges (({' '.join(f'{value:.4f}' for value in item.canonical_region)}))))"
        for item in catalog
    ]
    object_lines = [f"    {item.object_id} - {item.asset_id}" for item in catalog]
    init_lines = [
        f"    (On {item.object_id} main_table_{item.region_name})" for item in catalog
    ]
    return "\n".join(
        (
            "(define (problem Scene1_Tabletop_Manipulation)",
            "  (:domain robosuite)",
            "  (:language pick up the magnifying glass and place it into the box)",
            "  (:regions",
            *region_lines,
            "  )",
            "",
            "  (:fixtures",
            "    main_table - table",
            "  )",
            "",
            "  (:objects",
            *object_lines,
            "  )",
            "",
            "  (:obj_of_interest",
            "    scene1_fying_glass_1",
            "  )",
            "",
            "  (:init",
            *init_lines,
            "  )",
            "",
            "  (:goal",
            "    (And (In scene1_fying_glass_1 scene1_toolbox_1))",
            "  )",
            ")",
            "",
        )
    )


__all__ = [
    "SCENE1_GENERATED_OBJECT_CATALOG",
    "Scene1GeneratedCatalogObject",
    "Scene1GeneratedObjectPose",
    "render_scene1_generated_bddl",
]
