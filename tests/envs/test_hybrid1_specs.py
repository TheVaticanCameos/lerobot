from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from lerobot.envs.hybrid1_specs import (
    HYBRID1_OBJECT_NAMES,
    HYBRID1_PROMPT,
    HYBRID1_SCENE_VERSION,
    HYBRID1_SEMANTICS_ID,
    HYBRID1_SEMANTICS_VERSION,
    HYBRID1_SUCCESS,
    HYBRID1_TARGET_OBJECT,
    resolve_hybrid1_specs,
    validate_hybrid1_bddl,
)

DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "mujoco_hybrid1_libero"

def test_hybrid1_spec_and_bddl_bind_independent_scene_semantics() -> None:
    spec = resolve_hybrid1_specs("pick_and_place_fruit", "full_scene")[0]
    inventory = validate_hybrid1_bddl(
        spec, DATA_ROOT / "bddl" / "hybrid1_pick_and_place_fruit.bddl"
    )

    assert spec.prompt == HYBRID1_PROMPT
    assert spec.prompt == "pick up the fruit and place it into the box"
    assert HYBRID1_SEMANTICS_ID == "hybrid1.pick_and_place_fruit"
    assert HYBRID1_SEMANTICS_VERSION == 2
    assert HYBRID1_SCENE_VERSION == 2
    assert spec.target_object == HYBRID1_TARGET_OBJECT
    assert spec.semantics == HYBRID1_SUCCESS
    assert inventory.object_names == HYBRID1_OBJECT_NAMES
    assert inventory.object_names == {
        "hybrid1_fruit_1",
        "hybrid1_toolbox_1",
        "hybrid1_plate_1",
        "hybrid1_bottle_1",
        "hybrid1_toy_car_1",
    }
    assert ("In", "hybrid1_fruit_1", "hybrid1_toolbox_1") in (
        inventory.goal_relations
    )


def test_hybrid1_rejects_scene1_aliases_and_unknown_variants() -> None:
    with pytest.raises(ValueError, match="supports only task"):
        resolve_hybrid1_specs("pick_and_place_magnifying_glass")
    with pytest.raises(ValueError, match="supports only variant"):
        resolve_hybrid1_specs("pick_and_place_fruit", "task_objects")


def test_hybrid1_asset_manifest_is_locator_independent_and_matches_glb_source() -> None:
    manifest = json.loads(
        (DATA_ROOT / "assets" / "hybrid1_asset_manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert manifest["schema"] == "hybrid1-libero-assets/v3"
    assert manifest["source_glb_logical_path"] == "data/hybrid_1.glb"
    assert len(manifest["source_glb_sha256"]) == 64
    assert manifest["collision_mode"] == "coacd_convex"
    assert {item["object_category"] for item in manifest["movable_objects"]} == {
        "hybrid1_fruit",
        "hybrid1_toolbox",
        "hybrid1_plate",
        "hybrid1_bottle",
        "hybrid1_toy_car",
    }
    assert manifest["dynamics_policy"] == {
        "static_infrastructure": ["table", "robot_base", "robot_mount"],
        "dynamic_object_categories": [
            "hybrid1_bottle",
            "hybrid1_fruit",
            "hybrid1_plate",
            "hybrid1_toolbox",
            "hybrid1_toy_car",
        ],
    }
    assert all(item["dynamic"] for item in manifest["movable_objects"])
    assert {item["joint_type"] for item in manifest["movable_objects"]} == {
        "free"
    }
    assert all(not item["dynamic"] for item in manifest["static_geometry"])
    assert {item["joint_type"] for item in manifest["static_geometry"]} == {None}
    assert not any(
        str(Path.cwd()) in json.dumps(item) for item in manifest["movable_objects"]
    )


def test_hybrid1_runtime_uses_convex_mesh_collisions() -> None:
    manifest = json.loads(
        (DATA_ROOT / "assets" / "hybrid1_asset_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    movable_counts = {
        item["object_category"]: len(item["collision_meshes"])
        for item in manifest["movable_objects"]
    }
    static_counts = {
        item["source_geometry"]: len(item["collision_meshes"])
        for item in manifest["static_geometry"]
    }
    assert set(movable_counts) == {
        "hybrid1_fruit",
        "hybrid1_toolbox",
        "hybrid1_plate",
        "hybrid1_bottle",
        "hybrid1_toy_car",
    }
    assert set(static_counts) == {"table_2"}
    assert all(count > 0 for count in (*movable_counts.values(), *static_counts.values()))

    for category, expected_count in movable_counts.items():
        object_root = ET.parse(
            DATA_ROOT
            / "assets"
            / "hybrid1_objects"
            / category
            / f"{category}.xml"
        ).getroot()
        collision_geoms = object_root.findall(".//body[@name='object']/geom[@group='0']")
        assert len(collision_geoms) == expected_count
        assert {geom.get("type") for geom in collision_geoms} == {"mesh"}

    arena_root = ET.parse(
        DATA_ROOT / "assets" / "arenas" / "hybrid1_tabletop.xml"
    ).getroot()
    table_placeholder = arena_root.find(".//geom[@name='table_collision']")
    assert table_placeholder is not None
    assert table_placeholder.get("contype") == "0"
    assert table_placeholder.get("conaffinity") == "0"
    assert table_placeholder.get("group") == "1"
    static_collision_geoms = [
        geom
        for geom in arena_root.findall(".//geom[@group='0']")
        if geom.get("name", "").startswith("hybrid1_glb_")
    ]
    assert len(static_collision_geoms) == sum(static_counts.values())
    assert {geom.get("type") for geom in static_collision_geoms} == {"mesh"}
    arena_text = ET.tostring(arena_root, encoding="unicode")
    for migrated_category in ("plate", "bottle", "toy_car"):
        assert f"hybrid1_glb_{migrated_category}" not in arena_text
        assert f"hybrid1_{migrated_category}_mesh" not in arena_text
