from __future__ import annotations

import subprocess
import sys

import pytest

from lerobot.envs.libero_scenes.builtins import build_builtin_libero_scene_registry
from lerobot.envs.libero_scenes.contracts import LiberoSceneTaskRequest
from lerobot.envs.libero_scenes.hybrid1 import HYBRID1_LIBERO_SCENE_PLUGIN
from lerobot.envs.libero_scenes.scene1 import SCENE1_LIBERO_SCENE_PLUGIN


def test_builtin_registry_resolves_scene1_without_loading_simulator() -> None:
    registry = build_builtin_libero_scene_registry()

    resolved = registry.resolve(
        LiberoSceneTaskRequest(
            scene_id="scene1",
            task_id="pick_and_place_magnifying_glass",
            variant_id="full_scene",
        )
    )

    assert registry.scene_ids == ("hybrid1", "scene1")
    assert resolved.scene_version == 1
    assert resolved.env_type == "scene1_libero"
    assert resolved.semantics_id == "scene1.pick_and_place_magnifying_glass"
    assert resolved.success.to_dict() == {
        "kind": "pick_and_place",
        "schema_version": 1,
        "parameters": {
            "receptacle_object": "scene1_toolbox_1",
            "relation": "In",
            "minimum_target_height": None,
            "require_grasp": True,
            "require_release": True,
            "minimum_lift_height_delta": 0.03,
            "max_linear_speed": 0.05,
            "max_angular_speed": 0.5,
            "stable_steps": 3,
        },
    }
    assert resolved.bddl.path.is_file()
    assert all(item.path.is_file() for item in resolved.asset_files)
    assert all(item.path.is_file() for item in resolved.implementation_files)


def test_scene1_asset_manifest_contains_only_variant_inventory() -> None:
    resolved = SCENE1_LIBERO_SCENE_PLUGIN.resolve(
        LiberoSceneTaskRequest(
            scene_id="scene1",
            task_id="pick_and_place_magnifying_glass",
            variant_id="task_objects",
        )
    )
    logical_paths = {item.logical_path for item in resolved.asset_files}

    assert logical_paths
    assert all(
        path.startswith("assets/scene1_objects/scene1_fying_glass/")
        or path.startswith("assets/scene1_objects/scene1_toolbox/")
        for path in logical_paths
    )
    assert not any("toy_car" in path for path in logical_paths)
    assert resolved.task_metadata["object_names"] == [
        "scene1_fying_glass_1",
        "scene1_toolbox_1",
    ]


def test_hybrid1_plugin_binds_glb_derived_runtime_assets() -> None:
    resolved = HYBRID1_LIBERO_SCENE_PLUGIN.resolve(
        LiberoSceneTaskRequest(
            scene_id="hybrid1",
            task_id="pick_and_place_fruit",
            variant_id="full_scene",
        )
    )
    logical_paths = {item.logical_path for item in resolved.asset_files}

    assert resolved.env_type == "hybrid1_libero"
    assert resolved.scene_version == 2
    assert resolved.semantics_id == "hybrid1.pick_and_place_fruit"
    assert resolved.semantics_version == 2
    assert resolved.instruction == "pick up the fruit and place it into the box"
    assert resolved.task_metadata["source_scene"] == "hybrid_1.glb"
    assert "assets/arenas/hybrid1_tabletop.xml" in logical_paths
    assert "converted_scene/meshes/table_2.obj" in logical_paths
    assert "converted_scene/meshes/collision/table_2_col_000.obj" in logical_paths
    assert sum("/meshes/collision/" in path for path in logical_paths) == 96
    assert "assets/hybrid1_objects/hybrid1_fruit/hybrid1_fruit.xml" in logical_paths
    for category in ("plate", "bottle", "toy_car"):
        assert (
            f"assets/hybrid1_objects/hybrid1_{category}/hybrid1_{category}.xml"
            in logical_paths
        )
    assert any(item.role == "source_scene" for item in resolved.implementation_files)


def test_registry_rejects_unknown_scene_and_task() -> None:
    registry = build_builtin_libero_scene_registry()

    with pytest.raises(ValueError, match="Unknown LIBERO scene plugin"):
        registry.resolve(LiberoSceneTaskRequest("unknown", "task"))

    with pytest.raises(ValueError, match="Unknown task"):
        registry.resolve(LiberoSceneTaskRequest("scene1", "unknown"))


def test_registry_rejects_duplicate_plugin() -> None:
    registry = build_builtin_libero_scene_registry()

    with pytest.raises(ValueError, match="already registered"):
        registry.register(SCENE1_LIBERO_SCENE_PLUGIN)


def test_generic_registry_import_does_not_load_concrete_scenes() -> None:
    code = """
import sys
import lerobot.envs.libero_scenes.registry

assert "lerobot.envs.libero_scenes.scene1" not in sys.modules
assert "lerobot.envs.libero_scenes.hybrid1" not in sys.modules
assert "lerobot.envs.scene1_specs" not in sys.modules
assert "lerobot.envs.hybrid1_specs" not in sys.modules
assert not any(name == "SceneAgent" or name.startswith("SceneAgent.") for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
