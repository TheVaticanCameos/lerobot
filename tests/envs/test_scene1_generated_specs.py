from __future__ import annotations

from pathlib import Path

import pytest

from lerobot.envs.factory import make_env_config
from lerobot.envs.libero_scenes.contracts import LiberoSceneTaskRequest
from lerobot.envs.libero_scenes.scene1 import SCENE1_LIBERO_SCENE_PLUGIN
from lerobot.envs.scene1_generated_specs import (
    SCENE1_GENERATED_OBJECT_CATALOG,
    render_scene1_generated_bddl,
)
from lerobot.envs.scene1_specs import read_scene1_bddl_inventory

REQUIRED_OBJECTS = {"scene1_fying_glass_1", "scene1_toolbox_1"}


def test_generated_scene1_catalog_is_exact_and_bddl_is_deterministic(tmp_path: Path) -> None:
    assert len(SCENE1_GENERATED_OBJECT_CATALOG) == 9
    assert all(
        object_id == item.object_id and object_id.rsplit("_", 1)[0] == item.asset_id
        for object_id, item in SCENE1_GENERATED_OBJECT_CATALOG.items()
    )
    object_ids = (*REQUIRED_OBJECTS, "scene1_toy_car_8_1")

    first = render_scene1_generated_bddl(object_ids)
    second = render_scene1_generated_bddl(tuple(reversed(object_ids)))
    path = tmp_path / "generated.bddl"
    path.write_text(first, encoding="utf-8")
    inventory = read_scene1_bddl_inventory(path)

    assert first == second
    assert inventory.object_names == set(object_ids)
    assert inventory.target_object == "scene1_fying_glass_1"
    assert (
        "In",
        "scene1_fying_glass_1",
        "scene1_toolbox_1",
    ) in inventory.goal_relations
    assert first.count("scene1_toy_car_8_1 - scene1_toy_car_8") == 1
    assert first.count("(On scene1_toy_car_8_1 main_table_toy_car_8_region)") == 1


def test_generated_bddl_resolves_exact_enabled_assets(tmp_path: Path) -> None:
    enabled = REQUIRED_OBJECTS | {"scene1_toy_car_2_1"}
    path = tmp_path / "generated.bddl"
    path.write_text(render_scene1_generated_bddl(tuple(enabled)), encoding="utf-8")

    resolved = SCENE1_LIBERO_SCENE_PLUGIN.resolve(
        LiberoSceneTaskRequest(
            scene_id="scene1",
            task_id="pick_and_place_magnifying_glass",
            bddl_path=path,
        )
    )

    assert set(resolved.task_metadata["object_names"]) == enabled
    asset_paths = {item.logical_path for item in resolved.asset_files}
    assert any("scene1_toy_car_2" in path for path in asset_paths)
    assert not any("scene1_toy_car_8" in path for path in asset_paths)
    assert not any("scene1_blue_car" in path for path in asset_paths)


@pytest.mark.parametrize(
    "object_ids",
    [(), ("scene1_fying_glass_1",), (*REQUIRED_OBJECTS, "unknown_object_1")],
)
def test_generated_bddl_rejects_invalid_inventory(object_ids: tuple[str, ...]) -> None:
    with pytest.raises(ValueError):
        render_scene1_generated_bddl(object_ids)


def test_generated_variant_env_config_is_registered_and_strict(tmp_path: Path) -> None:
    bddl_path = tmp_path / "generated.bddl"
    bddl_path.write_text(
        render_scene1_generated_bddl(tuple(REQUIRED_OBJECTS)), encoding="utf-8"
    )
    object_poses = {
        object_id: {
            "position_m": [0.1 * index, 0.0, 0.0],
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        }
        for index, object_id in enumerate(sorted(REQUIRED_OBJECTS))
    }

    config = make_env_config(
        "scene1_generated_variant",
        scene_variant="a" * 64,
        bddl_path=str(bddl_path),
        assets_root=str(tmp_path),
        object_poses=object_poses,
        placement_mode="tabletop",
    )

    assert config.type == "scene1_generated_variant"
    assert config.gym_kwargs["episode_length"] == 600
    assert "domain_randomization" not in config.gym_kwargs
    config.domain_randomization = type(config.domain_randomization)(enabled=True)
    with pytest.raises(ValueError, match="does not accept domain randomization"):
        config.create_envs(1)
