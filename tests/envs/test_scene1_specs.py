from __future__ import annotations

from pathlib import Path

import pytest

from lerobot.envs.configs import Scene1LiberoEnv as Scene1LiberoConfig
from lerobot.envs.scene1_specs import (
    SCENE1_LAYOUTS,
    SCENE1_TASK_FAMILIES,
    SCENE1_VARIANTS,
    Scene1LayoutSpec,
    read_scene1_bddl_inventory,
    resolve_scene1_specs,
    validate_scene1_bddl,
)

BDDL_ROOT = Path(__file__).resolve().parents[2] / "data" / "mujoco_scene1_libero" / "bddl"


@pytest.mark.parametrize(
    ("variant", "bddl_file", "layout", "settle_steps"),
    [
        (
            "task_objects_near",
            "scene1_pick_and_place_magnifying_glass_task_objects.bddl",
            "task_objects_near",
            10,
        ),
        (
            "task_objects",
            "scene1_pick_and_place_magnifying_glass_task_objects.bddl",
            "original",
            10,
        ),
        (
            "sparse_distractors",
            "scene1_pick_and_place_magnifying_glass_sparse.bddl",
            "original",
            10,
        ),
        ("full_scene", "scene1_pick_and_place_magnifying_glass.bddl", "original", 10),
        (
            "near_clutter",
            "scene1_pick_and_place_magnifying_glass.bddl",
            "nearby_red_cars_magnifying_glass",
            40,
        ),
        (
            "stacked_clutter",
            "scene1_pick_and_place_magnifying_glass.bddl",
            "cluttered_red_cars_magnifying_glass",
            40,
        ),
    ],
)
def test_canonical_magnifying_glass_matrix(
    variant: str, bddl_file: str, layout: str, settle_steps: int
) -> None:
    spec = resolve_scene1_specs("pick_and_place_magnifying_glass", variant)[0]

    assert spec.variant_key == variant
    assert spec.bddl_file == bddl_file
    assert spec.layout.key == layout
    assert spec.settle_steps == settle_steps
    assert spec.target_object == "scene1_fying_glass_1"
    assert spec.prompt == "pick up the magnifying glass and place it into the box"
    assert spec.semantics.receptacle_object == "scene1_toolbox_1"
    assert spec.semantics.stable_steps == 3
    assert "scene1_toolbox_1" in spec.object_names


def test_magnifying_glass_registry_has_exact_canonical_keys() -> None:
    variants = {
        key
        for key, variant in SCENE1_VARIANTS.items()
        if variant.task_key == "pick_and_place_magnifying_glass"
    }
    assert variants == {
        "task_objects_near",
        "task_objects",
        "sparse_distractors",
        "full_scene",
        "near_clutter",
        "stacked_clutter",
    }


def test_task_defaults_and_legacy_magnifying_glass_names_are_rejected() -> None:
    assert resolve_scene1_specs("pick_red_car")[0].variant_key == "red_car_full"
    assert resolve_scene1_specs("pick_gray_box")[0].variant_key == "gray_box_full"
    assert resolve_scene1_specs("pick_and_place_magnifying_glass")[0].variant_key == "full_scene"

    with pytest.raises(ValueError, match="Unknown scene1 task"):
        resolve_scene1_specs("pick_magnifying_glass")
    with pytest.raises(ValueError, match="Unknown Scene1 variant"):
        resolve_scene1_specs("pick_and_place_magnifying_glass", "magnifying_glass_full")


def test_registry_referential_integrity() -> None:
    assert all(isinstance(layout, Scene1LayoutSpec) for layout in SCENE1_LAYOUTS.values())
    for key, variant in SCENE1_VARIANTS.items():
        assert key == variant.key
        assert variant.task_key in SCENE1_TASK_FAMILIES
        assert variant.layout_key in SCENE1_LAYOUTS
        task = SCENE1_TASK_FAMILIES[variant.task_key]
        assert task.target_object in variant.object_names
        if task.semantics.receptacle_object is not None:
            assert task.semantics.receptacle_object in variant.object_names
        assert SCENE1_LAYOUTS[variant.layout_key].required_objects <= variant.object_names
        assert SCENE1_LAYOUTS[variant.layout_key].pose_randomization_group <= variant.object_names


def test_all_variant_bddls_exist_and_match_declared_inventories() -> None:
    for variant in SCENE1_VARIANTS.values():
        spec = resolve_scene1_specs(variant.task_key, variant.key)[0]
        inventory = validate_scene1_bddl(
            spec,
            BDDL_ROOT / variant.bddl_file,
            require_canonical_inventory=True,
        )
        assert inventory.target_object == spec.target_object
        assert inventory.object_names == spec.object_names
        assert inventory.language == spec.prompt
        if spec.semantics.receptacle_object is not None:
            assert (
                spec.semantics.relation,
                spec.target_object,
                spec.semantics.receptacle_object,
            ) in inventory.goal_relations


def test_bddl_inventory_reports_missing_and_incoherent_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Scene1 BDDL file does not exist"):
        read_scene1_bddl_inventory(tmp_path / "missing.bddl")

    malformed = tmp_path / "malformed.bddl"
    malformed.write_text("(define (problem broken) (:language broken))", encoding="utf-8")
    with pytest.raises(ValueError, match=":objects section"):
        read_scene1_bddl_inventory(malformed)


def test_canonical_variant_rejects_conflicting_bddl_inventory() -> None:
    spec = resolve_scene1_specs("pick_and_place_magnifying_glass", "task_objects")[0]
    with pytest.raises(ValueError, match="conflicts with canonical variant 'task_objects'"):
        validate_scene1_bddl(
            spec,
            BDDL_ROOT / "scene1_pick_and_place_magnifying_glass.bddl",
            require_canonical_inventory=True,
        )


def test_bddl_validation_rejects_wrong_prompt_and_missing_receptacle_goal(tmp_path: Path) -> None:
    spec = resolve_scene1_specs("pick_and_place_magnifying_glass", "task_objects")[0]
    canonical = (BDDL_ROOT / spec.bddl_file).read_text(encoding="utf-8")

    wrong_prompt = tmp_path / "wrong_prompt.bddl"
    wrong_prompt.write_text(
        canonical.replace("place it into the box", "place it in the bin"), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="does not match task prompt"):
        validate_scene1_bddl(spec, wrong_prompt, require_canonical_inventory=True)

    wrong_goal = tmp_path / "wrong_goal.bddl"
    wrong_goal.write_text(
        canonical.replace(
            "(In scene1_fying_glass_1 scene1_toolbox_1)",
            "(On scene1_fying_glass_1 main_table_fying_glass_region)",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="goal must contain"):
        validate_scene1_bddl(spec, wrong_goal, require_canonical_inventory=True)


def test_near_clutter_keeps_red_cars_clear_of_target() -> None:
    layout = SCENE1_LAYOUTS["nearby_red_cars_magnifying_glass"].object_poses
    target = layout["scene1_fying_glass_1"]
    for name in ("scene1_toy_car_8_1", "scene1_toy_car_2_1"):
        pose = layout[name]
        distance = ((pose.x - target.x) ** 2 + (pose.y - target.y) ** 2) ** 0.5
        assert distance > 0.08


def test_stacked_clutter_randomizes_as_a_group_without_disabled_profiles() -> None:
    variant = SCENE1_VARIANTS["stacked_clutter"]
    assert variant.unsupported_randomization_profiles == frozenset()
    assert SCENE1_LAYOUTS[variant.layout_key].pose_randomization_group == {
        "scene1_fying_glass_1",
        "scene1_toy_car_8_1",
        "scene1_toy_car_2_1",
    }


def test_variant_must_belong_to_requested_task() -> None:
    with pytest.raises(ValueError, match="belongs to task"):
        resolve_scene1_specs("pick_gray_box", "task_objects")


def test_scene1_config_exposes_variant_and_custom_bddl_to_factory() -> None:
    config = Scene1LiberoConfig(
        scene_task="pick_and_place_magnifying_glass",
        scene_variant="sparse_distractors",
        bddl_path="custom.bddl",
    )
    assert config.scene_task == "pick_and_place_magnifying_glass"
    assert config.scene_variant == "sparse_distractors"
    assert config.bddl_path == "custom.bddl"
