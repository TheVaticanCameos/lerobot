from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from lerobot.envs.hybrid1_specs import HYBRID1_OBJECT_NAMES


@pytest.mark.skipif(
    os.environ.get("RUN_HYBRID1_SIM") != "1",
    reason="Set RUN_HYBRID1_SIM=1 for the high-density MuJoCo integration smoke.",
)
def test_hybrid1_libero_reset_renders_nonterminal_first_frame() -> None:
    hybrid1 = pytest.importorskip("lerobot.envs.hybrid1_libero")
    env = hybrid1.Hybrid1LiberoEnv(
        hybrid1.DEFAULT_BDDL_ROOT / "hybrid1_pick_and_place_fruit.bddl",
        hybrid1.DEFAULT_ASSETS_ROOT,
        observation_width=128,
        observation_height=128,
    )
    try:
        observation, info = env.reset(seed=20000)
        image = env.render()
        model = env._env.sim.model
        hybrid1_collision_ids = [
            geom_id
            for geom_id in range(model.ngeom)
            if model.geom_group[geom_id] == 0
            and (model.geom_id2name(geom_id) or "").startswith("hybrid1_")
        ]
        hybrid1_collision_types = {
            int(model.geom_type[geom_id]) for geom_id in hybrid1_collision_ids
        }
        free_joint_names = {
            model.joint_id2name(joint_id)
            for joint_id in range(model.njnt)
            if int(model.jnt_type[joint_id]) == 0
        }
        env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        dynamic_positions = []
        for object_name in HYBRID1_OBJECT_NAMES:
            joint_id = model.joint_name2id(f"{object_name}_joint0")
            address = int(model.jnt_qposadr[joint_id])
            dynamic_positions.append(env._env.sim.data.qpos[address : address + 3].copy())
    finally:
        env.close()

    assert observation["pixels"]["image"].shape == (128, 128, 3)
    assert observation["pixels"]["image2"].shape == (128, 128, 3)
    assert image.shape == (128, 128, 3)
    assert np.std(image) > 10.0
    data_root = Path(hybrid1.DEFAULT_ASSETS_ROOT)
    asset_manifest = json.loads(
        (data_root / "assets" / "hybrid1_asset_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    expected_collision_count = sum(
        len(item["collision_meshes"])
        for family in ("movable_objects", "static_geometry")
        for item in asset_manifest[family]
    )
    assert len(hybrid1_collision_ids) == expected_collision_count
    assert hybrid1_collision_types == {7}  # mjGEOM_MESH
    assert free_joint_names == {f"{name}_joint0" for name in HYBRID1_OBJECT_NAMES}
    assert np.isfinite(dynamic_positions).all()
    assert np.all(np.asarray(dynamic_positions)[:, 2] > 0.75)
    assert np.all(np.asarray(dynamic_positions)[:, 2] < 1.25)
    assert info["is_success"] is False
    assert info["task_progress"]["inside"] is False
    assert info["hybrid1"]["source_scene"] == "hybrid_1.glb"
