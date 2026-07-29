"""Runtime and reset/settle inspection for generated Scene1 variants."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np

from .libero import get_libero_dummy_action
from .scene1_generated_specs import Scene1GeneratedObjectPose
from .scene1_libero import Scene1LiberoEnv
from .scene1_specs import read_scene1_bddl_inventory


def _canonical_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _multiply_quaternions(
    left: np.ndarray, right: np.ndarray
) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )


def _inverse_quaternion(value: np.ndarray) -> np.ndarray:
    return np.array([value[0], -value[1], -value[2], -value[3]], dtype=np.float64)


class Scene1GeneratedVariantEnv(Scene1LiberoEnv):
    """Apply validated table-frame poses after the canonical LIBERO reset."""

    def __init__(
        self,
        *,
        bddl_path: str | Path,
        assets_root: str | Path,
        variant_hash: str,
        object_poses: Mapping[str, Scene1GeneratedObjectPose | Mapping[str, Any]],
        generated_settle_steps: int = 40,
        **kwargs: Any,
    ) -> None:
        if len(variant_hash) != 64 or any(character not in "0123456789abcdef" for character in variant_hash):
            raise ValueError("Generated Scene1 variant_hash must be a lowercase SHA-256 value.")
        if generated_settle_steps < 1:
            raise ValueError("Generated Scene1 settle steps must be positive.")
        inventory = read_scene1_bddl_inventory(bddl_path).object_names
        normalized = {
            object_id: (
                pose
                if isinstance(pose, Scene1GeneratedObjectPose)
                else Scene1GeneratedObjectPose.from_mapping(pose)
            )
            for object_id, pose in object_poses.items()
        }
        if set(normalized) != set(inventory):
            raise ValueError("Generated Scene1 poses must exactly match the BDDL inventory.")
        forbidden = {
            "scene_task",
            "scene_variant",
            "layout",
            "domain_randomization",
            "variant_settle_steps",
        } & set(kwargs)
        if forbidden:
            raise ValueError(f"Generated Scene1 runtime overrides reserved fields: {sorted(forbidden)}.")
        self.generated_variant_hash = variant_hash
        self.generated_object_poses = normalized
        self.generated_settle_steps = generated_settle_steps
        self._generated_baseline_z: dict[str, float] = {}
        self._generated_table_position = np.zeros(3, dtype=np.float64)
        self._generated_table_rotation = np.eye(3, dtype=np.float64)
        self._generated_table_quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        super().__init__(
            bddl_path=bddl_path,
            assets_root=assets_root,
            scene_task="pick_and_place_magnifying_glass",
            scene_variant=variant_hash,
            layout="original",
            domain_randomization={"enabled": False},
            variant_settle_steps=10,
            **kwargs,
        )

    def _capture_table_frame(self) -> None:
        assert self._env is not None
        sim = self._env.sim
        site_id = sim.model.site_name2id("table_top")
        self._generated_table_position = np.asarray(
            sim.data.site_xpos[site_id], dtype=np.float64
        ).copy()
        self._generated_table_rotation = np.asarray(
            sim.data.site_xmat[site_id], dtype=np.float64
        ).reshape(3, 3).copy()
        quaternion = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quaternion, self._generated_table_rotation.reshape(-1))
        self._generated_table_quaternion = quaternion

    def _apply_generated_poses(self) -> None:
        assert self._env is not None
        sim = self._env.sim
        self._capture_table_frame()
        for object_id, pose in self.generated_object_poses.items():
            joint_id = sim.model.joint_name2id(f"{object_id}_joint0")
            qpos_address = int(sim.model.jnt_qposadr[joint_id])
            dof_address = int(sim.model.jnt_dofadr[joint_id])
            baseline_z = float(sim.data.qpos[qpos_address + 2])
            self._generated_baseline_z[object_id] = baseline_z
            table_xy = np.array([pose.position_m[0], pose.position_m[1], 0.0])
            world_position = self._generated_table_position + self._generated_table_rotation @ table_xy
            world_position[2] = baseline_z + pose.position_m[2]
            world_quaternion = _multiply_quaternions(
                self._generated_table_quaternion,
                np.asarray(pose.quaternion_wxyz, dtype=np.float64),
            )
            sim.data.qpos[qpos_address : qpos_address + 3] = world_position
            sim.data.qpos[qpos_address + 3 : qpos_address + 7] = world_quaternion
            sim.data.qvel[dof_address : dof_address + 6] = 0.0
        sim.forward()

    def reset(self, seed: int | None = None, **kwargs: Any):
        super().reset(seed=seed, **kwargs)
        assert self._env is not None
        self._apply_generated_poses()
        raw_obs = self._env.env._get_observations()
        for _ in range(self.generated_settle_steps):
            raw_obs, _, _, _ = self._env.step(get_libero_dummy_action())
        self._last_raw_obs = raw_obs
        self._step_count = 0
        self._stable_success_steps = 0
        self._ever_grasped = False
        self._ever_lifted = False
        self._grasp_start_height = None
        metadata = self._make_episode_metadata(None)
        metadata["scene1"]["generated_variant_hash"] = self.generated_variant_hash
        metadata["scene1"]["generated_settle_steps"] = self.generated_settle_steps
        self._episode_metadata = metadata
        task_progress = self._task_progress(raw_obs, update_stability=False)
        return self._format_raw_obs(raw_obs), {
            "is_success": False,
            "task_progress": task_progress,
            **metadata,
        }

    def inspect_generated_state(
        self,
        info: Mapping[str, Any],
        *,
        placement_mode: str,
        maximum_position_drift_m: float,
        maximum_orientation_drift_rad: float,
    ) -> dict[str, Any]:
        if placement_mode not in {"tabletop", "challenge"}:
            raise ValueError(f"Unknown generated Scene1 placement mode {placement_mode!r}.")
        if maximum_position_drift_m <= 0 or maximum_orientation_drift_rad <= 0:
            raise ValueError("Generated Scene1 drift tolerances must be positive.")
        assert self._env is not None
        sim = self._env.sim
        model = sim.model
        data = sim.data
        enabled = tuple(sorted(self.generated_object_poses))
        instantiated = tuple(sorted(self._env.env.objects_dict))
        free_joints = tuple(
            sorted(
                name
                for joint_id in range(model.njnt)
                if int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE)
                for name in (model.joint_id2name(joint_id),)
                if name is not None
            )
        )
        expected_free_joints = tuple(sorted(f"{object_id}_joint0" for object_id in enabled))

        geom_owner: dict[str, str] = {}
        for object_id in enabled:
            for geom_name in self._env.env.objects_dict[object_id].contact_geoms:
                geom_owner[geom_name] = object_id
        table_geoms = {
            name
            for geom_id in range(model.ngeom)
            for name in (model.geom_id2name(geom_id),)
            if name is not None and name.endswith("table_collision")
        }
        direct_support: set[str] = set()
        object_edges: dict[str, set[str]] = {object_id: set() for object_id in enabled}
        disallowed_contacts: list[dict[str, Any]] = []
        contacts: list[dict[str, Any]] = []
        for contact in data.contact[: data.ncon]:
            geom1 = model.geom_id2name(int(contact.geom1)) or f"geom_{int(contact.geom1)}"
            geom2 = model.geom_id2name(int(contact.geom2)) or f"geom_{int(contact.geom2)}"
            owner1 = geom_owner.get(geom1)
            owner2 = geom_owner.get(geom2)
            distance = float(contact.dist)
            record = {"geom1": geom1, "geom2": geom2, "distance_m": distance}
            contacts.append(record)
            if owner1 is not None and geom2 in table_geoms:
                direct_support.add(owner1)
            if owner2 is not None and geom1 in table_geoms:
                direct_support.add(owner2)
            if owner1 is not None and owner2 is not None and owner1 != owner2:
                object_edges[owner1].add(owner2)
                object_edges[owner2].add(owner1)
                if placement_mode == "tabletop" and distance < -0.001:
                    disallowed_contacts.append(record)
            elif (owner1 is not None or owner2 is not None) and not (
                (owner1 is not None and geom2 in table_geoms)
                or (owner2 is not None and geom1 in table_geoms)
            ) and distance < -0.001:
                disallowed_contacts.append(record)

        supported = set(direct_support)
        changed = True
        while changed:
            changed = False
            for object_id, neighbors in object_edges.items():
                if object_id not in supported and neighbors & supported:
                    supported.add(object_id)
                    changed = True

        objects: dict[str, Any] = {}
        settled = True
        drift_ok = True
        for object_id in enabled:
            joint_id = model.joint_name2id(f"{object_id}_joint0")
            qpos_address = int(model.jnt_qposadr[joint_id])
            dof_address = int(model.jnt_dofadr[joint_id])
            position_world = np.asarray(data.qpos[qpos_address : qpos_address + 3], dtype=np.float64)
            quaternion_world = np.asarray(
                data.qpos[qpos_address + 3 : qpos_address + 7], dtype=np.float64
            )
            table_position = self._generated_table_rotation.T @ (
                position_world - self._generated_table_position
            )
            table_position[2] = position_world[2] - self._generated_baseline_z[object_id]
            table_quaternion = _multiply_quaternions(
                _inverse_quaternion(self._generated_table_quaternion), quaternion_world
            )
            if table_quaternion[0] < 0:
                table_quaternion = -table_quaternion
            linear_velocity = np.asarray(
                data.qvel[dof_address : dof_address + 3], dtype=np.float64
            )
            angular_velocity = np.asarray(
                data.qvel[dof_address + 3 : dof_address + 6], dtype=np.float64
            )
            linear_speed = float(np.linalg.norm(linear_velocity))
            angular_speed = float(np.linalg.norm(angular_velocity))
            requested = self.generated_object_poses[object_id]
            position_drift = float(np.linalg.norm(table_position - requested.position_m))
            quaternion_dot = min(
                1.0,
                abs(float(np.dot(table_quaternion, requested.quaternion_wxyz))),
            )
            orientation_drift = 2.0 * math.acos(quaternion_dot)
            settled = settled and linear_speed <= 0.05 and angular_speed <= 0.5
            drift_ok = (
                drift_ok
                and position_drift <= maximum_position_drift_m
                and orientation_drift <= maximum_orientation_drift_rad
            )
            objects[object_id] = {
                "position_m": [float(value) for value in table_position],
                "quaternion_wxyz": [float(value) for value in table_quaternion],
                "linear_velocity_m_s": [float(value) for value in linear_velocity],
                "angular_velocity_rad_s": [float(value) for value in angular_velocity],
                "position_drift_m": position_drift,
                "orientation_drift_rad": orientation_drift,
                "direct_table_support": object_id in direct_support,
                "support_path_to_table": object_id in supported,
            }

        progress = dict(info.get("task_progress", {}))
        image_observation = self._format_raw_obs(self._last_raw_obs)["pixels"]
        renders_valid = bool(image_observation) and all(
            np.asarray(image).size > 0 and np.isfinite(np.asarray(image)).all()
            for image in image_observation.values()
        )
        checks = {
            "inventory_exact": instantiated == enabled,
            "free_joints_exact": free_joints == expected_free_joints,
            "all_objects_supported": set(enabled) <= supported,
            "all_objects_settled": settled,
            "requested_realized_drift_ok": drift_ok,
            "no_disallowed_penetration": not disallowed_contacts,
            "initial_inside_false": progress.get("inside") is False,
            "initial_success_false": (
                info.get("is_success") is False
                and progress.get("success") is False
                and progress.get("stable_steps") == 0
            ),
            "renders_valid": renders_valid,
        }
        identity: dict[str, Any] = {
            "schema": "scene1-generated-runtime-inspection/v1",
            "variant_hash": self.generated_variant_hash,
            "placement_mode": placement_mode,
            "enabled_object_ids": list(enabled),
            "instantiated_object_ids": list(instantiated),
            "free_joint_names": list(free_joints),
            "objects": objects,
            "contacts": contacts,
            "disallowed_contacts": disallowed_contacts,
            "checks": checks,
            "valid": all(checks.values()),
        }
        return {**identity, "inspection_hash": _canonical_hash(identity)}


def run_scene1_generated_variant_smoke(
    *,
    bddl_path: str | Path,
    assets_root: str | Path,
    variant_hash: str,
    object_poses: Mapping[str, Mapping[str, Any]],
    placement_mode: str,
    seed: int = 0,
    settle_steps: int = 40,
    maximum_position_drift_m: float | None = None,
    maximum_orientation_drift_rad: float | None = None,
) -> dict[str, Any]:
    if maximum_position_drift_m is None:
        maximum_position_drift_m = 0.03 if placement_mode == "tabletop" else 0.06
    if maximum_orientation_drift_rad is None:
        maximum_orientation_drift_rad = 0.35 if placement_mode == "tabletop" else 0.7
    env = Scene1GeneratedVariantEnv(
        bddl_path=bddl_path,
        assets_root=assets_root,
        variant_hash=variant_hash,
        object_poses=object_poses,
        generated_settle_steps=settle_steps,
        observation_width=128,
        observation_height=128,
    )
    try:
        _, info = env.reset(seed=seed)
        return env.inspect_generated_state(
            info,
            placement_mode=placement_mode,
            maximum_position_drift_m=maximum_position_drift_m,
            maximum_orientation_drift_rad=maximum_orientation_drift_rad,
        )
    finally:
        env.close()


def create_scene1_generated_variant_envs(
    *,
    n_envs: int,
    env_cls: Callable[[Sequence[Callable[[], Any]]], Any],
    bddl_path: str | Path,
    assets_root: str | Path,
    variant_hash: str,
    object_poses: Mapping[str, Mapping[str, Any]],
    placement_mode: str,
    generated_settle_steps: int,
    maximum_position_drift_m: float,
    maximum_orientation_drift_rad: float,
    prompt: str | None = None,
    gym_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, dict[int, Any]]:
    """Create a vector env without resolving the generated hash as a catalog variant."""

    if n_envs < 1:
        raise ValueError("Generated Scene1 n_envs must be positive.")
    if placement_mode not in {"tabletop", "challenge"}:
        raise ValueError(f"Unknown generated Scene1 placement mode {placement_mode!r}.")
    gym_kwargs = dict(gym_kwargs or {})
    reserved = {
        "assets_root",
        "bddl_path",
        "domain_randomization",
        "generated_settle_steps",
        "layout",
        "object_poses",
        "placement_mode",
        "prompt",
        "scene_task",
        "scene_variant",
        "variant_hash",
        "variant_settle_steps",
    }
    conflicting = reserved & set(gym_kwargs)
    if conflicting:
        raise ValueError(
            "Generated Scene1 resolved fields cannot be overridden through gym_kwargs: "
            f"{sorted(conflicting)}."
        )
    bddl_path = Path(bddl_path).expanduser().resolve()
    assets_root = Path(assets_root).expanduser().resolve()

    def _make_env(**kwargs: Any) -> Scene1GeneratedVariantEnv:
        return Scene1GeneratedVariantEnv(
            bddl_path=bddl_path,
            assets_root=assets_root,
            variant_hash=variant_hash,
            object_poses=object_poses,
            generated_settle_steps=generated_settle_steps,
            prompt=prompt or "pick up the magnifying glass and place it into the box",
            **kwargs,
        )

    factories = [partial(_make_env, **gym_kwargs) for _ in range(n_envs)]
    vector_kwargs = {"context": "forkserver"} if env_cls is gym.vector.AsyncVectorEnv else {}
    try:
        from gymnasium.vector import AutoresetMode
    except ImportError:
        vector_env = env_cls(factories, **vector_kwargs)
    else:
        try:
            vector_env = env_cls(
                factories,
                autoreset_mode=AutoresetMode.SAME_STEP,
                **vector_kwargs,
            )
        except TypeError as error:
            if "autoreset_mode" not in str(error):
                raise
            vector_env = env_cls(factories, **vector_kwargs)
    vector_env.scene1_generated_placement_mode = placement_mode
    vector_env.scene1_generated_drift_tolerances = (
        maximum_position_drift_m,
        maximum_orientation_drift_rad,
    )
    return {"scene1_generated_variant": {0: vector_env}}


__all__ = [
    "Scene1GeneratedVariantEnv",
    "create_scene1_generated_variant_envs",
    "run_scene1_generated_variant_smoke",
]
