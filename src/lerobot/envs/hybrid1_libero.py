#!/usr/bin/env python

"""LIBERO runtime for the independent Hybrid1 GLB scene."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from libero.libero.envs.base_object import OBJECTS_DICT, register_object
from libero.libero.envs.bddl_base_domain import TASK_MAPPING, register_problem
from libero.libero.envs.regions import REGION_SAMPLERS

from .hybrid1_specs import (
    HYBRID1_BOTTLE_PROMPT,
    HYBRID1_BOTTLE_SEMANTICS_ID,
    HYBRID1_BOTTLE_SEMANTICS_VERSION,
    HYBRID1_BOTTLE_TARGET_OBJECT,
    HYBRID1_BOTTLE_TASK_ID,
    HYBRID1_NOMINAL_LAYOUT,
    HYBRID1_SUCCESS,
    HYBRID1_VARIANT_ID,
    Hybrid1ObjectLayout,
    ResolvedHybrid1Spec,
    read_hybrid1_bddl_inventory,
    resolve_hybrid1_specs,
    validate_hybrid1_bddl,
)
from .scene1_libero import Scene1LiberoEnv, Scene1Object

DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[3] / "data" / "mujoco_hybrid1_libero"
DEFAULT_ASSETS_ROOT = DEFAULT_DATA_ROOT
DEFAULT_BDDL_ROOT = DEFAULT_DATA_ROOT / "bddl"


def _assets_root() -> Path:
    return Path(os.environ.get("LEROBOT_HYBRID1_LIBERO_ASSETS", DEFAULT_ASSETS_ROOT)).expanduser().resolve()


class Hybrid1Object(Scene1Object):
    def __init__(self, name: str, joints: list[dict[str, str]] | None = None):
        if joints is None:
            joints = [{"type": "free", "damping": "0.0005"}]
        from robosuite.models.objects import MujocoXMLObject

        MujocoXMLObject.__init__(
            self,
            str(
                _assets_root()
                / "assets"
                / "hybrid1_objects"
                / self.object_category
                / f"{self.object_category}.xml"
            ),
            name=name,
            joints=joints,
            obj_type="all",
            duplicate_collision_geoms=False,
        )
        self.category_name = self.object_category
        self.rotation = (0, 0)
        self.rotation_axis = "z"
        self.object_properties = {"vis_site_names": {}}


@register_object
class Hybrid1Fruit(Hybrid1Object):
    object_category = "hybrid1_fruit"

    def __init__(self, name="hybrid1_fruit", joints=None):
        super().__init__(name=name, joints=joints)


@register_object
class Hybrid1Toolbox(Hybrid1Object):
    object_category = "hybrid1_toolbox"

    def __init__(self, name="hybrid1_toolbox", joints=None):
        super().__init__(name=name, joints=joints)

    def in_box(self, position, object_position):
        local = np.asarray(object_position) - np.asarray(position) - np.array([0.0, 0.0, 0.105])
        return bool(np.all(np.abs(local) <= np.array([0.085, 0.105, 0.085])))


@register_object
class Hybrid1Plate(Hybrid1Object):
    object_category = "hybrid1_plate"

    def __init__(self, name="hybrid1_plate", joints=None):
        super().__init__(name=name, joints=joints)


@register_object
class Hybrid1Bottle(Hybrid1Object):
    object_category = "hybrid1_bottle"

    def __init__(self, name="hybrid1_bottle", joints=None):
        super().__init__(name=name, joints=joints)


@register_object
class Hybrid1ToyCar(Hybrid1Object):
    object_category = "hybrid1_toy_car"

    def __init__(self, name="hybrid1_toy_car", joints=None):
        super().__init__(name=name, joints=joints)


OBJECTS_DICT.update(
    {
        "hybrid1_fruit": Hybrid1Fruit,
        "hybrid1_toolbox": Hybrid1Toolbox,
        "hybrid1_plate": Hybrid1Plate,
        "hybrid1_bottle": Hybrid1Bottle,
        "hybrid1_toy_car": Hybrid1ToyCar,
    }
)


class Hybrid1_Tabletop_Manipulation(  # noqa: N801
    TASK_MAPPING["libero_tabletop_manipulation"]
):
    def __init__(self, bddl_file_name, *args, **kwargs):
        self._hybrid1_target_object = read_hybrid1_bddl_inventory(bddl_file_name).target_object
        kwargs["scene_xml"] = str(_assets_root() / "assets" / "arenas" / "hybrid1_tabletop.xml")
        super().__init__(bddl_file_name, *args, **kwargs)

    def _check_success(self):
        return False

    def _setup_camera(self, mujoco_arena):
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[0.72, 0.0, 1.52],
            quat=[0.6380178, 0.3048497, 0.3048498, 0.6380178],
        )
        mujoco_arena.set_camera(
            camera_name="frontview",
            pos=[1.0, 0.0, 1.48],
            quat=[0.56, 0.43, 0.43, 0.56],
        )


register_problem(Hybrid1_Tabletop_Manipulation)
REGION_SAMPLERS["hybrid1_tabletop_manipulation"] = REGION_SAMPLERS["libero_tabletop_manipulation"]


class Hybrid1LiberoEnv(Scene1LiberoEnv):
    def __init__(
        self,
        bddl_path: str | Path,
        assets_root: str | Path,
        prompt: str = HYBRID1_BOTTLE_PROMPT,
        scene_task: str = HYBRID1_BOTTLE_TASK_ID,
        scene_variant: str = HYBRID1_VARIANT_ID,
        target_object: str = HYBRID1_BOTTLE_TARGET_OBJECT,
        receptacle_object: str = HYBRID1_SUCCESS.receptacle_object,
        semantics_id: str = HYBRID1_BOTTLE_SEMANTICS_ID,
        semantics_version: int = HYBRID1_BOTTLE_SEMANTICS_VERSION,
        success_semantics: Mapping[str, Any] | None = None,
        object_layout: Hybrid1ObjectLayout | Mapping[str, Any] | None = None,
        **kwargs,
    ) -> None:
        os.environ["LEROBOT_HYBRID1_LIBERO_ASSETS"] = str(Path(assets_root).expanduser().resolve())
        semantics = success_semantics or {
            name: getattr(HYBRID1_SUCCESS, name) for name in HYBRID1_SUCCESS.__dataclass_fields__
        }
        self.hybrid1_semantics_id = semantics_id
        self.hybrid1_semantics_version = semantics_version
        if object_layout is None:
            object_layout = HYBRID1_NOMINAL_LAYOUT
        elif isinstance(object_layout, Mapping):
            object_layout = Hybrid1ObjectLayout.from_mapping(object_layout)
        elif not isinstance(object_layout, Hybrid1ObjectLayout):
            raise TypeError("Hybrid1 object_layout must be a Hybrid1ObjectLayout or layout mapping.")
        self.hybrid1_object_layout = object_layout
        domain_randomization = kwargs.pop("domain_randomization", None)
        if _domain_randomization_enabled(domain_randomization):
            raise ValueError("Hybrid1 does not accept inherited Scene1 domain randomization.")
        super().__init__(
            bddl_path=bddl_path,
            assets_root=assets_root,
            prompt=prompt,
            scene_task=scene_task,
            scene_variant=scene_variant,
            layout="original",
            target_object=target_object,
            receptacle_object=receptacle_object,
            success_semantics=semantics,
            unsupported_randomization_profiles=(),
            domain_randomization=domain_randomization,
            **kwargs,
        )

    def _apply_original_scene_layout(self) -> None:
        assert self._env is not None
        sim = self._env.sim
        for object_name, pose in self.hybrid1_object_layout.poses.items():
            joint_id = sim.model.joint_name2id(f"{object_name}_joint0")
            qpos_address = int(sim.model.jnt_qposadr[joint_id])
            dof_address = int(sim.model.jnt_dofadr[joint_id])
            sim.data.qpos[qpos_address : qpos_address + 3] = np.asarray(pose.position)
            sim.data.qpos[qpos_address + 3 : qpos_address + 7] = np.asarray(pose.quaternion_wxyz)
            sim.data.qvel[dof_address : dof_address + 6] = 0.0
        sim.forward()

    def _realized_object_poses(self) -> dict[str, dict[str, list[float]]] | None:
        env = getattr(self, "_env", None)
        if env is None:
            return None
        sim = env.sim
        realized = {}
        for object_name in sorted(self.hybrid1_object_layout.poses):
            joint_id = sim.model.joint_name2id(f"{object_name}_joint0")
            address = int(sim.model.jnt_qposadr[joint_id])
            realized[object_name] = {
                "position": sim.data.qpos[address : address + 3].astype(float).tolist(),
                "quaternion_wxyz": sim.data.qpos[address + 3 : address + 7].astype(float).tolist(),
            }
        return realized

    def _make_episode_metadata(self, randomization: dict[str, Any] | None) -> dict[str, Any]:
        return {
            "hybrid1": {
                "task": self.scene_task,
                "variant": self.scene_variant,
                "instruction": self.task_description,
                "semantics_id": self.hybrid1_semantics_id,
                "semantics_version": self.hybrid1_semantics_version,
                "source_scene": "hybrid_1.glb",
                "bddl_file": Path(self.bddl_path).name,
                "target_object": self.target_object,
                "receptacle_object": self.receptacle_object,
                "object_layout": {
                    "schema": self.hybrid1_object_layout.schema,
                    "identity": self.hybrid1_object_layout.identity,
                    "profile": self.hybrid1_object_layout.profile,
                    "seed": self.hybrid1_object_layout.seed,
                    "requested_poses": self.hybrid1_object_layout.to_dict()["poses"],
                    "realized_poses": self._realized_object_poses(),
                },
                "success_semantics": {
                    "kind": self.success_semantics.kind,
                    "relation": self.success_semantics.relation,
                    "require_grasp": self.success_semantics.require_grasp,
                    "require_release": self.success_semantics.require_release,
                    "minimum_lift_height_delta": (self.success_semantics.minimum_lift_height_delta),
                    "max_linear_speed": self.success_semantics.max_linear_speed,
                    "max_angular_speed": self.success_semantics.max_angular_speed,
                    "stable_steps": self.success_semantics.stable_steps,
                },
            },
            "domain_randomization": randomization,
        }


def create_hybrid1_libero_envs(
    *,
    n_envs: int,
    env_cls: Callable[[Sequence[Callable[[], Any]]], Any],
    scene_task: str | Sequence[str],
    scene_variant: str | None = None,
    assets_root: str | Path,
    bddl_path: str | Path | None = None,
    prompt: str | None = None,
    object_layout: Hybrid1ObjectLayout | Mapping[str, Any] | None = None,
    gym_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, dict[int, Any]]:
    gym_kwargs = dict(gym_kwargs or {})
    conflicting_fields = {"layout", "object_layout"} & gym_kwargs.keys()
    if conflicting_fields:
        fields = ", ".join(sorted(conflicting_fields))
        raise ValueError(f"Hybrid1 layout fields cannot be overridden through gym_kwargs: {fields}")
    if _domain_randomization_enabled(gym_kwargs.get("domain_randomization")):
        raise ValueError("Hybrid1 does not accept inherited Scene1 domain randomization.")
    if isinstance(object_layout, Mapping):
        object_layout = Hybrid1ObjectLayout.from_mapping(object_layout)
    elif object_layout is not None and not isinstance(object_layout, Hybrid1ObjectLayout):
        raise TypeError("Hybrid1 object_layout must be a Hybrid1ObjectLayout or layout mapping.")
    specs = resolve_hybrid1_specs(scene_task, scene_variant)
    envs: dict[str, dict[int, Any]] = {"hybrid1_libero": {}}

    def _make_env(spec: ResolvedHybrid1Spec, task_id: int, **kwargs):
        selected_bddl = Path(bddl_path or DEFAULT_BDDL_ROOT / spec.bddl_file).resolve()
        validate_hybrid1_bddl(spec, selected_bddl)
        return Hybrid1LiberoEnv(
            bddl_path=selected_bddl,
            assets_root=assets_root,
            prompt=prompt or spec.prompt,
            task_id=task_id,
            scene_task=spec.task_key,
            scene_variant=spec.variant_key,
            target_object=spec.target_object,
            receptacle_object=spec.semantics.receptacle_object,
            semantics_id=spec.semantics_id,
            semantics_version=spec.semantics_version,
            object_layout=object_layout,
            success_semantics={
                name: getattr(spec.semantics, name) for name in spec.semantics.__dataclass_fields__
            },
            **kwargs,
        )

    for task_id, spec in enumerate(specs):
        fns = [partial(_make_env, spec, task_id, **gym_kwargs) for _ in range(n_envs)]
        envs["hybrid1_libero"][task_id] = env_cls(fns)
    return envs


def _domain_randomization_enabled(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, Mapping):
        return bool(value.get("enabled", False))
    return bool(getattr(value, "enabled", False))


__all__ = [
    "DEFAULT_ASSETS_ROOT",
    "DEFAULT_BDDL_ROOT",
    "Hybrid1Fruit",
    "Hybrid1Bottle",
    "Hybrid1LiberoEnv",
    "Hybrid1Plate",
    "Hybrid1ToyCar",
    "Hybrid1Toolbox",
    "Hybrid1_Tabletop_Manipulation",
    "create_hybrid1_libero_envs",
]
