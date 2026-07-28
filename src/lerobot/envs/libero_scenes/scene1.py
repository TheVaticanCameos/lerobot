"""Scene1 resolver exposed through the generic LIBERO scene contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from lerobot.envs.scene1_specs import (
    SCENE1_TASK_FAMILIES,
    resolve_scene1_specs,
    validate_scene1_bddl,
)

from .contracts import (
    LiberoSceneDescriptor,
    LiberoSceneTaskRequest,
    ResolvedLiberoSceneTask,
    ResolvedSceneFile,
    ResolvedSuccessContract,
)

_OBJECT_ASSET_DIRECTORIES = {
    "scene1_blue_car_1": "scene1_blue_car",
    "scene1_toy_car_8_1": "scene1_toy_car_8",
    "scene1_toy_car_2_1": "scene1_toy_car_2",
    "scene1_toolbox_1": "scene1_toolbox",
    "scene1_fying_glass_1": "scene1_fying_glass",
    "scene1_screwdriver_1": "scene1_screwdriver",
    "scene1_bottle_4_1": "scene1_bottle_4",
    "scene1_bottle_5_1": "scene1_bottle_5",
    "scene1_bottle_6_1": "scene1_bottle_6",
}
_SEMANTICS = {
    task_id: (f"scene1.{task_id}", 1) for task_id in SCENE1_TASK_FAMILIES
}


def _default_data_root() -> Path:
    return Path(__file__).resolve().parents[4] / "data" / "mujoco_scene1_libero"


def _files_under(root: Path, *, logical_root: str, role: str) -> tuple[ResolvedSceneFile, ...]:
    if not root.is_dir():
        raise FileNotFoundError(f"Scene1 asset directory does not exist: {root}")
    files = tuple(
        ResolvedSceneFile(
            logical_path=f"{logical_root}/{path.relative_to(root).as_posix()}",
            path=path,
            role=role,
        )
        for path in sorted(root.rglob("*"))
        if path.is_file()
    )
    if not files:
        raise ValueError(f"Scene1 asset directory contains no files: {root}")
    return files


@dataclass(frozen=True, slots=True)
class Scene1LiberoScenePlugin:
    descriptor: LiberoSceneDescriptor = LiberoSceneDescriptor(
        scene_id="scene1",
        scene_version=1,
        task_ids=tuple(sorted(SCENE1_TASK_FAMILIES)),
    )

    def resolve(self, request: LiberoSceneTaskRequest) -> ResolvedLiberoSceneTask:
        if request.scene_id != self.descriptor.scene_id:
            raise ValueError(
                f"Scene1 plugin cannot resolve scene {request.scene_id!r}."
            )
        resolved = resolve_scene1_specs(request.task_id, request.variant_id)[0]
        data_root = _default_data_root()
        assets_root = (
            request.assets_root.expanduser().resolve()
            if request.assets_root is not None
            else (data_root / "assets").resolve()
        )
        bddl_path = (
            request.bddl_path.expanduser().resolve()
            if request.bddl_path is not None
            else (data_root / "bddl" / resolved.bddl_file).resolve()
        )
        inventory = validate_scene1_bddl(
            resolved,
            bddl_path,
            require_canonical_inventory=(
                request.bddl_path is None or request.variant_id is not None
            ),
        )
        asset_files: list[ResolvedSceneFile] = []
        for object_name in sorted(inventory.object_names):
            try:
                directory_name = _OBJECT_ASSET_DIRECTORIES[object_name]
            except KeyError as error:
                raise ValueError(
                    f"Scene1 object {object_name!r} has no registered asset directory."
                ) from error
            asset_files.extend(
                _files_under(
                    assets_root / "scene1_objects" / directory_name,
                    logical_root=f"assets/scene1_objects/{directory_name}",
                    role="mujoco_asset",
                )
            )

        success_parameters = asdict(resolved.semantics)
        success_kind = success_parameters.pop("kind")
        envs_root = Path(__file__).resolve().parents[1]
        plugin_root = Path(__file__).resolve().parent
        implementation_files = tuple(
            ResolvedSceneFile(logical_path=logical_path, path=path, role="implementation")
            for logical_path, path in (
                ("lerobot/envs/scene1_specs.py", envs_root / "scene1_specs.py"),
                (
                    "lerobot/envs/scene1_randomization.py",
                    envs_root / "scene1_randomization.py",
                ),
                ("lerobot/envs/scene1_libero.py", envs_root / "scene1_libero.py"),
                ("lerobot/envs/libero_scenes/contracts.py", plugin_root / "contracts.py"),
                ("lerobot/envs/libero_scenes/registry.py", plugin_root / "registry.py"),
                ("lerobot/envs/libero_scenes/scene1.py", Path(__file__).resolve()),
            )
        )
        semantics_id, semantics_version = _SEMANTICS[resolved.task_key]
        return ResolvedLiberoSceneTask(
            scene_id=self.descriptor.scene_id,
            scene_version=self.descriptor.scene_version,
            task_id=resolved.task_key,
            variant_id=resolved.variant_key,
            instruction=resolved.prompt,
            semantics_id=semantics_id,
            semantics_version=semantics_version,
            success=ResolvedSuccessContract(
                kind=success_kind,
                schema_version=1,
                parameters=success_parameters,
            ),
            bddl=ResolvedSceneFile(
                logical_path=f"bddl/{bddl_path.name}",
                path=bddl_path,
                role="task_definition",
            ),
            assets_root=assets_root,
            asset_files=tuple(asset_files),
            implementation_files=implementation_files,
            env_type="scene1_libero",
            task_metadata={
                "target_object": resolved.target_object,
                "receptacle_object": resolved.semantics.receptacle_object,
                "object_names": sorted(inventory.object_names),
            },
            runtime_options={
                "layout_id": resolved.layout.key,
                "settle_steps": resolved.settle_steps,
                "unsupported_randomization_profiles": sorted(
                    resolved.unsupported_randomization_profiles
                ),
            },
        )


SCENE1_LIBERO_SCENE_PLUGIN = Scene1LiberoScenePlugin()


__all__ = ["SCENE1_LIBERO_SCENE_PLUGIN", "Scene1LiberoScenePlugin"]
