"""Resolver for the independent Hybrid1 GLB-backed LIBERO scene."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from lerobot.envs.hybrid1_specs import (
    HYBRID1_SCENE_ID,
    HYBRID1_SCENE_VERSION,
    HYBRID1_SEMANTICS_ID,
    HYBRID1_SEMANTICS_VERSION,
    HYBRID1_TASK_ID,
    resolve_hybrid1_specs,
    validate_hybrid1_bddl,
)

from .contracts import (
    LiberoSceneDescriptor,
    LiberoSceneTaskRequest,
    ResolvedLiberoSceneTask,
    ResolvedSceneFile,
    ResolvedSuccessContract,
)


def _data_root() -> Path:
    return Path(__file__).resolve().parents[4] / "data" / "mujoco_hybrid1_libero"


def _file(logical_path: str, path: Path, role: str) -> ResolvedSceneFile:
    if not path.is_file():
        raise FileNotFoundError(f"Hybrid1 asset file does not exist: {path}")
    return ResolvedSceneFile(logical_path=logical_path, path=path, role=role)


def _files_under(root: Path, logical_root: str) -> list[ResolvedSceneFile]:
    if not root.is_dir():
        raise FileNotFoundError(f"Hybrid1 asset directory does not exist: {root}")
    return [
        _file(
            f"{logical_root}/{path.relative_to(root).as_posix()}",
            path,
            "mujoco_asset",
        )
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


@dataclass(frozen=True, slots=True)
class Hybrid1LiberoScenePlugin:
    descriptor: LiberoSceneDescriptor = LiberoSceneDescriptor(
        scene_id=HYBRID1_SCENE_ID,
        scene_version=HYBRID1_SCENE_VERSION,
        task_ids=(HYBRID1_TASK_ID,),
    )

    def resolve(self, request: LiberoSceneTaskRequest) -> ResolvedLiberoSceneTask:
        if request.scene_id != HYBRID1_SCENE_ID:
            raise ValueError(f"Hybrid1 plugin cannot resolve {request.scene_id!r}.")
        spec = resolve_hybrid1_specs(request.task_id, request.variant_id)[0]
        data_root = _data_root()
        assets_root = (request.assets_root or data_root).resolve()
        bddl_path = (request.bddl_path or data_root / "bddl" / spec.bddl_file).resolve()
        inventory = validate_hybrid1_bddl(spec, bddl_path)

        asset_files = [
            _file(
                "assets/hybrid1_asset_manifest.json",
                assets_root / "assets" / "hybrid1_asset_manifest.json",
                "source_manifest",
            ),
        ]
        for category in (
            "hybrid1_fruit",
            "hybrid1_toolbox",
            "hybrid1_plate",
            "hybrid1_bottle",
            "hybrid1_toy_car",
        ):
            asset_files.extend(
                _files_under(
                    assets_root / "assets" / "hybrid1_objects" / category,
                    f"assets/hybrid1_objects/{category}",
                )
            )
        asset_files.extend(
            _files_under(assets_root / "assets" / "arenas", "assets/arenas")
        )
        asset_files.extend(
            _files_under(assets_root / "assets" / "textures", "assets/textures")
        )
        converted = assets_root / "converted_scene"
        for name in ("table_2", "plate_0", "bottle_4", "toy_car_8"):
            asset_files.append(
                _file(
                    f"converted_scene/meshes/{name}.obj",
                    converted / "meshes" / f"{name}.obj",
                    "mujoco_asset",
                )
            )
            asset_files.append(
                _file(
                    f"converted_scene/textures/{name}.png",
                    converted / "textures" / f"{name}.png",
                    "mujoco_asset",
                )
            )
        asset_files.extend(
            _files_under(
                converted / "meshes" / "collision",
                "converted_scene/meshes/collision",
            )
        )
        asset_files.extend(
            (
                _file(
                    "converted_scene/scene_1.xml",
                    converted / "scene_1.xml",
                    "conversion_output",
                ),
                _file(
                    "converted_scene/manifest.txt",
                    converted / "manifest.txt",
                    "conversion_manifest",
                ),
            )
        )

        envs_root = Path(__file__).resolve().parents[1]
        plugin_root = Path(__file__).resolve().parent
        implementation_files = tuple(
            _file(logical_path, path, role)
            for logical_path, path, role in (
                (
                    "lerobot/envs/hybrid1_specs.py",
                    envs_root / "hybrid1_specs.py",
                    "implementation",
                ),
                (
                    "lerobot/envs/hybrid1_libero.py",
                    envs_root / "hybrid1_libero.py",
                    "implementation",
                ),
                (
                    "lerobot/envs/libero_scenes/contracts.py",
                    plugin_root / "contracts.py",
                    "implementation",
                ),
                (
                    "lerobot/envs/libero_scenes/registry.py",
                    plugin_root / "registry.py",
                    "implementation",
                ),
                (
                    "lerobot/envs/libero_scenes/hybrid1.py",
                    Path(__file__).resolve(),
                    "implementation",
                ),
                (
                    "source/hybrid_1.glb",
                    data_root.parent / "hybrid_1.glb",
                    "source_scene",
                ),
            )
        )
        success = asdict(spec.semantics)
        success_kind = success.pop("kind")
        return ResolvedLiberoSceneTask(
            scene_id=HYBRID1_SCENE_ID,
            scene_version=HYBRID1_SCENE_VERSION,
            task_id=spec.task_key,
            variant_id=spec.variant_key,
            instruction=spec.prompt,
            semantics_id=HYBRID1_SEMANTICS_ID,
            semantics_version=HYBRID1_SEMANTICS_VERSION,
            success=ResolvedSuccessContract(success_kind, 1, success),
            bddl=_file(f"bddl/{bddl_path.name}", bddl_path, "task_definition"),
            assets_root=assets_root,
            asset_files=tuple(asset_files),
            implementation_files=implementation_files,
            env_type="hybrid1_libero",
            task_metadata={
                "source_scene": "hybrid_1.glb",
                "target_object": spec.target_object,
                "receptacle_object": spec.semantics.receptacle_object,
                "object_names": sorted(inventory.object_names),
            },
            runtime_options={"layout_id": "original", "settle_steps": spec.settle_steps},
        )


HYBRID1_LIBERO_SCENE_PLUGIN = Hybrid1LiberoScenePlugin()

__all__ = ["HYBRID1_LIBERO_SCENE_PLUGIN", "Hybrid1LiberoScenePlugin"]
