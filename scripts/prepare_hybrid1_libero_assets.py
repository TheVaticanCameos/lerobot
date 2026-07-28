#!/usr/bin/env python

"""Prepare Hybrid1 LIBERO runtime assets from the converted GLB geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
from convert_glb_to_mujoco import (
    _apply_y_up_to_z_up,
    _texture_from_mesh,
    _write_obj,
    _write_plain_obj,
)

_MOVABLE_GEOMETRIES = {
    "hybrid1_fruit": ("fruit_5", 120),
    "hybrid1_toolbox": ("toolbox_1", 150),
    "hybrid1_plate": ("plate_0", 120),
    "hybrid1_bottle": ("bottle_4", 120),
    "hybrid1_toy_car": ("toy car_8", 100),
}
_STATIC_GEOMETRIES = {
    "hybrid1_glb_table": "table_2",
}
_ARENA_GEOMETRIES = {
    "hybrid1_glb_table": "table",
    "hybrid1_glb_plate": "plate",
    "hybrid1_glb_bottle": "bottle",
    "hybrid1_glb_toy_car": "toy_car",
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _world_mesh(scene: trimesh.Scene, geometry_name: str) -> trimesh.Trimesh:
    matches = [
        node_name
        for node_name in scene.graph.nodes_geometry
        if scene.graph[node_name][1] == geometry_name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one node for geometry {geometry_name!r}, found {matches}."
        )
    transform, _ = scene.graph[matches[0]]
    mesh = scene.geometry[geometry_name].copy()
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Geometry {geometry_name!r} is not a triangular mesh.")
    mesh.apply_transform(transform)
    _apply_y_up_to_z_up(mesh)
    mesh.remove_unreferenced_vertices()
    return mesh


def _collision_parts(collision_root: Path, geometry_name: str) -> list[Path]:
    collision_name = geometry_name.replace(" ", "_")
    paths = sorted(collision_root.glob(f"{collision_name}_col_*.obj"))
    if not paths:
        raise FileNotFoundError(
            f"No convex collision parts found for {geometry_name!r} in {collision_root}."
        )
    expected_names = [
        f"{collision_name}_col_{index:03d}.obj" for index in range(len(paths))
    ]
    if [path.name for path in paths] != expected_names:
        raise ValueError(f"Collision parts for {geometry_name!r} are not contiguous.")
    return paths


def _load_collision_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Collision part {path} is not a triangular mesh.")
    mesh.remove_unreferenced_vertices()
    if not mesh.is_convex:
        raise ValueError(f"Collision part {path} is not convex.")
    return mesh


def _write_xml(root: ET.Element, path: Path) -> None:
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=False)


def _ensure_object_xml(
    object_dir: Path,
    object_category: str,
    local_bounds: np.ndarray,
    density: int,
) -> None:
    xml_path = object_dir / f"{object_category}.xml"
    if xml_path.is_file():
        return
    root = ET.Element("mujoco", {"model": object_category})
    asset = ET.SubElement(root, "asset")
    texture_name = f"tex_{object_category}"
    material_name = f"mat_{object_category}"
    visual_name = f"{object_category}_visual"
    ET.SubElement(
        asset,
        "texture",
        {"file": "texture.png", "name": texture_name, "type": "2d"},
    )
    ET.SubElement(
        asset,
        "material",
        {
            "name": material_name,
            "texture": texture_name,
            "texuniform": "false",
            "rgba": "1 1 1 1",
        },
    )
    ET.SubElement(
        asset,
        "mesh",
        {"file": f"{visual_name}.obj", "name": visual_name},
    )
    worldbody = ET.SubElement(root, "worldbody")
    wrapper = ET.SubElement(worldbody, "body")
    object_body = ET.SubElement(wrapper, "body", {"name": "object", "pos": "0 0 0"})
    ET.SubElement(
        object_body,
        "geom",
        {
            "type": "mesh",
            "mesh": visual_name,
            "material": material_name,
            "contype": "0",
            "conaffinity": "0",
            "group": "1",
            "density": str(density),
            "friction": "0.95 0.3 0.1",
        },
    )
    height = float(local_bounds[1, 2])
    center_z = height / 2.0
    horizontal_radius = float(np.max(np.abs(local_bounds[:, :2])))
    for name, pos in (
        ("bottom_site", (0.0, 0.0, 0.0)),
        ("top_site", (0.0, 0.0, height)),
        ("horizontal_radius_site", (horizontal_radius, 0.0, center_z)),
        ("center_site", (0.0, 0.0, center_z)),
    ):
        ET.SubElement(
            object_body,
            "site",
            {
                "name": name,
                "pos": " ".join(f"{value:.9g}" for value in pos),
                "size": "0.005",
                "rgba": "0 0 0 0",
            },
        )
    _write_xml(root, xml_path)


def _rewrite_object_xml(
    object_dir: Path,
    object_category: str,
    collision_paths: list[Path],
    density: int,
) -> None:
    xml_path = object_dir / f"{object_category}.xml"
    root = ET.parse(xml_path).getroot()
    asset = root.find("asset")
    object_body = root.find("./worldbody/body/body[@name='object']")
    if asset is None or object_body is None:
        raise ValueError(f"Hybrid1 object XML has an unexpected structure: {xml_path}")

    collision_prefix = f"{object_category}_collision_"
    for mesh_element in list(asset.findall("mesh")):
        if mesh_element.get("name", "").startswith(collision_prefix):
            asset.remove(mesh_element)
    for geom in list(object_body.findall("geom")):
        if geom.get("group") == "0":
            object_body.remove(geom)

    visual_geom = object_body.find("geom[@group='1']")
    if visual_geom is None:
        raise ValueError(f"Hybrid1 object XML has no visual geom: {xml_path}")
    insertion_index = list(object_body).index(visual_geom) + 1
    for index, collision_path in enumerate(collision_paths):
        mesh_name = f"{object_category}_collision_{index:03d}"
        ET.SubElement(
            asset,
            "mesh",
            {
                "file": f"collision/{collision_path.name}",
                "name": mesh_name,
            },
        )
        object_body.insert(
            insertion_index + index,
            ET.Element(
                "geom",
                {
                    "type": "mesh",
                    "mesh": mesh_name,
                    "group": "0",
                    "rgba": "0.8 0.8 0.8 0.25",
                    "density": str(density),
                    "friction": "0.95 0.3 0.1",
                    "solimp": "0.998 0.998 0.001",
                    "solref": "0.001 1",
                },
            ),
        )
    _write_xml(root, xml_path)


def _rewrite_arena_xml(output_root: Path, collision_root: Path) -> list[dict]:
    xml_path = output_root / "arenas" / "hybrid1_tabletop.xml"
    root = ET.parse(xml_path).getroot()
    asset = root.find("asset")
    worldbody = root.find("worldbody")
    if asset is None or worldbody is None:
        raise ValueError(f"Hybrid1 arena XML has an unexpected structure: {xml_path}")

    for mesh_element in list(asset.findall("mesh")):
        if mesh_element.get("name", "").startswith("hybrid1_static_collision_"):
            asset.remove(mesh_element)

    static_body_names = set(_STATIC_GEOMETRIES)
    for body_name, asset_suffix in _ARENA_GEOMETRIES.items():
        if body_name in static_body_names:
            continue
        body = worldbody.find(f"./body[@name='{body_name}']")
        if body is not None:
            worldbody.remove(body)
        removable_names = {
            f"tex_hybrid1_{asset_suffix}",
            f"mat_hybrid1_{asset_suffix}",
            f"hybrid1_{asset_suffix}_mesh",
        }
        for element in list(asset):
            if element.get("name") in removable_names:
                asset.remove(element)

    table_placeholder = worldbody.find("./body[@name='table']/geom[@name='table_collision']")
    if table_placeholder is None:
        raise ValueError("Hybrid1 arena is missing the TableArena collision placeholder.")
    table_placeholder.set("contype", "0")
    table_placeholder.set("conaffinity", "0")
    table_placeholder.set("group", "1")
    table_placeholder.set("rgba", "0 0 0 0")

    records = []
    for body_name, geometry_name in _STATIC_GEOMETRIES.items():
        body = worldbody.find(f"./body[@name='{body_name}']")
        if body is None:
            raise ValueError(f"Hybrid1 arena is missing body {body_name!r}.")
        for geom in list(body.findall("geom")):
            if geom.get("group") == "0":
                body.remove(geom)

        part_records = []
        for index, collision_path in enumerate(
            _collision_parts(collision_root, geometry_name)
        ):
            mesh_name = f"hybrid1_static_collision_{geometry_name}_{index:03d}"
            ET.SubElement(
                asset,
                "mesh",
                {
                    "name": mesh_name,
                    "file": (
                        "../../converted_scene/meshes/collision/"
                        f"{collision_path.name}"
                    ),
                },
            )
            ET.SubElement(
                body,
                "geom",
                {
                    "name": f"{body_name}_collision_{index:03d}",
                    "type": "mesh",
                    "mesh": mesh_name,
                    "group": "0",
                    "rgba": "0 0 0 0",
                    "friction": "0.95 0.3 0.1",
                    "solimp": "0.998 0.998 0.001",
                    "solref": "0.001 1",
                },
            )
            mesh = _load_collision_mesh(collision_path)
            part_records.append(
                {
                    "mesh": (
                        "converted_scene/meshes/collision/"
                        f"{collision_path.name}"
                    ),
                    "mesh_sha256": _file_sha256(collision_path),
                    "world_bounds": np.asarray(mesh.bounds, dtype=float).tolist(),
                }
            )
        records.append(
            {
                "arena_body": body_name,
                "source_geometry": geometry_name,
                "dynamic": False,
                "joint_type": None,
                "infrastructure_role": "table_fixture",
                "collision_meshes": part_records,
            }
        )
    _write_xml(root, xml_path)
    return records


def prepare_hybrid1_libero_assets(input_path: Path, output_root: Path) -> Path:
    input_path = input_path.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    scene = trimesh.load(input_path, force="scene")
    if not isinstance(scene, trimesh.Scene):
        raise TypeError(f"Expected a trimesh.Scene, got {type(scene).__name__}.")

    conversion_root = output_root.parent / "converted_scene"
    conversion_manifest = conversion_root / "manifest.txt"
    if "collision_mode=convex" not in conversion_manifest.read_text(encoding="utf-8"):
        raise ValueError("Hybrid1 conversion manifest must declare collision_mode=convex.")
    collision_root = conversion_root / "meshes" / "collision"

    records = []
    for object_category, (geometry_name, density) in _MOVABLE_GEOMETRIES.items():
        mesh = _world_mesh(scene, geometry_name)
        world_bounds = np.asarray(mesh.bounds, dtype=float)
        origin = np.array(
            [
                float(world_bounds[:, 0].mean()),
                float(world_bounds[:, 1].mean()),
                float(world_bounds[0, 2]),
            ]
        )
        mesh.apply_translation(-origin)

        object_dir = output_root / "hybrid1_objects" / object_category
        object_dir.mkdir(parents=True, exist_ok=True)
        visual_path = object_dir / f"{object_category}_visual.obj"
        if not _write_obj(visual_path, mesh):
            raise ValueError(f"Geometry {geometry_name!r} has no usable UV coordinates.")
        texture = _texture_from_mesh(mesh)
        if texture is None:
            raise ValueError(f"Geometry {geometry_name!r} has no base-color texture.")
        texture_path = object_dir / "texture.png"
        texture.save(texture_path)
        _ensure_object_xml(
            object_dir,
            object_category,
            np.asarray(mesh.bounds, dtype=float),
            density,
        )
        local_collision_dir = object_dir / "collision"
        if local_collision_dir.exists():
            shutil.rmtree(local_collision_dir)
        local_collision_dir.mkdir()
        collision_records = []
        local_collision_paths = []
        for source_collision_path in _collision_parts(collision_root, geometry_name):
            collision_mesh = _load_collision_mesh(source_collision_path)
            collision_mesh.apply_translation(-origin)
            local_collision_path = local_collision_dir / source_collision_path.name
            _write_plain_obj(
                local_collision_path,
                np.asarray(collision_mesh.vertices),
                np.asarray(collision_mesh.faces),
            )
            local_collision_paths.append(local_collision_path)
            collision_records.append(
                {
                    "mesh": str(local_collision_path.relative_to(output_root)),
                    "mesh_sha256": _file_sha256(local_collision_path),
                    "local_bounds": np.asarray(
                        collision_mesh.bounds, dtype=float
                    ).tolist(),
                }
            )
        _rewrite_object_xml(
            object_dir,
            object_category,
            local_collision_paths,
            density,
        )
        records.append(
            {
                "object_category": object_category,
                "source_geometry": geometry_name,
                "source_world_bounds": world_bounds.tolist(),
                "local_origin": origin.tolist(),
                "local_bounds": np.asarray(mesh.bounds, dtype=float).tolist(),
                "visual_mesh": str(visual_path.relative_to(output_root)),
                "visual_mesh_sha256": _file_sha256(visual_path),
                "texture": str(texture_path.relative_to(output_root)),
                "texture_sha256": _file_sha256(texture_path),
                "collision_mode": "coacd_convex",
                "dynamic": True,
                "joint_type": "free",
                "collision_meshes": collision_records,
            }
        )

    static_collision_records = _rewrite_arena_xml(output_root, collision_root)

    source_textures = Path(
        __import__("libero.libero", fromlist=["get_libero_path"]).get_libero_path(
            "assets"
        )
    ) / "textures"
    textures_dir = output_root / "textures"
    textures_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "light-gray-floor-tile.png",
        "light-gray-plaster.png",
        "martin_novak_wood_table.png",
    ):
        shutil.copy2(source_textures / name, textures_dir / name)

    manifest_path = output_root / "hybrid1_asset_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "hybrid1-libero-assets/v3",
                "source_glb_logical_path": "data/hybrid_1.glb",
                "source_glb_sha256": _file_sha256(input_path),
                "coordinate_transform": "glb-y-up-to-mujoco-z-up",
                "collision_mode": "coacd_convex",
                "dynamics_policy": {
                    "static_infrastructure": ["table", "robot_base", "robot_mount"],
                    "dynamic_object_categories": sorted(_MOVABLE_GEOMETRIES),
                },
                "conversion_manifest": "converted_scene/manifest.txt",
                "conversion_manifest_sha256": _file_sha256(conversion_manifest),
                "movable_objects": records,
                "static_geometry": static_collision_records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(prepare_hybrid1_libero_assets(args.input, args.output_root))


if __name__ == "__main__":
    main()
