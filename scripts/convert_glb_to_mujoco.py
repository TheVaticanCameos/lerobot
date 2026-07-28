#!/usr/bin/env python

"""Convert a GLB scene into MuJoCo-compatible OBJ meshes plus MJCF.

The generated MJCF uses the exported OBJ meshes for visuals and simple box
or convex-decomposed geoms for collision. This keeps large scanned scenes
loadable in MuJoCo while preserving the visual layout of the original GLB.
"""

from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from contextlib import suppress
from pathlib import Path

import numpy as np
import trimesh


def _sanitize_name(name: str, fallback: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_").lower()
    return clean or fallback


def _rgba_from_mesh(mesh: trimesh.Trimesh) -> tuple[float, float, float, float]:
    material = getattr(getattr(mesh, "visual", None), "material", None)
    color = getattr(material, "baseColorFactor", None)
    if color is None:
        color = getattr(material, "main_color", None)
    if color is None:
        return (0.7, 0.7, 0.7, 1.0)
    rgba = np.asarray(color, dtype=float).reshape(-1)
    if rgba.max(initial=1.0) > 1.0:
        rgba = rgba / 255.0
    if rgba.size == 3:
        rgba = np.concatenate([rgba, [1.0]])
    return tuple(float(np.clip(v, 0.0, 1.0)) for v in rgba[:4])


def _texture_from_mesh(mesh: trimesh.Trimesh):
    material = getattr(getattr(mesh, "visual", None), "material", None)
    if material is None:
        return None
    return getattr(material, "baseColorTexture", None) or getattr(material, "image", None)


def _write_obj(path: Path, mesh: trimesh.Trimesh) -> bool:
    uv = getattr(getattr(mesh, "visual", None), "uv", None)
    has_uv = uv is not None and len(uv) == len(mesh.vertices)

    with path.open("w", encoding="utf-8") as f:
        f.write(f"# Exported from {path.name}\n")
        for vertex in mesh.vertices:
            f.write(f"v {vertex[0]:.9g} {vertex[1]:.9g} {vertex[2]:.9g}\n")
        if has_uv:
            for texcoord in uv:
                f.write(f"vt {texcoord[0]:.9g} {texcoord[1]:.9g}\n")
        for face in mesh.faces:
            # OBJ indices are 1-based.
            if has_uv:
                f.write(
                    f"f {face[0] + 1}/{face[0] + 1} "
                    f"{face[1] + 1}/{face[1] + 1} "
                    f"{face[2] + 1}/{face[2] + 1}\n"
                )
            else:
                f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")
    return bool(has_uv)


def _write_plain_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# Exported from {path.name}\n")
        for vertex in vertices:
            f.write(f"v {vertex[0]:.9g} {vertex[1]:.9g} {vertex[2]:.9g}\n")
        for face in faces:
            f.write(f"f {int(face[0]) + 1} {int(face[1]) + 1} {int(face[2]) + 1}\n")


def _apply_y_up_to_z_up(mesh: trimesh.Trimesh) -> None:
    transform = np.eye(4)
    transform[:3, :3] = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    mesh.apply_transform(transform)


def _format_vec(values: np.ndarray | tuple[float, ...]) -> str:
    return " ".join(f"{float(v):.9g}" for v in values)


def _repair_mesh_topology(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    repaired = mesh.copy()
    repaired.remove_unreferenced_vertices()
    repaired.merge_vertices()
    repaired.update_faces(repaired.unique_faces())
    repaired.update_faces(repaired.nondegenerate_faces())
    repaired.remove_unreferenced_vertices()
    with suppress(Exception):
        trimesh.repair.fill_holes(repaired)
    try:
        trimesh.repair.fix_normals(repaired, multibody=True)
    except TypeError:
        trimesh.repair.fix_normals(repaired)
    repaired.remove_unreferenced_vertices()
    return repaired


def _simplify_for_collision(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    mesh = _repair_mesh_topology(mesh)
    if target_faces <= 0 or len(mesh.faces) <= target_faces:
        return mesh.copy()
    simplified = mesh.simplify_quadric_decimation(face_count=target_faces)
    return _repair_mesh_topology(simplified)


def _prepare_convex_part(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    part = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    part = _repair_mesh_topology(part)
    if not part.is_watertight:
        part = _repair_mesh_topology(part.convex_hull)
    return np.asarray(part.vertices, dtype=float), np.asarray(part.faces, dtype=np.int32)


def _coacd_parts(
    mesh: trimesh.Trimesh,
    *,
    threshold: float,
    max_convex_hull: int,
    preprocess_resolution: int,
    resolution: int,
    mcts_nodes: int,
    mcts_iterations: int,
    max_ch_vertex: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    try:
        import coacd
    except ImportError as exc:
        raise RuntimeError("collision_mode='convex' requires `pip install coacd`.") from exc

    with suppress(Exception):
        coacd.set_log_level("error")

    coacd_mesh = coacd.Mesh(mesh.vertices.astype(np.float64), mesh.faces.astype(np.int32))
    parts = coacd.run_coacd(
        coacd_mesh,
        threshold=threshold,
        max_convex_hull=max_convex_hull,
        preprocess_resolution=preprocess_resolution,
        resolution=resolution,
        mcts_nodes=mcts_nodes,
        mcts_iterations=mcts_iterations,
        max_ch_vertex=max_ch_vertex,
        merge=True,
        seed=seed,
    )
    return [_prepare_convex_part(vertices, faces) for vertices, faces in parts]


def _is_convex_skip(name: str, pattern: str) -> bool:
    return bool(pattern and re.search(pattern, name))


def convert_glb_to_mujoco(
    input_path: Path,
    output_dir: Path,
    *,
    y_up_to_z_up: bool = True,
    collision_mode: str = "box",
    convex_skip_name_regex: str = r"^table",
    collision_decimate_faces: int = 5000,
    coacd_threshold: float = 0.08,
    coacd_max_convex_hull: int = 16,
    coacd_preprocess_resolution: int = 50,
    coacd_resolution: int = 2000,
    coacd_mcts_nodes: int = 10,
    coacd_mcts_iterations: int = 80,
    coacd_max_ch_vertex: int = 128,
    coacd_seed: int = 0,
) -> Path:
    scene = trimesh.load(input_path, force="scene")
    if not isinstance(scene, trimesh.Scene):
        raise TypeError(f"Expected a trimesh.Scene from {input_path}, got {type(scene).__name__}")

    if collision_mode not in {"box", "convex"}:
        raise ValueError(f"Unsupported collision_mode={collision_mode!r}; expected 'box' or 'convex'.")

    meshes_dir = output_dir / "meshes"
    collision_meshes_dir = meshes_dir / "collision"
    textures_dir = output_dir / "textures"
    meshes_dir.mkdir(parents=True, exist_ok=True)
    collision_meshes_dir.mkdir(parents=True, exist_ok=True)
    textures_dir.mkdir(parents=True, exist_ok=True)

    mujoco = ET.Element("mujoco", {"model": input_path.stem})
    ET.SubElement(mujoco, "compiler", {"angle": "radian", "meshdir": "meshes", "texturedir": "textures"})
    ET.SubElement(mujoco, "option", {"gravity": "0 0 -9.81"})

    asset = ET.SubElement(mujoco, "asset")
    worldbody = ET.SubElement(mujoco, "worldbody")
    ET.SubElement(worldbody, "light", {"name": "key", "pos": "0 -3 4", "dir": "0 1 -1", "diffuse": "0.8 0.8 0.8"})
    ET.SubElement(worldbody, "light", {"name": "fill", "pos": "2 2 3", "dir": "-1 -1 -1", "diffuse": "0.4 0.4 0.4"})
    ET.SubElement(worldbody, "camera", {"name": "overview", "pos": "1.4 -1.8 1.2", "xyaxes": "0.8 0.6 0 -0.3 0.4 0.866"})

    used_names: set[str] = set()
    exported = []

    for index, node_name in enumerate(scene.graph.nodes_geometry):
        transform, geometry_name = scene.graph[node_name]
        mesh = scene.geometry[geometry_name].copy()
        if not isinstance(mesh, trimesh.Trimesh):
            continue
        mesh.apply_transform(transform)
        if y_up_to_z_up:
            _apply_y_up_to_z_up(mesh)
        mesh.remove_unreferenced_vertices()

        base_name = _sanitize_name(str(geometry_name), f"mesh_{index}")
        name = base_name
        suffix = 1
        while name in used_names:
            suffix += 1
            name = f"{base_name}_{suffix}"
        used_names.add(name)

        obj_file = f"{name}.obj"
        has_uv = _write_obj(meshes_dir / obj_file, mesh)

        texture_name = None
        texture_image = _texture_from_mesh(mesh)
        if texture_image is not None and has_uv:
            texture_name = f"tex_{name}"
            texture_file = f"{name}.png"
            texture_image.save(textures_dir / texture_file)
            ET.SubElement(asset, "texture", {"name": texture_name, "type": "2d", "file": texture_file})

        material_name = f"mat_{name}"
        ET.SubElement(asset, "mesh", {"name": name, "file": obj_file})
        material_attrs = {"name": material_name, "rgba": "1 1 1 1" if texture_name else _format_vec(_rgba_from_mesh(mesh))}
        if texture_name:
            material_attrs["texture"] = texture_name
        ET.SubElement(asset, "material", material_attrs)

        bounds = np.asarray(mesh.bounds, dtype=float)
        center = bounds.mean(axis=0)
        half_size = np.maximum((bounds[1] - bounds[0]) / 2.0, 1e-4)

        collision_parts: list[str] = []
        used_collision_mode = "box"
        if collision_mode == "convex" and not _is_convex_skip(name, convex_skip_name_regex):
            collision_mesh = _simplify_for_collision(mesh, collision_decimate_faces)
            parts = _coacd_parts(
                collision_mesh,
                threshold=coacd_threshold,
                max_convex_hull=coacd_max_convex_hull,
                preprocess_resolution=coacd_preprocess_resolution,
                resolution=coacd_resolution,
                mcts_nodes=coacd_mcts_nodes,
                mcts_iterations=coacd_mcts_iterations,
                max_ch_vertex=coacd_max_ch_vertex,
                seed=coacd_seed,
            )
            used_collision_mode = "convex"
            for part_index, (vertices, faces) in enumerate(parts):
                col_name = f"{name}_col_{part_index:03d}"
                col_file = f"collision/{col_name}.obj"
                _write_plain_obj(collision_meshes_dir / f"{col_name}.obj", vertices, faces)
                ET.SubElement(asset, "mesh", {"name": col_name, "file": col_file})
                collision_parts.append(col_name)

        body = ET.SubElement(worldbody, "body", {"name": name, "pos": "0 0 0"})
        ET.SubElement(
            body,
            "geom",
            {
                "name": f"{name}_visual",
                "type": "mesh",
                "mesh": name,
                "material": material_name,
                "contype": "0",
                "conaffinity": "0",
                "group": "2",
            },
        )
        if collision_parts:
            for part_index, col_name in enumerate(collision_parts):
                ET.SubElement(
                    body,
                    "geom",
                    {
                        "name": f"{name}_collision_{part_index:03d}",
                        "type": "mesh",
                        "mesh": col_name,
                        "rgba": "0 0 0 0",
                        "group": "3",
                        "density": "100",
                        "friction": "0.95 0.3 0.1",
                    },
                )
        else:
            ET.SubElement(
                body,
                "geom",
                {
                    "name": f"{name}_collision",
                    "type": "box",
                    "pos": _format_vec(center),
                    "size": _format_vec(half_size),
                    "rgba": "0 0 0 0",
                    "group": "3",
                    "density": "100",
                    "friction": "0.95 0.3 0.1",
                },
            )

        exported.append(
            (
                name,
                len(mesh.vertices),
                len(mesh.faces),
                bounds,
                texture_name is not None,
                has_uv,
                used_collision_mode,
                len(collision_parts) if collision_parts else 1,
            )
        )

    if not exported:
        raise RuntimeError(f"No mesh geometry was exported from {input_path}")

    ET.indent(mujoco, space="  ")
    xml_path = output_dir / "scene_1.xml"
    ET.ElementTree(mujoco).write(xml_path, encoding="utf-8", xml_declaration=True)

    manifest = output_dir / "manifest.txt"
    with manifest.open("w", encoding="utf-8") as f:
        f.write(f"source={input_path}\n")
        f.write(f"xml={xml_path}\n")
        f.write(f"y_up_to_z_up={y_up_to_z_up}\n")
        f.write(f"collision_mode={collision_mode}\n")
        f.write(f"convex_skip_name_regex={convex_skip_name_regex}\n")
        for name, n_vertices, n_faces, bounds, has_texture, has_uv, used_collision_mode, n_collision_parts in exported:
            f.write(
                f"{name}: vertices={n_vertices} faces={n_faces} "
                f"uv={has_uv} texture={has_texture} "
                f"collision={used_collision_mode} collision_parts={n_collision_parts} "
                f"bounds_min={_format_vec(bounds[0])} bounds_max={_format_vec(bounds[1])}\n"
            )

    return xml_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Path to the source .glb file")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for MJCF and OBJ files")
    parser.add_argument("--keep-y-up", action="store_true", help="Do not rotate GLB Y-up coordinates to MuJoCo Z-up")
    parser.add_argument("--collision-mode", choices=["box", "convex"], default="convex")
    # parser.add_argument("--convex-skip-name-regex", default=r"^table", help="Object names matching this regex keep box collision")
    parser.add_argument("--convex-skip-name-regex", default="", help="Object names matching this regex keep box collision")
    parser.add_argument("--collision-decimate-faces", type=int, default=20000)
    parser.add_argument("--coacd-threshold", type=float, default=0.04)
    parser.add_argument("--coacd-max-convex-hull", type=int, default=16)
    parser.add_argument("--coacd-preprocess-resolution", type=int, default=70)
    parser.add_argument("--coacd-resolution", type=int, default=3000)
    parser.add_argument("--coacd-mcts-nodes", type=int, default=15)
    parser.add_argument("--coacd-mcts-iterations", type=int, default=120)
    parser.add_argument("--coacd-max-ch-vertex", type=int, default=192)
    parser.add_argument("--coacd-seed", type=int, default=0)
    args = parser.parse_args()

    xml_path = convert_glb_to_mujoco(
        args.input,
        args.output_dir,
        y_up_to_z_up=not args.keep_y_up,
        collision_mode=args.collision_mode,
        convex_skip_name_regex=args.convex_skip_name_regex,
        collision_decimate_faces=args.collision_decimate_faces,
        coacd_threshold=args.coacd_threshold,
        coacd_max_convex_hull=args.coacd_max_convex_hull,
        coacd_preprocess_resolution=args.coacd_preprocess_resolution,
        coacd_resolution=args.coacd_resolution,
        coacd_mcts_nodes=args.coacd_mcts_nodes,
        coacd_mcts_iterations=args.coacd_mcts_iterations,
        coacd_max_ch_vertex=args.coacd_max_ch_vertex,
        coacd_seed=args.coacd_seed,
    )
    print(xml_path)


if __name__ == "__main__":
    main()
