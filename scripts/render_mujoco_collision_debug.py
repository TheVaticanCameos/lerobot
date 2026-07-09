#!/usr/bin/env python

"""Render MuJoCo collision boxes for a converted scene."""

from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image
import mujoco


COLORS = [
    "1 0.1 0.1 0.45",
    "0.1 0.8 1 0.45",
    "0.1 1 0.2 0.45",
    "1 0.8 0.1 0.45",
    "0.9 0.1 1 0.45",
    "1 0.45 0.05 0.45",
    "0.1 0.2 1 0.45",
    "0.2 1 0.8 0.45",
]


def make_debug_xml(
    source_xml: Path,
    output_xml: Path,
    *,
    collision_only: bool = False,
    height: int = 480,
    width: int = 640,
    hide_collision_name_regex: str = "",
) -> None:
    tree = ET.parse(source_xml)
    root = tree.getroot()

    compiler = root.find("compiler")
    if compiler is not None and output_xml.parent.resolve() != source_xml.parent.resolve():
        for attr in ("meshdir", "texturedir"):
            value = compiler.attrib.get(attr)
            if value and not Path(value).is_absolute():
                compiler.set(attr, str((source_xml.parent / value).resolve()))

    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    global_visual = visual.find("global")
    if global_visual is None:
        global_visual = ET.SubElement(visual, "global")
    global_visual.set("offheight", str(height))
    global_visual.set("offwidth", str(width))

    material_by_name = {mat.attrib["name"]: mat for mat in root.findall("./asset/material")}

    collision_index = 0
    for geom in root.findall(".//geom"):
        name = geom.attrib.get("name", "")
        if name.endswith("_visual"):
            if collision_only:
                geom.set("rgba", "1 1 1 0")
                geom.set("contype", "0")
                geom.set("conaffinity", "0")
                geom.set("group", "5")
            else:
                mat_name = geom.attrib.get("material")
                if mat_name in material_by_name:
                    material_by_name[mat_name].set("rgba", "1 1 1 0.22")
                else:
                    geom.set("rgba", "1 1 1 0.22")
        elif "_collision" in name:
            if hide_collision_name_regex and re.search(hide_collision_name_regex, name):
                geom.set("rgba", "0 0 0 0")
                geom.set("contype", "0")
                geom.set("conaffinity", "0")
                geom.set("group", "5")
                continue
            geom.set("rgba", COLORS[collision_index % len(COLORS)])
            geom.set("group", "1")
            collision_index += 1

    ET.indent(root, space="  ")
    tree.write(output_xml, encoding="utf-8", xml_declaration=True)


def render_xml(xml_path: Path, output_png: Path, *, height: int, width: int) -> None:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=height, width=width)
    renderer.update_scene(data, camera="overview")
    image = renderer.render()
    renderer.close()
    Image.fromarray(image).save(output_png)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xml", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--hide-collision-name-regex", default="")
    args = parser.parse_args()

    output_dir = args.output_dir or args.xml.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    overlay_xml = output_dir / f"{args.xml.stem}_collision_overlay.xml"
    collision_xml = output_dir / f"{args.xml.stem}_collision_only.xml"
    overlay_png = output_dir / f"{args.xml.stem}_collision_overlay.png"
    collision_png = output_dir / f"{args.xml.stem}_collision_only.png"

    make_debug_xml(
        args.xml,
        overlay_xml,
        collision_only=False,
        height=args.height,
        width=args.width,
        hide_collision_name_regex=args.hide_collision_name_regex,
    )
    make_debug_xml(
        args.xml,
        collision_xml,
        collision_only=True,
        height=args.height,
        width=args.width,
        hide_collision_name_regex=args.hide_collision_name_regex,
    )
    render_xml(overlay_xml, overlay_png, height=args.height, width=args.width)
    render_xml(collision_xml, collision_png, height=args.height, width=args.width)

    print(overlay_xml)
    print(overlay_png)
    print(collision_xml)
    print(collision_png)


if __name__ == "__main__":
    main()
