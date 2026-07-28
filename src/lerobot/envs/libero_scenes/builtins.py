"""Composition root for built-in catalog-backed LIBERO scenes."""

from __future__ import annotations

from .registry import LiberoScenePluginRegistry


def build_builtin_libero_scene_registry() -> LiberoScenePluginRegistry:
    from .hybrid1 import HYBRID1_LIBERO_SCENE_PLUGIN
    from .scene1 import SCENE1_LIBERO_SCENE_PLUGIN

    registry = LiberoScenePluginRegistry()
    registry.register(HYBRID1_LIBERO_SCENE_PLUGIN)
    registry.register(SCENE1_LIBERO_SCENE_PLUGIN)
    return registry


__all__ = ["build_builtin_libero_scene_registry"]
