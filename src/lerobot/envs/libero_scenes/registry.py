"""Explicit registry for LIBERO scene resolver plugins."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from types import MappingProxyType

from .contracts import LiberoScenePlugin, LiberoSceneTaskRequest, ResolvedLiberoSceneTask


class LiberoScenePluginRegistry:
    __slots__ = ("_plugins",)

    def __init__(self) -> None:
        self._plugins: dict[str, LiberoScenePlugin] = {}

    @property
    def plugins(self) -> Mapping[str, LiberoScenePlugin]:
        return MappingProxyType(self._plugins)

    @property
    def scene_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._plugins))

    def register(self, plugin: LiberoScenePlugin) -> None:
        if not isinstance(plugin, LiberoScenePlugin):
            raise TypeError("plugin must implement LiberoScenePlugin.")
        scene_id = plugin.descriptor.scene_id
        if scene_id in self._plugins:
            raise ValueError(f"LIBERO scene plugin {scene_id!r} is already registered.")
        self._plugins[scene_id] = plugin

    def get(self, scene_id: str) -> LiberoScenePlugin:
        try:
            return self._plugins[scene_id]
        except KeyError as error:
            available = ", ".join(self.scene_ids) or "<none>"
            raise ValueError(
                f"Unknown LIBERO scene plugin {scene_id!r}. Available scenes: {available}"
            ) from error

    def resolve(self, request: LiberoSceneTaskRequest) -> ResolvedLiberoSceneTask:
        plugin = self.get(request.scene_id)
        if request.task_id not in plugin.descriptor.task_ids:
            available = ", ".join(plugin.descriptor.task_ids)
            raise ValueError(
                f"Unknown task {request.task_id!r} for LIBERO scene {request.scene_id!r}. "
                f"Available tasks: {available}"
            )
        resolved = plugin.resolve(request)
        if (
            resolved.scene_id != plugin.descriptor.scene_id
            or resolved.scene_version != plugin.descriptor.scene_version
            or resolved.task_id != request.task_id
        ):
            raise RuntimeError("LIBERO scene plugin returned an inconsistent resolved identity.")
        return resolved

    def __iter__(self) -> Iterator[LiberoScenePlugin]:
        return iter(self._plugins[scene_id] for scene_id in self.scene_ids)


__all__ = ["LiberoScenePluginRegistry"]
