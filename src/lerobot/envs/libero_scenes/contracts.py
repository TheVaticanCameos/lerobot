"""Simulator-owned contracts for resolving catalog-backed LIBERO scenes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable


def _identifier(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    if value != value.strip() or any(character.isspace() for character in value):
        raise ValueError(f"{field_name} must not contain whitespace: {value!r}.")


def _version(value: int, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer.")


@dataclass(frozen=True, slots=True)
class LiberoSceneDescriptor:
    scene_id: str
    scene_version: int
    task_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _identifier(self.scene_id, "scene_id")
        _version(self.scene_version, "scene_version")
        if not self.task_ids or len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError("task_ids must be non-empty and unique.")
        for task_id in self.task_ids:
            _identifier(task_id, "task_id")


@dataclass(frozen=True, slots=True)
class LiberoSceneTaskRequest:
    scene_id: str
    task_id: str
    variant_id: str | None = None
    bddl_path: Path | None = None
    assets_root: Path | None = None

    def __post_init__(self) -> None:
        _identifier(self.scene_id, "scene_id")
        _identifier(self.task_id, "task_id")
        if self.variant_id is not None:
            _identifier(self.variant_id, "variant_id")


@dataclass(frozen=True, slots=True)
class ResolvedSceneFile:
    """One exact backend input with identity independent from its locator."""

    logical_path: str
    path: Path
    role: str

    def __post_init__(self) -> None:
        if not self.logical_path or self.logical_path.startswith("/"):
            raise ValueError("logical_path must be a non-empty relative path.")
        if ".." in Path(self.logical_path).parts:
            raise ValueError("logical_path must not contain parent traversal.")
        _identifier(self.role, "file role")
        object.__setattr__(self, "path", self.path.expanduser().resolve())


@dataclass(frozen=True, slots=True)
class ResolvedSuccessContract:
    kind: str
    schema_version: int
    parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        _identifier(self.kind, "success kind")
        _version(self.schema_version, "success schema_version")
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "schema_version": self.schema_version,
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True, slots=True)
class ResolvedLiberoSceneTask:
    scene_id: str
    scene_version: int
    task_id: str
    variant_id: str
    instruction: str
    semantics_id: str
    semantics_version: int
    success: ResolvedSuccessContract
    bddl: ResolvedSceneFile
    assets_root: Path
    asset_files: tuple[ResolvedSceneFile, ...]
    implementation_files: tuple[ResolvedSceneFile, ...]
    env_type: str
    task_metadata: Mapping[str, Any]
    runtime_options: Mapping[str, Any]

    def __post_init__(self) -> None:
        for value, name in (
            (self.scene_id, "scene_id"),
            (self.task_id, "task_id"),
            (self.variant_id, "variant_id"),
            (self.semantics_id, "semantics_id"),
            (self.env_type, "env_type"),
        ):
            _identifier(value, name)
        _version(self.scene_version, "scene_version")
        _version(self.semantics_version, "semantics_version")
        if not self.instruction.strip():
            raise ValueError("instruction must be non-empty.")
        object.__setattr__(self, "assets_root", self.assets_root.expanduser().resolve())
        object.__setattr__(self, "task_metadata", MappingProxyType(dict(self.task_metadata)))
        object.__setattr__(self, "runtime_options", MappingProxyType(dict(self.runtime_options)))
        logical_paths = [item.logical_path for item in (self.bddl, *self.asset_files)]
        if len(logical_paths) != len(set(logical_paths)):
            raise ValueError("Resolved BDDL and asset logical paths must be unique.")


@runtime_checkable
class LiberoScenePlugin(Protocol):
    descriptor: LiberoSceneDescriptor

    def resolve(self, request: LiberoSceneTaskRequest) -> ResolvedLiberoSceneTask: ...


__all__ = [
    "LiberoSceneDescriptor",
    "LiberoScenePlugin",
    "LiberoSceneTaskRequest",
    "ResolvedLiberoSceneTask",
    "ResolvedSceneFile",
    "ResolvedSuccessContract",
]
