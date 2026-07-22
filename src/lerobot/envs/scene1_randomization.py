#!/usr/bin/env python

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Scene1RandomizationConfig:
    enabled: bool = False
    profile: str = "nominal"
    table_margin: float = 0.05
    minimum_object_clearance: float = 0.08
    max_pose_sampling_attempts: int = 64

    def __post_init__(self) -> None:
        if self.profile not in SCENE1_RANDOMIZATION_PROFILES:
            available = ", ".join(SCENE1_RANDOMIZATION_PROFILES)
            raise ValueError(f"Unknown Scene1 randomization profile '{self.profile}'. Available: {available}")
        if self.table_margin < 0 or self.minimum_object_clearance < 0:
            raise ValueError("Scene1 randomization margins must be non-negative.")
        if self.max_pose_sampling_attempts < 1:
            raise ValueError("max_pose_sampling_attempts must be positive.")


@dataclass(frozen=True)
class ObjectPoseRanges:
    xy: float
    yaw: float


@dataclass(frozen=True)
class LightingRanges:
    exposure_ev: float
    position: float


@dataclass(frozen=True)
class CameraRanges:
    position: float
    rotation: float
    fov: float


@dataclass(frozen=True)
class Scene1RandomizationProfile:
    object_pose: ObjectPoseRanges | None = None
    lighting: LightingRanges | None = None
    camera: CameraRanges | None = None


@dataclass(frozen=True)
class ObjectPoseSample:
    position_delta: tuple[float, float, float]
    yaw_delta: float
    sampling_attempt: int


@dataclass(frozen=True)
class LightSample:
    name: str
    exposure_ev: float
    exposure_scale: float
    position_delta: tuple[float, float, float]


@dataclass(frozen=True)
class CameraSample:
    name: str
    position_delta: tuple[float, float, float]
    axis_angle_delta: tuple[float, float, float]
    fov_delta: float


@dataclass(frozen=True)
class Scene1RandomizationSample:
    profile: str
    episode_seed: int
    object_pose: ObjectPoseSample | None
    lights: tuple[LightSample, ...]
    camera: CameraSample | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ObjectPoseBaseline:
    qpos_address: int
    position: tuple[float, float, float]
    quaternion: tuple[float, float, float, float]


@dataclass(frozen=True)
class LightBaseline:
    name: str
    light_id: int
    position: tuple[float, float, float]
    diffuse: tuple[float, float, float]
    specular: tuple[float, float, float]


@dataclass(frozen=True)
class CameraBaseline:
    name: str
    camera_id: int
    position: tuple[float, float, float]
    quaternion: tuple[float, float, float, float]
    fov: float


@dataclass(frozen=True)
class Scene1RandomizationBaseline:
    object_pose: ObjectPoseBaseline
    lights: tuple[LightBaseline, ...]
    camera: CameraBaseline


_POSE_XY = (0.02, 0.05, 0.08)
_POSE_YAW_DEGREES = (10.0, 25.0, 45.0)
_LIGHT_EV = (0.5, 1.0, 1.5)
_LIGHT_POSITION = (0.10, 0.25, 0.50)
_CAMERA_POSITION = (0.02, 0.05, 0.10)
_CAMERA_ROTATION_DEGREES = (3.0, 7.0, 12.0)
_CAMERA_FOV_DEGREES = (3.0, 7.0, 12.0)


def _profile(
    *, pose: int | None = None, lighting: int | None = None, camera: int | None = None
) -> Scene1RandomizationProfile:
    object_pose = (
        ObjectPoseRanges(_POSE_XY[pose - 1], math.radians(_POSE_YAW_DEGREES[pose - 1]))
        if pose is not None
        else None
    )
    lighting_ranges = (
        LightingRanges(_LIGHT_EV[lighting - 1], _LIGHT_POSITION[lighting - 1])
        if lighting is not None
        else None
    )
    camera_ranges = (
        CameraRanges(
            _CAMERA_POSITION[camera - 1],
            math.radians(_CAMERA_ROTATION_DEGREES[camera - 1]),
            _CAMERA_FOV_DEGREES[camera - 1],
        )
        if camera is not None
        else None
    )
    return Scene1RandomizationProfile(object_pose, lighting_ranges, camera_ranges)


SCENE1_RANDOMIZATION_PROFILES: Mapping[str, Scene1RandomizationProfile] = {
    "nominal": Scene1RandomizationProfile(),
    **{f"pose_s{severity}": _profile(pose=severity) for severity in range(1, 4)},
    **{f"lighting_s{severity}": _profile(lighting=severity) for severity in range(1, 4)},
    **{f"camera_s{severity}": _profile(camera=severity) for severity in range(1, 4)},
    **{
        f"combined_s{severity}": _profile(pose=severity, lighting=severity, camera=severity)
        for severity in range(1, 4)
    },
}

_FACTOR_IDS = {
    "object_pose": 1,
    "light1_exposure": 2,
    "light1_position": 3,
    "light2_exposure": 4,
    "light2_position": 5,
    "camera_position": 6,
    "camera_rotation": 7,
    "camera_fov": 8,
}


def _rng(episode_seed: int, factor: str) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([episode_seed, _FACTOR_IDS[factor]]))


def _uniform_vector(rng: np.random.Generator, limit: float, size: int) -> tuple[float, ...]:
    return tuple(float(value) for value in rng.uniform(-limit, limit, size=size))


def is_valid_object_candidate(
    candidate_xy: tuple[float, float],
    other_object_xy: Mapping[str, tuple[float, float]],
    *,
    table_half_extents: tuple[float, float] = (0.5, 0.6),
    table_margin: float = 0.05,
    minimum_clearance: float = 0.08,
) -> bool:
    x, y = candidate_xy
    if abs(x) > table_half_extents[0] - table_margin or abs(y) > table_half_extents[1] - table_margin:
        return False
    return all(
        math.hypot(x - other_x, y - other_y) >= minimum_clearance
        for other_x, other_y in other_object_xy.values()
    )


def _sample_object_pose(
    ranges: ObjectPoseRanges,
    episode_seed: int,
    target_xy: tuple[float, float],
    other_object_xy: Mapping[str, tuple[float, float]],
    table_half_extents: tuple[float, float],
    table_margin: float,
    minimum_clearance: float,
    max_attempts: int,
) -> ObjectPoseSample:
    rng = _rng(episode_seed, "object_pose")
    for attempt in range(1, max_attempts + 1):
        delta_x, delta_y = _uniform_vector(rng, ranges.xy, 2)
        yaw = float(rng.uniform(-ranges.yaw, ranges.yaw))
        candidate = (target_xy[0] + delta_x, target_xy[1] + delta_y)
        if is_valid_object_candidate(
            candidate,
            other_object_xy,
            table_half_extents=table_half_extents,
            table_margin=table_margin,
            minimum_clearance=minimum_clearance,
        ):
            return ObjectPoseSample((delta_x, delta_y, 0.0), yaw, attempt)
    raise RuntimeError(f"Unable to sample a valid Scene1 target pose after {max_attempts} attempts.")


def sample_scene1_randomization(
    profile_name: str,
    episode_seed: int,
    *,
    target_xy: tuple[float, float] = (0.0, 0.0),
    other_object_xy: Mapping[str, tuple[float, float]] | None = None,
    table_half_extents: tuple[float, float] = (0.5, 0.6),
    table_margin: float = 0.05,
    minimum_clearance: float = 0.08,
    max_pose_sampling_attempts: int = 64,
) -> Scene1RandomizationSample:
    try:
        profile = SCENE1_RANDOMIZATION_PROFILES[profile_name]
    except KeyError as exc:
        available = ", ".join(SCENE1_RANDOMIZATION_PROFILES)
        raise ValueError(
            f"Unknown Scene1 randomization profile '{profile_name}'. Available: {available}"
        ) from exc

    object_pose = None
    if profile.object_pose is not None:
        object_pose = _sample_object_pose(
            profile.object_pose,
            episode_seed,
            target_xy,
            other_object_xy or {},
            table_half_extents,
            table_margin,
            minimum_clearance,
            max_pose_sampling_attempts,
        )

    lights: tuple[LightSample, ...] = ()
    if profile.lighting is not None:
        sampled_lights = []
        for name in ("light1", "light2"):
            exposure_ev = float(
                _rng(episode_seed, f"{name}_exposure").uniform(
                    -profile.lighting.exposure_ev, profile.lighting.exposure_ev
                )
            )
            sampled_lights.append(
                LightSample(
                    name,
                    exposure_ev,
                    2.0**exposure_ev,
                    _uniform_vector(_rng(episode_seed, f"{name}_position"), profile.lighting.position, 3),
                )
            )
        lights = tuple(sampled_lights)

    camera_sample = None
    if profile.camera is not None:
        rotation_rng = _rng(episode_seed, "camera_rotation")
        axis = rotation_rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = float(rotation_rng.uniform(-profile.camera.rotation, profile.camera.rotation))
        camera_sample = CameraSample(
            "agentview",
            _uniform_vector(_rng(episode_seed, "camera_position"), profile.camera.position, 3),
            tuple(float(value) for value in axis * angle),
            float(_rng(episode_seed, "camera_fov").uniform(-profile.camera.fov, profile.camera.fov)),
        )
    return Scene1RandomizationSample(profile_name, episode_seed, object_pose, lights, camera_sample)


def capture_scene1_randomization_baseline(sim: Any, target_object: str) -> Scene1RandomizationBaseline:
    joint_id = sim.model.joint_name2id(f"{target_object}_joint0")
    address = int(sim.model.jnt_qposadr[joint_id])
    object_baseline = ObjectPoseBaseline(
        address,
        tuple(float(value) for value in sim.data.qpos[address : address + 3]),
        tuple(float(value) for value in sim.data.qpos[address + 3 : address + 7]),
    )

    light_baselines = tuple(
        LightBaseline(
            light_name,
            light_id := int(sim.model.light_name2id(light_name)),
            tuple(float(value) for value in sim.model.light_pos[light_id]),
            tuple(float(value) for value in sim.model.light_diffuse[light_id]),
            tuple(float(value) for value in sim.model.light_specular[light_id]),
        )
        for light_name in ("light1", "light2")
    )

    camera_name = "agentview"
    camera_id = int(sim.model.camera_name2id(camera_name))
    camera_baseline = CameraBaseline(
        camera_name,
        camera_id,
        tuple(float(value) for value in sim.model.cam_pos[camera_id]),
        tuple(float(value) for value in sim.model.cam_quat[camera_id]),
        float(sim.model.cam_fovy[camera_id]),
    )
    return Scene1RandomizationBaseline(object_baseline, light_baselines, camera_baseline)


def _quaternion_multiply(
    left: tuple[float, float, float, float], right: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )


def _axis_angle_quaternion(axis_angle: tuple[float, float, float]) -> tuple[float, float, float, float]:
    angle = math.sqrt(sum(value * value for value in axis_angle))
    if angle == 0.0:
        return (1.0, 0.0, 0.0, 0.0)
    scale = math.sin(angle / 2.0) / angle
    return (math.cos(angle / 2.0), *(value * scale for value in axis_angle))


def apply_scene1_randomization(
    sim: Any, baseline: Scene1RandomizationBaseline, sample: Scene1RandomizationSample
) -> None:
    if sample.object_pose is not None:
        address = baseline.object_pose.qpos_address
        sim.data.qpos[address : address + 3] = np.add(
            baseline.object_pose.position, sample.object_pose.position_delta
        )
        yaw_quaternion = (
            math.cos(sample.object_pose.yaw_delta / 2.0),
            0.0,
            0.0,
            math.sin(sample.object_pose.yaw_delta / 2.0),
        )
        sim.data.qpos[address + 3 : address + 7] = _quaternion_multiply(
            baseline.object_pose.quaternion, yaw_quaternion
        )

    if sample.lights:
        for light, light_baseline in zip(sample.lights, baseline.lights, strict=True):
            sim.model.light_pos[light_baseline.light_id] = np.add(
                light_baseline.position, light.position_delta
            )
            sim.model.light_diffuse[light_baseline.light_id] = np.multiply(
                light_baseline.diffuse, light.exposure_scale
            )
            sim.model.light_specular[light_baseline.light_id] = np.multiply(
                light_baseline.specular, light.exposure_scale
            )

    if sample.camera is not None:
        camera_id = baseline.camera.camera_id
        sim.model.cam_pos[camera_id] = np.add(baseline.camera.position, sample.camera.position_delta)
        sim.model.cam_quat[camera_id] = _quaternion_multiply(
            baseline.camera.quaternion, _axis_angle_quaternion(sample.camera.axis_angle_delta)
        )
        sim.model.cam_fovy[camera_id] = baseline.camera.fov + sample.camera.fov_delta
    sim.forward()


def capture_scene1_realized_metadata(
    sim: Any,
    target_object: str,
    sample: Scene1RandomizationSample,
    baseline: Scene1RandomizationBaseline,
) -> dict[str, Any]:
    realized: dict[str, Any] = {}
    address = baseline.object_pose.qpos_address
    realized["target_object"] = {
        "name": target_object,
        "qpos": [float(value) for value in sim.data.qpos[address : address + 7]],
    }
    realized["lights"] = [
        {
            "name": light.name,
            "position": [float(value) for value in sim.model.light_pos[light.light_id]],
            "diffuse": [float(value) for value in sim.model.light_diffuse[light.light_id]],
            "specular": [float(value) for value in sim.model.light_specular[light.light_id]],
        }
        for light in baseline.lights
    ]
    camera = baseline.camera
    realized["camera"] = {
        "name": camera.name,
        "position": [float(value) for value in sim.model.cam_pos[camera.camera_id]],
        "quaternion": [float(value) for value in sim.model.cam_quat[camera.camera_id]],
        "fov": float(sim.model.cam_fovy[camera.camera_id]),
    }
    return {"requested": sample.to_dict(), "realized": realized}
