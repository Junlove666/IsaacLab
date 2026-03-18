# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Keyframe:
    """A keyframe in normalized phase [0, 1]."""

    phase: float
    # joint_name -> delta(rad) relative to default pose
    delta: dict[str, float]


def _smoothstep(x: float) -> float:
    """C1 smoothstep on [0,1]."""
    x = 0.0 if x < 0.0 else 1.0 if x > 1.0 else x
    return x * x * (3.0 - 2.0 * x)


def sample_keyframes(keyframes: list[Keyframe], phase: float) -> dict[str, float]:
    """Interpolate deltas for a given phase (wraps to [0,1))."""
    if not keyframes:
        return {}
    # wrap
    phase = phase % 1.0
    # find segment
    kfs = sorted(keyframes, key=lambda k: k.phase)
    # handle wrap-around by duplicating first frame at +1
    phases = [k.phase for k in kfs] + [kfs[0].phase + 1.0]
    frames = kfs + [Keyframe(kfs[0].phase + 1.0, kfs[0].delta)]
    # locate interval
    idx = 0
    while idx + 1 < len(phases) and not (phases[idx] <= phase < phases[idx + 1]):
        idx += 1
    a = frames[idx]
    b = frames[idx + 1]
    # normalized within segment
    seg = (phase - a.phase) / (b.phase - a.phase + 1e-9)
    w = _smoothstep(seg)
    # union of keys
    out: dict[str, float] = {}
    keys = set(a.delta.keys()) | set(b.delta.keys())
    for k in keys:
        va = a.delta.get(k, 0.0)
        vb = b.delta.get(k, 0.0)
        out[k] = (1.0 - w) * va + w * vb
    return out


"""
Three motion sequences (each designed for ~10s loop when phase runs 0->1):

0: Wave -> Clap -> Martial pose
1: Twist torso + arm swings
2: Two simple "kungfu" poses alternating

All deltas are relative to the default H1 pose.
"""

MOTION_NAMES = ["wave_clap_pose", "twist_swing", "kungfu_two_poses"]

MOTIONS: list[list[Keyframe]] = [
    # 0) Wave (right arm) -> Clap (both) -> Pose
    [
        Keyframe(0.00, {"right_shoulder_pitch": +0.6, "right_elbow": +0.6}),
        Keyframe(0.12, {"right_shoulder_pitch": +0.6, "right_elbow": -0.6}),
        Keyframe(0.24, {"right_shoulder_pitch": +0.6, "right_elbow": +0.6}),
        Keyframe(0.36, {"right_shoulder_pitch": +0.6, "right_elbow": -0.6}),
        # transition to clap: arms forward
        Keyframe(0.50, {"left_shoulder_pitch": +0.8, "right_shoulder_pitch": +0.8, "left_elbow": +0.3, "right_elbow": +0.3}),
        # clap close/open (shoulder_roll in/out)
        Keyframe(0.60, {"left_shoulder_pitch": +0.8, "right_shoulder_pitch": +0.8, "left_shoulder_roll": -0.6, "right_shoulder_roll": +0.6}),
        Keyframe(0.70, {"left_shoulder_pitch": +0.8, "right_shoulder_pitch": +0.8, "left_shoulder_roll": -0.15, "right_shoulder_roll": +0.15}),
        Keyframe(0.80, {"left_shoulder_pitch": +0.8, "right_shoulder_pitch": +0.8, "left_shoulder_roll": -0.6, "right_shoulder_roll": +0.6}),
        # martial pose: torso twist + arms set
        Keyframe(0.92, {"torso": +0.4, "left_shoulder_pitch": +0.2, "right_shoulder_pitch": +1.0, "right_elbow": -0.2}),
    ],
    # 1) Twist + swing
    [
        Keyframe(0.00, {"torso": +0.5, "left_shoulder_pitch": +0.3, "right_shoulder_pitch": +0.9}),
        Keyframe(0.25, {"torso": -0.5, "left_shoulder_pitch": +0.9, "right_shoulder_pitch": +0.3}),
        Keyframe(0.50, {"torso": +0.5, "left_shoulder_pitch": +0.3, "right_shoulder_pitch": +0.9}),
        Keyframe(0.75, {"torso": -0.5, "left_shoulder_pitch": +0.9, "right_shoulder_pitch": +0.3}),
    ],
    # 2) Two kungfu poses alternating
    [
        Keyframe(0.00, {"torso": +0.3, "left_shoulder_pitch": +1.0, "left_elbow": +0.3, "right_shoulder_pitch": +0.2}),
        Keyframe(0.45, {"torso": +0.3, "left_shoulder_pitch": +1.0, "left_elbow": +0.3, "right_shoulder_pitch": +0.2}),
        Keyframe(0.55, {"torso": -0.3, "right_shoulder_pitch": +1.0, "right_elbow": +0.3, "left_shoulder_pitch": +0.2}),
        Keyframe(1.00, {"torso": -0.3, "right_shoulder_pitch": +1.0, "right_elbow": +0.3, "left_shoulder_pitch": +0.2}),
    ],
]

