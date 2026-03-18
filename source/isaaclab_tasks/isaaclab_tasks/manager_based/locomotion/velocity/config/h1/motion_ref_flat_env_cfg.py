# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.flat_env_cfg import H1FlatEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.rough_env_cfg import H1Rewards
from isaaclab_tasks.manager_based.locomotion.velocity.mdp.commands.motion_ref_command import (
    MotionRefJointPosCommandCfg as Goal2MotionRefJointPosCommandCfg,
)


# -----------------------------
# Goal 2 (Flat): Zhanma bu + Punching
# -----------------------------

# Track full 19 joints (legs + arms + torso). These joint names must match articulation joint_names.
TRACK_JOINT_NAMES: list[str] = [
    "torso",
    "left_hip_yaw",
    "right_hip_yaw",
    "left_hip_roll",
    "right_hip_roll",
    "left_hip_pitch",
    "right_hip_pitch",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_shoulder_pitch",
    "right_shoulder_pitch",
    "left_shoulder_roll",
    "right_shoulder_roll",
    "left_shoulder_yaw",
    "right_shoulder_yaw",
    "left_elbow",
    "right_elbow",
]


# Default reference files for training (sampled randomly per env episode).
TRAIN_MOTION_REF_FILES: list[str] = [
    "scripts/motions/h1_zhanma_ref_train01.yaml",
    "scripts/motions/h1_zhanma_ref_train02.yaml",
    "scripts/motions/h1_zhanma_ref_train03.yaml",
    "scripts/motions/h1_zhanma_ref_train04.yaml",
    "scripts/motions/h1_zhanma_ref_train05.yaml",
]

# Default unseen motion file for play.
PLAY_MOTION_REF_FILE: str = "scripts/motions/h1_zhanma_ref_motion04.yaml"


@configclass
class H1MotionRefFlatRewards(H1Rewards):
    """Stay-in-place + track ref joint positions (legs + upper body)."""

    # Disable velocity tracking / stepping rewards.
    track_lin_vel_xy_exp = None
    track_ang_vel_z_exp = None
    feet_air_time = None
    feet_slide = None

    # Reward tracking reference joint positions (command provides ref_joint_pos directly).
    track_ref_pose = RewTerm(
        func=mdp.track_ref_joint_pos_exp,
        weight=1.5,
        params={
            "command_name": "base_velocity",
            "joint_names": TRACK_JOINT_NAMES,
            "std": 0.6,
        },
    )

    # Penalize x/y drift: encourage "stand" and stable stance (zhanma bu).
    root_lin_vel_xy_l2 = RewTerm(
        func=mdp.root_lin_vel_xy_l2,
        weight=-0.15,
    )

    # Small drift penalty in world xy (helps stop slow walking).
    root_pos_xy_l2 = RewTerm(
        func=mdp.root_pos_xy_l2,
        weight=-0.02,
    )


@configclass
class H1MotionRefFlatEnvCfg(H1FlatEnvCfg):
    """Flat training environment for Zhanma bu reference tracking."""

    rewards: H1MotionRefFlatRewards = H1MotionRefFlatRewards()

    def __post_init__(self):
        super().__post_init__()

        # No terrain randomization curriculum on flat.
        self.curriculum.terrain_levels = None

        # Episode duration should match reference motion duration.
        self.episode_length_s = 10.0

        # Replace command generator with ref joint position command.
        self.commands.base_velocity = Goal2MotionRefJointPosCommandCfg(
            asset_name="robot",
            ref_joint_names=TRACK_JOINT_NAMES,
            motion_ref_files=TRAIN_MOTION_REF_FILES,
            motion_duration_s=10.0,
            resampling_time_range=(10.0, 10.0),
        )

        # Tuning for upright stability.
        self.rewards.flat_orientation_l2.weight = -2.0
        # Reduce strong deviation penalties that could fight reference tracking.
        self.rewards.joint_deviation_hip.weight = -0.05
        self.rewards.joint_deviation_arms.weight = -0.05
        self.rewards.joint_deviation_torso.weight = -0.05


@configclass
class H1MotionRefFlatEnvCfg_PLAY(H1MotionRefFlatEnvCfg):
    """Play config for Zhanma bu reference tracking."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 10.0

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        self.commands.base_velocity.motion_ref_files = [PLAY_MOTION_REF_FILE]

