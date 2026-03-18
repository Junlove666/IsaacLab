# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.rough_env_cfg import H1RoughEnvCfg, H1Rewards

from . import motion_library

# -----------------------------
# Goal 2: reference-file tracking
# -----------------------------

# Upper-body reference joints we generate/track (must exist in the H1 articulation).
TRACK_JOINT_NAMES: list[str] = [
    "torso",
    "left_shoulder_pitch",
    "right_shoulder_pitch",
    "left_shoulder_roll",
    "right_shoulder_roll",
    "left_elbow",
    "right_elbow",
]

# Training set (sampled randomly per environment episode).
# You can add more files here to increase diversity.
TRAIN_MOTION_REF_FILES: list[str] = [
    "scripts/motions/h1_motion_ref_train01.yaml",
    "scripts/motions/h1_motion_ref_train02.yaml",
    "scripts/motions/h1_motion_ref_train03.yaml",
    "scripts/motions/h1_motion_ref_train04.yaml",
    "scripts/motions/h1_motion_ref_train05.yaml",
]

# Default motion for PLAY (can be overridden by CLI).
PLAY_MOTION_REF_FILE: str = "scripts/motions/h1_motion_ref_motion04.yaml"


# Curriculum is based on base_velocity tracking distance; for Goal 2 we repurpose base_velocity as
# ref_joint_pos, so we disable it to avoid semantic mismatch.
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import CurriculumCfg as LocomotionCurriculumCfg


@configclass
class H1MotionRefCurriculumCfg(LocomotionCurriculumCfg):
    terrain_levels = None


@configclass
class H1MotionRewards(H1Rewards):
    """Rewards for motion-sequence imitation with balance preservation."""

    # Disable velocity tracking / stepping-specific rewards.
    track_lin_vel_xy_exp = None
    track_ang_vel_z_exp = None
    feet_air_time = None

    # Encourage following motion reference on upper body + torso joints.
    track_motion_pose = RewTerm(
        func=mdp.track_motion_joint_pos_exp,
        weight=2.0,
        params={
            "command_name": "base_velocity",
            "joint_names": [
                "torso",
                "left_shoulder_pitch",
                "right_shoulder_pitch",
                "left_shoulder_roll",
                "right_shoulder_roll",
                "left_elbow",
                "right_elbow",
            ],
            "motions": motion_library.MOTIONS_DATA,
            "std": 0.45,
        },
    )


@configclass
class H1MotionRoughEnvCfg(H1RoughEnvCfg):
    """H1 motion-sequence task on rough terrain scene template.

    Uses a 3D command to keep observation size unchanged:
      [motion_id_scaled, phase_scaled, 0]
    """

    rewards: H1MotionRewards = H1MotionRewards()

    def __post_init__(self):
        super().__post_init__()

        # episode length to match one motion cycle
        self.episode_length_s = 10.0

        # swap command generator to motion sequence command (dim=4 to keep obs shape)
        self.commands.base_velocity = mdp.MotionSequenceCommandCfg(
            asset_name="robot",
            num_motions=len(motion_library.MOTIONS_DATA),
            motion_duration_s=10.0,
            resampling_time_range=(10.0, 10.0),
            debug_vis=False,
        )

        # reduce penalties that can over-constrain expressive upper-body motions
        self.rewards.joint_deviation_arms.weight = -0.05
        self.rewards.joint_deviation_torso.weight = -0.05

        # encourage staying upright
        self.rewards.flat_orientation_l2.weight = -2.0


@configclass
class H1MotionRoughEnvCfg_PLAY(H1MotionRoughEnvCfg):
    """Play config for motion-sequence task."""

    def __post_init__(self):
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 10.0

        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing / external forces
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class H1MotionRefRewards(H1Rewards):
    """Rewards for reference-file imitation with balance preservation."""

    # Disable velocity tracking / stepping-specific rewards.
    track_lin_vel_xy_exp = None
    track_ang_vel_z_exp = None
    feet_air_time = None

    # Track reference joint positions provided by the command term (Goal 2).
    track_ref_pose = RewTerm(
        func=mdp.track_ref_joint_pos_exp,
        weight=2.0,
        params={
            "command_name": "base_velocity",
            "joint_names": TRACK_JOINT_NAMES,
            "std": 0.45,
        },
    )


@configclass
class H1MotionRefRoughEnvCfg(H1RoughEnvCfg):
    """Reference-file motion task on rough terrain scene template (Goal 2).

    Implementation trick:
        - We repurpose ``commands.base_velocity`` to output ``ref_joint_pos`` directly.
        - The existing observation term ``velocity_commands`` then concatenates this reference into the policy obs.
    """

    rewards: H1MotionRefRewards = H1MotionRefRewards()
    curriculum: H1MotionRefCurriculumCfg = H1MotionRefCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()

        # episode length to match the reference duration (used for command resampling).
        self.episode_length_s = 10.0

        # swap command generator to reference joint position command.
        self.commands.base_velocity = mdp.MotionRefJointPosCommandCfg(
            asset_name="robot",
            ref_joint_names=TRACK_JOINT_NAMES,
            motion_ref_files=TRAIN_MOTION_REF_FILES,
            motion_duration_s=10.0,
            resampling_time_range=(10.0, 10.0),
        )

        # reduce penalties that can over-constrain expressive upper-body motions
        self.rewards.joint_deviation_arms.weight = -0.05
        self.rewards.joint_deviation_torso.weight = -0.05

        # encourage staying upright
        self.rewards.flat_orientation_l2.weight = -2.0


@configclass
class H1MotionRefRoughEnvCfg_PLAY(H1MotionRefRoughEnvCfg):
    """Play config for reference-file motion task."""

    def __post_init__(self):
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 10.0

        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        # default single file; play.py can override via CLI.
        self.commands.base_velocity.motion_ref_files = [PLAY_MOTION_REF_FILE]
