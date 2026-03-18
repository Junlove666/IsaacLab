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
            "motion_library": motion_library,
            "std": 0.45,
        },
    )


@configclass
class H1MotionRoughEnvCfg(H1RoughEnvCfg):
    """H1 motion-sequence task on rough terrain scene template.

    Uses a 4D command to keep observation size unchanged:
      [motion_id_scaled, phase_scaled, 0, 0]
    """

    rewards: H1MotionRewards = H1MotionRewards()

    def __post_init__(self):
        super().__post_init__()

        # episode length to match one motion cycle
        self.episode_length_s = 10.0

        # swap command generator to motion sequence command (dim=4 to keep obs shape)
        self.commands.base_velocity = mdp.MotionSequenceCommandCfg(
            asset_name="robot",
            num_motions=len(motion_library.MOTIONS),
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

