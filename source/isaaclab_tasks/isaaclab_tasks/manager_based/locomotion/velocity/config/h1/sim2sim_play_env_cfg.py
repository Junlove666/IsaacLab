# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.rough_env_cfg import H1RoughEnvCfg_PLAY
from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.motion_ref_flat_env_cfg import H1MotionRefFlatEnvCfg_PLAY


# Preset levels for Isaac-Sim-internal Sim2Sim robustness evaluation.
_LEVEL_PRESETS: dict[str, dict[str, object]] = {
    "easy": {
        "obs_noise_scale": 1.15,
        "friction_static": (0.7, 1.0),
        "friction_dynamic": (0.5, 0.8),
        "joint_pos_range": (0.9, 1.1),
        "reset_vel_abs": 0.15,
        "base_force_abs": 35.0,
        "base_torque_abs": 7.0,
        "push_vel_abs": 0.20,
        "push_interval": (12.0, 16.0),
    },
    "medium": {
        "obs_noise_scale": 1.35,
        "friction_static": (0.55, 1.1),
        "friction_dynamic": (0.4, 0.9),
        "joint_pos_range": (0.82, 1.18),
        "reset_vel_abs": 0.30,
        "base_force_abs": 70.0,
        "base_torque_abs": 14.0,
        "push_vel_abs": 0.35,
        "push_interval": (9.0, 13.0),
    },
    "hard": {
        "obs_noise_scale": 1.7,
        "friction_static": (0.4, 1.2),
        "friction_dynamic": (0.3, 1.0),
        "joint_pos_range": (0.72, 1.28),
        "reset_vel_abs": 0.45,
        "base_force_abs": 110.0,
        "base_torque_abs": 22.0,
        "push_vel_abs": 0.55,
        "push_interval": (6.0, 10.0),
    },
}


def _scale_observation_noise(cfg, scale: float):
    # Observation terms are dataclass-like objects with optional additive-uniform noise configs.
    for term_name in (
        "base_lin_vel",
        "base_ang_vel",
        "projected_gravity",
        "joint_pos",
        "joint_vel",
        "height_scan",
    ):
        term_cfg = getattr(cfg.observations.policy, term_name, None)
        if term_cfg is None or getattr(term_cfg, "noise", None) is None:
            continue
        term_cfg.noise.n_min *= scale
        term_cfg.noise.n_max *= scale


def _apply_sim2sim_level(cfg, level: str):
    p = _LEVEL_PRESETS[level]

    cfg.scene.num_envs = 64
    cfg.scene.env_spacing = 2.5
    cfg.observations.policy.enable_corruption = True

    _scale_observation_noise(cfg, float(p["obs_noise_scale"]))

    # Broaden contact material randomization.
    cfg.events.physics_material.params["static_friction_range"] = p["friction_static"]
    cfg.events.physics_material.params["dynamic_friction_range"] = p["friction_dynamic"]

    # Increase reset randomization.
    cfg.events.reset_robot_joints.params["position_range"] = p["joint_pos_range"]
    v = float(p["reset_vel_abs"])
    cfg.events.reset_base.params["velocity_range"] = {
        "x": (-v, v),
        "y": (-v, v),
        "z": (-v, v),
        "roll": (-v, v),
        "pitch": (-v, v),
        "yaw": (-v, v),
    }

    # Re-enable reset impulse and interval pushes for robustness testing.
    f = float(p["base_force_abs"])
    t = float(p["base_torque_abs"])
    cfg.events.base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*torso_link"]),
            "force_range": (-f, f),
            "torque_range": (-t, t),
        },
    )

    pv = float(p["push_vel_abs"])
    cfg.events.push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=p["push_interval"],
        params={"velocity_range": {"x": (-pv, pv), "y": (-pv, pv)}},
    )


@configclass
class H1RoughEnvCfg_Sim2SimEasy_PLAY(H1RoughEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _apply_sim2sim_level(self, "easy")


@configclass
class H1RoughEnvCfg_Sim2SimMedium_PLAY(H1RoughEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _apply_sim2sim_level(self, "medium")


@configclass
class H1RoughEnvCfg_Sim2SimHard_PLAY(H1RoughEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _apply_sim2sim_level(self, "hard")


@configclass
class H1MotionRefFlatEnvCfg_Sim2SimEasy_PLAY(H1MotionRefFlatEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _apply_sim2sim_level(self, "easy")


@configclass
class H1MotionRefFlatEnvCfg_Sim2SimMedium_PLAY(H1MotionRefFlatEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _apply_sim2sim_level(self, "medium")


@configclass
class H1MotionRefFlatEnvCfg_Sim2SimHard_PLAY(H1MotionRefFlatEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _apply_sim2sim_level(self, "hard")
