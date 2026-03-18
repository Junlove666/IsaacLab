# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned
    robot frame using an exponential kernel.
    """
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def stand_still_joint_deviation_l1(
    env, command_name: str, command_threshold: float = 0.06, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    command = env.command_manager.get_command(command_name)
    # Penalize motion when command is nearly zero.
    return mdp.joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :2], dim=1) < command_threshold)


def track_motion_joint_pos_exp(
    env,
    command_name: str,
    joint_names: list[str],
    motions: list[list[tuple[float, dict[str, float]]]],
    std: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking a reference joint pose defined by a motion library.

    Expects the command tensor to be at least 2D, with:
    - command[0]: motion_id scaled to [-1, 1]
    - command[1]: phase scaled to [-1, 1]
    Remaining dims (if any) are unused.
    """
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    # decode motion id and phase
    num_motions = len(motions)
    if num_motions <= 0:
        return torch.zeros((env.num_envs,), device=asset.data.joint_pos.device)
    motion_cont = (cmd[:, 0] + 1.0) * 0.5 * (num_motions - 1)
    motion_id = torch.clamp(torch.round(motion_cont), 0, num_motions - 1).to(torch.long)
    phase = torch.clamp((cmd[:, 1] + 1.0) * 0.5, 0.0, 1.0)

    # cache joint ids on env for performance
    cache_key = "_motion_track_joint_ids_" + "_".join(joint_names)
    if not hasattr(env, cache_key):
        name_to_id = {n: i for i, n in enumerate(asset.data.joint_names)}
        ids = []
        for n in joint_names:
            if n not in name_to_id:
                raise ValueError(f"Joint name '{n}' not found in asset joint_names.")
            ids.append(name_to_id[n])
        setattr(env, cache_key, torch.tensor(ids, device=asset.data.joint_pos.device, dtype=torch.long))
    joint_ids = getattr(env, cache_key)

    def _smoothstep(x: float) -> float:
        x = 0.0 if x < 0.0 else 1.0 if x > 1.0 else x
        return x * x * (3.0 - 2.0 * x)

    def _sample_motion(motion_data: list[tuple[float, dict[str, float]]], ph: float) -> dict[str, float]:
        if not motion_data:
            return {}
        ph = ph % 1.0
        frames = sorted(motion_data, key=lambda it: it[0])
        # wrap by duplicating first at +1
        phases = [p for p, _ in frames] + [frames[0][0] + 1.0]
        deltas = [d for _, d in frames] + [frames[0][1]]
        idx = 0
        while idx + 1 < len(phases) and not (phases[idx] <= ph < phases[idx + 1]):
            idx += 1
        p0, p1 = phases[idx], phases[idx + 1]
        d0, d1 = deltas[idx], deltas[idx + 1]
        seg = (ph - p0) / (p1 - p0 + 1e-9)
        w = _smoothstep(seg)
        keys = set(d0.keys()) | set(d1.keys())
        out: dict[str, float] = {}
        for k in keys:
            out[k] = (1.0 - w) * float(d0.get(k, 0.0)) + w * float(d1.get(k, 0.0))
        return out

    # build target deltas per env on CPU (small) then move to torch.
    target_delta = torch.zeros((env.num_envs, len(joint_names)), device=asset.data.joint_pos.device)
    for i in range(env.num_envs):
        deltas = _sample_motion(motions[int(motion_id[i].item())], float(phase[i].item()))
        for j, name in enumerate(joint_names):
            target_delta[i, j] = float(deltas.get(name, 0.0))

    q_default = asset.data.default_joint_pos[:, joint_ids]
    q_target = q_default + target_delta
    q = asset.data.joint_pos[:, joint_ids]
    err = torch.mean(torch.square(q - q_target), dim=1)
    return torch.exp(-err / (std**2))
