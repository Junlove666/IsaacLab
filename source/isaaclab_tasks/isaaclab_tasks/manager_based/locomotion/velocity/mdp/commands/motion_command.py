"""Motion command term for motion-sequence tasks.

This command outputs a 3D tensor to keep the policy observation shape unchanged for the H1 velocity tasks:
  [motion_id_scaled, phase_scaled, 0]
"""

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass


class MotionSequenceCommand(CommandTerm):
    """Outputs a 3D command: [motion_id_scaled, phase_scaled, 0]."""

    def __init__(self, cfg: "MotionSequenceCommandCfg", env):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]

        # buffers
        self._motion_id = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._phase = torch.zeros(self.num_envs, device=self.device)
        self._cmd = torch.zeros(self.num_envs, 3, device=self.device)

        # metrics
        self.metrics["motion_id"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._cmd

    def _update_metrics(self):
        self.metrics["motion_id"][:] = self._motion_id.float()

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        # sample new motion ids uniformly
        self._motion_id[env_ids] = torch.randint(
            low=0, high=int(self.cfg.num_motions), size=(len(env_ids),), device=self.device
        )
        # reset phase
        self._phase[env_ids] = 0.0

    def _update_command(self):
        # advance phase
        dt = self._env.step_dt
        self._phase += dt / float(self.cfg.motion_duration_s)
        self._phase %= 1.0

        # scale motion id to [-1, 1]
        if self.cfg.num_motions <= 1:
            motion_scaled = torch.zeros_like(self._phase)
        else:
            motion_scaled = (self._motion_id.float() / (self.cfg.num_motions - 1)) * 2.0 - 1.0
        # scale phase to [-1, 1]
        phase_scaled = self._phase * 2.0 - 1.0

        self._cmd[:, 0] = motion_scaled
        self._cmd[:, 1] = phase_scaled
        self._cmd[:, 2] = 0.0


@configclass
class MotionSequenceCommandCfg(CommandTermCfg):
    """Configuration for :class:`MotionSequenceCommand`."""

    # NOTE: Set directly here (do not patch later), otherwise config validation may see MISSING.
    class_type: type[CommandTerm] = MotionSequenceCommand

    asset_name: str = "robot"
    """Name of the robot articulation in the scene."""

    num_motions: int = 3
    """Number of motion sequences."""

    motion_duration_s: float = 10.0
    """Duration of one motion cycle in seconds."""

    def __post_init__(self):
        # Default: resample a new motion every motion_duration_s (roughly one cycle).
        if getattr(self, "resampling_time_range", MISSING) is MISSING or self.resampling_time_range is None:
            self.resampling_time_range = (float(self.motion_duration_s), float(self.motion_duration_s))
