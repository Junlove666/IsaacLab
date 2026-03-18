# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import os
import re
from collections.abc import Sequence
from dataclasses import MISSING
import json

import torch
try:
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    yaml = None  # type: ignore

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import retrieve_file_path


class MotionRefJointPosCommand(CommandTerm):
    """Outputs a reference joint position tensor with shape (num_envs, len(ref_joint_names)).

    YAML format:
        - optional: ``ramp_in_s`` (float, seconds)
        - ``joints``: list of items, each item has:
            - ``pattern`` (regex string)
            - ``amplitude`` (float, default 0)
            - ``frequency_hz`` (float, default 1)
            - ``phase_rad`` (float, default 0)
            - ``bias`` (float, default 0)

    The generated reference position is:
        ``q_ref = q_default + ramp(t) * (bias + amplitude * sin(2*pi*frequency*t + phase))``
    """

    def __init__(self, cfg: MotionRefJointPosCommandCfg, env):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        if len(cfg.ref_joint_names) == 0:
            raise ValueError("ref_joint_names must be non-empty.")
        if len(cfg.motion_ref_files) == 0:
            raise ValueError("motion_ref_files must be non-empty.")

        self.ref_joint_names = list(cfg.ref_joint_names)
        self.ref_dim = len(self.ref_joint_names)

        # map joint names -> joint indices in the articulation
        name_to_id = {n: i for i, n in enumerate(self.robot.data.joint_names)}
        joint_ids = []
        for n in self.ref_joint_names:
            if n not in name_to_id:
                raise ValueError(f"Joint name '{n}' not found in robot joint_names.")
            joint_ids.append(name_to_id[n])
        self._joint_ids = torch.tensor(joint_ids, device=self.device, dtype=torch.long)

        # buffers
        self._time = torch.zeros(self.num_envs, device=self.device)
        self._ref_idx = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._cmd = torch.zeros(self.num_envs, self.ref_dim, device=self.device)

        # Default joint positions (same for all envs, but stored per-env tensor for simplicity).
        self._q_default = self.robot.data.default_joint_pos[:, self._joint_ids]

        # Pre-load and pre-compile all reference motion files.
        self._ref_files_compiled: list[dict] = []
        for f in cfg.motion_ref_files:
            try:
                ref_path = retrieve_file_path(f)
            except FileNotFoundError:
                # If the runner is launched from the RL_Projects root, configs that use
                # "scripts/motions/..." (relative to IsaacLab/) would not exist.
                # Best-effort fallback: try "IsaacLab/<path>".
                ref_path = retrieve_file_path(os.path.join("IsaacLab", f))
            with open(ref_path, encoding="utf-8") as fp:
                text = fp.read()
            if yaml is not None:
                ref_cfg = yaml.safe_load(text)
            else:
                # Fallback: support JSON motion files when PyYAML is unavailable.
                ref_cfg = json.loads(text)
            if not isinstance(ref_cfg, dict):
                raise ValueError(f"Invalid YAML root in motion ref file: {ref_path}")

            ramp_in_s = float(ref_cfg.get("ramp_in_s", 0.0) or 0.0)
            joints_cfg = ref_cfg.get("joints", [])
            if not isinstance(joints_cfg, list):
                raise ValueError(f"Invalid YAML field 'joints' in motion ref file: {ref_path}")

            compiled_patterns: list[tuple[float, float, float, float, list[int]]] = []
            for item in joints_cfg:
                if not isinstance(item, dict):
                    continue
                pattern = item.get("pattern")
                if not pattern:
                    continue

                pat = re.compile(str(pattern))
                amp = float(item.get("amplitude", 0.0) or 0.0)
                freq_hz = float(item.get("frequency_hz", 1.0) or 1.0)
                phase_rad = float(item.get("phase_rad", 0.0) or 0.0)
                bias = float(item.get("bias", 0.0) or 0.0)

                matched = [i for i, jn in enumerate(self.ref_joint_names) if pat.search(jn) is not None]
                if len(matched) == 0:
                    continue

                compiled_patterns.append((amp, freq_hz, phase_rad, bias, matched))

            self._ref_files_compiled.append({"ramp_in_s": ramp_in_s, "patterns": compiled_patterns})

        # metrics (optional but handy for debugging)
        self.metrics["motion_ref_idx"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._cmd

    def _update_metrics(self):
        self.metrics["motion_ref_idx"][:] = self._ref_idx.float()

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        # sample a new reference file per env
        self._ref_idx[env_ids] = torch.randint(
            low=0, high=len(self._ref_files_compiled), size=(len(env_ids),), device=self.device
        )
        self._time[env_ids] = 0.0

    def _update_command(self):
        dt = float(self._env.step_dt)
        self._time += dt

        # Compute command by reference groups to avoid per-env branching.
        for ref_i in range(len(self._ref_files_compiled)):
            env_mask = self._ref_idx == ref_i
            if not torch.any(env_mask):
                continue
            env_ids = env_mask.nonzero(as_tuple=False).flatten()
            t = self._time[env_ids]

            ramp_in_s = float(self._ref_files_compiled[ref_i]["ramp_in_s"])
            if ramp_in_s > 0.0:
                ramp = torch.clamp(t / ramp_in_s, max=1.0)
            else:
                ramp = torch.ones_like(t)

            delta = torch.zeros((len(env_ids), self.ref_dim), device=self.device)
            for (amp, freq_hz, phase_rad, bias, matched) in self._ref_files_compiled[ref_i]["patterns"]:
                if amp == 0.0 and bias == 0.0:
                    continue
                v = bias + amp * torch.sin(2.0 * math.pi * freq_hz * t + phase_rad)
                v = v * ramp
                # apply the same v to all matched joints for this pattern
                delta[:, matched] += v.unsqueeze(1)

            self._cmd[env_ids] = self._q_default[env_ids] + delta


@configclass
class MotionRefJointPosCommandCfg(CommandTermCfg):
    """Generate a reference joint position vector from external YAML motion files.

    Notes:
        - This command term is plugged into ``commands.base_velocity`` so it automatically becomes part
          of the existing policy observation term ``velocity_commands``.
        - The output order matches ``ref_joint_names``.
    """

    # NOTE: set directly here (do not patch later), otherwise config validation may see MISSING.
    class_type: type[CommandTerm] = MotionRefJointPosCommand

    asset_name: str = "robot"
    ref_joint_names: list[str] = MISSING
    """Joint names (in this order) whose reference positions are generated."""

    motion_ref_files: list[str] = MISSING
    """List of YAML reference motion files (one file sampled per environment episode)."""

    # How long (s) each sampled motion should last before resampling.
    motion_duration_s: float = 10.0

    def __post_init__(self):
        # Ensure resampling matches one episode-length by default.
        if getattr(self, "resampling_time_range", MISSING) is MISSING or self.resampling_time_range is None:
            self.resampling_time_range = (float(self.motion_duration_s), float(self.motion_duration_s))

