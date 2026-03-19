# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Batch Sim2Sim evaluator for A/B checkpoints on easy/medium/hard presets."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Batch-evaluate Sim2Sim tasks for H1 A/B checkpoints.")
parser.add_argument("--a_checkpoint", type=str, required=True, help="Checkpoint path for model A (velocity rough).")
parser.add_argument("--b_checkpoint", type=str, required=True, help="Checkpoint path for model B (zhanma ref flat).")
parser.add_argument(
    "--b_motion_ref_file",
    type=str,
    default="scripts/motions/h1_zhanma_ref_motion04.yaml",
    help="Motion reference file used by B tasks.",
)
parser.add_argument("--num_envs", type=int, default=64, help="Parallel env count for each case.")
parser.add_argument("--episodes_per_case", type=int, default=200, help="Number of completed episodes per case.")
parser.add_argument("--seed", type=int, default=42, help="Evaluation seed.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Registry key for agent config.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry


@dataclass
class EvalCase:
    name: str
    task: str
    checkpoint: str
    model_type: str  # "A" | "B"


CASES: list[EvalCase] = [
    EvalCase("A_easy", "Isaac-Velocity-Rough-H1-Sim2Sim-Easy-Play-v0", args_cli.a_checkpoint, "A"),
    EvalCase("A_medium", "Isaac-Velocity-Rough-H1-Sim2Sim-Medium-Play-v0", args_cli.a_checkpoint, "A"),
    EvalCase("A_hard", "Isaac-Velocity-Rough-H1-Sim2Sim-Hard-Play-v0", args_cli.a_checkpoint, "A"),
    EvalCase("B_easy", "Isaac-ZhanmaRef-Flat-H1-Sim2Sim-Easy-Play-v0", args_cli.b_checkpoint, "B"),
    EvalCase("B_medium", "Isaac-ZhanmaRef-Flat-H1-Sim2Sim-Medium-Play-v0", args_cli.b_checkpoint, "B"),
    EvalCase("B_hard", "Isaac-ZhanmaRef-Flat-H1-Sim2Sim-Hard-Play-v0", args_cli.b_checkpoint, "B"),
]


def _to_float(value, default: float = math.nan) -> float:
    if value is None:
        return default
    if isinstance(value, (float, int)):
        return float(value)
    if torch.is_tensor(value):
        if value.numel() == 0:
            return default
        return float(value.item())
    return default


def _resolve_tracking_score(log_dict: dict) -> tuple[float, str]:
    if "Episode_Reward/track_ref_pose_legs" in log_dict:
        return _to_float(log_dict["Episode_Reward/track_ref_pose_legs"]), "track_ref_pose_legs"
    if "Episode_Reward/track_ref_pose" in log_dict:
        return _to_float(log_dict["Episode_Reward/track_ref_pose"]), "track_ref_pose"
    has_lin = "Episode_Reward/track_lin_vel_xy_exp" in log_dict
    has_ang = "Episode_Reward/track_ang_vel_z_exp" in log_dict
    if has_lin and has_ang:
        lin = _to_float(log_dict["Episode_Reward/track_lin_vel_xy_exp"])
        ang = _to_float(log_dict["Episode_Reward/track_ang_vel_z_exp"])
        return 0.5 * (lin + ang), "0.5*(track_lin_vel_xy_exp+track_ang_vel_z_exp)"
    if has_lin:
        return _to_float(log_dict["Episode_Reward/track_lin_vel_xy_exp"]), "track_lin_vel_xy_exp"
    if has_ang:
        return _to_float(log_dict["Episode_Reward/track_ang_vel_z_exp"]), "track_ang_vel_z_exp"
    return math.nan, "n/a"


def evaluate_case(case: EvalCase) -> dict[str, float | str]:
    env_cfg = load_cfg_from_registry(case.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(case.task, args_cli.agent)

    env_cfg.scene.num_envs = int(args_cli.num_envs)
    env_cfg.seed = int(args_cli.seed)
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device
    agent_cfg.seed = int(args_cli.seed)

    if case.model_type == "B":
        try:
            motion_path = retrieve_file_path(args_cli.b_motion_ref_file)
        except FileNotFoundError:
            motion_path = retrieve_file_path(f"IsaacLab/{args_cli.b_motion_ref_file}")
        env_cfg.commands.base_velocity.motion_ref_files = [motion_path]

    env = gym.make(case.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    ckpt = retrieve_file_path(case.checkpoint)
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(ckpt)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs = env.get_observations()
    target_episodes = int(args_cli.episodes_per_case)

    episodes = 0
    falls = 0
    drift_xy_sum = 0.0
    drift_xy_count = 0
    vel_xy_sum = 0.0
    vel_xy_count = 0
    track_sum = 0.0
    track_count = 0
    track_name = "n/a"

    while simulation_app.is_running() and episodes < target_episodes:
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, extras = env.step(actions)
            # reset recurrent states, matching play.py behavior
            if hasattr(policy, "reset"):
                policy.reset(dones)

        done_count = int(dones.sum().item())
        if done_count == 0:
            continue

        episodes += done_count

        terminated = env.unwrapped.reset_terminated.bool()
        time_outs = env.unwrapped.reset_time_outs.bool()
        falls += int((terminated & (~time_outs)).sum().item())

        log_dict = extras.get("log", {})
        if "Episode_Reward/root_pos_xy_l2" in log_dict:
            drift_xy_sum += _to_float(log_dict["Episode_Reward/root_pos_xy_l2"], 0.0) * done_count
            drift_xy_count += done_count
        if "Episode_Reward/root_lin_vel_xy_l2" in log_dict:
            vel_xy_sum += _to_float(log_dict["Episode_Reward/root_lin_vel_xy_l2"], 0.0) * done_count
            vel_xy_count += done_count

        t_score, t_name = _resolve_tracking_score(log_dict)
        if not math.isnan(t_score):
            track_sum += t_score * done_count
            track_count += done_count
            track_name = t_name

    env.close()

    denom = max(episodes, 1)
    return {
        "case": case.name,
        "task": case.task,
        "episodes": episodes,
        "fall_rate": falls / denom,
        "avg_root_pos_xy_l2": drift_xy_sum / max(drift_xy_count, 1) if drift_xy_count > 0 else math.nan,
        "avg_root_lin_vel_xy_l2": vel_xy_sum / max(vel_xy_count, 1) if vel_xy_count > 0 else math.nan,
        "tracking_metric": track_name,
        "tracking_score": track_sum / max(track_count, 1) if track_count > 0 else math.nan,
    }


def _fmt(v: float | str) -> str:
    if isinstance(v, str):
        return v
    if math.isnan(v):
        return "n/a"
    return f"{v:.4f}"


def main():
    results = []
    for case in CASES:
        print(f"[INFO] Evaluating {case.name} ({case.task}) ...")
        results.append(evaluate_case(case))

    print("\n=== Sim2Sim Batch Eval Results ===")
    header = (
        f"{'case':<10} {'episodes':>9} {'fall_rate':>10} "
        f"{'root_pos_xy_l2':>14} {'root_lin_xy_l2':>14} {'tracking_metric':>42} {'tracking_score':>14}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['case']:<10} "
            f"{int(r['episodes']):>9d} "
            f"{_fmt(r['fall_rate']):>10} "
            f"{_fmt(r['avg_root_pos_xy_l2']):>14} "
            f"{_fmt(r['avg_root_lin_vel_xy_l2']):>14} "
            f"{_fmt(r['tracking_metric']):>42} "
            f"{_fmt(r['tracking_score']):>14}"
        )


if __name__ == "__main__":
    main()
    simulation_app.close()
