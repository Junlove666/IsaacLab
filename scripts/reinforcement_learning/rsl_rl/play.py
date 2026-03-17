# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import math
import re

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--ref_traj",
    type=str,
    default=None,
    help=(
        "Optional YAML reference joint trajectory for residual mode. "
        "If provided, we run: action = policy(obs) * ref_traj_residual_scale + (ref_joint_pos - default_joint_pos)/scale "
        "for the specified joint position action term."
    ),
)
parser.add_argument(
    "--ref_traj_action_term",
    type=str,
    default="joint_pos",
    help="Name of the joint position action term in the env config. Defaults to 'joint_pos'.",
)
parser.add_argument(
    "--ref_traj_residual_scale",
    type=float,
    default=1.0,
    help="Scale multiplier applied to the policy residual before adding to reference action.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for installed RSL-RL version."""

import importlib.metadata as metadata

from packaging import version

installed_version = metadata.version("rsl-rl-lib")

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import torch
import yaml
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)

def _maybe_set_camera_view(env) -> None:
    """Best-effort camera placement to keep the main robot in view (env0).

    For headless + RecordVideo runs, the default camera is often not pointed at the robot.
    This helper tries common scene keys first, then falls back to the first articulation-like
    entity that exposes root pose data.
    """
    try:
        sim = env.unwrapped.sim
        scene = getattr(env.unwrapped, "scene", None)
        if scene is None:
            return

        # Try common keys first.
        robot = None
        for k in ("robot", "anymal", "agent"):
            try:
                robot = scene[k]
                break
            except Exception:
                robot = None

        # Fall back to first entity that looks like an articulation with root position.
        if robot is None:
            for k in scene.keys():
                if k in ("terrain",):
                    continue
                try:
                    ent = scene[k]
                    data = getattr(ent, "data", None)
                    if data is not None and hasattr(data, "root_pos_w"):
                        robot = ent
                        break
                except Exception:
                    continue

        if robot is None:
            return

        pos = robot.data.root_pos_w[0].detach()
        target = [float(pos[0]), float(pos[1]), float(pos[2])]
        eye = [target[0] + 6.0, target[1] + 6.0, target[2] + 6.0]
        sim.set_camera_view(eye, target)
    except Exception:
        return


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # handle deprecated configurations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # export the trained policy to JIT and ONNX formats
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    if version.parse(installed_version) >= version.parse("4.0.0"):
        # use the new export functions for rsl-rl >= 4.0.0
        runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
        runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
    else:
        # extract the neural network for rsl-rl < 4.0.0
        if version.parse(installed_version) >= version.parse("2.3.0"):
            policy_nn = runner.alg.policy
        else:
            policy_nn = runner.alg.actor_critic

        # extract the normalizer
        if hasattr(policy_nn, "actor_obs_normalizer"):
            normalizer = policy_nn.actor_obs_normalizer
        elif hasattr(policy_nn, "student_obs_normalizer"):
            normalizer = policy_nn.student_obs_normalizer
        else:
            normalizer = None

        # export to JIT and ONNX
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # optional residual reference trajectory (joint position targets)
    ref_traj_compiled = None
    ref_action_scale = None
    ref_action_offset = None
    joint_names_term = None
    if args_cli.ref_traj is not None:
        ref_traj_path = retrieve_file_path(args_cli.ref_traj)
        with open(ref_traj_path, encoding="utf-8") as f:
            ref_traj_cfg = yaml.safe_load(f)
        if not isinstance(ref_traj_cfg, dict) or "joints" not in ref_traj_cfg:
            raise ValueError(f"Invalid ref_traj YAML format: {ref_traj_path}. Expected top-level key: 'joints'.")

        # resolve action term scale + default joint positions
        action_term = env.unwrapped.action_manager.get_term(args_cli.ref_traj_action_term)
        # JointAction stores scale/offset as tensors shaped (num_envs, action_dim)
        if not hasattr(action_term, "_scale") or not hasattr(action_term, "_offset") or not hasattr(action_term, "_joint_ids"):
            raise ValueError(
                f"Action term '{args_cli.ref_traj_action_term}' does not look like a joint action term."
            )
        ref_action_scale = action_term._scale  # noqa: SLF001
        ref_action_offset = action_term._offset  # noqa: SLF001  # default joint positions when use_default_offset=True
        joint_ids = action_term._joint_ids  # noqa: SLF001

        joint_names_all = env.unwrapped.scene["robot"].data.joint_names
        # _joint_ids can be slice(None) when the term controls all joints (see isaaclab JointAction)
        if isinstance(joint_ids, slice):
            indices = list(range(len(joint_names_all)))[joint_ids]
        else:
            indices = list(joint_ids)
        joint_names_term = [joint_names_all[i] for i in indices]

        # pre-compile patterns
        compiled = []
        for item in ref_traj_cfg.get("joints", []):
            pattern = item.get("pattern")
            if not pattern:
                continue
            compiled.append(
                (
                    re.compile(pattern),
                    float(item.get("amplitude", 0.0)),
                    float(item.get("frequency_hz", 1.0)),
                    float(item.get("phase_rad", 0.0)),
                    float(item.get("bias", 0.0)),
                )
            )
        if len(compiled) == 0:
            raise ValueError(f"ref_traj YAML contains no valid joint patterns: {ref_traj_path}")
        ref_traj_compiled = compiled

    # reset environment
    obs = env.get_observations()
    if args_cli.video:
        _maybe_set_camera_view(env)
    step_count = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)

            # residual reference mode: action = policy + (ref_joint_pos - default_joint_pos)/scale
            if ref_traj_compiled is not None:
                t = step_count * dt
                # delta around default joint positions (same shape as action term)
                delta = torch.zeros_like(ref_action_offset)
                for (pat, amp, freq_hz, phase, bias) in ref_traj_compiled:
                    if amp == 0.0 and bias == 0.0:
                        continue
                    v = bias + amp * math.sin(2.0 * math.pi * freq_hz * t + phase)
                    if v == 0.0:
                        continue
                    for j, name in enumerate(joint_names_term):
                        if pat.search(name) is not None:
                            delta[:, j] += v

                # convert desired joint position target (default + delta) to raw action space:
                # processed = raw * scale + offset(default)
                # want processed = default + delta + (policy*scale)  => raw = (delta/scale) + policy
                if isinstance(ref_action_scale, torch.Tensor):
                    ref_raw = torch.where(
                        ref_action_scale != 0.0, delta / ref_action_scale, torch.zeros_like(delta)
                    )
                else:
                    ref_raw = delta / ref_action_scale if ref_action_scale != 0.0 else torch.zeros_like(delta)
                actions = torch.clamp(ref_raw + actions * float(args_cli.ref_traj_residual_scale), -1.0, 1.0)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)
            else:
                policy_nn.reset(dones)

        step_count += 1
        if args_cli.video:
            if step_count % 40 == 0:  #10
                _maybe_set_camera_view(env)
            # Exit the play loop after recording one video
            if step_count == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
