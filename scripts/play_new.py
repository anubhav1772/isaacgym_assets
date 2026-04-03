import isaacgym

assert isaacgym
import torch
import numpy as np

import glob
import pickle as pkl

from aliengo_gym.envs import *
from aliengo_gym.envs.base.legged_robot_config import BaseCfg
# from aliengo_gym.envs.aliengo.aliengo_config import config_aliengo
from aliengo_gym.envs.aliengo.velocity_tracking import VelocityTrackingEasyEnv
from aliengo_gym.utils.helpers import dict_to_env_cfg
from aliengo_gym.utils.logger import Logger

from tqdm import tqdm
from argparse import ArgumentParser
from scipy.spatial.transform import Rotation as R
import yaml
import os

def load_policy(logdir):
    body = torch.jit.load(logdir + '/checkpoints/body_latest.jit')
    import os
    adaptation_module = torch.jit.load(logdir + '/checkpoints/adaptation_module_latest.jit')

    def policy(obs, info={}):
        i = 0
        latent = adaptation_module.forward(obs["obs_history"].to('cpu'))
        action = body.forward(torch.cat((obs["obs_history"].to('cpu'), latent), dim=-1))
        info['latent'] = latent
        return action

    return policy

def load_env(logdir, headless=False):
    # logdir ="/home/anubhav1772/Documents/TEST/wtw-aliengo/runs/gait-conditioned-agility/2026-02-28/train/113800.235972"
    # logdir ="/home/anubhav1772/Documents/TEST/wtw-aliengo/runs/gait-conditioned-agility/2026-02-16/train/095453.580913"
    logdir ="/home/anubhav1772/Documents/TEST/wtw-aliengo/runs/gait-conditioned-agility/2026-03-20/train/113739.771308"
    # logdir ="/home/anubhav1772/Documents/TEST/wtw-aliengo/runs/gait-conditioned-agility/2025-10-21/train/022109.278720"
    with open(logdir + "/parameters.pkl", 'rb') as file:
        pkl_cfg = pkl.load(file)
        print(pkl_cfg.keys())
        cfg = pkl_cfg["Cfg"]
        # print(cfg.keys())
        # print(cfg.values())

        for key, value in cfg.items():
            if hasattr(BaseCfg, key):
                for key2, value2 in cfg[key].items():
                    setattr(getattr(BaseCfg, key), key2, value2)

    # turn off DR for evaluation script
    BaseCfg.env.num_recording_envs = 1
    BaseCfg.env.num_envs = 1
    BaseCfg.terrain.num_rows = 2 #1
    BaseCfg.terrain.num_cols = 1
    BaseCfg.terrain.terrain_length = 10. #15
    BaseCfg.terrain.terrain_width = 10. #15
    BaseCfg.terrain.border_size = 0
    BaseCfg.terrain.center_robots = True
    BaseCfg.terrain.center_span = 1
    BaseCfg.terrain.teleport_robots = True

    BaseCfg.commands.command_curriculum = False

    BaseCfg.domain_rand.push_robots = False
    BaseCfg.domain_rand.randomize_friction = False
    BaseCfg.domain_rand.randomize_gravity = False
    BaseCfg.domain_rand.randomize_restitution = False
    BaseCfg.domain_rand.randomize_motor_offset = False
    BaseCfg.domain_rand.randomize_motor_strength = False
    BaseCfg.domain_rand.randomize_friction_indep = False
    BaseCfg.domain_rand.randomize_ground_friction = False
    BaseCfg.domain_rand.randomize_base_mass = False
    BaseCfg.domain_rand.randomize_Kd_factor = False
    BaseCfg.domain_rand.randomize_Kp_factor = False
    BaseCfg.domain_rand.randomize_joint_friction = False
    BaseCfg.domain_rand.randomize_com_displacement = False

    BaseCfg.terrain.curriculum = False
    BaseCfg.domain_rand.push_robots = False

    BaseCfg.domain_rand.lag_timesteps = 6
    BaseCfg.domain_rand.randomize_lag_timesteps = True
    BaseCfg.control.control_type = "P"

    from aliengo_gym.envs.wrappers.history_wrapper import HistoryWrapper

    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=headless, cfg=BaseCfg)
    env = HistoryWrapper(env)

    # load policy
    from ml_logger import logger
    from aliengo_gym_learn.ppo_cse.actor_critic import ActorCritic

    policy = load_policy(logdir)

    return env, policy


def play_aliengo(model_dir, lin_x_speed, yaw_speed, headless=True, device='cuda:0'):
    from ml_logger import logger

    from pathlib import Path
    from aliengo_gym import MINI_GYM_ROOT_DIR
    import glob
    import os

    env, policy = load_env(model_dir, headless=headless)

    model_dir = f"{MINI_GYM_ROOT_DIR}/{model_dir}"
    os.makedirs(os.path.join(model_dir, "plots"), exist_ok=True)
    for env_idx in range(BaseCfg.env.num_envs):
        os.makedirs(os.path.join(os.path.join(model_dir, "plots"), f"env{env_idx}"), exist_ok=True)

    num_eval_steps = 500
    gaits = {"pronking": [0, 0, 0],
             "trotting": [0.5, 0, 0],
             "bounding": [0, 0.5, 0],
             "pacing": [0, 0, 0.5],
             "galloping": [0.25, 0, 0],
             "walking": [0.25, 0, 0.5]}

    x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 1.4, 0.0, 0.0
    body_height_cmd = 0.0
    step_frequency_cmd = 3.0
    gait = torch.tensor(gaits["trotting"])
    duration_cmd = 0.5
    footswing_height_cmd = 0.08
    pitch_cmd = 0.0
    roll_cmd = 0.0
    stance_width_cmd = 0.25
    stance_length_cmd = 0.40

    # x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 1.480125, 0.471418, 0.846521
    #
    # # body + gait parameters
    # body_height_cmd = -0.069051
    # step_frequency_cmd = 3.717114
    # duration_cmd = 0.457293
    # gait = torch.tensor(gaits["trotting"])
    # footswing_height_cmd = 0.26364
    # pitch_cmd = 0.016081
    # roll_cmd = 0.0
    #
    # stance_width_cmd = 0.37219
    # stance_length_cmd = 0.439592

    measured_x_vels = np.zeros(num_eval_steps)
    # target_x_vels = np.ones(num_eval_steps) * x_vel_cmd

    measured_y_vels = np.zeros(num_eval_steps)
    target_y_vels = np.ones(num_eval_steps) * y_vel_cmd

    measured_yaw_rate = np.zeros(num_eval_steps)
    target_yaw_rate = np.ones(num_eval_steps) * yaw_vel_cmd
    target_x_vels = []
    joint_positions = np.zeros((num_eval_steps, 12))

    obs = env.reset()
    base_env = env.env
    follow_cam = False #not headless

    WARMUP_STEPS = int(0.25 * num_eval_steps)
    distance = 0
    energy = 0

    for i in tqdm(range(num_eval_steps)):
        x_vel_cmd = lin_x_speed * i / num_eval_steps
        target_x_vels.append(x_vel_cmd)

        with torch.no_grad():
            actions = policy(obs)
            # actions = torch.zeros(1, 12)
        env.commands[:, 0] = x_vel_cmd
        env.commands[:, 1] = y_vel_cmd
        env.commands[:, 2] = yaw_vel_cmd
        env.commands[:, 3] = body_height_cmd
        env.commands[:, 5:8] = gait
        env.commands[:, 4] = step_frequency_cmd
        env.commands[:, 8] = duration_cmd
        env.commands[:, 9] = footswing_height_cmd
        env.commands[:, 10] = pitch_cmd
        env.commands[:, 11] = roll_cmd
        env.commands[:, 12] = stance_width_cmd
        env.commands[:, 13] = stance_length_cmd

        obs, rew, done, info = env.step(actions)

        torque = env.torques
        vel = env.dof_vel
        power = torch.abs(torque * vel).sum(dim=1)

        # if i < WARMUP_STEPS:
        #     continue

        energy += power * env.dt

        vel_xy = env.base_lin_vel[:, :2]
        speed = torch.norm(vel_xy, dim=1)
        distance += speed * env.dt

        measured_x_vels[i] = env.base_lin_vel[0, 0]
        measured_y_vels[i] = env.base_lin_vel[0, 1]
        measured_yaw_rate[i] = env.base_ang_vel[0, 2]
        joint_positions[i] = env.dof_pos[0, :].cpu()

    print(f"CoT (XY): {energy / (21.5 * 9.81 * distance)}")

    # plot target and measured forward velocity
    from matplotlib import pyplot as plt
    fig, axs = plt.subplots(3, 1, figsize=(12, 5))
    axs[0].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), measured_x_vels, color='black', linestyle="-", label="Measured")
    axs[0].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), np.array(target_x_vels), color='black', linestyle="--", label="Desired")
    axs[0].legend()
    axs[0].set_title("Forward Linear Velocity")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Velocity (m/s)")

    axs[1].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), measured_y_vels, color='black', linestyle="-", label="Measured")
    axs[1].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), np.array(target_y_vels), color='black', linestyle="--", label="Desired")
    axs[1].legend()
    axs[1].set_title("Lateral Linear Velocity")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Velocity (m/s)")

    axs[2].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), measured_yaw_rate, color='black', linestyle="-", label="Measured")
    axs[2].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), np.array(target_yaw_rate), color='black', linestyle="--", label="Desired")
    axs[2].legend()
    axs[2].set_title("Yaw Rate about Z")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Yaw Rate (rad/s)")

    # axs[3].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), joint_positions, linestyle="-", label="Measured")
    # axs[3].set_title("Joint Positions")
    # axs[3].set_xlabel("Time (s)")
    # axs[3].set_ylabel("Joint Position (rad)")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser(description="Evaluation script for WTW locomotion policy!")
    # model_dir is the relative path starting from this repo root.
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--model_dir", type=str,
                        default="runs/gait-conditioned-agility/2025-10-21/train/022109.278720",
                        help="num_obs = 58")
    parser.add_argument("--headless", default=False)
    parser.add_argument("--lin_speed", type=float, default=2.0)
    parser.add_argument("--ang_speed", type=float, default=0.0)
    parser.add_argument("--terrain_choice", type=str, default="flat")
    parser.add_argument("--terrain_diff", type=float, default=0.1)
    args = parser.parse_args()

    # label = f"pronking_{args.lin_speed}_{args.ang_speed}"

    play_aliengo(
        model_dir=args.model_dir,
        # label=label,
        lin_x_speed=args.lin_speed,
        yaw_speed=args.ang_speed,
        headless=args.headless,
        device="cuda:{}".format(args.device)
    )
