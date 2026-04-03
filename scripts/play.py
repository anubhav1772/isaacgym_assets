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
    logdir ="/home/anubhav1772/Documents/TEST/wtw-aliengo/runs/gait-conditioned-agility/2026-02-15/train/133633.404635"
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

    BaseCfg.env.num_recording_envs = 1
    BaseCfg.env.num_envs = 1
    BaseCfg.terrain.num_rows = 5
    BaseCfg.terrain.num_cols = 5
    BaseCfg.terrain.border_size = 0
    BaseCfg.terrain.center_robots = True
    BaseCfg.terrain.center_span = 1
    BaseCfg.terrain.teleport_robots = True

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

    num_eval_steps = 250
    gaits = {"pronking": [0, 0, 0],
             "trotting": [0.5, 0, 0],
             "bounding": [0, 0.5, 0],
             "pacing": [0, 0, 0.5],
             "galloping": [0.25, 0, 0]}

    gait_labels = ['pronking', 'trotting', 'bounding', 'pacing', 'galloping']

    x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.0, 0.0, 0.0
    body_height_cmd = 0.0
    step_frequency_cmd = 3.0
    # gait = torch.tensor(gaits["bounding"])
    footswing_height_cmd = 0.08
    pitch_cmd = 0.0
    roll_cmd = 0.0
    stance_width_cmd = 0.25
    # stance_length_cmd = 0.45

    measured_x_vels = np.zeros(num_eval_steps)
    target_x_vels = np.ones(num_eval_steps) * x_vel_cmd
    joint_positions = np.zeros((num_eval_steps, 12))

    # logger = Logger(env.env.dt, env.env.num_envs, env.env.dof_names, env.feet_names, lin_x_speed, yaw_speed, os.path.join(model_dir, "plots"), 200, device)

    obs = env.reset()
    base_env = env.env
    follow_cam = False #not headless

    for i in tqdm(range(num_eval_steps)):
        #x_vel_cmd = 1.8 * i / num_eval_steps

        with torch.no_grad():
            actions = policy(obs)
            # actions = torch.zeros(1, 12)
        # env.commands[:, 0] = x_vel_cmd
        # env.commands[:, 1] = y_vel_cmd
        # env.commands[:, 2] = yaw_vel_cmd
        # env.commands[:, 3] = body_height_cmd
        # env.commands[:, 4] = step_frequency_cmd
        # env.commands[:, 5:8] = gait
        # env.commands[:, 8] = 0.5
        # env.commands[:, 9] = footswing_height_cmd
        # env.commands[:, 10] = pitch_cmd
        # env.commands[:, 11] = roll_cmd
        # env.commands[:, 12] = stance_width_cmd
        # env.commands[:, 13] = stance_length_cmd

        env.commands[:, 0] = x_vel_cmd
        env.commands[:, 1] = y_vel_cmd
        env.commands[:, 2] = yaw_vel_cmd
        env.commands[:, 3] = body_height_cmd
        env.commands[:, 4] = step_frequency_cmd
        #env.commands[0, 5:8] = torch.tensor(gaits["pronking"])
        #env.commands[1, 5:8] = torch.tensor(gaits["trotting"])
        env.commands[0, 5:8] = torch.tensor(gaits["trotting"])
        #env.commands[3, 5:8] = torch.tensor(gaits["pacing"])
        #env.commands[4, 5:8] = torch.tensor(gaits["galloping"])
        env.commands[:, 8] = 1.0
        env.commands[:, 9] = footswing_height_cmd
        env.commands[:, 10] = pitch_cmd
        env.commands[:, 11] = roll_cmd
        env.commands[:, 12] = stance_width_cmd

        obs, rew, done, info = env.step(actions)

        # Set camera to follow the robot
        # if follow_cam and hasattr(base_env, "set_camera"):
        #     pos = base_env.base_pos[0].cpu().numpy()
        #     cam_pos = pos + np.array([-3.0, -3.0, 3.0])  # offset behind and above the robot
        #     base_env.set_camera(cam_pos, pos)

        # if i >= 0 and i <= 250:
        #     for env_idx in range(BaseCfg.env.num_envs):
        #         log_dict = {
        #             'command_x': env.env.commands[env_idx:env_idx+1, 0].cpu().numpy(),
        #             'command_y': env.env.commands[env_idx:env_idx+1, 1].cpu().numpy(),
        #             'command_yaw': env.env.commands[env_idx:env_idx+1, 2].cpu().numpy(),
        #         }
        #         log_dict['x_vel_cmd'] = x_vel_cmd
        #         log_dict['base_pos_x'] = env.env.base_pos[env_idx:env_idx+1, 0].cpu().numpy()
        #         log_dict['base_pos_y'] = env.env.base_pos[env_idx:env_idx+1, 1].cpu().numpy()
        #         log_dict['base_pos_z'] = env.env.base_pos[env_idx:env_idx+1, 2].cpu().numpy()
        #         log_dict['base_quat_x'] = env.env.base_quat[env_idx:env_idx+1, 0].cpu().numpy()
        #         log_dict['base_quat_y'] = env.env.base_quat[env_idx:env_idx+1, 1].cpu().numpy()
        #         log_dict['base_quat_z'] = env.env.base_quat[env_idx:env_idx+1, 2].cpu().numpy()
        #         log_dict['base_quat_w'] = env.env.base_quat[env_idx:env_idx+1, 3].cpu().numpy()
        #         log_dict['base_vel_x'] = env.env.base_lin_vel[env_idx:env_idx+1, 0].cpu().numpy()
        #         log_dict['base_vel_y'] = env.env.base_lin_vel[env_idx:env_idx+1, 1].cpu().numpy()
        #         log_dict['base_vel_z'] = env.env.base_lin_vel[env_idx:env_idx+1, 2].cpu().numpy()
        #         log_dict['base_vel_roll'] = env.env.base_ang_vel[env_idx:env_idx+1, 0].cpu().numpy()
        #         log_dict['base_vel_pitch'] = env.env.base_ang_vel[env_idx:env_idx+1, 1].cpu().numpy()
        #         log_dict['base_vel_yaw'] = env.env.base_ang_vel[env_idx:env_idx+1, 2].cpu().numpy()
        #         log_dict['contact_forces_x'] = env.env.contact_forces[env_idx:env_idx+1, env.env.feet_indices, 0].cpu().numpy()
        #         log_dict['contact_forces_y'] = env.env.contact_forces[env_idx:env_idx+1, env.env.feet_indices, 1].cpu().numpy()
        #         log_dict['contact_forces_z'] = env.env.contact_forces[env_idx:env_idx+1, env.env.feet_indices, 2].cpu().numpy()
        #         log_dict['reward'] = env.env.rew_buf[env_idx:env_idx+1].detach().clone().cpu().numpy()
        #         log_dict['energy_consume'] = env.env.energy_consume[env_idx:env_idx+1].cpu().numpy()
        #         log_dict['cot'] = env.env.cot[env_idx:env_idx+1].cpu().numpy()
        #         log_dict['dof_pos'] = env.env.dof_pos.cpu().numpy()[env_idx:env_idx+1]
        #         log_dict['dof_vel'] = env.env.dof_vel.cpu().numpy()[env_idx:env_idx+1]
        #         log_dict['dof_acc'] = env.env.dof_acc.cpu().numpy()[env_idx:env_idx+1]
        #         log_dict['dof_torque'] = env.env.torques.detach().clone().cpu().numpy()[env_idx:env_idx+1]
        #         log_dict['action_scaled'] = env.env.actions.detach().clone().cpu().numpy()[env_idx:env_idx+1] # * env_cfg.control.action_scale
        #
        #         # Convert quaternion into yaw angle.
        #         base_quat_x = env.env.base_quat[env_idx:env_idx+1, 0].cpu().numpy()
        #         base_quat_y = env.env.base_quat[env_idx:env_idx+1, 1].cpu().numpy()
        #         base_quat_z = env.env.base_quat[env_idx:env_idx+1, 2].cpu().numpy()
        #         base_quat_w = env.env.base_quat[env_idx:env_idx+1, 3].cpu().numpy()
        #
        #
        #         rotmat = R.from_quat(np.stack([base_quat_x, base_quat_y, base_quat_z, base_quat_w], axis=1)).as_matrix()
        #         headdir = rotmat @ np.array([[1.0], [0.0], [0.0]])
        #         log_dict['base_pos_yaw'] = np.arctan2(headdir[:, 1, 0], headdir[:, 0, 0])

                # for key, value in log_dict.items():
                #     print(f"{key}({value.shape})==>>{value}")

                #logger.log_states(log_dict, env_idx)

        measured_x_vels[i] = env.base_lin_vel[0, 0]
        joint_positions[i] = env.dof_pos[0, :].cpu()

    #logger.plot_save_states_multi(gait_labels, f"{lin_x_speed}_{yaw_speed}", [0])

    # plot target and measured forward velocity
    from matplotlib import pyplot as plt
    fig, axs = plt.subplots(2, 1, figsize=(12, 5))
    axs[0].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), measured_x_vels, color='black', linestyle="-", label="Measured")
    axs[0].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), target_x_vels, color='black', linestyle="--", label="Desired")
    axs[0].legend()
    axs[0].set_title("Forward Linear Velocity")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Velocity (m/s)")

    axs[1].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), joint_positions, linestyle="-", label="Measured")
    axs[1].set_title("Joint Positions")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Joint Position (rad)")

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
    parser.add_argument("--lin_speed", type=float, default=1.0)
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
