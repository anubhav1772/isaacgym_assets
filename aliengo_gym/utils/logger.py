import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import itertools
import os
import pickle
from scipy.spatial.transform import Rotation as R
import yaml
import torch

class Logger:
    def __init__(self, dt, num_envs, dof_names, feet_names, lin_speed, ang_speed, model_dir, load_iteration, device):
        self.state_log = defaultdict(list)
        self.dt = dt
        self.num_episodes = 0
        self.feet_names = feet_names
        self.dof_names = dof_names
        self.lin_speed = lin_speed
        self.ang_speed = ang_speed
        self.model_dir = model_dir
        self.load_iteration = load_iteration
        # self.label = label # subfolder label
        self.num_envs = num_envs
        self.device = device

    def log_state(self, key, value):
        """Log state for all environments at once"""
        self.state_log[key].append(value)

    def log_states(self, dict, env_idx=None):
        """Log states for a specific environment or all environments"""
        if env_idx is not None:
            for key, value in dict.items():
                if key not in self.state_log:
                    self.state_log[key] = []
                if len(self.state_log[key]) == 0:
                    self.state_log[key] = [[] for _ in range(self.num_envs)]

                self.state_log[key][env_idx].append(value)
        else:
            for key, value in dict.items():
                self.log_state(key, value)

    def reset(self):
        self.state_log.clear()
        self.rew_log.clear()

    def get_vel_x_stats(self, env_idx=None) -> (float, float):
        """Compute the x-direction average velocity in robot frame (by time).
        Averaged over all sub environments.
        Return the two values, mean velocity (by time) and the sqrt sum square error.
        """
        if env_idx is not None:
            command_vel_x = np.array(self.state_log["command_x"][env_idx])
            base_vel_x = np.array(self.state_log["base_vel_x"][env_idx])
        else:
            command_vel_x = np.array(self.state_log["command_x"])
            base_vel_x = np.array(self.state_log["base_vel_x"])
        error_vel_x_square = np.mean(np.mean(np.square(base_vel_x - command_vel_x), axis=0), axis=0)
        base_vel_x_mean = np.mean(base_vel_x, axis=0)
        return np.mean(base_vel_x_mean).item(), np.sqrt(error_vel_x_square).item()

    def get_vel_x(self, robot_index: int, env_idx: None) -> float:
        """Compute the x-direction average velocity in robot frame (by time).
        For specified sub environment.
        """
        if env_id is not None:
            base_vel = np.array(self.state_log["base_vel_x"][env_idx])
        else:
            base_vel = np.array(self.state_log["base_vel_x"])
        return np.mean(base_vel[:, robot_index]).item()

    def get_vel_y_stats(self, env_idx=None) -> (float, float):
        """Compute the y-direction average velocity in robot frame (by time).
        Averaged over all sub environments.
        Return the two values, mean velocity by time and the sqrt sum square error.
        """
        if env_idx is not None:
            command_vel_y = np.array(self.state_log["command_y"][env_idx])
            base_vel_y = np.array(self.state_log["base_vel_y"][env_idx])
        else:
            command_vel_y = np.array(self.state_log["command_y"])
            base_vel_y = np.array(self.state_log["base_vel_y"])
        error_vel_y_square = np.mean(np.mean(np.square(base_vel_y - command_vel_y), axis=0), axis=0)
        base_vel_y_mean = np.mean(base_vel_y, axis=0)
        return np.mean(base_vel_y_mean).item(), np.sqrt(error_vel_y_square).item()

    def get_vel_y(self, robot_index: int, env_idx: None) -> float:
        """Compute the y-direction average velocity in robot frame (by time).
        For specified sub environment.
        """
        if env_idx is not None:
            base_vel = np.array(self.state_log["base_vel_y"][env_idx])
        else:
            base_vel = np.array(self.state_log["base_vel_y"])
        return np.mean(base_vel[:, robot_index]).item()

    def get_vel_yaw_stats(self, env_idx=None) -> (float, float):
        """Compute the y-direction average velocity in robot frame (by time).
        Averaged over all sub environments.
        Return two values, mean yaw velocity by time and the sqrt sum square error.
        """
        if env_idx is not None:
            command_vel_yaw = np.array(self.state_log["command_yaw"][env_idx])
            base_vel_yaw = np.array(self.state_log["base_vel_yaw"][env_idx])
        else:
            command_vel_yaw = np.array(self.state_log["command_yaw"])
            base_vel_yaw = np.array(self.state_log["base_vel_yaw"])
        error_vel_yaw_square = np.mean(np.mean(np.square(base_vel_yaw - command_vel_yaw), axis=0), axis=0)
        base_vel_yaw_mean = np.mean(base_vel_yaw, axis=0)
        return np.mean(base_vel_yaw_mean).item(), np.sqrt(error_vel_yaw_square).item()

    def get_vel_yaw(self, robot_index: int, env_idx: None) -> float:
        """Compute the yaw-direction average velocity in robot frame (by time).
        For specified sub environment.
        """
        if env_idx is not None:
            base_vel = np.array(self.state_log["base_vel_yaw"][env_idx])
        else:
            base_vel = np.array(self.state_log["base_vel_yaw"])
        return np.mean(base_vel[: robot_index]).item()

    def get_energy_consume_watts_stats(self, env_idx=None) -> (float, float):
        """Compute the average energy consumption (by time) of this test.
        Averaged over all sub environments.
        Return the two values, mean consumed energy by time and its variance.
        """
        if env_idx is not None:
            energy_consume_watts = np.array(self.state_log["energy_consume"][env_idx])
        else:
            energy_consume_watts = np.array(self.state_log["energy_consume"])
        return np.mean(np.mean(energy_consume_watts, axis=0)).item(), np.std(np.mean(energy_consume_watts, axis=0)).item()

    def get_energy_consume_watts(self, robot_index: int, env_idx: int) -> float:
        """Compute the average energy consumption (by time).
        For specified sub environment.
        """
        energy_consume_watts = np.array(self.state_log["energy_consume"][env_idx])
        return np.mean(energy_consume_watts[:, robot_index], axis=0).item()

    def get_energy_consume_dist_stats(self, env_idx=None):
        """Compute the average energy consumption (by distance) of this test.
        Averaged over all sub environments.
        Return the two values, mean consumed energy by distance and its variance.
        """
        m = 21.5 # kg
        g = 9.81 # ms^-2
        if env_idx is not None:
            energy_consume = np.array(self.state_log["energy_consume"])[env_idx] * self.dt
            base_pos_x = np.array(self.state_log["base_pos_x"][env_idx])
            base_pos_y = np.array(self.state_log["base_pos_y"][env_idx])
            base_pos_yaw = np.array(self.state_log["base_pos_yaw"][env_idx])
        else:
            energy_consume = np.sum(np.array(self.state_log["energy_consume"][env_idx]) * self.dt, axis=0)
            base_pos_x = np.array(self.state_log["base_pos_x"])
            base_pos_y = np.array(self.state_log["base_pos_y"])
            base_pos_yaw = np.array(self.state_log["base_pos_yaw"])
        amount_moved = np.zeros(base_pos_x.shape[1])
        for i in range(base_pos_x.shape[0] - 1):
            if np.sqrt(np.square(base_pos_x[i + 1] - base_pos_x[i]) + np.square(base_pos_y[i + 1] - base_pos_y[i])) > 1.0:
                continue
            if np.abs(base_pos_yaw[i + 1] - base_pos_yaw[i]) > np.pi / 2.0:
                continue
            amount_moved = amount_moved + np.sqrt(np.square(base_pos_x[i + 1] - base_pos_x[i]) + np.square(base_pos_y[i + 1] - base_pos_y[i]))
            amount_moved = amount_moved + 0.5 * np.abs(base_pos_yaw[i + 1] - base_pos_yaw[i])

        # amount_moved[amount_moved <= 0.001] = 0.001
        # energy_consume_dist = energy_consume / (m * g * amount_moved)

        E_cum = np.cumsum(energy_consume)
        d_cum = amount_moved.copy()
        d_cum[d_cum < 1e-3] = 1e-3

        cot = E_cum / (m * g * d_cum)
        return cot, cot[-1], np.std(cot), energy_consume / (m * g * d_cum)
        # return energy_consume_dist, np.mean(energy_consume_dist).item(), np.std(energy_consume_dist).item()

    # def get_mean_energy_consumed_by_dist(self, env_idx: int) -> (float, float):
    #     """Compute normalized mean cost of transport (E / mgd) for a given environment.
    #     Returns mean and standard deviation over the evaluation horizon.
    #     """
    #     m = 21.5 # kg
    #     g = 9.81 # ms^-2
    #     energy_consume = np.array(self.state_log["energy_consume"][env_idx])
    #     base_pos_x = np.array(self.state_log["base_pos_x"][env_idx])
    #     base_pos_y = np.array(self.state_log["base_pos_y"][env_idx])
    #     base_pos_yaw = np.array(self.state_log["base_pos_yaw"][env_idx])
    #
    #     T = energy_consume.shape[0]
    #     warmup_frac = 0.25                 # discard first 25% of steps
    #     start_idx = int(T * warmup_frac)
    #
    #     # Slice to keep only the steady-state part
    #     energy_consume = energy_consume[start_idx:]
    #     base_pos_x     = base_pos_x[start_idx:]
    #     base_pos_y     = base_pos_y[start_idx:]
    #     base_pos_yaw   = base_pos_yaw[start_idx:]
    #
    #     E_total = np.sum(energy_consume * self.dt, axis=0)   # shape (num_dofs or 1,)
    #
    #     # Distance moved in XY (no yaw term)
    #     amount_moved = np.zeros(base_pos_x.shape[1])
    #     for i in range(base_pos_x.shape[0] - 1):
    #         dx = base_pos_x[i + 1] - base_pos_x[i]
    #         dy = base_pos_y[i + 1] - base_pos_y[i]
    #         # same teleport / large-turn filters as before
    #         if np.sqrt(dx**2 + dy**2) > 1.0:
    #             continue
    #         if np.abs(base_pos_yaw[i + 1] - base_pos_yaw[i]) > np.pi / 2.0:
    #             continue
    #         amount_moved += np.sqrt(dx**2 + dy**2)
    #
    #     amount_moved[amount_moved <= 1e-3] = 1e-3
    #     cot = E_total / (m * g * amount_moved)              # J / (m g d), dimensionless
    #
    #     return np.mean(cot).item(), np.std(cot).item()

    def get_cot_by_vel(self, vel):
        m = 21.5 # kg
        g = 9.81 # ms^-2
        energy_consume = np.array(self.state_log["energy_consume"][env_idx])



    def get_mean_energy_consumed_by_dist(self, env_idx: None) -> (float, float):
        """Compute normalized mean cost of transport (E / mgd) for a given environment.
        Returns mean and standard deviation over the evaluation horizon.
        """
        m = 21.5 # kg
        g = 9.81 # ms^-2

        if env_idx is not None:
            energy_consume = np.array(self.state_log["energy_consume"][env_idx])
            base_pos_x = np.array(self.state_log["base_pos_x"][env_idx])
            base_pos_y = np.array(self.state_log["base_pos_y"][env_idx])
            base_pos_yaw = np.array(self.state_log["base_pos_yaw"][env_idx])
        else:
            energy_consume = np.array(self.state_log["energy_consume"])
            base_pos_x = np.array(self.state_log["base_pos_x"])
            base_pos_y = np.array(self.state_log["base_pos_y"])
            base_pos_yaw = np.array(self.state_log["base_pos_yaw"])

        T = energy_consume.shape[0]
        warmup_frac = 0.25                 # discard first 25% of steps
        start_idx = int(T * warmup_frac)

        # Slice to keep only the steady-state part
        energy_consume = energy_consume[start_idx:]
        base_pos_x     = base_pos_x[start_idx:]
        base_pos_y     = base_pos_y[start_idx:]
        base_pos_yaw   = base_pos_yaw[start_idx:]

        E_total = np.sum(energy_consume * self.dt, axis=0)   # shape (num_dofs or 1,)

        # Distance moved in XY (no yaw term)
        amount_moved = np.zeros(base_pos_x.shape[1])
        for i in range(base_pos_x.shape[0] - 1):
            dx = base_pos_x[i + 1] - base_pos_x[i]
            dy = base_pos_y[i + 1] - base_pos_y[i]
            # same teleport / large-turn fillters as before
            if np.sqrt(dx**2 + dy**2) > 1.0:
                continue
            if np.abs(base_pos_yaw[i + 1] - base_pos_yaw[i]) > np.pi / 2.0:
                continue
            amount_moved += np.sqrt(dx**2 + dy**2)

        amount_moved[amount_moved <= 1e-3] = 1e-3
        cot = E_total / (m * g * amount_moved)              # J / (m g d), dimensionless

        return np.mean(cot).item(), np.std(cot).item()

    def get_energy_consume_dist(self, robot_index: int, env_idx: int) -> float:
        """Compute the average energy consumption (by distance) of this test.
        For specific sub environment.
        """
        energy_consume = np.sum(np.array(self.state_log["energy_consume"][env_idx]) * self.dt, axis=0)
        base_pos_x = np.array(self.state_log["base_pos_x"][env_idx])
        base_pos_y = np.array(self.state_log["base_pos_y"][env_idx])
        distance_moved = np.zeros(base_pos_x.shape[1])
        for i in range(base_pos_x.shape[0] - 1):
            if np.sqrt(np.square(base_pos_x[i + 1] - base_pos_x[i]) + np.square(base_pos_y[i + 1] - base_pos_y[i])) > 1.0:
                continue
            distance_moved = distance_moved + np.sqrt(np.square(base_pos_x[i + 1] - base_pos_x[i]) + np.square(base_pos_y[i + 1] - base_pos_y[i]))

        distance_moved[distance_moved <= 0.001] = 0.001
        energy_consume_dist = energy_consume[robot_index] / max(distance_moved[robot_index], 0.001)
        return energy_consume_dist.item()

    def get_cost_of_transportation(self, env_idx=None) -> float:
        """Compute the average energy consumption (by distance) of this test.
        For specific sub environment.
        Unit: J/m
        """
        if env_idx is not None:
            energy_consume = np.array(self.state_log["energy_consume"])[env_idx] * self.dt
            base_pos_x = np.array(self.state_log["base_pos_x"][env_idx])
            base_pos_y = np.array(self.state_log["base_pos_y"][env_idx])
        distance_moved = np.zeros(base_pos_x.shape[1])
        for i in range(base_pos_x.shape[0] - 1):
            if np.sqrt(np.square(base_pos_x[i + 1] - base_pos_x[i]) + np.square(base_pos_y[i + 1] - base_pos_y[i])) > 1.0:
                continue
            distance_moved = distance_moved + np.sqrt(np.square(base_pos_x[i + 1] - base_pos_x[i]) + np.square(base_pos_y[i + 1] - base_pos_y[i]))

        distance_moved[distance_moved <= 0.001] = 0.001
        energy_consume_dist = energy_consume / max(distance_moved, 0.001)
        return energy_consume_dist # J/m

    def get_rewards_stats(self, env_idx=None) -> (float, float):
        """Get the average reward obtained from this trajectory.
        Averaged over all sub environments.
        """
        if env_idx is not None:
            rewards = np.array(self.state_log["reward"][env_idx])
        else:
            rewards = np.array(self.state_log["reward"])
        return np.mean(np.mean(rewards, axis=0)).item(), np.std(np.mean(rewards, axis=0)).item()

    def get_rewards(self, robot_index: int, env_idx: None) -> float:
        """Get the average reward obtained from this trajectory.
        Averaged over all sub environments.
        """
        if env_idx is not None:
            rewards = np.array(self.state_log["reward"][env_idx])
        else:
            rewards = np.array(self.state_log["reward"])
        return np.mean(rewards[:, robot_index]).item()

    # def plot_save_states(self, test_name: str, robot_indices: list):
    #     # self._save_logs(test_name)
    #     for idx in robot_indices:
    #         self._plot_velocities(test_name, idx)
    #         self._plot_dof_pos(test_name, idx)
    #         self._plot_dof_vels(test_name, idx)
    #         self._plot_dof_accs(test_name, idx)
    #         self._plot_dof_torques(test_name, idx)
    #         self._plot_actions(test_name, idx)
    #         self._plot_gaits(test_name, idx)
    #         self._plot_tracking(test_name, idx)
    #         # self._plot_energy_consumed(test_name, idx)
    #         # self._plot_cumulative_cot(test_name, idx)
    #         # self._plot_cot(test_name)

    def plot_save_states_multi(self, gait_labels: list, test_name: str, robot_indices: list):
        self._save_logs(test_name)
        for idx in robot_indices:
            # self._plot_velocities(test_name, idx)
            # self._plot_dof_pos(test_name, idx)
            # self._plot_dof_vels(test_name, idx)
            # self._plot_dof_accs(test_name, idx)
            # self._plot_dof_torques(test_name, idx)
            # self._plot_actions(test_name, idx)
            # self._plot_gaits(test_name, idx)
            # self._plot_tracking(test_name, idx)
            self._plot_energy_consumed(test_name, idx) # for both instantaneous & cumulative energy consumption
            # self._plot_cumulative_cot(test_name, idx)
            # self._plot_cot(test_name)

    def _save_logs(self, test_name: str):
        """Save the logging dictionary into the model directory.
        File name is "{test_name}_log_data_{self.load_iteration}.pkl".
        Also, save statistics data of energy_consume_watts, energy_consume_dists, rewards, vel_x and vel_y.
        File name is "{test_name}_statistics_{self.load_iteration}.yaml"
        """
        with open(os.path.join(self.model_dir, f"{test_name}_log_data_{self.load_iteration}.pkl"), 'wb') as file:
            pickle.dump(self.state_log, file)

        # Save data statistics
        self.dump_dict = {
            'num_envs': self.num_envs,
            'dt': self.dt,
            'total_steps': len(self.state_log.get('base_vel_x', [])),
            'environments': {}
        }

        for env_idx in range(self.num_envs):
            vel_x_mean, vel_x_std = self.get_vel_x_stats(env_idx)
            vel_y_mean, vel_y_std = self.get_vel_y_stats(env_idx)
            vel_yaw_mean, vel_yaw_std = self.get_vel_yaw_stats(env_idx)
            energy_consume_watts_mean, energy_consume_watts_std = self.get_energy_consume_watts_stats(env_idx)
            energy_consume_dist, energy_consume_dist_mean, energy_consume_dist_std, _ = self.get_energy_consume_dist_stats(env_idx)
            rewards_mean, rewards_std = self.get_rewards_stats(env_idx)

            # Environment-specific data
            self.dump_dict['environments'][f'env_{env_idx}'] = {
                'vel_x_mean': float(vel_x_mean),
                'vel_x_std': float(vel_x_std),
                'vel_y_mean': float(vel_y_mean),
                'vel_y_std': float(vel_y_std),
                'vel_yaw_mean': float(vel_yaw_mean),
                'vel_yaw_std': float(vel_yaw_std),
                'energy_consume_watts_mean': float(energy_consume_watts_mean),
                'energy_consume_watts_std': float(energy_consume_watts_std),
                'energy_consume_dist_mean': float(energy_consume_dist_mean),
                'energy_consume_dist_std': float(energy_consume_dist_std),
                'rewards_mean': float(rewards_mean),
                'rewards_std': float(rewards_std)
            }

        # Aggregated statistics across all environments
        if self.num_envs > 0:
            all_vel_x_means = [self.dump_dict['environments'][f'env_{i}']['vel_x_mean'] for i in range(self.num_envs)]
            all_vel_y_means = [self.dump_dict['environments'][f'env_{i}']['vel_y_mean'] for i in range(self.num_envs)]
            all_rewards = [self.dump_dict['environments'][f'env_{i}']['rewards_mean'] for i in range(self.num_envs)]
            all_energy = [self.dump_dict['environments'][f'env_{i}']['energy_consume_watts_mean'] for i in range(self.num_envs)]

            self.dump_dict['aggregated'] = {
                'vel_x_mean_mean': float(np.mean(all_vel_x_means)),
                'vel_x_mean_std': float(np.std(all_vel_x_means)),
                'vel_y_mean_mean': float(np.mean(all_vel_y_means)),
                'vel_y_mean_std': float(np.std(all_vel_y_means)),
                'rewards_mean_mean': float(np.mean(all_rewards)),
                'rewards_mean_std': float(np.std(all_rewards)),
                'energy_mean_mean': float(np.mean(all_energy)),
                'energy_mean_std': float(np.std(all_energy))
            }

        with open(os.path.join(self.model_dir, f"{test_name}_statistics_{self.load_iteration}.yaml"), 'w') as file:
            yaml.dump(self.dump_dict, file)

    def _plot_energy_consumed(self, test_name: str, robot_index: int):
        """Simple plot of CoT over time for all environments."""
        gait_names = {
            0: "Pronking",
            1: "Trotting",
            2: "Bounding",
            3: "Pacing",
            4: "Galloping"
        }

        plt.figure(figsize=(10, 6))

        for key, value in self.state_log.items():
            time = np.linspace(0, len(value[0])*self.dt, len(value[0]))
            break

        vel = np.array(self.state_log['x_vel_cmd'])[0]

        for env_idx in range(self.num_envs):
            # cot_over_time, _, _, cot_instant_over_time = self.get_energy_consume_dist_stats(env_idx)
            cot = self.get_cost_of_transportation(env_idx)

            print(cot)

            gait_name = gait_names.get(env_idx, f"Gait {env_idx}")
            # plt.plot(time, np.cumsum(cot_over_time[0]), label=gait_name)
            # plt.plot(time, cot_over_time, label=gait_name)              # Cumulative energy vs time
            # plt.plot(vel, cot_over_time, label=gait_name)               # Cumulative energy vs vel
            # plt.plot(time, cot_instant_over_time, label=gait_name)      # Instantaneous cot vs time
            # plt.plot(vel, cot_instant_over_time, label=gait_name)       # Instantaneous cot vs vel
            plt.plot(vel, cot, label=gait_name)                           # Cost of Transportation (J/m)

        # plt.xlabel('Time (s)')
        # plt.ylabel('Cumulative Normalized Energy')
        # plt.title(f"Cumulative Normalized Energy vs time (V_x = 1 m/s, V_y = 0.3 m/s, V_yaw = 0 rad/s)")
        # plt.xlabel('Velocity (m/s)')
        # plt.ylabel('Instantaneous CoT')
        # plt.title(f"Instantaneous CoT vs Vel (V_x = 1.5 m/s, V_y = 0.0 m/s, V_yaw = 0 rad/s)")
        plt.xlabel('Velocity (m/s)')
        plt.ylabel('Cost of Transportation (J/m)')
        plt.title(f"Cost of Transportation (J/m) vs Vel (V_x = 1.8 m/s, V_y = 0.0 m/s, V_yaw = 0 rad/s)")
        plt.legend()
        plt.grid(False, alpha=0.3)
        plt.show()


    def _plot_velocities(self, test_name: str, robot_index: int):
        """Plot velocity tracking and save it into the model directory.
        File name is "{test_name}_vel_plot_{self.load_iteration}.png"
        """
        nb_rows = 2
        nb_cols = 3
        for env_idx in range(self.num_envs):
            fig, axs = plt.subplots(nb_rows, nb_cols)
            fig.set_size_inches(12, 6)
            for key, value in self.state_log.items():
                time = np.linspace(0, len(value[0])*self.dt, len(value[0]))
                break
            log = self.state_log
            # print(log.keys())

            # plot base vel x
            a = axs[0, 0]
            if log["base_vel_x"][env_idx]:
                log_base_vel_x = np.array(log["base_vel_x"][env_idx])
                a.plot(time, log_base_vel_x[:, robot_index], label='measured')

            if log["command_x"][env_idx]:
                log_command_x = np.array(log["command_x"][env_idx])
                a.plot(time, log_command_x[:, robot_index], label='commanded')
            a.set_xlabel('Time [s]', fontsize=8)
            a.set_ylabel('Base lin vel [m/s]', fontsize=8)
            a.set_title('Base velocity x', fontsize=10)
            a.set_ylim([0, 3])
            a.legend()

            # plot base vel y
            a = axs[0, 1]
            if log["base_vel_y"][env_idx]:
                log_base_vel_y = np.array(log["base_vel_y"][env_idx])
                a.plot(time, log_base_vel_y[:, robot_index], label='measured')
            if log["command_y"][env_idx]:
                log_command_y = np.array(log["command_y"][env_idx])
                a.plot(time, log_command_y[:, robot_index], label='commanded')
            a.set_xlabel('Time [s]', fontsize=8)
            a.set_ylabel('Base lin vel [m/s]', fontsize=8)
            a.set_title('Base velocity y', fontsize=10)
            a.set_ylim([-1, 1])
            a.legend()

            # plot base vel z
            # a = axs[0, 2]
            # if log["base_vel_z"][env_idx]:
            #     log_base_vel_z = np.array(log["base_vel_z"][env_idx])
            #     a.plot(time, log_base_vel_z[:, robot_index], label='measured')
            # if log["command_z"]:
            #     log_command_z = np.array(log["command_z"][env_idx])
            #     a.plot(time, log_command_z[:, robot_index], label='commanded')
            # a.set_xlabel('Time [s]', fontsize=8)
            # a.set_ylabel('Base lin vel [m/s]', fontsize=8)
            # a.set_title('Base velocity z', fontsize=10)
            # a.set_ylim([-1, 1])
            # a.legend()

            # plot base vel yaw
            a = axs[1, 0]
            if log["base_vel_yaw"]:
                log_base_vel_yaw = np.array(log["base_vel_yaw"][env_idx])
                a.plot(time, log_base_vel_yaw[:, robot_index], label='measured')
            if log["command_yaw"]:
                log_command_yaw = np.array(log["command_yaw"][env_idx])
                a.plot(time, log_command_yaw[:, robot_index], label='commanded')
            a.set_xlabel('Time [s]', fontsize=8)
            a.set_ylabel('Base ang vel [rad/s]', fontsize=8)
            a.set_title('Base velocity yaw', fontsize=10)
            a.set_ylim([-3, 3])
            a.legend()

            # plot base vel pitch
            # a = axs[1, 1]
            # if log["base_vel_pitch"][env_idx]:
            #     log_base_vel_pitch = np.array(log["base_vel_pitch"][env_idx])
            #     a.plot(time, log_base_vel_pitch[:, robot_index], label='measured')
            # if log["command_pitch"][env_idx]:
            #     log_command_pitch = np.array(log["command_pitch"][env_idx])
            #     a.plot(time, log_command_pitch[:, robot_index], label='commanded')
            # a.set_xlabel('Time [s]', fontsize=8)
            # a.set_ylabel('Base ang vel [rad/s]', fontsize=8)
            # a.set_title('Base velocity pitch', fontsize=10)
            # a.set_ylim([-3, 3])
            # a.legend()

            # plot contact roll
            # a = axs[1, 2]
            # if log["base_vel_roll"][env_idx]:
            #     log_base_vel_roll = np.array(log["base_vel_roll"][env_idx])
            #     a.plot(time, log_base_vel_roll[:, robot_index], label='measured')
            # if log["command_roll"][env_idx]:
            #     log_command_roll = np.array(log["command_roll"][env_idx])
            #     a.plot(time, log_command_roll[:, robot_index], label='commanded')
            # a.set_xlabel('Time [s]', fontsize=8)
            # a.set_ylabel('Base ang vel [rad/s]', fontsize=8)
            # a.set_title('Base velocity roll', fontsize=10)
            # a.set_ylim([-3, 3])
            # a.legend()

            fig.tight_layout()
            fig.savefig(os.path.join(self.model_dir, f"env{env_idx}", f"{test_name}_rob_{robot_index}_vel_plot_{self.load_iteration}.png"), dpi=100)
            plt.close()

    def _plot_dof_pos(self, test_name: str, robot_index: int):
        nb_rows = 3
        nb_cols = 4
        for env_idx in range(self.num_envs):
            fig, axs = plt.subplots(nb_rows, nb_cols)
            fig.set_size_inches(16, 6)
            for key, value in self.state_log.items():
                time = np.linspace(0, len(value[0])*self.dt, len(value[0]))
                break
            dof_pos = np.array(self.state_log['dof_pos'][env_idx])
            time = np.linspace(0, len(value[0]) * self.dt, len(value[0]))

            # Plot the status of every joint.
            for i, j in itertools.product(range(nb_rows), range(nb_cols)):
                dof_id = nb_cols * i + j
                a = axs[i, j]
                a.plot(time, dof_pos[:, robot_index, dof_id])
                a.set_xlabel('Time [s]', fontsize=8)
                a.set_ylabel('Dof Pos [rad]', fontsize=8)
                a.set_title(self.dof_names[dof_id], fontsize=10)

            fig.tight_layout()
            fig.savefig(os.path.join(self.model_dir, f"env{env_idx}", f"{test_name}_rob_{robot_index}_dof_pos_plot_{self.load_iteration}.png"), dpi=100)
            plt.close()

    def _plot_dof_vels(self, test_name: str, robot_index: int):
        nb_rows = 3
        nb_cols = 4
        for env_idx in range(self.num_envs):
            fig, axs = plt.subplots(nb_rows, nb_cols)
            fig.set_size_inches(16, 6)
            for key, value in self.state_log.items():
                time = np.linspace(0, len(value[0])*self.dt, len(value[0]))
                break
            dof_vels = np.array(self.state_log['dof_vel'][env_idx])
            time = np.linspace(0, len(value[0]) * self.dt, len(value[0]))

            # Plot the status of every joint.
            for i, j in itertools.product(range(nb_rows), range(nb_cols)):
                dof_id = nb_cols * i + j
                a = axs[i, j]
                a.plot(time, dof_vels[:, robot_index, dof_id])
                a.set_xlabel('Time [s]', fontsize=8)
                a.set_ylabel('Dof Vel [rad/s]', fontsize=8)
                a.set_title(self.dof_names[dof_id], fontsize=10)

            fig.tight_layout()
            fig.savefig(os.path.join(self.model_dir, f"env{env_idx}", f"{test_name}_rob_{robot_index}_dof_vel_plot_{self.load_iteration}.png"), dpi=100)
            plt.close()

    def _plot_dof_accs(self, test_name: str, robot_index: int):
        nb_rows = 3
        nb_cols = 4
        for env_idx in range(self.num_envs):
            fig, axs = plt.subplots(nb_rows, nb_cols)
            fig.set_size_inches(16, 6)
            for key, value in self.state_log.items():
                time = np.linspace(0, len(value[0])*self.dt, len(value[0]))
                break
            dof_accs = np.array(self.state_log['dof_acc'][env_idx])
            time = np.linspace(0, len(value[0]) * self.dt, len(value[0]))

            # Plot the status of every joint.
            for i, j in itertools.product(range(nb_rows), range(nb_cols)):
                dof_id = nb_cols * i + j
                a = axs[i, j]
                a.plot(time, dof_accs[:, robot_index, dof_id])
                a.set_xlabel('Time [s]', fontsize=8)
                a.set_ylabel('Dof Acc [rad/s2]', fontsize=8)
                a.set_title(self.dof_names[dof_id], fontsize=10)

            fig.tight_layout()
            fig.savefig(os.path.join(self.model_dir, f"env{env_idx}", f"{test_name}_rob_{robot_index}_dof_acc_plot_{self.load_iteration}.png"), dpi=100)
            plt.close()

    def _plot_dof_torques(self, test_name: str, robot_index: int):
        nb_rows = 3
        nb_cols = 4
        for env_idx in range(self.num_envs):
            fig, axs = plt.subplots(nb_rows, nb_cols)
            fig.set_size_inches(16, 6)
            for key, value in self.state_log.items():
                time = np.linspace(0, len(value[0])*self.dt, len(value[0]))
                break
            dof_torques = np.array(self.state_log['dof_torque'][env_idx])
            time = np.linspace(0, len(value[0]) * self.dt, len(value[0]))

            # Plot the status of every joint.
            for i, j in itertools.product(range(nb_rows), range(nb_cols)):
                dof_id = nb_cols * i + j
                a = axs[i, j]
                a.plot(time, dof_torques[:, robot_index, dof_id])
                a.set_xlabel('Time [s]', fontsize=8)
                a.set_ylabel('Dof Torque [N x m]', fontsize=8)
                a.set_title(self.dof_names[dof_id], fontsize=10)

            fig.tight_layout()
            fig.savefig(os.path.join(self.model_dir, f"env{env_idx}", f"{test_name}_rob_{robot_index}_dof_torque_plot_{self.load_iteration}.png"), dpi=100)
            plt.close()

    def _plot_actions(self, test_name: str, robot_index: int):
        nb_rows = 3
        nb_cols = 4
        for env_idx in range(self.num_envs):
            fig, axs = plt.subplots(nb_rows, nb_cols)
            fig.set_size_inches(16, 6)
            for key, value in self.state_log.items():
                time = np.linspace(0, len(value[0])*self.dt, len(value[0]))
                break
            actions_scaled = np.array(self.state_log['action_scaled'][env_idx])
            time = np.linspace(0, len(value[0]) * self.dt, len(value[0]))

            # Plot the status of every joint.
            for i, j in itertools.product(range(nb_rows), range(nb_cols)):
                dof_id = nb_cols * i + j
                a = axs[i, j]
                a.plot(time, actions_scaled[:, robot_index, dof_id])
                a.set_xlabel('Time [s]', fontsize=8)
                a.set_ylabel('Actions Scaled [rad]', fontsize=8)
                a.set_title(self.dof_names[dof_id], fontsize=10)

            fig.tight_layout()
            fig.savefig(os.path.join(self.model_dir, f"env{env_idx}", f"{test_name}_robo_{robot_index}_dof_action_plot_{self.load_iteration}.png"), dpi=100)
            plt.close()

    def _plot_gaits(self, test_name: str, robot_index: int):
        """Plot gait and save it into model directory.
        File name is "{test_name}_gait_plot_{self.load_iteration}.png".
        Save energy consumption, gait frequency, average rewards and step information into a yaml file.
        File name is "{test_name}_gait_info_{self.load_iteration}.yaml".
        """
        if self.num_envs == 1:
            fig, axs = plt.subplots(1, 1)
            fig.set_size_inches(16, 6)
            for key, value in self.state_log.items():
                time = np.linspace(0, len(value) * self.dt, len(value))
                break
            forces = np.array(self.state_log["contact_forces_z"])
            contact_gaps = []
            for i in range(forces.shape[2]):
                this_contact_ranges = []
                this_contact_steps = []
                for j in range(forces.shape[0]):
                    if forces[j, robot_index, i] > 5:
                        this_contact_steps.append(j)
                        if j == forces.shape[0] - 1:
                            this_contact_ranges.append([this_contact_steps[0], this_contact_steps[-1]])
                            this_contact_steps = []
                    elif len(this_contact_steps) != 0:
                        this_contact_ranges.append([this_contact_steps[0], this_contact_steps[-1]])
                        this_contact_steps = []
                contact_gaps.append(this_contact_ranges)
            contact_gaps_times = []
            for i in range(forces.shape[2]):
                this_contact_gap_times = []
                for j in range(len(contact_gaps[i])):
                    this_contact_gap_times.append([time[contact_gaps[i][j][0]], time[contact_gaps[i][j][1]]])
                contact_gaps_times.append(this_contact_gap_times)

            color_code = ['r', 'y', 'b', 'g']
            for i in range(forces.shape[2]):
                axs.add_patch(plt.Rectangle((time[0], i - 0.2), time[-1] - time[0], 0.4, edgecolor='none', facecolor=color_code[i], alpha=0.1))
                for j in range(len(contact_gaps_times[i])):
                    axs.add_patch(plt.Rectangle((contact_gaps_times[i][j][0], i - 0.2), contact_gaps_times[i][j][1] - contact_gaps_times[i][j][0], 0.4, edgecolor='none', facecolor=color_code[i]))
            axs.set_xlim([time[0], time[-1]])
            axs.set_ylim([-0.5, 3.5])
            axs.set_xlabel("Running time (seconds)", weight='bold')
            axs.set_ylabel("Foot name", weight='bold')
            axs.set_yticks(range(len(self.feet_names)), self.feet_names)
            axs.set_title("Gait Graph under Test Speed {:.3f} m/s".format(self.lin_speed), weight='bold')

            fig.tight_layout()
            fig.savefig(os.path.join(self.model_dir, f"gaits", f"{test_name}_robo_{robot_index}_gait_plot_ru.png"), dpi=100)
            plt.close()

            # Save the energy and gait informations into YAML file.
            # The velocity tracking error and rewards are averaged in time.
            total_steps = sum([len(contact_gaps[i]) for i in range(len(contact_gaps))])
            step_count = total_steps / len(contact_gaps)

            # dump_dict = {
            #     "step_count": step_count,
            #     "gait": None,
            #     'vel_x': self.get_vel_x(robot_index),
            #     'vel_y': self.get_vel_y(robot_index),
            #     'energy_consume_watts': self.get_energy_consume_watts(robot_index),
            #     'energy_consume_dist': self.get_energy_consume_dist(robot_index),
            #     'rewards': self.get_rewards(robot_index),
            # }
            # with open(os.path.join(self.model_dir, f"gaits", f"{test_name}_robo_{robot_index}_gait_info_{self.load_iteration}.yaml"), 'w') as file:
            #     yaml.dump(dump_dict, file)
            #     file.close()

        else:
            for env_idx in range(self.num_envs):
                fig, axs = plt.subplots(1, 1)
                fig.set_size_inches(16, 6)
                for key, value in self.state_log.items():
                    time = np.linspace(0, len(value[0]) * self.dt, len(value[0]))
                    break
                forces = np.array(self.state_log["contact_forces_z"][env_idx])
                contact_gaps = []
                for i in range(forces.shape[2]):
                    this_contact_ranges = []
                    this_contact_steps = []
                    for j in range(forces.shape[0]):
                        if forces[j, robot_index, i] > 5:
                            this_contact_steps.append(j)
                            if j == forces.shape[0] - 1:
                                this_contact_ranges.append([this_contact_steps[0], this_contact_steps[-1]])
                                this_contact_steps = []
                        elif len(this_contact_steps) != 0:
                            this_contact_ranges.append([this_contact_steps[0], this_contact_steps[-1]])
                            this_contact_steps = []
                    contact_gaps.append(this_contact_ranges)
                contact_gaps_times = []
                for i in range(forces.shape[2]):
                    this_contact_gap_times = []
                    for j in range(len(contact_gaps[i])):
                        this_contact_gap_times.append([time[contact_gaps[i][j][0]], time[contact_gaps[i][j][1]]])
                    contact_gaps_times.append(this_contact_gap_times)

                color_code = ['r', 'y', 'b', 'g']
                for i in range(forces.shape[2]):
                    axs.add_patch(plt.Rectangle((time[0], i - 0.2), time[-1] - time[0], 0.4, edgecolor='none', facecolor=color_code[i], alpha=0.1))
                    for j in range(len(contact_gaps_times[i])):
                        axs.add_patch(plt.Rectangle((contact_gaps_times[i][j][0], i - 0.2), contact_gaps_times[i][j][1] - contact_gaps_times[i][j][0], 0.4, edgecolor='none', facecolor=color_code[i]))
                axs.set_xlim([time[0], time[-1]])
                axs.set_ylim([-0.5, 3.5])
                axs.set_xlabel("Running time (seconds)", weight='bold')
                axs.set_ylabel("Foot name", weight='bold')
                axs.set_yticks(range(len(self.feet_names)), self.feet_names)
                axs.set_title("Gait Graph under Test Speed {:.3f} m/s".format(self.lin_speed), weight='bold')

                fig.tight_layout()
                fig.savefig(os.path.join(self.model_dir, f"env{env_idx}", f"{test_name}_robo_{robot_index}_gait_plot_{self.load_iteration}.png"), dpi=100)
                plt.close()

                # Save the energy and gait informations into YAML file.
                # The velocity tracking error and rewards are averaged in time.
                total_steps = sum([len(contact_gaps[i]) for i in range(len(contact_gaps))])
                step_count = total_steps / len(contact_gaps)

                self.dump_dict['environments'][f'env_{env_idx}'] = {
                    "step_count": step_count,
                    "gait": None,
                    'vel_x': self.get_vel_x(robot_index, env_idx),
                    'vel_y': self.get_vel_y(robot_index, env_idx),
                    'energy_consume_watts': self.get_energy_consume_watts(robot_index, env_idx),
                    'energy_consume_dist': self.get_energy_consume_dist(robot_index, env_idx),
                    'rewards': self.get_rewards(robot_index, env_idx),
                }
                with open(os.path.join(self.model_dir, f"env{env_idx}", f"{test_name}_robo_{robot_index}_gait_info_{self.load_iteration}.yaml"), 'w') as file:
                    yaml.dump(self.dump_dict, file)
                    file.close()

    def _plot_tracking(self, test_name: str, robot_index: int):
        for env_idx in range(self.num_envs):
            fig, axs = plt.subplots(1, 1)
            fig.set_size_inches(16, 6)
            base_pos_x = np.array(self.state_log['base_pos_x'][env_idx])[:, robot_index]
            base_pos_y = np.array(self.state_log['base_pos_y'][env_idx])[:, robot_index]
            current_yaw = np.array(self.state_log['base_pos_yaw'][env_idx])[0, robot_index]

            command_vel_x = np.array(self.state_log['command_x'][env_idx])[:, robot_index]
            command_vel_yaw = np.array(self.state_log['command_yaw'][env_idx])[:, robot_index]

            reference_pos_x = np.zeros_like(base_pos_x)
            reference_pos_y = np.zeros_like(base_pos_y)
            reference_pos_x[0] = base_pos_x[0]
            reference_pos_y[0] = base_pos_y[0]
            for i in range(base_pos_x.shape[0] - 1):
                reference_pos_x[i + 1] = command_vel_x[i] * np.cos(current_yaw) * self.dt + reference_pos_x[i]
                reference_pos_y[i + 1] = command_vel_x[i] * np.sin(current_yaw) * self.dt + reference_pos_y[i]
                current_yaw = current_yaw + command_vel_yaw[i] * self.dt

            axs.plot(reference_pos_x, reference_pos_y, label="Reference")
            axs.plot(base_pos_x, base_pos_y, label="Actual")
            axs.legend()
            axs.set_xlim([np.min(reference_pos_x) - 0.1, np.max(reference_pos_x) + 0.1])
            axs.set_ylim([np.min(reference_pos_y) - 0.1, np.max(reference_pos_y) + 0.1])
            axs.set_xlabel("x (m)", weight='bold')
            axs.set_ylabel("y (m)", weight='bold')
            axs.set_title("Tracking Plot", weight='bold')
            axs.set_aspect('equal', adjustable='box')
            fig.tight_layout()
            fig.savefig(os.path.join(self.model_dir, f"env{env_idx}", f"{test_name}_robo_{robot_index}_tracking_plot_{self.load_iteration}.png"), dpi=100)
            plt.close()

    def _plot_cot(self, test_name: str):
        for env_idx in range(self.num_envs):
            energy_t = np.array(self.state_log["energy_consume"][env_idx])  # Power (W)
            pos_x = np.array(self.state_log["base_pos_x"][env_idx])[:, 0]
            pos_y = np.array(self.state_log["base_pos_y"][env_idx])[:, 0]

            energy_cum = np.cumsum(energy_t[:, 0] * self.dt)  # J
            dist_cum = np.cumsum(np.sqrt(np.diff(pos_x, prepend=pos_x[0])**2 +
                                        np.diff(pos_y, prepend=pos_y[0])**2))  # m

            CoT_t = energy_cum / dist_cum

            _, CoT_mean, CoT_std = self.get_energy_consume_dist_stats(env_idx)
            print(f"Env: {env_idx} Mean CoT: {CoT_mean} Std CoT: {CoT_std}")

            for key, value in self.state_log.items():
                time = np.linspace(0, len(value[0])*self.dt, len(value[0]))
                break

            plt.plot(time, CoT_t, color='r')
            plt.xlabel("Time (s)")
            plt.ylabel("CoT (J/m)")
            plt.title("Cost of Transportation over Time")
            plt.show()

    def _plot_cumulative_cot(self, test_name: str, robot_mass: float = 21.5):
        for env_idx in range(self.num_envs):
            energy_t = np.array(self.state_log["energy_consume"][env_idx])  # Power (W)
            pos_x = np.array(self.state_log["base_pos_x"][env_idx])[:, 0]
            pos_y = np.array(self.state_log["base_pos_y"][env_idx])[:, 0]

            energy_cum = np.cumsum(energy_t[:, 0] * self.dt)  # J
            dist_cum = np.cumsum(np.sqrt(np.diff(pos_x, prepend=pos_x[0])**2 +
                                        np.diff(pos_y, prepend=pos_y[0])**2))  # m
            g = 9.81
            CoT_t = energy_cum / (robot_mass * g * dist_cum)

            for key, value in self.state_log.items():
                time = np.linspace(0, len(value[0])*self.dt, len(value[0]))
                break

            plt.plot(time, CoT_t)
            plt.xlabel("Time (s)")
            plt.ylabel("Cumulative CoT")
            plt.title("Cumulative Cost of Transport over Time")
            plt.show()
            # plt.grid(True)
            # plt.savefig(f"{os.path.join(self.model_dir, self.label)}/cot_over_time.png")
            # plt.close()
