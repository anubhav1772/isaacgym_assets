import gym
import torch
import numpy as np
from aliengo_gym.utils.math_utils import quat_apply_yaw, wrap_to_pi, get_scale_shift
from aliengo_gym.envs.base.curriculum import RewardThresholdCurriculum

import os
import matplotlib.pyplot as plt

class GaitPolicyWrapper(gym.Wrapper):
    """
    env action = gait params (from Policy 1)
    produces torques by calling Policy 2 internally
    """

    def __init__(self, env, policy2, cfg):
        super().__init__(env)
        self.env = env
        self.wtw = policy2                  # frozen locomotion controller (WTW)
        self.cfg = cfg
        self.device = env.device
        self.enable_curriculum = False      # default
        self.obs_scales = cfg.obs_scales

        self.num_envs = cfg.env.num_envs
        self.dt = env.dt
        self.num_gaits = cfg.env.num_gaits

        self.reward_scales = vars(cfg.reward_scales)

        # self.pi1_decimation = 20   # high-level acts every 20 pi2 steps
        # self.pi1_counter = 0
        # self.current_pi1_gaits = torch.zeros(self.cfg.env.num_envs, self.cfg.env.num_gaits, dtype=torch.float,
        #                         device=self.env.device, requires_grad=False)

        # allocate buffer for gait params
        self.gaits = torch.zeros(self.cfg.env.num_envs, self.cfg.env.num_gaits, dtype=torch.float,
                                device=self.env.device, requires_grad=False)
        # store last gait params
        self.last_gaits = torch.zeros_like(self.gaits)

        # self.commands_scale = torch.tensor([self.cfg.obs_scales.lin_vel, self.cfg.obs_scales.lin_vel, self.cfg.obs_scales.ang_vel],
        #                                    device=self.device, requires_grad=False, )[:self.cfg.commands.num_commands]

        # self.commands_value = torch.zeros(self.cfg.env.num_envs, self.cfg.commands.num_commands, dtype=torch.float,
        #                                   device=self.device, requires_grad=False)
        # self.commands = torch.zeros_like(self.commands_value)  # x vel, y vel, yaw vel

        # Velocity commands (used by Policy 1 and passed to Policy 2)
        # Shape: [num_envs, 3] = (vx, vy, yaw)
        self.current_vel_cmds = torch.zeros(self.cfg.env.num_envs, 3, dtype=torch.float, device=self.env.device, requires_grad=False)

        # gait phase
        # self.gait_indices = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.env.device,
        #                                 requires_grad=False)

        # self.pi1_history = torch.zeros((self.env.num_envs, self.pi1_obs_dim * self.pi1_history_length),
        #                                 dtype=torch.float,  device=self.env.device)

        self.pi1_obs_history_length = self.cfg.env.num_observation_history
        self.pi1_obs_dim = self.cfg.env.num_observations
        self.num_pi1_obs_history = self.pi1_obs_history_length * self.pi1_obs_dim

        # Current Obs
        #self.pi1_obs = torch.zeros(self.env.num_envs, self.pi1_obs_dim,
        #                                dtype=torch.float, device=self.device, requires_grad=False)

        #self.pi1_obss = torch.empty((self.env.num_envs, 0), dtype=torch.float, device=self.device)

        # Obs History [num_envs, num_observation_history * pi1_obs_dim]
        self.pi1_obs_history = torch.zeros(self.env.num_envs, self.num_pi1_obs_history,
                                        dtype=torch.float, device=self.device, requires_grad=False)

        # Privileged Obs
        self.num_pi1_privileged_obs = self.cfg.env.num_privileged_obs
        self.privileged_pi1_obss = torch.zeros(self.env.num_envs, self.num_pi1_privileged_obs,
                                        dtype=torch.float, device=self.device, requires_grad=False)

        # Noise
        #self.noise_scale_vec = self._get_noise_scale_vec(cfg)
        self.add_noise = self.cfg.noise.add_noise

        wtw_freq = 1.0 / env.dt        # env.env.dt = env.env.sim_params.dt * env.env.cfg.control.decimation
        # print(f"dt: {env.env.sim_params.dt}, \
        #         control_decimation: {env.env.cfg.control.decimation}, \
        #         policy 2 freq: {wtw_freq}")

        policy1_freq = 2
        self.hl_decimation = max(1, int(round(wtw_freq / policy1_freq)))

        self.win_len = self.hl_decimation
        self.lin_vel_buf = torch.zeros(self.num_envs, self.win_len, 3, device=self.device)
        self.ang_vel_buf = torch.zeros(self.num_envs, self.win_len, 3, device=self.device)
        self.dof_vel_buf = torch.zeros(self.num_envs, self.win_len, 12, device=self.device)
        self.power_buf   = torch.zeros(self.num_envs, self.win_len, device=self.device)
        #self.rew_ll_buf = torch.zeros(self.num_envs, self.win_len, device=self.device) # (num_envs, win_len)

        # self.gait_ranges = self._get_gait_ranges()
        self.gait_ranges = torch.tensor(self._get_gait_ranges(), dtype=torch.float32, device=self.sim_device)
        # self._prepare_reward_function()

        self.global_step = 0

        self.episode_sums = {
            "energy": torch.zeros(self.num_envs, device=self.device),
            "gait_smoothness": torch.zeros(self.num_envs, device=self.device),
            "tracking": torch.zeros(self.num_envs, device=self.device),
            "rew_ll": torch.zeros(self.num_envs, device=self.device),
            # "tracking_lin": torch.zeros(self.num_envs, device=self.device),
            # "tracking_ang": torch.zeros(self.num_envs, device=self.device),
            "total_reward": torch.zeros(self.num_envs, device=self.device),
        }

    def _get_gait_ranges(self):
        gait_cfg = self.cfg.gaits
        self.gait_param_names = np.array(gait_cfg.name)

        return np.array([
            gait_cfg.limit_body_height,
            gait_cfg.limit_gait_frequency,
            gait_cfg.limit_gait_phase,
            gait_cfg.limit_gait_offset,
            gait_cfg.limit_gait_bound,
            gait_cfg.limit_gait_duration,
            gait_cfg.limit_footswing_height,
            gait_cfg.limit_body_pitch,
            gait_cfg.limit_body_roll,
            gait_cfg.limit_stance_width,
            gait_cfg.limit_stance_length
        ])


    def set_velocity_commands(self, vel_cmds):
        assert vel_cmds.shape == self.current_vel_cmds.shape
        self.current_vel_cmds[:] = vel_cmds
        # if self.global_step % 100 == 0:
        #     print("cmd debug:", vel_cmds[0])

    def _sample_velocity_commands(self, env_ids):
        vx = torch.rand(env_ids.numel(), device=self.device) * (
            self.cfg.commands.lin_vel_x[1]
            - self.cfg.commands.lin_vel_x[0]
        ) + self.cfg.commands.lin_vel_x[0]

        vy = torch.rand(env_ids.numel(), device=self.device) * (
            self.cfg.commands.lin_vel_y[1]
            - self.cfg.commands.lin_vel_y[0]
        ) + self.cfg.commands.lin_vel_y[0]

        yaw = torch.rand(env_ids.numel(), device=self.device) * (
            self.cfg.commands.ang_vel_yaw[1]
            - self.cfg.commands.ang_vel_yaw[0]
        ) + self.cfg.commands.ang_vel_yaw[0]

        self.current_vel_cmds[env_ids, :] = torch.stack([vx, vy, yaw], dim=-1)

    def reset(self):
        _ = self.env.reset()

        # Reset HL state
        self.gaits.zero_()
        self.last_gaits.zero_()
        self.pi1_obs_history.zero_()

        # Reset episode integrals
        for k in self.episode_sums:
            self.episode_sums[k].zero_()

        # Reset window buffers
        self.lin_vel_buf.zero_()
        self.ang_vel_buf.zero_()
        self.power_buf.zero_()

        # Sample velocity ONCE per episode
        env_ids = torch.arange(self.num_envs, device=self.device)
        self._sample_velocity_commands(env_ids)

        # Build initial observations
        self._compute_pi1_obs()
        self._update_pi1_history()
        self._compute_pi1_privileged_obs()

        return {
            "obs": self.pi1_obss,
            "privileged_obs": self.privileged_pi1_obss,
            "obs_history": self.pi1_obs_history,
        }


    def reset_idx(self, env_ids):
        """
        Reset only selected environments.
        Called when LL terminates inside an HL window.
        """

        if env_ids.numel() == 0:
            return

        # Reset underlying envs
        self.env.reset_idx(env_ids)
        self.env.compute_observations()

        # HARD reset HL-side state
        self.gaits[env_ids].zero_()
        self.last_gaits[env_ids].zero_()
        self.pi1_obs_history[env_ids].zero_()

        # Reset episode integrals
        for k in self.episode_sums:
            self.episode_sums[k][env_ids] = 0.0

        # Reset window buffers
        self.lin_vel_buf[env_ids].zero_()
        self.ang_vel_buf[env_ids].zero_()
        self.power_buf[env_ids].zero_()

        # Sample new velocity for new episode
        self._sample_velocity_commands(env_ids)


    def step(self, gaits):
        """
        gaits: Policy 1 actions, shape [num_envs, num_gaits]
        """
        distance = torch.zeros(self.num_envs, device=self.device)
        distance_masked = torch.zeros(self.num_envs, device=self.device)
        done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        alive = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        valid_counts = torch.zeros(self.num_envs, device=self.device)
        rew_accum = torch.zeros(self.num_envs, device=self.device)

        self.lin_vel_buf.zero_()
        self.ang_vel_buf.zero_()
        self.power_buf.zero_()
        #self.rew_ll_buf.zero_()

        self.gaits[:] = gaits
        self.dgaits = self.gaits - self.last_gaits

        # Inject velocity commands + generated gaits into underlying env for Policy 2
        alive_ids = alive.nonzero(as_tuple=False).squeeze(-1)
        self.wtw.set_env_commands(vel_cmds=self.current_vel_cmds, gait_params=self.gaits, env_ids=alive_ids)

        # Defer LL resets during HL window
        self.env.defer_reset = True
        # Freeze LL-side command resampling
        self.env.defer_command_resample = True

        for i in range(self.win_len):
            if not alive.any():
                break

            alive_at_step_start = alive.clone()

            with torch.no_grad():
                obs_pi2 = self.env.get_observations()
                action_pi2 = self.wtw.policy(obs_pi2)

            obs_pi2_next, rew_pi2, done_step, info = self.env.step(action_pi2)
            done_step = done_step.bool()

            # Re-apply HL commands to freshly reset envs
            #reset_ids = done_step.nonzero(as_tuple=False).squeeze(-1)
            #if reset_ids.numel() > 0:
            #    self.wtw.set_env_commands(
            #        self.current_vel_cmds[reset_ids],
            #        self.gaits[reset_ids],
            #        env_ids=reset_ids
            #    )

            # Count this LL step if env was alive at step start
            valid_counts[alive_at_step_start] += 1

            # physics truth
            lin_vel = self.env.base_lin_vel
            ang_vel = self.env.base_ang_vel

            self.lin_vel_buf[alive, i] = lin_vel[alive]
            self.ang_vel_buf[alive, i] = ang_vel[alive]

            # LL reward (window-normalized)
            rew_accum += 25.0 * rew_pi2 * alive / self.win_len

            # power
            power = torch.sum(torch.abs(self.env.torques * self.env.dof_vel), dim=1) * alive.float()
            self.power_buf[:, i] = power

            #power = torch.sum(torch.abs(self.env.torques * self.env.dof_vel), dim=1)
            #self.power_buf[alive, i] = power[alive]

            # distance
            #step_dist = torch.norm(lin_vel[:, :2], dim=1) * self.dt
            #distance[alive] += step_dist[alive]
            distance_masked[alive] += (torch.norm(self.lin_vel_buf[alive, i, :2], dim=1) * self.dt)

            # Update alive mask
            newly_done = done_step & alive
            alive = alive & (~newly_done)
            done = done | newly_done

        # Re-enable LL resets
        self.env.defer_reset = False
        # Re-enable LL-side command resampling
        self.env.defer_command_resample = True

        if "train/episode" not in info:
            info["train/episode"] = {}

        if done.any():
            # self.lin_vel_buf[done] = 0.0
            # self.ang_vel_buf[done] = 0.0
            # self.power_buf[done] = 0.0
            # self.last_gaits[done] = 0.0

            done_ids = done.nonzero(as_tuple=False).squeeze(-1)

            # Total accumulated energy over the whole episode
            # Only for envs that just finished
            info["train/episode"]["rew_energy"] = (
                self.episode_sums["energy"][done_ids].mean().item()
            )
            info["train/episode"]["rew_total"] = (
                self.episode_sums["total_reward"][done_ids].mean().item()
            )
            info["train/episode"]["rew_gait_smoothness"] = (
                self.episode_sums["gait_smoothness"][done_ids].mean().item()
            )
            info["train/episode"]["rew_ll"] = (
                self.episode_sums["rew_ll"][done_ids].mean().item()
            )
            # info["train/episode"]["rew_tracking_lin"] = (
            #     self.episode_sums["tracking_lin"][done_ids].mean().item()
            # )
            # info["train/episode"]["rew_tracking_ang"] = (
            #     self.episode_sums["tracking_ang"][done_ids].mean().item()
            # )
            # reset per-env episode sums
            for k in self.episode_sums:
                self.episode_sums[k][done_ids] = 0.0

        # Reset all envs that terminated during the window
        dead_ids = done.nonzero(as_tuple=False).squeeze(-1)
        if dead_ids.numel() > 0:
            self.reset_idx(dead_ids)

        valid_counts = torch.clamp(valid_counts, min=1.0)

        mask = (torch.arange(self.win_len, device=self.device) < valid_counts.unsqueeze(1))

        # Means only for logging / obs
        #lin_vel_mean = self.lin_vel_buf.sum(dim=1) / valid_counts[:, None]
        #ang_vel_mean = self.ang_vel_buf.sum(dim=1) / valid_counts[:, None]
        # window-normalized sum
        lin_vel_mean = (self.lin_vel_buf * mask.unsqueeze(-1)).sum(dim=1) / valid_counts[:, None]
        ang_vel_mean = (self.ang_vel_buf * mask.unsqueeze(-1)).sum(dim=1) / valid_counts[:, None]
        rew_ll_win = rew_accum

        #power_total = self.power_buf.sum(dim=1)                 # Σ |τ·ω|
        power_total = (self.power_buf * mask).sum(dim=1)         # scalar per env
        mean_power = power_total / valid_counts
        energy = power_total * self.dt                           # J

        #distance = torch.norm(lin_vel_mean[:, :2], dim=1) * (valid_counts * self.dt)
        #distance = torch.norm(self.lin_vel_buf[:, :, :2], dim=2).sum(dim=1) * self.dt
        #distance = torch.clamp(distance, min=0.05 * valid_counts * self.dt)

        #cot = energy / (self.cfg.env.mg * (distance + 1e-6))
        distance = torch.clamp(distance_masked, min=1e-3)
        cot = energy / (self.cfg.env.mg * distance)
        #cot = power_total / (self.cfg.env.mg * torch.abs(lin_vel_mean[:, 0]) + 1e-6)

        # rew_ll_mean = 25 * rew_ll_mean
        rew_smooth = self._reward_gait_smooth()
        rew_energy = self._reward_energy_regularization(power_total, lin_vel_mean, ang_vel_mean)

        #rew_ll_safe = torch.clamp(rew_ll_mean, min=1e-1)
        #rew_ll_log = torch.log(rew_ll_mean + 0.1)
        reward = rew_ll_win + rew_energy + 0.2 * rew_smooth
        #reward = torch.clamp(reward, -3.0, 3.0)
        #reward = reward * valid_counts / self.win_len

        vx_cmd = self.current_vel_cmds[:, 0]
        vy_cmd = self.current_vel_cmds[:, 1]
        yaw_cmd = self.current_vel_cmds[:, 2]
        #vx_cmd, vy_cmd, yaw_cmd = self.current_vel_cmds.T

        vx = lin_vel_mean[:, 0]
        vy = lin_vel_mean[:, 1]
        #vx, vy, _ = lin_vel_mean.T
        yaw = ang_vel_mean[:, 2]

        # terminate envs that are too slow relative to command
        #cmd_mask = torch.abs(vx_cmd) > 0.2
        #slow_mask = cmd_mask & (torch.abs(vx) < 0.3 * torch.abs(vx_cmd))
        #if self.global_step > 10:
        #    done |= slow_mask

        #reward = (rew_log - rew_log.mean().detach())
        #reward = torch.clamp(reward, -3, 3)

        # Episode sums
        mask = ~done
        self.episode_sums["energy"] += rew_energy * mask
        self.episode_sums["gait_smoothness"] += rew_smooth * mask
        #self.episode_sums["rew_ll"] += (rew_ll_win / valid_counts) * mask
        # self.episode_sums["tracking_lin"] += rew_lin * mask
        # self.episode_sums["tracking_ang"] += rew_ang * mask
        self.episode_sums["total_reward"] += reward * mask

        if "train/step" not in info:
            info["train/step"] = {}

        info["train/step"]["valid_counts_mean"] = valid_counts.mean().item()
        info["train/step"]["valid_counts_min"]  = valid_counts.min().item()
        info["train/step"]["valid_counts_max"]  = valid_counts.max().item()

        valid_envs = valid_counts > 0

        self.episode_sums["rew_ll"][valid_envs] += (rew_ll_win[valid_envs] / valid_counts[valid_envs])
        # Step-wise metrics
        info["train/step"]["rew_energy"] = rew_energy[valid_envs].mean().item()
        info["train/step"]["rew_smooth"] = rew_smooth[valid_envs].mean().item()
        info["train/step"]["rew_ll_mean"] = rew_ll_win[valid_envs].mean().item()
        info['train/step']["rew_ll_min"] = rew_ll_win[valid_envs].min().item()
        info['train/step']["rew_ll_max"] = rew_ll_win[valid_envs].max().item()
        # info["train/step"]["rew_tracking_lin"] = rew_lin[valid_envs].mean().item()
        # info["train/step"]["rew_tracking_ang"] = rew_ang[valid_envs].mean().item()
        info["train/step"]["rew_mean"] = reward[valid_envs].mean().item()
        info["train/step"]["rew_min"] = reward[valid_envs].min().item()
        info["train/step"]["rew_max"] = reward[valid_envs].max().item()

        info["train/step"]["vx_cmd_abs"] = vx_cmd[valid_envs].abs().mean().item()
        info["train/step"]["vx_abs"] = vx[valid_envs].abs().mean().item()
        # info["train/step"]["vx_err"] = ((vx_cmd[valid_envs] - vx[valid_envs]) ** 2).mean().item()

        # info["train/step"]["yaw_cmd_abs"] = yaw_cmd[valid_envs].abs().mean().item()
        # info["train/step"]["yaw_abs"] = yaw[valid_envs].abs().mean().item()
        # info["train/step"]["yaw_err"] = ((yaw_cmd[valid_envs] - yaw[valid_envs]) ** 2).mean().item()
        #info["train/step"]["energy_per_meter"] = (energy[valid_envs] / distance[valid_envs]).mean().item()

        #info["train/step"]["energy"] = energy[valid_envs].mean().item()
        # info["train/step"]["distance"] = distance[valid_envs].mean().item()

        info["train/step"]["power_total"] = mean_power[valid_envs].mean().item()
        #info["train/step"]["CoT"] = self.compute_CoT()[valid_envs].mean().item()
        info["train/step"]["CoT"] = cot[valid_envs].mean().item()

        if "train/gait" not in info:
            info["train/gait"] = {}

        for i, name in enumerate(self.gait_param_names):
            info["train/gait"][name] = self.gaits[:, i].mean().item()
            # info["train/gait"][f"{name}_delta"] = (
            #     (self.gaits[:, i] - self.last_gaits[:, i]).abs().mean().item())

        obs = self.get_observations(lin_vel_mean, ang_vel_mean)
        # update last gaits
        self.last_gaits[:] = self.gaits
        self.global_step += 1

        return obs, reward, done, info

    def get_observations(self, lin_vel_mean, ang_vel_mean):
        """
        Called by Runner before rollout.
        We return the policy 1 obs dict, not policy 2's (i.e. wtw/unio4).
        """
        self._compute_pi1_obs(lin_vel_mean, ang_vel_mean)
        self._update_pi1_history()
        self._compute_pi1_privileged_obs()

        if not torch.isfinite(self.pi1_obs_history).all():
            raise RuntimeError("NaNs detected in Policy-1 observation history")

        return {
            "obs": self.pi1_obss,
            "privileged_obs": self.privileged_pi1_obss,
            "obs_history": self.pi1_obs_history,
        }


    def _compute_pi1_obs(self, lin_vel_mean=None, ang_vel_mean=None):
        """
        Construct policy 1 observation (no history).
        Here we will use only those features we want Policy 1 to see:
            - projected gravity vector
            - base linear (along x and y) and angular vels (yaw)
            - desired velocity commands (from Runner, not env.commands)
            - clock inputs (?)
            - foot contact forces (x, y, z) or resultant foot forces (?)
            - torques (?)
            - last gait parameters (?)
        """
        self.pi1_obss = torch.empty((self.env.num_envs, 0), dtype=torch.float, device=self.device)

        if lin_vel_mean is None:
            lin_vel_mean = self.env.base_lin_vel

        if ang_vel_mean is None:
            ang_vel_mean = self.env.base_ang_vel

        # Current low-level state we choose for gait generator
        # pi1_features = torch.cat([
        #     self.env.base_lin_vel,        # [N, 3]
        #     self.env.base_ang_vel,        # [N, 3]
        #     self.env.projected_gravity,   # [N, 3]
        #     self.env.commands[:, :self.cfg.num_commands.num_commands],     # commanded vel
        #     self.env.contact_forces[:, self.base_env.feet_indices].reshape(N, 12),
        #     self.env.last_action,
        #     self.gaits,        # 10
        #     self.gait_indices.unsqueeze(1),    # phase
        #     self.clock_inputs,                 # 4
        # ], dim=-1)

        if self.cfg.env.observe_gravity:
            self.pi1_obss = torch.cat((self.pi1_obss,
                                        self.env.projected_gravity), dim=-1)

        if self.cfg.env.observe_command:
            self.pi1_obss = torch.cat((self.pi1_obss,
                                        self.current_vel_cmds), dim=-1)

        if self.cfg.env.observe_timing_parameter:
            self.pi1_obss = torch.cat((self.pi1_obss,
                                      self.gait_indices.unsqueeze(1)), dim=-1)

        if self.cfg.env.observe_clock_inputs:
            self.pi1_obss = torch.cat((self.pi1_obss,
                                      self.env.clock_inputs), dim=-1) # implement clock inputs here if possible

        if self.cfg.env.observe_vel:
            if self.cfg.commands.global_reference:
                self.pi1_obss = torch.cat((self.root_states[:self.num_envs, 7:10] * self.cfg.obs_scales.lin_vel,
                                          self.base_ang_vel * self.cfg.obs_scales.ang_vel,
                                          self.pi1_obss), dim=-1)
            else:
                self.pi1_obss = torch.cat((self.base_lin_vel * self.cfg.obs_scales.lin_vel,
                                          self.base_ang_vel * self.cfg.obs_scales.ang_vel,
                                          self.pi1_obss), dim=-1)

        if self.cfg.env.observe_only_ang_vel:
            self.pi1_obss = torch.cat((self.env.base_ang_vel * self.cfg.obs_scales.ang_vel,
                                      self.pi1_obss), dim=-1)

        if self.cfg.env.observe_only_lin_vel:
            self.pi1_obss = torch.cat((self.env.base_lin_vel * self.cfg.obs_scales.lin_vel,
                                      self.pi1_obss), dim=-1)

        if self.cfg.env.observe_only_lin_vel_xy:
            #self.pi1_obss = torch.cat((self.env.base_lin_vel[:self.num_envs, :2] * self.cfg.obs_scales.lin_vel,
            #                          self.pi1_obss), dim=-1)
            self.pi1_obss = torch.cat([lin_vel_mean[:, :2] * self.cfg.obs_scales.lin_vel,
                                        self.pi1_obss], dim=-1)

        if self.cfg.env.observe_only_ang_vel_z:
            #self.pi1_obss = torch.cat((self.env.base_ang_vel[:self.num_envs, :1] * self.cfg.obs_scales.ang_vel,
            #                          self.pi1_obss), dim=-1)
            self.pi1_obss = torch.cat((ang_vel_mean[:self.num_envs, 2].unsqueeze(1) * self.cfg.obs_scales.ang_vel,
                                      self.pi1_obss), dim=-1)

        if self.cfg.env.observe_yaw:
            forward = quat_apply(self.env.base_quat, self.env.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0]).unsqueeze(1)
            # heading_error = torch.clip(0.5 * wrap_to_pi(heading), -1., 1.).unsqueeze(1)
            self.pi1_obss = torch.cat((self.pi1_obss,
                                      heading), dim=-1)

        if self.cfg.env.observe_contact_states:
            # Include foot forces only (Binary contact)
            # stance = 1, swing = 0
            foot_contact = (self.env.contact_forces[:, self.env.feet_indices, 2] > 1.).view(self.env.num_envs, -1) * 1.0
            self.pi1_obss = torch.cat((self.pi1_obss, foot_contact), dim=1)

        if self.cfg.env.observe_foot_forces:
            # Extract foot forces
            foot_forces = self.env.contact_forces[:, self.env.feet_indices, :]  # Shape (num_envs, num_feet, 3)
            foot_forces_flat = foot_forces.view(self.env.num_envs, -1)      # Flatten: Shape (num_envs, num_feet * 3)
            self.pi1_obss = torch.cat((self.pi1_obss, foot_forces_flat), dim=1)

        # add noise if needed
        if self.add_noise:
            self.pi1_obss += (2 * torch.rand_like(self.pi1_obss) - 1) * self.noise_scale_vec

        assert self.pi1_obss.shape[1] == self.pi1_obs_dim, \
            f"pi1 obs dim mismatch: got {self.pi1_obss.shape[1]}, expected {self.pi1_obs_dim}"


    def _compute_pi1_privileged_obs(self):
        """
        Build privileged obs for Policy 1.
        """

        # build privileged obs

        self.privileged_pi1_obss = torch.empty(self.num_envs, 0).to(self.device)
        # self.next_privileged_pi1_obss = torch.empty(self.num_envs, 0).to(self.device)

        if self.cfg.env.priv_observe_body_height:
            body_height_scale, body_height_shift = get_scale_shift(self.cfg.normalization.body_height_range)
            self.privileged_pi1_obss = torch.cat((self.privileged_pi1_obss,
                                                 ((self.env.root_states[:self.num_envs, 2]).view(
                                                     self.num_envs, -1) - body_height_shift) * body_height_scale),
                                                dim=1)

        if self.cfg.env.priv_observe_friction:
            friction_coeffs_scale, friction_coeffs_shift = get_scale_shift(self.cfg.normalization.friction_range)
            self.privileged_pi1_obss = torch.cat((self.privileged_pi1_obss,
                                                 (self.env.friction_coeffs[:, 0].unsqueeze(1) - friction_coeffs_shift) * friction_coeffs_scale),
                                                dim=1)
            # self.next_privileged_pi1_obss = torch.cat((self.next_privileged_pi1_obss,
            #                                           (self.friction_coeffs[:, 0].unsqueeze(1) - friction_coeffs_shift) * friction_coeffs_scale),
            #                                          dim=1)
        # if self.cfg.env.priv_observe_ground_friction:
        #     self.ground_friction_coeffs = self._get_ground_frictions(range(self.num_envs))
        #     ground_friction_coeffs_scale, ground_friction_coeffs_shift = get_scale_shift(
        #         self.cfg.normalization.ground_friction_range)
        #     self.privileged_pi1_obss = torch.cat((self.privileged_pi1_obss,
        #                                          (self.env.ground_friction_coeffs.unsqueeze(1) - ground_friction_coeffs_shift) * ground_friction_coeffs_scale),
        #                                         dim=1)
        #     self.next_privileged_pi1_obss = torch.cat((self.next_privileged_pi1_obss,
        #                                               (self.env.ground_friction_coeffs.unsqueeze(1) - friction_coeffs_shift) * friction_coeffs_scale),
        #                                              dim=1)
        if self.cfg.env.priv_observe_restitution:
            restitutions_scale, restitutions_shift = get_scale_shift(self.cfg.normalization.restitution_range)
            self.privileged_pi1_obss = torch.cat((self.privileged_pi1_obss,
                                                 (self.env.restitutions[:, 0].unsqueeze(1) - restitutions_shift) * restitutions_scale),
                                                dim=1)
            # self.next_privileged_pi1_obss = torch.cat((self.next_privileged_pi1_obss,
            #                                           (self.env.restitutions[:, 0].unsqueeze(1) - restitutions_shift) * restitutions_scale),
            #                                          dim=1)

        assert self.privileged_pi1_obss.shape[
                   1] == self.cfg.env.num_privileged_obs, f"num_privileged_obs ({self.cfg.env.num_privileged_obs}) != the number of privileged observations ({self.privileged_pi1_obss.shape[1]}), you will discard data from the student!"


    def _update_pi1_history(self):
        """
        Shift history left and append new observations at the end.
        """
        self.pi1_obs_history = torch.cat([
            self.pi1_obs_history[:, self.pi1_obs_dim:],
            self.pi1_obss
        ], dim=-1)  # Shape: [num_envs, H * pi1_obs_dim]


    def _prepare_reward_function(self):
        """Prepares a list of reward functions, whcih will be called to
        compute the total reward.
        """
        # Create reward container
        from aliengo_gym.envs.rewards.corl_rewards import CoRLRewards
        self.reward_container = CoRLRewards(self)

        # Filter and scale rewards
        scales = {}
        for name, scale in self.reward_scales.items():
            if abs(scale) > 0:
                # scales[name] = scale * self.dt  # scale by dt
                scales[name] = scale

        self.reward_scales = scales             # keep only non-zero ones

        # Collect callable reward function
        self.reward_names = []
        self.reward_functions = []
        for name in self.reward_scales.keys():

            fn_name = f"_reward_{name}"
            fn = getattr(self.reward_container, fn_name, None)

            if fn is not None:
                self.reward_names.append(name)
                self.reward_functions.append(fn)
            else:
                print(f"[WARNING] Reward '{fn_name}' not found")

        # print(f"reward func name: {self.reward_names}")

        # Episode tracking buffers
        # self.episode_sums = {
        #     name: torch.zeros(self.num_envs, device=self.device)
        #     for name in self.reward_scales
        # }
        self.episode_sums["energy"] = torch.zeros(self.num_envs, device=self.device)
        # self.episode_sums["tracking_lin"] = torch.zeros(self.num_envs, device=self.device)
        # self.episode_sums["tracking_ang"] = torch.zeros(self.num_envs, device=self.device)
        self.episode_sums["total"] = torch.zeros(self.num_envs, device=self.device)

    def _compute_reward(self):
        rew_energy = self._reward_energy_regularization()
        rew_lin = self._reward_tracking_lin_vel()
        rew_ang = self._reward_tracking_ang_vel()
        rew_smooth = self._reward_gait_smooth()

        # reward = rew_energy * (0.7 * rew_lin + 0.3 * rew_ang)

        # tracking = 0.7 * rew_lin + 0.3 * rew_ang
        # reward = tracking * (0.5 + 0.5 * rew_energy)

        # Regularizers
        # reward += -0.0003 * self.gaits_change
        # reward += -0.0010 * torch.mean(
        #     (self.env.actions - self.env.last_actions)**2, dim=1
        # )

        reward = rew_energy

        rew_log = torch.log(reward + 1e-6)
        reward = 3 * (rew_log - rew_log.mean().detach())
        reward = torch.clamp(reward, -2, 2)

        self.episode_sums["energy"] += rew_energy
        self.episode_sums["tracking_lin"] += rew_lin
        self.episode_sums["tracking_ang"] += rew_ang
        self.episode_sums["total"] += reward

        return reward

    def _reward_gait_smooth(self):
        ranges = self.gait_ranges[:, 1] - self.gait_ranges[:, 0]  # max - min
        ranges = torch.clamp(ranges, min=1e-6)

        gaits_norm = self.dgaits / ranges

        gait_smooth = torch.exp(-(gaits_norm ** 2).sum(dim=-1))
        return gait_smooth

    '''
    def _reward_energy_regularization(self, power_total, lin_vel_mean, ang_vel_mean):
        lin = torch.abs(lin_vel_mean[:, 0])
        ang = torch.abs(ang_vel_mean[:, 2])

        denom = (
            self.cfg.rewards.energy_sigma_lin * torch.clamp(lin, min=0.1)
            + self.cfg.rewards.energy_sigma_ang * torch.clamp(ang, min=0.1)
        )
        return torch.exp(-power_total / (denom + 1e-6))'''
    def _reward_energy_regularization(self, power_total, lin_vel_mean, ang_vel_mean):
        lin = torch.abs(lin_vel_mean[:, 0])
        ang = torch.abs(ang_vel_mean[:, 2])

        denom = (
            self.cfg.rewards.energy_sigma_lin * lin
            + self.cfg.rewards.energy_sigma_ang * ang
        )
        return torch.exp(-power_total / (denom + 1e-6))

    def _reward_tracking_lin_vel(self, lin_vel_mean):
        err = torch.sum(
            (self.current_vel_cmds[:, :2] - lin_vel_mean[:, :2]) ** 2,
            dim=1
        )
        return torch.exp(-err / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self, ang_vel_mean):
        err = (self.current_vel_cmds[:, 2] - ang_vel_mean[:, 2]) ** 2
        return torch.exp(-err / self.cfg.rewards.tracking_sigma_yaw)

    # def compute_CoT(self):
    #     # P / (mgv)
    #     P = torch.sum(torch.multiply(self.env.torques, self.env.dof_vel), dim=1)
    #     m = 21.5 # (env.default_body_mass + env.payloads).cpu()
    #     g = 9.81  # m/s^2
    #     v = torch.norm(self.env.base_lin_vel[:, 0:2], dim=1)
    #     return self.env.energy_consume / (m * g * v)

    def compute_CoT(self):
        m = 21.5
        g = 9.81

        # instantaneous mechanical power (always positive)
        power = torch.sum(torch.abs(self.env.torques * self.env.dof_vel), dim=1)

        # planar speed
        v = torch.norm(self.env.base_lin_vel[:, :2], dim=1)

        # avoid division blow-up
        v = torch.clamp(v, min=0.1)

        cot = power / (m * g * v)
        return cot

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_scales = self.cfg.noise_scales
        noise_level = self.cfg.noise.noise_level

        if self.cfg.env.observe_gravity:
            noise_vec = torch.ones(3) * noise_scales.gravity * noise_level

        # if self.cfg.env.observe_command:
        #     noise_vec = torch.cat((torch.ones(3) * noise_scales.gravity * noise_level,
        #                            torch.zeros(self.cfg.commands.num_commands)), dim=0)

        if self.cfg.env.observe_command:
            noise_vec = torch.zeros(self.cfg.commands.num_commands)

        # if self.cfg.env.observe_command:
        #     noise_vec = torch.cat((noise_vec,
        #                            torch.zeros(self.cfg.commands.num_commands)), dim=0)

        # noise_vec = torch.ones(3) * noise_scales.gravity * noise_level


        if self.cfg.env.observe_timing_parameter:
            noise_vec = torch.cat((noise_vec,
                                   torch.zeros(1)), dim=0)

        if self.cfg.env.observe_clock_inputs:
            noise_vec = torch.cat((noise_vec,
                                   torch.zeros(4)), dim=0)

        if self.cfg.env.observe_vel:
            noise_vec = torch.cat((torch.ones(3) * noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel,
                                   torch.ones(3) * noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel), dim=0)

        if self.cfg.env.observe_only_ang_vel:
            noise_vec = torch.cat((torch.ones(3) * noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel,
                                   noise_vec), dim=0)

        if self.cfg.env.observe_only_lin_vel:
            noise_vec = torch.cat((torch.ones(3) * noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel,
                                   noise_vec), dim=0)

        if self.cfg.env.observe_only_lin_vel_xy:
            noise_vec = torch.cat((torch.ones(2) * noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel,
                                   noise_vec), dim=0)

        if self.cfg.env.observe_only_ang_vel_z:
            noise_vec = torch.cat((torch.ones(1) * noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel,
                                   noise_vec), dim=0)

        if self.cfg.env.observe_yaw:
            noise_vec = torch.cat((noise_vec,
                                   torch.zeros(1)), dim=0)

        if self.cfg.env.observe_contact_states:
            noise_vec = torch.cat((noise_vec,
                                   torch.ones(4) * noise_scales.contact_states * noise_level), dim=0)

        if self.cfg.env.observe_foot_forces:
            noise_vec = torch.cat((noise_vec,
                                   torch.ones(12) * noise_scales.foot_forces * noise_level), dim=0)

        noise_vec = noise_vec.to(self.device)

        return noise_vec
