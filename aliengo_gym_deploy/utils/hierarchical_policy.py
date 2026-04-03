import torch
import numpy as np
import pickle as pkl

from aliengo_gym.envs.gait.base.gait_config import GaitCfg

class HLWindowStats:
    def __init__(self, win_len):
        self.win_len = win_len
        self.sum_lin = torch.zeros(3)
        self.sum_ang = torch.zeros(3)
        self.count = 0

    def reset(self):
        self.sum_lin.zero_()
        self.sum_ang.zero_()
        self.count = 0

    def update(self, lin_vel, ang_vel):
        if self.count < self.win_len:
            self.sum_lin += lin_vel
            self.sum_ang += ang_vel
            self.count += 1
    
    def mean(self):
        if self.count == 0:
            return torch.zeros(3), torch.zeros(3)
        return self.sum_lin / self.count, self.sum_ang / self.count

class HierarchicalPolicy:
    """
    Deployment-ready HRL controller.
    HL runs occasionally, LL runs every step.
    """

    def __init__(self, pi1_dir, pi2_dir, hl_decimation, win_len, device="cuda:0"):
        self.device = device
        self.hl_decimation = hl_decimation
        self.step_counter = 0

        # HL window statistics
        self.stats = HLWindowStats(win_len)

        # Load HL policy
        self._load_pi1(pi1_dir)

        # Load LL policy
        self.pi2 = self.load_pi2(pi2_dir)

        self.current_gaits = None

    # HL
    def _get_gait_ranges(self):
        gait_cfg = GaitCfg.gaits
        return torch.tensor([
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
            gait_cfg.limit_stance_length,
            gait_cfg.limit_aux_reward_coef,
        ], device=self.device)

    def _load_pi1(self, pi1_dir):
        self.pi1_body = torch.jit.load(f"{pi1_dir}/checkpoints/gaits/body_latest.jit").to(self.device)
        self.pi1_adapt = torch.jit.load(f"{pi1_dir}/checkpoints/gaits/adaptation_module_latest.jit").to(self.device)
        self.gait_ranges = self._get_gait_ranges()

    @torch.no_grad()
    def _run_pi1(self, obs_history):
        latent = self.pi1_adapt(obs_history)
        mean_raw = self.pi1_body(torch.cat((obs_history, latent), dim=-1))
        a_norm = torch.tanh(mean_raw)

        lows = self.gait_ranges[:, 0]
        highs = self.gait_ranges[:, 1]
        return lows + 0.5 * (a_norm + 1.0) * (highs - lows)

    # LL
    def load_pi2(self, pi2_dir):
        body = torch.jit.load(f"{pi2_dir}/checkpoints/body_latest.jit").to(self.device)
        adaptation = torch.jit.load(f"{pi2_dir}/checkpoints/adaptation_module_latest.jit").to(self.device)

        def pi2(obs_dict, info):
            obs_h = obs_dict["obs_history"].to(self.device)
            latent = adaptation(obs_h)
            act = body(torch.cat((obs_h, latent), dim=-1))
            info["latent"] = latent
            return act

        return pi2

    def reset(self):
        self.step_counter = 0
        self.current_gaits = None
        self.stats.reset()

    @torch.no_grad()
    def __call__(self, obs_dict, lin_vel, ang_vel, info):
        """
        Called every LL control step (e.g. 50 Hz).
        """

        # Update window statistics (LL-rate)
        self.stats.update(lin_vel, ang_vel)

        # HL step
        if self.step_counter % self.hl_decimation == 0 or self.current_gaits is None:
            lin_mean, ang_mean = self.stats.mean()

            # Build HL observation exactly like training
            obs_dict["lin_vel_mean"] = lin_mean.unsqueeze(0)
            obs_dict["ang_vel_mean"] = ang_mean.unsqueeze(0)

            self.current_gaits = self._run_pi1(obs_dict["obs_history"])
            info["hl_update"] = True
            info["gaits"] = self.current_gaits.detach().cpu().numpy()

            self.stats.reset()
        else:
            info["hl_update"] = False

        # Inject gaits for LL
        obs_dict["gaits"] = self.current_gaits

        action = self.pi2(obs_dict, info)
        self.step_counter += 1
        return action
