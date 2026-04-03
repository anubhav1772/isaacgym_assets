import isaacgym
assert isaacgym
import torch
import gym

class HistoryWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

        self.obs_history_length = self.env.cfg.env.num_observation_history

        self.num_obs_history = self.obs_history_length * self.num_obs
        self.obs_history = torch.zeros(self.env.num_envs, self.num_obs_history, dtype=torch.float,
                                       device=self.env.device, requires_grad=False)
        self.num_privileged_obs = self.num_privileged_obs

    def step(self, action):
        # privileged information and observation history are stored in info
        obs, rew, done, info = self.env.step(action)
        privileged_obs = info["privileged_obs"]

        self.obs_history = torch.cat((self.obs_history[:, self.env.num_obs:], obs), dim=-1)
        return {'obs': obs, 'privileged_obs': privileged_obs, 'obs_history': self.obs_history}, rew, done, info

    def get_observations(self):
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        self.obs_history = torch.cat((self.obs_history[:, self.env.num_obs:], obs), dim=-1)
        return {'obs': obs, 'privileged_obs': privileged_obs, 'obs_history': self.obs_history}

    def reset_idx(self, env_ids):  # it might be a problem that this isn't getting called!!
        ret = self.env.reset_idx(env_ids)
        self.obs_history[env_ids, :] = 0
        return ret

    def reset(self):
        ret = super().reset()
        privileged_obs = self.env.get_privileged_observations()
        self.obs_history[:, :] = 0
        return {"obs": ret, "privileged_obs": privileged_obs, "obs_history": self.obs_history}


# class HistoryWrapper(gym.Wrapper):
#     def __init__(self, env):
#         super().__init__(env)
#         self.env = env

#         self.obs_history_length = self.env.cfg.env.num_observation_history
#         self.num_obs_history = self.obs_history_length * self.num_obs

#         self.obs_history = torch.zeros(
#             self.env.num_envs,
#             self.num_obs_history,
#             dtype=torch.float,
#             device=self.env.device,
#             requires_grad=False,
#         )

#     def step(self, action):
#         obs, rew, done, info = self.env.step(action)
#         privileged_obs = info["privileged_obs"]

#         alive = ~done

#         self.obs_history[alive] = torch.cat(
#             (self.obs_history[alive, self.env.num_obs:], obs[alive]),
#             dim=-1
#         )

#         return {
#             "obs": obs,
#             "privileged_obs": privileged_obs,
#             "obs_history": self.obs_history,
#         }, rew, done, info

#     def get_observations(self):
#         obs = self.env.get_observations()
#         privileged_obs = self.env.get_privileged_observations()

#         if not getattr(self.env, "defer_reset", False):
#             alive = ~self.env.reset_buf
#             self.obs_history[alive] = torch.cat(
#                 (self.obs_history[alive, self.env.num_obs:], obs[alive]),
#                 dim=-1
#             )

#         return {
#             "obs": obs,
#             "privileged_obs": privileged_obs,
#             "obs_history": self.obs_history,
#         }

#     def reset_idx(self, env_ids):
#         if not hasattr(self.env, "reset_idx"):
#             raise RuntimeError("Underlying env has no reset_idx")

#         self.env.reset_idx(env_ids)

#         # THIS is the only place history is cleared
#         self.obs_history[env_ids] = 0.0

#     def reset(self):
#         obs = self.env.reset()
#         privileged_obs = self.env.get_privileged_observations()
#         self.obs_history.zero_()

#         return {
#             "obs": obs,
#             "privileged_obs": privileged_obs,
#             "obs_history": self.obs_history,
#         }


# if __name__ == "__main__":
#     from tqdm import trange
#     import matplotlib.pyplot as plt

#     import ml_logger as logger

#     from aliengo_gym_learn.ppo import Runner
#     from aliengo_gym.envs.wrappers.history_wrapper import HistoryWrapper
#     from aliengo_gym_learn.ppo.actor_critic import AC_Args

#     from aliengo_gym.envs.base.legged_robot_config import Cfg
#     from aliengo_gym.envs.mini_cheetah.mini_cheetah_config import config_mini_cheetah
#     config_mini_cheetah(Cfg)

#     test_env = gym.make("VelocityTrackingEasyEnv-v0", cfg=Cfg)
#     env = HistoryWrapper(test_env)

#     env.reset()
#     action = torch.zeros(test_env.num_envs, 12)
#     for i in trange(3):
#         obs, rew, done, info = env.step(action)
#         print(obs.keys())
#         print(f"obs: {obs['obs']}")
#         print(f"privileged obs: {obs['privileged_obs']}")
#         print(f"obs_history: {obs['obs_history']}")

#         img = env.render('rgb_array')
#         plt.imshow(img)
#         plt.show()
