import glob
import pickle as pkl
import lcm
import sys

from aliengo_gym_deploy.utils.deployment_runner import DeploymentRunner
from aliengo_gym_deploy.envs.lcm_agent import LCMAgent
from aliengo_gym_deploy.utils.cheetah_state_estimator import StateEstimator
from aliengo_gym_deploy.utils.command_profile import RCControllerProfile
import inspect
import torch

import pathlib

lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")

def load_and_run_policy(label, experiment_name, args, max_vel=1.0, max_yaw_vel=1.0):
    logdir = "/home/anubhav1772/Github/unio4-efficient-aliengo/WTW-Aliengo/runs/gait-conditioned-agility/2025-10-29/train/034208.466438"

    with open(logdir+"/parameters.pkl", 'rb') as file:
        pkl_cfg = pkl.load(file)
        cfg = pkl_cfg["Cfg"]

    se = StateEstimator(lc)

    control_dt = 0.02

    seconds_to_end = 5
    max_steps = 50 * seconds_to_end
    print(f'max steps {max_steps}')

    command_profile = RCControllerProfile(dt=control_dt, state_estimator=se, x_scale=max_vel, y_scale=1.0, yaw_scale=max_yaw_vel, max_steps=max_steps)

    hardware_agent = LCMAgent(cfg, se, command_profile)
    se.spin()

    from aliengo_gym_deploy.envs.history_wrapper import HistoryWrapper
    hardware_agent = HistoryWrapper(hardware_agent)
    if args.deploy_policy == 'sim':
        policy = load_policy_sim(logdir)
    elif args.deploy_policy == 'offline':
        policy = load_policy_offline(logdir)
    elif args.deploy_policy == 'online':
        policy = load_policy_online(logdir)

    root = f"{pathlib.Path(__file__).parent.resolve()}/../../logs/"
    pathlib.Path(root).mkdir(parents=True, exist_ok=True)
    deployment_runner = DeploymentRunner(experiment_name=experiment_name, se=None,
                                         log_root=f"{root}/{experiment_name}")
    deployment_runner.add_control_agent(hardware_agent, "hardware_closed_loop")
    deployment_runner.add_policy(policy)
    deployment_runner.add_command_profile(command_profile)

    deployment_runner.run(max_steps=max_steps, logging=True)

# 1-----for online fine-tuned policy deployment-----
def load_policy_online(logdir):
    print("Loading ONLINE policy...")
    from bppo import BehaviorCloning
    bc = BehaviorCloning("cuda:0", 58 * 5, [512, 256, 128], 3, 12, 1e-4, 512)
    bc.load(logdir + '/online_finetuned/pi_latest.pt')

    def policy(obs, info):
        action = bc._policy.mean(obs["obs_history"].to('cuda:0'))
        info['latent'] = 0
        return action
    return policy

# 1-----for offline fine-tuned policy deployment-----
def load_policy_offline(logdir):
    print("Loading OFFLINE policy...")
    from bppo import BehaviorCloning
    bc = BehaviorCloning("cuda:0", 58 * 5, [512, 256, 128], 3, 12, 1e-4, 512)
    bc.load(logdir + 'pi_off.pt')

    def policy(obs, info):
        action = bc._policy.mean(obs["obs_history"].to('cuda:0'))
        info['latent'] = 0
        return action
    return policy

# 0-----collect data using pre-trained policy in simulator (1,000,000 steps)-----
def load_policy_sim(logdir):
    from actor_critic import ActorCritic
    actor_critic = ActorCritic(58, 0, 5 * 58, 12) # num obs; num privileged obs; num history; num actions
    load_dir = logdir + '/checkpoints/ac_weights_last.pt'
    print('policy loaded from: {}'.format(str(load_dir)))
    weights = torch.load(logdir + '/checkpoints/ac_weights_last.pt')

    actor_critic.load_state_dict(state_dict=weights)
    actor_critic.to('cuda:0')
    def sample_policy(obs, info):
        action = actor_critic.mean(obs["obs_history"].to('cuda:0'))
        info['latent'] = 0
        return action
    return sample_policy

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-continuous")
    parser.add_argument("--deploy_policy", type=str, default='sim', help="choice: sim/offline/online trained policy")

    args = parser.parse_args()
    label = "gait-conditioned-agility/itmo/train"
    experiment_name = "aliengo_offline_dataset"
    load_and_run_policy(label, experiment_name=experiment_name, args=args, max_vel=1.0, max_yaw_vel=0.0)
