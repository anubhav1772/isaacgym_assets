import copy
import time
import os

import math
import select

import numpy as np
import torch
import lcm

import threading

from aliengo_gym_deploy.utils.logger import MultiLogger
from aliengo_gym_deploy.utils.HDF5_recorder import HDF5_recorder
from aliengo_gym_deploy.lcm_types.PositionGravityState import PositionGravityState

from tqdm import tqdm

lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")

class DeploymentRunner:
    def __init__(self, experiment_name="unnamed", se=None, log_root="."):
        self.agents = {}
        self.policy = None
        self.command_profile = None
        self.logger = MultiLogger()
        self.se = se
        self.vision_server = None

        self.log_root = log_root
        self.init_log_filename()
        self.control_agent_name = None
        self.command_agent_name = None

        self.triggered_commands = {i: None for i in range(4)}
        self.button_states = np.zeros(4)

        self.is_currently_probing = False
        self.is_currently_logging = [False, False, False, False]

        self.hdf5_recorder = HDF5_recorder()

    def init_log_filename(self):
        datetime = time.strftime("%Y/%m_%d/%H_%M_%S")

        for i in range(100):
            try:
                os.makedirs(f"{self.log_root}/{datetime}_{i}")
                self.log_filename = f"{self.log_root}/{datetime}_{i}/log.pkl"
                return
            except FileExistsError:
                continue

    def add_open_loop_agent(self, agent, name):
        self.agents[name] = agent
        self.logger.add_robot(name, agent.env.cfg)

    def add_control_agent(self, agent, name):
        self.control_agent_name = name
        self.agents[name] = agent
        self.logger.add_robot(name, agent.env.cfg)

    def add_vision_server(self, vision_server):
        self.vision_server = vision_server

    def set_command_agents(self, name):
        self.command_agent = name

    def add_policy(self, policy):
        self.policy = policy

    def add_command_profile(self, command_profile):
        self.command_profile = command_profile
    
    def poll(self, cb=None):
        t = time.time()
        try:
            while True:
                timeout = 0.01
                rfds, wfds, efds = select.select([lc.fileno()], [], [], timeout)
                if rfds:
                    lc.handle()
                else:
                    continue
        except KeyboardInterrupt:
            pass
    
    def spin(self):
        self.run_thread = threading.Thread(target=self.poll, daemon=False)
        self.run_thread.start()

    def calibrate(self, wait=True, low=False, start_controller=True):
        # first, if the robot is not in nominal pose, move slowly to the nominal pose
        for agent_name in self.agents.keys():
            if hasattr(self.agents[agent_name], "get_obs"):
                agent = self.agents[agent_name]
                agent.get_obs()
                joint_pos = agent.dof_pos
                if low:
                    final_goal = np.array([0., 0.3, -0.7,
                                           0., 0.3, -0.7,
                                           0., 0.3, -0.7,
                                           0., 0.3, -0.7,])
                else:
                    final_goal = np.zeros(12)
                nominal_joint_pos = agent.default_dof_pos

                print(f"About to calibrate; the robot will stand [Press R2 to calibrate]")
                
                if(not wait):
                    print("Dog reset!!!!")
                    self.hdf5_recorder.save_file()
                else:
                    print("Normally record")
                    self.hdf5_recorder.save_file()
                while wait:
                    self.button_states = self.command_profile.get_buttons()
                    if self.command_profile.state_estimator.right_lower_right_switch_pressed:
                        self.command_profile.state_estimator.right_lower_right_switch_pressed = False
                        break
                
                cal_action = np.zeros((agent.num_envs, agent.num_actions))
                target_sequence = []
                target = joint_pos - nominal_joint_pos
                while np.max(np.abs(target - final_goal)) > 0.01:
                    target -= np.clip((target - final_goal), -0.05, 0.05)
                    target_sequence += [copy.deepcopy(target)]
                for target in target_sequence:
                    next_target = target
                    if isinstance(agent.cfg, dict):
                        hip_reduction = agent.cfg["control"]["hip_scale_reduction"]
                        action_scale = agent.cfg["control"]["action_scale"]
                    else:
                        hip_reduction = agent.cfg.control.hip_scale_reduction
                        action_scale = agent.cfg.control.action_scale

                    next_target[[0, 3, 6, 9]] /= hip_reduction
                    next_target = next_target / action_scale
                    cal_action[:, 0:12] = next_target
                    is_calibrated = False
                    agent.step(torch.from_numpy(cal_action), calibrated=is_calibrated)
                    agent.get_obs()
                    time.sleep(0.05)
                if start_controller:
                    print("Starting pose calibrated [Press R2 to start controller]")
                    while True:
                        self.button_states = self.command_profile.get_buttons()
                        if self.command_profile.state_estimator.right_lower_right_switch_pressed:
                            self.command_profile.state_estimator.right_lower_right_switch_pressed = False
                            break

                    for agent_name in self.agents.keys():
                        obs = self.agents[agent_name].reset()
                        if agent_name == self.control_agent_name:
                            control_obs = obs

                    return control_obs

    def run(self, num_log_steps=1000, max_steps=10000, logging=True):
        assert self.control_agent_name is not None, "cannot deploy, runner has no control agent!"
        assert self.policy is not None, "cannot deploy, runner has no policy!"
        assert self.command_profile is not None, "cannot deploy, runner has no command profile!"

        for agent_name in self.agents.keys():
            obs = self.agents[agent_name].reset()
            if agent_name == self.control_agent_name:
                control_obs = obs

        self.gravity = None

        self.camera_data = None

        control_obs = self.calibrate(wait=True)
        obs_record = control_obs["obs"][0,:].detach().cpu().numpy().tolist()
        count = 0
        valuable_count = 0
        # now, run control loop
        try:
            with tqdm(total=max_steps) as pbar:
                iterations = 0
                while count < max_steps:
                    done = False

                    if count != 0:
                        control_obs = self.calibrate(wait=False, low=True)
                    obs_record = control_obs["obs"][0,:].detach().cpu().numpy().tolist()
                    history_obs = control_obs["obs_history"][0,:].detach().cpu().numpy()
                    
                    time_before_append = time.time()

                    obs_camera_subscribtion = lc.subscribe("camera_python", self._camera_cb)
                    self.spin()
                    if (self.camera_data is not None and len(self.camera_data.tolist()) != 18):
                        continue
                    
                    time_after_append = time.time()

                    while not done:
                        is_calibrated = True
                        policy_info = {}
                        action = self.policy(control_obs, policy_info)
                        act_record = action[0, :12].detach().cpu().numpy()

                        for agent_name in self.agents.keys():
                            obs, ret, _, info = self.agents[agent_name].step(action, calibrated=is_calibrated)
                            if agent_name == self.control_agent_name:
                                next_control_obs, control_ret, control_done, control_info = obs, ret, _, info
                                next_obs_record = next_control_obs["obs"][0,:].detach().cpu().numpy().tolist()
                                next_history_obs = next_control_obs["obs_history"][0,:].detach().cpu().numpy()

                        time_before_append = time.time()

                        if (self.camera_data is not None and len(self.camera_data.tolist()) != 18):
                            continue

                        time_after_append = time.time()
                        if(time_after_append-time_before_append>0.1):
                            count = valuable_count
                            break

                        rpy = self.agents[self.control_agent_name].se.get_rpy()
                        
                        if abs(rpy[0]) > 1.6 or abs(rpy[1]) > 1.6:
                            valuable_count = count
                            done = True
                        else:
                            done = False
                        if count == max_steps - 1:
                            done = True

                        if done and count != (max_steps - 1):
                            dw = True
                        else:
                            dw = False

                        obs_camera = self.camera_data.tolist()
                        while len(obs_camera) != 18:
                            continue

                        obs_record.extend(obs_camera)
                        if len(next_obs_record) == 58:
                            next_obs_record.extend(obs_camera)
                        self.hdf5_recorder.record_step(state=np.array(obs_record), action=act_record, next_state = np.array(next_obs_record), done=done)
                        count += 1
                        obs_record = next_obs_record[:58]
                        control_obs = next_control_obs
                        history_obs = next_history_obs
                        iterations += 1
                        pbar.update(1)

            # finally, return to the nominal pose
            _ = self.calibrate(wait=False, low=True, start_controller = False)
            
        except KeyboardInterrupt:
            self.logger.save(self.log_filename)


    def _camera_cb(self, channel, data):
        msg = PositionGravityState.decode(data)
        self.gravity = np.array(msg.gravity)
        self.camera_data = np.array(msg.data)
