# License: see [LICENSE, LICENSES/legged_gym/LICENSE]

from params_proto import PrefixProto, ParamsProto


class BaseCfg(PrefixProto, cli=False):
    class curriculum_thresholds(PrefixProto, cli=False):
        tracking_lin_vel = 0.8  # closer to 1 is tighter
        tracking_ang_vel = 0.7
        tracking_contacts_shaped_force = 0.9  # closer to 1 is tighter
        tracking_contacts_shaped_vel = 0.9

    # class obstacles:
    #     enable = True
    #     num_per_env = 3          # how many boxes per environment
    #     box_size = [0.4, 0.4, 0.8]  # [w, d, h] in meters
    #     randomize_pos = True
    #     x_range = [1.0, 5.0]
    #     y_range = [-2.5, 2.5]
    #     fixed_z = 0.4
    #     color = [0.8, 0.3, 0.1]  # RGB

    class init_state(PrefixProto, cli=False):
        pos = [0.0, 0.0, 0.51]  # x,y,z [m] #.6
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = {
            # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,  # [rad]
            'RL_hip_joint': 0.1,  # [rad]
            'FR_hip_joint': -0.1,  # [rad]
            'RR_hip_joint': -0.1,  # [rad]

            'FL_thigh_joint': 0.8,  # [rad]
            'RL_thigh_joint': 1.,  # [rad]
            'FR_thigh_joint': 0.8,  # [rad]
            'RR_thigh_joint': 1.,  # [rad]

            'FL_calf_joint': -1.5,  # [rad]
            'RL_calf_joint': -1.5,  # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5  # [rad]
        }

    class control(PrefixProto, cli=False):
        control_type = 'P' #'P'  # P: position, V: velocity, T: torques # actuator_net
        # PD Drive parameters:
        stiffness = {'joint': 30.}  # [N*m/rad]
        damping = {'joint': 0.8}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        hip_scale_reduction = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset(PrefixProto, cli=False):
        file = '{MINI_GYM_ROOT_DIR}/resources/robots/aliengo/urdf/aliengo.urdf'
        foot_name = "foot"  # name of the feet bodies, used to index body state and contact force tensors
        penalize_contacts_on = ["thigh", "calf"]                              # WTW | WTW-NEW-ADAPTIVE-FLAT
        # penalize_contacts_on = ["calf"]                                     # WTW-NEW-ADAPTIVE-TERRAIN
        # terminate_after_contacts_on = ["base", "thigh", "hip"]              # WTW-NEW-ADAPTIVE-TERRAIN
        terminate_after_contacts_on = ["base"]                                # WTW
        #terminate_after_contacts_on = ["base", "hip"]                        # WTW-NEW-ADAPTIVE-FLAT
        disable_gravity = False
        # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        collapse_fixed_joints = True
        fix_base_link = False  # fixe the base of the robot
        default_dof_drive_mode = 3  # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        # replace collision cylinders with capsules, leads to faster/more stable simulation
        replace_cylinder_with_capsule = True
        flip_visual_attachments =  False # Some .obj meshes must be flipped from y-up to z-up

        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class env(PrefixProto, cli=False):
        num_envs = 1
        num_observations = 58
        num_scalar_observations = 58
        # if not None a privilige_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise
        num_privileged_obs = 2 # friction, restitution, (possibly heightmap)
        privileged_future_horizon = 1
        observe_gaits = True
        num_actions = 12
        num_observation_history = 5 # 30 (WTW)
        env_spacing = 3.       # not used with heightfields/trimeshes
        send_timeouts = True   # send time out information to the algorithm
        episode_length_s = 20  # episode length in seconds
        observe_vel = False
        observe_only_ang_vel = False
        observe_only_lin_vel = False
        observe_yaw = False
        observe_contact_states = False
        #observe_foot_forces = True
        observe_command = True
        observe_height_command = False
        observe_gait_commands = True
        observe_timing_parameter = False # NOT that useful when already using clock_inputs which contains more info
        observe_clock_inputs = True
        observe_two_prev_actions = False
        observe_imu = False
        record_video = True
        recording_width_px = 360
        recording_height_px = 240
        recording_mode = "COLOR"
        num_recording_envs = 1
        debug_viz = False
        all_agents_share = False

        priv_observe_friction = True
        priv_observe_friction_indep = False
        priv_observe_ground_friction = False
        priv_observe_ground_friction_per_foot = False
        priv_observe_restitution = True
        priv_observe_base_mass = False
        priv_observe_com_displacement = False
        priv_observe_motor_strength = False
        priv_observe_motor_offset = False
        priv_observe_joint_friction = True
        priv_observe_Kp_factor = False
        priv_observe_Kd_factor = False
        priv_observe_contact_forces = False
        priv_observe_contact_states = False
        priv_observe_body_velocity = False
        priv_observe_foot_height = False
        priv_observe_body_height = False
        priv_observe_gravity = False
        priv_observe_terrain_type = False
        priv_observe_clock_inputs = False
        priv_observe_doubletime_clock_inputs = False
        priv_observe_halftime_clock_inputs = False
        priv_observe_desired_contact_states = False
        priv_observe_dummy_variable = False
        priv_observe_motion = False
        priv_observe_gravity_transformed_motion = False
        priv_observe_foot_displacement = False
        priv_observe_gravity_transformed_foot_displacement = False

    class terrain(PrefixProto, cli=False):
        mesh_type = "trimesh" # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 0  # 50 WTW # [m]
        curriculum = False
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0
        terrain_noise_magnitude = 0.0
        # rough terrain only:
        terrain_smoothness = 0.005
        measure_heights = False
        # 1mx1.6m rectangle (without center line)
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False  # select a unique terrain type and pass all arguments
        terrain_kwargs = None  # Dict of arguments for selected terrain
        min_init_terrain_level = 0
        max_init_terrain_level = 5  # starting curriculum state
        terrain_length = 5.
        terrain_width = 5.
        num_rows = 30 #10  # number of terrain rows (levels)
        num_cols = 30 #10  # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0, 0, 0, 0, 0, 0, 0, 0, 1.0]
        # trimesh only:
        slope_treshold = 0.75  # slopes above this threshold will be corrected to vertical surfaces
        difficulty_scale = 1.
        x_init_range = 0.2
        y_init_range = 0.2
        yaw_init_range = 3.14
        x_init_offset = 0.
        y_init_offset = 0.
        teleport_robots = False
        teleport_thresh = 0.3
        max_platform_height = 0.2
        center_robots = True
        center_span = 4

    class commands(PrefixProto, cli=False):
        command_curriculum = True
        max_reverse_curriculum = 1.
        max_forward_curriculum = 1.
        yaw_command_curriculum = False
        max_yaw_curriculum = 1.
        exclusive_command_sampling = False
        num_commands = 15
        resampling_time = 10.  # time before command are changed[s]
        subsample_gait = False
        gait_interval_s = 10.  # time between resampling gait params
        vel_interval_s = 10.
        jump_interval_s = 20.  # time between jumps
        jump_duration_s = 0.1  # duration of jump
        jump_height = 0.3
        heading_command = False  # if true: compute ang vel command from heading error
        global_reference = False
        observe_accel = False
        distributional_commands = True #(when training)
        curriculum_type = "RewardThresholdCurriculum"
        lipschitz_threshold = 0.9

        num_lin_vel_bins = 30
        lin_vel_step = 0.3
        num_ang_vel_bins = 30
        ang_vel_step = 0.3
        distribution_update_extension_distance = 1
        curriculum_seed = 100

        impulse_height_commands = False

        lin_vel_x = [-2.0, 2.0]                  # min max [m/s]
        lin_vel_y = [-0.75, 0.75]                  # min max [m/s]
        ang_vel_yaw = [-1.5, 1.5]                # min max [rad/s]
        body_height_cmd = [-0.25, 0.15]
        gait_frequency_cmd_range = [1.5, 4.0]
        gait_phase_cmd_range = [0.0, 0.5]
        gait_offset_cmd_range = [0.0, 0.5]
        gait_bound_cmd_range = [0.0, 0.5]
        gait_duration_cmd_range = [0.35, 0.5] #[0.35, 0.70]
        footswing_height_range = [0.03, 0.35]
        body_pitch_range = [-0.4, 0.4]
        body_roll_range = [-0.0, 0.0]
        stance_width_range = [0.10, 0.45]
        stance_length_range = [0.35, 0.45]
        aux_reward_coef_range = [0.0, 0.01]
        compliance_range = [0.0, 0.01]

        limit_vel_x = [-5.0, 5.0]           # [-1.8, 1.8]
        limit_vel_y = [-1.0, 1.0]
        limit_vel_yaw = [-5.0, 5.0]
        limit_body_height = [-0.25, 0.15]   # [-0.20, 0.20]
        limit_gait_frequency = [1.5, 4.0]   # [2.0, 4.0]
        limit_gait_phase = [0.0, 0.5]
        limit_gait_offset = [0.0, 0.5]
        limit_gait_bound = [0.0, 0.5]
        limit_gait_duration = [0.35, 0.5] #[0.35, 0.70]
        limit_footswing_height = [0.03, 0.35]
        limit_body_pitch = [-0.4, 0.4]
        limit_body_roll = [-0.0, 0.0]
        limit_stance_width = [0.10, 0.45]    # [0.15, 0.50]
        limit_stance_length =  [0.35, 0.45]  # [0.30, 0.55]

        limit_aux_reward_coef = [0.0, 0.01]
        limit_compliance = [0.0, 0.01]

        num_bins_vel_x = 21
        num_bins_vel_y = 1
        num_bins_vel_yaw = 21
        num_bins_body_height = 1
        num_bins_gait_frequency = 1
        num_bins_gait_phase = 1
        num_bins_gait_offset = 1
        num_bins_gait_bound = 1
        num_bins_gait_duration = 1
        num_bins_footswing_height = 1
        num_bins_body_pitch = 1
        num_bins_body_roll = 1
        num_bins_aux_reward_coef = 1
        num_bins_compliance = 1
        num_bins_compliance = 1
        num_bins_stance_width = 1
        num_bins_stance_length = 1

        heading = [-3.14, 3.14]

        exclusive_phase_offset = False
        binary_phases = True
        pacing_offset = False
        balance_gait_distribution = True
        gaitwise_curricula = True

    class domain_rand(PrefixProto, cli=False):
        rand_interval_s = 4
        randomize_rigids_after_start = False
        randomize_friction = True
        friction_range = [0.1, 3.0]  # increase range
        randomize_restitution = True
        restitution_range = [0, 0.4]
        randomize_base_mass = True
        # add link masses, increase range, randomize inertia, randomize joint properties
        added_mass_range = [-1., 3.]
        randomize_com_displacement = False
        # add link masses, increase range, randomize inertia, randomize joint properties
        com_displacement_range = [-0.15, 0.15]
        randomize_motor_strength = True
        motor_strength_range = [0.9, 1.1]
        randomize_Kp_factor = False
        Kp_factor_range = [0.8, 1.3]
        randomize_Kd_factor = False
        Kd_factor_range = [0.5, 1.5]
        gravity_rand_interval_s = 8.0
        gravity_impulse_duration = 0.99
        randomize_gravity = True
        gravity_range = [-1.0, 1.0]
        push_robots = False
        push_interval_s = 15
        max_push_vel_xy = 0.5
        max_push_vel_z = 0.5
        max_push_ang_rpy = 0.5
        randomize_lag_timesteps = True
        lag_timesteps = 6
        restitution = 0.5  # default terrain restitution
        randomize_friction_indep = False
        randomize_ground_friction = True
        ground_friction_range = [0.0, 1.0]
        randomize_motor_offset = True
        motor_offset_range = [-0.02, 0.02]
        rand_interval_s = 4
        tile_height_range = [-0.0, 0.0]
        tile_height_curriculum = False
        tile_height_update_interval = 1000000
        tile_height_curriculum_step = 0.01
        randomize_joint_friction = True

    class rewards(PrefixProto, cli=False):
        only_positive_rewards = False  # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards_ji22_style = True
        sigma_rew_neg = 0.02
        reward_container_name = "CoRLRewards"
        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        tracking_sigma_lat = 0.25  # tracking reward = exp(-error^2/sigma)
        tracking_sigma_long = 0.25  # tracking reward = exp(-error^2/sigma)
        tracking_sigma_yaw = 0.25  # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.9  # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.30
        max_contact_force = 100.  # forces above this value are penalized
        use_terminal_body_height = True
        terminal_body_height = 0.05
        use_terminal_foot_height = False
        terminal_foot_height = -0.005
        use_terminal_roll_pitch = True
        terminal_body_ori = 1.6
        kappa_gait_probs = 0.07
        gait_force_sigma = 100.
        gait_vel_sigma = 10.
        footswing_height = 0.09

    class reward_scales(ParamsProto, cli=False):
        termination = -0.0
        tracking_lin_vel = 1.0
        tracking_ang_vel = 0.5
        lin_vel_z = -0.02
        ang_vel_xy = -0.001
        orientation = 0.0
        torques = -0.0001
        dof_vel = -0.0001
        dof_acc = -2.5e-07
        base_height = 0.0
        feet_air_time = 0.0
        collision = -5.0
        feet_stumble = -0.0
        action_rate = -0.01
        stand_still = -4.0
        tracking_lin_vel_lat = 0.0
        tracking_lin_vel_long = 0.0
        tracking_contacts = 0.0
        tracking_contacts_shaped = 0.0
        tracking_contacts_shaped_force = 4.0
        tracking_contacts_shaped_vel = 4.0
        jump = 10.0
        energy = 0.0
        energy_expenditure = 0.0
        survival = 0.0
        dof_pos_limits = -10.0
        feet_contact_forces = 0.0
        feet_slip = -0.04
        feet_clearance_cmd_linear = -30.0
        dof_pos = -0.0
        action_smoothness_1 = -0.1
        action_smoothness_2 = -0.1
        base_motion = 0.0
        feet_impact_vel = -0.0
        raibert_heuristic = -5.0 #-10.0
        estimation_bonus = 0.0
        feet_clearance = -0.0
        feet_clearance_cmd = -0.0
        orientation_control = -5.0
        tracking_stance_width = -0.0
        tracking_stance_length = -0.0
        hop_symmetry = 0.0

    class normalization(PrefixProto, cli=False):
        clip_observations = 100.
        clip_actions = 10.

        friction_range = [0, 1.0]
        ground_friction_range = [1, 1.0]
        restitution_range = [0, 1.0]
        added_mass_range = [-1., 3.]
        com_displacement_range = [-0.1, 0.1]
        motor_strength_range = [0.9, 1.1]
        motor_offset_range = [-0.05, 0.05]
        Kp_factor_range = [0.8, 1.3]
        Kd_factor_range = [0.5, 1.5]
        joint_friction_range = [0.0, 0.7]
        contact_force_range = [0.0, 50.0]
        contact_state_range = [0.0, 1.0]
        body_velocity_range = [-6.0, 6.0]
        foot_height_range = [0.0, 0.15]
        body_height_range = [0.0, 0.60]
        gravity_range = [-1.0, 1.0]
        motion = [-0.01, 0.01]

    class obs_scales(PrefixProto, cli=False):
        lin_vel = 2.0
        ang_vel = 0.25
        dof_pos = 1.0
        dof_vel = 0.05
        dof_torque = 0.02
        imu = 0.1
        height_measurements = 5.0
        friction_measurements = 1.0
        body_height_cmd = 2.0
        gait_phase_cmd = 1.0
        gait_freq_cmd = 1.0
        footswing_height_cmd = 0.15
        body_pitch_cmd = 0.3
        body_roll_cmd = 0.3
        aux_reward_cmd = 1.0
        compliance_cmd = 1.0
        stance_width_cmd = 1.0
        stance_length_cmd = 1.0
        segmentation_image = 1.0
        rgb_image = 1.0
        depth_image = 1.0

    class noise(PrefixProto, cli=False):
        add_noise = True
        noise_level = 1.0  # scales other values

    class noise_scales(PrefixProto, cli=False):
        dof_pos = 0.01
        dof_vel = 1.5
        dof_torque = 0.01
        lin_vel = 0.1
        ang_vel = 0.2
        imu = 0.1
        gravity = 0.05
        contact_states = 0.05
        foot_forces = 0.05
        height_measurements = 0.1
        friction_measurements = 0.0
        segmentation_image = 0.0
        rgb_image = 0.0
        depth_image = 0.0

    # viewer camera:
    class viewer(PrefixProto, cli=False):
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]

    class sim(PrefixProto, cli=False):
        dt = 0.005
        substeps = 1
        gravity = [0., 0., -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        use_gpu_pipeline = True

        class physx(PrefixProto, cli=False):
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.5  # 0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2 ** 23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2  # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
