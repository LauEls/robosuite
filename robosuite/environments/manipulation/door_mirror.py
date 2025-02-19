from collections import OrderedDict
import numpy as np

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv

from robosuite.models.arenas import TableArena
from robosuite.models.objects import DoorObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.observables import Observable, sensor

import robosuite.macros as macros

from robosuite.controllers.gh360t_equilibrium_point import GH360TEquilibriumPointController


class DoorMirror(SingleArmEnv):
    """
    This class corresponds to the door opening task for a single robot arm.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!

        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        use_latch (bool): if True, uses a spring-loaded handle and latch to "lock" the door closed initially
            Otherwise, door is instantiated with a fixed handle

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        placement_initializer (ObjectPositionSampler): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

        camera_segmentations (None or str or list of str or list of list of str): Camera segmentation(s) to use
            for each camera. Valid options are:

                `None`: no segmentation sensor used
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level

            If not None, multiple types of segmentations can be specified. A [list of str / str or None] specifies
            [multiple / a single] segmentation(s) to use for all cameras. A list of list of str specifies per-camera
            segmentation setting(s) to use.

    Raises:
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        use_latch=True,
        table_offset=(0,0,0),
        action_punishment=False,
        force_punishment=False,
        stiffness_punishment=False,
        motor_obs = False,
        grasp_check=False,
        obs_optimization=False,
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=True,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,      # {None, instance, class, element}
        renderer="mujoco",
        renderer_config=None,
    ):
        # settings for table top (hardcoded since it's not an essential part of the environment)
        # self.table_full_size = (0.8, 0.3, 0.05)
        self.table_full_size = (0.8, 0.3, 0.05)
        # self.table_offset = (-0.2, -0.35, 0.8)
        self.table_offset = table_offset#(-0.2, 0.55, 0.8) #(-0.2, 0.35, 0.8)
        #self.table_offset = (-0.1, 0.35, 0.8)
        # self.resting_pos = [-0.26256637, 0.35293338, 1.04808937]
        self.resting_pos = [-0.36193448,  0.47809581, -0.06899795]
        self.resting_q_pos = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.action_punishment = action_punishment
        self.force_punishment = force_punishment
        self.stiffness_punishment = stiffness_punishment
        self.current_stiffness = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.grasp_check = grasp_check
        self.motor_obs = motor_obs
        # reward configuration
        self.use_latch = use_latch
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        self.obs_optimization = obs_optimization

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )

    def reward(self, action=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of 1.0 is provided if the door is opened

        Un-normalized summed components if using reward shaping:

            - Reaching: in [0, 0.25], proportional to the distance between door handle and robot arm
            - Rotating: in [0, 0.25], proportional to angle rotated by door handled
              - Note that this component is only relevant if the environment is using the locked door version

        Note that a successfully completed task (door opened) will return 1.0 irregardless of whether the environment
        is using sparse or shaped rewards

        Note that the final reward is normalized and scaled by reward_scale / 1.0 as
        well so that the max score is equal to reward_scale

        Args:
            action (np.array): [NOT USED]

        Returns:
            float: reward value
        """
        
        reward = 0.0
        force_punishment_scale = 0.1
        stiffness_punishment_scale = 0.1
        # print("action = ",action)
        # print("handle pos: ", self._handle_xpos)
        # print("door pos: ", self.sim.data.body_xpos[self.object_body_ids["door"]])
        # print("robot base pos: ", self.sim.data.get_body_xpos("robot0_base"))
        # print("shoulder0 pos: ", self.sim.data.get_body_xpos("robot0_shoulder0"))
        # print("eef pos: ", self._eef_xpos)

        # print(0.2 * (1- np.tanh(0.1*np.linalg.norm(self.resting_q_pos - self._joint_pos))))

        # sparse completion reward
        if self._check_success():
            reward = 1.0
            # reward += 0.2 * (1- np.tanh(10.0*np.linalg.norm(self.resting_pos - self._eef_xpos)))
            #reward += 0.2 * (1- np.tanh(1*np.linalg.norm(self.resting_q_pos - self._joint_pos)))

            #force_punishment_scale = 0.1

        # else, we consider only the case if we're using shaped rewards
        elif self.reward_shaping:
            # Add reaching component
            dist = np.linalg.norm(self._gripper_to_handle)
            reaching_reward = 0.2 * (1 - np.tanh(10.0 * dist))
            reward += reaching_reward
            # Add rotating component if we're using a locked door
            
            if self.grasp_check:
                if self._check_grasp(gripper=self.robots[0].gripper, object_geoms="handle"):
                    reward += 0.2

                    if self.use_latch:
                        handle_qpos = self.sim.data.qpos[self.handle_qpos_addr]
                        reward += np.clip(0.2 * np.abs(handle_qpos / (0.25 * np.pi)), -0.2, 0.2)
            else:
                if self.use_latch:
                    handle_qpos = self.sim.data.qpos[self.handle_qpos_addr]
                    hinge_qpos = self.sim.data.qpos[self.hinge_qpos_addr]
                    # reward += np.clip(0.25 * np.abs(handle_qpos / (0.25 * np.pi)), -0.25, 0.25)
                    if np.abs(handle_qpos) >= 0.1 and self._check_grasp(gripper=self.robots[0].gripper, object_geoms="handle"):
                        reward = 0.25
                        reward += np.clip(0.25 * np.abs(handle_qpos / (0.25 * np.pi)), 0.0, 0.25)
                    if np.abs(hinge_qpos) >= 0.1:
                        reward = 0.5
                        reward += np.clip(0.25 * np.abs(hinge_qpos / 0.3), 0, 0.25)



        # Scale reward if requested
        if self.reward_scale is not None:
            reward *= self.reward_scale / 1.0

        if self.action_punishment:
            for a in action:
                if a != 0:
                    reward -= 0.01
                    break

        if self.force_punishment:
            # print("Actuators: "+str(self.sim.data.qfrc_actuator))
            # print("Actuators: "+str(self.sim.data.ctrl))
            # print("Actuators: "+str(self.sim.data.actuator_force))
            percent = 0
            n_actuators = len(self.sim.data.ctrl)
            for actuator in range(n_actuators): #can maybe also use self.sim.data.ctrl, self.sim.data.actuator_force
                # print("actuator min/max: "+str(self.sim.model.actuator_ctrlrange[actuator]))
                min_force = self.sim.model.actuator_ctrlrange[actuator][0]
                max_force = self.sim.model.actuator_ctrlrange[actuator][1]
                force = self.sim.data.ctrl[actuator]
                # force = self.sim.data.qfrc_actuator[joint]
                percent = 0
                # if joint == 0:
                #     if force < 0:
                #         percent = abs(force) / abs(self.min_force_motor)
                #     elif force > 0:
                #         percent = force / self.max_force_motor
                # else:
                #     percent = force / self.max_force

                if force < 0:
                    percent += abs(force) / abs(min_force)
                elif force > 0:
                    percent += force / max_force

            percent = percent/n_actuators   
            reward -= force_punishment_scale * percent
                # print(force)

        if self.stiffness_punishment:
            delta_action = np.clip(action, -1, 1)
            delta_action = delta_action/10
            current_iter = [0,1,2,3,4,5]
            delta_iter = [1,3,5,7,9,12]
            for i in range(len(current_iter)):
                if delta_action[delta_iter[i]] > 0.0:
                    # print("Reward: ", abs(delta_action[delta_iter[i]]) / abs(0.1)*(stiffness_punishment_scale/6))
                    reward -= abs(delta_action[delta_iter[i]]) / abs(0.1)*(stiffness_punishment_scale/6)
                   
                # if self.current_stiffness[current_iter[i]] >= 0.0 and action[delta_iter[i]] > 0.0:
                #     reward -= abs(action[delta_iter[i]]) / abs(max_delta)*(stiffness_punishment_scale/6)
                # elif self.current_stiffness[current_iter[i]] <= 0.0 and action[delta_iter[i]] < 0.0:
                #     pass
                # else:
                #     pass
                # self.current_stiffness[current_iter[i]] + action[delta_iter[i]]
        # print(self.sim.data.qfrc_actuator)
        # print(self.sim.data.)
        # print("GH360 base pos: "+str(np.array(self.sim.data.site_xpos[self.sim.model.site_name2id("robot0_gh360_base")])))
        # print("GH360 EEF pos"+str(self._eef_xpos))
        # print("Door Handle pos: "+str(self._handle_xpos))
        # print(reward)
        return reward

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        print("xpos: ",xpos)
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # Modify default agentview camera
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[0.5986131746834771, -4.392035683362857e-09, 1.5903500240372423],
            quat=[0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349]
        )

        # initialize objects of interest
        self.door = DoorObject(
            name="Door",
            friction=0.0,
            damping=0.1,
            lock=self.use_latch,
            lock_type="LATCH",
            mirror=True,
        )

        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.door)
        else:
            self.placement_initializer = UniformRandomSampler(
                    name="ObjectSampler",
                    mujoco_objects=self.door,
                    # x_range=[-0.01, 0.01],
                    # y_range=[-0.01, 0.01],
                    # y_range=[-0.01, 0.01],
                    # x_range=[-0.05, 0.05],
                    # x_range=[-0.07, -0.09],
                    y_range=[-0.0, -0.0],
                    x_range=[-0.0, -0.0],
                    
                    #rotation=(-np.pi / 2. - 0.25, -np.pi / 2.),
                    #rotation=(np.pi / 2., np.pi / 2. + 0.25),
                    # rotation=(np.pi / 2. -0.1, np.pi / 2. + 0.1),
                    # rotation=(np.pi / 2. -0.25, np.pi / 2. + 0.25),
                    rotation=(np.pi / 2 + 0.0, np.pi / 2 + 0.0),
                    rotation_axis='z',
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=True,
                    reference_pos=self.table_offset,
                )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots], 
            mujoco_objects=self.door,
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.object_body_ids = dict()
        self.object_body_ids["door"] = self.sim.model.body_name2id(self.door.door_body)
        self.object_body_ids["frame"] = self.sim.model.body_name2id(self.door.frame_body)
        self.object_body_ids["latch"] = self.sim.model.body_name2id(self.door.latch_body)
        self.door_handle_site_id = self.sim.model.site_name2id(self.door.important_sites["handle"])
        self.hinge_qpos_addr = self.sim.model.get_joint_qpos_addr(self.door.joints[0])
        if self.use_latch:
            self.handle_qpos_addr = self.sim.model.get_joint_qpos_addr(self.door.joints[1])

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()
        print(self.sim.data.body_xpos[self.sim.model.body_name2id("robot0_shoulder0")])

        # low-level object information
        if self.use_object_obs:
            # Get robot prefix and define observables modality
            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"

            # Define sensor callbacks
            @sensor(modality=modality)
            def door_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.object_body_ids["door"]])

            @sensor(modality=modality)
            def handle_pos(obs_cache):
                return self._handle_xpos

            @sensor(modality=modality)
            def door_to_eef_pos(obs_cache):
                return obs_cache["door_pos"] - obs_cache[f"{pf}eef_pos"] if\
                    "door_pos" in obs_cache and f"{pf}eef_pos" in obs_cache else np.zeros(3)

            @sensor(modality=modality)
            def handle_to_eef_pos(obs_cache):
                return obs_cache["handle_pos"] - obs_cache[f"{pf}eef_pos"] if\
                    "handle_pos" in obs_cache and f"{pf}eef_pos" in obs_cache else np.zeros(3)

            @sensor(modality=modality)
            def hinge_qpos(obs_cache):
                return np.array([self.sim.data.qpos[self.hinge_qpos_addr]])

            sensors = [door_pos, handle_pos, door_to_eef_pos, handle_to_eef_pos, hinge_qpos]
            names = [s.__name__ for s in sensors]

            # Also append handle qpos if we're using a locked door version with rotatable handle
            if self.use_latch:
                @sensor(modality=modality)
                def handle_qpos(obs_cache):
                    return np.array([self.sim.data.qpos[self.handle_qpos_addr]])
                sensors.append(handle_qpos)
                names.append("handle_qpos")

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        if self.obs_optimization:
            if self.motor_obs:
                observable_list = [f"{pf}joint_pos", f"{pf}joint_vel", f"{pf}motor_pos", f"{pf}motor_vel" f"{pf}eef_quat", "handle_to_eef_pos", "hinge_qpos", "handle_qpos"]
            else:
                observable_list = [f"{pf}joint_pos", f"{pf}joint_vel", f"{pf}eef_quat", "handle_to_eef_pos", "hinge_qpos", "handle_qpos"]
        else:
            if self.motor_obs:
                observable_list = [f"{pf}joint_pos", f"{pf}joint_vel", f"{pf}motor_pos", f"{pf}motor_vel" f"{pf}eef_pos", f"{pf}eef_quat", "door_pos", "handle_pos", "handle_to_eef_pos", "hinge_qpos", "handle_qpos"]
            else:
                observable_list = [f"{pf}joint_pos", f"{pf}joint_vel", f"{pf}eef_pos", f"{pf}eef_quat", "door_pos", "handle_pos", "handle_to_eef_pos", "hinge_qpos", "handle_qpos"]
        # observable_list = [f"{pf}joint_pos", f"{pf}joint_vel", f"{pf}eef_pos", f"{pf}eef_quat", "door_pos", "handle_pos", "handle_to_eef_pos", "hinge_qpos", "handle_qpos"]
        # macros.CONCATENATE_ROBOT_STATE = False

        for key, value in observables.items():
            value.set_active(False)
            for list_item in observable_list:
                if key == list_item:
                    value.set_active(True)

        #print("observables: "+str(observables))

        return observables

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # We know we're only setting a single object (the door), so specifically set its pose
            door_pos, door_quat, _ = object_placements[self.door.name]
            door_body_id = self.sim.model.body_name2id(self.door.root_body)
            self.sim.model.body_pos[door_body_id] = door_pos
            self.sim.model.body_quat[door_body_id] = door_quat

    def _check_success(self):
        """
        Check if door has been opened.

        Returns:
            bool: True if door has been opened
        """
        hinge_qpos = self.sim.data.qpos[self.hinge_qpos_addr]
        if self.grasp_check:
            return hinge_qpos < -0.3
        else:
            handle_qpos = self.sim.data.qpos[self.handle_qpos_addr]
            return hinge_qpos < -0.3 and np.abs(handle_qpos) <= 0.1
    
    def _post_action(self, action):
        """
        In addition to super method, add additional info if requested

        Args:
            action (np.array): Action to execute within the environment

        Returns:
            3-tuple:

                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) info about current env step
        """
        reward, done, info = super()._post_action(action)

        total_wrench_ee = np.linalg.norm(np.array(self.robots[0].recent_ee_forcetorques.current))
        info["total_ee_wrench"] = total_wrench_ee
        info["task_completed"] = self._check_success()

        # Update force bias
        # if np.linalg.norm(self.ee_force_bias) == 0:
        #     self.ee_force_bias = self.robots[0].ee_force
        #     self.ee_torque_bias = self.robots[0].ee_torque

        # if self.get_info:
        #     info["add_vals"] = ["nwipedmarkers", "colls", "percent_viapoints_", "f_excess"]
        #     info["nwipedmarkers"] = len(self.wiped_markers)
        #     info["colls"] = self.collisions
        #     info["percent_viapoints_"] = len(self.wiped_markers) / self.num_markers
        #     info["f_excess"] = self.f_excess

        # allow episode to finish early if allowed
        # if self.early_terminations:
        #     done = done or self._check_success()
        
        # done = self._check_success()
        done = False

        return reward, done, info

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the door handle.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the door handle
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(
                gripper=self.robots[0].gripper,
                target=self.door.important_sites["handle"],
                target_type="site"
            )

    @property
    def _handle_xpos(self):
        """
        Grabs the position of the door handle handle.

        Returns:
            np.array: Door handle (x,y,z)
        """
        return self.sim.data.site_xpos[self.door_handle_site_id]

    @property
    def _gripper_to_handle(self):
        """
        Calculates distance from the gripper to the door handle.

        Returns:
            np.array: (x,y,z) distance between handle and eef
        """
        # print("Handle Pos: ",self._handle_xpos)
        # print("Eef Pos: ", self._eef_xpos)
        return self._handle_xpos - self._eef_xpos
    