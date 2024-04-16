from collections import OrderedDict
import numpy as np

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv

from robosuite.models.arenas import EmptyArena
from robosuite.models.objects import ViaPointVisualObject, MilkVisualObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler, SequentialCompositeSampler
from robosuite.utils.observables import Observable, sensor



class TrajectoryFollowing(SingleArmEnv):
    """
    This class corresponds to a task where the robot has to follow a trajectory defined by via points. When only one via point is given the robot should hold it's pose at that position.

    Args:
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
        q_vel_obs = True,
        task_state_obs = False,
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="sideview",
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

        self.action_punishment = action_punishment
        self.force_punishment = force_punishment

        self.motor_obs = motor_obs
        self.q_vel_obs = q_vel_obs
        self.task_state_obs = task_state_obs
        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        # self.target_joint_angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # self.via_point_offset = [-0.1576, 0.417, 1.05]
        self.via_point_offset = [-0.37, 0.47, 0.99]
        self.status = 0

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

        # sparse completion reward
        if self._check_success():
            reward = 1.0

        # else, we consider only the case if we're using shaped rewards
        elif self.reward_shaping:
            # Add reaching component
            if self.status == 0:
                dist = np.linalg.norm(self._gripper_to_via_point_1)
                reaching_reward = 0.25 * (1 - np.tanh(10.0 * dist))
                reward += reaching_reward
            elif self.status == 1:
                reward += 0.25
                dist = np.linalg.norm(self._gripper_to_via_point_2)
                reaching_reward = 0.25 * (1 - np.tanh(10.0 * dist))
                reward += reaching_reward
            elif self.status == 2:
                reward += 0.5
                dist = np.linalg.norm(self._gripper_to_via_point_3)
                reaching_reward = 0.25 * (1 - np.tanh(10.0 * dist))
                reward += reaching_reward
            elif self.status == 3:
                reward += 0.75
                dist = np.linalg.norm(self._gripper_to_via_point_4)
                reaching_reward = 0.25 * (1 - np.tanh(10.0 * dist))
                reward += reaching_reward
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
                if force < 0:
                    percent += abs(force) / abs(min_force)
                elif force > 0:
                    percent += force / max_force

            percent = percent/n_actuators   
            reward -= force_punishment_scale * percent
                # print(force)

        
        return reward

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["empty"]
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = EmptyArena()

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # Modify default agentview camera
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[0.5986131746834771, -4.392035683362857e-09, 1.5903500240372423],
            quat=[0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349]
        )

        self.via_points = []
        for i in range(4):
            self.via_points.append(ViaPointVisualObject(name="ViaPoint_"+str(i)))

        # self.via_point = ViaPointVisualObject(name="ViaPoint")
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

        # self.placement_initializer = UniformRandomSampler(
        #         name="ObjectSampler",
        #         mujoco_objects=self.via_point,
        #         # x_range=[-0.01, 0.01],
        #         # y_range=[-0.01, 0.01],
        #         x_range=[-0.1, 0.1],
        #         y_range=[-0.1, 0.1],
        #         z_range=[-0.1, 0.1],
        #         #rotation=(-np.pi / 2. - 0.25, -np.pi / 2.),
        #         #rotation=(np.pi / 2., np.pi / 2. + 0.25),
        #         #rotation=(np.pi / 2. -0.1, np.pi / 2. + 0.1),
        #         rotation=0.0,
        #         rotation_axis='z',
        #         ensure_object_boundary_in_range=False,
        #         ensure_valid_placement=False,
        #         reference_pos=self.via_point_offset,
        #         # z_offset=self.via_point_offset[2],
        #     )

        for via_point in self.via_points:
            self.placement_initializer.append_sampler(
                sampler=UniformRandomSampler(
                    name=f"{via_point.name}ObjectSampler",
                    mujoco_objects=via_point,
                    x_range=[-0.1, 0.1],
                    y_range=[-0.1, 0.1],
                    z_range=[-0.1, 0.1],
                    rotation=0.0,
                    rotation_axis='z',
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=False,
                    reference_pos=self.via_point_offset,
                    # z_offset=self.via_point_offset[2],
                )
            )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots], 
            mujoco_objects=self.via_points,
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # self.object_body_ids = dict()
        # self.object_body_ids["via_point"] = self.sim.model.body_name2id(self.via_points[0].root_body)
        self.via_point_site_ids = []
        for via_point in self.via_points:
            self.via_point_site_ids.append(self.sim.model.site_name2id(via_point.important_sites["via_point_site"]))
        # self.via_point_site_id = self.sim.model.site_name2id(self.via_points[0].important_sites["via_point_site"])
        

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()


        # low-level object information
        if self.use_object_obs:
            # Get robot prefix and define observables modality
            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"

            # Define sensor callbacks
            # @sensor(modality=modality)
            # def via_point_pos(obs_cache):
            #     return np.array(self.sim.data.body_xpos[self.object_body_ids["via_point"]])
            
            @sensor(modality=modality)
            def via_point_1_pos(obs_cache):
                return self._via_point_1_xpos
            
            @sensor(modality=modality)
            def via_point_2_pos(obs_cache):
                return self._via_point_2_xpos
            
            @sensor(modality=modality)
            def via_point_3_pos(obs_cache):
                return self._via_point_3_xpos
        
            @sensor(modality=modality)
            def via_point_4_pos(obs_cache):
                return self._via_point_3_xpos
            
            @sensor(modality=modality)
            def task_state(obs_cache):
                return self.via_point_state

            @sensor(modality=modality)
            def via_point_1_to_eef_pos(obs_cache):
                return obs_cache["via_point_1_pos"] - obs_cache[f"{pf}eef_pos"] if\
                    "via_point_1_pos" in obs_cache and f"{pf}eef_pos" in obs_cache else np.zeros(3)
            
            @sensor(modality=modality)
            def via_point_2_to_eef_pos(obs_cache):
                return obs_cache["via_point_2_pos"] - obs_cache[f"{pf}eef_pos"] if\
                    "via_point_2_pos" in obs_cache and f"{pf}eef_pos" in obs_cache else np.zeros(3)
            
            @sensor(modality=modality)
            def via_point_3_to_eef_pos(obs_cache):
                return obs_cache["via_point_3_pos"] - obs_cache[f"{pf}eef_pos"] if\
                    "via_point_3_pos" in obs_cache and f"{pf}eef_pos" in obs_cache else np.zeros(3)
            
            @sensor(modality=modality)
            def via_point_4_to_eef_pos(obs_cache):
                return obs_cache["via_point_4_pos"] - obs_cache[f"{pf}eef_pos"] if\
                    "via_point_4_pos" in obs_cache and f"{pf}eef_pos" in obs_cache else np.zeros(3)

            # @sensor(modality=modality)
            # def handle_to_eef_pos(obs_cache):
            #     return obs_cache["handle_pos"] - obs_cache[f"{pf}eef_pos"] if\
            #         "handle_pos" in obs_cache and f"{pf}eef_pos" in obs_cache else np.zeros(3)

            # @sensor(modality=modality)
            # def hinge_qpos(obs_cache):
            #     return np.array([self.sim.data.qpos[self.hinge_qpos_addr]])

            sensors = [via_point_1_pos, via_point_2_pos, via_point_3_pos, via_point_4_pos, via_point_1_to_eef_pos, via_point_2_to_eef_pos, via_point_3_to_eef_pos, via_point_4_to_eef_pos, task_state]
            names = [s.__name__ for s in sensors]

            # Also append handle qpos if we're using a locked door version with rotatable handle
            # if self.use_latch:
            #     @sensor(modality=modality)
            #     def handle_qpos(obs_cache):
            #         return np.array([self.sim.data.qpos[self.handle_qpos_addr]])
            #     sensors.append(handle_qpos)
            #     names.append("handle_qpos")

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        observable_list = [f"{pf}joint_pos", f"{pf}eef_pos", f"{pf}eef_quat", "via_point_1_pos", "via_point_1_to_eef_pos", "via_point_2_pos", "via_point_2_to_eef_pos", "via_point_3_pos", "via_point_3_to_eef_pos"]
        if self.motor_obs:
            observable_list.insert(1, f"{pf}motor_pos")
        if self.q_vel_obs:
            observable_list.insert(1, f"{pf}joint_vel")
        if self.task_state_obs:
            observable_list.append("task_state")

        for key, value in observables.items():
            value.set_active(False)
            for list_item in observable_list:
                if key == list_item:
                    value.set_active(True)


        return observables

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        self.status = 0

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            # object_placements = self.placement_initializer.sample()

            # # We know we're only setting a single object (the door), so specifically set its pose
            # # door_pos, door_quat, _ = object_placements[self.door.name]
            # via_point_pos, _, _ = object_placements[self.via_point.name]
            # via_point_body_id = self.sim.model.body_name2id(self.via_point.root_body)
            # self.sim.model.body_pos[via_point_body_id] = via_point_pos
            # self.sim.model.body_quat[door_body_id] = door_quat

            object_placements = self.placement_initializer.sample()

            for via_point in self.via_points:
                via_point_pos, _, _ = object_placements[via_point.name]
                via_point_body_id = self.sim.model.body_name2id(via_point.root_body)
                self.sim.model.body_pos[via_point_body_id] = via_point_pos


            # Loop through all objects and reset their positions
            # for obj_pos, obj_quat, obj in object_placements.values():
            #     # Set the visual object body locations
            #     if "via" in obj.name.lower():
            #         self.sim.model.body_pos[self.obj_body_id[obj.name]] = obj_pos
            #         self.sim.model.body_quat[self.obj_body_id[obj.name]] = obj_quat
            #     else:
            #         print("shouldn't be here")

    def _check_success(self):
        """
        Check if door has been opened.

        Returns:
            bool: True if door has been opened
        """
        # hinge_qpos = self.sim.data.qpos[self.hinge_qpos_addr]
        # print("Status: ", self.status)
        # print("Distance: ", self._gripper_to_via_point_1)
        if self.status == 0 and (np.abs(self._gripper_to_via_point_1) < 0.01).all():
            print("status 1")
            self.status = 1
        elif self.status == 1 and (np.abs(self._gripper_to_via_point_2) < 0.01).all():
            print("status 2")
            self.status = 2
        elif self.status == 2 and (np.abs(self._gripper_to_via_point_3) < 0.01).all():
            print("status 3")
            self.status = 3
        elif self.status == 3 and (np.abs(self._gripper_to_via_point_4) < 0.001).all():
            print("goal reached")
            return True
        return False
        # return (self._eef_xpos == self._via_point_xpos).all()

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
        # if vis_settings["grippers"]:
        #     self._visualize_gripper_to_target(
        #         gripper=self.robots[0].gripper,
        #         target=self.door.important_sites["handle"],
        #         target_type="site"
        #     )

    @property
    def _via_point_1_xpos(self):
        """
        Grabs the position of the door handle handle.

        Returns:
            np.array: Door handle (x,y,z)
        """
        return self.sim.data.site_xpos[self.via_point_site_ids[0]]
    
    @property
    def _via_point_2_xpos(self):
        """
        Grabs the position of the door handle handle.

        Returns:
            np.array: Door handle (x,y,z)
        """
        return self.sim.data.site_xpos[self.via_point_site_ids[1]]
    
    @property
    def _via_point_3_xpos(self):
        """
        Grabs the position of the door handle handle.

        Returns:
            np.array: Door handle (x,y,z)
        """
        return self.sim.data.site_xpos[self.via_point_site_ids[2]]
    
    @property
    def _via_point_4_xpos(self):
        """
        Grabs the position of the door handle handle.

        Returns:
            np.array: Door handle (x,y,z)
        """
        return self.sim.data.site_xpos[self.via_point_site_ids[3]]

    @property
    def _gripper_to_via_point_1(self):
        """
        Calculates distance from the gripper to the door handle.

        Returns:
            np.array: (x,y,z) distance between handle and eef
        """
        # print("Handle Pos: ",self._handle_xpos)
        # print("Eef Pos: ", self._eef_xpos)
        return self._via_point_1_xpos - self._eef_xpos
    
    @property
    def _gripper_to_via_point_2(self):
        """
        Calculates distance from the gripper to the door handle.

        Returns:
            np.array: (x,y,z) distance between handle and eef
        """
        return self._via_point_2_xpos - self._eef_xpos
    
    @property
    def _gripper_to_via_point_3(self):
        """
        Calculates distance from the gripper to the door handle.

        Returns:
            np.array: (x,y,z) distance between handle and eef
        """
        return self._via_point_3_xpos - self._eef_xpos
    
    @property
    def _gripper_to_via_point_4(self):
        """
        Calculates distance from the gripper to the door handle.

        Returns:
            np.array: (x,y,z) distance between handle and eef
        """
        return self._via_point_4_xpos - self._eef_xpos
    
    @property
    def via_point_state(self):
        if self.status == 0:
            return np.array([1, 0, 0, 0])
        elif self.status == 1:
            return np.array([0, 1, 0, 0])
        elif self.status == 2:
            return np.array([0, 0, 1, 0])
        elif self.status == 3:
            return np.array([0, 0, 0, 1])

    