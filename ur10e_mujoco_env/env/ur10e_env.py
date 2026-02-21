import time
import os
import numpy as np
np.set_printoptions(precision=3, suppress=True)
import mujoco
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces
from dm_control import mjcf
from .arena import StandardArena
from .arm import Arm
from .mocap import Mocap
from .utils import *
from ur10e_mujoco_env.controllers.operational_space_controller import OperationalSpaceController


class UR10eMjEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": None,
    }  # TODO add functionality to render_fps

    def __init__(self, render_mode=None):
        # TODO come up with an observation space that makes sense
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64
        )

        # TODO come up with an action space that makes sense
        self.action_space = spaces.Box(
            low=-0.1, high=0.1, shape=(6,), dtype=np.float64
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self._render_mode = render_mode
        
        # Checkerboard floor arena with pegs and holes
        self._arena = StandardArena(num_peg=1)

        # Mocap target that OSC will try to follow
        self._target = Mocap(self._arena.mjcf_model)

        assets_dir = os.path.join(os.path.dirname(__file__), '../assets')
        arm_xml_path = os.path.join(assets_dir, 'universal_robots_ur10e/ur10e.xml')
        gripper_xml_path = os.path.join(assets_dir, 'robotiq_2f85_v4/mjx_2f85.xml')

        # UR10e arm
        self._arm = Arm(
            xml_path= arm_xml_path,
            attachment_site_name='attachment_site'
        )
        gripper = mjcf.from_path(gripper_xml_path)
        gripper_frame = self._arm.attach_tool(gripper, pos=[0, 0, 0], quat=[0, 1, 0, 0], update_tcp=True, tool_tcp_name="pinch_site")
        # attach arm to arena
        self._arena.attach(
            self._arm.mjcf_model, pos=[0, 0, 0], quat=[0.7071068, 0, 0, -0.7071068]
        )

        self._expert_phase = 0
        self._wait_counter = 0

        # DEBUG for ease of visualizing frames in mujoco viewer
        # self._arm.mjcf_model.visual.scale.framelength = 0.2  # Default is often 1.0 (meters)
        # self._arm.mjcf_model.visual.scale.framewidth = 0.005 # Default is often 0.03
        
        # Generate model
        self._physics = mjcf.Physics.from_mjcf_model(self._arena.mjcf_model)

        # Set up OSC controller
        self._controller = OperationalSpaceController(
            physics=self._physics,
            site_name=self._arm._tcp,
            joint_names=self._arm.joints,
            actuator_names=self._arm.actuators
        )

        # For GUI and time keeping
        self._timestep = self._physics.model.opt.timestep
        self._viewer = None
        self._step_start = None

    def _get_obs(self) -> np.ndarray:
        # TODO come up with an observations that makes sense for your RL task
        return np.zeros(6)

    def _get_info(self) -> dict:
        # TODO come up with an info dict that makes sense for your RL task
        return {}

    def reset(self, seed=None, options=None) -> tuple:
        super().reset(seed=seed)

        with self._physics.reset_context():
            # Put arm in a reasonable starting position
            self._physics.bind(self._arm.joints).qpos = [0.0, -1.5707, 1.5707, -1.5707, -1.5707, 0.0]
            # Put target in a reasonable starting position
            self._target.set_mocap_pose(self._physics, position=[0.5, 0, 0.3], quaternion=[1, 0, 0, 0])

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def _get_pre_grasp_pose(self, peg_pose):
        """
        Calculates the pre-grasp pose 5cm above the peg's most upward facing flat surface.
        
        :param self: Description
        :param peg_pose: np.ndarray [x, y, z, w, x, y, z] of the peg in global space
        """

        normal_dist = 0.05 # 5cm

        peg_pos = peg_pose[:3]
        peg_quat = peg_pose[3:]

        # Extract the peg's local axes from its quaternion
        peg_mat = np.zeros(9)
        mujoco.mju_quat2Mat(peg_mat, peg_quat)
        peg_mat = peg_mat.reshape(3, 3)

        # Extract the face normals from local X, Y, Z unit vectors
        peg_x = peg_mat[:, 0]
        peg_y = peg_mat[:, 1] # Longitudinal axis of the peg
        peg_z = peg_mat[:, 2]
        face_normals = [peg_x, -peg_x, peg_z, -peg_z]

        # Find the normal that aligns the best with global up vector
        global_up = np.array([0, 0, 1])
        best_normal = max(face_normals, key=lambda n: np.dot(n, global_up))

        # Calculate target pose from the best normal
        target_pos = peg_pos + (best_normal * normal_dist)
        tcp_y = peg_y
        tcp_z = best_normal
        tcp_x = np.cross(tcp_y, tcp_z)
        target_mat = np.column_stack((tcp_x, tcp_y, tcp_z))
        target_quat = np.zeros(4)
        mujoco.mju_mat2Quat(target_quat, target_mat.flatten())

        return np.concatenate((target_pos, target_quat))

    def _get_grasp_pose(self, pre_grasp_pose):
        """
        Move the gripper down to pick up the peg
        
        :param self: Description
        :param peg_pose: np.ndarray [x, y, z, w, x, y, z] of the peg in global space
        """
        normal_dist = -0.055
        grasp_pose = pre_grasp_pose.copy()
        local_step = np.array([0.0, 0.0, normal_dist])
        global_step = np.zeros(3)

        mujoco.mju_rotVecQuat(global_step, local_step, grasp_pose[3:])

        grasp_pose[:3] += global_step

        return grasp_pose

    def _get_pre_insert_pose(self, hole_pose):
        """
        Calculates pre-insert pose 10cm above the hole
        
        :param self: Description
        :param hole_pose: np.ndarray [x, y, z, w, x, y, z] of the peg in global space
        """
        normal_dist = 0.1
        hole_pos = hole_pose[:3]
        hole_quat = hole_pose[3:]
        
        # Extract hole's local axes from quat
        hole_mat = np.zeros(9)
        mujoco.mju_quat2Mat(hole_mat, hole_quat)
        hole_mat = hole_mat.reshape(3,3)

        # Extract the face normals from local X, Y, Z unit vectors
        hole_x = hole_mat[:, 0]
        hole_y = hole_mat[:, 1]
        hole_z = hole_mat[:, 2] # Longitudinal axis of the hole
        face_normals = [hole_x, -hole_x, hole_y, -hole_y]

        # Find the normal that aligns the best with global up vector
        global_up = np.array([0, 0, 1])
        best_normal = max(face_normals, key=lambda n: np.dot(n, global_up))
        
        # Calculate target pos 10cm away from hole in Z 
        target_pos = hole_pos + (hole_z * normal_dist)

        # Calculate target quat
        tcp_y = -hole_z # TCP Y+ equals hole Z-
        tcp_z = best_normal
        tcp_x = np.cross(tcp_y, tcp_z)
        target_mat = np.column_stack((tcp_x, tcp_y, tcp_z))
        target_quat = np.zeros(4)
        mujoco.mju_mat2Quat(target_quat, target_mat.flatten())

        return np.concatenate((target_pos, target_quat))

    def _get_insert_pose(self, pre_insert_pose):
        """
        Move the gripper sideways to insert the peg
        
        :param self: Description
        :param peg_pose: np.ndarray [x, y, z, w, x, y, z] of the peg in global space
        """
        normal_dist = 0.09
        insert_pose = pre_insert_pose.copy()
        local_step = np.array([0.0, normal_dist, 0.0])
        global_step = np.zeros(3)

        mujoco.mju_rotVecQuat(global_step, local_step, insert_pose[3:])

        insert_pose[:3] += global_step

        return insert_pose

    def step(self, action: np.ndarray) -> tuple:
        # TODO use the action to control the arm
        # ---------------------------------------------------------
        # 1. Get Ground Truth Information
        # ---------------------------------------------------------
        gt_peg_poses = self._arena.get_peg_poses(self._physics)
        gt_hole_poses = self._arena.get_hole_poses(self._physics)
        peg_pose = gt_peg_poses[0]
        hole_pose = gt_hole_poses[0]

        # ---------------------------------------------------------
        # 2. Calculate Trajectory Waypoints
        # ---------------------------------------------------------
        pre_grasp_pose = self._get_pre_grasp_pose(peg_pose)
        grasp_pose = self._get_grasp_pose(pre_grasp_pose)
        pre_insert_pose = self._get_pre_insert_pose(hole_pose)
        insert_pose = self._get_insert_pose(pre_insert_pose)

        # # ---------------------------------------------------------
        # # 3. State Machine Logic for Scripted Sequence
        # # ---------------------------------------------------------
        current_tcp_pose = self._arm.get_tcp_pose(self._physics)
        gripper_action = 0.0 # 0.0 = Open, 1.0 = Closed
        target_pose = current_tcp_pose # Default to stay still

        # Helper function to check if we reached the waypoint
        def is_close(target, current, threshold=0.01):
            dist = np.linalg.norm(target[:3] - current[:3])
            ori_diff = np.linalg.norm(target[3:] - current[3:])
            return dist < threshold and ori_diff < threshold

        if self._expert_phase == 0:
            # Phase 0: Move to Pre-Grasp
            target_pose = pre_grasp_pose
            if is_close(target_pose, current_tcp_pose):
                self._expert_phase = 1
                print("Reached Pre-Grasp! Moving to Grasp.")

        elif self._expert_phase == 1:
            # Phase 1: Move down to Grasp
            target_pose = grasp_pose
            if is_close(target_pose, current_tcp_pose):
                self._expert_phase = 2
                print("Reached Grasp! Closing Gripper.")

        elif self._expert_phase == 2:
            # Phase 2: Close Gripper and wait a few steps for physics to settle
            target_pose = grasp_pose
            gripper_action = 1.0
            self._wait_counter += 1
            if self._wait_counter > 200: # Wait 200 simulation steps
                self._expert_phase = 3
                self._wait_counter = 0
                print("Gripper Closed! Moving to Pre-Insert.")

        elif self._expert_phase == 3:
            # Phase 3: Lift and move to Pre-Insert (above hole)
            target_pose = pre_insert_pose
            gripper_action = 1.0 # Keep holding!
            if is_close(target_pose, current_tcp_pose, threshold=0.015):
                self._expert_phase = 4
                print("Reached Pre-Insert! Inserting.")

        elif self._expert_phase == 4:
            # Phase 4: Push down into the hole
            target_pose = insert_pose
            gripper_action = 1.0
            if is_close(target_pose, current_tcp_pose):
                self._expert_phase = 5
                print("Insertion Complete!")

        elif self._expert_phase == 5:
            # Phase 5: Done, just hold it there
            target_pose = insert_pose
            gripper_action = 0.0

        # Mocap mode
        # target_pose = self._target.get_mocap_pose(self._physics)
        # gripper_action = 0

        # Run OSC controller to move to target pose
        self._controller.run(target_pose, gripper_action)

        # Step physics
        self._physics.step()

        # render frame
        if self._render_mode == "human":
            self._render_frame()
        
        # TODO come up with a reward, termination function that makes sense for your RL task
        observation = self._get_obs()
        reward = 0
        terminated = False
        info = self._get_info()

        return observation, reward, terminated, False, info

    def render(self) -> np.ndarray:
        """
        Renders the current frame and returns it as an RGB array if the render mode is set to "rgb_array".

        Returns:
            np.ndarray: RGB array of the current frame.
        """
        if self._render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self) -> None:
        """
        Renders the current frame and updates the viewer if the render mode is set to "human".
        """
        if self._viewer is None and self._render_mode == "human":
            # launch viewer
            self._viewer = mujoco.viewer.launch_passive(
                self._physics.model.ptr,
                self._physics.data.ptr,
                show_left_ui=False,
                show_right_ui=False,
            )
        if self._step_start is None and self._render_mode == "human":
            # initialize step timer
            self._step_start = time.time()

        if self._render_mode == "human":
            # render viewer
            self._viewer.sync()
            # TODO come up with a better frame rate keeping strategy
            time_until_next_step = self._timestep - (time.time() - self._step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

            self._step_start = time.time()

        else:  # rgb_array
            return self._physics.render()

    def close(self) -> None:
        """
        Closes the viewer if it's open.
        """
        if self._viewer is not None:
            self._viewer.close()
