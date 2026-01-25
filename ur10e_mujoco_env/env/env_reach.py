import time
import os
import numpy as np
import mujoco
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces
from dm_control import mjcf
from scipy.spatial.transform import Rotation as R

from arena import StandardArena
from arm import Arm
from mocap import Mocap
from operational_space_controller import OperationalSpaceController

class UR5eMjEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None):
        super().__init__()
        
        # --- 1. CONFIGURATION ---
        self.control_dt = 0.002  # Controller runs at 500Hz
        self.rl_dt = 0.1         # Agent acts at 10Hz
        self.n_substeps = int(self.rl_dt / self.control_dt)
        
        self._render_mode = render_mode
        self._viewer = None
        self._step_start = None

        # --- 2. ACTION SPACE ---
        # Action: [dx, dy, dz] (Move the Mocap Target relative to current pos)
        # Range: Move up to 5cm per step
        self.action_space = spaces.Box(low=-0.05, high=0.05, shape=(3,), dtype=np.float32)

        # --- 3. OBSERVATION SPACE ---
        # [EE_Pos(3), EE_Quat(4), Target_Pos(3), Error_Vec(3)] = 13 dims
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)

        # --- 4. PHYSICS SETUP ---
        self._arena = StandardArena()
        self._target = Mocap(self._arena.mjcf_model)
        
        # Load Arm
        xml_path = os.path.join(os.path.dirname(__file__), 'universal_robots_ur5e/ur5e.xml')
        self._arm = Arm(xml_path=xml_path, attachment_site_name='attachment_site')
        self._arena.attach(self._arm.mjcf_model, pos=[0, 0, 0], quat=[0.7071068, 0, 0, -0.7071068])
        
        # Build Physics
        self._physics = mjcf.Physics.from_mjcf_model(self._arena.mjcf_model)
        self._site_id = self._physics.bind(self._arm.attachment_site).element_id

        # Initialize OSC
        self._controller = OperationalSpaceController(
            physics=self._physics,
            site_name=self._arm.attachment_site,
            joint_names=self._arm.joints,
            actuator_names=self._arm.actuators,
            impedance_pos=[300, 300, 300], # Stiffer for accuracy
            impedance_ori=[200, 200, 200]
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        with self._physics.reset_context():
            # 1. Reset Arm to "Home" (Bent elbow)
            init_q = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]
            self._physics.bind(self._arm.joints).qpos = init_q
            
            # 2. Randomize Target Position (The "Task")
            # Random point in front of the robot [x: 0.3-0.6, y: -0.3-0.3, z: 0.2-0.5]
            target_pos = np.random.uniform([0.4, -0.3, 0.2], [0.6, 0.3, 0.5])
            self._target.set_mocap_pose(self._physics, position=target_pos, quaternion=[1,0,0,0])
            
            # 3. Teleport Mocap "Ghost" to EE to start (so we don't jerk)
            # Or leave it at target_pos to force the robot to move there.
            # Let's leave it at target_pos for this task.

        # Run physics briefly to settle
        for _ in range(50):
            self._physics.step()

        return self._get_obs(), {}

    def step(self, action):
        # --- 1. INTERPRET ACTION (Move the Mocap) ---
        # Action is relative motion of the target (The "Carrot")
        # In a real task, the "Action" usually moves the *Mocap*, and the *OSC* chases it.
        # But for "Reaching", we usually just set the Mocap to the Goal and let OSC run.
        # IF you want the agent to control the TRAJECTORY, mapped action -> mocap delta.
        
        current_mocap_pos = self._target.get_mocap_pose(self._physics)[:3]
        # Clip movement to workspace to prevent drifting away
        new_mocap_pos = current_mocap_pos + action
        new_mocap_pos = np.clip(new_mocap_pos, [0.2, -0.5, 0.1], [0.8, 0.5, 0.8])
        
        # Update Mocap State
        self._target.set_mocap_pose(self._physics, position=new_mocap_pos, quaternion=[0, 1, 0, 0]) # Fixed orientation downward

        # --- 2. PHYSICS SUB-STEPPING (The "Real-Time" Loop) ---
        for _ in range(self.n_substeps):
            # Get Targets
            target_pose = self._target.get_mocap_pose(self._physics)
            
            # Run OSC
            self._controller.run(target_pose)
            
            self._physics.step()

        # --- 3. REWARD & TERMINATION ---
        obs = self._get_obs()
        
        # Calculate Error
        ee_pos = obs[:3]
        target_pos = obs[7:10] # Based on observation structure below
        dist = np.linalg.norm(target_pos - ee_pos)
        
        # Dense Reward: Negative Distance (Max 0)
        # Cast from Numpy to float for compatibility with stable_baselines
        reward = float(-dist)
        
        # Bonus for Reaching
        terminated = False
        if dist < 0.02: # 2cm tolerance
            reward += 10.0
            terminated = True # Episode done on success
            print("Target Reached!")

        # Visualize
        if self._render_mode == "human":
            self._render_frame()

        return obs, reward, terminated, False, {}

    def _get_obs(self):
        # 1. EE State
        ee_pos = self._physics.data.site_xpos[self._site_id]
        ee_mat = self._physics.data.site_xmat[self._site_id].reshape(3,3)
        ee_quat = R.from_matrix(ee_mat).as_quat()
        
        # 2. Target State (The red sphere)
        target_pos = self._target.get_mocap_pose(self._physics)[:3]
        
        # 3. Error
        error = target_pos - ee_pos
        
        return np.concatenate([ee_pos, ee_quat, target_pos, error]).astype(np.float32)

    def _render_frame(self):
        if self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(self._physics.model.ptr, self._physics.data.ptr, show_left_ui=False, show_right_ui=False,)
        self._viewer.sync()
        time.sleep(self.rl_dt) # Slow down to real time
