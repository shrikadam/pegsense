import mujoco
import numpy as np

class OperationalSpaceController:
    def __init__(self, physics, site_name, joint_names, actuator_names,
                 impedance_pos=[200.0, 200.0, 200.0], 
                 impedance_ori=[100.0, 100.0, 100.0], 
                 damping_ratio=1.0, 
                 k_pos=0.95, 
                 k_ori=0.95, 
                 vmax_lin = 0.1,
                 vmax_ang = 0.05,
                 integration_dt=1.0,
                 gravity_comp=True):
        
        self.model = physics.model.ptr
        self.data = physics.data.ptr
        self.gravity_comp = gravity_comp
        self.integration_dt = integration_dt
        self.k_pos = k_pos
        self.k_ori = k_ori
        self.vmax_lin = vmax_lin
        self.vmax_ang = vmax_ang

        # 1. Get IDs for Site and Joints
        self.site_id = physics.bind(site_name).element_id
        self.dof_ids = physics.bind(joint_names).dofadr
        self.actuator_ids = physics.bind(actuator_names).element_id.astype(int)

        # 2. Compute Gains (Kp and Kd matrices)
        # Kp = stiffness, Kd = damping
        damping_pos = damping_ratio * 2 * np.sqrt(impedance_pos)
        damping_ori = damping_ratio * 2 * np.sqrt(impedance_ori)
        
        self.Kp = np.concatenate([impedance_pos, impedance_ori], axis=0)
        self.Kd = np.concatenate([damping_pos, damping_ori], axis=0)

        # 3. Pre-allocate Memory (Avoids garbage collection in loop)
        self.jac = np.zeros((6, self.model.nv))
        self.twist = np.zeros(6)
        self.site_quat = np.zeros(4)
        self.site_quat_conj = np.zeros(4)
        self.error_quat = np.zeros(4)
        self.M_inv = np.zeros((self.model.nv, self.model.nv))
        self.Mx = np.zeros((6, 6))

    def run(self, target_pose):
        """
        Calculates torque to reach target_pos (vec3) and target_quat (vec4, wxyz).
        Returns: tau (array of torques for controlled joints)
        """
        target_pos = target_pose[:3]
        target_quat = target_pose[3:]
        # 1. Compute Twist (Spatial Velocity Error)
        # -----------------------------------------------------------
        # Translational error
        dx = target_pos - self.data.site_xpos[self.site_id]
        self.twist[:3] = self.k_pos * dx / self.integration_dt

        # Rotational error
        mujoco.mju_mat2Quat(self.site_quat, self.data.site_xmat[self.site_id])
        mujoco.mju_negQuat(self.site_quat_conj, self.site_quat)
        mujoco.mju_mulQuat(self.error_quat, target_quat, self.site_quat_conj)
        
        # Convert relative quaternion to angular velocity
        mujoco.mju_quat2Vel(self.twist[3:], self.error_quat, 1.0)
        self.twist[3:] *= self.k_ori / self.integration_dt

        # Cap linear vel while preserving direction and angular vel while preserving orientation
        v_lin = np.linalg.norm(self.twist[:3])
        if v_lin > self.vmax_lin:
            self.twist[:3] = (self.twist[:3] / v_lin) * self.vmax_lin
        v_ang = np.linalg.norm(self.twist[3:])
        if v_ang > self.vmax_ang:
            self.twist[3:] = (self.twist[3:] / v_ang) * self.vmax_ang

        # 2. Compute Jacobian & Task Space Inertia (Mx)
        # -----------------------------------------------------------
        mujoco.mj_jacSite(self.model, self.data, self.jac[:3], self.jac[3:], self.site_id)
        
        # Solve M^-1 (Generalized Inertia Inverse)
        mujoco.mj_solveM(self.model, self.data, self.M_inv, np.eye(self.model.nv))
        
        # Mx^-1 = J * M^-1 * J^T
        Mx_inv = self.jac @ self.M_inv @ self.jac.T
        
        # Invert Mx (with singularity protection)
        if abs(np.linalg.det(Mx_inv)) >= 1e-2:
            self.Mx = np.linalg.inv(Mx_inv)
        else:
            self.Mx = np.linalg.pinv(Mx_inv, rcond=1e-2)

        # 3. Compute Generalized Forces (OSC Control Law)
        # -----------------------------------------------------------
        # F = Mx * (Kp * twist - Kd * J * qvel)
        # Note: We filter jac and qvel by dof_ids in the math
        current_vel = self.jac @ self.data.qvel
        
        # Forces in Task Space
        F_task = self.Mx @ (self.Kp * self.twist - self.Kd * current_vel)
        
        # Convert to Joint Space Torques: tau = J^T * F_task
        tau = self.jac.T @ F_task
        
        # Filter for the specific joints we control
        tau_controlled = tau[self.dof_ids]

        # Joint space velocity damping
        jvel_max = np.array([1.0, 1.0, 1.5, 2.0, 2.0, 20.0])
        joint_vels = self.data.qvel[self.dof_ids]
        excess_vel = np.abs(joint_vels) - jvel_max # How much is each joint exceeding the speed limit
        excess_vel = np.maximum(0, excess_vel) # Zero if under the limit
        j_damping_gain = np.array([50.0, 50.0, 20.0, 10.0, 10.0, 0.0])
        damping_tau = -j_damping_gain * excess_vel * np.sign(joint_vels)

        tau_controlled += damping_tau

        # 4. Add Gravity Compensation
        # -----------------------------------------------------------
        if self.gravity_comp:
            tau_controlled += self.data.qfrc_bias[self.dof_ids]
        
        # tau = np.clip(tau_controlled, *self.model.actuator_ctrlrange.T)
        # self.data.ctrl[:] = tau
        
        # 1. Get the control limits ONLY for the arm joints
        #    shape becomes (6, 2) instead of (Total_Actuators, 2)
        arm_ctrl_ranges = self.model.actuator_ctrlrange[self.actuator_ids]

        # 2. Extract min and max columns
        arm_min = arm_ctrl_ranges[:, 0]
        arm_max = arm_ctrl_ranges[:, 1]

        # 3. Clip the calculated torque against these specific limits
        tau_clipped = np.clip(tau_controlled, arm_min, arm_max)

        # 4. Write ONLY to the arm actuators in the global data array
        #    This leaves the gripper (index 6) untouched by this controller
        self.data.ctrl[self.actuator_ids] = tau_clipped
