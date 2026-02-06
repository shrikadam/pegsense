import numpy as np
import mujoco

def pose_offset(target_pose, offset_pos, offset_quat):
    """
    Computes a new pose relative to a target pose using native MuJoCo functions.
    
    Args:
        target_pose (np.array): [x, y, z, w, x, y, z] of the object (global frame).
        offset_pos (list/np.array): [x, y, z] translation offset in the object's frame.
        offset_quat (list/np.array): [w, x, y, z] rotation offset.
        
    Returns:
        np.array: New global pose [x, y, z, w, x, y, z]
    """
    # 1. Unpack inputs
    target_p = target_pose[:3]
    target_q = target_pose[3:]
    
    # Ensure inputs are numpy arrays for MuJoCo functions
    off_p = np.array(offset_pos, dtype=np.float64)
    off_q = np.array(offset_quat, dtype=np.float64)
    
    # 2. Compute New Orientation (Global Quat = Target Quat * Offset Quat)
    res_q = np.zeros(4)
    mujoco.mju_mulQuat(res_q, target_q, off_q)
    
    # 3. Compute New Position
    # First, rotate the offset vector into the global frame: v_global = q_target * v_local * q_target_inv
    rotated_offset = np.zeros(3)
    mujoco.mju_rotVecQuat(rotated_offset, off_p, target_q)
    
    # Then add to the target's original position
    res_p = target_p + rotated_offset
    
    return np.concatenate([res_p, res_q])