import os
import numpy as np
import mujoco
from dm_control import mjcf
from .targets import *

class StandardArena(object):
    def __init__(self, num_peg = 3) -> None:
        """
        Initializes the StandardArena object by creating a new MJCF model and adding a checkerboard floor and lights.
        """
        xml_path= os.path.join(os.path.dirname(__file__), '../assets/pegs_arena.xml')
        self._mjcf_model = mjcf.from_path(xml_path)
        self._pegs = []
        self._holes = []
        for i in range(num_peg):
            peg = Peg()
            rand_pos = [np.random.uniform(-0.6, 0.6), np.random.uniform(-0.6, 0.6), np.random.uniform(0.5, 1)]
            rand_quat = np.random.randn(4)
            rand_quat /= np.linalg.norm(rand_quat)
            self.attach_free(
                peg.mjcf_root, pos=rand_pos, quat=rand_quat
            )
            self._pegs.append(peg)
        
        # # Ensure Holes are not placed in the immediate work envelope of the robot
        def sample_excluding_center():
            if np.random.rand() < 0.5:
                return np.random.uniform(-0.6, -0.4)  # Left range
            else:
                return np.random.uniform(0.4, 0.6)   # Right range
        for i in range(num_peg):
            hole = Hole()
            rand_pos = [sample_excluding_center(), sample_excluding_center(), np.random.uniform(0.3, 0.6)]
            rand_quat = np.random.randn(4)
            rand_quat /= np.linalg.norm(rand_quat)
            self.attach(
                hole.mjcf_root, pos=rand_pos, quat=rand_quat
            )
            self._holes.append(hole)

    def attach(self, child, pos: list = [0, 0, 0], quat: list = [1, 0, 0, 0]) -> mjcf.Element:
        """
        Attaches a child element to the MJCF model at a specified position and orientation.

        Args:
            child: The child element to attach.
            pos: The position of the child element.
            quat: The orientation of the child element.

        Returns:
            The frame of the attached child element.
        """
        frame = self._mjcf_model.attach(child)
        frame.pos = pos
        frame.quat = quat
        return frame
    
    def attach_free(self, child,  pos: list = [0, 0, 0], quat: list = [1, 0, 0, 0]) -> mjcf.Element:
        """
        Attaches a child element to the MJCF model with a free joint.

        Args:
            child: The child element to attach.

        Returns:
            The frame of the attached child element.
        """
        frame = self.attach(child)
        frame.add('freejoint')
        frame.pos = pos
        frame.quat = quat
        return frame
    
    @property
    def mjcf_model(self) -> mjcf.RootElement:
        """
        Returns the MJCF model for the StandardArena object.

        Returns:
            The MJCF model.
        """
        return self._mjcf_model
    
    def get_peg_poses(self, physics):
        """
        Returns a list of dictionaries containing ground truth poses for all pegs.
        """
        poses = []
        
        for peg in self._pegs:
            peg_pos = physics.bind(peg.peg_body).xpos
            peg_quat = physics.bind(peg.peg_body).xquat
            peg_pose = np.concatenate((peg_pos, peg_quat))

            poses.append(peg_pose)
            
        return poses
    
    def get_hole_poses(self, physics):
        """
        Returns a list of dictionaries containing ground truth poses for all holes.
        """
        poses = []
        
        for hole in self._holes:
            hole_pos = physics.bind(hole.target_site).xpos
            hole_quat = np.zeros(4)
            mujoco.mju_mat2Quat(hole_quat, physics.bind(hole.target_site).xmat)
            hole_pose = np.concatenate((hole_pos, hole_quat))

            poses.append(hole_pose)
            
        return poses