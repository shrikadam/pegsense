import numpy as np
from dm_control import mjcf

class Peg(object):
    """
    A class representing a pool cue with motion capture capabilities.
    """

    def __init__(self, type='square'):
        """
        Initializes a new instance of the PoolCuepeg class.

        Args:
            mjcf_root: The root element of the MJCF model.
        """
        self._mjcf_root = mjcf.RootElement()
        self._peg_body = self._mjcf_root.worldbody.add("body", name="square_peg")
        if type == 'square':
            self._peg_body.add(
                "geom",
                type="box",
                size=[0.01, 0.01, 0.05],
                density=1000,
                rgba=[0, 1, 0, 1]
            )
        elif type == 'round':
            self._peg_body.add(
                "geom",
                type="cylinder",
                size=[0.01, 0.05],
                density=1000,
                rgba=[0, 0, 1, 1]
            )
        else:
            raise Exception("Incompatible peg type specified! Supported types are 'square' and 'round'")
        
    @property
    def mjcf_root(self) -> object:
        """
        Gets the root element of the MJCF model.

        Returns:
            The root element of the MJCF model.
        """
        return self._mjcf_root
    

class Hole(object):
    """
    A class representing a pool cue with motion capture capabilities.
    """

    def __init__(self, mjcf_root, type='square'):
        """
        Initializes a new instance of the PoolCuepeg class.

        Args:
            mjcf_root: The root element of the MJCF model.
        """
        self._mjcf_root = mjcf_root

        self.hole_body = self._mjcf_root.worldbody.add("body", name="square_peg", mocap=True)
        self.hole_body.add(
            "geom",
            type="box",
            size=[0.01, 0.01, 0.05],
            density=1000,
            rgba=[0, 1, 0, 1]
        )

    @property
    def mjcf_root(self) -> object:
        """
        Gets the root element of the MJCF model.

        Returns:
            The root element of the MJCF model.
        """
        return self._mjcf_root

    @property
    def peg_body(self) -> object:
        """
        Gets the peg body.

        Returns:
            The peg body.
        """
        return self._square_peg

    def set_peg_pose(self, physics, position=None, quaternion=None):
        """
        Sets the pose of the peg body.

        Args:
            physics: The physics simulation.
            position: The position of the peg body.
            quaternion: The quaternion orientation of the peg body.
        """

        # flip quaternion xyzw to wxyz
        quaternion = np.roll(np.array(quaternion), 1)

        if position is not None:
            physics.bind(self.peg_body).mocap_pos[:] = position
        if quaternion is not None:
            physics.bind(self.peg_body).mocap_quat[:] = quaternion

    def get_peg_pose(self, physics):
        
        position = physics.bind(self.peg_body).mocap_pos[:]
        quaternion = physics.bind(self.peg_body).mocap_quat[:]

        # flip quaternion wxyz to xyzw
        quaternion = np.roll(np.array(quaternion), -1)

        pose = np.concatenate([position, quaternion])

        return pose