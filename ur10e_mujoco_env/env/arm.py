import numpy as np
from dm_control import mjcf
import mujoco

class Arm():
    def __init__(self, xml_path, attachment_site_name, joint_names = None, actuator_names = None, name: str = None):
        self._mjcf_root = mjcf.from_path(xml_path)
        if name:
            self._mjcf_root.model = name

        # Find MJCF elements that will be exposed as attributes.
        if joint_names is None:
            self._joints = self.mjcf_model.find_all('joint')
        else:
            self._joints = [self._mjcf_root.find('joint', name) for name in joint_names]

        if actuator_names is None:
            self._actuators = self.mjcf_model.find_all('actuator')
        else:
            self._actuators = [self._mjcf_root.find('actuator', name) for name in actuator_names]

        self._attachment_site = self._mjcf_root.find('site', attachment_site_name)
        self._tcp = self._attachment_site

    @property
    def joints(self):
        """List of joint elements belonging to the arm."""
        return self._joints[:6]

    @property
    def tcp(self):
        """Returns the current Tool Center Point (site element)."""
        return self._tcp
    
    @property
    def actuators(self):
        """List of actuator elements belonging to the arm."""
        return self._actuators[:6]

    @property
    def mjcf_model(self):
        """Returns the `mjcf.RootElement` object corresponding to this robot."""
        return self._mjcf_root
    
    def attach_tool(self, child, pos: list = [0, 0, 0], quat: list = [1, 0, 0, 0], 
                    update_tcp: bool = False, tool_tcp_name: str = None) -> mjcf.Element:
        frame = self._attachment_site.attach(child)
        frame.pos = pos
        frame.quat = quat
        if update_tcp:
            if tool_tcp_name is None:
                raise ValueError("update_tcp is True, but no tool_tcp_name was provided!")
            # Find the site specifically within the attached child model
            new_tcp_site = child.find('site', tool_tcp_name)
            if new_tcp_site:
                self._tcp = new_tcp_site
            else:
                raise ValueError(f"Site '{tool_tcp_name}' not found in the attached tool XML.")
        return frame
    
    def get_tcp_pose(self, physics):
        tcp_pos = physics.bind(self._tcp).xpos
        tcp_quat = np.zeros(4)
        mujoco.mju_mat2Quat(tcp_quat, physics.bind(self._attachment_site).xmat)
        tcp_pose = np.concatenate((tcp_pos, tcp_quat))
        return tcp_pose
