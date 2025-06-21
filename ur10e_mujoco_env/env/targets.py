import os
from dm_control import mjcf

class Peg(object):
    def __init__(self, type='square'):
        self._mjcf_root = mjcf.RootElement()
        self.peg_body = self._mjcf_root.worldbody.add("body", name="peg")
        if type == 'square':
            self.peg_body.add(
                "geom",
                type="box",
                size=[0.01, 0.01, 0.05],
                mass=0.1,
                rgba=[0, 1, 0, 1]
            )
        elif type == 'round':
            self.peg_body.add(
                "geom",
                type="cylinder",
                size=[0.01, 0.05],
                mass=0.1,
                rgba=[0, 0, 1, 1]
            )
        else:
            raise Exception("Incompatible peg type specified! Supported types are 'square' and 'round'")
        
    @property
    def mjcf_root(self) -> object:
        return self._mjcf_root
    

class Hole(object):
    def __init__(self, type='square'):
        self._mjcf_root = mjcf.RootElement()
        self.hole_body = self._mjcf_root.worldbody.add("body", name="hole")
        if type == 'square':
            stl_path = os.path.join(os.path.dirname(__file__), '../assets/models/square_hole.stl')
            mesh = self._mjcf_root.asset.add('mesh', name='square_hole', file=stl_path, scale=[0.001, 0.001, 0.001])
            self.hole_body.add(
                "geom",
                type="mesh",
                mesh=mesh,
                mass=0.1,
                rgba=[1, 1, 0, 1]
            )
        elif type == 'round':
            stl_path = os.path.join(os.path.dirname(__file__), '../assets/models/round_hole.stl')
            mesh = self._mjcf_root.asset.add('mesh', name='round_hole', file=stl_path, scale=[0.001, 0.001, 0.001])
            self.hole_body.add(
                "geom",
                type="mesh",
                mesh=mesh,
                mass=0.1,
                rgba=[1, 0, 1, 1]
            )
        else:
            raise Exception("Incompatible peg type specified! Supported types are 'square' and 'round'")

    @property
    def mjcf_root(self) -> object:
        return self._mjcf_root