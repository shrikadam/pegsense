from dm_control import mjcf

class Peg(object):
    def __init__(self):
        self._mjcf_root = mjcf.RootElement()
        self.peg_body = self._mjcf_root.worldbody.add("body", name="peg")
        self.peg_body.add(
            "geom",
            type="box",
            size=[0.013, 0.05, 0.013],
            mass=0.1,
            rgba=[0, 1, 0, 1],
            contype="1",
            conaffinity="1",
            friction=[2.0, 0.01, 0.0005]
        )
        
    @property
    def mjcf_root(self) -> object:
        return self._mjcf_root
    

class Hole(object):
    def __init__(self):
        self._mjcf_root = mjcf.RootElement()
        self.hole_body = self._mjcf_root.worldbody.add("body", name="hole")
        half_size = 0.015
        wall_thick = 0.002
        height = 0.05

        # Wall 1 (+X)
        self.hole_body.add("geom", type="box", rgba=[1, 1, 0, 1], size=[wall_thick, half_size + wall_thick, height], 
                        pos=[half_size + wall_thick, 0, 0])
        # Wall 2 (-X)
        self.hole_body.add("geom", type="box", rgba=[1, 1, 0, 1], size=[wall_thick, half_size + wall_thick, height], 
                        pos=[-(half_size + wall_thick), 0, 0])
        # Wall 3 (+Y)
        self.hole_body.add("geom", type="box", rgba=[1, 1, 0, 1], size=[half_size, wall_thick, height], 
                        pos=[0, half_size + wall_thick, 0])
        # Wall 4 (-Y)
        self.hole_body.add("geom", type="box", rgba=[1, 1, 0, 1], size=[half_size, wall_thick, height], 
                        pos=[0, -(half_size + wall_thick), 0])
        # Floor (-Z) - optional if it sits on the ground
        self.hole_body.add("geom", type="box", rgba=[1, 1, 0, 1], size=[half_size + wall_thick, half_size + wall_thick, wall_thick], 
                        pos=[0, 0, -height])
        self.target_site = self.hole_body.add("site", name="hole_target", pos=[0, 0, height],
                                            rgba=[1, 0, 0, 1], size=[0.005])

    @property
    def mjcf_root(self) -> object:
        return self._mjcf_root