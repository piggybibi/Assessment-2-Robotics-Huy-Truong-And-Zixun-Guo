import os
from math import pi

import roboticstoolbox as rtb
from spatialmath import SE3

from ir_support.robots.UTSMeshRobot import UTSMeshRobot


class DoosanM0609(UTSMeshRobot):
    """Doosan Robotics M0609 (6-axis collaborative arm, 900 mm reach, 6 kg payload).

    Sources
    -------
    * Link geometry and meshes: Doosan Robotics ROS 2 package ``dsr_description2``
      (urdf/m0609.urdf, meshes/m0609_white), BSD-3-Clause, Doosan Robotics Inc.
      https://github.com/DoosanRobotics/doosan-robot2   (licence: MESH_LICENSE_Doosan_BSD3.txt)
    * Joint ranges / reach / payload: Doosan user manual, "Robot Specifications".
    * The standard DH parameters were derived from the URDF joint axes and match the URDF forward
      kinematics to numerical precision.

    Pose convention
    ---------------
    ``q = 0`` is the arm pointing straight up. ``home_q`` has the forearm forward and the flange
    pointing down. ``base`` is the pose of the MOUNTING SURFACE the robot bolts to: the base plate sits
    exactly on it (the cable on the official mesh hangs ~45 mm below that plane). ``BASE_LIFT`` adds
    extra height if you ever need it.
    """

    manufacturer_url = "https://manual.doosanrobotics.com/en/user-manual/3.5.0/1-m-h-series/robot-specifications"
    BASE_LIFT = 0.0  # [m] extra height above the mounting surface (0 = base plate sits exactly on it)

    # Teach-pendant ("TP") joint ranges from the manual, in degrees (smaller than the hardware ranges below)
    qlim_tp_deg = [(-360, 360), (-95, 95), (-125, 125), (-360, 360), (-135, 135), (-360, 360)]

    def __init__(self, base=None):
        d1, d3, a2, d4, d6 = 0.1345, 0.0062, 0.411, 0.368, 0.121
        links = [
            rtb.RevoluteDH(d=d1, a=0.0, alpha=pi / 2, offset=pi, qlim=self._qlim(-360, 360)),
            rtb.RevoluteDH(d=0.0, a=a2, alpha=0.0, offset=pi / 2, qlim=self._qlim(-360, 360)),
            rtb.RevoluteDH(d=d3, a=0.0, alpha=pi / 2, offset=pi / 2, qlim=self._qlim(-150, 150)),
            rtb.RevoluteDH(d=d4, a=0.0, alpha=pi / 2, offset=pi, qlim=self._qlim(-360, 360)),
            rtb.RevoluteDH(d=0.0, a=0.0, alpha=pi / 2, offset=pi, qlim=self._qlim(-360, 360)),
            rtb.RevoluteDH(d=d6, a=0.0, alpha=0.0, offset=0.0, qlim=self._qlim(-360, 360)),
        ]

        lift = SE3(0, 0, self.BASE_LIFT)
        base = lift if base is None else self._as_se3(base) * lift

        super().__init__(
            links=links,
            mesh_stem="DoosanM0609",
            mesh_dir=os.path.abspath(os.path.dirname(__file__)),
            name="DoosanM0609",
            home_q=[0.0, 0.0, pi / 2, 0.0, pi / 2, 0.0],
            base=base,
        )
