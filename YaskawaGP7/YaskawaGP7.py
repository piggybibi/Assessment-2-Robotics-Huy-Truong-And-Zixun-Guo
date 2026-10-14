import os
from math import pi, radians

import roboticstoolbox as rtb
from spatialmath import SE3

from ir_support.robots.UTSMeshRobot import UTSMeshRobot


class YaskawaGP7(UTSMeshRobot):
    """Yaskawa Motoman GP7 (6-axis, 927 mm reach, 7 kg payload).

    Sources
    -------
    * Link geometry and meshes: ROS-Industrial ``motoman_gp7_support`` (urdf/gp7_macro.xacro and
      meshes/visual/*.stl), BSD-3-Clause.  https://github.com/ros-industrial/motoman
      (licence: MESH_LICENSE_Motoman_BSD3.txt)
    * Joint ranges: same URDF (they match the Yaskawa GP7 data sheet).
    * The standard DH parameters were derived from the URDF joint axes and match the URDF forward
      kinematics to numerical precision (see gp7_dh_check.py / GP7_DH_NOTES.txt).

    Pose convention
    ---------------
    ``q = 0`` is the arm standing up with the forearm pointing +x and the flange pointing +x.
    ``home_q`` has the forearm forward and the flange pointing straight down. The z axis of the tool
    frame points OUT of the flange. ``base`` is the mounting surface the base plate sits on.
    """

    manufacturer_url = "https://www.motoman.com/en-us/products/robots/industrial/general-purpose/gp-series/gp7"

    # (lower, upper) in degrees, from the URDF
    qlim_deg = [(-170, 170), (-65, 145), (-70, 190), (-190, 190), (-135, 135), (-360, 360)]

    def __init__(self, base=None):
        q = self._qlim
        L = self.qlim_deg
        links = [
            rtb.RevoluteDH(d=0.330, a=0.040, alpha=-pi / 2, offset=0.0, qlim=q(*L[0])),
            rtb.RevoluteDH(d=0.000, a=0.445, alpha=pi, offset=-pi / 2, qlim=q(*L[1])),
            rtb.RevoluteDH(d=0.000, a=0.040, alpha=-pi / 2, offset=0.0, qlim=q(*L[2])),
            rtb.RevoluteDH(d=-0.440, a=0.0, alpha=pi / 2, offset=0.0, qlim=q(*L[3])),
            rtb.RevoluteDH(d=0.000, a=0.0, alpha=pi / 2, offset=pi, qlim=q(*L[4])),
            rtb.RevoluteDH(d=-0.080, a=0.0, alpha=pi, offset=0.0, qlim=q(*L[5])),
        ]
        super().__init__(
            links=links,
            mesh_stem="YaskawaGP7",
            mesh_dir=os.path.abspath(os.path.dirname(__file__)),
            name="YaskawaGP7",
            home_q=[0.0, radians(20), radians(-30), 0.0, radians(-40), 0.0],   # forearm forward, flange straight down
            base=base,
        )
