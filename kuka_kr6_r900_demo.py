"""
KUKA KR 6 R900 sixx (Agilus) — DH model for the Robotics Toolbox for Python

IMPORTANT: the a1, a2, a3, d1, d4, d6 values below are PLACEHOLDERS.
KUKA does not publish DH parameters directly — you must derive them
yourself from the official mechanical dimension drawing (side-view
diagram in the datasheet/brochure), then replace the values marked
TODO. Verify with the sanity check at the bottom of this file.

Joint limits below ARE from KUKA's published spec sheet for the
Agilus sixx family — double check against the exact variant your
team is using, since CR / HM-SC / base sixx differ slightly.
"""

import numpy as np
from roboticstoolbox import DHRobot, RevoluteDH
from spatialmath import SE3


class KR6R900(DHRobot):
    """
    Class that models a KUKA KR 6 R900 sixx (Agilus) 6-DOF industrial arm
    using standard DH convention.

    :references:
        - KUKA KR AGILUS sixx product datasheet (kuka.com)

    .. codeauthor:: < huy truong >
    """

    def __init__(self):
        deg = np.pi / 180

        # TODO: replace with values measured off the official dimension
        # drawing (convert mm -> m). These are placeholder magnitudes only.
        d1 = 0.400   # base height, axis1 -> axis2 offset along z0
        a1 = 0.025   # shoulder offset, axis1 -> axis2 offset along x1
        a2 = 0.455   # upper-arm length, axis2 -> axis3
        a3 = 0.035   # small forearm offset, axis3 -> axis4
        d4 = 0.420   # forearm length, axis4 -> axis5/6 region
        d6 = 0.080   # wrist/flange offset

        links = [
            RevoluteDH(d=d1, a=a1, alpha=-np.pi / 2,
                       qlim=[-170 * deg, 170 * deg]),
            RevoluteDH(d=0,  a=a2, alpha=0,
                       qlim=[-190 * deg, 45 * deg]),
            RevoluteDH(d=0,  a=a3, alpha=-np.pi / 2,
                       qlim=[-120 * deg, 156 * deg]),
            RevoluteDH(d=d4, a=0,  alpha=np.pi / 2,
                       qlim=[-185 * deg, 185 * deg]),
            RevoluteDH(d=0,  a=0,  alpha=-np.pi / 2,
                       qlim=[-120 * deg, 120 * deg]),
            RevoluteDH(d=d6, a=0,  alpha=0,
                       qlim=[-350 * deg, 350 * deg]),
        ]

        super().__init__(
            links,
            name="KR6_R900",
            manufacturer="KUKA",
        )

        self.qz = np.zeros(6)          # zero joint angle configuration
        self.addconfiguration("qz", self.qz)


if __name__ == "__main__":
    robot = KR6R900()
    print(robot)

    # Sanity check: max reach should land near KUKA's published 901.5 mm
    T = robot.fkine(robot.qz)
    print("End-effector position at qz:", T.t)
    print("Rough max reach estimate (a1+a2+a3+d4+d6) [m]:")
    # Sum your derived link dimensions here once filled in and compare
    # to 0.9015 m as a plausibility check.

    robot.plot(robot.qz, block=True)
