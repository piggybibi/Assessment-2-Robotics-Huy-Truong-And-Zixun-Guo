"""
KUKA KR 10 R1100 sixx (KR AGILUS) - DH model, 3D link geometry and Swift demo.
Requires: pip install roboticstoolbox-python swift-sim spatialgeometry

Joint angles q are KUKA axis angles A1..A6 (radians) with DH-native positive
directions. KUKA mastering pose is q = [0, -90, 90, 0, 0, 0] deg.

!! Link lengths below MUST be checked against the official KUKA KR 10 R1100
!! mechanical dimension drawing before you submit (cite the drawing/datasheet).
"""
import sys
import numpy as np
import spatialmath as sm
from spatialmath import SE3
import spatialgeometry as sg
from roboticstoolbox import DHRobot, RevoluteDH, jtraj

d2r = np.deg2rad

# ---- link lengths [m] - VERIFY against the KUKA drawing --------------------
D1, A1, A2, A3, D4, D6 = 0.400, 0.025, 0.560, 0.035, 0.515, 0.080

# ---- published joint limits (deg) from the datasheet -----------------------
LIM = [(-170, 170), (-190, 45), (-120, 156), (-185, 185), (-120, 120), (-350, 350)]


class KR10R1100(DHRobot):
    def __init__(self):
        links = [
            RevoluteDH(d=D1, a=A1, alpha=-np.pi / 2, qlim=d2r(LIM[0])),
            RevoluteDH(d=0,  a=A2, alpha=0,          qlim=d2r(LIM[1])),
            RevoluteDH(d=0,  a=A3, alpha=-np.pi / 2, offset=-np.pi / 2, qlim=d2r(LIM[2])),
            RevoluteDH(d=D4, a=0,  alpha=np.pi / 2,  qlim=d2r(LIM[3])),
            RevoluteDH(d=0,  a=0,  alpha=-np.pi / 2, qlim=d2r(LIM[4])),
            RevoluteDH(d=D6, a=0,  alpha=0,          qlim=d2r(LIM[5])),
        ]
        super().__init__(links, name="KUKA KR10 R1100")
        self.q_master = d2r([0, -90, 90, 0, 0, 0])
        self.addconfiguration("master", self.q_master)
        self._add_geometry()

    @staticmethod
    def _segment(T_link_inv, p0, p1, radius, color):
        """Cylinder between world points p0->p1, expressed in link frame."""
        v = p1 - p0
        L = np.linalg.norm(v)
        if L < 1e-6:
            return None
        z = v / L
        ref = np.array([1, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1, 0])
        x = np.cross(ref, z); x /= np.linalg.norm(x)
        y = np.cross(z, x)
        R = np.column_stack([x, y, z])
        mid = (p0 + p1) / 2
        T_world = SE3.Rt(R, mid)
        return sg.Cylinder(radius=radius, length=L, pose=T_link_inv * T_world, color=color)

    def _add_geometry(self):
        """Attach orange cylinders (arm) + grey joint spheres so Swift shows a 3D body."""
        orange, grey = (1.0, 0.45, 0.0, 1.0), (0.25, 0.25, 0.25, 1.0)
        Ts = self.fkine_all(self.q_master)           # frames 0..6 at mastering pose
        pts = [Ts[i].t for i in range(7)]
        radii = [0.085, 0.070, 0.058, 0.050, 0.044, 0.034]
        # extra joint-axis points so the offsets (A1, A3) look right
        for i, link in enumerate(self.links, start=1):
            Tinv = Ts[i].inv()
            seg = self._segment(Tinv, pts[i - 1], pts[i], radii[i - 1], orange)
            if seg is not None:
                link.geometry.append(seg)
            link.geometry.append(
                sg.Sphere(radius=radii[i - 1] * 1.15, pose=Tinv * SE3(*pts[i]), color=grey))
        # base plate
        self.links[0].geometry.append(
            sg.Cylinder(radius=0.13, length=0.05, pose=Ts[1].inv() * SE3(0, 0, 0.025), color=grey))


def checks(robot):
    """Quick numerical sanity checks for the report."""
    np.set_printoptions(precision=3, suppress=True)
    Tm = robot.fkine(robot.q_master)
    print("Mastering pose TCP [m]:", Tm.t)
    # wrist-centre (frame 4) reach with arm outstretched, compare with 901.5 mm datasheet
    q = d2r([0, 0, 0, 0, 0, 0])
    wc = robot.fkine_all(q)[4].t
    print(f"Outstretched wrist-centre reach (A1 axis -> A5): {np.hypot(wc[0], wc[1]) * 1000:.1f} mm "
          f"(datasheet max reach 1101 mm)")


def demo(robot, headless=False):
    # pick / place poses (tool pointing down), inside the workspace
    down = SE3.Rx(np.pi)
    pick = SE3(0.60, 0.25, 0.10) * down
    place = SE3(0.60, -0.25, 0.10) * down
    above = lambda T: SE3(0, 0, 0.15) * T

    waypoints = [above(pick), pick, above(pick), above(place), place, above(place)]
    q_prev = robot.q_master
    qs = [q_prev]
    for T in waypoints:
        sol = robot.ikine_LM(T, q0=q_prev, mask=[1, 1, 1, 1, 1, 0])  # ignore tool spin
        if not sol.success:
            raise RuntimeError(f"IK failed for {T.t}")
        qs.append(sol.q)
        q_prev = sol.q
    qs.append(robot.q_master)

    traj = np.vstack([jtraj(qs[i], qs[i + 1], 40).q for i in range(len(qs) - 1)])
    print(f"Trajectory: {len(traj)} steps, within limits:",
          bool(np.all((traj >= robot.qlim[0]) & (traj <= robot.qlim[1]))))
    if headless:
        return

    from swift import Swift
    env = Swift()
    env.launch(realtime=True)
    robot.q = robot.q_master
    env.add(robot)
    box = sg.Cuboid([0.04, 0.04, 0.04], pose=SE3(0.60, 0.25, 0.02), color=(0.1, 0.4, 0.9, 1))
    env.add(box)
    env.add(sg.Cuboid([0.45, 1.0, 0.002], pose=SE3(0.60, 0, -0.001), color=(0.7, 0.7, 0.7, 1)))

    for k, q in enumerate(traj):
        robot.q = q
        stage = k // 40                      # 0-1 approach, 2-4 carry, 5-6 retreat
        if 2 <= stage <= 4:                  # box is grasped: follows the tool
            box.T = (robot.fkine(q) * SE3(0, 0, 0.02)).A
        elif stage >= 5:                     # released at the place pose
            box.T = (SE3(0.60, -0.25, 0.02)).A
        env.step(0.02)
    env.hold()


if __name__ == "__main__":
    kr10 = KR10R1100()
    print(kr10)
    checks(kr10)
    demo(kr10, headless="--headless" in sys.argv)
