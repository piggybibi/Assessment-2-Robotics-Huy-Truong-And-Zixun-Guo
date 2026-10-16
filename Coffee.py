"""
Robot café drink line with a safety scanner  (Swift simulation)

    Doosan M0609 : takes a cup from the rack, puts it on the machine tray, takes the filled cup out
    UR3e         : puts a lid on the cup, then passes the cup across to the GP7
    Yaskawa GP7  : puts the finished drink in the next slot of the pickup counter

The three arms work as a pipeline (the Doosan is already on drink 2 while the UR3e lids drink 1 ...).

Sensors (all simulated)
    * 2-D laser scanner on the front of the bar. It sees the café walls and the customer. The customer is
      whatever the scanner sees that is NOT the known wall. From that the safety controller computes, for
      every arm, the clearance between the customer and the arm:
          clearance < 0.30 m  -> STOP  (arm freezes mid-motion; resumes 1.5 s after the customer has moved away)
          clearance < 0.80 m  -> SLOW  (arm runs at 35 % speed)
    * Status lamp on a pole beside every arm:  green = running, amber = slow, red = stopped (person)

Floor zones: the transparent platform in front of the bar shows where the customer makes an arm slow down
    (amber) or stop (red). It is computed from the planned robot motion, worst case over the whole drink cycle.

Control window (a separate desktop window, Tkinter - it ships with Python on Windows)
    Selected robot, then: Enable Manual / Start Coffee / Pause / Resume / Stop Task / Home-Ready, a speed slider,
    joint jog (J1-J6, - / +) and tool-point jog (world X, Y, Z, - / +) with adjustable steps, a Safety Inputs tab
    (scanner clearance per arm, customer ON/OFF), an Event Log tab, RESET / ACKNOWLEDGE and an EMERGENCY STOP
    (also the Esc key). Enable Manual freezes the selected arm (the others carry on until they have to wait for
    it) and unlocks its jog buttons; Disable Manual glides it back to where the plan had it. There is no collision
    checking in manual mode; the jog stops at the joint limits and refuses points out of reach.

Run:    python cafe_drink_line.py                 (opens Swift + the control window; start-up takes a little while,
                                                   the café has ~570 objects)
        python cafe_drink_line.py --no-customer   (window starts with the customer switched off)
        python cafe_drink_line.py --auto          (no control window: runs the whole café once as soon as Swift is up)
        python cafe_drink_line.py --headless      (plan + simulate + report, no window)

Needs:  ir_support with UR3e, DoosanM0609 and YaskawaGP7 (robots/YaskawaGP7 folder from YaskawaGP7.zip).
"""
import argparse
import sys
from math import pi
from types import SimpleNamespace

import numpy as np
import spatialgeometry as sg
from roboticstoolbox import ctraj, jtraj
from spatialmath import SE3

from ir_support.robots.DoosanM0609 import DoosanM0609
from ir_support.robots.UR3e import UR3e
from ir_support.robots.YaskawaGP7 import YaskawaGP7

# =============================== layout [m] ===============================
TABLE_H = 0.75
TABLE_L, TABLE_W = 3.00, 0.90
ROBOT_X = 1.00                                   # Doosan x = -1, UR3e x = 0, GP7 x = +1 (all on y = 0)
GP7_READY_DEG = [0, -17, -26, 0, -81, 0]         # GP7 waiting pose: tool down, held high and back

CUP_R, CUP_H = 0.035, 0.09
LID_R, LID_T = 0.036, 0.008
TRAY_T = 0.012                                    # sliding tray thickness
GRIP = 0.08                                       # tool frame -> object centre along the tool z axis
APPROACH = 0.15                                   # hover height above every pick / place
DWELL = 12                                        # frames the arm stays still while the fingers close / open

Z_CUP = TABLE_H + CUP_H / 2                       # cup centre height standing on the bar
CUP_RACK = [(-0.72, -0.36), (-0.72, -0.26), (-0.72, -0.16)]
TRAY_OUT = np.array([-0.62, -0.02])               # tray position where the Doosan puts the cup
TRAY_IN = np.array([-0.62, 0.22])                 # tray position under the nozzle
LID_STATION = np.array([-0.36, 0.0])              # Doosan puts the filled cup here, UR3e lids it
RELAY = np.array([0.36, 0.0])                     # UR3e puts the lidded cup here, GP7 collects it
LID_PICK = np.array([-0.10, 0.34])                # lid dispenser (UR3e side)
SLOTS = [np.array([0.62, -0.345]), np.array([0.62, -0.27]), np.array([0.62, -0.195])]   # pickup counter slots

# café room
WALL_FRONT_Y, WALL_BACK_Y, WALL_X = -3.4, 1.6, 5.0
DOOR_X0, DOOR_X1 = 2.2, 3.2
WALL_H = 2.8
WALL_T = 0.10
IN_FRONT, IN_BACK, IN_X = WALL_FRONT_Y + WALL_T / 2, WALL_BACK_Y - WALL_T / 2, WALL_X - WALL_T / 2   # inner wall faces

# sensors / safety
SCAN_ORIGIN = np.array([0.0, -0.58])              # laser scanner on a post at the front of the bar
SCAN_Z, SCAN_RANGE = 1.0, 6.0
STOP_D, SLOW_D = 0.30, 0.80                       # clearance thresholds [m]
RESUME_S = 1.5                                    # customer must be clear this long before the arm restarts
SLOW_FACTOR = 0.35
PERSON_R, LINK_R, PERSON_H = 0.22, 0.06, 1.75
ZONE_CELL = 0.05                                  # floor-zone resolution [m]
SECOND_VISIT_DELAY = 1.0                          # [s] the 2nd customer visit starts this long after the GP7 starts serving drink 2

DT = 0.02
SPEED, JSPEED = 0.55, 1.2                         # [m/s] line moves, [rad/s] joint moves
HIDDEN = SE3(0, 0, -10).A


# =============================== helpers: orientation + IK ===============================
def unwrap(robot, q_new, q_ref):
    q = np.array(q_new, float)
    lo, hi = robot.qlim
    for i in range(len(q)):
        q[i] = q_ref[i] + (q[i] - q_ref[i] + pi) % (2 * pi) - pi
        if q[i] > hi[i] + 1e-9 and q[i] - 2 * pi >= lo[i] - 1e-9:
            q[i] -= 2 * pi
        if q[i] < lo[i] - 1e-9 and q[i] + 2 * pi <= hi[i] + 1e-9:
            q[i] += 2 * pi
    return q


def _rotvec(R):
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    ang = np.arccos(c)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2.0
    if ang < 1e-7:
        return v
    if pi - ang < 1e-5:
        axis = np.sqrt(np.clip((np.diag(R) + 1.0) / 2.0, 0.0, None))
        return axis / (np.linalg.norm(axis) + 1e-12) * ang
    return v / np.sin(ang) * ang


def dls_ik(robot, Td, q0, mask, iters=300, lam=0.02, tol_p=1e-6, tol_r=1e-5):
    """Damped-least-squares IK that only uses robot.fkine (independent of the toolbox's own solver)."""
    q = np.array(q0, float)
    W = np.asarray(mask, float)
    n = len(q)
    for _ in range(iters):
        T = robot.fkine(q).A
        e = np.r_[Td[:3, 3] - T[:3, 3], _rotvec(Td[:3, :3] @ T[:3, :3].T)] * W
        if np.linalg.norm(e[:3]) < tol_p and np.linalg.norm(e[3:]) < tol_r:
            return q, True
        J = np.zeros((6, n))
        h = 1e-6
        for i in range(n):
            dq = q.copy()
            dq[i] += h
            Tn = robot.fkine(dq).A
            J[:3, i] = (Tn[:3, 3] - T[:3, 3]) / h
            J[3:, i] = _rotvec(Tn[:3, :3] @ T[:3, :3].T) / h
        J *= W[:, None]
        step = J.T @ np.linalg.solve(J @ J.T + (lam ** 2) * np.eye(6), e)
        m = np.max(np.abs(step))
        if m > 0.3:
            step *= 0.3 / m
        q = q + step
    return q, False


class Arm:
    """A robot plus the motion helpers used to build its trajectories (all in the world frame)."""

    def __init__(self, name, robot, q_ready):
        self.name, self.r = name, robot
        self.q_ready = np.array(q_ready, float)
        self.q = self.q_ready.copy()
        self.R_down = robot.fkine(self.q_ready).R          # tool-down orientation used everywhere
        self.min_z = 9.0

    def ik(self, T, seed=None):
        seed = self.q if seed is None else seed
        q_sol, ok = dls_ik(self.r, T.A, seed, np.ones(6))
        if not ok:
            try:
                sol = self.r.ikine_LM(T, q0=seed, joint_limits=False, slimit=1, ilimit=500, tol=1e-8)
                q_sol, ok = sol.q, bool(sol.success)
            except Exception:
                ok = False
        if not ok:
            raise RuntimeError(f"{self.name}: IK failed for target {np.round(T.t, 3)}")
        q = unwrap(self.r, q_sol, seed)
        lo, hi = self.r.qlim
        if np.any(q < lo - 1e-6) or np.any(q > hi + 1e-6):
            raise RuntimeError(f"{self.name}: IK solution outside joint limits at {np.round(T.t, 3)}")
        return q

    def _track(self, traj):
        for q in traj[::4]:
            zs = [T.t[2] for T in self.r.fkine_all(q)][2:]
            self.min_z = min(self.min_z, min(zs))

    def to_pose(self, p, seed=None):
        qg = self.ik(SE3.Rt(self.R_down, p), seed=seed)
        n = max(30, int(np.max(np.abs(qg - self.q)) / JSPEED / DT))
        traj = jtraj(self.q, qg, n).q
        self._track(traj)
        self.q = qg
        return list(traj)

    def to_q(self, qg):
        qg = unwrap(self.r, qg, self.q)
        n = max(30, int(np.max(np.abs(qg - self.q)) / JSPEED / DT))
        traj = jtraj(self.q, qg, n).q
        self._track(traj)
        self.q = np.array(qg, float)
        return list(traj)

    def line(self, p_goal):
        T0 = self.r.fkine(self.q)
        T1 = SE3.Rt(T0.R, p_goal)
        n = max(12, int(np.linalg.norm(T1.t - T0.t) / SPEED / DT))
        out, q = [], self.q.copy()
        for T in ctraj(T0, T1, n):
            q = self.ik(T, seed=q)
            out.append(q)
        if len(out) > 1 and np.max(np.abs(np.diff(np.array(out), axis=0))) > 0.12:
            raise RuntimeError(f"{self.name}: jerky line move towards {np.round(p_goal, 3)}")
        self._track(out)
        self.q = out[-1]
        return out

    def tcp_pos(self, q):
        return self.r.fkine(q).t


# =============================== gripper ===============================
class Gripper:
    """Two-finger parallel gripper on a tool frame (tool z out of the flange, fingers open along tool x)."""
    PALM = (0.13, 0.05, 0.03)
    FINGER = (0.01, 0.03, 0.07)
    OPEN = 0.065

    def __init__(self, color, finger_color):
        self.palm = sg.Cuboid(list(self.PALM), color=color)
        self.f1 = sg.Cuboid(list(self.FINGER), color=finger_color)
        self.f2 = sg.Cuboid(list(self.FINGER), color=finger_color)
        self.off = self.OPEN

    def shapes(self):
        return [self.palm, self.f1, self.f2]

    def update(self, T_tool, hold_half):
        target = self.OPEN if hold_half is None else hold_half + 0.005
        self.off += float(np.clip(target - self.off, -0.004, 0.004))
        zf = self.PALM[2] + self.FINGER[2] / 2
        self.palm.T = (T_tool * SE3(0, 0, self.PALM[2] / 2)).A
        self.f1.T = (T_tool * SE3(self.off, 0, zf)).A
        self.f2.T = (T_tool * SE3(-self.off, 0, zf)).A


# =============================== geometry for the safety distance ===============================
def seg_dist(p1, q1, p2, q2):
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    if a < 1e-12 and e < 1e-12:
        return np.linalg.norm(r)
    if a < 1e-12:
        s, t = 0.0, np.clip(f / e, 0, 1)
    else:
        c = d1 @ r
        if e < 1e-12:
            t, s = 0.0, np.clip(-c / a, 0, 1)
        else:
            b = d1 @ d2
            den = a * e - b * b
            s = np.clip((b * f - c * e) / den, 0, 1) if den > 1e-12 else 0.0
            t = (b * s + f) / e
            if t < 0:
                t, s = 0.0, np.clip(-c / a, 0, 1)
            elif t > 1:
                t, s = 1.0, np.clip((b - c) / a, 0, 1)
    return np.linalg.norm((p1 + d1 * s) - (p2 + d2 * t))


def arm_polyline(arm, q):
    """Joint-centre polyline of an arm, plus the gripper (tool frame -> fingertips)."""
    pts = [arm.r.base.t] + [T.t for T in arm.r.fkine_all(q)][1:]
    T = arm.r.fkine(q)
    pts.append(T.t + 0.10 * T.R[:, 2])
    return np.array(pts)


# =============================== scene ===============================
def box(size, pos, color):
    return sg.Cuboid(list(size), pose=SE3(*pos), color=color)


def cyl(r, length, pos, color):
    return sg.Cylinder(radius=r, length=length, pose=SE3(*pos), color=color)


def make_room():
    s = []
    wall, wains = (0.90, 0.86, 0.78, 1.0), (0.16, 0.40, 0.40, 1.0)
    t, cy, D = WALL_T, (WALL_FRONT_Y + WALL_BACK_Y) / 2, WALL_BACK_Y - WALL_FRONT_Y
    s.append(box([16, 11, 0.10], (0, cy, -0.04), (0.33, 0.34, 0.36, 1)))                          # ground outside
    s.append(box([2 * IN_X, D - t, 0.02], (0, cy, 0.001), (0.84, 0.80, 0.72, 1)))                  # café floor (top z = 0.011)
    for ix in range(10):                                                                            # checker tiles
        for iy in range(5):
            if (ix + iy) % 2 == 0:
                s.append(box([1.0, 1.0, 0.002], (-4.5 + ix, WALL_FRONT_Y + 0.5 + iy, 0.012), (0.66, 0.62, 0.55, 1)))
    # walls
    s.append(box([WALL_X * 2, t, WALL_H], (0, WALL_BACK_Y, WALL_H / 2), wall))                    # back wall
    s.append(box([DOOR_X0 + WALL_X, t, WALL_H], ((DOOR_X0 - WALL_X) / 2, WALL_FRONT_Y, WALL_H / 2), wall))
    s.append(box([WALL_X - DOOR_X1, t, WALL_H], ((DOOR_X1 + WALL_X) / 2, WALL_FRONT_Y, WALL_H / 2), wall))
    s.append(box([DOOR_X1 - DOOR_X0, t, WALL_H - 2.325], ((DOOR_X0 + DOOR_X1) / 2, WALL_FRONT_Y, (2.325 + WALL_H) / 2), wall))   # above the door
    for sx in (-WALL_X, WALL_X):
        s.append(box([t, D, WALL_H], (sx, cy, WALL_H / 2), wall))
    # teal lower wall (wainscot) on the inner faces, up to 0.9 m
    s.append(box([2 * IN_X, 0.03, 0.9], (0, IN_BACK - 0.015, 0.45), wains))
    for sx in (-1, 1):
        s.append(box([0.03, D - t, 0.9], (sx * (IN_X - 0.015), cy, 0.45), wains))
    s.append(box([DOOR_X0 + IN_X, 0.03, 0.9], ((DOOR_X0 - IN_X) / 2, IN_FRONT + 0.015, 0.45), wains))
    s.append(box([IN_X - DOOR_X1, 0.03, 0.9], ((DOOR_X1 + IN_X) / 2, IN_FRONT + 0.015, 0.45), wains))
    # entrance: frame, lintel, mat, exit sign
    dark = (0.30, 0.19, 0.12, 1)
    s.append(box([DOOR_X1 - DOOR_X0 + 0.2, t, 0.25], ((DOOR_X0 + DOOR_X1) / 2, WALL_FRONT_Y, 2.2), dark))
    for xj in (DOOR_X0 - 0.04, DOOR_X1 + 0.04):
        s.append(box([0.08, 0.14, 2.2], (xj, WALL_FRONT_Y, 1.1), dark))
    s.append(box([0.9, 0.6, 0.01], ((DOOR_X0 + DOOR_X1) / 2, -3.0, 0.017), (0.45, 0.12, 0.12, 1)))
    s.append(box([0.5, 0.05, 0.18], ((DOOR_X0 + DOOR_X1) / 2, IN_FRONT + 0.03, 2.55), (0.1, 0.65, 0.25, 1)))
    # windows in both side walls
    for sx in (-1, 1):
        for yc in (-2.1, 0.2):
            s += window(sx, yc)
    return s


def window(sx, yc, zc=1.8, w=1.5, h=1.1):
    x, f, white = sx * (IN_X - 0.012), 0.05, (0.95, 0.95, 0.93, 1)
    return [box([0.02, w, h], (x, yc, zc), (0.62, 0.80, 0.93, 0.35)),                                # glass
            box([0.05, w + 2 * f, f], (x, yc, zc + h / 2 + f / 2), white), box([0.05, w + 2 * f, f], (x, yc, zc - h / 2 - f / 2), white),
            box([0.05, f, h], (x, yc - w / 2 - f / 2, zc), white), box([0.05, f, h], (x, yc + w / 2 + f / 2, zc), white),
            box([0.04, 0.03, h], (x, yc, zc), white), box([0.04, w, 0.03], (x, yc, zc), white),      # mullions
            box([0.14, w + 0.2, 0.04], (sx * (IN_X - 0.07), yc, zc - h / 2 - f - 0.02), white)]       # sill


def chair(x, y, face, col):
    dk = (0.18, 0.18, 0.20, 1)
    s = [box([0.40, 0.40, 0.04], (x, y, 0.46), col)]
    for dx in (-0.17, 0.17):
        for dy in (-0.17, 0.17):
            s.append(box([0.03, 0.03, 0.44], (x + dx, y + dy, 0.22), dk))
    s.append(box([0.04, 0.40, 0.36], (x - face * 0.18, y, 0.66), col))
    return s


def cafe_table(x, y, col):
    wood, metal = (0.58, 0.40, 0.24, 1), (0.20, 0.20, 0.22, 1)
    s = [cyl(0.36, 0.03, (x, y, 0.735), wood), cyl(0.035, 0.70, (x, y, 0.35), metal), cyl(0.22, 0.02, (x, y, 0.02), metal),
         cyl(0.07, 0.01, (x + 0.10, y + 0.06, 0.755), (0.95, 0.95, 0.95, 1)), cyl(0.04, 0.08, (x + 0.10, y + 0.06, 0.80), (0.95, 0.95, 0.95, 1)),
         cyl(0.025, 0.12, (x - 0.12, y - 0.05, 0.81), (0.6, 0.8, 0.9, 0.5)), sg.Sphere(radius=0.04, pose=SE3(x - 0.12, y - 0.05, 0.89), color=(0.9, 0.2, 0.3, 1))]
    return s + chair(x - 0.62, y, +1, col) + chair(x + 0.62, y, -1, col)


def plant(x, y):
    return [cyl(0.16, 0.34, (x, y, 0.18), (0.72, 0.38, 0.22, 1)),
            sg.Sphere(radius=0.22, pose=SE3(x, y, 0.60), color=(0.18, 0.50, 0.22, 1)),
            sg.Sphere(radius=0.17, pose=SE3(x + 0.10, y - 0.05, 0.84), color=(0.25, 0.60, 0.28, 1)),
            sg.Sphere(radius=0.15, pose=SE3(x - 0.09, y + 0.06, 0.80), color=(0.14, 0.42, 0.20, 1))]


FONT = {"C": [".####", "#....", "#....", "#....", "#....", "#....", ".####"],
        "A": [".###.", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"],
        "F": ["#####", "#....", "#....", "####.", "#....", "#....", "#...."],
        "E": ["#####", "#....", "#....", "####.", "#....", "#....", "#####"]}


def pixel_sign(text, px, xc, zc, y, color):
    """Block letters built from flat boxes (one box per horizontal run of pixels)."""
    width = len(text) * 5 * px + (len(text) - 1) * 1.5 * px
    x0, s = xc - width / 2, []
    for k, ch in enumerate(text):
        for r, row in enumerate(FONT[ch]):
            c = 0
            while c < 5:
                if row[c] == "#":
                    c0 = c
                    while c < 5 and row[c] == "#":
                        c += 1
                    xl = x0 + k * 6.5 * px + c0 * px
                    s.append(box([(c - c0) * px, 0.04, px], (xl + (c - c0) * px / 2, y, zc + (3 - r) * px), color))
                else:
                    c += 1
    return s


def make_decor():
    """Furniture and fittings: counter panels, back bar, shelves, menu, sign, lamps, tables, plants."""
    s = []
    wood, dark = (0.35, 0.22, 0.12, 1), (0.30, 0.20, 0.14, 1)
    steel, white, glass = (0.76, 0.77, 0.80, 1), (0.95, 0.95, 0.95, 1), (0.82, 0.92, 1.0, 0.25)
    # ---- the bar becomes a closed counter (panels on all four sides, below the top)
    ph = TABLE_H - 0.05
    s.append(box([TABLE_L - 0.10, 0.03, ph], (0, -TABLE_W / 2 + 0.015, ph / 2), wood))
    s.append(box([TABLE_L - 0.10, 0.03, ph], (0, TABLE_W / 2 - 0.015, ph / 2), wood))
    for sx in (-1, 1):
        s.append(box([0.03, TABLE_W - 0.10, ph], (sx * (TABLE_L / 2 - 0.015), 0, ph / 2), wood))
    s.append(box([TABLE_L - 0.10, 0.03, 0.03], (0, -TABLE_W / 2, 0.50), (0.78, 0.55, 0.30, 1)))   # copper trim on the front (proud of the panel)
    # ---- on the bar, at both ends (away from the robots)
    s += [cyl(0.05, 0.12, (1.36, -0.32, 0.81), (0.8, 0.9, 0.95, 0.4)), cyl(0.04, 0.03, (1.36, -0.32, 0.765), (0.9, 0.75, 0.2, 1)),       # tip jar
          box([0.10, 0.14, 0.05], (1.36, -0.10, 0.775), (0.15, 0.15, 0.18, 1)), box([0.07, 0.09, 0.004], (1.36, -0.10, 0.802), (0.2, 0.5, 0.9, 1)),   # card reader
          box([0.08, 0.12, 0.14], (1.36, 0.30, 0.82), (0.7, 0.7, 0.72, 1)),                                                                # napkins
          box([0.10, 0.14, 0.07], (-1.36, -0.30, 0.785), white), cyl(0.04, 0.20, (-1.36, 0.20, 0.85), white)]                              # sugar, spare cups
    # ---- back counter with a second espresso machine, grinder, jars and pastry case
    s.append(box([4.8, 0.60, 0.88], (0, 1.25, 0.44), dark))
    s.append(box([4.9, 0.66, 0.04], (0, 1.25, 0.90), (0.88, 0.86, 0.82, 1)))
    z0 = 0.92
    s += [box([0.20, 0.20, 0.30], (-1.7, 1.30, z0 + 0.15), (0.12, 0.12, 0.14, 1)), cyl(0.07, 0.18, (-1.7, 1.30, z0 + 0.39), (0.45, 0.28, 0.15, 0.6)),   # grinder
          box([0.55, 0.38, 0.38], (-0.6, 1.30, z0 + 0.19), steel), box([0.55, 0.30, 0.03], (-0.6, 1.30, z0 + 0.395), (0.2, 0.2, 0.22, 1)),
          cyl(0.03, 0.07, (-0.72, 1.10, z0 + 0.17), (0.70, 0.45, 0.25, 1)), cyl(0.03, 0.07, (-0.48, 1.10, z0 + 0.17), (0.70, 0.45, 0.25, 1)),
          box([0.40, 0.08, 0.02], (-0.6, 1.08, z0 + 0.03), steel)]                                                       # espresso machine
    for x in (0.35, 0.60, 0.85):
        s += [cyl(0.055, 0.20, (x, 1.25, z0 + 0.10), (0.85, 0.9, 0.92, 0.35)), cyl(0.05, 0.12, (x, 1.25, z0 + 0.065), (0.30, 0.18, 0.10, 1))]    # bean jars
    s += [box([0.8, 0.42, 0.04], (1.7, 1.28, z0 + 0.02), steel), box([0.8, 0.42, 0.30], (1.7, 1.28, z0 + 0.19), glass)]                   # pastry case
    for dx in (-0.27, -0.09, 0.09, 0.27):
        s.append(sg.Sphere(radius=0.045, pose=SE3(1.7 + dx, 1.28, z0 + 0.085), color=(0.75, 0.50, 0.25, 1)))
    # ---- shelves, menu board, sign
    mugs = [(0.95, 0.95, 0.95, 1), (0.16, 0.45, 0.45, 1), (0.90, 0.70, 0.25, 1), (0.75, 0.30, 0.25, 1)]
    for sx in (-1, 1):
        for zb in (1.55, 1.95):
            s.append(box([1.40, 0.22, 0.04], (sx * 1.55, IN_BACK - 0.11, zb), wood))
            for k, dx in enumerate((-0.5, -0.17, 0.17, 0.5)):
                if k % 2 == 0:
                    s.append(cyl(0.04, 0.09, (sx * 1.55 + dx, IN_BACK - 0.11, zb + 0.065), mugs[(k + int(zb * 10)) % 4]))
                else:
                    s.append(cyl(0.06, 0.18, (sx * 1.55 + dx, IN_BACK - 0.11, zb + 0.11), (0.82, 0.88, 0.90, 0.45)))
    s.append(box([1.5, 0.04, 0.90], (0, IN_BACK - 0.02, 1.72), wood))
    s.append(box([1.4, 0.02, 0.80], (0, IN_BACK - 0.045, 1.72), (0.10, 0.13, 0.12, 1)))
    for k, (w, c) in enumerate(zip((0.9, 0.7, 0.8, 0.6, 0.75, 0.5), ((1, 0.9, 0.5, 1),) + ((0.92, 0.92, 0.9, 1),) * 5)):
        z = 1.72 + 0.30 - k * 0.11
        s.append(box([w, 0.006, 0.03], (-0.65 + w / 2 + 0.05, IN_BACK - 0.058, z), c))
        if k:
            s.append(box([0.14, 0.006, 0.03], (0.55, IN_BACK - 0.058, z), (0.9, 0.7, 0.3, 1)))
    s.append(box([1.65, 0.03, 0.55], (0, IN_BACK - 0.015, 2.51), (0.16, 0.10, 0.08, 1)))
    s += pixel_sign("CAFE", 0.06, 0.0, 2.51, IN_BACK - 0.05, (1.0, 0.82, 0.35, 1))
    # ---- ceiling beams and pendant lamps (over the bar ends and over the café tables)
    for bx in (-3.6, -2.0, 0.0, 2.0, 3.9):
        s.append(box([0.12, WALL_BACK_Y - WALL_FRONT_Y - WALL_T, 0.14], (bx, (WALL_FRONT_Y + WALL_BACK_Y) / 2, 2.72), (0.30, 0.19, 0.12, 1)))
    for (x, y) in ((-3.6, -1.9), (-2.0, 0.0), (2.0, 0.0), (3.9, -1.7)):
        s += [cyl(0.006, 0.38, (x, y, 2.46), (0.1, 0.1, 0.1, 1)), cyl(0.17, 0.14, (x, y, 2.20), (0.95, 0.78, 0.30, 1)),
              sg.Sphere(radius=0.05, pose=SE3(x, y, 2.12), color=(1.0, 0.97, 0.8, 1))]
    # ---- customer area: café tables with chairs, plants
    for (x, y, col) in ((-3.6, -1.2, (0.16, 0.45, 0.45, 1)), (-3.6, -2.6, (0.85, 0.55, 0.22, 1)),
                        (3.9, -1.0, (0.85, 0.55, 0.22, 1)), (3.9, -2.4, (0.16, 0.45, 0.45, 1))):
        s += cafe_table(x, y, col)
    for (x, y) in ((-4.62, -3.05), (4.62, -3.05), (-4.62, 1.05), (4.62, 1.05)):
        s += plant(x, y)
    return s


def make_bar():
    wood, dark = (0.55, 0.38, 0.22, 1.0), (0.25, 0.25, 0.28, 1.0)
    t, leg = 0.05, 0.06
    s = [box([TABLE_L, TABLE_W, t], (0, 0, TABLE_H - t / 2), wood)]
    for x in (-(TABLE_L / 2 - leg), TABLE_L / 2 - leg):
        for y in (-(TABLE_W / 2 - leg), TABLE_W / 2 - leg):
            s.append(box([leg, leg, TABLE_H - t], (x, y, (TABLE_H - t) / 2), dark))
    return s


def make_stations():
    """Static furniture: rack marks, machine, lid dispenser, counter plate."""
    s = []
    for (x, y) in CUP_RACK:                                                                       # cup rack marks
        s.append(cyl(0.042, 0.004, (x, y, TABLE_H + 0.002), (0.8, 0.8, 0.82, 1)))
    ry = [p[1] for p in CUP_RACK]                                                                 # rack plate, sized from the cup positions
    s.append(box([0.20, max(ry) - min(ry) + 0.12, 0.006], (CUP_RACK[0][0], (max(ry) + min(ry)) / 2, TABLE_H + 0.003), (0.8, 0.8, 0.82, 1)))
    # coffee machine: body at the back edge, head over the nozzle position
    mx, ymid = TRAY_IN[0], TRAY_IN[1]
    s.append(box([0.26, 0.16, 0.30], (mx, 0.365, TABLE_H + 0.15), (0.18, 0.18, 0.20, 1)))
    s.append(box([0.22, 0.20, 0.07], (mx, ymid, TABLE_H + 0.235), (0.28, 0.28, 0.32, 1)))
    s.append(cyl(0.012, 0.03, (mx, ymid, TABLE_H + 0.185), (0.8, 0.8, 0.85, 1)))                  # nozzle
    s.append(box([0.12, 0.06, 0.004], (mx, 0.0, TABLE_H + 0.002), (0.3, 0.3, 0.33, 1)))           # tray slot marking
    # lid dispenser
    s.append(cyl(0.032, 0.06, (LID_PICK[0], LID_PICK[1], TABLE_H + 0.03), (0.7, 0.7, 0.72, 1)))
    s.append(box([0.14, 0.05, 0.14], (LID_PICK[0], 0.425, TABLE_H + 0.07), (0.3, 0.3, 0.33, 1)))
    # lid station + relay marks
    for p in (LID_STATION, RELAY):
        s.append(cyl(0.05, 0.004, (p[0], p[1], TABLE_H + 0.002), (0.9, 0.75, 0.2, 1)))
    # pickup counter plate with slot marks
    sy = [p[1] for p in SLOTS]                                                                    # counter plate, sized from the slot positions
    s.append(box([0.16, max(sy) - min(sy) + 0.12, 0.006], (SLOTS[0][0], (max(sy) + min(sy)) / 2, TABLE_H + 0.003), (0.2, 0.6, 0.3, 1)))
    for p in SLOTS:
        s.append(cyl(0.042, 0.004, (p[0], p[1], TABLE_H + 0.008), (0.85, 0.95, 0.85, 1)))
    return s


class Lamp:
    """Status lamp on a pole: four coloured spheres on the same spot, only the active one is shown."""
    COLS = {"RUN": (0.1, 0.85, 0.2, 1), "SLOW": (1.0, 0.7, 0.0, 1), "STOP": (0.95, 0.1, 0.1, 1)}

    def __init__(self, x, y):
        self.pole = cyl(0.015, 0.55, (x, y, TABLE_H + 0.275), (0.2, 0.2, 0.22, 1))
        self.pos = np.array([x, y, TABLE_H + 0.60])
        self.lamps = {k: sg.Sphere(radius=0.04, pose=SE3(*self.pos), color=c) for k, c in self.COLS.items()}
        self.state = None
        self.show("RUN")

    def show(self, state):
        if state == self.state:
            return
        self.state = state
        for k, sh in self.lamps.items():
            sh.T = SE3(*self.pos).A if k == state else HIDDEN

    def shapes(self):
        return [self.pole] + list(self.lamps.values())


# =============================== world: cups, lids, machine ===============================
class World:
    def __init__(self, arms):
        self.arms = arms
        n = 3
        self.reset()
        # shapes
        self.cup = [cyl(CUP_R, CUP_H, p, (0.97, 0.97, 0.97, 0.55)) for p in self.cup_pos]
        self.lid = [cyl(LID_R, LID_T, p, (0.30, 0.13, 0.08, 1)) for p in self.lid_pos]
        self.layer = [[cyl(CUP_R - 0.004, 0.014, (0, 0, -10), (0.32, 0.17, 0.08, 1)) for _ in range(4)] for _ in range(n)]
        self.tray = box([0.11, 0.11, TRAY_T], (TRAY_OUT[0], TRAY_OUT[1], TABLE_H + TRAY_T / 2 + 0.004), (0.7, 0.72, 0.76, 1))
        self.stream_sh = cyl(0.004, 0.068, (0, 0, -10), (0.32, 0.17, 0.08, 1))

    def reset(self):
        """Start of the day: cups on the rack, one lid in the dispenser, tray out, nothing poured."""
        n = 3

        def put(name, value):
            cur = getattr(self, name, None)
            if isinstance(cur, list):
                cur[:] = value                         # in place: the motion plan's events hold references to these lists
            else:
                setattr(self, name, value)

        put("cup_pos", [np.array([CUP_RACK[k][0], CUP_RACK[k][1], Z_CUP]) for k in range(n)])
        put("cup_holder", [None] * n)                  # None | arm | 'tray' | 'person'
        put("lid_holder", ["dispenser"] + ["hidden"] * (n - 1))   # None | arm | 'on_cup' | 'dispenser' | 'hidden'
        put("lid_pos", [np.array([LID_PICK[0], LID_PICK[1], TABLE_H + 0.06 + LID_T / 2]) for _ in range(n)])
        put("level", [0] * n)                          # coffee layers 0..4
        self.tray_s = 0.0                              # 0 = out, 1 = under the nozzle
        self.stream = False
        self.stream_cup = 0
        self.person_hand = np.array([0.0, 0.0, 1.0])

    def shapes(self):
        return self.cup + self.lid + [s for L in self.layer for s in L] + [self.tray, self.stream_sh]

    def tray_pos(self):
        xy = TRAY_OUT + (TRAY_IN - TRAY_OUT) * self.tray_s
        return np.array([xy[0], xy[1], TABLE_H + 0.004 + TRAY_T / 2])

    def holding_half(self, nm):
        if any(h == nm for h in self.cup_holder):
            return CUP_R
        if any(h == nm for h in self.lid_holder):
            return LID_R
        return None

    def update(self):
        carried = {}
        for nm, a in self.arms.items():
            T = a.r.fkine(a.r.q)
            carried[nm] = T.t + GRIP * T.R[:, 2]
        tp = self.tray_pos()
        self.tray.T = SE3(*tp).A
        for k in range(3):
            h = self.cup_holder[k]
            if h in carried:
                self.cup_pos[k] = carried[h]
            elif h == "tray":
                self.cup_pos[k] = tp + np.array([0, 0, TRAY_T / 2 + CUP_H / 2])
            elif h == "person":
                self.cup_pos[k] = self.person_hand.copy()
            self.cup[k].T = SE3(*self.cup_pos[k]).A
            lh = self.lid_holder[k]
            if lh in carried:
                self.lid_pos[k] = carried[lh]
            elif lh == "on_cup":
                self.lid_pos[k] = self.cup_pos[k] + np.array([0, 0, CUP_H / 2 + LID_T / 2])
            elif lh == "dispenser":
                self.lid_pos[k] = np.array([LID_PICK[0], LID_PICK[1], TABLE_H + 0.06 + LID_T / 2])
            self.lid[k].T = HIDDEN if lh == "hidden" else SE3(*self.lid_pos[k]).A
            for i, sh in enumerate(self.layer[k]):
                sh.T = SE3(*(self.cup_pos[k] + np.array([0, 0, -CUP_H / 2 + 0.008 + i * 0.015]))).A if i < self.level[k] else HIDDEN
        if self.stream:
            c = self.cup_pos[self.stream_cup]
            self.stream_sh.T = SE3(TRAY_IN[0], TRAY_IN[1], c[2] + CUP_H / 2 + 0.034).A
        else:
            self.stream_sh.T = HIDDEN


# =============================== plan: tracks and phases ===============================
class Track:
    """Frames of one robot inside a phase, with events (fire when a frame is reached) and gates (wait until true)."""

    def __init__(self):
        self.frames, self.events, self.gates = [], {}, {}

    def add(self, frames, event=None, gate=None):
        i = len(self.frames)
        self.frames.extend(frames)
        if event:
            self.events.setdefault(i, []).append(event)
        if gate:
            self.gates[i] = gate


class Phase:
    def __init__(self, name):
        self.name = name
        self.tracks = {}
        self.process = None        # (n_frames, fn(i, n)) machine animation, not affected by the safety stops
        self.on_start = []

    def add(self, arm, frames, **kw):
        self.tracks.setdefault(arm.name, Track()).add(frames, **kw)


def build_plan(arms, world, flags, log):
    dsn, ur, gp7 = arms["dsn"], arms["ur"], arms["gp7"]
    up = np.array([0.0, 0.0, APPROACH])
    tcp = lambda c: np.asarray(c) + np.array([0.0, 0.0, GRIP])
    xyz = lambda p, z: np.array([p[0], p[1], z])

    def pick(ph, arm, c, on_grasp, from_ready):
        ph.add(arm, arm.to_pose(tcp(c) + up) if from_ready else arm.line(tcp(c) + up))
        ph.add(arm, arm.line(tcp(c)))
        ph.add(arm, [arm.q.copy()] * DWELL, event=on_grasp)
        ph.add(arm, arm.line(tcp(c) + up))

    def place(ph, arm, c, on_release, swing=0, gate=None, joint=False):
        if joint:                                                   # joint-space transfer (avoids a near-singular straight line)
            ph.add(arm, arm.to_pose(tcp(c) + up))
        elif swing:                                                   # half turn about the base: pure joint-1 swing
            seed = arm.q.copy()
            seed[0] += swing * pi
            seed[5] += pi
            ph.add(arm, arm.to_pose(tcp(c) + up, seed=seed))
        else:
            ph.add(arm, arm.line(tcp(c) + up))
        if gate:
            ph.add(arm, [arm.q.copy()], gate=gate)
        ph.add(arm, arm.line(tcp(c)))
        ph.add(arm, [arm.q.copy()] * DWELL, event=on_release)
        ph.add(arm, arm.line(tcp(c) + up))

    def ready(ph, arm):
        ph.add(arm, arm.to_q(arm.q_ready))

    def setter(lst, k, val, extra=None):
        def f():
            lst[k] = val
            if extra:
                extra()
        return f

    z_tray = TABLE_H + 0.004 + TRAY_T + CUP_H / 2

    # ---------- Doosan: stage A (cup rack -> tray)
    def dsn_a(ph, d):
        c = xyz(CUP_RACK[d], Z_CUP)
        pick(ph, dsn, c, setter(world.cup_holder, d, "dsn"), True)
        place(ph, dsn, xyz(TRAY_OUT, z_tray), setter(world.cup_holder, d, "tray"))
        ready(ph, dsn)

    # ---------- machine: tray in, fill, tray out
    def machine(d):
        n_in, n_fill, n_out = 50, 125, 50

        def fn(i, n):
            if i < n_in:
                world.tray_s = i / n_in
                world.stream = False
            elif i < n_in + n_fill:
                world.tray_s = 1.0
                world.stream, world.stream_cup = True, d
                world.level[d] = min(4, 1 + int(4 * (i - n_in) / n_fill))
            else:
                world.stream = False
                world.level[d] = 4
                world.tray_s = max(0.0, 1.0 - (i - n_in - n_fill) / n_out)
        return n_in + n_fill + n_out, fn

    # ---------- Doosan: stage B (tray -> lid station)
    def dsn_b(ph, d):
        pick(ph, dsn, xyz(TRAY_OUT, z_tray), setter(world.cup_holder, d, "dsn"), True)
        place(ph, dsn, xyz(LID_STATION, Z_CUP), setter(world.cup_holder, d, None))
        ready(ph, dsn)

    # ---------- UR3e: stage A (lid dispenser -> lid on the cup)
    def ur_a(ph, u):
        def take_lid():
            world.lid_holder[u] = "ur"
            if u + 1 < 3:
                world.lid_holder[u + 1] = "dispenser"
        pick(ph, ur, xyz(LID_PICK, TABLE_H + 0.06 + LID_T / 2), take_lid, True)
        place(ph, ur, xyz(LID_STATION, TABLE_H + CUP_H + LID_T / 2), setter(world.lid_holder, u, "on_cup"), joint=True)

    # ---------- UR3e: stage B (lidded cup -> relay), swinging over the back of the bar
    def ur_b(ph, u):
        pick(ph, ur, xyz(LID_STATION, Z_CUP), setter(world.cup_holder, u, "ur"), False)
        place(ph, ur, xyz(RELAY, Z_CUP), setter(world.cup_holder, u, None), swing=-1)
        ready(ph, ur)

    # ---------- GP7: relay -> pickup slot
    def gp7_stage(ph, g):
        pick(ph, gp7, xyz(RELAY, Z_CUP), setter(world.cup_holder, g, "gp7"), True)
        place(ph, gp7, xyz(SLOTS[g], Z_CUP), setter(world.cup_holder, g, None))
        ready(ph, gp7)

    phases = []
    for t in range(5):
        d, u, g = t, t - 1, t - 2
        ph = Phase(f"step {t}: stage 1")
        if d < 3:
            dsn_a(ph, d)
        if 0 <= u < 3:
            ur_a(ph, u)
        if 0 <= g < 3:
            gp7_stage(ph, g)
            if g == 0:
                ph.on_start.append(lambda: flags.__setitem__("go", True))     # first customer arrives
            if g == 1:
                ph.on_start.append(lambda: flags.__setitem__("go2", True))    # second visit: collects drink 1 while the GP7 serves drink 2
        if ph.tracks:
            phases.append(ph)
        ph = Phase(f"step {t}: stage 2")
        if d < 3:
            ph.process = machine(d)
        if 0 <= u < 3:
            ur_b(ph, u)
        if ph.tracks or ph.process:
            phases.append(ph)
        ph = Phase(f"step {t}: stage 3")
        if d < 3:
            dsn_b(ph, d)
        if ph.tracks:
            phases.append(ph)
    return phases



# =============================== floor zones: where does the robot slow down / stop? ===============================
def compute_zones(arms, phases):
    """Worst case over the whole drink cycle: floor positions (customer centre) at which at least one arm
    would slow down or stop.  Uses exactly the clearance rule of the safety controller."""
    pts = []

    def add(arm, q):
        pl = arm_polyline(arm, q)
        for k in range(len(pl) - 1):
            a, b = pl[k], pl[k + 1]
            n = max(2, int(np.linalg.norm(b - a) / 0.04) + 1)
            pts.append(a + (b - a) * np.linspace(0.0, 1.0, n)[:, None])

    for a in arms.values():
        add(a, a.q_ready)
    for ph in phases:
        for nm, tr in ph.tracks.items():
            for q in tr.frames[::6]:
                add(arms[nm], q)
            add(arms[nm], tr.frames[-1])
    P = np.vstack(pts)
    P[:, 2] = np.clip(P[:, 2] - PERSON_H, 0.0, None)            # the customer's axis is vertical: only height above his head counts
    P = np.unique(np.round(P / 0.02).astype(int), axis=0) * 0.02
    xs = np.arange(-2.6, 2.6 + 1e-9, ZONE_CELL)
    ys = np.arange(-2.3, -(TABLE_W / 2) + 1e-9, ZONE_CELL)      # from the front edge of the bar outwards
    gx, gy = np.meshgrid(xs, ys)
    Q = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
    try:
        from scipy.spatial import cKDTree
        d = cKDTree(P).query(Q)[0]
    except ImportError:
        d = np.concatenate([np.sqrt(((Q[i:i + 150, None, :] - P[None, :, :]) ** 2).sum(-1)).min(axis=1) for i in range(0, len(Q), 150)])
    d = d.reshape(gx.shape)
    stop = d < STOP_D + PERSON_R + LINK_R
    slow = (d < SLOW_D + PERSON_R + LINK_R) & ~stop
    return xs, ys, stop, slow


def mask_to_boxes(mask, xs, ys, color, h=0.004):
    """Merge a boolean floor grid into a few flat transparent boxes (runs along x, equal neighbouring rows merged)."""
    rows = []
    for j in range(len(ys)):
        runs, i = [], 0
        while i < len(xs):
            if mask[j, i]:
                i0 = i
                while i < len(xs) and mask[j, i]:
                    i += 1
                runs.append((i0, i))
            else:
                i += 1
        rows.append(tuple(runs))
    out, j, c = [], 0, ZONE_CELL
    while j < len(ys):
        runs, j0 = rows[j], j
        while j < len(ys) and rows[j] == runs:
            j += 1
        for i0, i1 in runs:
            x0, x1 = xs[i0] - c / 2, xs[i1 - 1] + c / 2
            y0, y1 = ys[j0] - c / 2, ys[j - 1] + c / 2
            out.append(box([x1 - x0 + 0.004, y1 - y0 + 0.004, h], ((x0 + x1) / 2, (y0 + y1) / 2, 0.014 + h / 2), color))   # thin, tiny overlap -> no seams
    return out


# =============================== sensors: laser scanner + customer ===============================
def _ray_seg(o, d, a, b):
    ab = b - a
    den = d[0] * ab[1] - d[1] * ab[0]
    if abs(den) < 1e-12:
        return np.inf
    ao = a - o
    t = (ao[0] * ab[1] - ao[1] * ab[0]) / den
    s = (ao[0] * d[1] - ao[1] * d[0]) / den
    return t if (t > 0 and 0 <= s <= 1) else np.inf


def _ray_circle(o, d, c, r):
    f = o - c
    b = f @ d
    disc = b * b - (f @ f - r * r)
    if disc < 0:
        return np.inf
    t = -b - np.sqrt(disc)
    return t if t > 0 else np.inf


class Scanner:
    """2-D laser scanner: 46 rays over the half plane in front of the bar. Wall hits are drawn grey, customer hits red."""

    WALLS = [((-IN_X, IN_FRONT), (DOOR_X0, IN_FRONT)), ((DOOR_X1, IN_FRONT), (IN_X, IN_FRONT)),
             ((-IN_X, IN_FRONT), (-IN_X, IN_BACK)), ((IN_X, IN_FRONT), (IN_X, IN_BACK))]

    def __init__(self):
        self.ang = np.deg2rad(np.linspace(-178, -2, 46))
        self.dirs = np.column_stack([np.cos(self.ang), np.sin(self.ang)])
        self.wall_range = np.array([min(min(_ray_seg(SCAN_ORIGIN, d, np.array(a), np.array(b)) for a, b in self.WALLS), SCAN_RANGE)
                                    for d in self.dirs])
        self.post = cyl(0.03, SCAN_Z - 0.05, (SCAN_ORIGIN[0], SCAN_ORIGIN[1], (SCAN_Z - 0.05) / 2), (0.15, 0.15, 0.18, 1))   # stands on the floor
        self.head = cyl(0.07, 0.10, (SCAN_ORIGIN[0], SCAN_ORIGIN[1], SCAN_Z), (0.1, 0.1, 0.12, 1))                          # scanner head at ray height
        self.wall_mk = [sg.Sphere(radius=0.035, pose=SE3(*self._pt(i, self.wall_range[i])), color=(0.25, 0.55, 0.95, 1))
                        if self.wall_range[i] < SCAN_RANGE - 1e-6 else sg.Sphere(radius=0.01, pose=SE3(0, 0, -10)) for i in range(46)]
        self.per_mk = [sg.Sphere(radius=0.035, pose=SE3(0, 0, -10), color=(1.0, 0.15, 0.1, 1)) for _ in range(46)]
        self.last_mask = np.zeros(46, bool)

    def _pt(self, i, r):
        p = SCAN_ORIGIN + self.dirs[i] * r
        return (p[0], p[1], SCAN_Z)

    def shapes(self):
        return [self.post, self.head] + self.wall_mk + self.per_mk

    def scan(self, person_xy):
        """Returns the estimated customer position (x, y) or None, from the ranges alone."""
        per = np.full(46, np.inf)
        if person_xy is not None:
            per = np.array([_ray_circle(SCAN_ORIGIN, d, np.asarray(person_xy), PERSON_R) for d in self.dirs])
        mask = per < self.wall_range - 0.15                     # closer than the known wall = something new
        for i in range(46):                                     # draw
            if mask[i]:
                self.per_mk[i].T = SE3(*self._pt(i, per[i])).A
            elif self.last_mask[i]:
                self.per_mk[i].T = HIDDEN
            if mask[i] != self.last_mask[i] and self.wall_range[i] < SCAN_RANGE - 1e-6:
                self.wall_mk[i].T = HIDDEN if mask[i] else SE3(*self._pt(i, self.wall_range[i])).A
        self.last_mask = mask
        if not mask.any():
            return None
        pts = np.array([SCAN_ORIGIN + self.dirs[i] * per[i] for i in range(46) if mask[i]])
        c = pts.mean(axis=0)
        u = (c - SCAN_ORIGIN) / np.linalg.norm(c - SCAN_ORIGIN)
        return c + PERSON_R * u                                 # hits are on the near surface -> add the body radius


class Customer:
    """A scripted customer. Steps: wait_flag / show / walk / wait / do / hide."""

    def __init__(self, script, world):
        self.script = script
        self.world = world
        self.reset()
        c1, c2, c3 = (0.2, 0.35, 0.8, 1), (0.9, 0.75, 0.6, 1), (0.15, 0.15, 0.2, 1)
        self.legs = [cyl(0.06, 0.80, (0, 0, -10), c3) for _ in range(2)]
        self.torso = cyl(0.17, 0.62, (0, 0, -10), c1)
        self.head = sg.Sphere(radius=0.11, pose=SE3(0, 0, -10), color=c2)

    def reset(self):
        self.i, self.timer = 0, 0.0
        self.pos = np.array([0.0, -9.0])
        self.active = False

    def shapes(self):
        return self.legs + [self.torso, self.head]

    def xy(self):
        return self.pos if self.active else None

    def update(self, dt, flags):
        while self.i < len(self.script):
            kind, arg = self.script[self.i]
            if kind == "wait_flag":
                if not flags.get(arg):
                    break
            elif kind == "show":
                self.pos, self.active = np.array(arg, float), True
            elif kind == "hide":
                self.active = False
            elif kind == "do":
                arg()
            elif kind == "wait":
                self.timer += dt
                if self.timer < arg:
                    break
                self.timer = 0.0
            elif kind == "walk":
                target = np.array(arg, float)
                d = target - self.pos
                dist = np.linalg.norm(d)
                step = 1.1 * dt
                if dist > step:
                    self.pos = self.pos + d / dist * step
                    break
                self.pos = target
            self.i += 1
        self.world.person_hand = np.array([self.pos[0], self.pos[1] + 0.25, 1.0])
        if self.active:
            x, y = self.pos
            self.legs[0].T = SE3(x - 0.08, y, 0.40).A
            self.legs[1].T = SE3(x + 0.08, y, 0.40).A
            self.torso.T = SE3(x, y, 1.11).A
            self.head.T = SE3(x, y, 1.52).A
        else:
            for s in self.shapes():
                s.T = HIDDEN

    @property
    def finished(self):
        return self.i >= len(self.script)


# =============================== safety controller ===============================
class Safety:
    def __init__(self, name):
        self.name, self.state, self.clear_t, self.blocked = name, "RUN", None, False

    def update(self, t, clearance, log):
        old = self.state
        if clearance is None:
            clearance = 9.0
        if self.state == "STOP":
            if clearance > STOP_D + 0.05:
                self.clear_t = t if self.clear_t is None else self.clear_t
                if t - self.clear_t >= RESUME_S:
                    self.state = "SLOW" if clearance < SLOW_D else "RUN"
                    self.clear_t = None
            else:
                self.clear_t = None
        elif clearance < STOP_D:
            self.state, self.clear_t = "STOP", None
        else:
            self.state = "SLOW" if clearance < SLOW_D else "RUN"
        if self.state != old:
            what = {"RUN": "running", "SLOW": "slowing down (customer nearby)", "STOP": "STOPPED (customer too close)"}[self.state]
            if old == "STOP":
                what = "resumes" + ("" if self.state == "RUN" else " at slow speed")
            log(f"[t={t:5.1f} s] {self.name:>4}: {what}   clearance {clearance:.2f} m")
        return self.state


# =============================== simulation loop ===============================
class Controls:
    """What the control panel can do to the arms while the simulation runs.

    mode[nm]:  'auto'    the plan moves the arm
               'manual'  the user holds / jogs the arm; the plan is frozen for this arm (the others carry on until
                         they have to wait for it)
               'glide'   the arm is returning to the pose the plan left it in; the plan stays frozen until it arrives
    """

    def __init__(self, arms):
        self.arms = arms
        self.mode = {nm: "auto" for nm in arms}
        self.plan_q = {nm: None for nm in arms}
        self.glide = {nm: [] for nm in arms}
        self.goto = {nm: [] for nm in arms}                      # manual 'Home / Ready' moves
        self.safety = {nm: "RUN" for nm in arms}
        self.clearance = {nm: None for nm in arms}               # customer-to-arm clearance seen by the scanner [m]
        self.person = None                                       # customer position estimated by the scanner
        self.t, self.phase = 0.0, ""

    def reset(self):
        for nm in self.arms:
            self.mode[nm], self.plan_q[nm], self.glide[nm], self.safety[nm] = "auto", None, [], "RUN"
            self.goto[nm], self.clearance[nm] = [], None
        self.person = None
        self.t, self.phase = 0.0, ""

    def owns(self, nm):
        """True while the plan must not move this arm."""
        return self.mode[nm] != "auto"

    def take(self, nm):
        if self.mode[nm] == "auto":
            self.plan_q[nm] = self.arms[nm].r.q.copy()          # where the plan has the arm right now
        self.mode[nm] = "manual"
        self.glide[nm] = []
        self.goto[nm] = []

    def release(self, nm):
        if self.mode[nm] != "manual":
            return
        r = self.arms[nm].r
        n = max(25, int(np.max(np.abs(self.plan_q[nm] - r.q)) / JSPEED / DT))
        self.glide[nm] = list(jtraj(r.q, self.plan_q[nm], n).q)
        self.mode[nm] = "glide"

    def advance(self):
        """One frame of every glide back to the plan."""
        for nm, a in self.arms.items():
            if self.mode[nm] == "glide":
                a.r.q = self.glide[nm].pop(0)
                if not self.glide[nm]:
                    self.mode[nm] = "auto"
            elif self.mode[nm] == "manual" and self.goto[nm]:
                a.r.q = self.goto[nm].pop(0)


def sim_gen(phases, arms, world, lamps, scanner, customer, flags, grippers, log, ctl):
    """The whole café run as a generator: every next() is one 20 ms tick. Returns (time, stats, closest approach)."""
    t = 0.0
    safety = {nm: Safety(nm) for nm in arms}
    stats = {nm: {"stop_s": 0.0, "slow_s": 0.0, "stops": 0} for nm in arms}
    min_gap = [9.0, ""]
    for ph in phases:
        log(f"[t={t:5.1f} s] -- {ph.name}")
        ctl.phase = ph.name
        for fn in ph.on_start:
            fn()
        st = {nm: {"p": 0.0, "fired": -1, "passed": set()} for nm in ph.tracks}
        proc_i = 0
        guard = 0
        while True:
            # customer + scanner + safety
            customer.update(DT, flags)
            p_est = scanner.scan(customer.xy())
            ctl.person = p_est
            for nm, a in arms.items():
                clr = None
                if p_est is not None:
                    pl = arm_polyline(a, a.r.q)
                    axis = (np.array([p_est[0], p_est[1], 0.0]), np.array([p_est[0], p_est[1], PERSON_H]))
                    clr = min(seg_dist(axis[0], axis[1], pl[k], pl[k + 1]) for k in range(len(pl) - 1)) - PERSON_R - LINK_R
                was = safety[nm].state
                safety[nm].update(t, clr, log)
                ctl.safety[nm] = safety[nm].state
                ctl.clearance[nm] = clr
                if safety[nm].state == "STOP" and was != "STOP":
                    stats[nm]["stops"] += 1
            # advance tracks
            all_done = True
            for nm, tr in ph.tracks.items():
                s = st[nm]
                n = len(tr.frames)
                safety[nm].blocked = False
                done = s["p"] >= n - 1 and s["fired"] >= max(tr.events, default=-1)
                if not done:
                    all_done = False
                    if ctl.owns(nm):                                    # the user has this arm: the plan waits
                        continue
                    idx = int(s["p"])
                    g = tr.gates.get(idx)
                    if g is not None and idx not in s["passed"]:
                        if g():
                            s["passed"].add(idx)
                        else:
                            safety[nm].blocked = True
                    state = safety[nm].state
                    speed = 0.0 if (state == "STOP" or safety[nm].blocked) else (SLOW_FACTOR if state == "SLOW" else 1.0)
                    newp = min(s["p"] + speed, n - 1)
                    for gi in sorted(tr.gates):                         # never run past a gate that has not been passed
                        if gi not in s["passed"] and int(s["p"]) < gi <= int(newp):
                            newp = float(gi)
                    s["p"] = newp
                    hi = int(newp)
                    for k in range(s["fired"] + 1, hi + 1):
                        for ev in tr.events.get(k, []):
                            ev()
                        s["fired"] = k
                    i0 = int(s["p"])
                    i1 = min(i0 + 1, n - 1)
                    f = s["p"] - i0
                    arms[nm].r.q = tr.frames[i0] + (tr.frames[i1] - tr.frames[i0]) * f
            if ph.process:
                n_p, fn = ph.process
                if proc_i < n_p:
                    fn(proc_i, n_p)
                    proc_i += 1
                    all_done = False
            # status + stats
            for nm in arms:
                s = safety[nm]
                lamps[nm].show(s.state)
                if s.state == "STOP":
                    stats[nm]["stop_s"] += DT
                elif s.state == "SLOW":
                    stats[nm]["slow_s"] += DT
            if int(t / DT) % 5 == 0:                                    # closest approach between two different robots
                pls = {nm: arm_polyline(a, a.r.q) for nm, a in arms.items()}
                names = list(arms)
                for i in range(3):
                    for j in range(i + 1, 3):
                        A, B = pls[names[i]], pls[names[j]]
                        d = min(seg_dist(A[a], A[a + 1], B[b], B[b + 1]) for a in range(len(A) - 1) for b in range(len(B) - 1))
                        if d < min_gap[0]:
                            min_gap[:] = [d, f"{names[i]}-{names[j]}"]
            world.update()
            for nm, gr in grippers.items():
                gr.update(arms[nm].r.fkine(arms[nm].r.q), world.holding_half(nm))
            ctl.t = t
            yield
            t += DT
            if not any(ctl.owns(n) for n in arms):                  # (a held arm is not a deadlock)
                guard += 1
            if all_done:
                break
            if guard > 60000:
                raise RuntimeError(f"phase '{ph.name}' did not finish (deadlock?)")
    # let the customer finish leaving
    for _ in range(3000):
        if customer.finished:
            break
        customer.update(DT, flags)
        scanner.scan(customer.xy())
        world.update()
        ctl.t = t
        yield
        t += DT
    return t, stats, min_gap




def run_sim(phases, arms, world, lamps, scanner, customer, flags, grippers, step_fn, log, ctl=None):
    """Run the whole café once, calling step_fn() after every tick (no control panel)."""
    gen = sim_gen(phases, arms, world, lamps, scanner, customer, flags, grippers, log, ctl or Controls(arms))
    while True:
        try:
            next(gen)
        except StopIteration as stop:
            return stop.value
        step_fn()


def report(result, log):
    t, stats, gap = result
    log(f"\ncompleted in {t:.0f} s of simulated time")
    for nm, s in stats.items():
        log(f"{nm:>3}: stopped {s['stops']}x ({s['stop_s']:.1f} s), slowed {s['slow_s']:.1f} s")
    log(f"closest approach between two robots' links = {gap[0]:.3f} m ({gap[1]})")




# =============================== the simulation as a state machine ===============================
ROBOT_TITLE = {"dsn": "Doosan M0609", "ur": "UR3e", "gp7": "Yaskawa GP7"}


class Session:
    """The café run, driven by the control window:   READY -> RUNNING <-> PAUSED -> DONE
    (Stop Task / Reset goes back to READY; the emergency stop latches on top of any state)."""

    def __init__(self, sc, log=print):
        self.sc, self._out = sc, log
        self.arms, self.world = sc.arms, sc.world
        self.ctl = Controls(sc.arms)
        self.state, self.gen, self.result = "READY", None, None
        self.use_customer, self.speed, self.quit = True, 1.0, False
        self.estop, self.resume_required = False, False
        self.events = []                                           # every log line, for the window's Event Log tab
        self.window = None
        self.frames, self._acc = 0, 0.0

    def log(self, text):
        self._out(text)
        self.events.append(text.strip("\n"))

    def refresh(self):
        """Redraw cups, lids and grippers from the arms' current joint angles (needed while the plan is not ticking)."""
        self.world.update()
        for nm, gr in self.sc.grippers.items():
            gr.update(self.arms[nm].r.fkine(self.arms[nm].r.q), self.world.holding_half(nm))

    # ---- whole café
    def start(self):
        if self.estop:
            return
        if self.state == "DONE":
            self.reset(quiet=True)
        if self.state == "READY":
            sc = self.sc
            sc.customer.script = sc.script if self.use_customer else [("hide", None)]
            sc.customer.reset()
            self.gen = sim_gen(sc.phases, sc.arms, sc.world, sc.lamps, sc.scanner, sc.customer, sc.flags,
                               sc.grippers, self.log, self.ctl)
            self.log("-- START")
            self.state = "RUNNING"
        elif self.state == "PAUSED":
            self.resume()

    def pause(self):
        if self.state == "RUNNING":
            self.state = "PAUSED"
            self.log("-- PAUSE (the task is kept; press Resume)")

    def resume(self):
        if self.state == "PAUSED" and not self.estop:
            self.state, self.resume_required = "RUNNING", False
            self.log("-- RESUME")

    stop = pause                                                   # (older name)

    def stop_task(self):
        """Cancel the task: everything goes back to the start."""
        self.reset(quiet=True)
        self.log("-- STOP TASK (cancelled, everything back at the start)")

    def reset(self, quiet=False):
        sc = self.sc
        self.gen, self.result, self._acc = None, None, 0.0
        self.resume_required = False
        sc.flags.clear()
        sc.world.reset()
        sc.customer.reset()
        self.ctl.reset()
        for a in sc.arms.values():
            a.r.q = a.q_ready.copy()
        for lamp in sc.lamps.values():
            lamp.show("STOP" if self.estop else "RUN")
        sc.customer.update(0.0, sc.flags)                          # hides the customer again
        sc.scanner.scan(None)
        self.refresh()
        self.state = "READY"
        if not quiet:
            self.log("-- RESET")

    # ---- emergency stop
    def emergency_stop(self):
        if self.estop:
            return
        self.estop = True
        for lamp in self.sc.lamps.values():
            lamp.show("STOP")
        self.log("!! EMERGENCY STOP - everything frozen")

    def acknowledge(self):
        """RESET / ACKNOWLEDGE: clears the latch. A running task is kept but must be resumed by hand."""
        if not self.estop:
            return
        self.estop = False
        if self.state == "RUNNING":
            self.state, self.resume_required = "PAUSED", True
        for nm, lamp in self.sc.lamps.items():
            lamp.show(self.ctl.safety[nm])
        self.log("E-stop cleared" + (" - press Resume to continue the task" if self.resume_required else ""))

    # ---- one robot, by hand
    def enable_manual(self, nm):
        if self.ctl.mode[nm] == "auto":
            self.ctl.take(nm)
            self.log(f"{ROBOT_TITLE[nm]}: manual mode (the line waits for this arm)")

    def disable_manual(self, nm):
        if self.ctl.mode[nm] == "manual":
            self.ctl.release(nm)
            self.log(f"{ROBOT_TITLE[nm]}: returning to the plan")

    def jog(self, nm, j, deg):
        """Set joint j of one arm to deg (it leaves the plan: manual mode). Stops at the joint limits."""
        r = self.arms[nm].r
        self.ctl.take(nm)
        self.ctl.goto[nm] = []
        q = r.q.copy()
        q[j] = float(np.clip(np.radians(deg), r.qlim[0, j], r.qlim[1, j]))
        r.q = q

    def jog_joint(self, nm, j, delta_deg):
        if self.estop or self.ctl.mode[nm] != "manual":
            return False
        r = self.arms[nm].r
        target = np.degrees(r.q[j]) + delta_deg
        lo, hi = np.degrees(r.qlim[0, j]), np.degrees(r.qlim[1, j])
        if not lo <= target <= hi:
            self.log(f"{ROBOT_TITLE[nm]}: joint {j + 1} at its limit ({lo:.0f} to {hi:.0f} deg)")
        self.jog(nm, j, target)
        return True

    def tcp(self, nm):
        """World position of the tool point (centre of the gripper pads)."""
        T = self.arms[nm].r.fkine(self.arms[nm].r.q)
        return T.t + GRIP * T.R[:, 2]

    def jog_tcp(self, nm, axis, delta_m):
        """Move the tool point along a world axis, keeping the gripper orientation (inverse kinematics)."""
        if self.estop or self.ctl.mode[nm] != "manual":
            return False
        r = self.arms[nm].r
        T = r.fkine(r.q)
        d = np.zeros(3)
        d[axis] = delta_m
        q, ok = dls_ik(r, SE3.Rt(T.R, T.t + d).A, r.q, np.ones(6), iters=100)
        if ok:
            q = unwrap(r, q, r.q)
            ok = bool(np.all(q >= r.qlim[0] - 1e-6) and np.all(q <= r.qlim[1] + 1e-6))
        if not ok:
            self.log(f"{ROBOT_TITLE[nm]}: that point is out of reach or past a joint limit")
            return False
        self.ctl.goto[nm] = []
        r.q = q
        return True

    def home(self, nm):
        """Manual mode: glide the selected arm to its ready pose."""
        if self.estop or self.ctl.mode[nm] != "manual":
            return
        r, a = self.arms[nm].r, self.arms[nm]
        n = max(25, int(np.max(np.abs(a.q_ready - r.q)) / JSPEED / DT))
        self.ctl.goto[nm] = list(jtraj(r.q, a.q_ready, n).q)
        self.log(f"{ROBOT_TITLE[nm]}: going to its ready pose")

    # ---- frame loop
    def frame(self):
        """Call once per Swift frame."""
        if not self.estop:
            self.ctl.advance()
        ticked = False
        if self.state == "RUNNING" and not self.estop:
            self._acc += self.speed
            for _ in range(8):
                if self._acc < 1.0:
                    break
                self._acc -= 1.0
                try:
                    next(self.gen)
                    ticked = True
                except StopIteration as stop:
                    self.result = stop.value
                    self.state = "DONE"
                    report(stop.value, self.log)
                    break
            self._acc = min(self._acc, 1.0)
        if not ticked:
            self.refresh()
        self.frames += 1
        if self.window is not None:
            self.window.poll(self.frames)

    def drinks_ready(self):
        w = self.world
        return sum(1 for k in range(3) if w.lid_holder[k] == "on_cup" and
                   (w.cup_holder[k] == "person" or (w.cup_holder[k] is None and w.cup_pos[k][0] > 0.5)))


# =============================== control window (Tkinter, separate desktop window) ===============================
STATE_BADGE = {"READY": ("IDLE", "#5b7aa5"), "RUNNING": ("RUNNING", "#2e8b3d"), "PAUSED": ("PAUSED", "#d19a00"),
               "DONE": ("FINISHED", "#3a6ea5"), "ESTOP": ("E-STOP", "#c62828")}


class ControlWindow:
    """'Workcell Control' window: pick a robot, enable manual mode, jog its joints or its tool point, and start / pause /
    resume / stop the coffee process, with a speed slider, a safety tab, an event log and an emergency stop."""

    JOINT_STEP_DEG, TCP_STEP_MM = 2.0, 5.0
    REPEAT_FIRST_MS, REPEAT_MS = 350, 70

    def __init__(self, sess):
        import tkinter as tk
        from tkinter import font, ttk
        self.tk, self.ttk, self.sess = tk, ttk, sess
        self.titles = {ROBOT_TITLE[nm]: nm for nm in sess.arms}
        self._log_i = 0
        root = self.root = tk.Tk()
        root.title("Café Drink Line - Workcell Control")
        root.minsize(860, 700)
        root.protocol("WM_DELETE_WINDOW", lambda: setattr(sess, "quit", True))
        root.bind("<Escape>", lambda e: sess.emergency_stop())
        base = font.nametofont("TkDefaultFont")
        big = base.copy()
        big.configure(size=base.cget("size") + 8, weight="bold")
        bold = base.copy()
        bold.configure(weight="bold")

        # ---- title + state badge
        top = ttk.Frame(root, padding=(12, 10, 12, 4))
        top.pack(fill="x")
        top.columnconfigure(0, weight=1)
        ttk.Label(top, text="Café Drink Line Workcell Control", font=big).grid(row=0, column=0, sticky="w")
        self.badge = tk.Label(top, text="IDLE", width=16, fg="white", bg="#5b7aa5", font=bold, pady=5)
        self.badge.grid(row=0, column=1, sticky="e")

        # ---- robot selection
        sel = ttk.Frame(root, padding=(12, 2))
        sel.pack(fill="x")
        sel.columnconfigure(2, weight=1)
        ttk.Label(sel, text="Selected robot:").grid(row=0, column=0, sticky="w")
        self.robot_var = tk.StringVar(value=ROBOT_TITLE["dsn"])
        self.combo = ttk.Combobox(sel, textvariable=self.robot_var, values=list(self.titles), state="readonly", width=18)
        self.combo.grid(row=0, column=1, padx=6, sticky="w")
        self.combo.bind("<<ComboboxSelected>>", lambda e: self.refresh(force=True))
        self.scene_lbl = ttk.Label(sel, text="Scene: READY")
        self.scene_lbl.grid(row=0, column=3, padx=20)
        self.mode_lbl = ttk.Label(sel, text="Mode: AUTO", font=bold)
        self.mode_lbl.grid(row=0, column=4)

        # ---- main buttons
        bar = ttk.Frame(root, padding=(12, 6))
        bar.pack(fill="x")
        self.btn = {}
        for k, (key, text, cmd) in enumerate([
                ("manual", "Enable Manual", self._toggle_manual), ("start", "Start Coffee", sess.start),
                ("pause", "Pause", sess.pause), ("resume", "Resume", sess.resume),
                ("stop", "Stop Task", sess.stop_task), ("home", "Home / Ready", self._home)]):
            b = ttk.Button(bar, text=text, command=cmd)
            b.grid(row=0, column=k, sticky="ew", padx=3, ipady=4)
            bar.columnconfigure(k, weight=1, uniform="b")
            self.btn[key] = b

        # ---- speed
        sp = ttk.Frame(root, padding=(12, 0))
        sp.pack(fill="x")
        sp.columnconfigure(1, weight=1)
        ttk.Label(sp, text="Speed:").grid(row=0, column=0, sticky="w")
        self.speed_var = tk.DoubleVar(value=100.0)
        ttk.Scale(sp, from_=10, to=300, variable=self.speed_var, command=self._on_speed).grid(row=0, column=1, sticky="ew", padx=8)
        self.speed_lbl = ttk.Label(sp, text="100%", width=6)
        self.speed_lbl.grid(row=0, column=2)
        ttk.Label(sp, text="Pause keeps the task. Stop Task cancels it.").grid(row=0, column=3, padx=(24, 0))

        # ---- notebook
        nb = self.nb = ttk.Notebook(root)
        nb.pack(fill="both", expand=True, padx=12, pady=8)
        jog = ttk.Frame(nb, padding=8)
        safety = ttk.Frame(nb, padding=8)
        logf = ttk.Frame(nb, padding=8)
        nb.add(jog, text="Teach / Jog")
        nb.add(safety, text="Safety Inputs")
        nb.add(logf, text="Event Log")
        self._build_jog(jog, bold)
        self._build_safety(safety, bold)
        self._build_log(logf)

        # ---- shared safety controls
        sh = ttk.LabelFrame(root, text="Shared Safety Controls", padding=8)
        sh.pack(fill="x", padx=12, pady=(0, 6))
        for c in range(4):
            sh.columnconfigure(c, weight=1)
        self.sc_lbl = {k: ttk.Label(sh, text=f"{k}: -") for k in ("Scanner", "Zone", "Latch", "Resume")}
        for c, lbl in enumerate(self.sc_lbl.values()):
            lbl.grid(row=0, column=c, sticky="w")
        self.ack = ttk.Button(sh, text="RESET / ACKNOWLEDGE", command=sess.acknowledge)
        self.ack.grid(row=1, column=0, columnspan=2, sticky="ew", padx=(0, 8), pady=(8, 0), ipady=8)
        self.estop = tk.Button(sh, text="EMERGENCY STOP  [Esc]", command=sess.emergency_stop, bg="#d32020", fg="white",
                               activebackground="#a81010", activeforeground="white", font=bold, relief="flat")
        self.estop.grid(row=1, column=2, columnspan=2, sticky="ew", pady=(8, 0), ipady=8)
        ttk.Label(root, text="Enable Manual freezes the selected arm (the line waits for it) and unlocks the jog buttons; "
                             "Disable Manual sends it back to where the plan had it. No collision checking in manual mode.",
                  wraplength=820, padding=(12, 0, 12, 8)).pack(fill="x")
        self.refresh(force=True)

    # ---------------------------------------------------------------- tabs
    def _hold(self, button, fn):
        """Call fn once on press, then repeatedly while the mouse button stays down."""
        job = {"id": None}

        def again():
            fn()
            job["id"] = self.root.after(self.REPEAT_MS, again)

        def press(_):
            if button.instate(["disabled"]):
                return
            fn()
            job["id"] = self.root.after(self.REPEAT_FIRST_MS, again)

        def release(_):
            if job["id"] is not None:
                self.root.after_cancel(job["id"])
                job["id"] = None

        button.bind("<ButtonPress-1>", press, add="+")
        button.bind("<ButtonRelease-1>", release, add="+")
        button.bind("<Leave>", release, add="+")

    def _build_jog(self, parent, bold):
        tk, ttk = self.tk, self.ttk
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
        jf = ttk.LabelFrame(parent, text="Joint Jog", padding=8)
        jf.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        tf = ttk.LabelFrame(parent, text="TCP Jog - World XYZ", padding=8)
        tf.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        for c in (0, 1, 2, 3):
            jf.columnconfigure(c, weight=1)
            tf.columnconfigure(c, weight=1)
        ttk.Label(jf, text="Joint", font=bold).grid(row=0, column=0)
        ttk.Label(jf, text="Current angle", font=bold).grid(row=0, column=2)
        self.jbtn, self.jang = [], []
        for j in range(6):
            ttk.Label(jf, text=f"J{j + 1}").grid(row=j + 1, column=0, pady=3)
            m = ttk.Button(jf, text="-", width=4)
            p = ttk.Button(jf, text="+", width=4)
            m.grid(row=j + 1, column=1, pady=3, ipady=2)
            p.grid(row=j + 1, column=3, pady=3, ipady=2)
            lbl = ttk.Label(jf, text="0.00°")
            lbl.grid(row=j + 1, column=2)
            self._hold(m, lambda j=j: self._jog_joint(j, -1))
            self._hold(p, lambda j=j: self._jog_joint(j, +1))
            self.jbtn += [m, p]
            self.jang.append(lbl)
        ttk.Label(jf, text="Step:").grid(row=8, column=1, sticky="e", pady=(8, 0))
        self.jstep = tk.DoubleVar(value=self.JOINT_STEP_DEG)
        ttk.Spinbox(jf, from_=0.1, to=20, increment=0.5, textvariable=self.jstep, width=6).grid(row=8, column=2, pady=(8, 0))
        ttk.Label(jf, text="deg").grid(row=8, column=3, sticky="w", pady=(8, 0))

        ttk.Label(tf, text="TCP position", font=bold).grid(row=0, column=1, columnspan=2)
        self.tbtn, self.tpos = [], []
        for k, ax in enumerate("XYZ"):
            ttk.Label(tf, text=ax).grid(row=k + 1, column=0, pady=6)
            m = ttk.Button(tf, text="-", width=4)
            p = ttk.Button(tf, text="+", width=4)
            m.grid(row=k + 1, column=1, ipady=2)
            p.grid(row=k + 1, column=3, ipady=2)
            lbl = ttk.Label(tf, text="0.0000 m")
            lbl.grid(row=k + 1, column=2)
            self._hold(m, lambda k=k: self._jog_tcp(k, -1))
            self._hold(p, lambda k=k: self._jog_tcp(k, +1))
            self.tbtn += [m, p]
            self.tpos.append(lbl)
        ttk.Label(tf, text="Step:").grid(row=4, column=1, sticky="e", pady=(10, 0))
        self.tstep = tk.DoubleVar(value=self.TCP_STEP_MM)
        ttk.Spinbox(tf, from_=1, to=50, increment=1, textvariable=self.tstep, width=6).grid(row=4, column=2, pady=(10, 0))
        ttk.Label(tf, text="mm").grid(row=4, column=3, sticky="w", pady=(10, 0))
        self.tcp_note = ttk.Label(tf, text="", wraplength=330, justify="left")
        self.tcp_note.grid(row=5, column=0, columnspan=4, sticky="w", pady=(10, 0))

    def _build_safety(self, parent, bold):
        tk, ttk = self.tk, self.ttk
        for c, h in enumerate(("Robot", "Clearance to customer", "Safety state", "Mode")):
            ttk.Label(parent, text=h, font=bold).grid(row=0, column=c, sticky="w", padx=8, pady=4)
            parent.columnconfigure(c, weight=1)
        self.srow = {}
        for r, nm in enumerate(self.sess.arms, start=1):
            cells = [ttk.Label(parent, text="-") for _ in range(4)]
            cells[0].configure(text=ROBOT_TITLE[nm])
            for c, w in enumerate(cells):
                w.grid(row=r, column=c, sticky="w", padx=8, pady=4)
            self.srow[nm] = cells
        self.person_lbl = ttk.Label(parent, text="Customer: not detected")
        self.person_lbl.grid(row=5, column=0, columnspan=4, sticky="w", padx=8, pady=(14, 4))
        self.cust_var = tk.BooleanVar(value=self.sess.use_customer)
        ttk.Checkbutton(parent, text="Customer walks in (used at the next Start Coffee)", variable=self.cust_var,
                        command=lambda: setattr(self.sess, "use_customer", bool(self.cust_var.get()))
                        ).grid(row=6, column=0, columnspan=4, sticky="w", padx=8, pady=4)
        ttk.Label(parent, text=f"The scanner slows an arm below {SLOW_D:.2f} m clearance and stops it below {STOP_D:.2f} m. "
                               f"It restarts {RESUME_S:.1f} s after the customer has moved away.",
                  wraplength=780).grid(row=7, column=0, columnspan=4, sticky="w", padx=8, pady=4)

    def _build_log(self, parent):
        tk, ttk = self.tk, self.ttk
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)
        self.log_text = tk.Text(parent, height=10, state="disabled", wrap="word", font="TkFixedFont")
        sb = ttk.Scrollbar(parent, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=sb.set)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")

    # ---------------------------------------------------------------- actions
    def nm(self):
        return self.titles[self.robot_var.get()]

    def _on_speed(self, v):
        pct = float(v)
        self.sess.speed = pct / 100.0
        self.speed_lbl.configure(text=f"{pct:.0f}%")

    def _toggle_manual(self):
        nm = self.nm()
        if self.sess.ctl.mode[nm] == "auto":
            self.sess.enable_manual(nm)
        else:
            self.sess.disable_manual(nm)
        self.refresh(force=True)

    def _home(self):
        self.sess.home(self.nm())

    def _jog_joint(self, j, sign):
        self.sess.jog_joint(self.nm(), j, sign * self._num(self.jstep, self.JOINT_STEP_DEG))
        self.refresh(force=True)

    def _jog_tcp(self, axis, sign):
        self.sess.jog_tcp(self.nm(), axis, sign * self._num(self.tstep, self.TCP_STEP_MM) / 1000.0)
        self.refresh(force=True)

    @staticmethod
    def _num(var, default):
        try:
            return float(var.get())
        except Exception:
            return default

    # ---------------------------------------------------------------- display
    def poll(self, frame_no):
        """Once per Swift frame: let Tk handle clicks, refresh the read-outs about 10 times a second."""
        try:
            self.root.update()
        except self.tk.TclError:                       # window was closed
            self.sess.quit = True
            return
        if frame_no % 5 == 0:
            self.refresh()

    def close(self):
        try:
            self.root.destroy()
        except Exception:
            pass

    def _enable(self, widget, on):
        widget.state(["!disabled"] if on else ["disabled"])

    def refresh(self, force=False):
        s, ctl = self.sess, self.sess.ctl
        nm = self.nm()
        st = "ESTOP" if s.estop else s.state
        text, colour = STATE_BADGE[st]
        self.badge.configure(text=text, bg=colour)
        self.scene_lbl.configure(text=f"Scene: {s.state}   t = {ctl.t:5.1f} s   drinks ready {s.drinks_ready()}/3")
        mode = ctl.mode[nm]
        manual = mode == "manual"
        self.mode_lbl.configure(text="Mode: " + {"auto": "AUTO", "manual": "MANUAL", "glide": "RETURNING"}[mode])
        running, paused = s.state == "RUNNING", s.state == "PAUSED"
        ok = not s.estop
        self.btn["manual"].configure(text="Disable Manual" if manual else "Enable Manual")
        self._enable(self.btn["manual"], ok and mode != "glide")
        self._enable(self.btn["start"], ok and s.state in ("READY", "DONE"))
        self._enable(self.btn["pause"], ok and running)
        self._enable(self.btn["resume"], ok and paused)
        self._enable(self.btn["stop"], ok and s.state != "READY")
        self._enable(self.btn["home"], ok and manual)
        for b in self.jbtn + self.tbtn:
            self._enable(b, ok and manual)
        self._enable(self.ack, s.estop)
        r = s.arms[nm].r
        q = np.degrees(r.q)
        for j in range(6):
            self.jang[j].configure(text=f"{q[j]:7.2f}°")
        tcp = s.tcp(nm)
        for k in range(3):
            self.tpos[k].configure(text=f"{tcp[k]:7.4f} m")
        self.tcp_note.configure(text=f"{ROBOT_TITLE[nm]}: TCP = centre of the gripper pads ({GRIP * 1000:.0f} mm beyond the "
                                     f"flange). Grasp and release run automatically during the coffee process.")
        # safety tab + shared safety controls
        worst = "RUN"
        for n, cells in self.srow.items():
            clr = ctl.clearance.get(n)
            cells[1].configure(text="-" if clr is None else f"{clr:5.2f} m")
            state = ctl.safety[n]
            cells[2].configure(text={"RUN": "RUN (clear)", "SLOW": "SLOW", "STOP": "STOP"}[state])
            cells[3].configure(text=ctl.mode[n].upper())
            worst = "STOP" if state == "STOP" or worst == "STOP" else ("SLOW" if state == "SLOW" else worst)
        p = ctl.person
        self.person_lbl.configure(text="Customer: not detected" if p is None else f"Customer detected by the scanner at x = {p[0]:.2f} m, y = {p[1]:.2f} m")
        self.sc_lbl["Scanner"].configure(text="Scanner: " + ("CLEAR" if p is None else "CUSTOMER SEEN"))
        self.sc_lbl["Zone"].configure(text="Zone: " + {"RUN": "CLEAR", "SLOW": "SLOW", "STOP": "STOP"}[worst])
        self.sc_lbl["Latch"].configure(text="Latch: " + ("E-STOP" if s.estop else "CLEAR"))
        self.sc_lbl["Resume"].configure(text="Resume: " + ("REQUIRED" if s.resume_required or s.estop else "--"))
        # event log
        if self._log_i < len(s.events):
            self.log_text.configure(state="normal")
            for line in s.events[self._log_i:]:
                self.log_text.insert("end", line + "\n")
            self._log_i = len(s.events)
            self.log_text.see("end")
            self.log_text.configure(state="disabled")


# =============================== main ===============================
def build(log):
    """Plan everything once: robots, world, motion plan, floor zones, scene objects. Nothing is drawn yet."""
    dsn_r, ur_r, gp7_r = DoosanM0609(), UR3e(), YaskawaGP7()
    dsn_r.base = SE3(-ROBOT_X, 0, TABLE_H)
    ur_r.base = SE3(0, 0, TABLE_H)
    gp7_r.base = SE3(ROBOT_X, 0, TABLE_H) * SE3.Rz(pi)          # GP7 faces the other two robots

    # UR3e waiting pose: tool straight down above the bar, found with IK
    Rur = np.diag([1.0, -1.0, -1.0])
    q_ur = None
    for s in ([0, -90, 90, -90, -90, 0], [180, -90, 90, -90, -90, 0], [90, -90, 90, -90, -90, 0], [-90, -90, 90, -90, -90, 0]):
        q, ok = dls_ik(ur_r, SE3.Rt(Rur, [0.0, 0.30, TABLE_H + 0.40]).A, np.radians(s), np.ones(6))
        if ok:
            q_ur = unwrap(ur_r, q, np.radians(s))
            break
    if q_ur is None:
        sys.exit("could not find a tool-down waiting pose for the UR3e")

    arms = {"dsn": Arm("dsn", dsn_r, dsn_r.home_q),
            "ur": Arm("ur", ur_r, q_ur),
            "gp7": Arm("gp7", gp7_r, np.radians(GP7_READY_DEG))}
    for a in arms.values():
        a.r.q = a.q_ready

    world = World(arms)
    flags = {}
    phases = build_plan(arms, world, flags, log)
    for a in arms.values():
        a.q = a.q_ready.copy()                                   # (plan ends with every arm back at ready)
    n_frames = sum(max([len(tr.frames) for tr in p.tracks.values()] + [p.process[0] if p.process else 0]) for p in phases)
    log(f"{len(phases)} phases, about {n_frames * DT:.0f} s of robot motion without interruptions")
    for a in arms.values():
        log(f"{a.name:>3}: lowest joint frame {a.min_z - TABLE_H:+.3f} m above the bar")

    xs, ys, stop_m, slow_m = compute_zones(arms, phases)
    log(f"floor zones: STOP area {stop_m.sum() * ZONE_CELL ** 2:.1f} m2 (red), SLOW area {slow_m.sum() * ZONE_CELL ** 2:.1f} m2 (amber)")
    zone_shapes = mask_to_boxes(stop_m, xs, ys, (1.0, 0.12, 0.12, 0.55)) + mask_to_boxes(slow_m, xs, ys, (1.0, 0.78, 0.1, 0.42))

    lamps = {"dsn": Lamp(-ROBOT_X, 0.40), "ur": Lamp(0.0, 0.40), "gp7": Lamp(ROBOT_X, 0.40)}
    grippers = {"dsn": Gripper((0.35, 0.35, 0.38, 1), (0.9, 0.9, 0.9, 1)),
                "ur": Gripper((0.25, 0.25, 0.3, 1), (0.8, 0.8, 0.85, 1)),
                "gp7": Gripper((0.2, 0.2, 0.25, 1), (0.95, 0.75, 0.1, 1))}
    scanner = Scanner()

    def take_cup():
        world.cup_holder[0] = "person"
        log("customer takes drink 1")

    door, door_in = (2.7, -3.9), (2.7, -3.0)
    script = [("wait_flag", "go"), ("show", door), ("walk", door_in), ("walk", (0.62, -0.78)), ("wait", 2.5),
              ("walk", (1.7, -2.3)), ("wait_flag", "go2"), ("wait", SECOND_VISIT_DELAY),
              ("walk", (0.62, -0.80)), ("wait", 1.0), ("do", take_cup), ("wait", 0.8),
              ("walk", door_in), ("walk", door), ("hide", None)]
    customer = Customer(script, world)
    return SimpleNamespace(arms=arms, world=world, flags=flags, phases=phases, lamps=lamps, grippers=grippers,
                           scanner=scanner, customer=customer, script=script, zone_shapes=zone_shapes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true", help="plan, simulate and report only, no Swift window")
    ap.add_argument("--no-customer", action="store_true", help="start with the customer switched off")
    ap.add_argument("--auto", action="store_true", help="no control panel: start at once and run the café one time")
    args = ap.parse_args()
    log = print

    sc = build(log)

    if args.headless:
        sc.customer.script = [("hide", None)] if args.no_customer else sc.script
        report(run_sim(sc.phases, sc.arms, sc.world, sc.lamps, sc.scanner, sc.customer, sc.flags, sc.grippers,
                       lambda: None, log), log)
        return

    from swift import Swift
    env = Swift()
    env.launch(realtime=True)
    env.set_camera_pose([-2.0, -3.0, 1.9], [0.2, -0.1, 0.95])
    sess = Session(sc, log)
    sess.use_customer = not args.no_customer
    if not args.auto:
        try:
            sess.window = ControlWindow(sess)                      # separate desktop window, there while the café loads
        except Exception as exc:                                   # no tkinter / no display
            log(f"could not open the control window ({exc}); running with --auto instead")
            args.auto = True
    for a in sc.arms.values():
        a.r.add_to_env(env)
    for sh in make_room() + make_decor() + make_bar() + make_stations() + sc.zone_shapes + sc.world.shapes() + sc.scanner.shapes() + sc.customer.shapes():
        env.add(sh)
    for lamp in sc.lamps.values():
        for sh in lamp.shapes():
            env.add(sh)
    for g in sc.grippers.values():
        for sh in g.shapes():
            env.add(sh)
    sess.refresh()
    env.step(0.5)

    if args.auto:
        sess.start()
    else:
        log("control window is open - press Start Coffee")
    try:
        while not sess.quit:
            env.step(DT)
            sess.frame()
            if args.auto and sess.state == "DONE":
                break
    except KeyboardInterrupt:
        return
    finally:
        if sess.window is not None:
            sess.window.close()
    if args.auto and sess.state == "DONE":
        log("done - close the Swift window or press Ctrl+C to exit")
        env.hold()


if __name__ == "__main__":
    main()