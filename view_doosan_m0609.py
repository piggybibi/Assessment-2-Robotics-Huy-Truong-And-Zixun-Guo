"""
View the Doosan M0609 in Swift.

Opens the robot on a small stand, then moves each joint through its range, one joint at a time, so you
can check the 3D model and the joint limits.

Setup
-----
Copy the DoosanM0609 folder into  <your venv>/Lib/site-packages/ir_support/robots/  (next to UR3e), then:

    python view_doosan_m0609.py              # open Swift and sweep the joints
    python view_doosan_m0609.py --still      # open Swift and just show the home pose
    python view_doosan_m0609.py --headless   # no window: print the DH table, joint limits and sweep ranges
"""
import argparse
from math import degrees, radians

import numpy as np
import spatialgeometry as sg
from spatialmath import SE3

from ir_support.robots.DoosanM0609 import DoosanM0609

STAND_H = 0.45                 # height of the stand the robot is mounted on (m)
SWEEP_MAX = radians(90)        # sweep each joint at most this far either side of home
STEP = 0.02                    # seconds per Swift frame
FRAMES = 50                    # frames for each leg of a joint sweep


def sweep_range(robot, q_home, j):
    """Lowest and highest value for joint j: within its limits, within SWEEP_MAX of home, and stopping
    before any joint of the arm would drop below the top of the stand."""
    floor_z = robot.base.t[2]

    def ok(v):
        q = q_home.copy()
        q[j] = v
        return min(T.t[2] for T in robot.fkine_all(q)) >= floor_z - 1e-6

    lo_lim, hi_lim = robot.qlim[0, j], robot.qlim[1, j]
    ends = []
    for sign in (-1, 1):
        v = q_home[j]
        while True:
            nxt = v + sign * radians(1)
            if abs(nxt - q_home[j]) > SWEEP_MAX + 1e-9 or not (lo_lim - 1e-9 <= nxt <= hi_lim + 1e-9) or not ok(nxt):
                break
            v = nxt
        ends.append(v)
    return ends[0], ends[1]


def joint_sweep(q_home, j, lo, hi, frames=FRAMES):
    """Joint configurations that take joint j from home down to lo, up to hi, then back home."""
    legs = [(q_home[j], lo, frames), (lo, hi, 2 * frames), (hi, q_home[j], frames)]
    for a, b, n in legs:
        for v in np.linspace(a, b, n):
            q = q_home.copy()
            q[j] = v
            yield q


def main():
    ap = argparse.ArgumentParser(description="View the Doosan M0609 model")
    ap.add_argument("--still", action="store_true", help="show the home pose only, no joint sweep")
    ap.add_argument("--headless", action="store_true", help="no Swift window; print a report only")
    args = ap.parse_args()

    robot = DoosanM0609(base=SE3(0, 0, STAND_H))
    q_home = np.array(robot.home_q, dtype=float)
    ranges = [sweep_range(robot, q_home, j) for j in range(robot.n)]

    print(robot)                                         # DH table
    print("\njoint   limits (deg)        sweep shown (deg)")
    for j in range(robot.n):
        print(f"  {j + 1}   {degrees(robot.qlim[0, j]):7.1f} to {degrees(robot.qlim[1, j]):6.1f}"
              f"   {degrees(ranges[j][0]) + 0.0:7.1f} to {degrees(ranges[j][1]) + 0.0:6.1f}")
    print(f"\nhome pose (deg): {np.round(np.degrees(q_home), 1)}")

    if args.headless:
        return

    from swift import Swift
    env = Swift()
    env.launch(realtime=True)
    env.set_camera_pose([1.4, -1.4, 1.3], [0.0, 0.0, 0.85])

    env.add(sg.Cuboid(scale=[3.0, 3.0, 0.01], pose=SE3(0, 0, -0.005), color=(0.80, 0.80, 0.82, 1)))
    env.add(sg.Cuboid(scale=[0.30, 0.30, STAND_H], pose=SE3(0, 0, STAND_H / 2), color=(0.35, 0.37, 0.40, 1)))
    robot.add_to_env(env)
    robot.q = q_home
    env.step(0.5)

    if not args.still:
        for j in range(robot.n):
            print(f"sweeping joint {j + 1} ...")
            for q in joint_sweep(q_home, j, *ranges[j]):
                robot.q = q
                env.step(STEP)
        robot.q = q_home
        env.step(STEP)

    print("done - close the Swift window or press Ctrl+C to exit")
    env.hold()


if __name__ == "__main__":
    main()
