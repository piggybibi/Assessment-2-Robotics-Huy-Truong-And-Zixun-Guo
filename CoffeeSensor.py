"""
Laser scanner demo - the café's safety sensor on its own, in its own desktop window (Tkinter, no Swift, no robots)

The scanner is a 2-D laser on a post at the front of the bar. It fires 46 rays over the half plane in front of it
(range 6 m). The café walls are known in advance, so a ray that comes back SHORTER than its wall (by more than 15 cm)
has hit something new: the customer. The customer's position is estimated from those rays alone.

Top view (left)    grey lines = laser rays, blue dots = wall hits, red dots = customer hits,
                   green cross = position the scanner estimates, grey circle = where the customer really is
Range plot (below) the distance every ray measured: blue = the known wall, red = rays that hit the customer
Right              readout + controls

Use it
    * click or drag anywhere in the top view to put the customer there (arrow keys move it 5 cm)
    * "Auto walk" lets the customer walk in through the door, to the counter and out again
    * "Range noise" adds Gaussian noise to every range reading, to see how robust the detection is
In the café this estimate feeds the safety controller: an arm slows when the customer is within 0.80 m of it and
stops within 0.30 m (see cafe_drink_line.py).

Run:    python scanner_demo.py              (opens the window)
        python scanner_demo.py --selftest   (no window: prints the detection accuracy at a few spots)

Needs:  Python 3 with numpy and tkinter (both come with a normal python.org install + numpy).
"""
import argparse
import sys

import numpy as np

# ---- geometry (same numbers as cafe_drink_line.py) [m]
WALL_FRONT_Y, WALL_BACK_Y, WALL_X = -3.4, 1.6, 5.0
WALL_T = 0.10
IN_FRONT, IN_BACK, IN_X = WALL_FRONT_Y + WALL_T / 2, WALL_BACK_Y - WALL_T / 2, WALL_X - WALL_T / 2   # inner wall faces
DOOR_X0, DOOR_X1 = 2.2, 3.2
SCAN_ORIGIN = np.array([0.0, -0.58])
SCAN_RANGE = 6.0
N_RAYS = 46
PERSON_R = 0.22
NEW_OBJECT_MARGIN = 0.15            # a ray must be this much shorter than the known wall to count as a hit
BAR = (-1.5, 1.5, -0.45, 0.45)      # bar top (x0, x1, y0, y1)
ROBOTS = {"Doosan": -1.0, "UR3e": 0.0, "GP7": 1.0}
WALLS = [((-IN_X, IN_FRONT), (DOOR_X0, IN_FRONT)), ((DOOR_X1, IN_FRONT), (IN_X, IN_FRONT)),
         ((-IN_X, IN_FRONT), (-IN_X, IN_BACK)), ((IN_X, IN_FRONT), (IN_X, IN_BACK))]
DOOR, DOOR_IN = (2.7, -3.9), (2.7, -3.0)
WALK = [("walk", DOOR), ("walk", DOOR_IN), ("walk", (0.62, -0.78)), ("wait", 2.5), ("walk", (1.7, -2.3)),
        ("walk", (0.62, -0.80)), ("wait", 1.5), ("walk", DOOR_IN), ("walk", DOOR), ("wait", 1.5)]
WALK_SPEED = 1.1                    # m/s


# =============================== the sensor ===============================
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
    def __init__(self):
        self.ang = np.deg2rad(np.linspace(-178, -2, N_RAYS))
        self.dirs = np.column_stack([np.cos(self.ang), np.sin(self.ang)])
        walls = [(np.array(a), np.array(b)) for a, b in WALLS]
        # the known background: how far every ray goes when the café is empty
        self.wall_range = np.array([min(min(_ray_seg(SCAN_ORIGIN, d, a, b) for a, b in walls), SCAN_RANGE) for d in self.dirs])

    def measure(self, person_xy, noise_sd=0.0, rng=None):
        """What the scanner reads: the range of every ray (wall or customer, whichever is nearer)."""
        per = np.full(N_RAYS, np.inf)
        if person_xy is not None:
            per = np.array([_ray_circle(SCAN_ORIGIN, d, np.asarray(person_xy, float), PERSON_R) for d in self.dirs])
        rng_m = np.minimum(self.wall_range, per)
        if noise_sd > 0:
            rng_m = np.clip(rng_m + (rng or np.random).normal(0, noise_sd, N_RAYS), 0.05, SCAN_RANGE)
        return rng_m

    def detect(self, ranges):
        """From the ranges alone: which rays hit something new, and where the customer is. Returns (mask, (x, y) or None)."""
        mask = ranges < self.wall_range - NEW_OBJECT_MARGIN
        if not mask.any():
            return mask, None
        pts = np.array([SCAN_ORIGIN + self.dirs[i] * ranges[i] for i in range(N_RAYS) if mask[i]])
        c = pts.mean(axis=0)
        u = (c - SCAN_ORIGIN) / np.linalg.norm(c - SCAN_ORIGIN)
        return mask, c + PERSON_R * u                            # hits are on the near surface -> add the body radius


def selftest():
    sc = Scanner()
    spots = [(0.62, -0.78), (0.0, -1.5), (-2.0, -2.0), (2.7, -3.0), (3.5, -2.0), (-4.0, -3.0), (0.62, -0.30)]
    print(f"scanner at {SCAN_ORIGIN}, {N_RAYS} rays, {int((sc.wall_range < SCAN_RANGE - 1e-6).sum())} of them end on a wall")
    print("customer at          rays hit   estimated position     error")
    worst = 0.0
    for p in spots:
        mask, est = sc.detect(sc.measure(p))
        if est is None:
            print(f"  {p!s:<18}  {int(mask.sum()):>3}      not detected")
            worst = np.inf
            continue
        err = float(np.linalg.norm(est - p))
        worst = max(worst, err)
        print(f"  {p!s:<18}  {int(mask.sum()):>3}      ({est[0]:6.2f}, {est[1]:6.2f})        {err * 100:5.1f} cm")
    mask, est = sc.detect(sc.measure(None))
    print(f"  nobody             {int(mask.sum()):>3}      {'-' if est is None else est}")
    rng = np.random.default_rng(1)
    fp = sum(sc.detect(sc.measure(None, 0.03, rng))[1] is not None for _ in range(1000))
    print(f"empty café, 3 cm range noise: false detections in 1000 scans = {fp}")
    return worst


# =============================== the window ===============================
class Demo:
    S = 73.0                                 # pixels per metre
    X0, Y0 = -5.2, 1.85                      # world corner at the canvas top-left
    W, H = 760, 400
    PW, PH = 760, 190

    def __init__(self):
        import tkinter as tk
        from tkinter import font, ttk
        self.tk, self.ttk = tk, ttk
        self.sc = Scanner()
        self.rng = np.random.default_rng()
        self.person = np.array([0.62, -0.78])
        self.present = True
        self.auto, self.auto_i, self.auto_wait = False, 0, 0.0
        self.dragging = False

        root = self.root = tk.Tk()
        root.title("Laser scanner demo - café safety sensor")
        root.resizable(False, False)
        base = font.nametofont("TkDefaultFont")
        big = base.copy()
        big.configure(size=base.cget("size") + 4, weight="bold")
        bold = base.copy()
        bold.configure(weight="bold")
        mono = font.nametofont("TkFixedFont").copy()

        left = ttk.Frame(root, padding=8)
        left.grid(row=0, column=0)
        right = ttk.Frame(root, padding=(4, 8, 10, 8))
        right.grid(row=0, column=1, sticky="n")
        ttk.Label(left, text="Top view: the scanner, the café and the customer", font=bold).pack(anchor="w")
        self.cv = tk.Canvas(left, width=self.W, height=self.H, bg="#fafafa", highlightthickness=1, highlightbackground="#999")
        self.cv.pack()
        ttk.Label(left, text="Range plot: what every ray measured   (blue line = known wall, dashed = detection threshold, red = customer)", font=bold).pack(anchor="w", pady=(8, 0))
        self.pv = tk.Canvas(left, width=self.PW, height=self.PH, bg="#fafafa", highlightthickness=1, highlightbackground="#999")
        self.pv.pack()

        ttk.Label(right, text="Laser scanner", font=big).pack(anchor="w")
        self.badge = tk.Label(right, text="CLEAR", width=26, fg="white", bg="#2e8b3d", font=bold, pady=8)
        self.badge.pack(pady=8, fill="x")
        self.info = tk.Text(right, width=36, height=11, font=mono, relief="flat", bg=root.cget("bg"), state="disabled")
        self.info.pack(anchor="w")
        ttk.Separator(right).pack(fill="x", pady=8)
        self.btn_walk = ttk.Button(right, text="Auto walk", command=self.toggle_auto)
        self.btn_walk.pack(fill="x", ipady=4, pady=2)
        self.btn_present = ttk.Button(right, text="Remove customer", command=self.toggle_present)
        self.btn_present.pack(fill="x", ipady=4, pady=2)
        self.noise = tk.DoubleVar(value=0.0)
        ttk.Label(right, text="Range noise (std dev):").pack(anchor="w", pady=(10, 0))
        nf = ttk.Frame(right)
        nf.pack(fill="x")
        ttk.Scale(nf, from_=0, to=4, variable=self.noise, command=lambda v: self.noise_lbl.configure(text=f"{float(v):.1f} cm")
                  ).pack(side="left", fill="x", expand=True)
        self.noise_lbl = ttk.Label(nf, text="0.0 cm", width=8)
        self.noise_lbl.pack(side="left")
        self.show_rays = tk.BooleanVar(value=True)
        self.show_true = tk.BooleanVar(value=True)
        ttk.Checkbutton(right, text="Show laser rays", variable=self.show_rays).pack(anchor="w", pady=(8, 0))
        ttk.Checkbutton(right, text="Show where the customer really is", variable=self.show_true).pack(anchor="w")
        ttk.Label(right, wraplength=270, justify="left", padding=(0, 12, 0, 0),
                  text="Click or drag in the top view to move the customer, or use the arrow keys. In the café this position "
                       "decides when an arm slows down (clearance under 0.80 m) or stops (under 0.30 m).").pack(anchor="w")

        self.cv.bind("<Button-1>", self.on_mouse)
        self.cv.bind("<B1-Motion>", self.on_mouse)
        for key, d in (("Left", (-1, 0)), ("Right", (1, 0)), ("Up", (0, 1)), ("Down", (0, -1))):
            root.bind(f"<{key}>", lambda e, d=d: self.nudge(d))
        self.last = 0.0
        self.tick()

    # ---- coordinates
    def px(self, x, y):
        return (x - self.X0) * self.S, (self.Y0 - y) * self.S

    def world(self, px, py):
        return np.array([px / self.S + self.X0, self.Y0 - py / self.S])

    # ---- input
    def on_mouse(self, e):
        self.auto = False
        self.present = True
        self.person = self.world(e.x, e.y)

    def nudge(self, d):
        self.auto = False
        self.present = True
        self.person = self.person + 0.05 * np.array(d)

    def toggle_auto(self):
        self.auto = not self.auto
        if self.auto:
            self.present, self.auto_i, self.auto_wait = True, 0, 0.0
            self.person = np.array(WALK[0][1], float)

    def toggle_present(self):
        self.present = not self.present
        self.auto = False

    def walk(self, dt):
        kind, arg = WALK[self.auto_i % len(WALK)]
        if kind == "wait":
            self.auto_wait += dt
            if self.auto_wait >= arg:
                self.auto_wait, self.auto_i = 0.0, self.auto_i + 1
            return
        d = np.array(arg, float) - self.person
        dist = float(np.linalg.norm(d))
        step = WALK_SPEED * dt
        if dist <= step:
            self.person, self.auto_i = np.array(arg, float), self.auto_i + 1
        else:
            self.person = self.person + d / dist * step

    # ---- one frame
    def tick(self):
        import time
        now = time.time()
        dt = min(0.1, now - self.last) if self.last else 0.03
        self.last = now
        if self.auto:
            self.walk(dt)
        person = self.person if self.present else None
        ranges = self.sc.measure(person, self.noise.get() / 100.0, self.rng)
        mask, est = self.sc.detect(ranges)
        self.draw_top(ranges, mask, est)
        self.draw_plot(ranges, mask)
        self.update_info(ranges, mask, est)
        self.root.after(30, self.tick)

    def draw_top(self, ranges, mask, est):
        cv, sc = self.cv, self.sc
        cv.delete("all")
        # room
        x0, y0 = self.px(-IN_X, IN_BACK)
        x1, y1 = self.px(IN_X, IN_FRONT)
        cv.create_rectangle(x0, y0, x1, y1, fill="#f1ebe0", outline="")
        for (a, b) in WALLS:
            cv.create_line(*self.px(*a), *self.px(*b), width=4, fill="#6b4e32")
        cv.create_line(*self.px(-IN_X, IN_BACK), *self.px(IN_X, IN_BACK), width=4, fill="#6b4e32")
        dx0, dy = self.px(DOOR_X0, IN_FRONT)
        dx1, _ = self.px(DOOR_X1, IN_FRONT)
        cv.create_line(dx0, dy, dx1, dy, width=4, fill="#f1ebe0")
        cv.create_text((dx0 + dx1) / 2, dy - 10, text="door", fill="#6b4e32")
        # bar + robots
        bx0, by0 = self.px(BAR[0], BAR[3])
        bx1, by1 = self.px(BAR[1], BAR[2])
        cv.create_rectangle(bx0, by0, bx1, by1, fill="#a9805a", outline="#6b4e32")
        cv.create_text((bx0 + bx1) / 2, by1 + 11, text="bar", fill="#6b4e32")
        for name, x in ROBOTS.items():
            cx, cy = self.px(x, 0)
            cv.create_oval(cx - 9, cy - 9, cx + 9, cy + 9, fill="#4a5568", outline="")
            cv.create_text(cx, cy - 18, text=name, font=("TkDefaultFont", 8), fill="#333")
        # rays
        ox, oy = self.px(*SCAN_ORIGIN)
        if self.show_rays.get():
            for i in range(N_RAYS):
                p = SCAN_ORIGIN + sc.dirs[i] * ranges[i]
                cv.create_line(ox, oy, *self.px(*p), fill="#ff9a9a" if mask[i] else "#d7d7d7")
        # hits
        for i in range(N_RAYS):
            p = SCAN_ORIGIN + sc.dirs[i] * ranges[i]
            hx, hy = self.px(*p)
            if mask[i]:
                cv.create_oval(hx - 4, hy - 4, hx + 4, hy + 4, fill="#e53935", outline="")
            elif sc.wall_range[i] < SCAN_RANGE - 1e-6:
                cv.create_oval(hx - 3, hy - 3, hx + 3, hy + 3, fill="#3b82f6", outline="")
        # the real customer
        if self.present and self.show_true.get():
            cx, cy = self.px(*self.person)
            r = PERSON_R * self.S
            cv.create_oval(cx - r, cy - r, cx + r, cy + r, outline="#777", dash=(4, 3), width=2)
        # estimate
        if est is not None:
            ex, ey = self.px(*est)
            cv.create_line(ex - 9, ey, ex + 9, ey, fill="#16a34a", width=3)
            cv.create_line(ex, ey - 9, ex, ey + 9, fill="#16a34a", width=3)
        # scanner
        cv.create_oval(ox - 7, oy - 7, ox + 7, oy + 7, fill="#111", outline="#fff", width=2)
        cv.create_text(ox - 12, oy + 4, text="scanner", anchor="e", fill="#111", font=("TkDefaultFont", 9, "bold"))

    def draw_plot(self, ranges, mask):
        pv, sc = self.pv, self.sc
        pv.delete("all")
        L, R, T, B = 46, self.PW - 10, 10, self.PH - 30
        xs = lambda i: L + (R - L) * i / (N_RAYS - 1)
        ys = lambda r: B - (B - T) * min(r, SCAN_RANGE) / SCAN_RANGE
        for m in range(0, 7):
            pv.create_line(L, ys(m), R, ys(m), fill="#e4e4e4")
            pv.create_text(L - 6, ys(m), anchor="e", text=f"{m} m", fill="#777", font=("TkDefaultFont", 8))
        pv.create_text((L + R) / 2, self.PH - 10, text="ray number (left of the scanner  ->  right of the scanner)", fill="#777",
                       font=("TkDefaultFont", 8))
        wall = [c for i in range(N_RAYS) for c in (xs(i), ys(sc.wall_range[i]))]
        thr = [c for i in range(N_RAYS) for c in (xs(i), ys(max(sc.wall_range[i] - NEW_OBJECT_MARGIN, 0)))]
        pv.create_line(*wall, fill="#3b82f6", width=2)
        pv.create_line(*thr, fill="#93c5fd", dash=(4, 3))
        for i in range(N_RAYS):
            col = "#e53935" if mask[i] else "#9aa0a6"
            pv.create_oval(xs(i) - 3, ys(ranges[i]) - 3, xs(i) + 3, ys(ranges[i]) + 3, fill=col, outline="")

    def update_info(self, ranges, mask, est):
        hit = est is not None
        self.badge.configure(text="CUSTOMER DETECTED" if hit else "CLEAR - nobody in view", bg="#c62828" if hit else "#2e8b3d")
        lines = [f"rays hitting something new: {int(mask.sum()):2d}/{N_RAYS}"]
        if hit:
            d = float(np.linalg.norm(est - SCAN_ORIGIN))
            lines += ["", "estimated position",
                      f"  x = {est[0]:6.2f} m   y = {est[1]:6.2f} m",
                      f"  distance from scanner  {d:5.2f} m"]
            if self.present:
                err = float(np.linalg.norm(est - self.person))
                lines += ["", "real position",
                          f"  x = {self.person[0]:6.2f} m   y = {self.person[1]:6.2f} m",
                          f"  estimation error     {err * 100:5.1f} cm"]
        else:
            lines += ["", "no ray is shorter than its wall,", "so nothing new is in view."]
            if self.present:
                lines += ["", "(the customer is outside the", " scanner's view or behind the bar)"]
        self.info.configure(state="normal")
        self.info.delete("1.0", "end")
        self.info.insert("1.0", "\n".join(lines))
        self.info.configure(state="disabled")
        self.btn_walk.configure(text="Stop auto walk" if self.auto else "Auto walk")
        self.btn_present.configure(text="Remove customer" if self.present else "Put customer back")

    def run(self):
        self.root.mainloop()


def main():
    ap = argparse.ArgumentParser(description="Laser scanner demo (safety sensor of the café)")
    ap.add_argument("--selftest", action="store_true", help="no window: print the detection accuracy at a few spots")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    try:
        demo = Demo()
    except Exception as exc:                                  # no tkinter / no display
        sys.exit(f"could not open the window ({exc}). Try  python scanner_demo.py --selftest")
    demo.run()


if __name__ == "__main__":
    main()