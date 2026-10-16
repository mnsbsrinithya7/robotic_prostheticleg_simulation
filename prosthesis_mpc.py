"""
Model Predictive Control of a Two-Link Robotic Prosthetic Leg (Lagrangian dynamics)
==================================================================================
Based on: Ma, Zhang, Xu, "Robotic Leg Prosthesis: A Survey From Dynamic Model to
Adaptive Control for Gait Coordination", IEEE TNSRE vol. 32, 2024.

What is implemented (paper -> code)
-----------------------------------
 Eq.(3)-(8)  Lagrangian model   M(q)q'' + C(q,q')q' + N(q) = u + J^T Fc      -> class Leg
 Eq.(7)      spring-damper GRF  Fc = k*delta + c*delta'                      -> Environment.contact
 Eq.(9)      stochastic model   Wiener-process disturbance + random GRF      -> Environment
 Eq.(10)     state constraints  q_inf <= q <= q_sup (soft in the MPC)        -> MPC (slack variables)
 Eq.(13)     gait coordination  e(t) = q_p(t) - q_L(t - Gamma),  Gamma=Tg/2  -> HealthyGait + metrics

Additions in this version
-------------------------
 * Stairs (up / down): moving-hip Lagrangian model. The hip (pelvis) follows a prescribed world
   trajectory p_h(t) that climbs/descends one stair per half gait cycle. Lagrange's equations with a
   prescribed base give one extra generalised force:
        M(q)q'' + C(q,q')q' + N(q) + B(q) p_h''(t) = u + J^T Fc,
        B(q) = m1*Jc1(q)^T + m2*Jc2(q)^T      (Jci = Jacobian of link-i centre of mass)
   (the base velocity terms cancel exactly; only the base acceleration enters).
   The stance phase of the stair gait is built by inverse kinematics so the foot stays on its tread
   while the pelvis moves; the swing phase clears the stair nosings. Steps: riser 17 cm, tread 28 cm.
   The staircase is a solid contact surface (treads + risers), so a low swing foot hits the riser.
 * Level walking / jogging / running: road obstacles (stones = raised bumps, cracks = narrow
   depressions) placed along the path. The foot's world position x_w = v*t + x_foot uses the
   kinematically consistent ground speed v (stance foot stationary on the road).
 * MPC with vs without the disturbance estimator: logged estimate d_hat, true lumped disturbance,
   per-cycle errors and a dedicated comparison (separate + comparison figures and metrics).

Model: planar two-link leg (q1 = hip/thigh angle from vertical, q2 = knee angle relative to thigh,
q2 <= 0 means knee flexion). Both joints are torque-actuated by the prosthesis.
The "healthy limb" gait is a synthetic periodic reference; the prosthesis must track it with a
half-cycle phase lag.

Controllers compared: MPC (LTV, constrained, with / without disturbance estimator) vs Computed-Torque
PD vs PD + gravity compensation. The controllers only know the NOMINAL model; the simulated plant has
parameter mismatch, joint friction, terrain contact and random torque noise.

Run:  python prosthesis_mpc.py                         (legacy level gait, all controllers, ./outputs)
      python prosthesis_mpc.py --activity stairsup     (stairs with the moving-hip model)
      python prosthesis_mpc.py --activity walking --obstacles
      python prosthesis_mpc.py --help
"""
import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize  # noqa: F401  (kept for users who extend the solver)

GRAV = 9.81
DEG = 180.0 / np.pi

# Activity presets (shared with the Streamlit app). f = gait frequency [Hz].
# Level activities use the parametric HealthyGait (h0, hA, kc, kp); stairs use StairGait.
ACT = {"walking": dict(n="Walking", f=0.90, h0=0.05, hA=0.30, kc=0.15, kp=1.00),
       "jogging": dict(n="Jogging", f=1.15, h0=0.12, hA=0.40, kc=0.25, kp=1.35),
       "running": dict(n="Running", f=1.40, h0=0.15, hA=0.50, kc=0.35, kp=1.60),
       "stairsup": dict(n="Stairs Up", f=0.70, h0=0.45, hA=0.35, kc=0.20, kp=1.50, stairs=+1),
       "stairsdown": dict(n="Stairs Down", f=0.70, h0=0.10, hA=0.30, kc=0.50, kp=1.10, stairs=-1)}


# =============================================================================
# 1. LAGRANGIAN DYNAMICS  (paper Eq. 8)  + prescribed-base term for stairs
# =============================================================================
class Leg:
    """Two-link planar leg. State q=[q1,q2]; angles measured from downward vertical.
    Optional argument `a` everywhere = world acceleration of the hip [ax, ay] (moving-base model)."""

    def __init__(self, m1=7.0, m2=4.0, l1=0.45, l2=0.45, lc1=0.20, lc2=0.20, b=(0.0, 0.0)):
        self.m1, self.m2, self.l1, self.l2, self.lc1, self.lc2 = m1, m2, l1, l2, lc1, lc2
        self.I1 = m1 * l1 ** 2 / 12.0
        self.I2 = m2 * l2 ** 2 / 12.0
        self.b = np.array(b, float)           # viscous joint friction (unknown to controller)

    def M(self, q):
        c2 = np.cos(q[1])
        a = self.m2 * self.l1 * self.lc2 * c2
        m11 = self.m1 * self.lc1 ** 2 + self.m2 * (self.l1 ** 2 + self.lc2 ** 2) + 2 * a + self.I1 + self.I2
        m12 = self.m2 * self.lc2 ** 2 + a + self.I2
        m22 = self.m2 * self.lc2 ** 2 + self.I2
        return np.array([[m11, m12], [m12, m22]])

    def Cqd(self, q, qd):                     # Coriolis/centrifugal vector  C(q,qd)*qd
        h = -self.m2 * self.l1 * self.lc2 * np.sin(q[1])
        return np.array([h * (2 * qd[0] * qd[1] + qd[1] ** 2), -h * qd[0] ** 2])

    def N(self, q):                           # gravity vector
        s12 = np.sin(q[0] + q[1])
        return np.array([(self.m1 * self.lc1 + self.m2 * self.l1) * GRAV * np.sin(q[0]) + self.m2 * self.lc2 * GRAV * s12,
                         self.m2 * self.lc2 * GRAV * s12])

    def Bbase(self, q):
        """B(q) = m1*Jc1^T + m2*Jc2^T  (2x2). Generalised force of a hip acceleration a: B(q) @ a.
        Its y-column equals N(q)/g, i.e. a vertical hip acceleration acts like extra gravity."""
        c1, s1 = np.cos(q[0]), np.sin(q[0])
        c12, s12 = np.cos(q[0] + q[1]), np.sin(q[0] + q[1])
        m1, m2, l1, lc1, lc2 = self.m1, self.m2, self.l1, self.lc1, self.lc2
        return np.array([[m1 * lc1 * c1 + m2 * (l1 * c1 + lc2 * c12), m1 * lc1 * s1 + m2 * (l1 * s1 + lc2 * s12)],
                         [m2 * lc2 * c12, m2 * lc2 * s12]])

    def base_force(self, q, a):
        return np.zeros(2) if a is None else self.Bbase(q) @ np.asarray(a, float)

    # --- kinematics (used for ground contact, Eq. 6-7) ---
    def foot(self, q):
        x = self.l1 * np.sin(q[0]) + self.l2 * np.sin(q[0] + q[1])
        y = -(self.l1 * np.cos(q[0]) + self.l2 * np.cos(q[0] + q[1]))
        return np.array([x, y])

    def jac(self, q):
        c1, c12 = np.cos(q[0]), np.cos(q[0] + q[1])
        s1, s12 = np.sin(q[0]), np.sin(q[0] + q[1])
        return np.array([[self.l1 * c1 + self.l2 * c12, self.l2 * c12],
                         [self.l1 * s1 + self.l2 * s12, self.l2 * s12]])

    def inverse_dynamics(self, q, qd, qdd, a=None):
        return self.M(q) @ qdd + self.Cqd(q, qd) + self.N(q) + self.base_force(q, a)

    def accel(self, q, qd, tau, tau_ext=0.0, a=None):
        rhs = tau + tau_ext - self.Cqd(q, qd) - self.N(q) - self.b * qd - self.base_force(q, a)
        return np.linalg.solve(self.M(q), rhs)

    def f(self, x, tau, tau_ext=0.0, a=None):
        return np.concatenate([x[2:], self.accel(x[:2], x[2:], tau, tau_ext, a)])

    def rk4(self, x, tau, dt, tau_ext=0.0, a=None):
        k1 = self.f(x, tau, tau_ext, a)
        k2 = self.f(x + 0.5 * dt * k1, tau, tau_ext, a)
        k3 = self.f(x + 0.5 * dt * k2, tau, tau_ext, a)
        k4 = self.f(x + dt * k3, tau, tau_ext, a)
        return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def leg_ik(F, l1=0.45, l2=0.45):
    """Inverse kinematics of the two-link leg, knee-flexed branch (q2 <= 0)."""
    c = np.clip((F @ F - l1 * l1 - l2 * l2) / (2 * l1 * l2), -1.0, 1.0)
    q2 = -np.arccos(c)
    q1 = np.arctan2(F[0], -F[1]) - np.arctan2(l2 * np.sin(q2), l1 + l2 * np.cos(q2))
    return np.array([q1, q2])


# =============================================================================
# 2. HEALTHY-LIMB GAIT REFERENCE  and  coordination delay (paper Eq. 13)
# =============================================================================
class HealthyGait:
    """Periodic healthy hip/knee trajectories q_L(t). Prosthesis reference = q_L(t - Gamma)."""
    stairs = 0

    def __init__(self, Tg=1.1, n=2200, h0=0.05, hA=0.30, kc=0.15, kp=1.00):
        self.Tg = Tg
        ph = np.linspace(0, 1, n + 1)

        def bump(p, c, w):
            d = (p - c + 0.5) % 1.0 - 0.5
            return np.exp(-0.5 * (d / w) ** 2)

        hip = h0 + hA * np.cos(2 * np.pi * (ph - 0.03))
        knee = 0.02 + kc * bump(ph, 0.15, 0.06) + kp * bump(ph, 0.72, 0.10)
        Y = np.stack([hip, -knee], axis=1)      # q2 <= 0  : knee flexion
        Y[-1] = Y[0]
        self.spl = CubicSpline(ph * Tg, Y, bc_type="periodic")

    def ref(self, t, shift=0.0):
        """returns q, qd, qdd of healthy limb evaluated at (t - shift)."""
        te = (t - shift) % self.Tg
        return self.spl(te), self.spl(te, 1), self.spl(te, 2)


class StairGait:
    """
    Stair ascent (direction=+1) / descent (direction=-1) reference, same interface as HealthyGait.

    Pelvis (hip) world trajectory, as a function of the gait phase p of the leg that owns this gait:
        p_h(p) = v*p + [Ax sin(4*pi*p), Az sin(4*pi*p)],   v = (2*tread, +-2*riser) per cycle
    (one stair per half cycle, vertical "bounce" of the pelvis at twice the gait frequency).
    Stance (p in [0, stance]): joint angles from inverse kinematics of  foot = W - p_h(p),
    i.e. the foot is fixed on its tread while the pelvis climbs/descends.
    Swing: knot points (hip, knee-flexion in deg) shaped to typical stair-gait ranges and tuned so the
    ankle passes >= 7 cm from every stair nosing. Everything is joined by one periodic cubic spline.
    NOTE: this is a synthetic, kinematically consistent template, not measured gait data.
    """
    TEMPLATE = {
        +1: dict(ic=(49.0, 43.0),
                 swing=[(0.50, 16, 16), (0.58, 23, 51), (0.66, 42, 90), (0.74, 50, 94),
                        (0.82, 68, 88), (0.90, 66, 65), (0.95, 54, 48)],
                 toe_room=0.17),      # ankle -> next riser [m] (foot sits behind the riser)
        -1: dict(ic=(16.0, 10.0),
                 swing=[(0.50, 31, 81), (0.58, 35, 95), (0.66, 46, 95), (0.74, 46, 52),
                        (0.82, 30, 23), (0.90, 23, 12), (0.95, 19, 9)],
                 toe_room=0.14)}      # ankle -> stair edge [m]

    def __init__(self, Tg=1.4, direction=+1, riser=0.17, tread=0.28, bounce=0.015, surge=0.008,
                 stance=0.42, n_stance=22, l1=0.45, l2=0.45, offset=0.004):
        self.Tg, self.stairs, self.riser, self.tread = Tg, int(direction), riser, tread
        self.l1, self.l2, self.stance, self.offset = l1, l2, stance, offset
        tpl = self.TEMPLATE[self.stairs]
        self.toe_room = tpl["toe_room"]
        self.v = np.array([2 * tread, direction * 2 * riser])      # pelvis displacement per cycle
        self.amp = np.array([surge, bounce])
        h, k = np.radians(tpl["ic"])
        self.W = self.pelvis_phase(0.0)[0] + self._fk(np.array([h, -k]))   # foothold of cycle 0
        ph = np.linspace(0.0, stance, n_stance)
        Q = [leg_ik(self.W - self.pelvis_phase(p)[0], l1, l2) for p in ph]
        kn = [(p, q[0], q[1]) for p, q in zip(ph, Q)] + \
             [(p, np.radians(hh), -np.radians(kk)) for p, hh, kk in tpl["swing"]]
        kn = np.array(kn)
        P = np.r_[kn[:, 0], 1.0]
        Y = np.r_[kn[:, 1:], kn[:1, 1:]]
        self.spl = CubicSpline(P * Tg, Y, bc_type="periodic")

    def _fk(self, q):
        return np.array([self.l1 * np.sin(q[0]) + self.l2 * np.sin(q[0] + q[1]),
                         -(self.l1 * np.cos(q[0]) + self.l2 * np.cos(q[0] + q[1]))])

    def pelvis_phase(self, p):
        """pelvis position, d/dp, d2/dp2 at (unwrapped) phase p."""
        w = 4 * np.pi
        s, c = np.sin(w * p), np.cos(w * p)
        return self.v * p + self.amp * s, self.v + self.amp * w * c, -self.amp * w * w * s

    def pelvis(self, t, shift=0.0):
        """world hip position, velocity, acceleration at time t (prosthesis phase = (t-shift)/Tg)."""
        p0, p1, p2 = self.pelvis_phase((t - shift) / self.Tg)
        return p0, p1 / self.Tg, p2 / self.Tg ** 2

    def ref(self, t, shift=0.0):
        te = (t - shift) % self.Tg
        return self.spl(te), self.spl(te, 1), self.spl(te, 2)

    # ---- staircase geometry (contact surface for the ankle point) ----
    def tread_index(self, x):
        return int(np.floor((x - self.W[0] - self.toe_room) / self.tread)) + 1

    def tread_box(self, j):
        """(x0, x1, ytop) of tread j; prosthesis stands on even j, healthy limb on odd j."""
        x1 = self.W[0] + j * self.tread + self.toe_room
        return x1 - self.tread, x1, self.W[1] + j * self.stairs * self.riser + self.offset


# =============================================================================
# 3. ENVIRONMENT: terrain (flat road + obstacles, or stairs), random torque (paper Eq. 7, 9)
# =============================================================================
def level_ground_speed(plant, gait, shift, band=0.015, n=600):
    """Kinematically consistent ground speed: mean backward speed of the reference foot relative to
    the hip over the contiguous stance segment where the foot is within `band` of its lowest point.
    On a road moving at this speed the stance foot does not skate over obstacles."""
    ts = np.linspace(0, gait.Tg, n, endpoint=False)
    F = np.array([plant.foot(gait.ref(t, shift)[0]) for t in ts])
    lim = F[:, 1].min() + band
    i = int(np.argmin(F[:, 1]))
    a = b = i
    while F[(a - 1) % n, 1] < lim and b - a < n - 1:
        a -= 1
    while F[(b + 1) % n, 1] < lim and b - a < n - 1:
        b += 1
    if b == a:
        return 0.0
    dt = (b - a) * gait.Tg / n
    return float(max(0.0, -(F[b % n, 0] - F[a % n, 0]) / dt))


class Environment:
    def __init__(self, plant, gait, shift, T, dt, seed=0, uneven=0.002, torque_noise=0.3,
                 contact=True, k=2.0e4, c=100.0, ct=50.0, obstacles=False, obstacle_density=4.0,
                 placement="path"):
        rng = np.random.default_rng(seed)
        n = int(round(T / dt)) + 2
        self.dt, self.contact, self.k, self.c, self.ct = dt, contact, k, c, ct
        self.gait, self.shift, self.T = gait, shift, T
        self.stairs = int(getattr(gait, "stairs", 0))
        if self.stairs:
            self.ground0 = gait.tread_box(0)[2]                   # first prosthesis tread (contact level)
            self.speed = float(np.hypot(*gait.v) / gait.Tg)
        else:
            # ground level: just above the lowest foot point of the reference cycle
            ts = np.linspace(0, gait.Tg, 400)
            ymin = min(plant.foot(gait.ref(t, shift)[0])[1] for t in ts)
            self.ground0 = ymin + 0.004
            self.speed = level_ground_speed(plant, gait, shift)
        # Ornstein-Uhlenbeck ground roughness (stochastically uneven terrain)
        tau_c = 0.2
        a = np.exp(-dt / tau_c)
        r = np.zeros(n)
        for i in range(1, n):
            r[i] = a * r[i - 1] + uneven * np.sqrt(1 - a * a) * rng.standard_normal()
        self.rough = r
        # white torque disturbance (discretised Wiener increment dW/dt)
        self.tau_noise = torque_noise * rng.standard_normal((n, 2)) / np.sqrt(dt)
        # road obstacles: separate RNG so that roughness/noise stay identical with obstacles on/off
        self.obstacles = []
        if obstacles and not self.stairs and obstacle_density > 0:
            orng = np.random.default_rng(seed + 7919)

            def make(x):
                if orng.random() < 0.6:     # stone: raised bump, 2-4.5 cm high
                    return dict(k="stone", x=float(x), w=float(orng.uniform(0.08, 0.18)),
                                h=float(orng.uniform(0.02, 0.045)))
                return dict(k="crack", x=float(x), w=float(orng.uniform(0.03, 0.08)),   # crack: depression
                            h=-float(orng.uniform(0.012, 0.025)))
            if placement == "path":
                # perturbation course: obstacles at upcoming touch-downs of the reference foot path
                tt = np.arange(0.0, T, dt)
                F = np.array([plant.foot(gait.ref(t, shift)[0]) for t in tt])
                y = F[:, 1]
                mins = np.where((y[1:-1] <= y[:-2]) & (y[1:-1] < y[2:]) & (y[1:-1] < self.ground0 + 0.01))[0] + 1
                p_hit = min(1.0, obstacle_density * self.speed * gait.Tg / 10.0)
                for i in mins:
                    if tt[i] < 0.5 * gait.Tg or orng.random() > p_hit:
                        continue
                    self.obstacles.append(make(self.speed * tt[i] + F[i, 0] + orng.uniform(-0.10, 0.06)))
            else:
                x, x_end = 1.0, self.speed * T + 2.0
                while True:
                    x += orng.exponential(10.0 / obstacle_density) + 0.25
                    if x > x_end:
                        break
                    self.obstacles.append(make(x))
            self.obstacles.sort(key=lambda o: o["x"])
        self._ox = np.array([o["x"] for o in self.obstacles])
        self.hits = 0                      # substeps with foot contact on an obstacle / stair riser
        self._special = False

    # ---- hip (pelvis) world motion ----
    def hip(self, t):
        if self.stairs:
            return self.gait.pelvis(t, self.shift)
        return np.array([self.speed * t, 0.0]), np.array([self.speed, 0.0]), np.zeros(2)

    def base_acc(self, t):
        """hip acceleration used by the moving-base Lagrangian model (None = fixed hip)."""
        return self.gait.pelvis(t, self.shift)[2] if self.stairs else None

    # ---- obstacle profile (raised-cosine bumps / depressions) ----
    def obstacle_height(self, x):
        if not self.obstacles:
            return 0.0, 0.0
        j = int(np.argmin(np.abs(self._ox - x)))
        o = self.obstacles[j]
        d = x - o["x"]
        if abs(d) >= o["w"] / 2:
            return 0.0, 0.0
        w = 2 * np.pi / o["w"]
        return 0.5 * o["h"] * (1 + np.cos(w * d)), -0.5 * o["h"] * w * np.sin(w * d)

    # ---- contact: penetration depth, outward normal, tangent ----
    def _penetration(self, x, y, i):
        r = self.rough[min(i, len(self.rough) - 1)]
        if self.stairs:
            g = self.gait
            j = g.tread_index(x)
            x0, x1, ytop = g.tread_box(j)
            ytop += r
            dv = ytop - y
            if dv <= 0:
                return 0.0, None
            best, n = dv, np.array([0.0, 1.0])
            if g.tread_box(j - 1)[2] + r < y and x - x0 < best:        # exit through the left riser face
                best, n = x - x0, np.array([-1.0, 0.0])
            if g.tread_box(j + 1)[2] + r < y and x1 - x < best:        # exit through the right riser face
                best, n = x1 - x, np.array([1.0, 0.0])
            return best, n
        h, s = self.obstacle_height(x)
        pen = self.ground0 + r + h - y
        if pen <= 0:
            return 0.0, None
        nrm = np.sqrt(1 + s * s)
        return pen / nrm, np.array([-s, 1.0]) / nrm

    def contact_force(self, plant, x, i):
        """returns (joint torque J^T Fc, world contact force Fc)."""
        if not self.contact:
            return np.zeros(2), np.zeros(2)
        q, qd = x[:2], x[2:]
        p = plant.foot(q)
        J = plant.jac(q)
        v = J @ qd                                        # foot velocity relative to the hip
        t = i * self.dt
        hp, hv, _ = self.hip(t)
        delta, n = self._penetration(hp[0] + p[0], hp[1] + p[1], i)
        if n is None:
            return np.zeros(2), np.zeros(2)
        self._special = (abs(n[0]) > 0.5) if self.stairs else (self.obstacle_height(hp[0] + p[0])[0] != 0.0)
        vw = v + hv                                       # foot velocity relative to the (static) ground
        if self.stairs:
            tvec = np.array([n[1], -n[0]])
            Fn = max(0.0, self.k * delta + self.c * (-(vw @ n)))   # Fc = k*delta + c*delta'   (Eq. 7)
            F = Fn * n - self.ct * (vw @ tvec) * tvec
        else:
            # level road: same law as the original model (tangential damping in the hip frame);
            # on an obstacle the normal follows the surface slope
            if n[0] == 0.0:
                F = np.array([-self.ct * v[0], max(0.0, self.k * delta + self.c * (-v[1]))])
            else:
                tvec = np.array([n[1], -n[0]])
                Fn = max(0.0, self.k * delta + self.c * (-(vw @ n)))
                F = Fn * n - self.ct * (v @ tvec) * tvec
        return J.T @ F, F                                 # Fe = J^T Fc               (Eq. 6)

    def contact_torque(self, plant, x, i):
        return self.contact_force(plant, x, i)[0]

    def disturbance(self, plant, x, i):
        self._special = False
        tau = self.contact_torque(plant, x, i)
        self.hits += int(self._special)
        return tau + self.tau_noise[i]

    # ---- geometry for plots / animations ----
    def terrain_polyline(self, x_from, x_to, step=0.01):
        """world (x, y) polyline of the contact surface (without the mm-scale random roughness)."""
        if self.stairs:
            g = self.gait
            pts = []
            for j in range(g.tread_index(x_from) - 1, g.tread_index(x_to) + 2):
                x0, x1, y = g.tread_box(j)
                pts += [(x0, y), (x1, y)]
            return np.array(pts)
        if not self.obstacles:
            return np.array([(x_from, self.ground0), (x_to, self.ground0)])
        pts = [(x_from, self.ground0)]
        for o in self.obstacles:
            if o["x"] + o["w"] < x_from or o["x"] - o["w"] > x_to:
                continue
            for xx in np.linspace(o["x"] - o["w"] / 2, o["x"] + o["w"] / 2, 13):
                pts.append((xx, self.ground0 + self.obstacle_height(xx)[0]))
        pts.append((x_to, self.ground0))
        return np.array(pts)

    def viz(self, hip_x):
        """compact terrain description for the HTML animations (world coordinates)."""
        x0, x1 = float(np.min(hip_x)) - 3.0, float(np.max(hip_x)) + 3.0
        d = dict(kind="stairs" if self.stairs else "road", ground0=float(self.ground0), speed=self.speed,
                 poly=np.round(self.terrain_polyline(x0, x1), 4).tolist())
        if self.stairs:
            g = self.gait
            d.update(riser=g.riser, tread=g.tread, dir=g.stairs, W=[float(g.W[0]), float(g.W[1])],
                     treads=[[round(v, 4) for v in g.tread_box(j)]
                             for j in range(g.tread_index(x0) - 1, g.tread_index(x1) + 2)])
        else:
            d["obstacles"] = self.obstacles
        return d


# =============================================================================
# 4. CONTROLLERS   (base = optional callable t -> hip acceleration, known from the stair plan)
# =============================================================================
class PDGravity:
    name = "PD + gravity comp."

    def __init__(self, nominal, gait, shift, umax, Kp=(120.0, 60.0), Kd=(12.0, 3.5), base=None):
        self.nom, self.gait, self.shift, self.umax = nominal, gait, shift, umax
        self.Kp, self.Kd = np.array(Kp), np.array(Kd)
        self.base = base

    def __call__(self, t, x):
        qr, qdr, _ = self.gait.ref(t, self.shift)
        a = self.base(t) if self.base else None
        u = self.nom.N(x[:2]) + self.nom.base_force(x[:2], a) + self.Kp * (qr - x[:2]) + self.Kd * (qdr - x[2:])
        return np.clip(u, -self.umax, self.umax)


class ComputedTorque:
    name = "Computed-torque PD"

    def __init__(self, nominal, gait, shift, umax, wn=(25.0, 25.0), zeta=1.0, base=None):
        self.nom, self.gait, self.shift, self.umax = nominal, gait, shift, umax
        self.Kp = np.array(wn) ** 2
        self.Kd = 2 * zeta * np.array(wn)
        self.base = base

    def __call__(self, t, x):
        qr, qdr, qddr = self.gait.ref(t, self.shift)
        e, ed = qr - x[:2], qdr - x[2:]
        acc = qddr + self.Kd * ed + self.Kp * e
        u = self.nom.inverse_dynamics(x[:2], x[2:], acc, self.base(t) if self.base else None)
        return np.clip(u, -self.umax, self.umax)


class MPC:
    """
    Linear time-varying MPC around the gait reference.
      error state  e = x - x_ref(t),  x=[q1,q2,qd1,qd2]
      model        e_{k+1} = A_k e_k + B_k du_k + c_k      (RK4-discretised nominal Lagrangian model,
                                                            linearised by finite differences around
                                                            (x_ref, u_ff); on stairs it includes the
                                                            planned hip acceleration B(q) p_h'')
      input        u = u_ff + du,  u_ff = nominal inverse dynamics of the reference
      cost         sum e'Qe + du'R du + e_N'Qf e_N + rho*(bound violation)^2
      constraints  |u| <= umax (hard),  q_inf <= q <= q_sup (soft penalty, paper Eq. 10)
    The condensed QP (box-constrained, soft state constraints) is solved with a projected Newton method (numpy only).
      An optional disturbance estimator (offset-free MPC) compensates unmodelled torques.
    """
    name = "MPC"

    def __init__(self, nominal, gait, shift, umax, qmin, qmax, N=15, Ts=0.02,
                 Q=(2000.0, 2000.0, 20.0, 20.0), R=(0.002, 0.002), Qf_scale=5.0, rho=1e5, dist_est=True, alpha=0.4,
                 base=None):
        self.nom, self.gait, self.shift = nominal, gait, shift
        self.umax, self.qmin, self.qmax = np.asarray(umax, float), np.asarray(qmin, float), np.asarray(qmax, float)
        self.N, self.Ts, self.rho = N, Ts, rho
        self.Q, self.R = np.diag(Q), np.diag(R)
        self.Qf = Qf_scale * self.Q
        self.nx, self.nu = 4, 2
        self._z_prev = None
        self.dist_est, self.alpha = dist_est, alpha
        self.dhat = np.zeros(2)                   # estimated lumped torque disturbance [Nm]
        self._x_prev = self._u_prev = self._a_prev = None
        self.solve_times, self.fail_count = [], 0
        self.base = base
        if not dist_est:
            self.name = "MPC (no estimator)"

        # constant parts of the QP
        N_, nx, nu = N, self.nx, self.nu
        self.Qb = np.zeros((N_ * nx, N_ * nx))
        for k in range(N_):
            self.Qb[k * nx:(k + 1) * nx, k * nx:(k + 1) * nx] = self.Qf if k == N_ - 1 else self.Q
        self.Rb = np.kron(np.eye(N_), self.R)
        self.P = np.zeros((N_ * 2, N_ * nx))          # selects joint positions from stacked error
        for k in range(N_):
            self.P[k * 2:(k + 1) * 2, k * nx:k * nx + 2] = np.eye(2)

    # discrete nominal model with zero-order hold over Ts
    def F(self, x, u, a=None):
        xn, sub = x, 2
        for _ in range(sub):
            xn = self.nom.rk4(xn, u + self.dhat, self.Ts / sub, a=a)    # u + estimated disturbance
        return xn

    def update_disturbance(self, x):
        """Offset-free MPC: estimate unmodelled torque (contact, friction, mismatch) from the
        one-step prediction residual and low-pass it (alpha = filter gain)."""
        if not self.dist_est or self._x_prev is None:
            return
        pred = self.F(self._x_prev, self._u_prev, self._a_prev)
        r = x[2:] - pred[2:]                                       # velocity residual
        self.dhat = np.clip(self.dhat + self.alpha * (self.nom.M(x[:2]) @ r) / self.Ts, -40.0, 40.0)

    def linearize(self, xb, ub, a=None, eps=1e-6):
        A = np.zeros((4, 4))
        B = np.zeros((4, 2))
        f0 = self.F(xb, ub, a)
        for i in range(4):
            d = np.zeros(4)
            d[i] = eps
            A[:, i] = (self.F(xb + d, ub, a) - self.F(xb - d, ub, a)) / (2 * eps)
        for j in range(2):
            d = np.zeros(2)
            d[j] = eps
            B[:, j] = (self.F(xb, ub + d, a) - self.F(xb, ub - d, a)) / (2 * eps)
        return A, B, f0

    def reference_x(self, t):
        q, qd, qdd = self.gait.ref(t, self.shift)
        a = self.base(t) if self.base else None
        return np.concatenate([q, qd]), self.nom.inverse_dynamics(q, qd, qdd, a)

    @staticmethod
    def _projected_newton(fun, H, PSu, lo, hi, rho, z, zl, zu, iters=25, tol=1e-8):
        """Projected Newton for  min f(z)  s.t. zl<=z<=zu  with piecewise-quadratic f
        (QP + hinge-squared soft state constraints). Converges in a handful of iterations."""
        f, gr = fun(z)
        for _ in range(iters):
            at_lo, at_hi = (z <= zl + 1e-10) & (gr > 0), (z >= zu - 1e-10) & (gr < 0)
            free = ~(at_lo | at_hi)
            if np.linalg.norm(gr[free]) < tol * max(1.0, abs(f)):
                return z, True
            p = PSu @ z
            act = ((p < lo) | (p > hi)).astype(float)
            Hs = H + rho * (PSu.T * act) @ PSu
            d = np.zeros_like(z)
            d[free] = -np.linalg.solve(Hs[np.ix_(free, free)] + 1e-9 * np.eye(free.sum()), gr[free])
            step, ok = 1.0, False
            for _ in range(30):                                  # backtracking line search
                zn = np.clip(z + step * d, zl, zu)
                fn, gn = fun(zn)
                if fn <= f + 1e-4 * gr @ (zn - z):
                    ok = True
                    break
                step *= 0.5
            if not ok:
                return z, False
            z, f, gr = zn, fn, gn
        return z, False

    def __call__(self, t, x):
        N, nx, nu = self.N, self.nx, self.nu
        self.update_disturbance(x)
        # ---- build prediction model along the reference over the horizon
        xb, ub, ab = [], [], []
        for k in range(N + 1):
            xr, ur = self.reference_x(t + k * self.Ts)
            xb.append(xr)
            ub.append(np.clip(ur - self.dhat, -self.umax, self.umax))   # feedforward minus estimated disturbance
            ab.append(self.base(t + k * self.Ts) if self.base else None)
        A, B, c = [], [], []
        for k in range(N):
            Ak, Bk, fk = self.linearize(xb[k], ub[k], ab[k])
            A.append(Ak); B.append(Bk); c.append(fk - xb[k + 1])

        Sx = np.zeros((N * nx, nx)); Su = np.zeros((N * nx, N * nu)); Sc = np.zeros(N * nx)
        pSx, pSu, pSc = np.eye(nx), np.zeros((nx, N * nu)), np.zeros(nx)
        for k in range(N):
            cSx = A[k] @ pSx
            cSu = A[k] @ pSu
            cSu[:, k * nu:(k + 1) * nu] += B[k]
            cSc = A[k] @ pSc + c[k]
            Sx[k * nx:(k + 1) * nx] = cSx; Su[k * nx:(k + 1) * nx] = cSu; Sc[k * nx:(k + 1) * nx] = cSc
            pSx, pSu, pSc = cSx, cSu, cSc

        e0 = x - xb[0]
        free = Sx @ e0 + Sc                                     # error prediction with du = 0
        H = Su.T @ self.Qb @ Su + self.Rb
        g = Su.T @ self.Qb @ free
        # soft state constraints  qmin <= q <= qmax  (quadratic hinge penalty, paper Eq. 10)
        qb = np.concatenate([xb[k + 1][:2] for k in range(N)])
        PSu, Pfree = self.P @ Su, self.P @ free
        lo = np.tile(self.qmin, N) - qb - Pfree          # need  PSu du >= lo
        hi = np.tile(self.qmax, N) - qb - Pfree          # need  PSu du <= hi
        rho = self.rho

        def fun(du):
            p = PSu @ du
            vl, vh = np.maximum(0.0, lo - p), np.maximum(0.0, p - hi)
            val = 0.5 * du @ H @ du + g @ du + 0.5 * rho * (vl @ vl + vh @ vh)
            grad = H @ du + g + rho * (PSu.T @ (vh - vl))
            return val, grad

        ub_arr = np.concatenate([ub[k] for k in range(N)])
        du_lo = np.tile(-self.umax, N) - ub_arr          # hard torque limits |u| <= umax
        du_hi = np.tile(self.umax, N) - ub_arr
        z0 = np.zeros(N * nu) if self._z_prev is None else self._z_prev
        z0 = np.clip(z0, du_lo, du_hi)
        t0 = time.perf_counter()
        z, converged = self._projected_newton(fun, H, PSu, lo, hi, rho, z0, du_lo, du_hi)
        self.solve_times.append(time.perf_counter() - t0)
        if not converged:
            self.fail_count += 1
        # warm start: shift the input sequence by one step
        zs = z.copy()
        zs[:(N - 1) * nu] = z[nu:N * nu]
        self._z_prev = zs
        u = ub[0] + z[:nu]
        u = np.clip(u, -self.umax, self.umax)
        self._x_prev, self._u_prev, self._a_prev = x.copy(), u.copy(), ab[0]
        return u


# =============================================================================
# 5. SIMULATION  &  METRICS
# =============================================================================
def simulate(controller, plant, env, gait, shift, T, Ts, dt, nominal=None):
    """Closed-loop simulation. If `nominal` is given, the true lumped disturbance as seen by the
    nominal model, d_true = ID_nominal(q, qd, qdd_plant) - u  (contact + friction + mass error,
    without the white torque noise), is logged for comparison with the MPC estimate d_hat."""
    steps = int(round(T / Ts))
    sub = int(round(Ts / dt))
    q0, qd0, _ = gait.ref(0.0, shift)
    x = np.concatenate([q0 + np.array([0.04, -0.04]), qd0])      # small initial offset
    log = {k: [] for k in ("t", "q", "qd", "qref", "u", "dist", "fc", "hip", "dhat", "dtrue", "obst")}
    i = 0
    for s in range(steps):
        t = s * Ts
        u = controller(t, x)
        qr, _, _ = gait.ref(t, shift)
        tau_c, Fc = env.contact_force(plant, x, i)
        a0 = env.base_acc(t)
        log["t"].append(t); log["q"].append(x[:2].copy()); log["qd"].append(x[2:].copy())
        log["qref"].append(qr.copy()); log["u"].append(u.copy())
        log["dist"].append(tau_c.copy()); log["fc"].append(Fc.copy())
        log["hip"].append(env.hip(t)[0].copy())
        log["dhat"].append(np.array(getattr(controller, "dhat", np.zeros(2)), float).copy())
        if nominal is not None:
            qdd = plant.accel(x[:2], x[2:], u, tau_c, a0)
            log["dtrue"].append(nominal.inverse_dynamics(x[:2], x[2:], qdd, a0) - u)
        else:
            log["dtrue"].append(np.full(2, np.nan))
        h0 = env.hits
        for _ in range(sub):
            x = plant.rk4(x, u, dt, env.disturbance(plant, x, i), a=env.base_acc(i * dt))
            i += 1
        log["obst"].append((env.hits - h0) / sub)
    return {k: np.array(v) for k, v in log.items()}


def metrics(log, Tg, qmin, qmax, umax):
    m = log["t"] > Tg                                           # discard first cycle (transient)
    e = log["q"][m] - log["qref"][m]
    rmse = np.sqrt(np.mean(e ** 2, axis=0)) * DEG
    coord = np.linalg.norm(e, axis=1) * DEG                      # ||q_p(t) - q_L(t-Gamma)||  (Eq. 13)
    u = log["u"][m]
    power = np.maximum(0, np.sum(u * log["qd"][m], axis=1))
    viol = np.mean(np.any((log["q"][m] < qmin) | (log["q"][m] > qmax), axis=1)) * 100
    out = {"RMSE hip [deg]": rmse[0], "RMSE knee [deg]": rmse[1],
           "Coord. error mean [deg]": coord.mean(), "Coord. error max [deg]": coord.max(),
           "Torque RMS hip [Nm]": np.sqrt(np.mean(u[:, 0] ** 2)),
           "Torque RMS knee [Nm]": np.sqrt(np.mean(u[:, 1] ** 2)),
           "Mean positive power [W]": power.mean(),
           "Sat. time [%]": np.mean(np.any(np.abs(u) >= umax - 1e-6, axis=1)) * 100,
           "State-bound violation [%]": viol}
    if "fc" in log:
        out["Peak contact force [N]"] = float(np.max(np.linalg.norm(log["fc"][m], axis=1)))
    if "obst" in log:
        out["Obstacle/riser contact [%]"] = float(np.mean(log["obst"][m]) * 100)
    return out


def per_cycle_rmse(log, Tg):
    """coordination-error RMSE [deg] of every complete gait cycle (cycle 1 = first one)."""
    t = log["t"]
    err = np.linalg.norm(log["q"] - log["qref"], axis=1) * DEG
    n = int(np.floor(t[-1] / Tg + 1e-9))
    return np.array([np.sqrt(np.mean(err[(t >= k * Tg) & (t < (k + 1) * Tg)] ** 2)) for k in range(n)
                     if np.any((t >= k * Tg) & (t < (k + 1) * Tg))])


def lowpass(x, Ts, tau=0.1):
    a = Ts / (tau + Ts)
    y = np.zeros_like(x)
    for i in range(len(x)):
        y[i] = x[i] if i == 0 else y[i - 1] + a * (x[i] - y[i - 1])
    return y


def estimator_metrics(log, Tg, Ts, solve_times=None):
    """extra metrics for the estimator study (computed after the first gait cycle)."""
    m = log["t"] > Tg
    e = (log["q"][m] - log["qref"][m]) * DEG
    coord = np.linalg.norm(e, axis=1)
    cyc = per_cycle_rmse(log, Tg)
    du = np.diff(log["u"][m], axis=0) / Ts
    out = {"Abs. mean bias hip [deg]": float(abs(np.mean(e[:, 0]))),
           "Abs. mean bias knee [deg]": float(abs(np.mean(e[:, 1]))),
           "Coord. error P95 [deg]": float(np.percentile(coord, 95)),
           "First-cycle RMSE [deg]": float(cyc[0]) if len(cyc) else np.nan,
           "Last-cycle RMSE [deg]": float(cyc[-1]) if len(cyc) else np.nan,
           "Torque rate RMS [Nm/s]": float(np.sqrt(np.mean(du ** 2))) if len(du) else np.nan}
    if np.all(np.isfinite(log["dtrue"])):
        dref = lowpass(log["dtrue"], Ts)[m]
        dh = log["dhat"][m]
        out["Disturbance RMS hip [Nm]"] = float(np.sqrt(np.mean(dref[:, 0] ** 2)))
        out["Disturbance RMS knee [Nm]"] = float(np.sqrt(np.mean(dref[:, 1] ** 2)))
        out["Unexplained disturbance RMS [Nm]"] = float(np.sqrt(np.mean(np.sum((dref - dh) ** 2, axis=1))))
    if solve_times is not None and len(solve_times):
        st = np.asarray(solve_times) * 1000
        out["Solve time mean [ms]"] = float(st.mean())
        out["Solve time max [ms]"] = float(st.max())
    return out


def estimator_comparison_table(row_with, row_without):
    """pandas table: metric x [with, without, difference, change %]. Negative change = estimator better
    for error/effort metrics."""
    import pandas as pd
    keys = [k for k in row_with if k in row_without]
    rows = []
    for k in keys:
        a, b = float(row_with[k]), float(row_without[k])
        rows.append((k, a, b, a - b, (a - b) / abs(b) * 100 if abs(b) > 1e-9 else np.nan))
    return pd.DataFrame(rows, columns=["Metric", "MPC + estimator", "MPC (no estimator)",
                                       "Difference", "Change [%]"]).set_index("Metric")


# =============================================================================
# 6. SCENARIO BUILDER (used by the CLI and the Streamlit app)
# =============================================================================
def make_gait(activity=None, gait_period=1.1):
    if activity is None:
        return HealthyGait(Tg=gait_period)
    ac = ACT[activity]
    Tg = 1.0 / ac["f"]
    if ac.get("stairs"):
        return StairGait(Tg=Tg, direction=ac["stairs"])
    return HealthyGait(Tg=Tg, h0=ac["h0"], hA=ac["hA"], kc=ac["kc"], kp=ac["kp"])


def state_limits(activity, gait):
    """q_inf / q_sup (Eq. 10). Level activities keep the original formula; stairs use the
    reference range with the same margins."""
    if activity is None:
        return np.array([-0.60, -1.45]), np.array([0.80, 0.03])
    ac = ACT[activity]
    if ac.get("stairs"):
        y = gait.spl(np.linspace(0, gait.Tg, 400))
        return (np.array([y[:, 0].min() - 0.2, y[:, 1].min() - 0.35]),
                np.array([y[:, 0].max() + 0.25, 0.03]))
    return (np.array([ac["h0"] - ac["hA"] - 0.2, -(ac["kp"] + 0.35)]),
            np.array([ac["h0"] + ac["hA"] + 0.25, 0.03]))


def run_scenario(activity="walking", duration=6.0, horizon=15, delay_ratio=0.5, mismatch=0.25, friction=1.5,
                 noise=0.3, terrain=0.002, contact=True, hip_lim=150.0, knee_lim=120.0, seed=1,
                 controllers=("pd", "ctc", "mpc", "mpc_noest"), obstacles=False, obstacle_density=4.0,
                 gait_period=1.1, verbose=False, placement="path"):
    """Builds plant/gait/environment, runs the chosen controllers on identical noise/terrain and
    returns (logs, metrics dict, extra estimator metrics dict, healthy-limb angles, info dict)."""
    Ts, dt = 0.02, 0.005
    gait = make_gait(activity, gait_period)
    Tg = gait.Tg
    shift = delay_ratio * Tg
    nominal = Leg()
    plant = Leg(m1=nominal.m1 * (1 + mismatch), m2=nominal.m2 * (1 + mismatch), b=(friction, friction))
    umax = np.array([hip_lim, knee_lim], float)
    qmin, qmax = state_limits(activity, gait)
    probe = Environment(plant, gait, shift, duration, dt, seed=seed, uneven=terrain, torque_noise=noise,
                        contact=contact, obstacles=obstacles, obstacle_density=obstacle_density,
                        placement=placement)
    base = probe.base_acc if probe.stairs else None
    mk = lambda est: MPC(nominal, gait, shift, umax, qmin, qmax, N=horizon, Ts=Ts, dist_est=est, base=base)
    factory = {"pd": lambda: PDGravity(nominal, gait, shift, umax, base=base),
               "ctc": lambda: ComputedTorque(nominal, gait, shift, umax, base=base),
               "mpc": lambda: mk(True), "mpc_noest": lambda: mk(False)}
    logs, rows, extra, solve = {}, {}, {}, {}
    for key in controllers:
        c = factory[key]()
        env = Environment(plant, gait, shift, duration, dt, seed=seed, uneven=terrain, torque_noise=noise,
                          contact=contact, obstacles=obstacles, obstacle_density=obstacle_density,
                          placement=placement)                                  # same for all
        t0 = time.perf_counter()
        logs[c.name] = simulate(c, plant, env, gait, shift, duration, Ts, dt, nominal=nominal)
        wall = time.perf_counter() - t0
        rows[c.name] = metrics(logs[c.name], Tg, qmin, qmax, umax)
        st = getattr(c, "solve_times", None)
        extra[c.name] = estimator_metrics(logs[c.name], Tg, Ts, st)
        if st is not None:
            solve[c.name] = dict(mean=float(np.mean(st) * 1000), max=float(np.max(st) * 1000), warn=c.fail_count)
        if verbose:
            s = solve.get(c.name)
            print(f"  [{c.name}] simulated in {wall:.1f}s" +
                  (f" | MPC solve time mean {s['mean']:.1f} ms, max {s['max']:.1f} ms, solver warnings {s['warn']}"
                   if s else ""))
    t = next(iter(logs.values()))["t"]
    healthy = np.array([gait.ref(tt, 0.0)[0] for tt in t])
    hip_x = next(iter(logs.values()))["hip"][:, 0]
    info = dict(Ts=Ts, ground=float(probe.ground0), qmin=qmin.tolist(), qmax=qmax.tolist(), Tg=Tg,
                act=activity, l1=plant.l1, l2=plant.l2, shift=shift, umax=umax.tolist(),
                solve_ms=solve.get("MPC", {}).get("mean"), solve=solve, speed=probe.speed,
                stairs=probe.stairs, terrain=probe.viz(hip_x),
                n_stones=sum(o["k"] == "stone" for o in probe.obstacles),
                n_cracks=sum(o["k"] == "crack" for o in probe.obstacles))
    return logs, rows, extra, healthy, info


# =============================================================================
# 7. PLOTS & ANIMATION
# =============================================================================
COLORS = {"MPC": "tab:red", "Computed-torque PD": "tab:blue", "PD + gravity comp.": "tab:green",
          "MPC (no estimator)": "tab:orange"}


def make_plots(logs, mets, Tg, outdir):
    names = list(logs)
    t = logs[names[0]]["t"]
    fig, ax = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    for j, jn in enumerate(["Hip (q1)", "Knee (q2)"]):
        ax[j].plot(t, logs[names[0]]["qref"][:, j] * DEG, "k--", lw=2, label="Healthy limb delayed q_L(t-Γ)")
        for n in names:
            ax[j].plot(t, logs[n]["q"][:, j] * DEG, color=COLORS.get(n), lw=1.3, label=n)
        ax[j].set_ylabel(jn + " [deg]"); ax[j].grid(alpha=.3)
    ax[0].legend(ncol=4, fontsize=8, loc="upper right")
    for n in names:
        e = np.linalg.norm(logs[n]["q"] - logs[n]["qref"], axis=1) * DEG
        ax[2].plot(t, e, color=COLORS.get(n), label=n)
        ax[3].plot(t, logs[n]["u"][:, 1], color=COLORS.get(n), label=n)
    ax[2].set_ylabel("‖e(t)‖ [deg]\n(coordination error)"); ax[2].grid(alpha=.3)
    ax[3].set_ylabel("Knee torque [Nm]"); ax[3].set_xlabel("time [s]"); ax[3].grid(alpha=.3)
    plt.suptitle("Gait coordination of a two-link prosthetic leg")
    plt.tight_layout(); plt.savefig(f"{outdir}/tracking.png", dpi=150); plt.close()

    # bar chart comparison
    keys = ["RMSE hip [deg]", "RMSE knee [deg]", "Coord. error max [deg]", "Mean positive power [W]"]
    fig, ax = plt.subplots(1, len(keys), figsize=(4 * len(keys), 3.6))
    for a, k in zip(ax, keys):
        a.bar(range(len(names)), [mets[n][k] for n in names], color=[COLORS.get(n) for n in names])
        a.set_xticks(range(len(names))); a.set_xticklabels([n.replace(" ", "\n", 1) for n in names], fontsize=7)
        a.set_title(k, fontsize=9)
    plt.tight_layout(); plt.savefig(f"{outdir}/comparison.png", dpi=150); plt.close()

    # phase portrait hip-knee (gait cycle shape)
    plt.figure(figsize=(5, 5))
    plt.plot(logs[names[0]]["qref"][:, 0] * DEG, logs[names[0]]["qref"][:, 1] * DEG, "k--", lw=2, label="reference")
    for n in names:
        plt.plot(logs[n]["q"][:, 0] * DEG, logs[n]["q"][:, 1] * DEG, color=COLORS.get(n), lw=1, label=n)
    plt.xlabel("hip q1 [deg]"); plt.ylabel("knee q2 [deg]"); plt.legend(fontsize=7); plt.grid(alpha=.3)
    plt.title("Hip-knee cyclogram"); plt.tight_layout(); plt.savefig(f"{outdir}/cyclogram.png", dpi=150); plt.close()


def plot_mpc_single(log, name, Ts, path):
    """separate figure for one MPC variant: tracking, error, torques, disturbance estimate."""
    t = log["t"]
    fig, ax = plt.subplots(5, 1, figsize=(11, 13), sharex=True)
    for j, jn in enumerate(["Hip q1", "Knee q2"]):
        ax[j].plot(t, log["qref"][:, j] * DEG, "k--", lw=2, label="reference q_L(t−Γ)")
        ax[j].plot(t, log["q"][:, j] * DEG, color=COLORS.get(name), lw=1.4, label=name)
        ax[j].set_ylabel(jn + " [deg]"); ax[j].grid(alpha=.3)
    ax[0].legend(fontsize=8, loc="upper right")
    ax[2].plot(t, np.linalg.norm(log["q"] - log["qref"], axis=1) * DEG, color=COLORS.get(name))
    ax[2].set_ylabel("‖e(t)‖ [deg]"); ax[2].grid(alpha=.3)
    ax[3].plot(t, log["u"][:, 0], label="hip"); ax[3].plot(t, log["u"][:, 1], label="knee")
    ax[3].set_ylabel("torque [Nm]"); ax[3].legend(fontsize=8); ax[3].grid(alpha=.3)
    if np.all(np.isfinite(log["dtrue"])):
        dl = lowpass(log["dtrue"], Ts)
        ax[4].plot(t, dl[:, 0], color="tab:blue", alpha=.45, label="true disturbance hip (low-pass)")
        ax[4].plot(t, dl[:, 1], color="tab:purple", alpha=.45, label="true disturbance knee (low-pass)")
    ax[4].plot(t, log["dhat"][:, 0], color="tab:blue", lw=1.6, label="estimate d̂ hip")
    ax[4].plot(t, log["dhat"][:, 1], color="tab:purple", lw=1.6, label="estimate d̂ knee")
    ax[4].set_ylabel("disturbance [Nm]"); ax[4].set_xlabel("time [s]"); ax[4].legend(fontsize=7, ncol=2)
    ax[4].grid(alpha=.3)
    plt.suptitle(name); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_estimator_comparison(lw, lo, Tg, path):
    names = ["MPC", "MPC (no estimator)"]
    logs = {"MPC": lw, "MPC (no estimator)": lo}
    t = lw["t"]
    fig = plt.figure(figsize=(13, 10))
    a1 = fig.add_subplot(3, 2, (1, 2))
    for n in names:
        a1.plot(t, np.linalg.norm(logs[n]["q"] - logs[n]["qref"], axis=1) * DEG, color=COLORS[n], label=n)
    a1.set_ylabel("‖e(t)‖ [deg]"); a1.set_xlabel("time [s]"); a1.legend(); a1.grid(alpha=.3)
    a1.set_title("Coordination error over time")
    a2 = fig.add_subplot(3, 2, 3)
    cw, co = per_cycle_rmse(lw, Tg), per_cycle_rmse(lo, Tg)
    k = np.arange(1, min(len(cw), len(co)) + 1)
    a2.bar(k - 0.2, cw[:len(k)], 0.4, color=COLORS["MPC"], label="MPC")
    a2.bar(k + 0.2, co[:len(k)], 0.4, color=COLORS["MPC (no estimator)"], label="no estimator")
    a2.set_xlabel("gait cycle"); a2.set_ylabel("RMSE [deg]"); a2.set_title("Per-cycle coordination RMSE")
    a2.legend(fontsize=8); a2.grid(alpha=.3, axis="y")
    a3 = fig.add_subplot(3, 2, 4)
    m = t > Tg
    data = [np.linalg.norm(logs[n]["q"][m] - logs[n]["qref"][m], axis=1) * DEG for n in names]
    try:
        a3.boxplot(data, tick_labels=["MPC", "no estimator"])       # matplotlib >= 3.9
    except TypeError:
        a3.boxplot(data, labels=["MPC", "no estimator"])            # older matplotlib
    a3.set_ylabel("‖e‖ [deg]"); a3.set_title("Error distribution (after cycle 1)"); a3.grid(alpha=.3)
    for j, jn in enumerate(["hip", "knee"]):
        a = fig.add_subplot(3, 2, 5 + j)
        for n in names:
            a.plot(t, (logs[n]["q"][:, j] - logs[n]["qref"][:, j]) * DEG, color=COLORS[n], lw=1, label=n)
        a.axhline(0, color="k", lw=.6); a.set_title(f"Signed {jn} error (bias)"); a.set_xlabel("time [s]")
        a.set_ylabel("[deg]"); a.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_terrain(log, env_info, plant_l, path, every=2):
    """world view: terrain profile and the prosthesis foot path (ankle point)."""
    hip = log["hip"]
    l1, l2 = plant_l
    q = log["q"]
    fx = hip[:, 0] + l1 * np.sin(q[:, 0]) + l2 * np.sin(q[:, 0] + q[:, 1])
    fy = hip[:, 1] - l1 * np.cos(q[:, 0]) - l2 * np.cos(q[:, 0] + q[:, 1])
    P = np.array(env_info["poly"])
    x0, x1 = fx.min() - 0.3, fx.max() + 0.3
    inside = (P[:, 0] > x0 - 0.5) & (P[:, 0] < x1 + 0.5)
    y0 = min(P[inside, 1].min() if inside.any() else P[:, 1].min(), fy.min()) - 0.15
    plt.figure(figsize=(12, 5))
    plt.fill_between(P[:, 0], P[:, 1], y0, color="#d9c7a8")
    plt.plot(P[:, 0], P[:, 1], color="#8b6b3e", lw=2, label="contact surface (ankle level)")
    plt.plot(fx[::every], fy[::every], color="tab:red", lw=1, label="prosthesis ankle path")
    plt.plot(hip[::every, 0], hip[::every, 1], color="k", lw=1, ls="--", label="hip (pelvis)")
    plt.xlim(x0, x1); plt.ylim(y0, hip[:, 1].max() + 0.1)
    if env_info["kind"] == "stairs":
        plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("x [m]"); plt.ylabel("y [m]"); plt.legend(fontsize=8, loc="upper left"); plt.grid(alpha=.3)
    plt.title(f"Terrain: {env_info['kind']}"); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def make_animation(log, gait, plant, shift, env, outdir, fps=25):
    from matplotlib.animation import FuncAnimation, PillowWriter
    t = log["t"]
    fig, ax = plt.subplots(figsize=(4, 5))
    ax.set_xlim(-0.7, 0.7); ax.set_ylim(-1.0, 0.15); ax.set_aspect("equal"); ax.axis("off")
    (gl,) = ax.plot([], [], color="saddlebrown", lw=3)
    (hl,) = ax.plot([], [], "o-", color="tab:blue", lw=6, alpha=.45, label="healthy limb")
    (pl,) = ax.plot([], [], "o-", color="tab:red", lw=4, label="prosthesis (MPC)")
    ttl = ax.text(0, 0.08, "", ha="center"); ax.legend(loc="lower center", fontsize=7)
    hip = log["hip"]
    P = env.terrain_polyline(hip[:, 0].min() - 2, hip[:, 0].max() + 2)

    def pts(q):
        k = np.array([plant.l1 * np.sin(q[0]), -plant.l1 * np.cos(q[0])])
        f = plant.foot(q)
        return [0, k[0], f[0]], [0, k[1], f[1]]

    def upd(i):
        qh, _, _ = gait.ref(t[i], 0.0)
        gl.set_data(P[:, 0] - hip[i, 0], P[:, 1] - hip[i, 1])
        hl.set_data(*pts(qh)); pl.set_data(*pts(log["q"][i])); ttl.set_text(f"t = {t[i]:.2f} s")
        return hl, pl, ttl, gl

    FuncAnimation(fig, upd, frames=range(0, len(t), 2), blit=False).save(
        f"{outdir}/animation.gif", writer=PillowWriter(fps=fps))
    plt.close()


# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="MPC for a two-link robotic prosthetic leg")
    ap.add_argument("--controller", default="all", choices=["all", "mpc", "mpc_noest", "ctc", "pd"])
    ap.add_argument("--activity", default=None, choices=list(ACT),
                    help="activity preset (default: legacy level gait with --gait-period)")
    ap.add_argument("--duration", type=float, default=6.0, help="simulation time [s]")
    ap.add_argument("--horizon", type=int, default=15, help="MPC prediction horizon (steps of 20 ms)")
    ap.add_argument("--gait-period", type=float, default=1.1, help="gait cycle Tg [s] (legacy mode only)")
    ap.add_argument("--delay-ratio", type=float, default=0.5, help="Gamma = ratio*Tg  (paper: 0.5)")
    ap.add_argument("--mismatch", type=float, default=0.25, help="plant mass error vs nominal model (0.25 = +25%%)")
    ap.add_argument("--friction", type=float, default=1.5, help="unmodelled viscous joint friction [Nms/rad]")
    ap.add_argument("--torque-noise", type=float, default=0.3, help="random torque disturbance intensity")
    ap.add_argument("--terrain", type=float, default=0.002, help="terrain roughness std [m] (stochastic)")
    ap.add_argument("--no-contact", action="store_true", help="disable foot-ground contact")
    ap.add_argument("--obstacles", action="store_true", help="stones & cracks on the road (level activities)")
    ap.add_argument("--obstacle-density", type=float, default=4.0, help="obstacles per 10 m of road")
    ap.add_argument("--placement", default="path", choices=["path", "random"],
                    help="path = at upcoming foot touch-downs (perturbation course), random = anywhere on the road")
    ap.add_argument("--hip-limit", type=float, default=None, help="default 100 (legacy) / 150 (activity)")
    ap.add_argument("--knee-limit", type=float, default=None, help="default 70 (legacy) / 120 (activity)")
    ap.add_argument("--no-dist-est", action="store_true", help="only with --controller mpc: disable the estimator")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--animate", action="store_true", help="also write animation.gif")
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    legacy = a.activity is None
    hip_lim = a.hip_limit if a.hip_limit is not None else (100.0 if legacy else 150.0)
    knee_lim = a.knee_limit if a.knee_limit is not None else (70.0 if legacy else 120.0)

    if a.controller == "all":
        chosen = ["pd", "ctc", "mpc", "mpc_noest"]
    elif a.controller == "mpc" and a.no_dist_est:
        chosen = ["mpc_noest"]
    else:
        chosen = [a.controller]
    gait = make_gait(a.activity, a.gait_period)
    print(f"Activity {a.activity or 'legacy level gait'}: Tg={gait.Tg:.3f}s, coordination delay "
          f"Gamma={a.delay_ratio * gait.Tg:.3f}s, mass mismatch {a.mismatch*100:.0f}%, friction {a.friction}, "
          f"contact={not a.no_contact}, obstacles={a.obstacles}")
    logs, mets, extra, healthy, info = run_scenario(
        a.activity, a.duration, a.horizon, a.delay_ratio, a.mismatch, a.friction, a.torque_noise, a.terrain,
        not a.no_contact, hip_lim, knee_lim, a.seed, chosen, a.obstacles, a.obstacle_density, a.gait_period,
        verbose=True, placement=a.placement)
    if info["stairs"]:
        print(f"Stairs: riser {gait.riser*100:.0f} cm, tread {gait.tread*100:.0f} cm, "
              f"{'ascent' if gait.stairs > 0 else 'descent'}, pelvis speed {info['speed']:.2f} m/s")
    elif a.obstacles:
        print(f"Road: {info['n_stones']} stones, {info['n_cracks']} cracks, ground speed {info['speed']:.2f} m/s")

    import pandas as pd
    table = pd.DataFrame(mets).T.round(3)
    print("\n=== Results (after first gait cycle) ===")
    print(table.to_string())
    table.to_csv(f"{a.out}/metrics.csv")
    make_plots(logs, mets, gait.Tg, a.out)
    plot_terrain(next(iter(logs.values())), info["terrain"], (info["l1"], info["l2"]), f"{a.out}/terrain.png")

    if "MPC" in logs and "MPC (no estimator)" in logs:
        Ts = info["Ts"]
        rw = {**mets["MPC"], **extra["MPC"]}
        ro = {**mets["MPC (no estimator)"], **extra["MPC (no estimator)"]}
        cmp_ = estimator_comparison_table(rw, ro).round(3)
        print("\n=== MPC with vs without disturbance estimator ===")
        print(cmp_.to_string())
        cmp_.to_csv(f"{a.out}/mpc_estimator_metrics.csv")
        plot_mpc_single(logs["MPC"], "MPC", Ts, f"{a.out}/mpc_with_estimator.png")
        plot_mpc_single(logs["MPC (no estimator)"], "MPC (no estimator)", Ts, f"{a.out}/mpc_without_estimator.png")
        plot_estimator_comparison(logs["MPC"], logs["MPC (no estimator)"], gait.Tg,
                                  f"{a.out}/mpc_estimator_comparison.png")
    if a.animate:
        key = "MPC" if "MPC" in logs else next(iter(logs))
        plant = Leg(m1=7.0 * (1 + a.mismatch), m2=4.0 * (1 + a.mismatch), b=(a.friction, a.friction))
        env = Environment(plant, gait, a.delay_ratio * gait.Tg, a.duration, 0.005, seed=a.seed, uneven=a.terrain,
                          torque_noise=a.torque_noise, contact=not a.no_contact, obstacles=a.obstacles,
                          obstacle_density=a.obstacle_density, placement=a.placement)
        make_animation(logs[key], gait, plant, a.delay_ratio * gait.Tg, env, a.out)
    print(f"\nSaved figures/metrics in ./{a.out}")


if __name__ == "__main__":
    main()
