"""Figures for the lecture slides introduction_control_theory.tex.

Run from the repository root:

    .venv/bin/python lectures/01_introduction_control_theory/make_figures.py [preview_dir]

The figures reuse the house style of the notebooks (notebooks/plotting.py), so the slides
and the notebooks look alike.  PDFs go to images/ next to this script; if a preview
directory is given, PNG copies are written there as well.  The last section writes the
Brownian-motion animation: a GIF that plays on its own, the matching multipage PDF that
Beamer's animate package embeds at FPS frames per second, and a PNG poster of the final
frame.  The residual-layer, Ising and Markov-chain diagrams are native TikZ in images/.
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "notebooks"))

import numpy as np
import torch
import matplotlib
import plotting as P                       # the course palette and figure helpers

matplotlib.use("agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Arc, Circle, FancyArrowPatch, Rectangle
from matplotlib.transforms import Affine2D
from PIL import Image

OUT = pathlib.Path(__file__).resolve().parent / "images"
OUT.mkdir(exist_ok=True)
PREVIEW = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else None
rng = np.random.default_rng(0)
trapz = getattr(np, "trapezoid", None) or np.trapz
FPS = 20                                   # frame rate of the Brownian-motion animation


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    if PREVIEW:
        fig.savefig(PREVIEW / f"{name}.png", dpi=170)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------------------------------
# Numerics shared by several figures
# ---------------------------------------------------------------------------------------
def odeint(f, x0, ts):
    """RK4 for x' = f(t, x) along the grid ts."""
    xs = [np.asarray(x0, float)]
    for t, t1 in zip(ts[:-1], ts[1:]):
        x, h = xs[-1], t1 - t
        k1 = f(t, x)
        k2 = f(t + h / 2, x + h / 2 * k1)
        k3 = f(t + h / 2, x + h / 2 * k2)
        k4 = f(t + h, x + h * k3)
        xs.append(x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4))
    return np.array(xs)


def expm(M):
    return torch.matrix_exp(torch.as_tensor(M, dtype=torch.float64)).numpy()


def lti(A, B, u):
    return lambda t, x: A @ x + B @ u(t)


def gramian(A, B, T, n=600):
    taus = np.linspace(0, T, n + 1)
    vals = np.array([expm(A * tau) @ B @ B.T @ expm(A * tau).T for tau in taus])
    return trapz(vals, taus, axis=0)


def l2_norm(u, ts):
    vals = np.array([u(s) for s in ts])
    return float(np.sqrt(trapz((vals ** 2).sum(-1), ts)))


# The harmonic oscillator x1'' + x1 = u, the running example of the lecture
A_osc = np.array([[0.0, 1.0], [-1.0, 0.0]])
B_osc = np.array([[0.0], [1.0]])
T = 5.0
x0 = np.array([1.0, 0.0])
x1 = np.array([0.0, 0.0])
ts = np.linspace(0, T, 1001)

# "Draw a curve, read off the control" (notebook 02, section 2)
curve = lambda s: 1 - 3 * (s / T) ** 2 + 2 * (s / T) ** 3
curve_dd = lambda s: -6 / T ** 2 + 12 * s / T ** 3
u_curve = lambda s: np.array([curve_dd(s) + curve(s)])

x_free = odeint(lti(A_osc, B_osc, lambda t: np.zeros(1)), x0, ts)
x_curve = odeint(lti(A_osc, B_osc, u_curve), x0, ts)

# The Gramian control (G) (notebook 02, section 6)
G = gramian(A_osc, B_osc, T)
d = x1 - expm(A_osc * T) @ x0
c_G = np.linalg.solve(G, d)
u_G = lambda t: B_osc.T @ expm(A_osc.T * (T - t)) @ c_G
x_G = odeint(lti(A_osc, B_osc, u_G), x0, ts)
print("Gramian G_T =", G.round(4).tolist())
print(f"|x(T)-x1| with (G): {np.linalg.norm(x_G[-1] - x1):.1e}")
print(f"L2 norm of the curve control: {l2_norm(u_curve, ts):.4f}")
print(f"L2 norm of the control (G):   {l2_norm(u_G, ts):.4f}   (d G^-1 d)^1/2 = {np.sqrt(d @ c_G):.4f}")


# ---------------------------------------------------------------------------------------
# 1. Three dynamical systems: discrete time, continuous time, infinite dimension
# ---------------------------------------------------------------------------------------
fig, (a1, a2, a3) = P.panels(3, size=(2.45, 2.25))

r = 3.9                                                   # the logistic map
for x, col, lab in [(0.3, P.BLUE, "$x_0=0.3$"), (0.3001, P.ORANGE, "$x_0=0.3001$")]:
    xs = [x]
    for _ in range(40):
        xs.append(r * xs[-1] * (1 - xs[-1]))
    a1.plot(range(41), xs, ".-", color=col, lw=0.9, ms=3, label=lab)
P.label(a1, "$k$", "$x_k$", title="discrete:  $x_{k+1}=r\\,x_k(1-x_k)$", ylim=(0, 1.35),
        legend=dict(loc="upper left", ncol=2, handlelength=1.2, columnspacing=0.8))

th, v = np.meshgrid(np.linspace(-2 * np.pi, 2 * np.pi, 400), np.linspace(-3.2, 3.2, 300))
H = v ** 2 / 2 - np.cos(th)                                # the pendulum
a2.contour(th, v, H, levels=[-0.8, -0.3, 0.4, 1.8, 2.9], colors=[P.BLUE], linewidths=1.1, linestyles="solid")
a2.contour(th, v, H, levels=[1.0], colors=[P.INK], linewidths=1.3, linestyles="--")
a2.plot([-np.pi, np.pi], [0, 0], "o", color=P.ORANGE, ms=4)
a2.plot([0], [0], "o", color=P.ORANGE, ms=4)
a2.set_xticks([-2 * np.pi, 0, 2 * np.pi], ["$-2\\pi$", "0", "$2\\pi$"])
P.label(a2, r"$\theta$", r"$\theta'$", title=r"continuous:  $\theta''=-\sin\theta$")

xx = np.linspace(0, 1, 400)                                # the heat equation
n = np.arange(1, 400)[:, None]
b = 2 / (n * np.pi) * (np.cos(0.3 * n * np.pi) - np.cos(0.7 * n * np.pi))
modes = np.sin(n * np.pi * xx)
times = [0.0, 0.002, 0.01, 0.03, 0.1]
shades = plt.cm.viridis(np.linspace(0, 0.85, len(times)))
for t, col in zip(times, shades):
    y = (b * np.exp(-(n * np.pi) ** 2 * t) * modes).sum(0)
    a3.plot(xx, y, color=col, label=f"$t={t:g}$")
P.label(a3, "$x$", "$y(t,x)$", title="infinite-dim.:  $y_t=y_{xx}$", ylim=(-0.05, 1.5),
        legend=dict(loc="upper right", fontsize=7, handlelength=1.2))
save(fig, "dynamical_systems")


# ---------------------------------------------------------------------------------------
# 2. The spring-mass schematic and the free oscillator in the phase plane
# ---------------------------------------------------------------------------------------
fig, ax = P.panels(figsize=(4.6, 2.0))
P.spring_mass_damper(ax, pos=1.1)
save(fig, "spring_mass")

fig, ax = P.panels(figsize=(2.6, 2.5))
ax.plot(x_free[:, 0], x_free[:, 1], color=P.BLUE)
ax.plot(x_free[0, 0], x_free[0, 1], "o", color=P.INK, ms=5)
ax.annotate("", xy=(x_free[60, 0], x_free[60, 1]), xytext=(x_free[40, 0], x_free[40, 1]),
            arrowprops=dict(arrowstyle="-|>", color=P.BLUE, lw=1.5))
ax.text(1.08, 0.12, "$x^0$", fontsize=10)
P.label(ax, "$x_1$ (position)", "$x_2$ (velocity)", title="$u=0$: energy is conserved",
        xlim=(-1.4, 1.4), ylim=(-1.4, 1.4), equal=True, zero="both")
save(fig, "oscillator_free")


# ---------------------------------------------------------------------------------------
# 3. Three questions, one oscillator: controllability, optimal control, feedback.
#    The target is a new rest position, away from the origin, so that the first two
#    panels do not look like the third.
# ---------------------------------------------------------------------------------------
target = np.array([-0.5, 0.0])
d_target = target - expm(A_osc * T) @ x0
controls = {"controllability": np.linalg.solve(G, d_target),         # control (G)
            "optimal": np.linalg.solve(G + np.eye(2), d_target)}     # minimizer of |x(T)-x1|^2 + ∫|u|^2
x_q = {k: odeint(lti(A_osc, B_osc, lambda t, c=c: B_osc.T @ expm(A_osc.T * (T - t)) @ c), x0, ts)
       for k, c in controls.items()}
ts_fb = np.linspace(0, 15, 1501)
x_fb = odeint(lambda t, x: np.array([[0.0, 1.0], [-1.0, -1.0]]) @ x, x0, ts_fb)
x_q["stabilization"] = x_fb
print(f"optimal control: x(T) = {x_q['optimal'][-1].round(3)}")

for name, traj in x_q.items():
    fig, ax = P.panels(figsize=(1.9, 1.15))
    ax.plot(traj[:, 0], traj[:, 1], color=P.BLUE)
    ax.plot(*x0, "o", color=P.INK, ms=4)
    ax.text(1.0, 0.1, "$x^0$", fontsize=9, ha="center")
    if name != "stabilization":
        ax.plot(*target, "*", color=P.ORANGE, ms=10, zorder=5)
        ax.text(target[0] - 0.1, target[1] + 0.1, "$x^1$", fontsize=9, ha="right")
    if name == "optimal":
        ax.plot(*traj[-1], "o", color=P.BLUE, ms=4)
        ax.text(traj[-1, 0] + 0.1, traj[-1, 1] + 0.02, "$x(T)$", fontsize=9, color=P.BLUE)
    P.label(ax, xlim=(-1.0, 1.2), ylim=(-0.85, 0.45), equal=True, zero="both", off=True)
    save(fig, f"three_questions_{name}")


# ---------------------------------------------------------------------------------------
# 4. Open loop versus closed loop
# ---------------------------------------------------------------------------------------
fig, ax = P.panels(figsize=(5.0, 3.7))
P.feedback_diagram(ax)
save(fig, "feedback_diagram")


# ---------------------------------------------------------------------------------------
# 5. Two first examples (notebook 02, section 2)
# ---------------------------------------------------------------------------------------
A1, B1 = np.eye(2), np.array([[1.0], [0.0]])
y0, T1 = np.array([0.5, 0.5]), 1.0
ts1 = np.linspace(0, T1, 301)


def random_control(k=1.0):
    a, b, w = rng.normal(size=3)
    return lambda t: k * np.array([a + 2 * b * np.cos((1.5 + w) * t) + 0.2 * np.sin(5 * t)])


fig, (a1, a2) = P.panels(2, size=(3.1, 2.5))
a1.plot(*odeint(lti(A1, B1, lambda t: np.zeros(1)), y0, ts1).T, color=P.GREY, lw=1.2)
for k, c0 in enumerate(np.linspace(-2.2, 2.2, 7)):
    u_k = (lambda c0, a: (lambda t: np.array([c0 + a * np.sin(2 * np.pi * t)])))(c0, 2.0 * (-1) ** k)
    traj = odeint(lti(A1, B1, u_k), y0, ts1)
    a1.plot(traj[:, 0], traj[:, 1], color=P.BLUE, lw=0.9, alpha=0.8)
    a1.plot(traj[-1, 0], traj[-1, 1], "o", color=P.BLUE, ms=3.5)
a1.axhline(y0[1] * np.exp(T1), color=P.INK, lw=1.2, ls="--")
a1.text(-5.6, y0[1] * np.exp(T1) + 0.1, r"$\mathcal{R}(T,x^0)=\{x_2=x_2^0e^{T}\}$", fontsize=9)
a1.plot(*y0, "o", color=P.INK, ms=5)
a1.text(0.65, 0.3, "$x^0$", fontsize=9)
P.label(a1, "$x_1$", "$x_2$", title="Example 1: $x_1'=x_1+u,\\ x_2'=x_2$",
        xlim=(-6, 6), ylim=(0, 2.1))

a2.plot(ts, x_free[:, 0], color=P.GREY, lw=1.2, label="free motion")
a2.plot(ts, x_curve[:, 0], color=P.BLUE, label="$x_1(t)$ (the cubic)")
a2.plot(ts, [u_curve(s)[0] for s in ts], color=P.ORANGE, ls="--", label="$u=x_1''+x_1$")
a2.plot(T, 0, "*", color=P.ORANGE, ms=11, zorder=5)
P.label(a2, "$t$", None, title="Example 2: $x_1''+x_1=u$", zero="h",
        legend=dict(loc="upper right", fontsize=7.5))
save(fig, "two_examples")


# ---------------------------------------------------------------------------------------
# 6. The reachable set: an affine subspace whose offset moves with T
# ---------------------------------------------------------------------------------------
fig = plt.figure(figsize=(6.4, 2.75))
a1 = fig.add_subplot(1, 2, 1)
a2 = fig.add_subplot(1, 2, 2, projection="3d")

Ts = [0.4, 0.8, 1.2]
shades = plt.cm.viridis(np.linspace(0, 0.85, len(Ts)))
tt = np.linspace(0, 1.3, 100)
a1.plot(0.5 * np.exp(tt), 0.5 * np.exp(tt), color=P.GREY, lw=1.2)
for Tk, col in zip(Ts, shades):
    off = y0[1] * np.exp(Tk)
    a1.axhline(off, color=col, lw=1.3)
    a1.text(-5.6, off + 0.08, f"$T={Tk}$", color=col, fontsize=8)
    ends = [odeint(lti(A1, B1, random_control(1.5)), y0, np.linspace(0, Tk, 200))[-1]
            for _ in range(10)]
    a1.plot(*np.array(ends).T, "o", color=col, ms=3)
    a1.plot(off, off, "o", color=P.INK, ms=5, zorder=5)
a1.plot(*y0, "o", color=P.INK, ms=5)
a1.text(0.6, 0.25, "$x^0$", fontsize=9)
a1.text(2.2, 0.55 * np.exp(1.2) + 0.1, "$e^{AT}x^0$", fontsize=9)
P.label(a1, "$x_1$", "$x_2$", title=r"Example 1: direction $e_1$, offset $e^{AT}x^0$",
        xlim=(-6, 6), ylim=(0.2, 2.3))

A3 = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
B3 = np.array([[0.0], [1.0], [0.0]])                      # the family at eps = 0
z0, T3 = np.array([0.0, 0.0, 1.0]), 2.0
ends = np.array([odeint(lti(A3, B3, random_control(1.2)), z0, np.linspace(0, T3, 200))[-1]
                 for _ in range(40)])
gx, gy = np.meshgrid(np.linspace(-4, 4, 2), np.linspace(-4, 4, 2))
a2.plot_surface(gx, gy, np.full_like(gx, np.exp(-T3)), color=P.BLUE, alpha=0.18, lw=0)
a2.scatter(ends[:, 0], ends[:, 1], ends[:, 2], color=P.BLUE, s=8, depthshade=False)
a2.scatter([0], [0], [np.exp(-T3)], color=P.INK, s=40, depthshade=False, zorder=10)
a2.scatter(*z0, color=P.INK, s=25, depthshade=False)
a2.plot([0, 0], [0, 0], [np.exp(-T3), 1.0], color=P.GREY, lw=1.2)
a2.text(0.2, 0.2, 1.0, "$x^0$", fontsize=9)
a2.text(1.2, 1.2, np.exp(-T3) + 0.16, "$e^{AT}x^0$", fontsize=9)
a2.set(xlabel="$x_1$", ylabel="$x_2$", zlabel="$x_3$", zlim=(0, 1.05))
a2.set_title(r"$\varepsilon=0$: the plane $\{x_3=e^{-T}\}$", fontsize=10)
a2.view_init(elev=18, azim=-55)
a2.tick_params(labelsize=7)
save(fig, "reachable_set")


# ---------------------------------------------------------------------------------------
# 7. The rank block by block: the chain of integrators x''' = u
# ---------------------------------------------------------------------------------------
Ac = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
Bc = np.array([[0.0], [0.0], [1.0]])
bump = lambda t: np.array([1.0 if t < 1.0 else 0.0])
tsc = np.linspace(0, 3, 601)
zc = odeint(lti(Ac, Bc, bump), np.zeros(3), tsc)
fig, ax = P.panels(figsize=(3.3, 2.4))
ax.step(tsc, [bump(s)[0] for s in tsc], color=P.GREY, lw=1.2, label="$u$")
ax.plot(tsc, zc[:, 2], color=P.BLUE, label="$x''$  ($B=e_3$)")
ax.plot(tsc, zc[:, 1], color=P.ORANGE, label="$x'$  ($AB=e_2$)")
ax.plot(tsc, zc[:, 0], color=P.AQUA, label="$x$  ($A^2B=e_1$)")
P.label(ax, "$t$", None, title="$x'''=u$: one integration per block", zero="h",
        legend=dict(loc="upper left", fontsize=7.5))
save(fig, "integrator_chain")


# ---------------------------------------------------------------------------------------
# 8. Parking: several movements combine into a sideways displacement
# ---------------------------------------------------------------------------------------
L, W, LEN = 2.5, 1.8, 4.0                 # wheelbase, width, length of the car
REAR = 0.7                                # rear overhang behind the rear axle
PHI = 0.5                                 # full steering lock (rad)
PSI = np.deg2rad(41.4)                    # heading reached at the end of the first arc
t_arc = PSI * L / np.tan(PHI)             # reversing time of each arc at |v| = 1
t_steer = 0.6
moves = [                                 # (name, v, omega, duration)
    ("steer right", 0.0, -PHI / t_steer, t_steer),
    ("reverse", -1.0, 0.0, t_arc),
    ("steer left", 0.0, 2 * PHI / t_steer, t_steer),
    ("reverse", -1.0, 0.0, t_arc),
    ("straighten", 0.0, -PHI / t_steer, t_steer),
    ("forward", 1.0, 0.0, 1.2),
]


def car(v, om):                           # Nelson's car: state (x, y, theta, phi)
    return lambda t, z: np.array([v * np.cos(z[2]), v * np.sin(z[2]), v / L * np.tan(z[3]), om])


y_lane = 1.1 + 2 * L / np.tan(PHI) * (1 - np.cos(PSI))
z = np.array([2.8, y_lane, 0.0, 0.0])
path, snaps = [z], [(0, z.copy())]
for k, (name, v, om, dur) in enumerate(moves, 1):
    zs = odeint(car(v, om), path[-1], np.linspace(0, dur, 80))
    path.extend(zs[1:])
    if v != 0:
        snaps.append((k, zs[-1].copy()))
path = np.array(path)
print("parking: final state", path[-1].round(3), " lane y", round(y_lane, 3))


def draw_car(ax, z, color, alpha=1.0, fill="white", zorder=3):
    x, y, th, ph = z
    body = Affine2D().rotate(th).translate(x, y) + ax.transData
    ax.add_patch(Rectangle((-REAR, -W / 2), LEN, W, transform=body, fc=fill, ec=color,
                           lw=1.4, alpha=alpha, zorder=zorder))
    ax.add_patch(Rectangle((0.9, -W / 2 + 0.2), 1.6, W - 0.4, transform=body, fc=color,
                           ec="none", alpha=0.12 * alpha, zorder=zorder))
    for wx in (0.0, L):
        for wy in (-W / 2 + 0.12, W / 2 - 0.12):
            wheel = Affine2D().rotate(ph if wx > 0 else 0.0).translate(wx, wy) + body
            ax.add_patch(Rectangle((-0.32, -0.1), 0.64, 0.2, transform=wheel, fc=color,
                                   ec="none", alpha=alpha, zorder=zorder + 1))


fig, ax = P.panels(figsize=(6.6, 2.35))
ax.fill_between([-12, 10.5], -1.3, 0, color=P.GHOST, lw=0)
ax.plot([-12, 10.5], [0, 0], color=P.GREY, lw=2.5)
ax.text(-11.7, -0.95, "curb", fontsize=8, color=P.NOTE)
for xr in (-8.8, 4.7):
    draw_car(ax, (xr, 1.1, 0.0, 0.0), P.GREY, fill="#f1f0ea")
ax.plot(path[:, 0], path[:, 1], ":", color=P.BLUE, lw=1.1, zorder=2)
for j, (k, zk) in enumerate(snaps):
    a = 0.3 + 0.7 * j / (len(snaps) - 1)
    draw_car(ax, zk, P.BLUE, alpha=a, fill="#eef3fb")
    cx, cy = zk[0] + 1.3 * np.cos(zk[2]), zk[1] + 1.3 * np.sin(zk[2])   # centre of the body
    ax.add_patch(Circle((cx, cy), 0.42, fc="white", ec=P.BLUE, lw=1.2, zorder=8))
    ax.text(cx, cy, str(k), ha="center", va="center", fontsize=8, color=P.BLUE, zorder=9)
P.label(ax, xlim=(-12, 10.5), ylim=(-1.3, 6.0), equal=True, off=True)
save(fig, "parking")


# ---------------------------------------------------------------------------------------
# 9. The Gramian control (G) stops the oscillator (notebook 02, section 6)
# ---------------------------------------------------------------------------------------
fig, (a1, a2) = P.panels(2, size=(3.1, 2.45))
a1.plot(ts, x_free[:, 0], color=P.GREY, lw=1.2, label="free motion")
a1.plot(ts, x_curve[:, 0], color=P.ORANGE, ls="--", label="the cubic (Ex. 2)")
a1.plot(ts, x_G[:, 0], color=P.BLUE, label="control (G)")
a1.plot(T, 0, "*", color=P.ORANGE, ms=11, zorder=5)
P.label(a1, "$t$", "$x_1(t)$", title="position", zero="h", legend=dict(loc="upper right", fontsize=7.5))
a2.plot(ts, [u_curve(s)[0] for s in ts], color=P.ORANGE, ls="--",
        label=f"$u=x_1''+x_1$,  $\\Vert u\\Vert_{{L^2}}={l2_norm(u_curve, ts):.2f}$")
a2.plot(ts, [u_G(s)[0] for s in ts], color=P.BLUE,
        label=f"(G),  $\\Vert u\\Vert_{{L^2}}={l2_norm(u_G, ts):.2f}$")
P.label(a2, "$t$", "$u(t)$", title="two controls that stop it", zero="h",
        legend=dict(loc="upper right", fontsize=7.5))
save(fig, "gramian_control")


# ---------------------------------------------------------------------------------------
# 10. Quantitative controllability: the near-uncontrollable family (notebook 02, section 7)
# ---------------------------------------------------------------------------------------
def family(eps):
    return A3, np.array([[0.0], [1.0], [eps]])


epsilons = np.logspace(-3, 0, 13)
target = np.array([0.0, 0.0, 1.0])
energy = []
for eps in epsilons:
    Ae, Be = family(eps)
    Ge = gramian(Ae, Be, T)
    de = target - expm(Ae * T) @ np.zeros(3)
    energy.append(de @ np.linalg.solve(Ge, de))
energy = np.array(energy)
print("energy at eps=1e-3, 1:", energy[0], energy[-1])
fig, ax = P.panels(figsize=(3.3, 2.6))
ax.loglog(epsilons, energy, "o-", color=P.ORANGE, label="energy of (G)")
ax.loglog(epsilons, energy[-1] / epsilons ** 2, "--", color=P.GREY, lw=1, label=r"$\propto\varepsilon^{-2}$")
P.label(ax, r"actuator strength $\varepsilon$", r"$\int_0^T|u|^2\,dt$",
        legend=dict(loc="upper right", fontsize=8))
save(fig, "near_uncontrollable")


# ---------------------------------------------------------------------------------------
# 11. The overdamping paradox (notebook 01, section 4)
# ---------------------------------------------------------------------------------------
ks = [0.5, 1.0, 2.0, 4.0, 8.0]
ts15 = np.linspace(0, 15, 1501)
shades = plt.cm.viridis(np.linspace(0, 0.85, len(ks)))
fig, (a1, a2) = P.panels(nrows=2, size=(3.0, 1.9), sharex=True)
for k, col in zip(ks, shades):
    Ak = np.array([[0.0, 1.0], [-1.0, -k]])
    xk = odeint(lambda t, x: Ak @ x, x0, ts15)
    a1.plot(ts15, xk[:, 0], color=col, label=f"$k={k:g}$")
    a2.semilogy(ts15, 0.5 * (xk ** 2).sum(1), color=col)
P.label(a1, None, "$x_1(t)$", title="position", zero="h", legend=dict(fontsize=7, loc="upper right"))
P.label(a2, "$t$", "$e(t)$", title="energy", ylim=(1e-12, 1))
save(fig, "overdamping")


# ---------------------------------------------------------------------------------------
# 12. The three dynamical systems one by one, for the slides that discuss them separately
# ---------------------------------------------------------------------------------------
fig, ax = P.panels(figsize=(3.3, 2.5))
for x, col, lab in [(0.3, P.BLUE, "$x_0=0.3$"), (0.3001, P.ORANGE, "$x_0=0.3001$")]:
    xs = [x]
    for _ in range(40):
        xs.append(r * xs[-1] * (1 - xs[-1]))
    ax.plot(range(41), xs, ".-", color=col, lw=0.9, ms=3, label=lab)
P.label(ax, "$k$", "$x_k$", title="the logistic map, $r=3.9$", ylim=(0, 1.35),
        legend=dict(loc="upper left", ncol=2, handlelength=1.2, columnspacing=0.8))
save(fig, "ds_discrete")

fig, ax = P.panels(figsize=(3.3, 2.5))
TH, VV = np.meshgrid(np.linspace(-2 * np.pi, 2 * np.pi, 400), np.linspace(-3.2, 3.2, 300))
HH = VV ** 2 / 2 - np.cos(TH)
ax.contour(TH, VV, HH, levels=[-0.8, -0.3, 0.4, 1.8, 2.9], colors=[P.BLUE], linewidths=1.1, linestyles="solid")
ax.contour(TH, VV, HH, levels=[1.0], colors=[P.INK], linewidths=1.3, linestyles="--")
ax.plot([-np.pi, np.pi], [0, 0], "o", color=P.ORANGE, ms=4)
ax.plot([0], [0], "o", color=P.ORANGE, ms=4)
ax.set_xticks([-2 * np.pi, 0, 2 * np.pi], ["$-2\\pi$", "0", "$2\\pi$"])
P.label(ax, r"$\theta$", r"$\theta'$", title=r"the pendulum $\theta''=-\sin\theta$")
save(fig, "ds_pendulum")

fig, ax = P.panels(figsize=(3.3, 2.5))
for t, col in zip(times, plt.cm.viridis(np.linspace(0, 0.85, len(times)))):
    y = (b * np.exp(-(n * np.pi) ** 2 * t) * modes).sum(0)
    ax.plot(xx, y, color=col, label=f"$t={t:g}$")
P.label(ax, "$x$", "$y(t,x)$", title="the heat equation $y_t=y_{xx}$", ylim=(-0.05, 1.5),
        legend=dict(loc="upper right", fontsize=7, handlelength=1.2))
save(fig, "ds_heat")


# ---------------------------------------------------------------------------------------
# 13. Three more dynamical systems: gradient descent, the wave and the transport equation
# ---------------------------------------------------------------------------------------
fig, ax = P.panels(figsize=(3.3, 2.5))
w = np.linspace(-0.5, 2.65, 400)                           # gradient descent on F(w) = w^2/2
ax.plot(w, 0.5 * w ** 2, color=P.BLUE, lw=1.5)
eta = 0.35
iterates = 2.4 * (1 - eta) ** np.arange(7)
losses = 0.5 * iterates ** 2
for left, right, f_left, f_right in zip(iterates[:-1], iterates[1:], losses[:-1], losses[1:]):
    ax.annotate("", xy=(right, f_right), xytext=(left, f_left),
                arrowprops=dict(arrowstyle="-|>", color=P.ORANGE, lw=1.2, shrinkA=3, shrinkB=3))
ax.plot(iterates, losses, "o", color=P.ORANGE, ms=4)
ax.plot(0, 0, "*", color=P.INK, ms=8, zorder=4)
ax.annotate("$w_0$", (iterates[0], losses[0]), xytext=(-22, 2), textcoords="offset points",
            fontsize=9, color=P.NOTE)
P.label(ax, "$w$", "$F(w)$", xlim=(-0.5, 2.7), ylim=(-0.2, 3.6))
ax.locator_params(axis="both", nbins=4)
save(fig, "ds_gradient_descent")

x = np.linspace(0, 1, 700)                                 # finer than xx, for the narrow pulses
colours, styles = [P.BLUE, P.AQUA, P.ORANGE], ["-", "--", "-."]

# d'Alembert's formula on the real line with zero initial velocity: the pulse splits in two
initial_wave = lambda z: np.exp(-((z - 0.5) / 0.075) ** 2)
fig, ax = P.panels(figsize=(3.3, 2.5))
for t, col, style in zip([0, 0.18, 0.36], colours, styles):
    ax.plot(x, 0.5 * (initial_wave(x - t) + initial_wave(x + t)), color=col, ls=style, lw=1.5,
            label=f"$t={t:g}$")
P.label(ax, "$x$", "$y(t,x)$", xlim=(0, 1), ylim=(-0.04, 1.45),
        legend=dict(loc="upper right", fontsize=8, handlelength=1.5))
ax.locator_params(axis="both", nbins=4)
save(fig, "ds_wave")

# Exact translation at speed c = 1, preserving the height and width of the pulse
initial_transport = lambda z: np.exp(-((z - 0.18) / 0.065) ** 2)
fig, ax = P.panels(figsize=(3.3, 2.5))
for t, col, style in zip([0, 0.28, 0.56], colours, styles):
    ax.plot(x, initial_transport(x - t), color=col, ls=style, lw=1.5, label=f"$t={t:g}$")
ax.annotate("", xy=(0.58, 1.2), xytext=(0.19, 1.2),
            arrowprops=dict(arrowstyle="->", color=P.NOTE, lw=1.2))
P.label(ax, "$x$", "$y(t,x)$", xlim=(0, 1), ylim=(-0.04, 1.45),
        legend=dict(loc="upper right", fontsize=8, handlelength=1.5))
ax.locator_params(axis="both", nbins=4)
save(fig, "ds_transport")


# ---------------------------------------------------------------------------------------
# 14. Two controlled examples: a motor at the pivot of the pendulum, a heater in the rod.
#     The control is drawn in P.ORANGE, the colour of the control in the slide equations.
# ---------------------------------------------------------------------------------------
fig, ax = P.panels(figsize=(3.3, 2.5))
theta, length = np.deg2rad(35), 1.55
bob = length * np.array([np.sin(theta), -np.cos(theta)])
ax.add_patch(Rectangle((-1.42, -0.55), 0.1, 1.1, color=P.GREY))           # the wall
for yy in np.linspace(-0.55, 0.37, 6):                                     # its hatching
    ax.plot([-1.6, -1.42], [yy, yy + 0.18], color=P.GREY, lw=1)
ax.plot([-1.32, 0], [0, 0], color=P.GREY, lw=2.5, solid_capstyle="butt")
ax.plot([0, 0], [0, -1.6], ls="--", color=P.RULE, lw=1)
ax.plot([0, bob[0]], [0, bob[1]], color=P.INK, lw=1.6)
ax.add_patch(Circle(bob, 0.17, fc="#eef3fb", ec=P.INK, lw=1.4, zorder=3))
ax.add_patch(Circle((0, 0), 0.1, fc="#fdf0e8", ec=P.ORANGE, lw=1.4, zorder=3))
ax.add_patch(Arc((0, 0), 1.3, 1.3, theta1=-90, theta2=-90 + np.rad2deg(theta), color=P.NOTE, lw=1))
ax.text(0.2, -0.95, r"$\theta$", fontsize=11, color=P.INK)
ax.add_patch(FancyArrowPatch((0.36, 0.13), (-0.36, 0.13), connectionstyle="arc3,rad=0.75",
                             arrowstyle="-|>", mutation_scale=12, color=P.ORANGE, lw=1.5))
ax.text(0.0, 0.62, "$u(t)$", ha="center", fontsize=11, color=P.ORANGE)
P.label(ax, xlim=(-1.7, 1.25), ylim=(-1.7, 0.92), equal=True, off=True)
save(fig, "ctrl_pendulum")

# Heat with a heater on omega = (0.2, 0.4): y_t = y_xx + 1_omega u, y(0) = 0, constant u = 25.
# Each sine mode solves c_n' = -(n pi)^2 c_n + u f_n exactly (n: the mode index of section 1).
a_w, b_w, u_heat = 0.2, 0.4, 25.0
lam = (n * np.pi) ** 2
f_w = 2 / (n * np.pi) * (np.cos(n * np.pi * a_w) - np.cos(n * np.pi * b_w))
modes_x = np.sin(n * np.pi * x)
times_u = [0.005, 0.02, 0.05, 0.3]
fig, ax = P.panels(figsize=(3.3, 2.5))
ax.axvspan(a_w, b_w, color=P.ORANGE, alpha=0.15, lw=0)
ax.text((a_w + b_w) / 2, 1.3, r"$\omega$", ha="center", fontsize=11, color=P.ORANGE)
for t, col in zip(times_u, plt.cm.viridis(np.linspace(0, 0.85, len(times_u)))):
    y = (u_heat * f_w / lam * (1 - np.exp(-lam * t)) * modes_x).sum(0)
    ax.plot(x, y, color=col, lw=1.5, label=f"$t={t:g}$")
P.label(ax, "$x$", "$y(t,x)$", xlim=(0, 1), ylim=(-0.04, 1.45),
        legend=dict(loc="upper right", fontsize=8, handlelength=1.5))
ax.locator_params(axis="both", nbins=4)
save(fig, "ctrl_heat")


# ---------------------------------------------------------------------------------------
# 15. The free oscillator and the feedback u = -x2 (friction)
# ---------------------------------------------------------------------------------------
fig, (a1, a2) = P.panels(2, size=(3.1, 2.5))
a1.plot(ts_fb, np.cos(ts_fb), color=P.GREY, label="$u=0$")
a1.plot(ts_fb, x_fb[:, 0], color=P.BLUE, label="$u=-x_2$")
P.label(a1, "$t$", "$x_1(t)$", title="position", zero="h", legend=dict(loc="upper right", fontsize=8))
a2.plot(np.cos(ts_fb), -np.sin(ts_fb), color=P.GREY, label="$u=0$")
a2.plot(x_fb[:, 0], x_fb[:, 1], color=P.BLUE, label="$u=-x_2$")
a2.plot(1, 0, "o", color=P.INK, ms=5)
a2.text(1.06, 0.1, "$x^0$", fontsize=9)
P.label(a2, "$x_1$", "$x_2$", title="phase plane", xlim=(-1.3, 1.3), ylim=(-1.3, 1.3),
        equal=True, zero="both", legend=dict(loc="upper left", fontsize=8))
save(fig, "oscillator_free_feedback")


# ---------------------------------------------------------------------------------------
# 16. The two first examples, one figure each
# ---------------------------------------------------------------------------------------
fig, ax = P.panels(figsize=(3.4, 2.6))
ax.plot(*odeint(lti(A1, B1, lambda t: np.zeros(1)), y0, ts1).T, color=P.GREY, lw=1.2, label="$u=0$")
for k, c0 in enumerate(np.linspace(-2.2, 2.2, 7)):
    u_k = (lambda c0, a: (lambda t: np.array([c0 + a * np.sin(2 * np.pi * t)])))(c0, 2.0 * (-1) ** k)
    traj = odeint(lti(A1, B1, u_k), y0, ts1)
    ax.plot(traj[:, 0], traj[:, 1], color=P.BLUE, lw=0.9, alpha=0.8)
    ax.plot(traj[-1, 0], traj[-1, 1], "o", color=P.BLUE, ms=3.5)
ax.axhline(y0[1] * np.exp(T1), color=P.INK, lw=1.2, ls="--")
ax.text(-5.6, y0[1] * np.exp(T1) + 0.1, r"$\mathcal{R}(T,x^0)=\{x_2=x_2^0e^{T}\}$", fontsize=9)
ax.plot(*y0, "o", color=P.INK, ms=5)
ax.text(0.65, 0.3, "$x^0$", fontsize=9)
P.label(ax, "$x_1$", "$x_2$", title="seven controls, $T=1$", xlim=(-6, 6), ylim=(0, 2.1), legend=False)
save(fig, "example_uncontrollable")

fig, ax = P.panels(figsize=(3.4, 2.6))
ax.plot(ts, x_free[:, 0], color=P.GREY, lw=1.2, label="free motion")
ax.plot(ts, x_curve[:, 0], color=P.BLUE, label="$x_1(t)$ (the cubic)")
ax.plot(T, 0, "*", color=P.ORANGE, ms=11, zorder=5)
P.label(ax, "$t$", None, zero="h", legend=dict(loc="upper right", fontsize=7.5))
save(fig, "example_curve")


# ---------------------------------------------------------------------------------------
# 17. The thermostat: open loop against feedback, with an unknown disturbance
# ---------------------------------------------------------------------------------------
dist = 0.5
tt = np.linspace(0, 6, 400)
fig, ax = P.panels(figsize=(3.4, 2.5))
ax.plot(tt, (1 + dist) * (1 - np.exp(-tt)), color=P.GREY, label="open loop $u\\equiv1$")
for L, col in zip([2, 10], [P.ORANGE, P.BLUE]):
    ax.plot(tt, (1 + L + dist) / (1 + L) * (1 - np.exp(-(1 + L) * tt)), color=col,
            label=f"feedback, $L={L}$")
ax.axhline(1, color=P.INK, lw=1, ls="--")
ax.text(5.9, 1.05, "setpoint", ha="right", fontsize=8, color=P.NOTE)
P.label(ax, "$t$", "$x(t)$", title="$x'=-x+u+d$, unknown $d=0.5$", ylim=(0, 1.7),
        legend=dict(loc="lower right", fontsize=7.5))
save(fig, "thermostat")


# ---------------------------------------------------------------------------------------
# 18. Brownian motion: a GIF that plays on its own, the matching vector PDF frames that
#     Beamer's animate package embeds at FPS frames per second, and a poster of the last frame
# ---------------------------------------------------------------------------------------
# Independent N(0, dt) coordinate increments sample standard planar Brownian motion on a
# fine time grid; the displayed segments interpolate the samples.
rng_bm = np.random.default_rng(42)
steps, duration = 1200, 6.0
increments = rng_bm.normal(scale=np.sqrt(duration / steps), size=(steps, 2))
positions = np.vstack([np.zeros(2), np.cumsum(increments, axis=0)])
stops = np.linspace(0, steps, 120, dtype=int)

fig, ax = P.panels(figsize=(3.3, 2.5), dpi=180)
fig.set_layout_engine(None)                # fix the layout and limits so that neither the
fig.subplots_adjust(left=0.17, right=0.97, bottom=0.22, top=0.95)   # axes nor the frame jumps
low, high = positions.min(axis=0), positions.max(axis=0)
centre = (low + high) / 2
radius = 0.56 * np.max(high - low)
P.label(ax, "$X_1(t)$", "$X_2(t)$", equal=True,
        xlim=(centre[0] - radius, centre[0] + radius), ylim=(centre[1] - radius, centre[1] + radius))
ax.locator_params(axis="both", nbins=4)
trace, = ax.plot([], [], color=P.BLUE, lw=1.15)
ax.plot(0, 0, "o", color=P.INK, ms=3.5, zorder=3)
particle, = ax.plot([], [], "o", color=P.ORANGE, ms=6, zorder=4)
clock = ax.text(0.03, 0.96, "", transform=ax.transAxes, va="top", fontsize=9, color=P.NOTE)

frames = []
with PdfPages(OUT / "brownian_motion.pdf") as pdf:
    for stop in stops:
        trace.set_data(positions[:stop + 1, 0], positions[:stop + 1, 1])
        particle.set_data(positions[stop:stop + 1, 0], positions[stop:stop + 1, 1])
        clock.set_text(f"$t={duration * stop / steps:.1f}$")
        pdf.savefig(fig)
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert("RGB"))
    for _ in range(FPS):                   # hold the completed path for a second before replaying;
        pdf.savefig(fig)                   # repeated pages keep the GIF and Beamer timings identical
frames[0].save(OUT / "brownian_motion.gif", save_all=True,
               append_images=frames[1:] + [frames[-1]] * FPS, duration=1000 // FPS, loop=0)
frames[-1].save(OUT / "brownian_motion_poster.png")
if PREVIEW:
    frames[-1].save(PREVIEW / "brownian_motion.png")
plt.close(fig)
print("wrote brownian_motion (pdf frames, gif, poster)")
