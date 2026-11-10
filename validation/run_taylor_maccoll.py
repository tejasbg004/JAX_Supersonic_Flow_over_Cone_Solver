"""Inviscid check of the cone solver against the Taylor-Maccoll solution (Mach 5, 10-degree cone).

Runs the solver's own MacCormack scheme (flux, metric and source functions imported from
JAX_Supersonic_Cone_Solver.py) with viscosity switched off, a slip wall, and light
pressure-switched artificial dissipation, on a 200 x 200 grid. Then compares the surface
pressure and shock angle with the Taylor-Maccoll solution and saves
figures/taylor_maccoll_validation.png.

    python validation/run_taylor_maccoll.py            # full run, 36,000 steps
    python validation/run_taylor_maccoll.py --quick    # short smoke test
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import harness as h
import references as rf

quick = "--quick" in sys.argv
r = h.run(M=5.0, nx=200, ny=200, viscous=False, slip=True, source_J=True, diss=0.2,
          dt=6e-12, iters=300 if quick else 36000, report=0 if quick else 2000)

cone_deg = np.degrees(np.arctan(h.TAN_C))
tm = rf.taylor_maccoll(r["M"], cone_deg)
s = np.hypot(r["xx"][:, 0] - r["xx"][0, 0], r["yy"][:, 0] - r["yy"][0, 0])
pw = r["P"][:, 1] / r["Pinf"]                  # surface pressure (first off-wall row)
shock, _, _ = h.shock_angle(r, (0.3, 0.95))

aft = s / s[-1] > 0.6
print(f"Taylor-Maccoll: shock {tm['shock_deg']:.2f} deg, surface p/pinf {tm['pc']:.3f}")
print(f"Solver:         shock {shock:.2f} deg, surface p/pinf {pw[aft].mean():.3f} "
      f"(max deviation {100 * np.abs(pw[aft] / tm['pc'] - 1).max():.2f}% for s/L > 0.6)")

fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
ax[0].plot(s / s[-1], pw, color="tab:blue", lw=2, label="JAX solver (inviscid, 200×200 grid)")
ax[0].axhline(tm["pc"], color="k", ls=":", lw=1.5, label=f"Taylor–Maccoll ({tm['pc']:.2f})")
ax[0].axhspan(0.99 * tm["pc"], 1.01 * tm["pc"], color="0.85", label="±1%")
ax[0].set(xlabel="distance along cone, s/L", ylabel="surface pressure p / p∞",
          title="Mach 5, 10° cone: surface pressure", ylim=(0, 1.6 * tm["pc"]))
ax[0].legend(fontsize=9, loc="lower right")
pc = ax[1].pcolormesh(r["xx"] * 1e6, r["yy"] * 1e6, r["P"] / r["Pinf"], cmap="viridis", shading="auto")
xl = np.linspace(0, r["xx"].max() * 1e6, 50)
ax[1].plot(xl, xl * np.tan(np.radians(tm["shock_deg"])), "w--", lw=1.5,
           label=f"Taylor–Maccoll shock, {tm['shock_deg']:.1f}°  (solver: {shock:.1f}°)")
ax[1].plot(xl, xl * h.TAN_C, "r-", lw=1.5, label="cone surface, 10°")
ax[1].set(xlabel="x (µm)", ylabel="r (µm)", title="Pressure field p / p∞")
ax[1].legend(fontsize=9, loc="upper left")
plt.colorbar(pc, ax=ax[1])
plt.tight_layout()
out = os.path.join(HERE, "..", "figures", "taylor_maccoll_validation_quick.png" if quick else "taylor_maccoll_validation.png")
plt.savefig(out, dpi=130)
print("saved", os.path.normpath(out))
