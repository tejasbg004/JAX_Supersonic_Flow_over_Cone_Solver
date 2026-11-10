"""Validation harness for Tejas's JAX cone solver.

Reuses the solver's own derivative, flux, metric, grid and source functions
(imported unchanged) and rebuilds the MacCormack step with three switches:
  source_J   : multiply the axisymmetric source term by the grid Jacobian J
  slip_wall  : inviscid slip wall (velocity tangent to cone, adiabatic) instead of no-slip
  diss       : pressure-switched 2nd-order artificial dissipation coefficient
Freestream conditions and geometry can be changed for the Tracy case.
"""
import sys, time
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import JAX_Supersonic_Cone_Solver as S

TAN_C = 0.1763  # cone surface slope used in make_cone_grid (10 deg)


def scaled_cone_grid(nx, ny, L, H):
    """Same construction as make_cone_grid, with the shear decrement and the number of
    skewed near-wall columns scaled so a refined grid has the same shape."""
    x = jnp.linspace(0.0, L, nx); y = jnp.linspace(0.0, H, ny)
    xx, yy = jnp.meshgrid(x, y, indexing="ij")
    yy1, xx1 = yy, xx
    k = (ny - 1) / 99.0
    n, j = 0.1763, 0
    while n > 0:
        yy1 = yy1.at[:, j].set(n * xx[:, j] + y[j]); n -= 0.0040 / k; j += 1
    c = int(round(3 * k)); nn = -0.2 / k
    xx1 = xx1.at[1:-1, 0:c].set(nn * yy1[1:-1, 0:c] + xx[1:-1, 0:1] - nn * yy1[1:-1, 0:1])
    for i in range(1, nx):
        m = (xx1[i, c - 1] - xx1[i, -1]) / (yy1[i, c - 1] - yy1[i, -1])
        xx1 = xx1.at[i, c - 1:].set(m * yy1[i, c - 1:] + xx1[i, c - 1] - m * yy1[i, c - 1])
    return xx1, yy1, jnp.maximum(yy1, 1.0e-7 * L / 1.0e-5), xx1


def clustered_cone_grid(nx, ny, L, top_deg=17.0, h0=None, beta=1.5):
    """Validation-only grid: vertical i-lines, j-lines packed toward the cone wall (tanh),
    top boundary along a ray at top_deg so the domain hugs the shock layer."""
    h0 = 0.03 * L if h0 is None else h0
    x = jnp.linspace(0.0, L, nx)
    eta = jnp.linspace(0.0, 1.0, ny)
    st = 1 - jnp.tanh(beta * (1 - eta)) / jnp.tanh(beta)
    yw = TAN_C * x
    hgt = h0 + x * (np.tan(np.radians(top_deg)) - TAN_C)
    yy = yw[:, None] + hgt[:, None] * st[None, :]
    xx = jnp.broadcast_to(x[:, None], yy.shape)
    return xx, yy


def bcs(rho, u, v, T, P, R, Uinf, Tinf, Pinf, slip, Twall):
    u = u.at[-1, :].set(2 * u[-2, :] - u[-3, :])
    v = v.at[-1, :].set(2 * v[-2, :] - v[-3, :])
    T = T.at[-1, :].set(2 * T[-2, :] - T[-3, :])
    P = P.at[-1, :].set(2 * P[-2, :] - P[-3, :])
    if slip:
        c = 1 / np.sqrt(1 + TAN_C ** 2); s = TAN_C * c
        vt = u[:, 1] * c + v[:, 1] * s
        u = u.at[:, 0].set(vt * c); v = v.at[:, 0].set(vt * s)
        T = T.at[:, 0].set(T[:, 1])
    else:
        u = u.at[:, 0].set(0.0); v = v.at[:, 0].set(0.0)
        T = T.at[:, 0].set(Twall)
    P = P.at[:, 0].set(2 * P[:, 1] - P[:, 2])
    u = u.at[0, :].set(Uinf); u = u.at[:, -1].set(Uinf)
    v = v.at[0, :].set(0.0); v = v.at[:, -1].set(0.0)
    P = P.at[0, :].set(Pinf); P = P.at[:, -1].set(Pinf)
    T = T.at[0, :].set(Tinf); T = T.at[:, -1].set(Tinf)
    u = u.at[0, 0].set(0.0 if not slip else Uinf); v = v.at[0, 0].set(0.0)
    P = P.at[0, 0].set(Pinf); T = T.at[0, 0].set(Tinf)
    return P / (R * T), u, v, T, P


def dissipation(Uc, P, C):
    """Pressure-switched 2nd-order smoothing in both index directions (interior only)."""
    if C == 0:
        return Uc
    out = Uc
    for ax in (1, 2):
        Pp, Pm = jnp.roll(P, -1, ax - 1), jnp.roll(P, 1, ax - 1)
        sw = jnp.abs(Pp - 2 * P + Pm) / (Pp + 2 * P + Pm)
        Up, Um = jnp.roll(Uc, -1, ax), jnp.roll(Uc, 1, ax)
        out = out + C * sw[None] * (Up - 2 * Uc + Um)
    # keep boundary rows/cols untouched (BCs reset them anyway)
    mask = jnp.zeros(P.shape).at[1:-1, 1:-1].set(1.0)
    return Uc + (out - Uc) * mask[None]


def run(M=5.0, Tinf=288.15, Pinf=101300.0, L=1.0e-5, H=0.8e-5, nx=100, ny=100,
        mu_o=1.735e-5, Twall=None, slip=False, source_J=True, diss=0.0,
        dt=1e-11, iters=6000, report=1000, viscous=True, grid=None):
    cp, cv, R, Pr = 1005.0, 718.0, 287.0, 0.71
    g = cp / cv
    Uinf = M * np.sqrt(g * R * Tinf)
    Twall = Tinf if Twall is None else Twall
    if grid is not None:
        xx, yy = grid
        Rad = jnp.maximum(yy, 1e-3 * float(yy[1, 0]))
    else:
        xx, yy, Rad, _ = (S.make_cone_grid(nx, ny, L, H) if (nx, ny) == (100, 100) and L == 1.0e-5
                          else scaled_cone_grid(nx, ny, L, H))
    Y = yy
    dxi, deta = L / (nx - 1), H / (ny - 1)
    J, xix, etax, xiy, etay = S.compute_metrics(xx, yy, dxi, deta)
    JRad = J * Rad
    Jsrc = J if source_J else jnp.ones_like(J)
    mu_fn = (lambda T: S.sutherland(T, mu_o)) if viscous else (lambda T: jnp.zeros_like(T))
    B = lambda *a: bcs(*a, R, Uinf, Tinf, Pinf, slip, Twall)

    def half(rho, u, v, T, P, dX, dY, dE, dF):
        mu = mu_fn(T); K = cp / Pr * mu
        E = S.flux_E(rho, u, v, T, P, cv, mu, K,
                     dX(u), S.ddy_central(v, dxi, deta, xiy, etay), dX(T),
                     S.ddy_central(u, dxi, deta, xiy, etay), dX(v), Rad)
        dvdy = dY(v); dudx_c = S.ddx_central(u, dxi, deta, xix, etax)
        F = S.flux_F(rho, u, v, T, P, cv, mu, K, dY(u),
                     S.ddx_central(v, dxi, deta, xix, etax), dvdy, dudx_c, dY(T), Rad)
        Sr = S.axisym_radial_source(v, P, mu, dudx_c, dvdy, Rad)
        Ee = J[None] * E * xix[None] * Rad[None] + J[None] * F * xiy[None] * Y[None]
        Fe = J[None] * E * etax[None] * Rad[None] + J[None] * F * etay[None] * Y[None]
        res = -(jnp.stack([dE(Ee[k], dxi) for k in range(4)]) +
                jnp.stack([dF(Fe[k], deta) for k in range(4)]))
        return res.at[2].add(Jsrc * Sr)

    bx = lambda f: S.ddx_bwd(f, dxi, deta, xix, etax); fx = lambda f: S.ddx_fwd(f, dxi, deta, xix, etax)
    by = lambda f: S.ddy_bwd(f, dxi, deta, xiy, etay); fy = lambda f: S.ddy_fwd(f, dxi, deta, xiy, etay)

    @jax.jit
    def step(rho, u, v, T, P):
        U0 = S.prim2cons(rho, u, v, T, cv) * JRad[None]
        Ub = U0 + dt * half(rho, u, v, T, P, bx, by, S.ddxi_fwd, S.deta_fwd)
        rb, ub, vb, Tb, Pb = B(*S.cons2prim(Ub / JRad[None], R, cv))
        Ub = S.prim2cons(rb, ub, vb, Tb, cv) * JRad[None]
        Un = 0.5 * (U0 + Ub + dt * half(rb, ub, vb, Tb, Pb, fx, fy, S.ddxi_bwd, S.deta_bwd))
        Uph = Un / JRad[None]
        _, _, _, _, Pt = S.cons2prim(Uph, R, cv)
        Uph = dissipation(Uph, Pt, diss)
        rn, un, vn, Tn, Pn = B(*S.cons2prim(Uph, R, cv))
        resid = jnp.linalg.norm(Uph[1] - U0[1] / JRad) / jnp.linalg.norm(U0[1] / JRad)
        return rn, un, vn, Tn, Pn, resid

    u = jnp.zeros((nx, ny)).at[1:, 1:-1].set(Uinf)
    v = jnp.zeros((nx, ny))
    T = jnp.full((nx, ny), Tinf); P = jnp.full((nx, ny), Pinf)
    rho, u, v, T, P = B(jnp.zeros((nx, ny)), u, v, T, P)
    hist = []
    t0 = time.time()
    for n in range(iters):
        rho, u, v, T, P, r = step(rho, u, v, T, P)
        if n == iters - 2001:
            P_prev, T_prev = np.array(P), np.array(T)
        if n % 50 == 0:
            rr = float(r); hist.append((n, rr))
            if not np.isfinite(rr):
                print("  diverged at iter", n); break
        if report and n % report == 0:
            print(f"  iter {n:6d} resid {float(r):.3e}  ({time.time()-t0:.0f}s)", flush=True)
    if iters > 2001:
        change = float(np.max(np.abs(np.array(P) - P_prev) / np.array(P)))
        rel = np.abs(np.array(P) - P_prev) / np.array(P)
        loc = np.unravel_index(np.argmax(rel), rel.shape)
        print(f"  max relative change in P over the last 2000 steps: {change:.2e} at (i,j)={loc}; "
              f"99th percentile {np.percentile(rel, 99):.1e}; wall-row max {rel[:, :3].max():.1e}")
    return dict(xx=np.array(xx), yy=np.array(yy), rho=np.array(rho), u=np.array(u),
                v=np.array(v), T=np.array(T), P=np.array(P), hist=np.array(hist),
                Pinf=Pinf, Tinf=Tinf, Uinf=Uinf, M=M, L=L, mu_o=mu_o, Twall=Twall,
                dt=dt, iters=iters, slip=slip, source_J=source_J, diss=diss, viscous=viscous)


def shock_angle(res, x_frac=(0.4, 0.95)):
    """Shock position at each x-station = outermost grid line where P exceeds Pinf by 50%
    of the local post-shock rise; fit a line y_s(x) and return its angle from the axis."""
    xx, yy, P, Pinf = res["xx"], res["yy"], res["P"], res["Pinf"]
    nx = P.shape[0]; xs, ys = [], []
    for i in range(int(x_frac[0] * nx), int(x_frac[1] * nx)):
        p = P[i]; pmax = p[1:].max()
        thr = Pinf + 0.5 * (pmax - Pinf)
        idx = np.where(p > thr)[0]
        if len(idx) == 0:
            continue
        j = idx.max()
        if j + 1 >= len(p):
            continue
        f = (p[j] - thr) / (p[j] - p[j + 1])
        xs.append(xx[i, j] + f * (xx[i, j + 1] - xx[i, j]))
        ys.append(yy[i, j] + f * (yy[i, j + 1] - yy[i, j]))
    xs, ys = np.array(xs), np.array(ys)
    slope = np.polyfit(xs, ys, 1)[0]
    return np.degrees(np.arctan(slope)), xs, ys
