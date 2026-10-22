"""
Axisymmetric supersonic cone flow solver — MacCormack predictor/corrector
scheme in generalized curvilinear coordinates.

"""


import jax
import jax.numpy as jnp
from functools import partial

jax.config.update("jax_enable_x64", True)  


## Numerical Differentiation approximation function (Physical Coordinates).

def ddxi_central(f, dxi):
    """Central diff along axis 0 (xi/i), 2nd-order one-sided at the ends.
    Matches Derivative_Functions/ddxi_central.m exactly."""
    d_mid = (f[2:, :] - f[:-2, :]) / (2 * dxi)
    d0 = ((4 * f[1, :] - 3 * f[0, :] - f[2, :]) / (2 * dxi))[None, :]
    dL = ((-4 * f[-2, :] + 3 * f[-1, :] + f[-3, :]) / (2 * dxi))[None, :]
    return jnp.concatenate([d0, d_mid, dL], axis=0)


def ddeta_central(f, deta):
    """Central diff along axis 1 (eta/j), 2nd-order one-sided at the ends.
    Matches Derivative_Functions/ddeta_central.m exactly."""
    d_mid = (f[:, 2:] - f[:, :-2]) / (2 * deta)
    d0 = ((-3 * f[:, 0] + 4 * f[:, 1] - f[:, 2]) / (2 * deta))[:, None]
    dL = ((3 * f[:, -1] - 4 * f[:, -2] + f[:, -3]) / (2 * deta))[:, None]
    return jnp.concatenate([d0, d_mid, dL], axis=1)


def ddxi_fwd(f, dxi):
    """Forward diff along axis 0, backward at the last row.
    Matches Derivative_Functions/ddxi_fwd.m (internal name ddx_fwd)."""
    d_main = (f[1:, :] - f[:-1, :]) / dxi          # rows 0..nx-2
    d_last = ((f[-1, :] - f[-2, :]) / dxi)[None, :]  # row nx-1
    return jnp.concatenate([d_main, d_last], axis=0)


def ddxi_bwd(f, dxi):
    """Backward diff along axis 0, forward at the first row.
    Matches Derivative_Functions/ddxi_bwd.m (internal name ddx_bwd)."""
    d_first = ((f[1, :] - f[0, :]) / dxi)[None, :]   # row 0
    d_main = (f[1:, :] - f[:-1, :]) / dxi            # rows 1..nx-1
    return jnp.concatenate([d_first, d_main], axis=0)


def deta_fwd(f, deta):
    """Forward diff along axis 1, backward at the last column.
    Matches Derivative_Functions/deta_fwd.m (internal name ddy_fwd)."""
    d_main = (f[:, 1:] - f[:, :-1]) / deta           # cols 0..ny-2
    d_last = ((f[:, -1] - f[:, -2]) / deta)[:, None]  # col ny-1
    return jnp.concatenate([d_main, d_last], axis=1)


def deta_bwd(f, deta):
    """Backward diff along axis 1, forward at the first column.
    Matches Derivative_Functions/deta_bwd.m (internal name ddy_bwd)."""
    d_first = ((f[:, 1] - f[:, 0]) / deta)[:, None]  # col 0
    d_main = (f[:, 1:] - f[:, :-1]) / deta           # cols 1..ny-1
    return jnp.concatenate([d_first, d_main], axis=1)




def compute_metrics(xx, yy, dxi, deta):
    dxdxi = ddxi_central(xx, dxi)
    dxdeta = ddeta_central(xx, deta)
    dydxi = ddxi_central(yy, dxi)
    dydeta = ddeta_central(yy, deta)
    J = dxdxi * dydeta - dxdeta * dydxi
    xix = dydeta / J
    etax = -dydxi / J
    xiy = -dxdeta / J
    etay = dxdxi / J
    return J, xix, etax, xiy, etay


## Calculating derivatives in computational domain

def ddx_bwd(f, dxi, deta, xix, etax):
    return ddxi_bwd(f, dxi) * xix + deta_bwd(f, deta) * etax


def ddx_fwd(f, dxi, deta, xix, etax):
    return ddxi_fwd(f, dxi) * xix + deta_fwd(f, deta) * etax


def ddx_central(f, dxi, deta, xix, etax):
    return ddxi_central(f, dxi) * xix + ddeta_central(f, deta) * etax


def ddy_bwd(f, dxi, deta, xiy, etay):
    return ddxi_bwd(f, dxi) * xiy + deta_bwd(f, deta) * etay


def ddy_fwd(f, dxi, deta, xiy, etay):
    return ddxi_fwd(f, dxi) * xiy + deta_fwd(f, deta) * etay


def ddy_central(f, dxi, deta, xiy, etay):
    return ddxi_central(f, dxi) * xiy + ddeta_central(f, deta) * etay



# Sutherlands law
def sutherland(T, mu0, S1=110.4, T0=288.15):
    return mu0 * (T / T0) ** 1.5 * (T0 + S1) / (T + S1)

# Converting from primitive to conservative and vice versa

def prim2cons(rho, u, v, T, cv):
    U1 = rho
    U2 = rho * u
    U3 = rho * v
    U4 = rho * (cv * T + 0.5 * (u ** 2 + v ** 2))
    return jnp.stack([U1, U2, U3, U4], axis=0)  # (4, nx, ny)


def cons2prim(U, R, cv):
    rho = U[0]
    u = U[1] / rho
    v = U[2] / rho
    T = (U[3] - 0.5 * rho * (u ** 2 + v ** 2)) / (rho * cv)
    p = rho * R * T
    return rho, u, v, T, p



# Axisymetrical continuity equation in cylindrical coordinate
def _axisym_div(dudx, dvdy, v, Rad):
    """Axisymmetric velocity divergence du/dx + dv/dy + v/y (y = Rad).
    Shared by flux_E, flux_F, and axisym_radial_source so tau_xx, tau_yy,
    and the geometric source term can never drift out of consistency with
    each other."""
    return dudx + dvdy + v / Rad


def flux_E(rho, u, v, T, P, cv, mu, K, dudxi, dvdeta, dTdxi, dudeta, dvdxi, Rad):
    div = _axisym_div(dudxi, dvdeta, v, Rad)
    tau_xx = 2 * mu * (dudxi - div / 3.0)
    tau_xy = mu * (dudeta + dvdxi)
    E1 = rho * u
    E2 = rho * u * u + P - tau_xx
    E3 = rho * u * v - tau_xy
    E4 = ((rho * (cv * T + 0.5 * (u ** 2 + v ** 2)) + P) * u
          - u * tau_xx - v * tau_xy - K * dTdxi)
    return jnp.stack([E1, E2, E3, E4], axis=0)


def flux_F(rho, u, v, T, P, cv, mu, K, dudeta, dvdxi, dvdeta, dudxi, dTdeta, Rad):
    div = _axisym_div(dudxi, dvdeta, v, Rad)
    tau_xy = mu * (dudeta + dvdxi)
    tau_yy = 2 * mu * (dvdeta - div / 3.0)
    F1 = rho * v
    F2 = rho * u * v - tau_xy
    F3 = rho * v * v + P - tau_yy
    F4 = ((rho * (cv * T + 0.5 * (u ** 2 + v ** 2)) + P) * v
          - u * tau_xy - v * tau_yy - K * dTdeta)
    return jnp.stack([F1, F2, F3, F4], axis=0)

# Additional axisymmetric source term due to axisymmertric physical coordinates.
def axisym_radial_source(v, P, mu, dudx, dvdy, Rad):

    div = _axisym_div(dudx, dvdy, v, Rad)
    tau_yy = 2 * mu * (dvdy - div / 3.0)
    tau_theta_theta = 2 * mu * (v / Rad - div / 3.0)
    return P - tau_theta_theta  #P - tau_yy + tau_theta_theta
 
# Boundary conditions

def apply_bcs(rho, u, v, T, P, R, Uinf, Tinf, Pinf):
    nx = u.shape[0]

    # outflow (2nd-order extrapolation), last i-row
    u = u.at[-1, :].set(2 * u[-2, :] - u[-3, :])
    v = v.at[-1, :].set(2 * v[-2, :] - v[-3, :])
    T = T.at[-1, :].set(2 * T[-2, :] - T[-3, :])
    P = P.at[-1, :].set(2 * P[-2, :] - P[-3, :])

    # wall, j = 0
    u = u.at[:, 0].set(0.0)
    v = v.at[:, 0].set(0.0)
    T = T.at[:, 0].set(Tinf)
    P = P.at[:, 0].set(2 * P[:, 1] - P[:, 2])

    # far field: i = 0 and j = ny-1
    u = u.at[0, :].set(Uinf)
    u = u.at[:, -1].set(Uinf)
    v = v.at[0, :].set(0.0)
    v = v.at[:, -1].set(0.0)
    P = P.at[0, :].set(Pinf)
    P = P.at[:, -1].set(Pinf)
    T = T.at[0, :].set(Tinf)
    T = T.at[:, -1].set(Tinf)

    # leading edge (0,0)
    u = u.at[0, 0].set(0.0)
    v = v.at[0, 0].set(0.0)
    P = P.at[0, 0].set(Pinf)
    T = T.at[0, 0].set(Tinf)

    rho = P / (R * T)
    return rho, u, v, T, P


def apply_bcs_init(rho, u, v, T, P, R, Uinf, Tinf, Pinf):
   
    return apply_bcs(rho, u, v, T, P, R, Uinf, Tinf, Pinf)


# MacCormack Scheme (predictor + corrector) 


def make_step_fn(dxi, deta, dt, cp, cv, R, Prandtl, mu_o, Rad, Y, J,
                  xix, etax, xiy, etay, Uinf, Tinf, Pinf):
    

    JRad = J * Rad

    def predictor(rho, u, v, T, P):
        uinit_phys = prim2cons(rho, u, v, T, cv)      # physical, pre-weight
        uinit_w = uinit_phys * JRad[None, :, :]        # J*Rad-weighted
        U = uinit_w

        mu = sutherland(T, mu_o)
        K = (cp / Prandtl) * mu

        dudxi = ddx_bwd(u, dxi, deta, xix, etax)
        dvdeta = ddy_central(v, dxi, deta, xiy, etay)
        dTdxi = ddx_bwd(T, dxi, deta, xix, etax)
        dudeta = ddy_central(u, dxi, deta, xiy, etay)
        dvdxi = ddx_bwd(v, dxi, deta, xix, etax)
        Enew = flux_E(rho, u, v, T, P, cv, mu, K,
                      dudxi, dvdeta, dTdxi, dudeta, dvdxi, Rad)

        dudeta_F = ddy_bwd(u, dxi, deta, xiy, etay)
        dvdxi_F = ddx_central(v, dxi, deta, xix, etax)
        dvdeta_F = ddy_bwd(v, dxi, deta, xiy, etay)
        dudxi_F = ddx_central(u, dxi, deta, xix, etax)
        dTdeta = ddy_bwd(T, dxi, deta, xiy, etay)
        Fnew = flux_F(rho, u, v, T, P, cv, mu, K,
                      dudeta_F, dvdxi_F, dvdeta_F, dudxi_F, dTdeta, Rad)

        
        S2 = axisym_radial_source(v, P, mu, dudxi_F, dvdeta_F, Rad)
        S = jnp.zeros_like(U).at[2].set(S2)

        E_eps = J[None] * Enew * xix[None] * Rad[None] + \
                J[None] * Fnew * xiy[None] * Y[None]
        F_eps = J[None] * Enew * etax[None] * Rad[None] + \
                J[None] * Fnew * etay[None] * Y[None]

        dEdx = jnp.stack([ddxi_fwd(E_eps[k], dxi) for k in range(4)], axis=0)
        dFdy = jnp.stack([deta_fwd(F_eps[k], deta) for k in range(4)], axis=0)

        Ubar = U - dt * (dEdx + dFdy) + dt * S # Source termmmmmmmmmmmmmmmmmmmmm
        Ubar = Ubar / JRad[None, :, :]

        rho_b, u_b, v_b, T_b, P_b = cons2prim(Ubar, R, cv)
        rho_b, u_b, v_b, T_b, P_b = apply_bcs(
            rho_b, u_b, v_b, T_b, P_b, R, Uinf, Tinf, Pinf)

        Utmp = prim2cons(rho_b, u_b, v_b, T_b, cv) * JRad[None, :, :]
        return Utmp, uinit_w, uinit_phys, rho_b, u_b, v_b, T_b, P_b

    def corrector(Ubar, uinit_w, rho, u, v, T, P):
        mu = sutherland(T, mu_o)
        K = (cp / Prandtl) * mu

        dudxi = ddx_fwd(u, dxi, deta, xix, etax)
        dvdeta = ddy_central(v, dxi, deta, xiy, etay)
        dTdxi = ddx_fwd(T, dxi, deta, xix, etax)
        dudeta = ddy_central(u, dxi, deta, xiy, etay)
        dvdxi = ddx_fwd(v, dxi, deta, xix, etax)
        Enew = flux_E(rho, u, v, T, P, cv, mu, K,
                      dudxi, dvdeta, dTdxi, dudeta, dvdxi, Rad)

        dudeta_F = ddy_fwd(u, dxi, deta, xiy, etay)
        dvdxi_F = ddx_central(v, dxi, deta, xix, etax)
        dvdeta_F = ddy_fwd(v, dxi, deta, xiy, etay)
        dudxi_F = ddx_central(u, dxi, deta, xix, etax)
        dTdeta = ddy_fwd(T, dxi, deta, xiy, etay)
        Fnew = flux_F(rho, u, v, T, P, cv, mu, K,
                      dudeta_F, dvdxi_F, dvdeta_F, dudxi_F, dTdeta, Rad)

        S2 = axisym_radial_source(v, P, mu, dudxi_F, dvdeta_F, Rad)
        S = jnp.zeros_like(Ubar).at[2].set(S2)

        E_eps = J[None] * Enew * xix[None] * Rad[None] + \
                J[None] * Fnew * xiy[None] * Y[None]
        F_eps = J[None] * Enew * etax[None] * Rad[None] + \
                J[None] * Fnew * etay[None] * Y[None]

        dEdx = jnp.stack([ddxi_bwd(E_eps[k], dxi) for k in range(4)], axis=0)
        dFdy = jnp.stack([deta_bwd(F_eps[k], deta) for k in range(4)], axis=0)

        U = 0.5 * ((Ubar + uinit_w) - dt * dEdx - dt * dFdy + dt * S)
        U_phys = U / JRad[None, :, :]   # physical-space U, BEFORE cons2prim/BCs

        rho_n, u_n, v_n, T_n, P_n = cons2prim(U_phys, R, cv)
        rho_n, u_n, v_n, T_n, P_n = apply_bcs(
            rho_n, u_n, v_n, T_n, P_n, R, Uinf, Tinf, Pinf)
        return rho_n, u_n, v_n, T_n, P_n, U_phys

    @jax.jit
    def step(rho, u, v, T, P):
        Ubar, uinit_w, uinit_phys, rho_b, u_b, v_b, T_b, P_b = predictor(
            rho, u, v, T, P)
        rho_n, u_n, v_n, T_n, P_n, U_phys = corrector(
            Ubar, uinit_w, rho_b, u_b, v_b, T_b, P_b) 

        residual = jnp.linalg.norm(uinit_phys[1] - U_phys[1])
        return rho_n, u_n, v_n, T_n, P_n, residual

    return step

# Grid generation (Similar to wedge)
def make_cone_grid(nx=100, ny=100, L=1.0e-5, H=0.8e-5, fix_apex_singularity=True,
                    rad_floor=1.0e-7):
    x = jnp.linspace(0.0, L, nx)
    y = jnp.linspace(0.0, H, ny)
    xx, yy = jnp.meshgrid(x, y, indexing="ij")   # xx[i,j]=x[i], yy[i,j]=y[j]

    yy1 = yy
    xx1 = xx
    x1 = float(x[0])  # = 0

    n = 0.1763
    j = 0
    while n > 0:
        col = n * xx[:, j] + y[j] - n * x1
        yy1 = yy1.at[:, j].set(col)
        n -= 0.0040
        j += 1

    
    nn = -0.2
    xx1 = xx1.at[1:-1, 0:3].set(
        nn * yy1[1:-1, 0:3] + xx[1:-1, 0:1] - nn * yy1[1:-1, 0:1]
    )

    
    for i in range(1, nx):  # MATLAB i=2:nx (1-based) -> 0-based 1..nx-1
        m = (xx1[i, 2] - xx1[i, -1]) / (yy1[i, 2] - yy1[i, -1])
        xx1 = xx1.at[i, 2:].set(m * yy1[i, 2:] + xx1[i, 2] - m * yy1[i, 2])

    Rad = yy1 / jnp.cos(0.0)   # == yy1
    if fix_apex_singularity:
        Rad = jnp.maximum(Rad, rad_floor)
    Z = xx1                   # == xx1, kept only for naming parity
    return xx1, yy1, Rad, Z


def make_placeholder_grid(nx, ny, L, H, r0_frac=0.02):
    """Simple algebraic fallback grid — only used if you call run_solver
    without a real grid and don't want make_cone_grid's shearing. Prefer
    make_cone_grid (the actual ported grid generator) for real results.
    """
    xi = jnp.linspace(0.0, L, nx)
    r0 = r0_frac * H
    eta = jnp.linspace(r0, H, ny)
    xx1, yy1 = jnp.meshgrid(xi, eta, indexing="ij")
    return xx1, yy1


def run_solver(nx=100, ny=100, L=1.0e-5, H=0.8e-5, M=5, iters=1500,
               dt=1e-11, xx1=None, yy1=None, plot_every=100,
               verbose_every=100, fix_apex_singularity=True,
               rad_floor=1.0e-7, live_plot=False):
    Tinf, Pinf = 288.15, 101300.0
    mu_o, Prandtl = 1.735e-5, 0.71
    cp, cv = 1005.0, 718.0
    gamma = cp / cv
    R = 287.0
    Uinf = M * jnp.sqrt(gamma * R * Tinf)

    xi = jnp.linspace(0.0, L, nx)
    eta = jnp.linspace(0.0, H, ny)
    dxi = xi[1] - xi[0]
    deta = eta[1] - eta[0]

    if xx1 is None or yy1 is None:
        xx1, yy1, Rad, _Z = make_cone_grid(
            nx, ny, L, H, fix_apex_singularity=fix_apex_singularity,
            rad_floor=rad_floor)
    else:
        Rad = yy1  # confirmed: Grid_genration.m sets Rad = yy1/cos(0) = yy1
        if fix_apex_singularity:
            Rad = jnp.maximum(Rad, rad_floor)
    Y = yy1

    J, xix, etax, xiy, etay = compute_metrics(xx1, yy1, dxi, deta)

    
    u = jnp.zeros((nx, ny))
    v = jnp.zeros((nx, ny))
    T = jnp.zeros((nx, ny))
    P = jnp.zeros((nx, ny))
    u = u.at[1:, 1:-1].set(Uinf)
    P = P.at[1:, 1:-1].set(Pinf)
    T = T.at[1:, 1:-1].set(Tinf)
    rho, u, v, T, P = apply_bcs(jnp.zeros((nx, ny)), u, v, T, P,
                                 R, Uinf, Tinf, Pinf)

    step = make_step_fn(dxi, deta, dt, cp, cv, R, Prandtl, mu_o, Rad, Y, J,
                         xix, etax, xiy, etay, Uinf, Tinf, Pinf)

    live_fig = live_axs = None
    if live_plot:
        live_fig, live_axs = _init_live_figure()

    residuals = []
    snapshots = []
    for n in range(iters):
        rho, u, v, T, P, residual = step(rho, u, v, T, P)
        residuals.append(float(residual))
        if verbose_every and n % verbose_every == 0:
            print(f"iter {n:5d}  residual = {float(residual):.6e}")
        if plot_every and n % plot_every == 0:
            snapshots.append((n, jnp.array(u), jnp.array(v),
                               jnp.array(rho), jnp.array(P), jnp.array(T)))
            if live_plot:
                _update_live_figure(live_fig, live_axs, xx1, yy1,
                                     u, v, rho, P, T, n)

    if live_plot:

        _update_live_figure(live_fig, live_axs, xx1, yy1, u, v, rho, P, T,
                             iters - 1)

    mach = u / jnp.sqrt(gamma * R * T)
    return dict(rho=rho, u=u, v=v, T=T, P=P, mach=mach,
                residuals=jnp.array(residuals), xx1=xx1, yy1=yy1,
                snapshots=snapshots, deta=deta, M=M)




_FIELD_SPECS = [
    ("u-velocity", "u"), ("v-velocity", "v"), ("Density", "rho"),
    ("Pressure", "P"), ("Internal Energy", "e"), ("Temperature", "T"),
]


def _in_jupyter():
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False


def _init_live_figure():
    import matplotlib.pyplot as plt
    if _in_jupyter():
        plt.ioff()  # don't let a figure auto-display the moment it's created
    fig, axs = plt.subplots(3, 2, figsize=(10, 12))
    if not _in_jupyter():
        plt.ion()
        fig.show()
    return fig, axs


def _update_live_figure(fig, axs, xx1, yy1, u, v, rho, P, T, n, out_dir=None):
    import numpy as np

    xx1n, yy1n = np.array(xx1), np.array(yy1)
    fields = dict(u=np.array(u), v=np.array(v), rho=np.array(rho),
                  P=np.array(P), e=718.0 * np.array(T), T=np.array(T))

    for ax, (title, key) in zip(axs.ravel(), _FIELD_SPECS):
        ax.clear()
        ax.pcolormesh(xx1n, yy1n, fields[key], cmap="jet", shading="auto")
        ax.set_title(f"{title} (iter {n})", fontsize=12)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_aspect("equal")
    fig.tight_layout()

    if _in_jupyter():
        import io
        from IPython.display import display, clear_output, Image
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        clear_output(wait=True)
        display(Image(data=buf.getvalue()))
        buf.close()
    else:
        fig.canvas.draw()
        fig.canvas.flush_events()
        import matplotlib.pyplot as plt
        plt.pause(0.001)  # lets a GUI event loop actually repaint the window



def plot_results(results, out_dir="."):
    import os
    import matplotlib.pyplot as plt   # uses whatever backend is already active
    import numpy as np

    os.makedirs(out_dir, exist_ok=True)

    xx1, yy1 = np.array(results["xx1"]), np.array(results["yy1"])
    M = results["M"]

    # figure(1)-style final field snapshot
    fields = [("u-velocity", results["u"]), ("v-velocity", results["v"]),
              ("Density", results["rho"]), ("Pressure", results["P"]),
              ("Internal Energy", 718.0 * results["T"]),
              ("Temperature", results["T"])]
    fig, axs = plt.subplots(3, 2, figsize=(10, 12))
    for ax, (title, field) in zip(axs.ravel(), fields):
        pc = ax.pcolormesh(xx1, yy1, np.array(field), cmap="jet", shading="auto")
        fig.colorbar(pc, ax=ax)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/fields.png", dpi=600,bbox_inches="tight")
    plt.close(fig)

    
    fig, ax = plt.subplots()
    ax.plot(np.array(results["residuals"]))
    ax.set_xlabel("Number Of Time Steps")
    ax.set_ylabel("U-velocity residual")
    ax.set_title(f"Convergence plot of velocity for M={M}")
    fig.savefig(f"{out_dir}/residual.png", dpi=600,bbox_inches="tight")
    plt.close(fig)

    
    fig, ax = plt.subplots()
    pc = ax.pcolormesh(xx1, yy1, np.array(results["mach"]), cmap="jet", shading="auto")
    fig.colorbar(pc, ax=ax)
    ax.set_aspect("equal")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("Mach number")
    fig.savefig(f"{out_dir}/mach.png", dpi=600,bbox_inches="tight")
    plt.close(fig)

  
    deta = float(results["deta"])
    inde = int(np.floor(1e-5 / (deta * 4))) + 1
    inde = min(inde, yy1.shape[1] - 1)
    for title, ylabel, field, fname in [
        ("Mach number variation along x-direction", "Machnumber",
         results["mach"], "mach_line.png"),
        ("Pressure variation along x-direction", "Pressure",
         results["P"], "pressure_line.png"),
        ("Temperature variation along x-direction", "Temperature",
         results["T"], "temperature_line.png"),
    ]:
        fig, ax = plt.subplots()
        ax.plot(yy1[:, inde], np.array(field)[:, inde])
        ax.set_xlabel("x"); ax.set_ylabel(ylabel)
        ax.set_title(f"{title} for M={M}")
        fig.savefig(f"{out_dir}/{fname}", dpi=600,bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    results = run_solver(nx=100, ny=100, iters=3000, live_plot=False)
    plot_results(results, out_dir="./outputs")