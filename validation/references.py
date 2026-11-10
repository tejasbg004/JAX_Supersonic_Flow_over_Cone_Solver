"""Analytical references: Taylor-Maccoll sharp-cone solution and
Eckert reference-temperature laminar heat flux with the Mangler (sqrt 3) cone factor."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

CP, CV, R, PR = 1005.0, 718.0, 287.0, 0.71
G = CP / CV


def sutherland(T, mu0=1.735e-5, S1=110.4, T0=288.15):  # same law and constants as the solver
    return mu0 * (T / T0) ** 1.5 * (T0 + S1) / (T + S1)


def taylor_maccoll(M, cone_deg, g=G):
    def rhs(t, y):
        V, dV = y
        A = (g - 1) / 2 * (1 - V ** 2 - dV ** 2)
        return [dV, (dV ** 2 * V - A * (2 * V + dV / np.tan(t))) / (A - dV ** 2)]

    def post(bs):
        b = np.radians(bs)
        Mn1 = M * np.sin(b)
        Mn2 = np.sqrt((1 + (g - 1) / 2 * Mn1 ** 2) / (g * Mn1 ** 2 - (g - 1) / 2))
        d = np.arctan(2 / np.tan(b) * (M ** 2 * np.sin(b) ** 2 - 1) / (M ** 2 * (g + np.cos(2 * b)) + 2))
        return b, Mn1, Mn2, d, Mn2 / np.sin(b - d)

    def cone(bs):
        b, Mn1, Mn2, d, M2 = post(bs)
        V = (2 / ((g - 1) * M2 ** 2) + 1) ** -0.5
        ev = lambda t, y: y[1]; ev.terminal = True
        s = solve_ivp(rhs, [b, 1e-3], [V * np.cos(b - d), -V * np.sin(b - d)],
                      events=ev, rtol=1e-10, atol=1e-12, max_step=1e-3)
        return np.degrees(s.t_events[0][0]), s.y_events[0][0][0]

    bs = brentq(lambda x: cone(x)[0] - cone_deg, np.degrees(np.arcsin(1 / M)) + 0.3, 45)
    _, Vc = cone(bs)
    b, Mn1, Mn2, d, M2 = post(bs)
    p2 = 1 + 2 * g / (g + 1) * (Mn1 ** 2 - 1)
    Mc = np.sqrt(2 / (g - 1) * Vc ** 2 / (1 - Vc ** 2))
    pc = p2 * ((1 + (g - 1) / 2 * M2 ** 2) / (1 + (g - 1) / 2 * Mc ** 2)) ** (g / (g - 1))
    T0_over_T = 1 + (g - 1) / 2 * M ** 2
    Tc = T0_over_T / (1 + (g - 1) / 2 * Mc ** 2)  # Tc/Tinf
    return dict(shock_deg=bs, Mc=Mc, pc=pc, Tc=Tc)


def eckert_cone_heat_flux(x, M, Tinf, Pinf, Twall, cone_deg):
    """Laminar heat flux q_w(x) [W/m^2] on a sharp cone: flat-plate Eckert reference-temperature
    result at Taylor-Maccoll edge conditions, times sqrt(3) (Mangler transformation)."""
    tm = taylor_maccoll(M, cone_deg)
    Te, pe, Me = tm["Tc"] * Tinf, tm["pc"] * Pinf, tm["Mc"]
    ue = Me * np.sqrt(G * R * Te)
    Taw = Te * (1 + np.sqrt(PR) * (G - 1) / 2 * Me ** 2)
    Ts = Te * (1 + 0.032 * Me ** 2 + 0.58 * (Twall / Te - 1))
    rs, mus = pe / (R * Ts), sutherland(Ts)
    Re = rs * ue * x / mus
    St = 0.332 / np.sqrt(Re) * PR ** (-2 / 3)
    return np.sqrt(3) * rs * ue * CP * (Taw - Twall) * St
