from __future__ import annotations
from dataclasses import dataclass, asdict
import math
import numpy as np

@dataclass(frozen=True)
class PhysicsConfig:
    Lx: float = 1.0
    Ly: float = 0.4
    x_n_end: float = 0.42
    x_s_end: float = 0.58
    t_final: float = 1.0
    current_ramp_time: float = 0.05
    I_app: float = 0.12
    eps_n: float = 0.33
    eps_s: float = 0.47
    eps_p: float = 0.3
    D_e_n: float = 0.006
    D_e_s: float = 0.009
    D_e_p: float = 0.005
    kappa_n: float = 1.0
    kappa_s: float = 1.2
    kappa_p: float = 0.9
    sigma_n: float = 20.0
    sigma_p: float = 15.0
    a_n: float = 1.0
    a_p: float = 1.0
    k0_n: float = 0.8
    k0_p: float = 0.7
    beta_bv: float = 3.0
    thermo_coeff: float = 0.15
    chi_ln_ce: float = 0.15
    Csolid_n: float = 0.9
    Csolid_p: float = 0.9
    D_s_n: float = 0.004
    D_s_p: float = 0.003
    gamma_li: float = 0.05
    rho_cp: float = 1.0
    kT_n: float = 0.01
    kT_s: float = 0.006
    kT_p: float = 0.009
    heat_scale: float = 0.05
    entropic_n: float = -0.01
    entropic_p: float = 0.012
    h_top: float = 0.025
    h_bottom: float = 0.015
    h_variation: float = 0.35
    T_amb: float = 1.0
    c_e0: float = 1.0
    theta_n0: float = 0.35
    theta_p0: float = 0.7
    T0: float = 1.0
    bv_argument_guard: float = 20.0
CFG = PhysicsConfig()

def config_dict() -> dict:
    return asdict(CFG)

def current_profile(t, cfg: PhysicsConfig=CFG):
    a = np.asarray(t, dtype=float)
    if cfg.current_ramp_time <= 0:
        return np.zeros_like(a) + cfg.I_app
    return cfg.I_app * np.clip(a / cfg.current_ramp_time, 0.0, 1.0)

def torch_current_profile(t, cfg: PhysicsConfig=CFG):
    import torch
    if cfg.current_ramp_time <= 0:
        return torch.zeros_like(t) + cfg.I_app
    return cfg.I_app * torch.clamp(t / cfg.current_ramp_time, 0.0, 1.0)

def region_layout(nx: int, ny: int, cfg: PhysicsConfig=CFG):
    dx = cfg.Lx / nx
    dy = cfg.Ly / ny
    i_n = int(round(cfg.x_n_end / dx))
    i_p = int(round(cfg.x_s_end / dx))
    i_n = max(1, min(nx - 2, i_n))
    i_p = max(i_n + 1, min(nx - 1, i_p))
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dy
    X, Y = np.meshgrid(x, y)
    mn = np.zeros((ny, nx), dtype=bool)
    mn[:, :i_n] = True
    ms = np.zeros((ny, nx), dtype=bool)
    ms[:, i_n:i_p] = True
    mp = np.zeros((ny, nx), dtype=bool)
    mp[:, i_p:] = True
    return {'dx': dx, 'dy': dy, 'x': x, 'y': y, 'X': X, 'Y': Y, 'i_n': i_n, 'i_p': i_p, 'x_n_face': i_n * dx, 'x_s_face': i_p * dx, 'mn': mn, 'ms': ms, 'mp': mp}

def piecewise_np(layout, nval, sval, pval):
    return np.where(layout['mn'], nval, np.where(layout['ms'], sval, pval))

def epsilon_np(layout, cfg=CFG):
    return piecewise_np(layout, cfg.eps_n, cfg.eps_s, cfg.eps_p)

def D_e_np(layout, c_e, T, cfg=CFG):
    base = piecewise_np(layout, cfg.D_e_n, cfg.D_e_s, cfg.D_e_p)
    return base * np.clip(c_e, 0.2, 2.5) ** 0.15 * np.exp(cfg.thermo_coeff * (T - 1.0))

def kappa_np(layout, c_e, T, cfg=CFG):
    base = piecewise_np(layout, cfg.kappa_n, cfg.kappa_s, cfg.kappa_p)
    return base * np.clip(c_e, 0.2, 2.5) ** 0.5 * np.exp(cfg.thermo_coeff * (T - 1.0))

def kT_np(layout, cfg=CFG):
    return piecewise_np(layout, cfg.kT_n, cfg.kT_s, cfg.kT_p)

def h_top_np(x, cfg=CFG):
    return cfg.h_top * (1.0 + cfg.h_variation * np.cos(2.0 * np.pi * np.asarray(x) / cfg.Lx))

def h_bottom_np(x, cfg=CFG):
    return cfg.h_bottom * (1.0 - 0.5 * cfg.h_variation * np.cos(2.0 * np.pi * np.asarray(x) / cfg.Lx))

def h_top_torch(x, cfg=CFG):
    import torch
    return cfg.h_top * (1.0 + cfg.h_variation * torch.cos(2.0 * math.pi * x / cfg.Lx))

def h_bottom_torch(x, cfg=CFG):
    import torch
    return cfg.h_bottom * (1.0 - 0.5 * cfg.h_variation * torch.cos(2.0 * math.pi * x / cfg.Lx))

def ocp_n_np(theta):
    theta = np.clip(theta, 1e-06, 1.0 - 1e-06)
    return 0.12 + 0.72 * (1.0 - theta) + 0.035 * np.log((1.0 - theta) / theta)

def ocp_p_np(theta):
    theta = np.clip(theta, 1e-06, 1.0 - 1e-06)
    return 4.1 - 0.85 * theta + 0.025 * np.log((1.0 - theta) / theta)

def exchange_current_np(k: str, c_e, theta, T, cfg=CFG):
    theta = np.clip(theta, 1e-06, 1.0 - 1e-06)
    c_e = np.clip(c_e, 1e-08, None)
    k0 = cfg.k0_n if k == 'n' else cfg.k0_p
    return k0 * np.sqrt(c_e * theta * (1.0 - theta)) * np.exp(cfg.thermo_coeff * (T - 1.0))

def faradaic_np(k: str, c_e, theta, phi_s, phi_e, T, cfg=CFG):
    U = ocp_n_np(theta) if k == 'n' else ocp_p_np(theta)
    eta = phi_s - phi_e - U
    i0 = exchange_current_np(k, c_e, theta, T, cfg)
    arg_raw = cfg.beta_bv * eta / np.clip(T, 0.5, 2.0)
    arg = np.clip(arg_raw, -cfg.bv_argument_guard, cfg.bv_argument_guard)
    j = 2.0 * i0 * np.sinh(arg)
    return (j, eta, i0, U, arg_raw)

def particle_flux_factor(k: str, cfg=CFG):
    if k == 'n':
        return cfg.gamma_li * cfg.a_n / (3.0 * cfg.Csolid_n)
    return cfg.gamma_li * cfg.a_p / (3.0 * cfg.Csolid_p)

def ocp_n_torch(theta):
    import torch
    theta = torch.clamp(theta, 1e-06, 1.0 - 1e-06)
    return 0.12 + 0.72 * (1.0 - theta) + 0.035 * torch.log((1.0 - theta) / theta)

def ocp_p_torch(theta):
    import torch
    theta = torch.clamp(theta, 1e-06, 1.0 - 1e-06)
    return 4.1 - 0.85 * theta + 0.025 * torch.log((1.0 - theta) / theta)

def exchange_current_torch(k: str, c_e, theta, T, cfg=CFG):
    import torch
    theta = torch.clamp(theta, 1e-06, 1.0 - 1e-06)
    c_e = torch.clamp(c_e, min=1e-08)
    k0 = cfg.k0_n if k == 'n' else cfg.k0_p
    return k0 * torch.sqrt(c_e * theta * (1.0 - theta)) * torch.exp(cfg.thermo_coeff * (T - 1.0))

def faradaic_torch(k: str, c_e, theta, phi_s, phi_e, T, cfg=CFG):
    import torch
    U = ocp_n_torch(theta) if k == 'n' else ocp_p_torch(theta)
    eta = phi_s - phi_e - U
    i0 = exchange_current_torch(k, c_e, theta, T, cfg)
    arg = cfg.beta_bv * eta / torch.clamp(T, 0.5, 2.0)
    g = cfg.bv_argument_guard
    arg_safe = g * torch.tanh(arg / g)
    j = 2.0 * i0 * torch.sinh(arg_safe)
    return (j, eta, i0, U, arg)

def coeff_torch(region: str, c_e, T, cfg=CFG):
    import torch
    if region == 'n':
        eps, De0, kap0, kT0 = (cfg.eps_n, cfg.D_e_n, cfg.kappa_n, cfg.kT_n)
    elif region == 's':
        eps, De0, kap0, kT0 = (cfg.eps_s, cfg.D_e_s, cfg.kappa_s, cfg.kT_s)
    elif region == 'p':
        eps, De0, kap0, kT0 = (cfg.eps_p, cfg.D_e_p, cfg.kappa_p, cfg.kT_p)
    else:
        raise ValueError(region)
    De = De0 * torch.clamp(c_e, 0.2, 2.5) ** 0.15 * torch.exp(cfg.thermo_coeff * (T - 1.0))
    kap = kap0 * torch.clamp(c_e, 0.2, 2.5) ** 0.5 * torch.exp(cfg.thermo_coeff * (T - 1.0))
    return (eps, De, kap, kT0)
