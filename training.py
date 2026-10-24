from __future__ import annotations
from pathlib import Path
import argparse, csv, json, random, time, math
import numpy as np
import torch
from physics import CFG, config_dict, coeff_torch, particle_flux_factor, torch_current_profile, h_top_torch, h_bottom_torch
from models import build_model

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def grad(u, z):
    return torch.autograd.grad(u, z, torch.ones_like(u), create_graph=True, retain_graph=True)[0]

def div_xy(fx, fy, z):
    return grad(fx, z)[:, 0:1] + grad(fy, z)[:, 1:2]

def nmse_residual(x, scale):
    s = max(float(scale), 1e-12)
    return torch.mean((x / s) ** 2)

def region_length(k):
    if k == 'n':
        return CFG.x_n_end
    if k == 's':
        return CFG.x_s_end - CFG.x_n_end
    if k == 'p':
        return CFG.Lx - CFG.x_s_end
    raise ValueError(k)

def scales():
    I = max(abs(CFG.I_app), 1e-08)
    Lmin = min(region_length('n'), region_length('p'))
    return {'I': I, 'ce_rate': max(CFG.gamma_li * I / Lmin, 0.0001), 'heat_rate': max(CFG.heat_scale * I * 0.15 / Lmin, 0.0002), 'particle_rate': max(CFG.gamma_li * I / (min(CFG.Csolid_n, CFG.Csolid_p) * Lmin), 0.0001), 'ce_flux': max(CFG.gamma_li * I, 1e-05), 'heat_flux': max(5e-05, CFG.heat_scale * I * 0.01), 'phi': 0.1}
SCALES = scales()

def sample_time(n, device, dtype, early_fraction=0.4):
    u = torch.rand(n, 1, device=device, dtype=dtype)
    all_t = CFG.t_final * torch.rand(n, 1, device=device, dtype=dtype)
    early_end = min(CFG.t_final, max(2.0 * CFG.current_ramp_time, 0.1 * CFG.t_final))
    early_t = early_end * torch.rand(n, 1, device=device, dtype=dtype)
    return torch.where(u < early_fraction, early_t, all_t)

def sample_region(n, k, device, dtype, fixed_t=None):
    xy = torch.rand(n, 2, device=device, dtype=dtype)
    if k == 'n':
        x0, x1 = (0.0, CFG.x_n_end)
    elif k == 's':
        x0, x1 = (CFG.x_n_end, CFG.x_s_end)
    elif k == 'p':
        x0, x1 = (CFG.x_s_end, CFG.Lx)
    else:
        raise ValueError(k)
    x = x0 + (x1 - x0) * xy[:, 0:1]
    y = CFG.Ly * xy[:, 1:2]
    t = sample_time(n, device, dtype) if fixed_t is None else torch.zeros(n, 1, device=device, dtype=dtype) + fixed_t
    return torch.cat([x, y, t], 1).requires_grad_(True)

def sample_plane(n, x=None, y=None, device='cpu', dtype=torch.float32, fixed_t=None):
    xx = CFG.Lx * torch.rand(n, 1, device=device, dtype=dtype)
    yy = CFG.Ly * torch.rand(n, 1, device=device, dtype=dtype)
    tt = sample_time(n, device, dtype) if fixed_t is None else torch.zeros(n, 1, device=device, dtype=dtype) + fixed_t
    if x is not None:
        xx = torch.zeros_like(xx) + float(x)
    if y is not None:
        yy = torch.zeros_like(yy) + float(y)
    return torch.cat([xx, yy, tt], 1).requires_grad_(True)

def macro_physics(model, z, k):
    f = model.region(z, k)
    ce, pe, T = (f['ce'], f['phi_e'], f['T'])
    eps, De, kap, kT = coeff_torch(k, ce, T)
    gc = grad(ce, z)
    gp = grad(pe, z)
    gT = grad(T, z)
    diffx = De * gc[:, 0:1]
    diffy = De * gc[:, 1:2]
    iex = -kap * gp[:, 0:1] + CFG.chi_ln_ce * kap * gc[:, 0:1] / ce
    iey = -kap * gp[:, 1:2] + CFG.chi_ln_ce * kap * gc[:, 1:2] / ce
    out = {**f, 'eps': eps, 'De': De, 'kap': kap, 'kT': kT, 'gc': gc, 'gpe': gp, 'gT': gT, 'diffx': diffx, 'diffy': diffy, 'iex': iex, 'iey': iey}
    if k in ('n', 'p'):
        gps = grad(f['phi_s'], z)
        sigma = CFG.sigma_n if k == 'n' else CFG.sigma_p
        out.update({'gps': gps, 'sigma': sigma, 'isx': -sigma * gps[:, 0:1], 'isy': -sigma * gps[:, 1:2]})
    return out

def stratified_r(n, device, dtype):
    u = torch.rand(n, 1, device=device, dtype=dtype)
    r = (torch.arange(n, device=device, dtype=dtype).reshape(-1, 1) + u) / float(n)
    perm = torch.randperm(n, device=device)
    return r[perm]

def pde_losses(model, n, device, dtype):
    comp = {}
    for k in ('n', 's', 'p'):
        z = sample_region(n, k, device, dtype)
        f = macro_physics(model, z, k)
        source = 0.0
        if k == 'n':
            source = CFG.a_n * f['j']
        elif k == 'p':
            source = CFG.a_p * f['j']
        rce = f['eps'] * f['gc'][:, 2:3] - div_xy(f['diffx'], f['diffy'], z) - CFG.gamma_li * source
        rqe = div_xy(f['iex'], f['iey'], z) - source
        qohme = (f['iex'] ** 2 + f['iey'] ** 2) / f['kap']
        Lk = region_length(k)
        charge_scale = SCALES['I'] / Lk
        It = torch_current_profile(z[:, 2:3])
        if k in ('n', 'p'):
            rqs = div_xy(f['isx'], f['isy'], z) + source
            qohms = (f['isx'] ** 2 + f['isy'] ** 2) / f['sigma']
            ent = CFG.entropic_n if k == 'n' else CFG.entropic_p
            q = CFG.heat_scale * (f['j'] * f['eta'] + qohme + qohms + ent * f['j'] * f['T'])
            comp[f'charge_s_{k}'] = nmse_residual(rqs, charge_scale)
            comp[f'total_current_{k}'] = nmse_residual(f['iex'] + f['isx'] + It, SCALES['I'])
            Cs = CFG.Csolid_n if k == 'n' else CFG.Csolid_p
            aa = CFG.a_n if k == 'n' else CFG.a_p
            dthdt = grad(f['theta_avg'], z)[:, 2:3]
            rmom = Cs * dthdt + CFG.gamma_li * aa * f['j']
            moment_scale = max(CFG.gamma_li * SCALES['I'] / Lk, 1e-08)
            comp[f'particle_moment_{k}'] = nmse_residual(rmom, moment_scale)
        else:
            q = CFG.heat_scale * qohme
            comp['total_current_s'] = nmse_residual(f['iex'] + It, SCALES['I'])
        rT = CFG.rho_cp * f['gT'][:, 2:3] - div_xy(f['kT'] * f['gT'][:, 0:1], f['kT'] * f['gT'][:, 1:2], z) - q
        comp[f'ce_{k}'] = nmse_residual(rce, SCALES['ce_rate'])
        comp[f'charge_e_{k}'] = nmse_residual(rqe, charge_scale)
        comp[f'heat_{k}'] = nmse_residual(rT, SCALES['heat_rate'])
    for k in ('n', 'p'):
        nr = 4
        nz = max(2, int(math.ceil(n / nr)))
        z0 = sample_region(nz, k, device, dtype).detach()
        q, _ = np.polynomial.legendre.leggauss(nr)
        rnodes = torch.tensor(0.5 * (q + 1.0), device=device, dtype=dtype).reshape(1, nr, 1)
        xyt = z0[:, None, :].expand(nz, nr, 3).reshape(-1, 3)
        rr = rnodes.expand(nz, nr, 1).reshape(-1, 1)
        zr = torch.cat([xyt[:, 0:2], rr, xyt[:, 2:3]], 1).requires_grad_(True)
        th = model.particle(zr, k)
        g = grad(th, zr)
        thr = g[:, 2:3]
        Ds = CFG.D_s_n if k == 'n' else CFG.D_s_p
        radial_flux = zr[:, 2:3] ** 2 * thr
        rpde = zr[:, 2:3] ** 2 * g[:, 3:4] - Ds * grad(radial_flux, zr)[:, 2:3]
        comp[f'particle_radial_{k}'] = nmse_residual(rpde, SCALES['particle_rate'])
    return comp

def outer_bc_losses(model, n, device, dtype):
    c = {}
    zl = sample_plane(n, x=0.0, device=device, dtype=dtype)
    fl = macro_physics(model, zl, 'n')
    I = torch_current_profile(zl[:, 2:3])
    c['left_ce_flux'] = nmse_residual(fl['gc'][:, 0:1], SCALES['ce_flux'] / max(CFG.D_e_n, 1e-08))
    c['left_ionic_flux'] = nmse_residual(fl['iex'], SCALES['I'])
    c['left_solid_current_hard_audit'] = nmse_residual(fl['isx'] + I, SCALES['I'])
    c['left_heat_flux'] = nmse_residual(CFG.kT_n * fl['gT'][:, 0:1], SCALES['heat_flux'])
    zr = sample_plane(n, x=CFG.Lx, device=device, dtype=dtype)
    fr = macro_physics(model, zr, 'p')
    I = torch_current_profile(zr[:, 2:3])
    c['right_ce_flux'] = nmse_residual(fr['gc'][:, 0:1], SCALES['ce_flux'] / max(CFG.D_e_p, 1e-08))
    c['right_ionic_flux'] = nmse_residual(fr['iex'], SCALES['I'])
    c['right_solid_current_hard_audit'] = nmse_residual(fr['isx'] + I, SCALES['I'])
    c['right_heat_flux'] = nmse_residual(CFG.kT_p * fr['gT'][:, 0:1], SCALES['heat_flux'])
    for k, x0, x1 in [('n', 0, CFG.x_n_end), ('s', CFG.x_n_end, CFG.x_s_end), ('p', CFG.x_s_end, CFG.Lx)]:
        for side, yv in [('bottom', 0.0), ('top', CFG.Ly)]:
            z = sample_plane(n, y=yv, device=device, dtype=dtype).detach()
            z[:, 0] = x0 + (x1 - x0) * torch.rand(n, device=device, dtype=dtype)
            z.requires_grad_(True)
            f = macro_physics(model, z, k)
            c[f'{k}_{side}_ce_flux'] = nmse_residual(f['gc'][:, 1:2], SCALES['ce_flux'] / max(float(f['De'].detach().mean().cpu()), 1e-08))
            c[f'{k}_{side}_ionic_flux'] = nmse_residual(f['iey'], SCALES['I'])
            if k in ('n', 'p'):
                c[f'{k}_{side}_solid_flux'] = nmse_residual(f['isy'], SCALES['I'])
            if side == 'top':
                r = -f['kT'] * f['gT'][:, 1:2] - h_top_torch(z[:, 0:1]) * (f['T'] - CFG.T_amb)
            else:
                r = f['kT'] * f['gT'][:, 1:2] - h_bottom_torch(z[:, 0:1]) * (f['T'] - CFG.T_amb)
            c[f'{k}_{side}_thermal'] = nmse_residual(r, SCALES['heat_flux'])
    return c

def interface_losses(model, n, device, dtype):
    out = {}
    for name, xv, ka, kb in [('ns', CFG.x_n_end, 'n', 's'), ('sp', CFG.x_s_end, 's', 'p')]:
        z = sample_plane(n, x=xv, device=device, dtype=dtype)
        fa = macro_physics(model, z, ka)
        fb = macro_physics(model, z, kb)
        hard_traces = bool(model.architecture_dict().get('hard_interface_traces', False))
        suffix = '_hard_audit' if hard_traces else ''
        out[f'{name}_ce_trace{suffix}'] = nmse_residual(fa['ce'] - fb['ce'], 1.0)
        out[f'{name}_phi_trace{suffix}'] = nmse_residual(fa['phi_e'] - fb['phi_e'], 0.1)
        out[f'{name}_T_trace{suffix}'] = nmse_residual(fa['T'] - fb['T'], 1.0)
        out[f'{name}_ce_flux'] = nmse_residual(fa['diffx'] - fb['diffx'], SCALES['ce_flux'])
        out[f'{name}_ionic_current'] = nmse_residual(fa['iex'] - fb['iex'], SCALES['I'])
        out[f'{name}_heat_flux'] = nmse_residual(fa['kT'] * fa['gT'][:, 0:1] - fb['kT'] * fb['gT'][:, 0:1], SCALES['heat_flux'])
        if ka == 'n':
            out['ns_solid_block_hard_audit'] = nmse_residual(fa['isx'], SCALES['I'])
        if kb == 'p':
            out['sp_solid_block_hard_audit'] = nmse_residual(fb['isx'], SCALES['I'])
    return out

def particle_hard_audits(model, n, device, dtype):
    out = {}
    for k in ('n', 'p'):
        z = sample_region(n, k, device, dtype).detach()
        f = model.region(z, k)
        j = f['j'].detach()
        for rvalue, label in [(0.0, 'center'), (1.0, 'surface')]:
            zr = torch.cat([z[:, 0:2], torch.zeros(n, 1, device=device, dtype=dtype) + rvalue, z[:, 2:3]], 1).requires_grad_(True)
            th = model.particle(zr, k, j=j)
            thr = grad(th, zr)[:, 2:3]
            if label == 'center':
                out[f'particle_center_{k}'] = nmse_residual(thr, 1.0)
            else:
                Ds = CFG.D_s_n if k == 'n' else CFG.D_s_p
                chi = particle_flux_factor(k)
                out[f'particle_surface_{k}'] = nmse_residual(-Ds * thr - chi * j, max(chi * SCALES['I'] / region_length(k), 1e-08))
    return out

def gauge_loss(model, n, device, dtype):
    z = torch.zeros(n, 3, device=device, dtype=dtype)
    z[:, 2:3] = sample_time(n, device, dtype)
    z.requires_grad_(True)
    return nmse_residual(model.region(z, 'n')['phi_e'], SCALES['phi'])

def reaction_compatibility(model, device, dtype, nx=12, ny=3):
    qx, wx = np.polynomial.legendre.leggauss(nx)
    qy, wy = np.polynomial.legendre.leggauss(ny)
    qx = torch.tensor(0.5 * (qx + 1), device=device, dtype=dtype)
    wx = torch.tensor(0.5 * wx, device=device, dtype=dtype)
    qy = torch.tensor(0.5 * (qy + 1) * CFG.Ly, device=device, dtype=dtype)
    wy = torch.tensor(0.5 * wy * CFG.Ly, device=device, dtype=dtype)
    times = sorted(set([float(CFG.current_ramp_time), 0.25 * CFG.t_final, 0.5 * CFG.t_final, 0.75 * CFG.t_final, CFG.t_final]))
    tv = torch.tensor(times, device=device, dtype=dtype)
    nt = len(times)
    sq = []
    rels = []
    tol = max(0.03 * abs(CFG.I_app) * CFG.Ly, 1e-08)
    for k, sgn in [('n', -1.0), ('p', 1.0)]:
        x0, x1 = (0.0, CFG.x_n_end) if k == 'n' else (CFG.x_s_end, CFG.Lx)
        L = x1 - x0
        a = CFG.a_n if k == 'n' else CFG.a_p
        X = x0 + L * qx[:, None].expand(nx, ny)
        Y = qy[None, :].expand(nx, ny)
        W = wx[:, None] * wy[None, :] * L
        X3 = X[None, :, :].expand(nt, nx, ny)
        Y3 = Y[None, :, :].expand(nt, nx, ny)
        T3 = tv[:, None, None].expand(nt, nx, ny)
        Z = torch.stack([X3.reshape(-1), Y3.reshape(-1), T3.reshape(-1)], 1)
        j = model.region(Z, k)['j'].reshape(nt, nx, ny)
        integ = a * torch.sum(j * W[None, :, :], dim=(1, 2))
        I = torch_current_profile(tv[:, None]).reshape(-1)
        target = sgn * I * CFG.Ly
        err = integ - target
        sq.extend(list((err / tol) ** 2))
        rels.extend(list(torch.abs(err) / torch.clamp(torch.abs(target), min=1e-08)))
    return (torch.mean(torch.stack(sq)), torch.stack(rels))

def _fixed_times(device, dtype):
    tr = float(CFG.current_ramp_time)
    vals = sorted(set([max(0.5 * tr, 0.01 * CFG.t_final), min(1.5 * tr, 0.2 * CFG.t_final), 0.25 * CFG.t_final, 0.5 * CFG.t_final, 0.75 * CFG.t_final, CFG.t_final]))
    return torch.tensor(vals, device=device, dtype=dtype)

def _fixed_region_grid(k, device, dtype, nx=4, ny=3):
    if k == 'n':
        x0, x1 = (0.0, CFG.x_n_end)
    elif k == 's':
        x0, x1 = (CFG.x_n_end, CFG.x_s_end)
    elif k == 'p':
        x0, x1 = (CFG.x_s_end, CFG.Lx)
    else:
        raise ValueError(k)
    L = x1 - x0
    xs = torch.linspace(x0 + 0.1 * L, x1 - 0.1 * L, nx, device=device, dtype=dtype)
    ys = torch.linspace(0.1 * CFG.Ly, 0.9 * CFG.Ly, ny, device=device, dtype=dtype)
    ts = _fixed_times(device, dtype)
    X, Y, T = torch.meshgrid(xs, ys, ts, indexing='ij')
    return torch.stack([X.reshape(-1), Y.reshape(-1), T.reshape(-1)], 1).requires_grad_(True)

def deterministic_physics_metrics(model, device, dtype):
    transport = []
    charge = []
    current = []
    moments = []
    radial = []
    boundaries = []
    interfaces = []
    for k in ('n', 's', 'p'):
        z = _fixed_region_grid(k, device, dtype)
        f = macro_physics(model, z, k)
        Lk = region_length(k)
        source = 0.0
        if k == 'n':
            source = CFG.a_n * f['j']
        elif k == 'p':
            source = CFG.a_p * f['j']
        rce = f['eps'] * f['gc'][:, 2:3] - div_xy(f['diffx'], f['diffy'], z) - CFG.gamma_li * source
        rqe = div_xy(f['iex'], f['iey'], z) - source
        qohme = (f['iex'] ** 2 + f['iey'] ** 2) / f['kap']
        charge_scale = SCALES['I'] / Lk
        It = torch_current_profile(z[:, 2:3])
        if k in ('n', 'p'):
            rqs = div_xy(f['isx'], f['isy'], z) + source
            qohms = (f['isx'] ** 2 + f['isy'] ** 2) / f['sigma']
            ent = CFG.entropic_n if k == 'n' else CFG.entropic_p
            q = CFG.heat_scale * (f['j'] * f['eta'] + qohme + qohms + ent * f['j'] * f['T'])
            charge.append(nmse_residual(rqs, charge_scale))
            current.append(nmse_residual(f['iex'] + f['isx'] + It, SCALES['I']))
            Cs = CFG.Csolid_n if k == 'n' else CFG.Csolid_p
            aa = CFG.a_n if k == 'n' else CFG.a_p
            rmom = Cs * grad(f['theta_avg'], z)[:, 2:3] + CFG.gamma_li * aa * f['j']
            moments.append(nmse_residual(rmom, max(CFG.gamma_li * SCALES['I'] / Lk, 1e-08)))
        else:
            q = CFG.heat_scale * qohme
            current.append(nmse_residual(f['iex'] + It, SCALES['I']))
        rT = CFG.rho_cp * f['gT'][:, 2:3] - div_xy(f['kT'] * f['gT'][:, 0:1], f['kT'] * f['gT'][:, 1:2], z) - q
        transport.extend([nmse_residual(rce, SCALES['ce_rate']), nmse_residual(rT, SCALES['heat_rate'])])
        charge.append(nmse_residual(rqe, charge_scale))
    qr, _ = np.polynomial.legendre.leggauss(5)
    rr = torch.tensor(0.5 * (qr + 1.0), device=device, dtype=dtype)
    ts = _fixed_times(device, dtype)[1::2]
    for k in ('n', 'p'):
        x0, x1 = (0.0, CFG.x_n_end) if k == 'n' else (CFG.x_s_end, CFG.Lx)
        xs = torch.linspace(x0 + 0.2 * (x1 - x0), x1 - 0.2 * (x1 - x0), 3, device=device, dtype=dtype)
        ys = torch.linspace(0.25 * CFG.Ly, 0.75 * CFG.Ly, 2, device=device, dtype=dtype)
        X, Y, R, T = torch.meshgrid(xs, ys, rr, ts, indexing='ij')
        zr = torch.stack([X.reshape(-1), Y.reshape(-1), R.reshape(-1), T.reshape(-1)], 1).requires_grad_(True)
        th = model.particle(zr, k)
        g = grad(th, zr)
        Ds = CFG.D_s_n if k == 'n' else CFG.D_s_p
        rpde = zr[:, 2:3] ** 2 * g[:, 3:4] - Ds * grad(zr[:, 2:3] ** 2 * g[:, 2:3], zr)[:, 2:3]
        radial.append(nmse_residual(rpde, SCALES['particle_rate']))
    ys = torch.linspace(0.0, CFG.Ly, 5, device=device, dtype=dtype)
    ts = _fixed_times(device, dtype)
    for xv, ka, kb in [(CFG.x_n_end, 'n', 's'), (CFG.x_s_end, 's', 'p')]:
        Y, T = torch.meshgrid(ys, ts, indexing='ij')
        z = torch.stack([torch.zeros_like(Y).reshape(-1) + xv, Y.reshape(-1), T.reshape(-1)], 1).requires_grad_(True)
        fa = macro_physics(model, z, ka)
        fb = macro_physics(model, z, kb)
        interfaces.extend([nmse_residual(fa['diffx'] - fb['diffx'], SCALES['ce_flux']), nmse_residual(fa['iex'] - fb['iex'], SCALES['I']), nmse_residual(fa['kT'] * fa['gT'][:, 0:1] - fb['kT'] * fb['gT'][:, 0:1], SCALES['heat_flux'])])
    for k, xv in [('n', 0.0), ('p', CFG.Lx)]:
        ys = torch.linspace(0.0, CFG.Ly, 5, device=device, dtype=dtype)
        Y, T = torch.meshgrid(ys, ts, indexing='ij')
        z = torch.stack([torch.zeros_like(Y).reshape(-1) + xv, Y.reshape(-1), T.reshape(-1)], 1).requires_grad_(True)
        f = macro_physics(model, z, k)
        boundaries.extend([nmse_residual(f['gc'][:, 0:1], SCALES['ce_flux'] / max(CFG.D_e_n if k == 'n' else CFG.D_e_p, 1e-08)), nmse_residual(f['iex'], SCALES['I']), nmse_residual((CFG.kT_n if k == 'n' else CFG.kT_p) * f['gT'][:, 0:1], SCALES['heat_flux'])])
    for k in ('n', 's', 'p'):
        x0, x1 = (0.0, CFG.x_n_end) if k == 'n' else (CFG.x_n_end, CFG.x_s_end) if k == 's' else (CFG.x_s_end, CFG.Lx)
        xs = torch.linspace(x0, x1, 4, device=device, dtype=dtype)
        for side, yv in [('bottom', 0.0), ('top', CFG.Ly)]:
            X, T = torch.meshgrid(xs, ts, indexing='ij')
            z = torch.stack([X.reshape(-1), torch.zeros_like(X).reshape(-1) + yv, T.reshape(-1)], 1).requires_grad_(True)
            f = macro_physics(model, z, k)
            boundaries.extend([nmse_residual(f['gc'][:, 1:2], SCALES['ce_flux'] / max(float(f['De'].detach().mean().cpu()), 1e-08)), nmse_residual(f['iey'], SCALES['I'])])
            if k in ('n', 'p'):
                boundaries.append(nmse_residual(f['isy'], SCALES['I']))
            if side == 'top':
                rb = -f['kT'] * f['gT'][:, 1:2] - h_top_torch(z[:, 0:1]) * (f['T'] - CFG.T_amb)
            else:
                rb = f['kT'] * f['gT'][:, 1:2] - h_bottom_torch(z[:, 0:1]) * (f['T'] - CFG.T_amb)
            boundaries.append(nmse_residual(rb, SCALES['heat_flux']))
    compat, _ = reaction_compatibility(model, device, dtype)
    yg = torch.linspace(0.0, CFG.Ly, 5, device=device, dtype=dtype)
    Y, T = torch.meshgrid(yg, ts, indexing='ij')
    zg = torch.stack([torch.zeros_like(Y).reshape(-1), Y.reshape(-1), T.reshape(-1)], 1).requires_grad_(True)
    gauge = nmse_residual(model.region(zg, 'n')['phi_e'], SCALES['phi'])
    mean = lambda vals: sum(vals) / max(len(vals), 1)
    metrics = {'transport': mean(transport), 'charge_divergence': mean(charge), 'total_current': mean(current), 'particle_moment': mean(moments), 'particle_radial': mean(radial), 'boundary': mean(boundaries), 'interface_flux': mean(interfaces), 'reaction_compatibility': compat, 'gauge': gauge}
    weights = {'transport': 1.0, 'charge_divergence': 2.0, 'total_current': 2.5, 'particle_moment': 1.5, 'particle_radial': 0.1, 'boundary': 2.0, 'interface_flux': 4.0, 'reaction_compatibility': 1.0, 'gauge': 0.2}
    denom = sum(weights.values())
    metrics['composite'] = sum((weights[k] * torch.log1p(metrics[k]) for k in weights)) / denom
    return metrics

def total_loss(model, ncol, device, dtype, progress):
    npde = max(4, ncol)
    nb = max(4, ncol // 4)
    pde = pde_losses(model, npde, device, dtype)
    bc = outer_bc_losses(model, nb, device, dtype)
    inter = interface_losses(model, nb, device, dtype)
    gauge = gauge_loss(model, max(4, nb // 2), device, dtype)
    compat, compat_rels = reaction_compatibility(model, device, dtype)
    pa = particle_hard_audits(model, max(4, nb // 2), device, dtype)
    charge_div = [v for k, v in pde.items() if k.startswith('charge_')]
    current = [v for k, v in pde.items() if k.startswith('total_current_')]
    moment = [v for k, v in pde.items() if k.startswith('particle_moment_')]
    radial = [v for k, v in pde.items() if k.startswith('particle_radial_')]
    transport = [v for k, v in pde.items() if not (k.startswith('charge_') or k.startswith('total_current_') or k.startswith('particle_'))]
    mean = lambda a: sum(a) / max(len(a), 1)
    Ltransport = mean(transport)
    Lcharge = mean(charge_div)
    Lcurrent = mean(current)
    Lmoment = mean(moment)
    Lradial = mean(radial)
    hard_keys = ('hard_audit',)
    bc_train = [v for k, v in bc.items() if not any((s in k for s in hard_keys))]
    bc_charge = [v for k, v in bc.items() if ('ionic' in k or 'solid_flux' in k) and 'hard_audit' not in k]
    bc_other = [v for k, v in bc.items() if v not in bc_charge and 'hard_audit' not in k]
    bc_charge = [v for k, v in bc.items() if ('ionic' in k or 'solid_flux' in k) and 'hard_audit' not in k]
    bc_other = [v for k, v in bc.items() if not (('ionic' in k or 'solid_flux' in k) or 'hard_audit' in k)]
    if_flux = [v for k, v in inter.items() if 'hard_audit' not in k]
    Lbcq = mean(bc_charge)
    Lbct = mean(bc_other)
    Lif = mean(if_flux)
    wradial = 0.05
    radial_obj = torch.log1p(Lradial)
    total = Ltransport + 2.0 * Lcharge + 2.5 * Lcurrent + 1.5 * Lmoment + wradial * radial_obj + 2.0 * Lbct + 2.5 * Lbcq + 4.0 * Lif + compat + 0.2 * gauge
    groups = {'transport_pde': Ltransport, 'charge_divergence': Lcharge, 'total_current': Lcurrent, 'particle_moment': Lmoment, 'particle_pde': Lradial, 'particle_radial_objective': radial_obj, 'particle_weight': torch.tensor(wradial, device=device, dtype=dtype), 'bc_transport': Lbct, 'bc_charge': Lbcq, 'interface_flux': Lif, 'reaction_compatibility': compat, 'reaction_integral_mean_rel': torch.mean(compat_rels), 'reaction_integral_max_rel': torch.max(compat_rels), 'gauge': gauge, 'particle_surface': 0.5 * (pa['particle_surface_n'] + pa['particle_surface_p']), 'particle_center': 0.5 * (pa['particle_center_n'] + pa['particle_center_p'])}
    return (total, groups)

def lr_for_step(step, steps, start=5e-05, peak=0.0003, end=5e-05):
    warm = max(5, int(round(0.1 * steps)))
    warm = min(warm, max(1, steps // 2))
    if step <= warm:
        a = step / max(warm, 1)
        return start + (peak - start) * a
    q = (step - warm) / max(steps - warm, 1)
    return end + 0.5 * (peak - end) * (1.0 + math.cos(math.pi * q))

def train(model_name='fain', steps=12000, width=None, depth=4, particle_width=64, particle_depth=3, ncol=256, seed=20260913, device=None, dtype_name='float32', outdir=None, log_every=None):
    seed_all(seed)
    model_name = model_name.lower().replace('-', '_')
    outdir = Path(outdir or f'outputs/formal/{model_name}')
    outdir.mkdir(parents=True, exist_ok=True)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float64 if dtype_name == 'float64' else torch.float32
    model = build_model(model_name, width, depth, particle_width, particle_depth).to(device=device, dtype=dtype)
    opt = torch.optim.Adam(model.parameters(), lr=5e-05)
    log_every = log_every or max(1, steps // 100)
    ema_decay = 0.995 if steps <= 3000 else 0.999
    ema_start = max(20, int(0.2 * steps))
    ema_state = None
    hist = []
    t0 = time.time()
    display = model.architecture_dict()['name']
    for step in range(1, steps + 1):
        lr = lr_for_step(step, steps)
        for pg in opt.param_groups:
            pg['lr'] = lr
        prog = step / max(steps, 1)
        model.alpha = float(np.clip((prog - 0.08) / 0.3, 0.0, 1.0)) if model_name == 'fain' else 1.0
        opt.zero_grad(set_to_none=True)
        loss, g = total_loss(model, ncol, device, dtype, prog)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Non-finite {display} loss at step {step}')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        if step == ema_start:
            ema_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif step > ema_start and ema_state is not None:
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    if torch.is_floating_point(v):
                        ema_state[k].mul_(ema_decay).add_(v.detach(), alpha=1.0 - ema_decay)
                    else:
                        ema_state[k].copy_(v)
        if step == 1 or step == steps or step % log_every == 0:
            row = {'step': step, 'total': float(loss.detach().cpu()), 'alpha': float(model.alpha), 'lr': lr}
            for k, v in g.items():
                row[k] = float(v.detach().cpu())
            hist.append(row)
            print(f"[{display}] {step:6d}/{steps} total={row['total']:.3e} charge={row['charge_divergence']:.3e} current={row['total_current']:.3e} react={row['reaction_integral_mean_rel']:.3e} particle={row['particle_pde']:.3e} lr={lr:.2e}")
    final_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    candidates = {'final': final_state}
    if ema_state is not None:
        candidates['ema'] = ema_state
    scores = {}
    score_details = {}
    for name, state in candidates.items():
        model.load_state_dict(state)
        model.alpha = 1.0
        vm = deterministic_physics_metrics(model, device, dtype)
        score_details[name] = {k: float(v.detach().cpu()) for k, v in vm.items()}
        scores[name] = score_details[name]['composite']
    selected = min(scores, key=scores.get)
    model.load_state_dict(candidates[selected])
    model.alpha = 1.0
    arch = model.architecture_dict()
    ck = {'state_dict': final_state, 'ema_state_dict': ema_state, 'evaluation_state': selected, 'physics_validation_scores': scores, 'deterministic_physics_validation': score_details, 'architecture': arch, 'seed': seed, 'dtype': dtype_name, 'steps': steps, 'ncol': ncol}
    torch.save(ck, outdir / 'model.pt')
    with open(outdir / 'training_history.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(hist[0].keys()))
        w.writeheader()
        w.writerows(hist)
    meta = {'method': arch['name'], 'full_name': arch['full_name'], 'version': arch['version'], 'zero_reference_supervision': True, 'supervised_reference_fraction': 0.0, 'reference_solution_read_during_training': False, 'physics_protocol_is_training_input': True, 'residuals_physically_scaled': True, 'primary_solid_potential': True, 'BV_derived_reaction': True, 'hard_galvanostatic_reaction': False, 'hard_BV_closure': True, 'hard_solid_current_BCs': True, 'hard_interface_traces': bool(arch.get('hard_interface_traces', False)), 'explicit_interface_value_loss': bool(arch.get('explicit_interface_value_loss', False)), 'hard_particle_radial_BCs': True, 'particle_moment_representation': True, 'baseline_preserving_particle_moments': True, 'zero_spatial_mean_particle_corrections': True, 'diffusion_boundary_layer_particle_basis': True, 'local_particle_moment_conservation': True, 'deterministic_reaction_compatibility': True, 'reaction_compatibility_tolerance_fraction': 0.03, 'reference_free_checkpoint_selection': True, 'deterministic_checkpoint_selection': True, 'evaluation_state': selected, 'physics_validation_scores': scores, 'deterministic_physics_validation': score_details, 'ema_checkpoint': ema_state is not None, 'ema_decay': ema_decay, 'early_time_sampling': True, 'warmup_learning_rate': True, 'device': device, 'runtime_seconds': time.time() - t0, 'architecture': arch, 'seed': seed, 'steps': steps, 'ncol_per_region': ncol, 'residual_scales': SCALES, 'physics': config_dict()}
    (outdir / 'training_config.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(f'[{display}] physics-only checkpoint selection: {selected} {scores}')
    return outdir / 'model.pt'