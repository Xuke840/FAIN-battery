from __future__ import annotations
from pathlib import Path
import argparse, csv, json, math, shutil
import numpy as np
import torch
from physics import CFG, region_layout, current_profile, epsilon_np, kappa_np, coeff_torch, ocp_n_np, ocp_p_np, exchange_current_np, torch_current_profile, particle_flux_factor
from models import build_model

def load_model(path, device):
    ck = torch.load(path, map_location=device)
    a = ck['architecture']
    name = str(a.get('name', '')).lower()
    if name == 'fain':
        key = 'fain'
    elif 'pure' in name:
        key = 'pure_pinn'
    elif 'res' in name:
        key = 'res_pinn'
    elif 'parameter' in name:
        key = 'param_mlp'
    elif 'siren' in name:
        key = 'siren_pinn'
    elif 'xpinn' in name:
        key = 'xpinn'
    else:
        raise ValueError(a.get('name'))
    m = build_model(key, a.get('width'), a.get('depth', 4), a.get('particle_width', 64), a.get('particle_depth', 3)).to(device)
    state = ck.get('ema_state_dict') if ck.get('evaluation_state') == 'ema' and ck.get('ema_state_dict') is not None else ck['state_dict']
    m.load_state_dict(state, strict=True)
    m.eval()
    m.alpha = 1.0
    return (m, ck)

def _eval_region(model, z_np, k, device, dtype, chunk=32768):
    outs = {'ce': [], 'phi_e': [], 'T': []}
    if k in ('n', 'p'):
        outs['phi_s'] = []
        outs['theta_avg'] = []
        outs['theta_surf'] = []
    with torch.no_grad():
        for i in range(0, len(z_np), chunk):
            z = torch.tensor(z_np[i:i + chunk], device=device, dtype=dtype)
            f = model.region(z, k)
            for key in outs:
                outs[key].append(f[key].detach().cpu().numpy().ravel())
    return {k0: np.concatenate(v) for k0, v in outs.items()}

def _eval_particle(model, xy_t, k, rvals, device, dtype, chunk=32768):
    n = len(xy_t)
    nr = len(rvals)
    rows = np.repeat(xy_t, nr, axis=0)
    rr = np.tile(np.asarray(rvals, float), n)[:, None]
    zr = np.c_[rows[:, 0:2], rr, rows[:, 2]]
    vals = []
    with torch.no_grad():
        for i in range(0, len(zr), chunk):
            z = torch.tensor(zr[i:i + chunk], device=device, dtype=dtype)
            vals.append(model.particle(z, k).detach().cpu().numpy().ravel())
    return np.concatenate(vals).reshape(n, nr)

def evaluate_model(model, x, y, t, device, dtype, nr_eval=24):
    nx, ny = (len(x), len(y))
    g = region_layout(nx, ny)
    X, Y = np.meshgrid(x, y)
    state = {k: [] for k in ['ce', 'T', 'phi_e', 'phi_sn', 'phi_sp', 'theta_n_surf', 'theta_p_surf', 'theta_n_avg', 'theta_p_avg']}
    faces = np.linspace(0, 1, nr_eval + 1)
    rv = (faces[1:] ** 3 - faces[:-1] ** 3) / 3
    rcent = 0.5 * (faces[:-1] + faces[1:])
    for it, tt in enumerate(t):
        ce = np.empty((ny, nx))
        T = np.empty_like(ce)
        pe = np.empty_like(ce)
        psn = np.full_like(ce, np.nan)
        psp = np.full_like(ce, np.nan)
        thns = np.full_like(ce, np.nan)
        thps = np.full_like(ce, np.nan)
        thna = np.full_like(ce, np.nan)
        thpa = np.full_like(ce, np.nan)
        for k, mask in [('n', g['mn']), ('s', g['ms']), ('p', g['mp'])]:
            pts = np.c_[X[mask], Y[mask], np.full(np.sum(mask), tt)]
            o = _eval_region(model, pts, k, device, dtype)
            ce[mask] = o['ce']
            T[mask] = o['T']
            pe[mask] = o['phi_e']
            if k == 'n':
                psn[mask] = o['phi_s']
            if k == 'p':
                psp[mask] = o['phi_s']
            if k in ('n', 'p'):
                if k == 'n':
                    thns[mask] = o['theta_surf']
                    thna[mask] = o['theta_avg']
                else:
                    thps[mask] = o['theta_surf']
                    thpa[mask] = o['theta_avg']
        for key, a in [('ce', ce), ('T', T), ('phi_e', pe), ('phi_sn', psn), ('phi_sp', psp), ('theta_n_surf', thns), ('theta_p_surf', thps), ('theta_n_avg', thna), ('theta_p_avg', thpa)]:
            state[key].append(a)
        if it == 0 or it == len(t) - 1 or it % max(1, len(t) // 10) == 0:
            print(f'[POST] FAIN state {it + 1}/{len(t)}')
    return ({k: np.asarray(v) for k, v in state.items()}, g)

def _regional_gradient(a, g, dy, dx):
    gy = np.empty_like(a, dtype=float)
    gx = np.empty_like(a, dtype=float)
    cuts = [(0, g['i_n']), (g['i_n'], g['i_p']), (g['i_p'], a.shape[1])]
    for i0, i1 in cuts:
        q = np.asarray(a[:, i0:i1], float)
        if q.shape[1] >= 2:
            qy, qx = np.gradient(q, dy, dx, edge_order=1)
        else:
            qy = np.gradient(q, dy, axis=0, edge_order=1)
            qx = np.zeros_like(q)
        gy[:, i0:i1] = qy
        gx[:, i0:i1] = qx
    return (gy, gx)

def derive(state, g, x, y, t):
    ny, nx = (len(y), len(x))
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    mn, ms, mp = (g['mn'], g['ms'], g['mp'])
    nn = g['i_n']
    ip = g['i_p']
    for key in ['jn', 'jp', 'eta_n', 'eta_p', 'i0_n', 'i0_p', 'i_e_x', 'i_e_y', 'i_s_n_x', 'i_s_n_y', 'i_s_p_x', 'i_s_p_y', 'q_rxn', 'q_ohm_e', 'q_ohm_s', 'q_rev', 'q_gen']:
        state[key] = []
    for it, tt in enumerate(t):
        ce = state['ce'][it]
        T = state['T'][it]
        pe = state['phi_e'][it]
        psn = state['phi_sn'][it][:, :nn]
        psp = state['phi_sp'][it][:, ip:]
        thn = state['theta_n_surf'][it][:, :nn]
        thp = state['theta_p_surf'][it][:, ip:]
        kap = kappa_np(g, ce, T)
        dpey, dpex = _regional_gradient(pe, g, dy, dx)
        lnc = np.log(np.clip(ce, 1e-12, None))
        dlcy, dlcx = _regional_gradient(lnc, g, dy, dx)
        iex = -kap * dpex + CFG.chi_ln_ce * kap * dlcx
        iey = -kap * dpey + CFG.chi_ln_ce * kap * dlcy
        dny, dnx = np.gradient(psn, dy, dx, edge_order=1)
        dpy, dpx = np.gradient(psp, dy, dx, edge_order=1)
        isnx = -CFG.sigma_n * dnx
        isny = -CFG.sigma_n * dny
        ispx = -CFG.sigma_p * dpx
        ispy = -CFG.sigma_p * dpy
        eta_n = psn - pe[:, :nn] - ocp_n_np(thn)
        eta_p = psp - pe[:, ip:] - ocp_p_np(thp)
        i0n = exchange_current_np('n', ce[:, :nn], thn, T[:, :nn])
        i0p = exchange_current_np('p', ce[:, ip:], thp, T[:, ip:])
        raw_n = CFG.beta_bv * eta_n / np.clip(T[:, :nn], 0.5, 2)
        raw_p = CFG.beta_bv * eta_p / np.clip(T[:, ip:], 0.5, 2)
        jn = 2 * i0n * np.sinh(np.clip(raw_n, -CFG.bv_argument_guard, CFG.bv_argument_guard))
        jp = 2 * i0p * np.sinh(np.clip(raw_p, -CFG.bv_argument_guard, CFG.bv_argument_guard))

        def full(local, where):
            a = np.full((ny, nx), np.nan)
            a[:, where] = local
            return a
        jnf = full(jn, slice(0, nn))
        jpf = full(jp, slice(ip, nx))
        enf = full(eta_n, slice(0, nn))
        epf = full(eta_p, slice(ip, nx))
        i0nf = full(i0n, slice(0, nn))
        i0pf = full(i0p, slice(ip, nx))
        isnxf = full(isnx, slice(0, nn))
        isnyf = full(isny, slice(0, nn))
        ispxf = full(ispx, slice(ip, nx))
        ispyf = full(ispy, slice(ip, nx))
        qrx = np.zeros_like(ce)
        qrx[:, :nn] = jn * eta_n
        qrx[:, ip:] = jp * eta_p
        qoe = (iex ** 2 + iey ** 2) / np.clip(kap, 1e-12, None)
        qos = np.zeros_like(ce)
        qos[:, :nn] = (isnx ** 2 + isny ** 2) / CFG.sigma_n
        qos[:, ip:] = (ispx ** 2 + ispy ** 2) / CFG.sigma_p
        qrv = np.zeros_like(ce)
        qrv[:, :nn] = CFG.entropic_n * jn * T[:, :nn]
        qrv[:, ip:] = CFG.entropic_p * jp * T[:, ip:]
        qg = CFG.heat_scale * (qrx + qoe + qos + qrv)
        vals = {'jn': jnf, 'jp': jpf, 'eta_n': enf, 'eta_p': epf, 'i0_n': i0nf, 'i0_p': i0pf, 'i_e_x': iex, 'i_e_y': iey, 'i_s_n_x': isnxf, 'i_s_n_y': isnyf, 'i_s_p_x': ispxf, 'i_s_p_y': ispyf, 'q_rxn': qrx, 'q_ohm_e': qoe, 'q_ohm_s': qos, 'q_rev': qrv, 'q_gen': qg}
        for k, v in vals.items():
            state[k].append(v)
    for k in list(state):
        state[k] = np.asarray(state[k])
    return state

def align_potential_gauge(pred, ref):
    for it in range(pred['phi_e'].shape[0]):
        delta = pred['phi_e'][it, 0, 0] - ref['phi_e'][it, 0, 0]
        pred['phi_e'][it] -= delta
        pred['phi_sn'][it] -= delta
        pred['phi_sp'][it] -= delta
    return pred

def field_metric(a, b, valid):
    av = a[valid]
    bv = b[valid]
    d = bv - av
    mse = float(np.mean(d * d))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(d)))
    rel = float(np.linalg.norm(d) / max(np.linalg.norm(av), 1e-14))
    linf = float(np.max(np.abs(d)))
    return (rmse, mse, mae, rel, linf)

def write_rows(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

def diagnostics(state, g, x, y, t, prefix, outdir):
    outdir = Path(outdir)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    X, Y = np.meshgrid(x, y)
    mn, ms, mp = (g['mn'], g['ms'], g['mp'])
    nn = g['i_n']
    ip = g['i_p']
    eps = epsilon_np(g)
    global_rows = []
    region_rows = {'negative': [], 'separator': [], 'positive': []}
    Li0 = None
    for it, tt in enumerate(t):
        ce = state['ce'][it]
        T = state['T'][it]
        I = float(current_profile(tt))
        thna = state['theta_n_avg'][it]
        thpa = state['theta_p_avg'][it]
        Li = float((np.sum(eps * ce) + CFG.Csolid_n * np.nansum(thna) + CFG.Csolid_p * np.nansum(thpa)) * dx * dy)
        if Li0 is None:
            Li0 = Li
        Qn = CFG.a_n * np.nansum(state['jn'][it]) * dx * dy
        Qp = CFG.a_p * np.nansum(state['jp'][it]) * dx * dy
        if abs(I) > 1e-12:
            Ib = max(abs(Qn + I * CFG.Ly), abs(Qp - I * CFG.Ly), abs(Qn + Qp)) / abs(I * CFG.Ly)
        else:
            Ib = max(abs(Qn), abs(Qp))
        total_x = state['i_e_x'][it].copy()
        total_x += np.nan_to_num(state['i_s_n_x'][it], nan=0.0)
        total_x += np.nan_to_num(state['i_s_p_x'][it], nan=0.0)
        ymean_total = np.mean(total_x, axis=0)
        if abs(I) > 1e-12:
            total_err = ymean_total + I
            It_rms = float(np.sqrt(np.mean(total_err ** 2)) / abs(I))
            It_max = float(np.max(np.abs(total_err)) / abs(I))
        else:
            It_rms = float(np.sqrt(np.mean(ymean_total ** 2)))
            It_max = float(np.max(np.abs(ymean_total)))
        Vn = float(np.nanmean(state['phi_sn'][it][:, 0] - 0.5 * dx * (I / CFG.sigma_n)))
        Vp = float(np.nanmean(state['phi_sp'][it][:, -1] + 0.5 * dx * (I / CFG.sigma_p)))
        grow = {'time': float(tt), 'I_app': I, 'V_n': Vn, 'V_p': Vp, 'V_cell': Vp - Vn, 'SOC_n': float(np.nanmean(thna)), 'SOC_p': float(np.nanmean(thpa)), 'Li_inventory': Li, 'Li_balance_rel': abs(Li - Li0) / max(abs(Li0), 1e-14), 'reaction_integral_balance_rel': float(Ib), 'local_total_current_rms_rel': It_rms, 'local_total_current_max_rel': It_max, 'T_min': float(np.min(T)), 'T_max': float(np.max(T)), 'T_mean': float(np.mean(T)), 'ce_min': float(np.min(ce)), 'ce_max': float(np.max(ce))}
        global_rows.append(grow)
        gy, gx = _regional_gradient(ce, g, dy, dx)
        for name, mask, k in [('negative', mn, 'n'), ('separator', ms, 's'), ('positive', mp, 'p')]:
            row = {'time': float(tt), 'I_app': I, 'ce_min': float(np.min(ce[mask])), 'ce_max': float(np.max(ce[mask])), 'ce_mean': float(np.mean(ce[mask])), 'phi_e_min': float(np.min(state['phi_e'][it][mask])), 'phi_e_max': float(np.max(state['phi_e'][it][mask])), 'phi_e_mean': float(np.mean(state['phi_e'][it][mask])), 'T_min': float(np.min(T[mask])), 'T_max': float(np.max(T[mask])), 'T_mean': float(np.mean(T[mask])), 'q_gen_max': float(np.max(state['q_gen'][it][mask])), 'q_gen_mean': float(np.mean(state['q_gen'][it][mask])), 'q_ohm_e_max': float(np.max(state['q_ohm_e'][it][mask])), 'i_e_abs_max': float(np.max(np.sqrt(state['i_e_x'][it][mask] ** 2 + state['i_e_y'][it][mask] ** 2))), 'grad_ce_abs_max': float(np.max(np.sqrt(gx[mask] ** 2 + gy[mask] ** 2)))}
            xs = np.unique(X[mask])
            prof = np.array([np.mean(ce[:, np.argmin(np.abs(x - xx))]) for xx in xs])
            mid = 0.5 * (prof.min() + prof.max())
            row['ce_front_x'] = float(xs[np.argmin(abs(prof - mid))])
            if k in ('n', 'p'):
                jf = state['jn'][it] if k == 'n' else state['jp'][it]
                ths = state['theta_n_surf'][it] if k == 'n' else state['theta_p_surf'][it]
                tha = state['theta_n_avg'][it] if k == 'n' else state['theta_p_avg'][it]
                eta = state['eta_n'][it] if k == 'n' else state['eta_p'][it]
                i0 = state['i0_n'][it] if k == 'n' else state['i0_p'][it]
                isx = state['i_s_n_x'][it] if k == 'n' else state['i_s_p_x'][it]
                isy = state['i_s_n_y'][it] if k == 'n' else state['i_s_p_y'][it]
                vals = jf[mask]
                av = np.abs(vals)
                den = av.sum()
                area = mask.sum() * dx * dy
                if den > 1e-15:
                    cx = float((X[mask] * av).sum() / den)
                    cy = float((Y[mask] * av).sum() / den)
                    ii = int(np.argmax(av))
                    px = float(X[mask][ii])
                    py = float(Y[mask][ii])
                    loc = area * float(np.sum(vals ** 2) * dx * dy) / float(np.sum(av) * dx * dy) ** 2
                    sgn = -1 if k == 'n' else 1
                    rev = float(np.maximum(-sgn * vals, 0).sum() / den)
                else:
                    cx = cy = px = py = loc = rev = np.nan
                row.update({'j_min': float(vals.min()), 'j_max': float(vals.max()), 'j_abs_max': float(av.max()), 'j_mean': float(vals.mean()), 'theta_surf_min': float(np.nanmin(ths[mask])), 'theta_surf_max': float(np.nanmax(ths[mask])), 'theta_surf_mean': float(np.nanmean(ths[mask])), 'theta_avg_mean': float(np.nanmean(tha[mask])), 'eta_mean': float(np.nanmean(eta[mask])), 'i0_mean': float(np.nanmean(i0[mask])), 'reaction_centroid_x': cx, 'reaction_centroid_y': cy, 'reaction_peak_x': px, 'reaction_peak_y': py, 'reaction_localization': loc, 'reverse_reaction_fraction': rev, 'solid_current_abs_max': float(np.nanmax(np.sqrt(isx[mask] ** 2 + isy[mask] ** 2))), 'q_rxn_mean': float(np.mean(state['q_rxn'][it][mask])), 'q_ohm_s_mean': float(np.mean(state['q_ohm_s'][it][mask])), 'q_rev_mean': float(np.mean(state['q_rev'][it][mask]))})
            region_rows[name].append(row)
    write_rows(outdir / f'{prefix}_global_cell.csv', global_rows)
    for name, rows in region_rows.items():
        write_rows(outdir / f'{prefix}_{name}.csv', rows)
    return (global_rows, region_rows)

def _torch_grad(u, z):
    return torch.autograd.grad(u, z, torch.ones_like(u), create_graph=True, retain_graph=True)[0]

def _torch_currents(model, z, k):
    f = model.region(z, k)
    ce, pe, T = (f['ce'], f['phi_e'], f['T'])
    _, _, kap, _ = coeff_torch(k, ce, T)
    gc = _torch_grad(ce, z)
    gp = _torch_grad(pe, z)
    iex = -kap * gp[:, 0:1] + CFG.chi_ln_ce * kap * gc[:, 0:1] / ce
    out = {'iex': iex, **f}
    if k in ('n', 'p'):
        gps = _torch_grad(f['phi_s'], z)
        sigma = CFG.sigma_n if k == 'n' else CFG.sigma_p
        out['isx'] = -sigma * gps[:, 0:1]
    return out

def physics_constraint_audit(model, outdir, device, dtype):
    rows = []

    def add(name, value, kind='rms', units='dimensionless'):
        rows.append({'metric': name, 'value': float(value), 'kind': kind, 'units': units})
    times = [float(CFG.current_ramp_time), 0.25 * CFG.t_final, 0.5 * CFG.t_final, 0.75 * CFG.t_final, CFG.t_final]
    ys = torch.linspace(0.0, CFG.Ly, 7, device=device, dtype=dtype)
    for label, xv, ka, kb in [('ns', CFG.x_n_end, 'n', 's'), ('sp', CFG.x_s_end, 's', 'p')]:
        vals = {q: [] for q in ('ce', 'phi_e', 'T')}
        for tv in times:
            z = torch.stack([torch.zeros_like(ys) + xv, ys, torch.zeros_like(ys) + tv], 1)
            with torch.no_grad():
                fa = model.region(z, ka)
                fb = model.region(z, kb)
            for q in vals:
                vals[q].append((fa[q] - fb[q]).detach().cpu().numpy().ravel())
        for q, parts in vals.items():
            a = np.concatenate(parts)
            add(f'interface_{label}_{q}_jump_rms', np.sqrt(np.mean(a * a)))
            add(f'interface_{label}_{q}_jump_max', np.max(np.abs(a)), 'max')
    bspec = [('collector_left', 'n', 0.0, -1.0), ('solid_block_ns', 'n', CFG.x_n_end, 0.0), ('solid_block_sp', 'p', CFG.x_s_end, 0.0), ('collector_right', 'p', CFG.Lx, -1.0)]
    for name, k, xv, coef in bspec:
        errs = []
        for tv in times:
            z = torch.stack([torch.zeros_like(ys) + xv, ys, torch.zeros_like(ys) + tv], 1).requires_grad_(True)
            f = _torch_currents(model, z, k)
            I = torch_current_profile(z[:, 2:3])
            target = coef * I
            errs.append(((f['isx'] - target) / max(abs(CFG.I_app), 1e-12)).detach().cpu().numpy().ravel())
        a = np.concatenate(errs)
        add(name + '_rms_rel', np.sqrt(np.mean(a * a)))
        add(name + '_max_rel', np.max(np.abs(a)), 'max')
    for k in ('n', 's', 'p'):
        x0, x1 = (0.0, CFG.x_n_end) if k == 'n' else (CFG.x_n_end, CFG.x_s_end) if k == 's' else (CFG.x_s_end, CFG.Lx)
        xs = torch.linspace(x0, x1, 17, device=device, dtype=dtype)
        yq = torch.linspace(0.0, CFG.Ly, 5, device=device, dtype=dtype)
        X, Y = torch.meshgrid(xs, yq, indexing='ij')
        errs = []
        final = []
        for tv in times:
            z = torch.stack([X.reshape(-1), Y.reshape(-1), torch.zeros(X.numel(), device=device, dtype=dtype) + tv], 1).requires_grad_(True)
            f = _torch_currents(model, z, k)
            I = torch_current_profile(z[:, 2:3])
            total = f['iex'] + (f['isx'] if k in ('n', 'p') else 0.0)
            e = ((total + I) / max(abs(CFG.I_app), 1e-12)).detach().cpu().numpy().ravel()
            errs.append(e)
            if abs(tv - CFG.t_final) < 1e-12:
                final = e
        a = np.concatenate(errs)
        add(f'total_current_{k}_rms_rel', np.sqrt(np.mean(a * a)))
        add(f'total_current_{k}_final_rms_rel', np.sqrt(np.mean(np.asarray(final) ** 2)))
    qx, wx = np.polynomial.legendre.leggauss(16)
    qy, wy = np.polynomial.legendre.leggauss(5)
    qx_t = torch.tensor(0.5 * (qx + 1), device=device, dtype=dtype)
    wx_t = torch.tensor(0.5 * wx, device=device, dtype=dtype)
    qy_t = torch.tensor(0.5 * (qy + 1) * CFG.Ly, device=device, dtype=dtype)
    wy_t = torch.tensor(0.5 * wy * CFG.Ly, device=device, dtype=dtype)
    for k, sgn in [('n', -1.0), ('p', 1.0)]:
        x0, x1 = (0.0, CFG.x_n_end) if k == 'n' else (CFG.x_s_end, CFG.Lx)
        L = x1 - x0
        aa = CFG.a_n if k == 'n' else CFG.a_p
        rel = []
        X = x0 + L * qx_t[:, None].expand(len(qx), len(qy))
        Y = qy_t[None, :].expand(len(qx), len(qy))
        W = wx_t[:, None] * wy_t[None, :] * L
        for tv in times:
            z = torch.stack([X.reshape(-1), Y.reshape(-1), torch.zeros(X.numel(), device=device, dtype=dtype) + tv], 1)
            with torch.no_grad():
                j = model.region(z, k)['j'].reshape(len(qx), len(qy))
            integ = aa * torch.sum(W * j)
            I = float(current_profile(tv))
            target = sgn * I * CFG.Ly
            rel.append(abs(float(integ.detach().cpu()) - target) / max(abs(target), 1e-12))
        add(f'reaction_integral_{k}_mean_rel', np.mean(rel))
        add(f'reaction_integral_{k}_max_rel', np.max(rel), 'max')
    qr, wr = np.polynomial.legendre.leggauss(48)
    rr = torch.tensor(0.5 * (qr + 1), device=device, dtype=dtype)
    wr = torch.tensor(0.5 * wr, device=device, dtype=dtype)
    for k, xv in [('n', 0.5 * CFG.x_n_end), ('p', 0.5 * (CFG.x_s_end + CFG.Lx))]:
        center = []
        surf = []
        moment = []
        Ds = CFG.D_s_n if k == 'n' else CFG.D_s_p
        chi = particle_flux_factor(k)
        for tv in times:
            z = torch.tensor([[xv, 0.5 * CFG.Ly, tv]], device=device, dtype=dtype)
            f = model.region(z, k)
            j = f['j'].detach()
            for rv, bag in [(0.0, center), (1.0, surf)]:
                zr = torch.tensor([[xv, 0.5 * CFG.Ly, rv, tv]], device=device, dtype=dtype, requires_grad=True)
                th = model.particle(zr, k, j=j)
                dr = _torch_grad(th, zr)[:, 2:3]
                val = dr if rv == 0.0 else -Ds * dr - chi * j
                bag.append(float(val.detach().cpu()))
            zr = torch.stack([torch.zeros_like(rr) + xv, torch.zeros_like(rr) + 0.5 * CFG.Ly, rr, torch.zeros_like(rr) + tv], 1)
            with torch.no_grad():
                th = model.particle(zr, k, j=torch.zeros_like(rr[:, None]) + j).reshape(-1)
            avg = 3.0 * torch.sum(wr * th * rr * rr)
            moment.append(float((avg - f['theta_avg'].reshape(())).detach().cpu()))
        add(f'particle_center_{k}_abs_rms', np.sqrt(np.mean(np.asarray(center) ** 2)), units='theta/r')
        add(f'particle_surface_flux_{k}_abs_rms', np.sqrt(np.mean(np.asarray(surf) ** 2)), units='flux')
        add(f'particle_mean_identity_{k}_abs_rms', np.sqrt(np.mean(np.asarray(moment) ** 2)), units='theta')
    qx, wx = np.polynomial.legendre.leggauss(16)
    qy, wy = np.polynomial.legendre.leggauss(7)
    qx_t = torch.tensor(0.5 * (qx + 1.0), device=device, dtype=dtype)
    wx_t = torch.tensor(0.5 * wx, device=device, dtype=dtype)
    qy_t = torch.tensor(0.5 * (qy + 1.0) * CFG.Ly, device=device, dtype=dtype)
    wy_t = torch.tensor(0.5 * wy * CFG.Ly, device=device, dtype=dtype)
    for k in ('n', 'p'):
        x0, x1 = (0.0, CFG.x_n_end) if k == 'n' else (CFG.x_s_end, CFG.Lx)
        L = x1 - x0
        aa = CFG.a_n if k == 'n' else CFG.a_p
        Cs = CFG.Csolid_n if k == 'n' else CFG.Csolid_p
        X = x0 + L * qx_t[:, None].expand(len(qx), len(qy))
        Y = qy_t[None, :].expand(len(qx), len(qy))
        W = wx_t[:, None] * wy_t[None, :] * L
        inventory = []
        surfcorr = []
        moment_res = []
        moment_scale = max(CFG.gamma_li * abs(CFG.I_app) / L, 1e-12)
        for tv in times:
            z = torch.stack([X.reshape(-1), Y.reshape(-1), torch.zeros(X.numel(), device=device, dtype=dtype) + tv], 1).requires_grad_(True)
            f = model.region(z, k)
            avg = f['theta_avg'].reshape(len(qx), len(qy))
            base_avg, base_surf = model._particle_baselines(torch.tensor([[tv]], device=device, dtype=dtype), k)
            avg_int = torch.sum(W * avg) / (L * CFG.Ly)
            inventory.append(float((avg_int - base_avg.reshape(())).detach().cpu()))
            surfcorr.append((f['theta_surf'] - base_surf).detach().cpu().numpy().ravel())
            dthdt = _torch_grad(f['theta_avg'], z)[:, 2:3]
            rmom = (Cs * dthdt + CFG.gamma_li * aa * f['j']) / moment_scale
            moment_res.append(rmom.detach().cpu().numpy().ravel())
        inv = np.asarray(inventory)
        sc = np.concatenate(surfcorr)
        mr = np.concatenate(moment_res)
        add(f'particle_inventory_baseline_{k}_abs_rms', np.sqrt(np.mean(inv * inv)), units='theta')
        add(f'particle_inventory_baseline_{k}_abs_max', np.max(np.abs(inv)), 'max', 'theta')
        add(f'particle_surface_correction_{k}_abs_rms', np.sqrt(np.mean(sc * sc)), units='theta')
        add(f'particle_moment_conservation_{k}_rms', np.sqrt(np.mean(mr * mr)))
    for k in ('n', 's', 'p'):
        x0, x1 = (0.0, CFG.x_n_end) if k == 'n' else (CFG.x_n_end, CFG.x_s_end) if k == 's' else (CFG.x_s_end, CFG.Lx)
        xs = torch.linspace(x0, x1, 9, device=device, dtype=dtype)
        ys2 = torch.linspace(0.0, CFG.Ly, 5, device=device, dtype=dtype)
        corr = []
        for tv in times:
            X, Y = torch.meshgrid(xs, ys2, indexing='ij')
            z = torch.stack([X.reshape(-1), Y.reshape(-1), torch.zeros(X.numel(), device=device, dtype=dtype) + tv], 1)
            with torch.no_grad():
                f = model.region(z, k)
                base = model._psi_base(z[:, 0:1], z[:, 2:3], k)
            corr.append(((f['psi_e'] - base) / max(float(model.phi_scale), 1e-12)).detach().cpu().numpy().ravel())
        a = np.concatenate(corr)
        add(f'psi_correction_{k}_rms_over_phi_scale', np.sqrt(np.mean(a * a)))
        add(f'psi_correction_{k}_max_over_phi_scale', np.max(np.abs(a)), 'max')
    write_rows(Path(outdir) / 'physics_constraint_audit.csv', rows)
    return rows

def compare(ref_path, model_path, outdir='outputs/comparison', nr_eval=24):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    R = np.load(ref_path)
    x, y, t = (R['x'], R['y'], R['t'])
    ref = {k: R[k] for k in R.files if k not in ('x', 'y', 't')}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, ck = load_model(model_path, device)
    dtype = next(model.parameters()).dtype
    audit_rows = physics_constraint_audit(model, outdir, device, dtype)
    pred, g = evaluate_model(model, x, y, t, device, dtype, nr_eval)
    pred = align_potential_gauge(pred, ref)
    pred = derive(pred, g, x, y, t)
    np.savez_compressed(outdir / 'model_fields_on_reference_grid.npz', x=x, y=y, t=t, **pred)
    metrics = []
    regions = {'full': np.ones((len(y), len(x)), bool), 'negative': g['mn'], 'separator': g['ms'], 'positive': g['mp']}
    fields = ['ce', 'T', 'phi_e', 'phi_sn', 'phi_sp', 'theta_n_surf', 'theta_p_surf', 'theta_n_avg', 'theta_p_avg', 'jn', 'jp', 'q_gen']
    for field in fields:
        if field not in ref:
            continue
        for rname, mask2 in regions.items():
            valid = np.broadcast_to(mask2, ref[field].shape) & np.isfinite(ref[field]) & np.isfinite(pred[field])
            if not np.any(valid):
                continue
            rmse, mse, mae, rel, linf = field_metric(ref[field], pred[field], valid)
            metrics.append({'field': field, 'region': rname, 'RMSE': rmse, 'MSE': mse, 'MAE': mae, 'relative_L2': rel, 'Linf': linf})
    write_rows(outdir / 'field_metrics.csv', metrics)
    pg, pr = diagnostics(pred, g, x, y, t, 'model', outdir)
    refdir = Path(ref_path).parent
    pairs = [('global_cell', pg), ('negative_electrode', pr['negative']), ('separator', pr['separator']), ('positive_electrode', pr['positive'])]
    traj = []
    for fname, prows in pairs:
        rfile = refdir / (fname + '.csv')
        if not rfile.exists():
            continue
        with open(rfile, encoding='utf-8') as f:
            rrows = list(csv.DictReader(f))
        common = set(prows[0]).intersection(rrows[0])
        common.discard('time')
        for key in sorted(common):
            try:
                r = np.array([float(q[key]) for q in rrows])
                p = np.array([float(q[key]) for q in prows])
            except Exception:
                continue
            good = np.isfinite(r) & np.isfinite(p)
            if not np.any(good):
                continue
            d = p[good] - r[good]
            refnorm = float(np.linalg.norm(r[good]))
            zero_target = key.endswith('_balance_rel') or refnorm < 1e-10
            rel = float('nan') if zero_target else float(np.linalg.norm(d) / refnorm)
            traj.append({'group': fname, 'quantity': key, 'RMSE': float(np.sqrt(np.mean(d * d))), 'relative_L2': rel, 'Linf': float(np.max(np.abs(d)))})
    write_rows(outdir / 'trajectory_metrics.csv', traj)
    summary = {'model': ck.get('architecture', {}), 'reference': str(ref_path), 'zero_reference_supervision': True, 'field_metric_count': len(metrics), 'trajectory_metric_count': len(traj), 'physics_constraint_audit_count': len(audit_rows), 'device': device, 'evaluation_state': ck.get('evaluation_state'), 'deterministic_physics_validation': ck.get('deterministic_physics_validation', {})}
    (outdir / 'comparison_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    return summary

def read_csv(path):
    with open(path, encoding='utf-8') as f:
        return list(csv.DictReader(f))

def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = fields or list(rows[0].keys())
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

def selected_time_indices(t):
    targets = [0.0, CFG.current_ramp_time, 0.25 * CFG.t_final, 0.5 * CFG.t_final, 0.75 * CFG.t_final, CFG.t_final]
    return sorted(set((int(np.argmin(np.abs(t - q))) for q in targets)))

def _region_x_indices(x, g, name):
    if name == 'negative':
        return np.arange(0, g['i_n'])
    if name == 'separator':
        return np.arange(g['i_n'], g['i_p'])
    if name == 'positive':
        return np.arange(g['i_p'], len(x))
    return np.arange(len(x))

def export_profiles(npz_path, outdir, label):
    a = np.load(npz_path)
    x, y, t = (a['x'], a['y'], a['t'])
    g = region_layout(len(x), len(y))
    tids = selected_time_indices(t)
    fields = [k for k in ['ce', 'phi_e', 'T', 'phi_sn', 'phi_sp', 'jn', 'jp', 'theta_n_surf', 'theta_p_surf', 'eta_n', 'eta_p', 'q_gen', 'q_rxn', 'q_ohm_e', 'q_ohm_s', 'q_rev'] if k in a.files]
    outdir = Path(outdir)
    for region in ['fullcell', 'negative', 'separator', 'positive']:
        idx = _region_x_indices(x, g, region if region != 'fullcell' else 'full')
        rows = []
        for it in tids:
            for ix in idx:
                r = {'source': label, 'time': float(t[it]), 'x': float(x[ix])}
                for fld in fields:
                    vals = np.asarray(a[fld][it, :, ix], float)
                    good = np.isfinite(vals)
                    r[fld] = float(np.mean(vals[good])) if np.any(good) else np.nan
                rows.append(r)
        family = {'fullcell': '04_fullcell_profiles', 'negative': '05_negative_profiles', 'separator': '06_separator_profiles', 'positive': '07_positive_profiles'}[region]
        write_csv(outdir / family / f'{label}_ymean_x_profiles.csv', rows)
    iy = int(np.argmin(np.abs(y - 0.5 * CFG.Ly)))
    rows = []
    for it in tids:
        for ix, xx in enumerate(x):
            r = {'source': label, 'time': float(t[it]), 'y': float(y[iy]), 'x': float(xx)}
            for fld in fields:
                v = float(a[fld][it, iy, ix])
                r[fld] = v
            rows.append(r)
    write_csv(outdir / '04_fullcell_profiles' / f'{label}_midplane_x_profiles.csv', rows)

def export_xt_manifest(ref_npz, pred_npz, outdir):
    r = np.load(ref_npz)
    fields = r.files
    g = region_layout(len(r['x']), len(r['y']))
    families = {'00_fullcell_common_fields': ['ce', 'phi_e', 'T', 'q_gen', 'q_ohm_e', 'i_e_x', 'i_e_y'], '01_negative_fields': ['ce', 'phi_e', 'phi_sn', 'T', 'jn', 'theta_n_surf', 'theta_n_avg', 'eta_n', 'i0_n', 'q_gen', 'q_rxn', 'q_ohm_s', 'q_rev', 'i_s_n_x', 'i_s_n_y'], '02_separator_fields': ['ce', 'phi_e', 'T', 'q_gen', 'q_ohm_e', 'i_e_x', 'i_e_y'], '03_positive_fields': ['ce', 'phi_e', 'phi_sp', 'T', 'jp', 'theta_p_surf', 'theta_p_avg', 'eta_p', 'i0_p', 'q_gen', 'q_rxn', 'q_ohm_s', 'q_rev', 'i_s_p_x', 'i_s_p_y']}
    manifest = {'reference_fields_npz': str(Path(ref_npz).resolve()), 'fain_fields_npz': str(Path(pred_npz).resolve()) if pred_npz else None, 'note': 'Large x-y-t fields remain in authoritative NPZ archives; this manifest defines paper field families without duplicating them.', 'families': {}}
    for fam, fl in families.items():
        use = [q for q in fl if q in fields]
        manifest['families'][fam] = {'fields': use, 'recommended_region': fam, 'reference_available': True, 'fain_available': bool(pred_npz)}
        (Path(outdir) / fam).mkdir(parents=True, exist_ok=True)
    (Path(outdir) / 'FIELD_FAMILY_MANIFEST.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

def export_time_curves(refdir, compdir, outdir):
    outdir = Path(outdir) / '10_time_curves'
    outdir.mkdir(parents=True, exist_ok=True)
    groups = ['global_cell', 'negative_electrode', 'separator', 'positive_electrode']
    count = 0
    for group in groups:
        rr = read_csv(Path(refdir) / (group + '.csv'))
        shutil.copy2(Path(refdir) / (group + '.csv'), outdir / f'reference_{group}.csv')
        fain_file = Path(compdir) / f"fain_{group.replace('_electrode', '')}.csv" if compdir else None
        if group == 'global_cell':
            fain_file = Path(compdir) / 'fain_global_cell.csv' if compdir else None
        elif group == 'negative_electrode':
            fain_file = Path(compdir) / 'fain_negative.csv' if compdir else None
        elif group == 'positive_electrode':
            fain_file = Path(compdir) / 'fain_positive.csv' if compdir else None
        elif group == 'separator':
            fain_file = Path(compdir) / 'fain_separator.csv' if compdir else None
        if fain_file and fain_file.exists():
            pp = read_csv(fain_file)
            common = [k for k in rr[0] if k in pp[0] and k != 'time']
            rows = []
            for r, p in zip(rr, pp):
                row = {'time': float(r['time'])}
                for k in common:
                    try:
                        rv = float(r[k])
                        pv = float(p[k])
                    except Exception:
                        continue
                    row[f'ref_{k}'] = rv
                    row[f'fain_{k}'] = pv
                    row[f'abs_error_{k}'] = abs(pv - rv) if np.isfinite(rv) and np.isfinite(pv) else np.nan
                rows.append(row)
            write_csv(outdir / f'comparison_{group}.csv', rows)
            count += len(common)
    return count

def export_particle_data(refdir, model_path, outdir):
    outdir = Path(outdir)
    p = np.load(Path(refdir) / 'particle_snapshots.npz')
    R = np.load(Path(refdir) / 'reference_fields.npz')
    x, y = (R['x'], R['y'])
    g = region_layout(len(x), len(y))
    nr = p['theta_n'].shape[-1]
    r = (np.arange(nr) + 0.5) / nr
    iy = int(np.argmin(abs(y - 0.5 * CFG.Ly)))
    nmap = np.nanmean(p['theta_n'], axis=1)
    pmap = np.nanmean(p['theta_p'], axis=1)
    np.savez_compressed(outdir / '08_particle_maps' / 'reference_particle_xr_maps.npz', t=p['t'], r=r, x_n=x[:g['i_n']], x_p=x[g['i_p']:], theta_n=nmap, theta_p=pmap)
    ixn = g['i_n'] // 2
    ixp = g['i_p'] + (len(x) - g['i_p']) // 2
    rows = []
    for it, tt in enumerate(p['t']):
        for ir, rv in enumerate(r):
            rows.append({'source': 'reference', 'electrode': 'negative', 'time': float(tt), 'x': float(x[ixn]), 'y': float(y[iy]), 'r_over_R': float(rv), 'theta': float(p['theta_n'][it, iy, ixn, ir])})
            localp = ixp - g['i_p']
            rows.append({'source': 'reference', 'electrode': 'positive', 'time': float(tt), 'x': float(x[ixp]), 'y': float(y[iy]), 'r_over_R': float(rv), 'theta': float(p['theta_p'][it, iy, localp, ir])})
    if model_path and Path(model_path).exists():
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        m, _ = load_model(model_path, device)
        dtype = next(m.parameters()).dtype
        rplot = np.linspace(0, 1, 101)
        for k, ix in [('n', ixn), ('p', ixp)]:
            for tt in p['t']:
                pt = np.array([[x[ix], y[iy], tt]], float)
                vals = _eval_particle(m, pt, k, rplot, device, dtype)[0]
                for rv, v in zip(rplot, vals):
                    rows.append({'source': 'fain', 'electrode': 'negative' if k == 'n' else 'positive', 'time': float(tt), 'x': float(x[ix]), 'y': float(y[iy]), 'r_over_R': float(rv), 'theta': float(v)})
    write_csv(outdir / '09_particle_profiles' / 'representative_radial_profiles.csv', rows)

def export_paper_data(reference, comparison=None, model=None, outdir='outputs/paper_data'):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for d in ['00_fullcell_common_fields', '01_negative_fields', '02_separator_fields', '03_positive_fields', '04_fullcell_profiles', '05_negative_profiles', '06_separator_profiles', '07_positive_profiles', '08_particle_maps', '09_particle_profiles', '10_time_curves']:
        (outdir / d).mkdir(parents=True, exist_ok=True)
    ref = Path(reference)
    refdir = ref.parent
    pred = Path(comparison) / 'model_fields_on_reference_grid.npz' if comparison and (Path(comparison) / 'model_fields_on_reference_grid.npz').exists() else None
    export_xt_manifest(ref, pred, outdir)
    export_profiles(ref, outdir, 'reference')
    if pred:
        export_profiles(pred, outdir, 'fain')
    curve_count = export_time_curves(refdir, comparison, outdir)
    export_particle_data(refdir, model, outdir)
    summary = {'reference': str(ref), 'comparison': str(comparison) if comparison else None, 'model': str(model) if model else None, 'field_families': 4, 'primary_region_field_groups': 3, 'profile_families': 4, 'time_curve_quantities_with_fain_error': curve_count, 'particle_maps': True, 'particle_profiles': True, 'plotting_style_deferred_to_codex': True}
    (outdir / 'PAPER_DATA_SUMMARY.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    return summary
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference', default='outputs/reference/reference_fields.npz')
    ap.add_argument('--model', default='outputs/fain/model.pt')
    ap.add_argument('--outdir', default='outputs/comparison')
    ap.add_argument('--nr-eval', type=int, default=24)
    ap.add_argument('--paper-data-outdir', default=None, help='If set, also export compact plotting data views.')
    a = ap.parse_args()
    compare(a.reference, a.model, a.outdir, a.nr_eval)
    if a.paper_data_outdir:
        export_paper_data(a.reference, a.outdir, a.model, a.paper_data_outdir)