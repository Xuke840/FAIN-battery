from __future__ import annotations
from pathlib import Path
import argparse, csv, json, math, time
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags, eye, lil_matrix
from scipy.sparse.linalg import splu, spsolve
from scipy.optimize import brentq
from physics import CFG, config_dict, current_profile, region_layout, epsilon_np, D_e_np, kappa_np, kT_np, h_top_np, h_bottom_np, ocp_n_np, ocp_p_np, exchange_current_np, particle_flux_factor

def harmonic(a, b):
    return 2.0 * a * b / np.clip(a + b, 1e-30, None)

def fv_matrix(k, dx, dy):
    k = np.asarray(k, dtype=float)
    ny, nx = k.shape
    N = nx * ny
    rows, cols, data = ([], [], [])
    diag = np.zeros(N, dtype=float)
    if nx > 1:
        jj, ii = np.meshgrid(np.arange(ny), np.arange(nx - 1), indexing='ij')
        p = (jj * nx + ii).ravel()
        q = (jj * nx + ii + 1).ravel()
        c = (harmonic(k[:, :-1], k[:, 1:]) / dx ** 2).ravel()
        diag[p] += c
        diag[q] += c
        rows.extend(p.tolist())
        cols.extend(q.tolist())
        data.extend((-c).tolist())
        rows.extend(q.tolist())
        cols.extend(p.tolist())
        data.extend((-c).tolist())
    if ny > 1:
        jj, ii = np.meshgrid(np.arange(ny - 1), np.arange(nx), indexing='ij')
        p = (jj * nx + ii).ravel()
        q = ((jj + 1) * nx + ii).ravel()
        c = (harmonic(k[:-1, :], k[1:, :]) / dy ** 2).ravel()
        diag[p] += c
        diag[q] += c
        rows.extend(p.tolist())
        cols.extend(q.tolist())
        data.extend((-c).tolist())
        rows.extend(q.tolist())
        cols.extend(p.tolist())
        data.extend((-c).tolist())
    idx = np.arange(N)
    rows.extend(idx.tolist())
    cols.extend(idx.tolist())
    data.extend(diag.tolist())
    return coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()

def divergence_grad(g, k, dx, dy):
    g = np.asarray(g, dtype=float)
    k = np.asarray(k, dtype=float)
    ny, nx = g.shape
    out = np.zeros_like(g)
    if nx > 1:
        cf = harmonic(k[:, :-1], k[:, 1:])
        f = cf * (g[:, 1:] - g[:, :-1]) / dx
        out[:, :-1] += f / dx
        out[:, 1:] -= f / dx
    if ny > 1:
        cf = harmonic(k[:-1, :], k[1:, :])
        f = cf * (g[1:, :] - g[:-1, :]) / dy
        out[:-1, :] += f / dy
        out[1:, :] -= f / dy
    return out

def prepare_neumann_solver(A):
    M = A.tolil(copy=True)
    M[0, :] = 0.0
    M[0, 0] = 1.0
    return splu(M.tocsc())

def solve_neumann(lu, rhs, gauge=0.0):
    b = np.asarray(rhs, dtype=float).ravel().copy()
    correction = float(np.mean(b))
    b -= correction
    b[0] = gauge
    return (lu.solve(b).reshape(np.asarray(rhs).shape), correction)

def radial_operator(nr, D):
    dr = 1.0 / nr
    faces = np.arange(nr + 1, dtype=float) * dr
    volumes = (faces[1:] ** 3 - faces[:-1] ** 3) / 3.0
    A = lil_matrix((nr, nr), dtype=float)
    for i in range(nr - 1):
        g = D * faces[i + 1] ** 2 / dr
        A[i, i] += g / volumes[i]
        A[i, i + 1] -= g / volumes[i]
        A[i + 1, i] -= g / volumes[i + 1]
        A[i + 1, i + 1] += g / volumes[i + 1]
    return (A.tocsr(), volumes)

def radial_average(theta, shell_volumes):
    return 3.0 * np.sum(theta * shell_volumes, axis=-1)

def scalar_bv_shift(base_eta, i0, T, target_integral, a, dx, dy, cfg=CFG):

    def residual(delta):
        raw = cfg.beta_bv * (base_eta + delta) / np.clip(T, 0.5, 2.0)
        arg = np.clip(raw, -cfg.bv_argument_guard, cfg.bv_argument_guard)
        j = 2.0 * i0 * np.sinh(arg)
        return a * np.sum(j) * dx * dy - target_integral
    lo, hi = (-1.0, 1.0)
    flo, fhi = (residual(lo), residual(hi))
    for _ in range(12):
        if flo * fhi <= 0.0:
            break
        lo *= 2.0
        hi *= 2.0
        flo, fhi = (residual(lo), residual(hi))
    if flo * fhi > 0.0:
        raise RuntimeError('Could not bracket galvanostatic BV electrode offset.')
    delta = brentq(residual, lo, hi, xtol=1e-13, rtol=1e-13, maxiter=200)
    raw = cfg.beta_bv * (base_eta + delta) / np.clip(T, 0.5, 2.0)
    guard_count = int(np.count_nonzero(np.abs(raw) > cfg.bv_argument_guard))
    j = 2.0 * i0 * np.sinh(np.clip(raw, -cfg.bv_argument_guard, cfg.bv_argument_guard))
    return (delta, j, guard_count, raw)

class ReferenceSolver:

    def __init__(self, nx=50, ny=20, nt=81, nr=16, cfg=CFG):
        if min(nx, ny, nt, nr) < 3:
            raise ValueError('Grid counts are too small.')
        self.cfg = cfg
        self.nx, self.ny, self.nt, self.nr = (int(nx), int(ny), int(nt), int(nr))
        self.g = region_layout(self.nx, self.ny, cfg)
        self.dx, self.dy = (self.g['dx'], self.g['dy'])
        self.nn, self.ip = (self.g['i_n'], self.g['i_p'])
        self.npcols = self.nx - self.ip
        self.t = np.linspace(0.0, cfg.t_final, self.nt)
        self.dt = self.t[1] - self.t[0]
        self.A_sn = fv_matrix(np.full((self.ny, self.nn), cfg.sigma_n), self.dx, self.dy)
        self.A_sp = fv_matrix(np.full((self.ny, self.npcols), cfg.sigma_p), self.dx, self.dy)
        self.lu_sn = prepare_neumann_solver(self.A_sn)
        self.lu_sp = prepare_neumann_solver(self.A_sp)
        self.Ar_n, self.rv_n = radial_operator(self.nr, cfg.D_s_n)
        self.Ar_p, self.rv_p = radial_operator(self.nr, cfg.D_s_p)
        self.lu_rn = splu((eye(self.nr, format='csr') + self.dt * self.Ar_n).tocsc())
        self.lu_rp = splu((eye(self.nr, format='csr') + self.dt * self.Ar_p).tocsc())
        kT = kT_np(self.g, cfg)
        A_T = fv_matrix(kT, self.dx, self.dy)
        N = self.nx * self.ny
        M_T = (diags(np.full(N, cfg.rho_cp / self.dt)) + A_T).tolil()
        x = self.g['x']
        ht = h_top_np(x, cfg)
        hb = h_bottom_np(x, cfg)
        for i in range(self.nx):
            ib = i
            it = (self.ny - 1) * self.nx + i
            M_T[ib, ib] += hb[i] / self.dy
            M_T[it, it] += ht[i] / self.dy
        self.lu_T = splu(M_T.tocsc())
        self.ht, self.hb = (ht, hb)

    def _rescale_previous_current(self, jprev, I):
        c = self.cfg
        j = np.zeros((self.ny, self.nx), dtype=float)
        if jprev is None or abs(I) < 1e-15:
            if abs(I) > 0:
                j[:, :self.nn] = -I / (c.a_n * self.nn * self.dx)
                j[:, self.ip:] = I / (c.a_p * self.npcols * self.dx)
            return j
        for sl, a, target in [((slice(None), slice(0, self.nn)), c.a_n, -I * c.Ly), ((slice(None), slice(self.ip, self.nx)), c.a_p, I * c.Ly)]:
            block = np.asarray(jprev[sl], dtype=float).copy()
            integ = a * np.sum(block) * self.dx * self.dy
            if abs(integ) < 1e-20:
                area_x = self.nn if sl[1].stop == self.nn else self.npcols
                block[:] = target / (a * area_x * self.dx * c.Ly)
            else:
                block *= target / integ
            j[sl] = block
        return j

    def solve_charge(self, ce, T, theta_n_edge, theta_p_edge, I, jprev=None, max_iter=100, tol=2e-10):
        c = self.cfg
        g = self.g
        mn, ms, mp = (g['mn'], g['ms'], g['mp'])
        kap = kappa_np(g, ce, T, c)
        dr = 1.0 / self.nr
        chi_n = particle_flux_factor('n', c)
        chi_p = particle_flux_factor('p', c)

        def reconstructed_surface(j_full):
            jn = j_full[:, :self.nn]
            jp = j_full[:, self.ip:]
            thn = theta_n_edge - 0.5 * dr * chi_n * jn / c.D_s_n
            thp = theta_p_edge - 0.5 * dr * chi_p * jp / c.D_s_p
            if np.min(thn) <= 0 or np.max(thn) >= 1 or np.min(thp) <= 0 or (np.max(thp) >= 1):
                raise RuntimeError('Reconstructed particle surface state left (0,1); no clipping is allowed.')
            return (thn, thp)
        if abs(I) < 1e-14:
            pe = np.zeros_like(ce)
            theta_n_surf = theta_n_edge.copy()
            theta_p_surf = theta_p_edge.copy()
            psn = ocp_n_np(theta_n_surf).copy()
            psp = ocp_p_np(theta_p_surf).copy()
            j = np.zeros_like(ce)
            return {'phi_e': pe, 'phi_sn_local': psn, 'phi_sp_local': psp, 'j': j, 'kappa': kap, 'iterations': 0, 'iteration_rel': 0.0, 'bv_fixed_point_rel': 0.0, 'bv_guard_count': 0, 'charge_residual_rel': 0.0, 'compatibility_correction': 0.0, 'i0n': exchange_current_np('n', ce[:, :self.nn], theta_n_surf, T[:, :self.nn], c), 'i0p': exchange_current_np('p', ce[:, self.ip:], theta_p_surf, T[:, self.ip:], c), 'theta_n_surf': theta_n_surf, 'theta_p_surf': theta_p_surf}
        A_e = fv_matrix(kap, self.dx, self.dy)
        lu_e = prepare_neumann_solver(A_e)
        j = self._rescale_previous_current(jprev, I)
        pe = np.zeros_like(ce)
        psn0 = np.zeros((self.ny, self.nn))
        psp0 = np.zeros((self.ny, self.npcols))
        delta_n = delta_p = 0.0
        guard_count = 0
        rel = np.inf
        corr_e = corr_n = corr_p = 0.0
        for it in range(1, max_iter + 1):
            reaction_source = c.a_n * j * mn + c.a_p * j * mp
            div_known = divergence_grad(np.log(np.clip(ce, 1e-12, None)), c.chi_ln_ce * kap, self.dx, self.dy)
            rhs_e = reaction_source - div_known
            pe, corr_e = solve_neumann(lu_e, rhs_e, gauge=0.0)
            rhs_n = -c.a_n * j[:, :self.nn].copy()
            rhs_n[:, 0] += -I / self.dx
            psn0, corr_n = solve_neumann(self.lu_sn, rhs_n, gauge=0.0)
            rhs_p = -c.a_p * j[:, self.ip:].copy()
            rhs_p[:, -1] -= -I / self.dx
            psp0, corr_p = solve_neumann(self.lu_sp, rhs_p, gauge=0.0)
            theta_n_surf, theta_p_surf = reconstructed_surface(j)
            i0n = exchange_current_np('n', ce[:, :self.nn], theta_n_surf, T[:, :self.nn], c)
            i0p = exchange_current_np('p', ce[:, self.ip:], theta_p_surf, T[:, self.ip:], c)
            base_n = psn0 - pe[:, :self.nn] - ocp_n_np(theta_n_surf)
            base_p = psp0 - pe[:, self.ip:] - ocp_p_np(theta_p_surf)
            delta_n, jn, gn, raw_n = scalar_bv_shift(base_n, i0n, T[:, :self.nn], -I * c.Ly, c.a_n, self.dx, self.dy, c)
            delta_p, jp, gp, raw_p = scalar_bv_shift(base_p, i0p, T[:, self.ip:], I * c.Ly, c.a_p, self.dx, self.dy, c)
            guard_count = gn + gp
            j_new = np.zeros_like(j)
            j_new[:, :self.nn] = jn
            j_new[:, self.ip:] = jp
            rel = float(np.linalg.norm(j_new - j) / max(np.linalg.norm(j_new), 1e-14))
            if rel < tol:
                j = j_new
                break
            j = 0.55 * j + 0.45 * j_new
        else:
            raise RuntimeError(f'Charge iteration did not converge: rel={rel:.3e}')
        reaction_source = c.a_n * j * mn + c.a_p * j * mp
        div_known = divergence_grad(np.log(np.clip(ce, 1e-12, None)), c.chi_ln_ce * kap, self.dx, self.dy)
        rhs_e = reaction_source - div_known
        pe, corr_e = solve_neumann(lu_e, rhs_e, gauge=0.0)
        rhs_n = -c.a_n * j[:, :self.nn].copy()
        rhs_n[:, 0] += -I / self.dx
        psn0, corr_n = solve_neumann(self.lu_sn, rhs_n, gauge=0.0)
        rhs_p = -c.a_p * j[:, self.ip:].copy()
        rhs_p[:, -1] -= -I / self.dx
        psp0, corr_p = solve_neumann(self.lu_sp, rhs_p, gauge=0.0)
        i0n = exchange_current_np('n', ce[:, :self.nn], theta_n_surf, T[:, :self.nn], c)
        i0p = exchange_current_np('p', ce[:, self.ip:], theta_p_surf, T[:, self.ip:], c)
        base_n = psn0 - pe[:, :self.nn] - ocp_n_np(theta_n_surf)
        base_p = psp0 - pe[:, self.ip:] - ocp_p_np(theta_p_surf)
        delta_n, jn_chk, gn, raw_n = scalar_bv_shift(base_n, i0n, T[:, :self.nn], -I * c.Ly, c.a_n, self.dx, self.dy, c)
        delta_p, jp_chk, gp, raw_p = scalar_bv_shift(base_p, i0p, T[:, self.ip:], I * c.Ly, c.a_p, self.dx, self.dy, c)
        j_chk = np.zeros_like(j)
        j_chk[:, :self.nn] = jn_chk
        j_chk[:, self.ip:] = jp_chk
        bv_fixed_point_rel = float(np.linalg.norm(j_chk - j) / max(np.linalg.norm(j_chk), 1e-14))
        j = j_chk
        guard_count = gn + gp
        theta_n_surf, theta_p_surf = reconstructed_surface(j)
        psn = psn0 + delta_n
        psp = psp0 + delta_p
        re = (A_e @ pe.ravel()).reshape(pe.shape) - rhs_e
        rn = (self.A_sn @ psn0.ravel()).reshape(psn0.shape) - rhs_n
        rp = (self.A_sp @ psp0.ravel()).reshape(psp0.shape) - rhs_p
        re.flat[0] = 0.0
        rn.flat[0] = 0.0
        rp.flat[0] = 0.0
        scale = max(abs(I) / max(self.dx, 1e-12), 1.0)
        charge_res = float(max(np.max(np.abs(re)), np.max(np.abs(rn)), np.max(np.abs(rp))) / scale)
        compatibility = float(max(abs(corr_e), abs(corr_n), abs(corr_p)))
        return {'phi_e': pe, 'phi_sn_local': psn, 'phi_sp_local': psp, 'j': j, 'kappa': kap, 'iterations': it, 'iteration_rel': rel, 'bv_fixed_point_rel': bv_fixed_point_rel, 'bv_guard_count': guard_count, 'charge_residual_rel': charge_res, 'compatibility_correction': compatibility, 'i0n': i0n, 'i0p': i0p, 'theta_n_surf': theta_n_surf, 'theta_p_surf': theta_p_surf}

    def radial_step(self, theta, k, j_local):
        c = self.cfg
        if k == 'n':
            lu, vols, chi = (self.lu_rn, self.rv_n, particle_flux_factor('n', c))
        else:
            lu, vols, chi = (self.lu_rp, self.rv_p, particle_flux_factor('p', c))
        flat = theta.reshape(-1, self.nr)
        rhs = flat.copy()
        rhs[:, -1] -= self.dt * chi * j_local.reshape(-1) / vols[-1]
        new = lu.solve(rhs.T).T.reshape(theta.shape)
        return new

    def derived_fields(self, ce, T, theta_n_surf, theta_p_surf, theta_n_avg, theta_p_avg, charge):
        c = self.cfg
        g = self.g
        dx, dy = (self.dx, self.dy)
        mn, ms, mp = (g['mn'], g['ms'], g['mp'])
        pe = charge['phi_e']
        psn = charge['phi_sn_local']
        psp = charge['phi_sp_local']
        j = charge['j']
        kap = charge['kappa']
        jn = j[:, :self.nn]
        jp = j[:, self.ip:]
        dpey, dpex = np.gradient(pe, dy, dx, edge_order=1)
        lnce = np.log(np.clip(ce, 1e-12, None))
        dlcy, dlcx = np.gradient(lnce, dy, dx, edge_order=1)
        iex = -kap * dpex + c.chi_ln_ce * kap * dlcx
        iey = -kap * dpey + c.chi_ln_ce * kap * dlcy
        dny, dnx = np.gradient(psn, dy, dx, edge_order=1)
        dpy, dpx = np.gradient(psp, dy, dx, edge_order=1)
        isnx = -c.sigma_n * dnx
        isny = -c.sigma_n * dny
        ispx = -c.sigma_p * dpx
        ispy = -c.sigma_p * dpy
        eta_n = psn - pe[:, :self.nn] - ocp_n_np(theta_n_surf)
        eta_p = psp - pe[:, self.ip:] - ocp_p_np(theta_p_surf)
        q_rxn = np.zeros_like(ce)
        q_ohm_e = (iex ** 2 + iey ** 2) / np.clip(kap, 1e-12, None)
        q_ohm_s = np.zeros_like(ce)
        q_rev = np.zeros_like(ce)
        q_rxn[:, :self.nn] = jn * eta_n
        q_rxn[:, self.ip:] = jp * eta_p
        q_ohm_s[:, :self.nn] = (isnx ** 2 + isny ** 2) / c.sigma_n
        q_ohm_s[:, self.ip:] = (ispx ** 2 + ispy ** 2) / c.sigma_p
        q_rev[:, :self.nn] = c.entropic_n * jn * T[:, :self.nn]
        q_rev[:, self.ip:] = c.entropic_p * jp * T[:, self.ip:]
        q_gen = c.heat_scale * (q_rxn + q_ohm_e + q_ohm_s + q_rev)
        phi_sn = np.full_like(ce, np.nan)
        phi_sp = np.full_like(ce, np.nan)
        phi_sn[:, :self.nn] = psn
        phi_sp[:, self.ip:] = psp
        jn_full = np.full_like(ce, np.nan)
        jp_full = np.full_like(ce, np.nan)
        jn_full[:, :self.nn] = jn
        jp_full[:, self.ip:] = jp
        thns = np.full_like(ce, np.nan)
        thps = np.full_like(ce, np.nan)
        thna = np.full_like(ce, np.nan)
        thpa = np.full_like(ce, np.nan)
        thns[:, :self.nn] = theta_n_surf
        thps[:, self.ip:] = theta_p_surf
        thna[:, :self.nn] = theta_n_avg
        thpa[:, self.ip:] = theta_p_avg
        eta_n_full = np.full_like(ce, np.nan)
        eta_p_full = np.full_like(ce, np.nan)
        eta_n_full[:, :self.nn] = eta_n
        eta_p_full[:, self.ip:] = eta_p
        i0n_full = np.full_like(ce, np.nan)
        i0p_full = np.full_like(ce, np.nan)
        i0n_full[:, :self.nn] = charge['i0n']
        i0p_full[:, self.ip:] = charge['i0p']
        isnx_full = np.full_like(ce, np.nan)
        isny_full = np.full_like(ce, np.nan)
        ispx_full = np.full_like(ce, np.nan)
        ispy_full = np.full_like(ce, np.nan)
        isnx_full[:, :self.nn] = isnx
        isny_full[:, :self.nn] = isny
        ispx_full[:, self.ip:] = ispx
        ispy_full[:, self.ip:] = ispy
        return {'phi_e': pe, 'phi_sn': phi_sn, 'phi_sp': phi_sp, 'jn': jn_full, 'jp': jp_full, 'theta_n_surf': thns, 'theta_p_surf': thps, 'theta_n_avg': thna, 'theta_p_avg': thpa, 'eta_n': eta_n_full, 'eta_p': eta_p_full, 'i0_n': i0n_full, 'i0_p': i0p_full, 'i_e_x': iex, 'i_e_y': iey, 'i_s_n_x': isnx_full, 'i_s_n_y': isny_full, 'i_s_p_x': ispx_full, 'i_s_p_y': ispy_full, 'q_rxn': q_rxn, 'q_ohm_e': q_ohm_e, 'q_ohm_s': q_ohm_s, 'q_rev': q_rev, 'q_gen': q_gen}

    def lithium_inventory(self, ce, theta_n_avg, theta_p_avg):
        c = self.cfg
        eps = epsilon_np(self.g, c)
        return float((np.sum(eps * ce) + c.Csolid_n * np.sum(theta_n_avg) + c.Csolid_p * np.sum(theta_p_avg)) * self.dx * self.dy)

    def diagnostic_row(self, tt, I, ce, T, d, charge, Li0, thermal_balance_rel):
        c = self.cfg
        g = self.g
        x, y = (g['x'], g['y'])
        X, Y = np.meshgrid(x, y)
        mn, ms, mp = (g['mn'], g['ms'], g['mp'])
        jn = d['jn']
        jp = d['jp']
        thna = d['theta_n_avg'][:, :self.nn]
        thpa = d['theta_p_avg'][:, self.ip:]
        Li = self.lithium_inventory(ce, thna, thpa)
        Li_err = abs(Li - Li0) / max(abs(Li0), 1e-14)
        Qn = c.a_n * np.nansum(jn) * self.dx * self.dy
        Qp = c.a_p * np.nansum(jp) * self.dx * self.dy
        if abs(I) > 1e-12:
            denom = abs(I * c.Ly)
            current_err = max(abs(Qn + I * c.Ly), abs(Qp - I * c.Ly), abs(Qn + Qp)) / denom
        else:
            current_err = max(abs(Qn), abs(Qp))
        Vn = float(np.nanmean(d['phi_sn'][:, 0] - 0.5 * self.dx * (I / c.sigma_n)))
        Vp = float(np.nanmean(d['phi_sp'][:, -1] + 0.5 * self.dx * (I / c.sigma_p)))
        V = Vp - Vn
        return {'time': float(tt), 'I_app': float(I), 'V_n': Vn, 'V_p': Vp, 'V_cell': V, 'SOC_n': float(np.mean(thna)), 'SOC_p': float(np.mean(thpa)), 'Li_inventory': Li, 'Li_balance_rel': Li_err, 'current_balance_rel': float(current_err), 'thermal_balance_rel': float(thermal_balance_rel) if np.isfinite(thermal_balance_rel) else np.nan, 'charge_residual_rel': charge['charge_residual_rel'], 'bv_fixed_point_rel': charge['bv_fixed_point_rel'], 'compatibility_correction': charge['compatibility_correction'], 'bv_guard_count': charge['bv_guard_count'], 'charge_iterations': charge['iterations'], 'T_min': float(np.min(T)), 'T_max': float(np.max(T)), 'T_mean': float(np.mean(T)), 'ce_min': float(np.min(ce)), 'ce_max': float(np.max(ce))}

    def region_diagnostics(self, tt, I, ce, T, d):
        c = self.cfg
        g = self.g
        x, y = (g['x'], g['y'])
        X, Y = np.meshgrid(x, y)
        out = {}
        for key, mask, k in [('negative', g['mn'], 'n'), ('separator', g['ms'], 's'), ('positive', g['mp'], 'p')]:
            row = {'time': float(tt), 'I_app': float(I), 'ce_min': float(np.min(ce[mask])), 'ce_max': float(np.max(ce[mask])), 'ce_mean': float(np.mean(ce[mask])), 'phi_e_min': float(np.min(d['phi_e'][mask])), 'phi_e_max': float(np.max(d['phi_e'][mask])), 'phi_e_mean': float(np.mean(d['phi_e'][mask])), 'T_min': float(np.min(T[mask])), 'T_max': float(np.max(T[mask])), 'T_mean': float(np.mean(T[mask])), 'q_gen_max': float(np.max(d['q_gen'][mask])), 'q_gen_mean': float(np.mean(d['q_gen'][mask])), 'q_ohm_e_max': float(np.max(d['q_ohm_e'][mask])), 'i_e_abs_max': float(np.max(np.sqrt(d['i_e_x'][mask] ** 2 + d['i_e_y'][mask] ** 2)))}
            candidates = [ce[mask].ravel()]
            if key == 'negative' and self.nn >= 2:
                candidates += [(1.5 * ce[:, self.nn - 1] - 0.5 * ce[:, self.nn - 2]).ravel(), ce[:, 0].ravel()]
            elif key == 'separator' and self.ip - self.nn >= 2:
                candidates += [(1.5 * ce[:, self.nn] - 0.5 * ce[:, self.nn + 1]).ravel(), (1.5 * ce[:, self.ip - 1] - 0.5 * ce[:, self.ip - 2]).ravel()]
            elif key == 'positive' and self.npcols >= 2:
                candidates += [(1.5 * ce[:, self.ip] - 0.5 * ce[:, self.ip + 1]).ravel(), ce[:, -1].ravel()]
            ccat = np.concatenate(candidates)
            row['ce_min_reconstructed'] = float(np.min(ccat))
            row['ce_max_reconstructed'] = float(np.max(ccat))
            gy, gx = np.gradient(ce, self.dy, self.dx, edge_order=1)
            row['grad_ce_abs_max'] = float(np.max(np.sqrt(gx[mask] ** 2 + gy[mask] ** 2)))
            xs = np.unique(X[mask])
            prof = np.array([np.mean(ce[:, np.argmin(np.abs(x - xx))]) for xx in xs])
            mid = 0.5 * (float(np.min(prof)) + float(np.max(prof)))
            row['ce_front_x'] = float(xs[np.argmin(np.abs(prof - mid))])
            if k in ('n', 'p'):
                jfield = d['jn'] if k == 'n' else d['jp']
                ths = d['theta_n_surf'] if k == 'n' else d['theta_p_surf']
                tha = d['theta_n_avg'] if k == 'n' else d['theta_p_avg']
                eta = d['eta_n'] if k == 'n' else d['eta_p']
                i0 = d['i0_n'] if k == 'n' else d['i0_p']
                isx = d['i_s_n_x'] if k == 'n' else d['i_s_p_x']
                isy = d['i_s_n_y'] if k == 'n' else d['i_s_p_y']
                vals = jfield[mask]
                av = np.abs(vals)
                denom = float(np.sum(av))
                if denom > 1e-15:
                    cx = float(np.sum(X[mask] * av) / denom)
                    cy = float(np.sum(Y[mask] * av) / denom)
                    idx = int(np.argmax(av))
                    px = float(X[mask][idx])
                    py = float(Y[mask][idx])
                    area = float(np.sum(mask) * self.dx * self.dy)
                    loc = area * float(np.sum(vals ** 2) * self.dx * self.dy) / float(np.sum(av) * self.dx * self.dy) ** 2
                    expected = -1.0 if k == 'n' else 1.0
                    rev = float(np.sum(np.maximum(-expected * vals, 0.0)) / denom)
                else:
                    cx = cy = px = py = np.nan
                    loc = rev = np.nan
                row.update({'j_min': float(np.min(vals)), 'j_max': float(np.max(vals)), 'j_abs_max': float(np.max(av)), 'j_mean': float(np.mean(vals)), 'theta_surf_min': float(np.nanmin(ths[mask])), 'theta_surf_max': float(np.nanmax(ths[mask])), 'theta_surf_mean': float(np.nanmean(ths[mask])), 'theta_avg_mean': float(np.nanmean(tha[mask])), 'eta_mean': float(np.nanmean(eta[mask])), 'i0_mean': float(np.nanmean(i0[mask])), 'reaction_centroid_x': cx, 'reaction_centroid_y': cy, 'reaction_peak_x': px, 'reaction_peak_y': py, 'reaction_localization': loc, 'reverse_reaction_fraction': rev, 'solid_current_abs_max': float(np.nanmax(np.sqrt(isx[mask] ** 2 + isy[mask] ** 2))), 'q_rxn_mean': float(np.mean(d['q_rxn'][mask])), 'q_ohm_s_mean': float(np.mean(d['q_ohm_s'][mask])), 'q_rev_mean': float(np.mean(d['q_rev'][mask]))})
            out[key] = row
        return out

    def solve(self, outdir='outputs/reference', save_fields=True, verbose=True):
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        c = self.cfg
        g = self.g
        mn, ms, mp = (g['mn'], g['ms'], g['mp'])
        ce = np.full((self.ny, self.nx), c.c_e0, dtype=float)
        T = np.full_like(ce, c.T0)
        theta_n = np.full((self.ny, self.nn, self.nr), c.theta_n0, dtype=float)
        theta_p = np.full((self.ny, self.npcols, self.nr), c.theta_p0, dtype=float)
        theta_n_avg = radial_average(theta_n, self.rv_n)
        theta_p_avg = radial_average(theta_p, self.rv_p)
        Li0 = self.lithium_inventory(ce, theta_n_avg, theta_p_avg)
        jprev = None
        thermal_balance_rel = np.nan
        fields = {k: [] for k in ['ce', 'T', 'phi_e', 'phi_sn', 'phi_sp', 'jn', 'jp', 'theta_n_surf', 'theta_p_surf', 'theta_n_avg', 'theta_p_avg', 'eta_n', 'eta_p', 'i0_n', 'i0_p', 'i_e_x', 'i_e_y', 'i_s_n_x', 'i_s_n_y', 'i_s_p_x', 'i_s_p_y', 'q_rxn', 'q_ohm_e', 'q_ohm_s', 'q_rev', 'q_gen']}
        global_rows = []
        reg_rows = {'negative': [], 'separator': [], 'positive': []}
        snap_idx = sorted(set([0, self.nt // 4, self.nt // 2, 3 * self.nt // 4, self.nt - 1]))
        snap_t = []
        snap_n = []
        snap_p = []
        t0 = time.time()
        for m, tt in enumerate(self.t):
            I = float(current_profile(tt, c))
            thn_edge = theta_n[..., -1]
            thp_edge = theta_p[..., -1]
            theta_n_avg = radial_average(theta_n, self.rv_n)
            theta_p_avg = radial_average(theta_p, self.rv_p)
            charge = self.solve_charge(ce, T, thn_edge, thp_edge, I, jprev)
            jprev = charge['j'].copy()
            thns = charge['theta_n_surf']
            thps = charge['theta_p_surf']
            d = self.derived_fields(ce, T, thns, thps, theta_n_avg, theta_p_avg, charge)
            grow = self.diagnostic_row(tt, I, ce, T, d, charge, Li0, thermal_balance_rel)
            global_rows.append(grow)
            rr = self.region_diagnostics(tt, I, ce, T, d)
            for k in reg_rows:
                reg_rows[k].append(rr[k])
            if save_fields:
                fields['ce'].append(ce.copy())
                fields['T'].append(T.copy())
                for k in fields:
                    if k not in ('ce', 'T'):
                        fields[k].append(d[k].copy())
            if m in snap_idx:
                snap_t.append(tt)
                snap_n.append(theta_n.copy())
                snap_p.append(theta_p.copy())
            if verbose and (m == 0 or m == self.nt - 1 or m % max(1, self.nt // 10) == 0):
                print(f"[FVM] {m + 1:4d}/{self.nt} t={tt:.4f} V={grow['V_cell']:.6f} ce=[{grow['ce_min']:.5f},{grow['ce_max']:.5f}] LiErr={grow['Li_balance_rel']:.2e} IErr={grow['current_balance_rel']:.2e}")
            if m == self.nt - 1:
                break
            eps = epsilon_np(g, c)
            De = D_e_np(g, ce, T, c)
            A_D = fv_matrix(De, self.dx, self.dy)
            M = diags((eps / self.dt).ravel()) + A_D
            source = c.gamma_li * (c.a_n * charge['j'] * mn + c.a_p * charge['j'] * mp)
            rhs = (eps / self.dt * ce + source).ravel()
            ce_new = spsolve(M, rhs).reshape(self.ny, self.nx)
            theta_n_new = self.radial_step(theta_n, 'n', charge['j'][:, :self.nn])
            theta_p_new = self.radial_step(theta_p, 'p', charge['j'][:, self.ip:])
            rhsT = (c.rho_cp / self.dt * T + d['q_gen']).ravel()
            for i in range(self.nx):
                rhsT[i] += self.hb[i] * c.T_amb / self.dy
                rhsT[(self.ny - 1) * self.nx + i] += self.ht[i] * c.T_amb / self.dy
            T_new = self.lu_T.solve(rhsT).reshape(self.ny, self.nx)
            dE = c.rho_cp * np.sum(T_new - T) * self.dx * self.dy / self.dt
            Qgen = np.sum(d['q_gen']) * self.dx * self.dy
            Qcool = np.sum(self.ht * (T_new[-1, :] - c.T_amb)) * self.dx + np.sum(self.hb * (T_new[0, :] - c.T_amb)) * self.dx
            thermal_scale = abs(Qgen) + abs(Qcool) + abs(dE)
            thermal_balance_rel = 0.0 if thermal_scale < 1e-12 else abs(dE - (Qgen - Qcool)) / thermal_scale
            ce, T, theta_n, theta_p = (ce_new, T_new, theta_n_new, theta_p_new)
            if not (np.all(np.isfinite(ce)) and np.all(np.isfinite(T)) and np.all(np.isfinite(theta_n)) and np.all(np.isfinite(theta_p))):
                raise FloatingPointError('Non-finite state generated by reference solver.')
            if np.min(ce) <= 0 or np.min(theta_n) <= 0 or np.max(theta_n) >= 1 or (np.min(theta_p) <= 0) or (np.max(theta_p) >= 1) or (np.min(T) <= 0):
                raise RuntimeError('Reference state left its admissible physical range; no clipping is allowed.')
        runtime = time.time() - t0

        def write_dict_rows(path, rows):
            keys = list(rows[0].keys())
            with open(path, 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(rows)
        write_dict_rows(outdir / 'global_cell.csv', global_rows)
        write_dict_rows(outdir / 'negative_electrode.csv', reg_rows['negative'])
        write_dict_rows(outdir / 'separator.csv', reg_rows['separator'])
        write_dict_rows(outdir / 'positive_electrode.csv', reg_rows['positive'])
        if save_fields:
            arrays = {k: np.asarray(v) for k, v in fields.items()}
            np.savez_compressed(outdir / 'reference_fields.npz', x=g['x'], y=g['y'], t=self.t, **arrays)
        np.savez_compressed(outdir / 'particle_snapshots.npz', t=np.asarray(snap_t), theta_n=np.asarray(snap_n), theta_p=np.asarray(snap_p))
        G = global_rows
        summary = {'solver': 'conservative cell-centred FVM + spherical FV particles', 'nx': self.nx, 'ny': self.ny, 'nt': self.nt, 'nr': self.nr, 'dx': self.dx, 'dy': self.dy, 'dt': self.dt, 'represented_x_n_interface': g['x_n_face'], 'represented_x_s_interface': g['x_s_face'], 'runtime_seconds': runtime, 'max_Li_balance_rel': float(np.nanmax([r['Li_balance_rel'] for r in G])), 'max_current_balance_rel': float(np.nanmax([r['current_balance_rel'] for r in G])), 'max_thermal_balance_rel': float(np.nanmax([r['thermal_balance_rel'] for r in G[1:]])) if len(G) > 1 else np.nan, 'max_charge_residual_rel': float(np.nanmax([r['charge_residual_rel'] for r in G])), 'max_bv_fixed_point_rel': float(np.nanmax([r['bv_fixed_point_rel'] for r in G])), 'max_compatibility_correction': float(np.nanmax([r['compatibility_correction'] for r in G])), 'total_bv_guard_activations': int(np.nansum([r['bv_guard_count'] for r in G])), 'min_ce': float(np.min([r['ce_min'] for r in G])), 'max_ce': float(np.max([r['ce_max'] for r in G])), 'min_T': float(np.min([r['T_min'] for r in G])), 'max_T': float(np.max([r['T_max'] for r in G])), 'final_V_cell': float(G[-1]['V_cell']), 'note': 'Scaled benchmark parameterization; chemistry-specific calibration is a separate step.'}
        (outdir / 'reference_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        cfg_out = {'physics': config_dict(), 'grid': {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in g.items() if k not in ('mn', 'ms', 'mp', 'X', 'Y')}}
        (outdir / 'reference_config.json').write_text(json.dumps(cfg_out, indent=2), encoding='utf-8')
        if verbose:
            print('[FVM] summary:', json.dumps(summary, indent=2))
        return summary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nx', type=int, default=50)
    ap.add_argument('--ny', type=int, default=20)
    ap.add_argument('--nt', type=int, default=81)
    ap.add_argument('--nr', type=int, default=16)
    ap.add_argument('--outdir', default='outputs/reference')
    ap.add_argument('--quiet', action='store_true')
    a = ap.parse_args()
    ReferenceSolver(a.nx, a.ny, a.nt, a.nr).solve(a.outdir, save_fields=True, verbose=not a.quiet)
if __name__ == '__main__':
    main()