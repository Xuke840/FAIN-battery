from __future__ import annotations
from pathlib import Path
import argparse, csv, json, math
import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.optimize import brentq
from physics import CFG
from reference import ReferenceSolver, fv_matrix, radial_operator
from training import train
from evaluation import compare

def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

def macro_mms(outdir):
    rows = []
    prev = None
    for nx in [20, 40, 80, 160]:
        ny = max(4, int(round(nx * CFG.Ly / CFG.Lx)))
        dx = CFG.Lx / nx
        dy = CFG.Ly / ny
        x = (np.arange(nx) + 0.5) * dx
        y = (np.arange(ny) + 0.5) * dy
        X, Y = np.meshgrid(x, y)
        u = np.cos(np.pi * X / CFG.Lx) * np.cos(np.pi * Y / CFG.Ly)
        exact = ((np.pi / CFG.Lx) ** 2 + (np.pi / CFG.Ly) ** 2) * u
        A = fv_matrix(np.ones_like(u), dx, dy)
        num = (A @ u.ravel()).reshape(u.shape)
        err = float(np.sqrt(np.mean((num - exact) ** 2)))
        order = np.nan if prev is None else math.log(prev / err) / math.log(2.0)
        rows.append({'nx': nx, 'ny': ny, 'L2_error': err, 'observed_order': order})
        prev = err
    write_csv(Path(outdir) / 'mms_macro_order.csv', rows)
    return rows

def particle_mms(outdir):
    rows = []
    prev = None
    for nr in [20, 40, 80, 160]:
        A, V = radial_operator(nr, 1.0)
        r = (np.arange(nr) + 0.5) / nr
        u = np.cos(np.pi * r)
        exact = np.pi ** 2 * np.cos(np.pi * r) + 2 * np.pi * np.sin(np.pi * r) / r
        num = A @ u
        err = float(np.sqrt(3.0 * np.sum(V * (num - exact) ** 2)))
        order = np.nan if prev is None else math.log(prev / err) / math.log(2.0)
        rows.append({'nr': nr, 'L2_error': err, 'observed_order': order})
        prev = err
    write_csv(Path(outdir) / 'mms_particle_order.csv', rows)
    return rows

def read_csv_table(path):
    arr = np.genfromtxt(path, delimiter=',', names=True, dtype=float, encoding=None)
    if arr.shape == ():
        arr = np.array([arr], dtype=arr.dtype)
    return arr

def apparent_order(Ecm, Emf, hc, hm, hf):
    if min(Ecm, Emf) <= 0:
        return np.nan
    ratio = Ecm / Emf

    def f(p):
        den = hm ** p - hf ** p
        return (hc ** p - hm ** p) / den - ratio
    grid = np.linspace(0.05, 8.0, 400)
    vals = [f(p) for p in grid]
    for a, b, fa, fb in zip(grid[:-1], grid[1:], vals[:-1], vals[1:]):
        if np.isfinite(fa) and np.isfinite(fb) and (fa * fb <= 0):
            return float(brentq(f, a, b))
    return np.nan

def trajectory_self_convergence(run_dirs, grids, outdir):
    tabs = [read_csv_table(Path(d) / 'global_cell.csv') for d in run_dirs]
    ptabs = [read_csv_table(Path(d) / 'positive_electrode.csv') for d in run_dirs]
    qs = [('V_cell', [t['V_cell'] for t in tabs]), ('ce_min_positive', [t['ce_min_reconstructed'] for t in ptabs]), ('reaction_centroid_x_positive', [t['reaction_centroid_x'] for t in ptabs])]
    hs = [CFG.Lx / g[0] for g in grids]
    rows = []
    for name, series in qs:
        c, m, f = [np.asarray(v, float) for v in series]
        valid = np.isfinite(c) & np.isfinite(m) & np.isfinite(f)
        Ecm = float(np.max(np.abs(c[valid] - m[valid])))
        Emf = float(np.max(np.abs(m[valid] - f[valid])))
        p = apparent_order(Ecm, Emf, *hs)
        scale = max(float(np.max(np.abs(f[valid]))), 1e-12)
        rows.append({'quantity': name, 'coarse_medium_Linf': Ecm, 'medium_fine_Linf': Emf, 'medium_fine_normalized': Emf / scale, 'observed_order': p})
    write_csv(Path(outdir) / 'self_convergence.csv', rows)
    return rows

def final_field_independence(run_dirs, outdir):
    med = np.load(Path(run_dirs[1]) / 'reference_fields.npz')
    fine = np.load(Path(run_dirs[2]) / 'reference_fields.npz')
    xm, ym = (med['x'], med['y'])
    xf, yf = (fine['x'], fine['y'])
    Xm, Ym = np.meshgrid(xm, ym)
    pts = np.c_[Ym.ravel(), Xm.ravel()]
    rows = []
    for field in ['ce', 'T', 'phi_e', 'q_gen', 'theta_n_surf', 'theta_p_surf']:
        am = np.asarray(med[field][-1], float)
        af = np.asarray(fine[field][-1], float)
        interp = RegularGridInterpolator((yf, xf), af, bounds_error=False, fill_value=np.nan)
        afi = interp(pts).reshape(am.shape)
        valid = np.isfinite(am) & np.isfinite(afi)
        if field == 'theta_n_surf':
            valid &= Xm < CFG.x_n_end - 1.1 * (xm[1] - xm[0])
        if field == 'theta_p_surf':
            valid &= Xm > CFG.x_s_end + 1.1 * (xm[1] - xm[0])
        d = am[valid] - afi[valid]
        rmse = float(np.sqrt(np.mean(d * d)))
        rel = float(np.linalg.norm(d) / max(np.linalg.norm(afi[valid]), 1e-14))
        linf = float(np.max(np.abs(d)))
        rows.append({'field': field, 'RMSE_medium_vs_fine': rmse, 'relative_L2_medium_vs_fine': rel, 'Linf_medium_vs_fine': linf})
    write_csv(Path(outdir) / 'grid_independence_fields.csv', rows)
    return rows

def run_suite(mode='quick', outdir='outputs/reference_verification'):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    mm = macro_mms(outdir)
    pm = particle_mms(outdir)
    if mode == 'quick':
        grids = [(20, 8), (30, 12), (40, 16)]
        nt = 31
        nr = 10
    elif mode == 'formal':
        grids = [(50, 20), (100, 40), (150, 60)]
        nt = 81
        nr = 96
    else:
        raise ValueError(mode)
    run_dirs = []
    summaries = []
    for nx, ny in grids:
        d = outdir / f'grid_{nx}x{ny}'
        d.mkdir(exist_ok=True)
        print(f'\n[VERIFY] reference grid {nx} x {ny}, nt={nt}, nr={nr}')
        summary = ReferenceSolver(nx, ny, nt, nr).solve(d, save_fields=True, verbose=False)
        summaries.append(summary)
        run_dirs.append(d)
        print(json.dumps({k: summary[k] for k in ['runtime_seconds', 'max_Li_balance_rel', 'max_current_balance_rel', 'max_thermal_balance_rel', 'total_bv_guard_activations']}, indent=2))
    conv = trajectory_self_convergence(run_dirs, grids, outdir)
    fieldrows = final_field_independence(run_dirs, outdir)
    finest = summaries[-1]
    mms_macro_orders = [r['observed_order'] for r in mm if np.isfinite(r['observed_order'])]
    mms_particle_orders = [r['observed_order'] for r in pm if np.isfinite(r['observed_order'])]
    conv_orders = [r['observed_order'] for r in conv if np.isfinite(r['observed_order'])]
    max_grid_rel = max((r['relative_L2_medium_vs_fine'] for r in fieldrows))
    gates = {'macro_MMS_second_order': min(mms_macro_orders) > 1.9, 'particle_MMS_second_order': min(mms_particle_orders) > 1.9, 'Li_conservation': finest['max_Li_balance_rel'] < 1e-10, 'current_balance': finest['max_current_balance_rel'] < 1e-08, 'thermal_balance': finest['max_thermal_balance_rel'] < 1e-08, 'charge_residual': finest['max_charge_residual_rel'] < 1e-08, 'BV_guard_inactive': finest['total_bv_guard_activations'] == 0, 'field_grid_independence': max_grid_rel < (0.1 if mode == 'quick' else 0.02)}
    if mode == 'formal':
        gates['scientific_self_convergence_order'] = len(conv_orders) == 3 and min(conv_orders) > 1.5
    summary = {'mode': mode, 'grids': grids, 'nt': nt, 'nr': nr, 'gates': gates, 'all_required_gates_pass': bool(all(gates.values())), 'max_medium_fine_field_relative_L2': float(max_grid_rel), 'scientific_observed_orders': {r['quantity']: r['observed_order'] for r in conv}}
    (outdir / 'verification_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('\nREFERENCE VERIFICATION SUMMARY')
    for k, v in gates.items():
        print(f"  {k:38s}: {('PASS' if v else 'FAIL')}")
    print(json.dumps(summary, indent=2))
    return summary

def _read(path):
    a = np.genfromtxt(path, delimiter=',', names=True, dtype=float, encoding=None)
    if a.shape == ():
        a = np.array([a], dtype=a.dtype)
    return a

def _write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

def _series_error(tc, yc, tm, ym):
    fi = interp1d(tm, ym, kind='linear', bounds_error=True)
    return float(np.max(np.abs(yc - fi(tc))))

def temporal_convergence(outdir, quick=False):
    outdir = Path(outdir) / 'temporal'
    outdir.mkdir(parents=True, exist_ok=True)
    if quick:
        nx, ny, nr = (30, 12, 10)
        nts = [11, 21, 41]
    else:
        nx, ny, nr = (50, 20, 96)
        nts = [41, 81, 161]
    dirs = []
    for nt in nts:
        d = outdir / f'nt_{nt}'
        d.mkdir(exist_ok=True)
        print(f'[TIME] nx={nx} ny={ny} nr={nr} nt={nt}')
        ReferenceSolver(nx, ny, nt, nr).solve(d, save_fields=False, verbose=False)
        dirs.append(d)
    G = [_read(d / 'global_cell.csv') for d in dirs]
    P = [_read(d / 'positive_electrode.csv') for d in dirs]
    quantities = [('V_cell', [g['V_cell'] for g in G], G), ('T_max', [g['T_max'] for g in G], G), ('SOC_p', [g['SOC_p'] for g in G], G), ('ce_min_positive', [p['ce_min_reconstructed'] for p in P], P)]
    rows = []
    for name, ys, tables in quantities:
        Ecm = _series_error(tables[0]['time'], ys[0], tables[1]['time'], ys[1])
        Emf = _series_error(tables[1]['time'], ys[1], tables[2]['time'], ys[2])
        p = np.nan if min(Ecm, Emf) <= 1e-16 else math.log(Ecm / Emf, 2.0)
        scale = max(float(np.max(np.abs(ys[2]))), 1e-14)
        rows.append({'quantity': name, 'coarse_medium_Linf': Ecm, 'medium_fine_Linf': Emf, 'medium_fine_normalized': Emf / scale, 'observed_temporal_order': p})
    _write(outdir / 'temporal_convergence.csv', rows)
    return rows

def radial_convergence(outdir, quick=False):
    outdir = Path(outdir) / 'radial'
    outdir.mkdir(parents=True, exist_ok=True)
    if quick:
        nx, ny, nt = (30, 12, 31)
        nrs = [6, 10, 18]
    else:
        nx, ny, nt = (50, 20, 81)
        nrs = [24, 48, 96]
    dirs = []
    for nr in nrs:
        d = outdir / f'nr_{nr}'
        d.mkdir(exist_ok=True)
        print(f'[RADIAL] nx={nx} ny={ny} nt={nt} nr={nr}')
        ReferenceSolver(nx, ny, nt, nr).solve(d, save_fields=True, verbose=False)
        dirs.append(d)
    data = [np.load(d / 'reference_fields.npz') for d in dirs]
    rows = []
    for fld in ['theta_n_surf', 'theta_p_surf', 'theta_n_avg', 'theta_p_avg']:
        c, m, f = [a[fld][-1] for a in data]
        valid = np.isfinite(c) & np.isfinite(m) & np.isfinite(f)
        Ecm = float(np.max(np.abs(c[valid] - m[valid])))
        Emf = float(np.max(np.abs(m[valid] - f[valid])))
        p = np.nan if min(Ecm, Emf) <= 1e-16 else math.log(Ecm / Emf, 2.0)
        scale = max(float(np.max(np.abs(f[valid]))), 1e-14)
        rows.append({'quantity': fld, 'coarse_medium_Linf': Ecm, 'medium_fine_Linf': Emf, 'medium_fine_normalized': Emf / scale, 'observed_radial_order': p})
    _write(outdir / 'radial_convergence.csv', rows)
    return rows

def run_extended(outdir='outputs/extended_reference_verification', mode='quick'):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    quick = mode == 'quick'
    tr = temporal_convergence(outdir, quick)
    rr = radial_convergence(outdir, quick)
    t_orders = [r['observed_temporal_order'] for r in tr if np.isfinite(r['observed_temporal_order'])]
    r_orders = [r['observed_radial_order'] for r in rr if np.isfinite(r['observed_radial_order'])]
    max_t = max((r['medium_fine_normalized'] for r in tr))
    max_r = max((r['medium_fine_normalized'] for r in rr))
    gates = {'temporal_independence': max_t < (0.05 if quick else 0.002), 'radial_independence': max_r < (0.05 if quick else 0.002), 'temporal_order_defined': len(t_orders) >= 3, 'radial_order_defined': len(r_orders) >= 3}
    if not quick:
        gates['temporal_order_consistent_with_BE'] = np.nanmedian(t_orders) > 0.7
        gates['radial_refinement_consistent'] = np.nanmedian(r_orders) > 0.8
    result = {'mode': mode, 'gates': {k: bool(v) for k, v in gates.items()}, 'all_required_gates_pass': bool(all(gates.values())), 'max_temporal_medium_fine_normalized': float(max_t), 'max_radial_medium_fine_normalized': float(max_r), 'temporal_orders': {r['quantity']: r['observed_temporal_order'] for r in tr}, 'radial_orders': {r['quantity']: r['observed_radial_order'] for r in rr}}
    (outdir / 'extended_verification_summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    return result

def run_verification(mode='formal', outroot='outputs'):
    root = Path(outroot)
    root.mkdir(parents=True, exist_ok=True)
    primary = run_suite(mode, root / 'reference_verification')
    extended = run_extended(root / 'extended_reference_verification', mode)
    combined = {'mode': mode, 'primary': primary, 'extended': extended, 'all_required_gates_pass': bool(primary.get('all_required_gates_pass') and extended.get('all_required_gates_pass')), 'methodology': {'macro_spatial_order': 'three-grid self-convergence at fixed radially-converged Nr', 'operator_order': 'independent manufactured solutions for macro and spherical FV operators', 'temporal_order': 'separate three-dt study; backward-Euler expectation approximately first order', 'radial_order': 'separate three-Nr study using reconstructed physical particle-surface concentration', 'reference_leakage_into_FAIN_training': False}}
    (root / 'verification_master_summary.json').write_text(json.dumps(combined, indent=2), encoding='utf-8')
    print('\nMASTER VERIFICATION:', 'PASS' if combined['all_required_gates_pass'] else 'FAIL')
    return combined
CORE_FIELDS = ('ce', 'T', 'phi_e', 'theta_n_surf', 'theta_p_surf', 'jn', 'jp', 'q_gen')

def read_metrics(path):
    with open(path, encoding='utf-8') as f:
        return list(csv.DictReader(f))

def summarize_case(label, kind, value, run_dir):
    rows = read_metrics(Path(run_dir) / 'comparison' / 'field_metrics.csv')
    vals = [float(r['relative_L2']) for r in rows if r['field'] in CORE_FIELDS and r['region'] == 'full']
    return {'label': label, 'kind': kind, 'value': value, 'mean_core_relative_L2': float(np.mean(vals)), 'max_core_relative_L2': float(np.max(vals)), 'metric_count': len(vals)}

def prediction_difference(run_a, run_b):
    a = np.load(Path(run_a) / 'comparison' / 'fain_fields_on_reference_grid.npz')
    b = np.load(Path(run_b) / 'comparison' / 'fain_fields_on_reference_grid.npz')
    rows = []
    for fld in CORE_FIELDS:
        if fld not in a.files or fld not in b.files:
            continue
        av = np.asarray(a[fld], float)
        bv = np.asarray(b[fld], float)
        good = np.isfinite(av) & np.isfinite(bv)
        if not np.any(good):
            continue
        d = av[good] - bv[good]
        rel = float(np.linalg.norm(d) / max(np.linalg.norm(bv[good]), 1e-14))
        rows.append({'field': fld, 'relative_L2_A_vs_B': rel, 'RMSE_A_vs_B': float(np.sqrt(np.mean(d * d))), 'Linf_A_vs_B': float(np.max(np.abs(d)))})
    return rows

def write_rows(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

def run_independence(reference, mode='quick', outdir='outputs/fain_independence', device=None, include_seeds=False):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if mode == 'quick':
        steps = 1
        network = [12, 16]
        collocation = [2, 3]
        basew = 12
        pwidth = 10
        depth = 1
        pdepth = 1
    else:
        steps = 8000
        network = [64, 96, 128]
        collocation = [128, 256, 512]
        basew = 96
        pwidth = 64
        depth = 4
        pdepth = 3
    summaries = []
    network_dirs = []
    collocation_dirs = []
    seed_dirs = []
    for w in network:
        d = outdir / f'network_width_{w}'
        model = train(steps, w, depth, pwidth, pdepth, collocation[len(collocation) // 2], 20260913, device, 'float32', d / 'train')
        compare(reference, model, d / 'comparison', nr_eval=12 if mode == 'quick' else 96)
        summaries.append(summarize_case(f'width_{w}', 'network_width', w, d))
        network_dirs.append(d)
    for ncol in collocation:
        d = outdir / f'collocation_{ncol}'
        model = train(steps, basew, depth, pwidth, pdepth, ncol, 20260913, device, 'float32', d / 'train')
        compare(reference, model, d / 'comparison', nr_eval=12 if mode == 'quick' else 96)
        summaries.append(summarize_case(f'ncol_{ncol}', 'collocation_points_per_region', ncol, d))
        collocation_dirs.append(d)
    if include_seeds:
        for seed in [20260913, 20260914, 20260915]:
            d = outdir / f'seed_{seed}'
            model = train(steps, basew, depth, pwidth, pdepth, collocation[len(collocation) // 2], seed, device, 'float32', d / 'train')
            compare(reference, model, d / 'comparison', nr_eval=12 if mode == 'quick' else 96)
            summaries.append(summarize_case(f'seed_{seed}', 'seed', seed, d))
            seed_dirs.append(d)
    write_rows(outdir / 'independence_summary.csv', summaries)
    groups = {}
    for kind in sorted(set((r['kind'] for r in summaries))):
        a = np.array([r['mean_core_relative_L2'] for r in summaries if r['kind'] == kind])
        groups[kind] = {'min': float(a.min()), 'max': float(a.max()), 'relative_spread': float((a.max() - a.min()) / max(a.mean(), 1e-14))}
    direct = {}
    if len(network_dirs) >= 2:
        rows = prediction_difference(network_dirs[-2], network_dirs[-1])
        write_rows(outdir / 'network_medium_high_prediction_difference.csv', rows)
        direct['network_medium_high'] = {'mean_relative_L2': float(np.mean([r['relative_L2_A_vs_B'] for r in rows])), 'max_relative_L2': float(np.max([r['relative_L2_A_vs_B'] for r in rows]))}
    if len(collocation_dirs) >= 2:
        rows = prediction_difference(collocation_dirs[-2], collocation_dirs[-1])
        write_rows(outdir / 'collocation_medium_high_prediction_difference.csv', rows)
        direct['collocation_medium_high'] = {'mean_relative_L2': float(np.mean([r['relative_L2_A_vs_B'] for r in rows])), 'max_relative_L2': float(np.max([r['relative_L2_A_vs_B'] for r in rows]))}
    if len(seed_dirs) >= 2:
        pair = []
        for i in range(len(seed_dirs)):
            for j in range(i + 1, len(seed_dirs)):
                rows = prediction_difference(seed_dirs[i], seed_dirs[j])
                pair.append(float(np.max([r['relative_L2_A_vs_B'] for r in rows])))
        direct['seed_pairwise'] = {'mean_max_field_relative_L2': float(np.mean(pair)), 'worst_max_field_relative_L2': float(np.max(pair))}
    gates = {}
    if mode == 'formal':
        if 'network_medium_high' in direct:
            gates['network_capacity_independence'] = direct['network_medium_high']['max_relative_L2'] < 0.05
        if 'collocation_medium_high' in direct:
            gates['collocation_independence'] = direct['collocation_medium_high']['max_relative_L2'] < 0.05
        if 'seed_pairwise' in direct:
            gates['seed_robustness'] = direct['seed_pairwise']['worst_max_field_relative_L2'] < 0.1
    result = {'mode': mode, 'steps_each': steps, 'zero_reference_supervision': True, 'groups': groups, 'direct_prediction_independence': direct, 'gates': gates, 'all_defined_gates_pass': bool(all(gates.values())) if gates else None, 'runs': len(summaries)}
    (outdir / 'independence_summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    return result