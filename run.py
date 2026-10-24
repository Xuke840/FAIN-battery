from pathlib import Path
import argparse
from reference import ReferenceSolver
from training import train
from evaluation import compare
from verification import run_verification, run_independence
MODELS = ('fain', 'pure_pinn', 'res_pinn', 'param_mlp', 'siren_pinn', 'xpinn')

def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest='command', required=True)
    q = sp.add_parser('reference')
    q.add_argument('--nx', type=int, default=150)
    q.add_argument('--ny', type=int, default=60)
    q.add_argument('--nt', type=int, default=81)
    q.add_argument('--nr', type=int, default=96)
    q.add_argument('--outdir', default='outputs/reference_formal')
    q.add_argument('--quiet', action='store_true')
    q = sp.add_parser('train')
    q.add_argument('--model', choices=MODELS, required=True)
    q.add_argument('--steps', type=int, default=12000)
    q.add_argument('--ncol', type=int, default=256)
    q.add_argument('--seed', type=int, default=20260913)
    q.add_argument('--device', default=None)
    q.add_argument('--dtype', choices=['float32', 'float64'], default='float32')
    q.add_argument('--outdir', default=None)
    q = sp.add_parser('evaluate')
    q.add_argument('--reference', required=True)
    q.add_argument('--checkpoint', required=True)
    q.add_argument('--outdir', required=True)
    q.add_argument('--nr-eval', type=int, default=24)
    q = sp.add_parser('verify')
    q.add_argument('--mode', choices=['quick', 'formal'], default='formal')
    q.add_argument('--outroot', default='outputs')
    q = sp.add_parser('independence')
    q.add_argument('--reference', required=True)
    q.add_argument('--mode', choices=['quick', 'formal'], default='quick')
    q.add_argument('--outdir', default='outputs/fain_independence')
    q.add_argument('--device', default=None)
    q.add_argument('--include-seeds', action='store_true')
    q = sp.add_parser('all')
    q.add_argument('--device', default=None)
    q.add_argument('--dtype', choices=['float32', 'float64'], default='float32')
    q.add_argument('--outroot', default='outputs')
    a = p.parse_args()
    if a.command == 'reference':
        ReferenceSolver(a.nx, a.ny, a.nt, a.nr).solve(a.outdir, save_fields=True, verbose=not a.quiet)
    elif a.command == 'train':
        train(a.model, 12000 if a.steps is None else a.steps, None, 4, 64, 3, a.ncol, a.seed, a.device, a.dtype, a.outdir)
    elif a.command == 'evaluate':
        compare(a.reference, a.checkpoint, a.outdir, a.nr_eval)
    elif a.command == 'verify':
        run_verification(a.mode, a.outroot)
    elif a.command == 'independence':
        run_independence(a.reference, a.mode, a.outdir, a.device, a.include_seeds)
    else:
        out = Path(a.outroot)
        refdir = out / 'reference_formal'
        ref = refdir / 'reference_fields.npz'
        ReferenceSolver(150, 60, 81, 96).solve(refdir, save_fields=True, verbose=True)
        for m in MODELS:
            md = out / 'formal' / m
            ck = train(m, 12000, None, 4, 64, 3, 256, 20260913, a.device, a.dtype, md)
            compare(ref, ck, md / 'compare', 24)
if __name__ == '__main__':
    main()
