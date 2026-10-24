# FAIN v0.13 paper code

This repository contains the integrated code used for the FAIN battery manuscript. The numerical problem is a scaled transient two-dimensional electrochemical-thermal DFN-type porous-electrode model with spherical particle diffusion. Training uses zero finite-volume reference-field supervision.

## Repository structure

- `physics.py`: physical configuration and constitutive closures.
- `models.py`: FAIN and all five comparison architectures.
- `training.py`: common physics-informed training and reference-free checkpoint selection.
- `reference.py`: conservative cell-centred finite-volume reference solver with spherical finite-volume particles.
- `evaluation.py`: common reference-grid evaluation and trajectory/field post-processing.
- `verification.py`: MMS, grid/self-convergence, temporal/radial checks, conservation gates and independence studies.
- `run.py`: unified command-line entry point.
- `configs/formal.json`: settings used in the formal paper runs.
- `SOURCE_PROVENANCE.json`: source provenance and reconstruction status.

The Python source is intentionally distributed without inline comments or docstrings. Scientific scope, provenance and usage are documented here and in `docs/`.

## Provenance

The FAIN, finite-volume, verification, SIREN-PINN and XPINN implementations derive from the historically retained v0.13 source. The original Python source files for Pure PINN, Res-PINN and the parameter-matched MLP were absent from the archived September 2026 output package. Their network definitions were reconstructed from the archived checkpoint state dictionaries, architecture metadata and the retained v0.13 physical representation. The reconstructed definitions match the archived parameter counts exactly and strictly load the formal checkpoints with no missing or unexpected keys. This distinction is recorded in `SOURCE_PROVENANCE.json`.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Linux/macOS, activate with `source .venv/bin/activate`.

## Formal reference

```bash
python run.py reference --nx 150 --ny 60 --nt 81 --nr 96 --outdir outputs/reference_formal
```

## Train one model

```bash
python run.py train --model fain --steps 12000 --ncol 256 --seed 20260913
python run.py train --model pure_pinn --steps 12000 --ncol 256 --seed 20260913
python run.py train --model res_pinn --steps 12000 --ncol 256 --seed 20260913
python run.py train --model param_mlp --steps 12000 --ncol 256 --seed 20260913
python run.py train --model siren_pinn --steps 12000 --ncol 256 --seed 20260913
python run.py train --model xpinn --steps 12000 --ncol 256 --seed 20260913
```

## Evaluate a checkpoint

```bash
python run.py evaluate --reference outputs/reference_formal/reference_fields.npz --checkpoint outputs/formal/fain/model.pt --outdir outputs/formal/fain/compare
```

Field evaluation records relative L2, RMSE, MSE, MAE and maximum absolute error. The current manuscript uses relative L2, RMSE, MSE and MAE in its principal comparison.

## Verification

```bash
python run.py verify --mode formal --outroot outputs
```

The verification module contains the macroscopic and spherical MMS checks, the spatial grid/self-convergence analysis, temporal and radial refinement checks, and conservation/charge/Butler-Volmer gates.

## Full six-model run

```bash
python run.py all --outroot outputs
```

This command is computationally expensive.

## Formal settings

The paper reference uses `150 x 60` macroscopic cells, `81` time levels and `96` radial shells. The formal neural runs use 12,000 optimization steps, 256 collocation points per material region and seed `20260913`.

## Checkpoints and large outputs

Formal checkpoints and large numerical output archives are distributed separately from the source tree.

## Citation

Citation metadata are provided in `CITATION.cff`. Update the preferred citation with the final journal DOI after publication.

## License

No software license has been selected yet. A license should be added before the repository is described as formally open-source software.
