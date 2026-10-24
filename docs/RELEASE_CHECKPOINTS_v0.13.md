# FAIN v0.13 formal checkpoints release

Recommended tag: `v0.13-paper`

Recommended release title:

`FAIN v0.13 — Formal six-model checkpoints`

Recommended release description:

Formal trained checkpoints and archived training configurations corresponding to the six-model comparison reported in the FAIN battery manuscript.

Included models:

- FAIN — 354,640 parameters — EMA checkpoint
- Pure PINN — 120,688 parameters — EMA checkpoint
- Res-PINN — 168,016 parameters — final checkpoint
- Parameter-matched MLP — 353,621 parameters — EMA checkpoint
- SIREN-PINN — 83,440 parameters — EMA checkpoint
- XPINN — 139,981 parameters — final checkpoint

All models were trained with zero finite-volume reference-field supervision. The associated source provenance is documented in `SOURCE_PROVENANCE.json`.

Release asset:

`FAIN_GITHUB_RELEASE_CHECKPOINTS_20261024.zip`

SHA-256:

`66c9d6ef9df48b1e1afaf958f59a673dc5af57ceedfc2ccd83a16109f22e7df5`
