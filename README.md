# GLaDiT DiT on MegaFlow2D Ellipse

This branch implements the original high-resolution mesh UVP task: file frames 1,16,…,961, one observation and 64 predictions, Train782 / Validation100 / Test100. All training uses four GPUs and checkpoint selection uses full Validation100.

Start with [ELLIPSE.md](ELLIPSE.md) for data preparation, training, recovery, budgets and verification status. Use [TEST.md](TEST.md) for final evaluation. Architecture and optimizer settings are inherited from the Airfoil four-GPU source; [AIRFOIL.md](AIRFOIL.md) is historical reference, and its launch commands do not define this branch’s task.

```bash
export RAW_DATA_DIR=/data/datasets/megaflow2d
export DATA_DIR=/data/datasets/megaflow2d_ellipse_stride15
export CUDA_VISIBLE_DEVICES=0,1,2,3
export RESULT_ROOT=/data/runs/ellipse_gladit_seed0
bash scripts/ellipse_4gpu.sh train
```

## Automatic VGAE → DiT handoff

For an existing selected representation, set AUTOENCODER to its `dit_autoencoder.pt`. ARTIFACTS_DIR defaults to `${AUTOENCODER}.ellipse_latents` and is bound to that export and the Train statistics. For automatic representation training followed by DiT, leave AUTOENCODER unset and set:

```bash
export VGAE_REPO=/code/vgae-cylinderflow-stride8
export VGAE_RESULT_ROOT=/data/runs/ellipse_vgae_seed0
export RESULT_ROOT=/data/runs/ellipse_gladit_seed0
bash scripts/ellipse_4gpu.sh train
```

Both checkouts must use this Ellipse branch. The wrapper completes or resumes VGAE, uses its selected export, prepares all Train65 latent frames, verifies the representation, then starts DiT. ELLIPSE_CONFIG defaults to `configs/ellipse_h1_w512_d24_4gpu.json`. Final weights come from the completed run's `selection.json`; Test does not select another checkpoint.
