# Checkpoint curves and training cost

The exporter reads retained JSON records from CylinderFlow attention, Airfoil,
and scaling runs. It produces intermediate Validation24 scores against measured
trainer elapsed time, plus the cost of a specified denoiser checkpoint and its
shared autoencoder. Existing monitor evaluations supply the scores. The supplied
September 25 result tables contain intermediate points and the Airfoil 150k
cost; this entrypoint makes future exports reproducible from the run evidence.

## Inputs and attention curves

Use a stable copy of each run containing its configuration, status, evaluation
records, and every attempt's launch record and training log. Preserve the original
run directory and all attempts. Execute this standalone script from its checkout;
input directories can come from any of the three experiment branches. Python
3.10+ is sufficient for JSON/CSV export; optional plotting uses the existing
Matplotlib dependency.

```bash
python scripts/export_checkpoint_cost.py curves \
  --dataset Airfoil \
  --run H1 /shared/airfoil/h1_seed0 \
  --run H2 /shared/airfoil/h2_seed0 \
  --run Full /shared/airfoil/full_seed0 \
  --output-dir /shared/reports/airfoil_attention_time --plot
```

Run CylinderFlow separately with `--dataset CylinderFlow` and its own directories.
Assign each run a unique label. Different hardware types, GPU counts, and training
seeds receive separate panels. The exporter retains every completed intermediate
monitor from unfinished Full runs. Each curve ends at its last available point.
Scores are the best raw/EMA weight set at each update, with the trainer's configured
weight order breaking ties. All expected weights and complete finite Validation24
scores must be present. The original per-weight scores remain in the CSV.

The plot compares configurations after removing only the training seed, protocol
label, attention metadata, attention mode, and hop limit. Other settings, including
model dimensions, schedule, budget, precision, sampling, and representation, must
match. A cache, controlled-configuration, or recorded-runtime mismatch withholds
the panel and leaves the input points and issue in the report. Review the saved
launch evidence for software and device-topology equivalence before making a
controlled speed claim.
Training progress, incomplete evaluations, and failed-run status remain visible.

The output contains `checkpoints.csv`, `report.json`, and optional PDF/PNG panels.
The report retains full configurations, launch/source identities, source locations,
and unresolved issues. Use a new output directory for each export.

## Time definitions

Two independent counters remain separate:

- **Elapsed hours** use the exact update's training log. They include earlier
  in-training monitoring and checkpoint I/O. The current update's save and monitor
  occur after this log entry. Offline evaluation, preparation, and downtime between
  attempts have separate costs.
- **Training-update hours** include the timed training-update loop. The existing
  scaling report uses this counter for its training GPU-hours axis.

GPU-hours multiply the respective counter by the recorded GPU count. These are
different accounting scopes. A missing exact-update log, ambiguous duplicate logs,
or a mismatched checkpoint/cache/runtime identity leaves the timing blank.
The exporter preserves recorded times without interpolation or extrapolation.
After recovery, cumulative counters describe the trainer's restored lineage;
discarded work beyond a recovery point contributes separately to campaign resource
accounting. All original attempt records remain in the report.

## Cost at the evaluated checkpoint

Prepare an autoencoder cost record from its retained timing evidence. It contains
`measurement: "measured"`, `artifact_id` matching the DiT candidate, the selected
autoencoder `checkpoint`, exact `hardware` name, `gpu_count`, `elapsed_hours`,
`time_scope`, and `evidence` pointing to the source record. This explicit binding
associates the shared representation with the denoiser run. Use the measured
autoencoder training scope adopted by the paper and state it in `time_scope`.

```bash
python scripts/export_checkpoint_cost.py cost \
  --run /shared/airfoil/h1_seed0 --update 150000 --weights ema_0.999 \
  --autoencoder-cost /shared/evidence/airfoil_autoencoder_cost.json \
  --output-dir /shared/reports/airfoil_150k_cost
```

`checkpoint_cost.json` records the requested update and weight set, both stages,
their timing sources, the summed sequential stage hours, and total GPU-hours when
both stages use the same GPU model. Different hardware keeps stage GPU-hours
separate. The shared autoencoder is counted once per model's end-to-end training
cost and once for a multi-seed campaign. Full-budget projections belong in a
separate estimate with their assumptions.

## Verification

This delivery received syntax parsing, static lint, and Git whitespace checks.
Execution against retained production records and target-environment plotting
remain pending. Existing training, monitoring, selection, and evaluation entrypoints
continue to supply the source evidence. Train/Validation permissions remain in
force and Test remains sealed.
