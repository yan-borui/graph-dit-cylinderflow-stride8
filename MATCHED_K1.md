# Matched 250K K1 evaluation

This entrypoint evaluates the original seed-0 ablation runs at update 250000 with EMA 0.999.
It never reads or changes selection.json. Use the exact runs from the previous K8 table,
not the submitted 550K CylinderFlow model or Validation-selected alternatives.

GLaDiT and temporal GNN use S6 and three independent K1 predictions, with sampling labels
0, 1, 2. Score each decoded prediction first, average scores within each trajectory,
then give each of the 100 trajectories equal weight. This is **not physical-field averaging**.
The deterministic Transformer is evaluated once; its repeated predictions would be identical.
The shared PRNG mapping is ((label+1)*1000003 + trajectory_index*9176) % (2**31-1).

## Branches and input identities

| Model | Dataset | Branch |
| --- | --- | --- |
| GLaDiT | CylinderFlow | main |
| GLaDiT | Airfoil | feature/airfoil-uvp-4gpu |
| Deterministic Transformer | Both | feature/deterministic-graph-transformer-4gpu |
| Temporal GNN diffusion | Both | feature/graph-latent-diffusion-4gpu |

Run from the corresponding checkout in its existing production CUDA environment.
Each evaluation uses one visible GPU and FP32. Keep the original frozen autoencoder,
latent/physical normalization and official 100-trajectory split; only future frames 1–64
are scored. CylinderFlow retains known inlet/wall velocity writeback; Airfoil predicts all nodes.

For each model/dataset, set RUN, CHECKPOINT and RUN_ID to that model's original run directory,
its 250K checkpoint (relative to RUN), and its recorded run UUID. The checkpoint loader checks
the UUID, update, training seed and EMA availability. Do not edit the original selection record.
For GLaDiT, ARTIFACTS is its original representation export; DATA is the prepared Train/Validation
directory and TEST_DATA is the official Test65 directory. For the controls, CACHE is the
original matching Train65 cache, DATA is the prepared Train/Validation directory and TEST_DATA
is the prepared Test65 directory. Use the existing official preparation tools if these inputs
are missing. OUTPUT must be a new directory for this model/dataset.

## All 12 evaluations

After setting the variables above for the named model/dataset, execute its two commands.
These commands intentionally keep each split in a separate output directory.

### GLaDiT: CylinderFlow (main)

```bash
python -m graph_dit.matched_k1 evaluate --model gladit --dataset cylinderflow --split validation --run "$RUN" --checkpoint "$CHECKPOINT" --run-id "$RUN_ID" --artifacts "$ARTIFACTS" --data-dir "$DATA" --output-dir "$OUTPUT/validation"
python -m graph_dit.matched_k1 evaluate --model gladit --dataset cylinderflow --split test --run "$RUN" --checkpoint "$CHECKPOINT" --run-id "$RUN_ID" --artifacts "$ARTIFACTS" --data-dir "$DATA" --test-data-dir "$TEST_DATA" --output-dir "$OUTPUT/test"
```

### GLaDiT: Airfoil (feature/airfoil-uvp-4gpu)

```bash
python -m graph_dit.matched_k1 evaluate --model gladit --dataset airfoil --split validation --run "$RUN" --checkpoint "$CHECKPOINT" --run-id "$RUN_ID" --artifacts "$ARTIFACTS" --data-dir "$DATA" --output-dir "$OUTPUT/validation"
python -m graph_dit.matched_k1 evaluate --model gladit --dataset airfoil --split test --run "$RUN" --checkpoint "$CHECKPOINT" --run-id "$RUN_ID" --artifacts "$ARTIFACTS" --data-dir "$DATA" --test-data-dir "$TEST_DATA" --output-dir "$OUTPUT/test"
```

### Deterministic Transformer: CylinderFlow

```bash
python -m graph_dit.matched_k1 evaluate --model regression --dataset cylinderflow --split validation --run "$RUN" --checkpoint "$CHECKPOINT" --run-id "$RUN_ID" --cache-dir "$CACHE" --data-dir "$DATA" --output-dir "$OUTPUT/validation"
python -m graph_dit.matched_k1 evaluate --model regression --dataset cylinderflow --split test --run "$RUN" --checkpoint "$CHECKPOINT" --run-id "$RUN_ID" --cache-dir "$CACHE" --data-dir "$TEST_DATA" --output-dir "$OUTPUT/test"
```

### Deterministic Transformer: Airfoil

```bash
python -m graph_dit.matched_k1 evaluate --model regression --dataset airfoil --split validation --run "$RUN" --checkpoint "$CHECKPOINT" --run-id "$RUN_ID" --cache-dir "$CACHE" --data-dir "$DATA" --output-dir "$OUTPUT/validation"
python -m graph_dit.matched_k1 evaluate --model regression --dataset airfoil --split test --run "$RUN" --checkpoint "$CHECKPOINT" --run-id "$RUN_ID" --cache-dir "$CACHE" --data-dir "$TEST_DATA" --output-dir "$OUTPUT/test"
```

### Temporal GNN diffusion: CylinderFlow

```bash
python -m graph_dit.matched_k1 evaluate --model gnn --dataset cylinderflow --split validation --run "$RUN" --checkpoint "$CHECKPOINT" --run-id "$RUN_ID" --cache-dir "$CACHE" --data-dir "$DATA" --output-dir "$OUTPUT/validation"
python -m graph_dit.matched_k1 evaluate --model gnn --dataset cylinderflow --split test --run "$RUN" --checkpoint "$CHECKPOINT" --run-id "$RUN_ID" --cache-dir "$CACHE" --data-dir "$TEST_DATA" --output-dir "$OUTPUT/test"
```

### Temporal GNN diffusion: Airfoil

```bash
python -m graph_dit.matched_k1 evaluate --model gnn --dataset airfoil --split validation --run "$RUN" --checkpoint "$CHECKPOINT" --run-id "$RUN_ID" --cache-dir "$CACHE" --data-dir "$DATA" --output-dir "$OUTPUT/validation"
python -m graph_dit.matched_k1 evaluate --model gnn --dataset airfoil --split test --run "$RUN" --checkpoint "$CHECKPOINT" --run-id "$RUN_ID" --cache-dir "$CACHE" --data-dir "$TEST_DATA" --output-dir "$OUTPUT/test"
```

## Outputs and comparison

Each run writes identity.json, case_INDEX_sampleLABEL.json, trajectory_INDEX.json,
failures.json and summary.json. Individual records include the actual PRNG seed and metrics;
trajectory records contain the mean of the independently scored repetitions, including
per-frame metrics. There are 300 diffusion sample records or 100 deterministic records.
Any failed sample marks its trajectory and the full evaluation incomplete, preserves the
failure and yields a nonzero exit. Incomplete reports have no population metric estimate.

Collect the twelve summary.json files (expand these globs only over this new result root):

```bash
python -m graph_dit.matched_k1 collect --summaries "$RESULTS"/*/*/*/summary.json --output-dir "$RESULTS/comparison"
```

The expected layout for that glob is RESULTS/model/dataset/split/summary.json. Explicitly
listing all twelve files is also supported. Collection checks the protocol, repetition
count, labels, steps, weights, update, frame count, completeness, matching trajectory
indices and representation identities. Validation and Test must share each model's run UUID.
It writes comparison.csv and comparison.md with UV relative RMSE in percent.
Do not mix these results with the old K8 summaries or overwrite them.

This delivery is statically checked. GPU evaluation must run in the verified production
environment; no new training is required.
