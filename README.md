# guide-assignment-metrics

Evaluation module for the [guide-assignment](https://github.com/tmp-perturb/guide-assignment)
Omnibenchmark. Three entrypoints:

| Entrypoint | Role |
|---|---|
| `run.py` (default) | per-lineage metrics: `tier1`, `construct_set`, `kd`, `discovery`, `mismatch`, `strat_tier1`, `strat_mismatch_loc`, `capacity`, `strat_delta_kd` (selected via `--metric`) |
| `difficulty_run.py` | difficulty table (`--phase table`) and delta-KD validation (`--phase validate`) |
| `collectors_run.py` | cross-lineage collectors: `jaccard`, `strat_jaccard`, `extraction_shift`, `mismatch` |

- Env: `assignment_metrics`
