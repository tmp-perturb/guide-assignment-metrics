#!/usr/bin/env python3
"""Aggregate collector: flatten every per-lineage metric JSON into one tidy table.

Fans in the score JSONs from all metric stages (metrics_gt / metrics_gex /
metrics_discovery) — plus any collector JSONs passed — and writes a single
long-format table (one row per scalar value), matching the omni-perturb style:

    columns: dataset, method, metric, submetric, value

Output: metrics.tsv (always) + metrics.parquet (if pyarrow is available).
Running the full plan therefore yields the individual JSONs AND one aggregated
data file that contains all three metric layers together.
"""
import argparse
import csv
import glob
import json
import os
import re

BC = re.compile(r"variant|threshold|crispat_method")
CANON = {"pgmm": {"mle": "pgmm_em", "map_e2": "pgmm_em_e2"},
         "crispat": {"pgmm": "crispat_pgmm", "2beta": "crispat_2beta"}}


def _method_of(path):
    """Recover method (with param variant) from an OB output path via its sibling
    parameters.json on the guide_assignment ancestor; falls back to the module id."""
    parts = os.path.normpath(path).split(os.sep)
    if "guide_assignment" not in parts:
        return "unknown"
    mod = parts[parts.index("guide_assignment") + 1]
    # walk up to the assignment node dir (…/guide_assignment/<mod>/.<hash>/) for parameters.json
    ga_idx = parts.index("guide_assignment")
    node_dir = os.sep.join(parts[: ga_idx + 3])
    params = {}
    pj = os.path.join(node_dir, "parameters.json")
    if os.path.exists(pj):
        try:
            params = json.load(open(pj))
        except Exception:
            params = {}
    if mod == "pgmm":
        return CANON["pgmm"].get(str(params.get("variant", "mle")), "pgmm_em")
    if mod == "umi":
        t = params.get("threshold")
        return f"umi_t{t}" if t is not None else "umi"
    if mod == "crispat":
        return CANON["crispat"].get(str(params.get("crispat_method")), mod)
    return mod


def _dataset_of(path):
    parts = os.path.normpath(path).split(os.sep)
    return parts[parts.index("data") + 1] if "data" in parts else "unknown"


def _flatten(obj, prefix=""):
    """Yield (submetric, value) for every numeric leaf, one nesting level deep."""
    for k, v in obj.items():
        if k in ("metric", "dataset", "guide_design", "match_regime", "method", "tool"):
            continue
        name = f"{prefix}{k}"
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            yield name, v
        elif isinstance(v, dict):
            for sk, sv in v.items():
                if isinstance(sv, (int, float)) and not isinstance(sv, bool):
                    yield f"{name}.{sk}", sv
                elif isinstance(sv, dict):
                    for ssk, ssv in sv.items():
                        if isinstance(ssv, (int, float)) and not isinstance(ssv, bool):
                            yield f"{name}.{sk}.{ssk}", ssv


def main():
    p = argparse.ArgumentParser(description="Aggregate per-lineage metric JSONs into one tidy table")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--name", default="metrics")
    # OB injects one flag per fan-in input id; accept any *.scores plus generic tokens.
    for flag in ("metrics_gt.scores", "metrics_gex.scores", "metrics_discovery.scores", "scores"):
        p.add_argument(f"--{flag}", action="append", default=[])
    p.add_argument("--scores_dir", action="append", default=[])
    args, extra = p.parse_known_args()

    files = []
    for flag in ("metrics_gt.scores", "metrics_gex.scores", "metrics_discovery.scores", "scores"):
        files += list(getattr(args, flag.replace(".", "_")) if hasattr(args, flag.replace(".", "_")) else getattr(args, flag))
    for d in args.scores_dir:
        files += glob.glob(os.path.join(d, "**", "*.scores.json"), recursive=True)
    files += [t for t in extra if t.endswith(".scores.json")]
    files = sorted(set(files))

    rows = []
    for f in files:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        metric = d.get("metric") or os.path.basename(f).split(".scores")[0]
        dataset = d.get("dataset") or _dataset_of(f)
        method = _method_of(f)
        for sub, val in _flatten(d):
            rows.append({"dataset": dataset, "method": method, "metric": metric,
                         "submetric": sub, "value": val})

    os.makedirs(args.output_dir, exist_ok=True)
    tsv = os.path.join(args.output_dir, "metrics.tsv")
    cols = ["dataset", "method", "metric", "submetric", "value"]
    with open(tsv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    try:
        import pandas as pd
        pd.DataFrame(rows, columns=cols).to_parquet(
            os.path.join(args.output_dir, "metrics.parquet"), index=False)
        pq = " + metrics.parquet"
    except Exception:
        pq = " (parquet skipped: pyarrow not available)"
    print(f"aggregate: {len(rows)} rows from {len(files)} metric files -> metrics.tsv{pq}")


if __name__ == "__main__":
    main()
