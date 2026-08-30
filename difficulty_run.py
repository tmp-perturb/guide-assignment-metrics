#!/usr/bin/env python3
"""Omnibenchmark module: guide_assignment_difficulty.

phase=table   -> Phase 0 per-cell difficulty table from the MEX trio
                 (entropy_lib/entropy_det/delta/perplexity/k80/libsize_pctl_in_lane).
                 Method-independent artifact. Logic copied VERBATIM from the
                 vendored scripts/difficulty_phase0_build_table.py per-cell loop.

Contract:
    --output_dir <dir> --name <dataset_id> [--phase table]
    --data.matrix / --data.barcodes / --data.features
Output:
    <output_dir>/{name}_cell_difficulty.tsv

NB the Phase-2 stratified knockdown analysis lives in run.py as the
`strat_delta_kd` metric (stratum-binned KD, wired into the metrics_gex stage).
"""
import argparse
import csv
import gzip
import os
import re
import time

import numpy as np
import scipy.io

BC_LANE = re.compile(r'^([ACGT]{16})-L(\d+)$')


def _mex_dir(matrix, barcodes, features, workdir):
    d = os.path.join(workdir, "mex")
    os.makedirs(d, exist_ok=True)
    for src, name in ((matrix, "merged_matrix.mtx.gz"),
                      (barcodes, "merged_barcodes.tsv.gz"),
                      (features, "merged_features.tsv.gz")):
        dst = os.path.join(d, name)
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(os.path.abspath(src), dst)
    return d


def phase_table(args):
    """Verbatim Phase-0 per-cell computation from difficulty_phase0_build_table.py."""
    import tempfile
    matrix = getattr(args, "data.matrix")
    barcodes_p = getattr(args, "data.barcodes")
    features = getattr(args, "data.features")
    out_path = os.path.join(args.output_dir, f"{args.name}_cell_difficulty.tsv")
    label = args.name
    ts = time.time()

    with tempfile.TemporaryDirectory() as work:
        mex_dir = _mex_dir(matrix, barcodes_p, features, work)
        with gzip.open(f"{mex_dir}/merged_matrix.mtx.gz", "rt") as f:
            mtx = scipy.io.mmread(f).tocsr()
        with gzip.open(f"{mex_dir}/merged_barcodes.tsv.gz", "rt") as f:
            barcodes = [l.strip() for l in f]
    n_total_guides = mtx.shape[1]
    n_cells = mtx.shape[0]
    print(f"  MEX: {n_cells:,} cells × {n_total_guides} guides, {mtx.nnz:,} nnz")

    lanes = np.zeros(n_cells, dtype=np.int16)
    cell_ids = []
    libsizes = np.zeros(n_cells, dtype=np.float64)
    n_detected = np.zeros(n_cells, dtype=np.int32)
    top1_umis = np.zeros(n_cells, dtype=np.int32)
    top2_umis = np.zeros(n_cells, dtype=np.int32)
    deltas = np.zeros(n_cells, dtype=np.float64)
    ent_lib = np.zeros(n_cells, dtype=np.float64)
    ent_det = np.zeros(n_cells, dtype=np.float64)
    perplexities = np.zeros(n_cells, dtype=np.float64)
    k80s = np.zeros(n_cells, dtype=np.int16)

    for i in range(n_cells):
        bc = barcodes[i]
        m = BC_LANE.match(bc)
        lanes[i] = int(m.group(2)) if m else 0
        cell_ids.append(bc)
        vals = mtx[i, :].data
        if len(vals) == 0:
            perplexities[i] = 1.0
            continue
        libsize = vals.sum()
        n_det = len(vals)
        sorted_vals = np.sort(vals)[::-1]
        t1 = sorted_vals[0]
        t2 = sorted_vals[1] if n_det >= 2 else 0
        dlt = (t1 - t2) / max(libsize, 1e-8)
        freqs = vals.astype(np.float64) / libsize
        H = -np.sum(freqs * np.log2(freqs + 1e-300))
        e_lib = H / np.log2(max(n_total_guides, 2))
        e_det = H / np.log2(max(n_det, 2))
        perp = 2.0 ** H
        cumsum = np.cumsum(sorted_vals)
        k = int(np.searchsorted(cumsum, 0.80 * libsize, side='right') + 1)
        k = min(k, n_det)
        libsizes[i] = libsize
        n_detected[i] = n_det
        top1_umis[i] = t1
        top2_umis[i] = t2
        deltas[i] = dlt
        ent_lib[i] = e_lib
        ent_det[i] = e_det
        perplexities[i] = perp
        k80s[i] = k

    unique_lanes = sorted(set(lanes))
    libsize_pct = np.zeros(n_cells, dtype=np.float64)
    for l in unique_lanes:
        mask = lanes == l
        lv = libsizes[mask]
        ranks = np.searchsorted(np.sort(lv), lv, side='right') - 1
        libsize_pct[mask] = ranks / max(len(lv) - 1, 1) * 100.0

    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(["cell_id", "lane", "extraction", "libsize", "n_detected",
                    "top1_umi", "top2_umi", "delta",
                    "entropy_lib", "entropy_det", "perplexity", "k80",
                    "libsize_pctl_in_lane"])
        for i in range(n_cells):
            w.writerow([
                cell_ids[i], lanes[i], label,
                int(libsizes[i]), int(n_detected[i]),
                int(top1_umis[i]), int(top2_umis[i]),
                f"{deltas[i]:.6f}",
                f"{ent_lib[i]:.6f}", f"{ent_det[i]:.6f}",
                f"{perplexities[i]:.4f}", int(k80s[i]),
                f"{libsize_pct[i]:.2f}",
            ])
    print(f"difficulty(table): wrote {os.path.basename(out_path)} "
          f"[{time.time()-ts:.0f}s, {n_cells:,} cells]")


def main():
    p = argparse.ArgumentParser(description="Omnibenchmark module: guide_assignment_difficulty")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--name", default="dataset")
    p.add_argument("--phase", default="table", choices=["table"])
    p.add_argument("--data.matrix")
    p.add_argument("--data.barcodes")
    p.add_argument("--data.features")
    args, _ = p.parse_known_args()
    os.makedirs(args.output_dir, exist_ok=True)
    phase_table(args)


if __name__ == "__main__":
    main()
