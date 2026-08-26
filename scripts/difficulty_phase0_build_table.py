#!/usr/bin/env python3
"""
Phase 0 — Per-Cell Difficulty Table (model-independent, pre-assignment).

Reads raw guide count MEX (CSR sparse), iterates per-cell non-zero entries.
Outputs one TSV per extraction with per-cell difficulty coordinates.

No GEX dependency. No assignment CSV dependency.
Peak RSS ~500 MB, single-threaded, ~2 min per extraction.

Columns emitted:
  cell_id, lane, extraction, libsize, n_detected,
  top1_umi, top2_umi, delta,
  entropy_lib, entropy_det, perplexity, k80,
  libsize_pctl_in_lane

Output:
  benchmark_output/difficulty_stratification/cell_difficulty_ham.tsv
  benchmark_output/difficulty_stratification/cell_difficulty_simpleaf.tsv
"""
import gzip, re, os, sys, time, csv
import numpy as np
from scipy import sparse
import scipy.io

STARTER = "/data/yunzliu/assignment_benchmark_starter"

DATASETS = {
    "replogle2022": {
        "extractions": {
            "ham":     "/data/yunzliu/results/guide_extraction/hash_matcher_final/merged",
            "simpleaf":"/data/yunzliu/assignment_benchmark_starter/04_extraction_mex/simpleaf_k15",
        },
        "out_dir": os.path.join(STARTER, "benchmark_output/difficulty_stratification"),
        "barcode_pattern": r'^([ACGT]{16})-L(\d+)$',
    },
    "papalexi2021": {
        "extractions": {
            "ham":     "/data/yunzliu/results/guide_extraction/papalexi_2021/ham/merged",
            "simpleaf":"/data/yunzliu/results/guide_extraction/papalexi_2021/simpleaf/merged",
        },
        "out_dir": os.path.join(STARTER, "11_papalexi_benchmark/02_results/benchmark/difficulty_stratification"),
        "barcode_pattern": r'^([ACGT]{16})-L(\d+)$',
    },
}

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="replogle2022", choices=["replogle2022","papalexi2021"])
args = ap.parse_args()

cfg = DATASETS[args.dataset]
MEX_PATHS = cfg["extractions"]
OUT_DIR = cfg["out_dir"]
os.makedirs(OUT_DIR, exist_ok=True)

BC_LANE = re.compile(cfg["barcode_pattern"])

# ── MEX loader: returns CSR + barcodes + total guides in library ──
def load_mex_csr(mex_dir):
    """Load MEX trio → (csr_matrix, barcodes, n_total_guides)."""
    t0 = time.time()
    with gzip.open(f"{mex_dir}/merged_matrix.mtx.gz", "rt") as f:
        mtx = scipy.io.mmread(f).tocsr()
    with gzip.open(f"{mex_dir}/merged_barcodes.tsv.gz", "rt") as f:
        barcodes = [l.strip() for l in f]
    n_guides = mtx.shape[1]
    print(f"  MEX: {mtx.shape[0]:,} cells × {n_guides} guides, {mtx.nnz:,} nnz [{time.time()-t0:.0f}s]")
    return mtx, barcodes, n_guides

# ── Per-cell difficulty computation ──
def compute_row(row_data, n_total_guides):
    """Compute difficulty metrics from one CSR row (numpy array of UMI values)."""
    vals = row_data.astype(np.float64)
    libsize = vals.sum()
    n_det = len(vals)
    if n_det == 0:
        return (0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 1)

    # Top-1, Top-2
    sorted_vals = np.sort(vals)[::-1]
    top1 = sorted_vals[0]
    top2 = sorted_vals[1] if n_det >= 2 else 0
    delta = (top1 - top2) / max(libsize, 1e-8)

    # Shannon entropy
    freqs = vals / libsize
    H = -np.sum(freqs * np.log2(freqs + 1e-300))
    entropy_lib = H / np.log2(n_total_guides)
    entropy_det = H / np.log2(max(n_det, 2))
    perplexity = 2.0 ** H

    # k80
    cumsum = np.cumsum(sorted_vals)
    k80 = int(np.searchsorted(cumsum, 0.80 * libsize, side='right') + 1)
    k80 = min(k80, n_det)

    return (libsize, n_det, top1, top2, delta, entropy_lib, entropy_det, perplexity, k80, k80)

# ── Main ──
ts = time.time()
print("=" * 60)
print("Phase 0 — Per-Cell Difficulty Table")
print("=" * 60)

for label, mex_dir in MEX_PATHS.items():
    print(f"\n[{label}]")
    mtx, barcodes, n_total_guides = load_mex_csr(mex_dir)

    # Pre-allocate arrays
    n_cells = mtx.shape[0]
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

    # Iterate rows
    for i in range(n_cells):
        bc = barcodes[i]
        m = BC_LANE.match(bc)
        lane_id = int(m.group(2)) if m else 0
        cell_id = bc
        lanes[i] = lane_id
        cell_ids.append(cell_id)

        row = mtx[i, :]
        vals = row.data
        if len(vals) == 0:
            libsizes[i] = 0; n_detected[i] = 0
            top1_umis[i] = 0; top2_umis[i] = 0
            deltas[i] = 0.0; ent_lib[i] = 0.0; ent_det[i] = 0.0
            perplexities[i] = 1.0; k80s[i] = 0
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

        libsizes[i] = libsize; n_detected[i] = n_det
        top1_umis[i] = t1; top2_umis[i] = t2
        deltas[i] = dlt; ent_lib[i] = e_lib; ent_det[i] = e_det
        perplexities[i] = perp; k80s[i] = k

        if (i + 1) % 100000 == 0:
            print(f"  processed {i+1:,}/{n_cells:,} cells [{time.time()-ts:.0f}s]")

    # Lane-normalized libsize percentile
    unique_lanes = sorted(set(lanes))
    libsize_pct = np.zeros(n_cells, dtype=np.float64)
    for l in unique_lanes:
        mask = lanes == l
        lv = libsizes[mask]
        ranks = np.searchsorted(np.sort(lv), lv, side='right') - 1
        libsize_pct[mask] = ranks / max(len(lv) - 1, 1) * 100.0

    # Write TSV
    out_path = os.path.join(OUT_DIR, f"cell_difficulty_{label.lower()}.tsv")
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

    print(f"  → {out_path}")

    # Summary stats
    mask_pos = libsizes > 0
    print(f"  cells with libsize>0: {mask_pos.sum():,}")
    print(f"  libsize:    p25={np.percentile(libsizes[mask_pos],25):.0f} med={np.median(libsizes[mask_pos]):.0f} p75={np.percentile(libsizes[mask_pos],75):.0f}")
    print(f"  delta:      p25={np.percentile(deltas[mask_pos],25):.4f} med={np.median(deltas[mask_pos]):.4f} p75={np.percentile(deltas[mask_pos],75):.4f}")
    print(f"  entropy_lib: p25={np.percentile(ent_lib[mask_pos],25):.4f} med={np.median(ent_lib[mask_pos]):.4f} p75={np.percentile(ent_lib[mask_pos],75):.4f}")
    print(f"  n_detected: p25={np.percentile(n_detected[mask_pos],25):.0f} med={np.median(n_detected[mask_pos]):.0f} p75={np.percentile(n_detected[mask_pos],75):.0f}")

    del mtx

print(f"\nDone [{time.time()-ts:.0f}s]")
