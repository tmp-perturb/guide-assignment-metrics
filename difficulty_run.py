#!/usr/bin/env python3
"""Omnibenchmark module: guide_assignment_difficulty.

phase=table   -> Phase 0 per-cell difficulty table from the MEX trio
                 (entropy_lib/entropy_det/delta/perplexity/k80/libsize_pctl_in_lane).
                 Method-independent artifact. Logic copied VERBATIM from the
                 vendored scripts/difficulty_phase0_build_table.py per-cell loop.
phase=validate -> Phase 2 delta-KD validation (difficulty.table + umi_t3 ref + GEX).
                 Vendored scripts/difficulty_phase2_delta_kd_ham.py logic.

Contract:
    --output_dir <dir> --name <dataset_id> --phase <table|validate>
    table:    --data.matrix / --data.barcodes / --data.features
    validate: --data.difficulty_table --guide_assignment.assignments --data.gex [--data.spec]
Output:
    table    -> <output_dir>/{name}_cell_difficulty.tsv
    validate -> <output_dir>/{name}_phase2_delta_kd.json
"""
import argparse
import csv
import gzip
import os
import re
import sys
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


def phase_validate(args):
    """Phase 2 delta-KD validation. Copied verbatim from
    difficulty_phase2_delta_kd_ham.py, parameterised on injected inputs.
    Method-independent (uses the wired assignment, conventionally umi_t3, as ref)."""
    import json
    from collections import defaultdict
    import h5py  # noqa: F401
    import anndata as ad
    EPS = 0.01
    DECILES = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    diff_table = getattr(args, "data.difficulty_table")
    assignments = getattr(args, "guide_assignment.assignments")
    gex = getattr(args, "data.gex")
    guide_csv = getattr(args, "data.guide_map")
    sym_map = "/data/yunzliu/assignment_benchmark_starter/benchmark_output/gene_symbol_to_ensg.json"
    ts = time.time()

    delta_per_cell = {}
    with open(diff_table) as f:
        for row in csv.DictReader(f, delimiter='\t'):
            m = BC_LANE.match(row['cell_id'])
            if m:
                delta_per_cell[(int(m.group(2)), m.group(1))] = float(row['delta'])

    sg2gene = {}
    nt_sgrnas = set()
    with open(guide_csv) as gf:
        for line in csv.reader(gf):
            if line[0].startswith('unique'):
                continue
            if len(line) < 8:
                continue
            gene, sgA, sgB = line[1], line[4], line[6]
            if gene == 'non-targeting':
                if sgA:
                    nt_sgrnas.add(sgA)
                if sgB:
                    nt_sgrnas.add(sgB)
            sg2gene[sgA] = gene
            sg2gene[sgB] = gene

    top1_guide = {}
    with open(assignments) as f:
        for row in csv.DictReader(f):
            cell = row.get('cell', '').strip()
            guide = row.get('gRNA', '').strip()
            if not cell or not guide:
                continue
            m = BC_LANE.match(cell)
            if not m:
                continue
            k = (int(m.group(2)), m.group(1))
            score = int(float(row.get('UMI_counts', 0) or 0))
            if k not in top1_guide or score > top1_guide[k][1]:
                top1_guide[k] = (guide, score)
    for k in top1_guide:
        top1_guide[k] = top1_guide[k][0]

    needed_genes = set()
    guide_to_cells = defaultdict(list)
    for k, g in top1_guide.items():
        delta = delta_per_cell.get(k)
        if delta is None:
            continue
        gene = sg2gene.get(g)
        if g in nt_sgrnas:
            continue
        if gene:
            needed_genes.add(gene)
            guide_to_cells[(g, gene)].append((k, delta))
    for g_nt in nt_sgrnas:
        needed_genes.add(g_nt)

    with open(sym_map) as f:
        sym2ensg = json.load(f)
    sc = ad.read_h5ad(gex)
    nc = sc.shape[0]
    ensg2col = {}
    for i, v in enumerate(sc.var_names):
        s = v.decode('utf-8') if isinstance(v, bytes) else str(v)
        ensg2col[s.replace('_S', '')] = i
    gene2col = {}
    for sym in sorted(needed_genes):
        es = sym2ensg.get(sym)
        if es and es in ensg2col:
            gene2col[sym] = ensg2col[es]
    needed_cols = sorted(gene2col.values())
    col2loc = {c: i for i, c in enumerate(needed_cols)}
    X_small = sc.X[:, needed_cols].toarray().astype(np.float32)
    cell_lookup = {}
    for i in range(nc):
        seq = str(sc.obs.barcode_16mer.iloc[i])
        lane = int(sc.obs.lane.iloc[i])
        cell_lookup[(lane, seq)] = i
    del sc

    nt_set = set()
    for k, g in top1_guide.items():
        if g in nt_sgrnas:
            idx = cell_lookup.get(k)
            if idx is not None:
                nt_set.add(idx)
    nt_list = sorted(nt_set)
    gene_means, nt_means = {}, {}
    for sym in gene2col:
        idx = col2loc[gene2col[sym]]
        gene_means[sym] = float(X_small[:, idx].mean())
        nt_means[sym] = float(X_small[nt_list, idx].mean()) if nt_list else gene_means[sym]

    all_delta_kd, per_gene_corr, nt_delta_kd = [], [], []
    for (guide, gene), cells_deltas in guide_to_cells.items():
        if len(cells_deltas) < 20:
            continue
        g_col = gene2col.get(gene)
        if g_col is None:
            continue
        g_idx = col2loc[g_col]
        cells_deltas.sort(key=lambda x: x[1])
        deltas = np.array([d for _, d in cells_deltas])
        bounds = [np.percentile(deltas, p) for p in DECILES]
        for di in range(10):
            lo, hi = bounds[di], bounds[di + 1]
            mask = (deltas >= lo) & (deltas <= hi) if di < 9 else (deltas >= lo)
            bin_cells = [(k, d) for (k, d), mm in zip(cells_deltas, mask) if mm]
            if len(bin_cells) < 5:
                continue
            exprs = []
            for k, d in bin_cells:
                idx = cell_lookup.get(k)
                if idx is None:
                    continue
                exprs.append(X_small[idx, g_idx])
            if not exprs:
                continue
            mean_expr = np.mean(exprs)
            em = gene_means.get(gene, 1e-8)
            enm = nt_means.get(gene, em)
            epsg = EPS * max(em, 1e-8)
            kd = float(np.log2(max(mean_expr + epsg, epsg)) - np.log2(max(enm + epsg, epsg)))
            all_delta_kd.append((di, kd))
        cell_kds = []
        for k, d in cells_deltas:
            idx = cell_lookup.get(k)
            if idx is None:
                continue
            expr = float(X_small[idx, g_idx])
            em = gene_means.get(gene, 1e-8)
            enm = nt_means.get(gene, em)
            epsg = EPS * max(em, 1e-8)
            kd = np.log2(max(expr + epsg, epsg)) - np.log2(max(enm + epsg, epsg))
            cell_kds.append((d, kd))
        if len(cell_kds) >= 10:
            from scipy.stats import spearmanr
            d_rank = np.array([d for d, _ in cell_kds])
            kd_arr = np.array([v for _, v in cell_kds])
            r, p = spearmanr(d_rank, kd_arr)
            per_gene_corr.append((r, p))

    non_nt_genes = sorted(set(sg2gene.values()) - {'non-targeting'})
    rng = np.random.RandomState(42)
    nt_guide_cells = defaultdict(list)
    for k, g in top1_guide.items():
        if g in nt_sgrnas:
            delta = delta_per_cell.get(k)
            if delta is not None:
                nt_guide_cells[g].append((k, delta))
    for g_nt, cells_deltas in nt_guide_cells.items():
        if len(cells_deltas) < 10:
            continue
        cells_deltas.sort(key=lambda x: x[1])
        deltas = np.array([d for _, d in cells_deltas])
        if np.max(deltas) == np.min(deltas):
            continue
        bounds = [np.percentile(deltas, p) for p in DECILES]
        t_rand = rng.choice(non_nt_genes)
        g_col = gene2col.get(t_rand)
        if g_col is None:
            continue
        g_idx = col2loc[g_col]
        for di in range(10):
            lo, hi = bounds[di], bounds[di + 1]
            mask = (deltas >= lo) & (deltas <= hi) if di < 9 else (deltas >= lo)
            bin_cells = [(k, d) for (k, d), mm in zip(cells_deltas, mask) if mm]
            if len(bin_cells) < 5:
                continue
            exprs = []
            for k, d in bin_cells:
                idx = cell_lookup.get(k)
                if idx is None:
                    continue
                exprs.append(X_small[idx, g_idx])
            if not exprs:
                continue
            mean_expr = np.mean(exprs)
            em = gene_means.get(t_rand, 1e-8)
            enm = nt_means.get(t_rand, em)
            epsg = EPS * max(em, 1e-8)
            kd = float(np.log2(max(mean_expr + epsg, epsg)) - np.log2(max(enm + epsg, epsg)))
            nt_delta_kd.append((di, kd))

    target_by_decile, nt_by_decile = defaultdict(list), defaultdict(list)
    for di, kd in all_delta_kd:
        target_by_decile[di].append(kd)
    for di, kd in nt_delta_kd:
        nt_by_decile[di].append(kd)
    results = {"target": {}, "nt_control": {}, "within_guide_spearman": []}
    for di in range(10):
        if target_by_decile[di]:
            arr = np.array(target_by_decile[di])
            results["target"][f"D{di}"] = {"n": len(arr), "kd_median": float(np.median(arr)),
                                           "kd_mean": float(arr.mean()), "kd_std": float(arr.std())}
        if nt_by_decile[di]:
            arr = np.array(nt_by_decile[di])
            results["nt_control"][f"D{di}"] = {"n": len(arr), "kd_median": float(np.median(arr)),
                                               "kd_mean": float(arr.mean()), "kd_std": float(arr.std())}
    if per_gene_corr:
        rs = [r for r, p in per_gene_corr]
        ps = [p for r, p in per_gene_corr]
        results["within_guide_spearman"] = {
            "n_genes": len(rs), "r_median": float(np.median(rs)), "r_mean": float(np.mean(rs)),
            "frac_p_lt_005": float((np.array(ps) < 0.05).mean())}
    out = os.path.join(args.output_dir, f"{args.name}_phase2_delta_kd.json")
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"difficulty(validate): wrote {os.path.basename(out)} [{time.time()-ts:.0f}s]")


def main():
    p = argparse.ArgumentParser(description="Omnibenchmark module: guide_assignment_difficulty")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--name", default="dataset")
    p.add_argument("--phase", required=True, choices=["table", "validate"])
    p.add_argument("--data.matrix")
    p.add_argument("--data.barcodes")
    p.add_argument("--data.features")
    p.add_argument("--data.difficulty_table")
    p.add_argument("--guide_assignment.assignments")
    p.add_argument("--data.gex")
    p.add_argument("--data.spec")
    args, _ = p.parse_known_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.phase == "table":
        phase_table(args)
    else:
        phase_validate(args)


if __name__ == "__main__":
    main()
