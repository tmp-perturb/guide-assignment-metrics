#!/usr/bin/env python3
"""Phase 2: crispat-style discovery + FPR using Mann-Whitney U + BH correction."""
import sys, os, json, time, csv, re, argparse, gzip
import numpy as np
from collections import defaultdict
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from multiprocessing import Pool
import h5py

sys.path.insert(0, os.path.dirname(__file__))
from benchmark_assignments import (load_standard, load_crispat_2beta,
                                    load_crispat_umi, load_fishash_topk, STARTER_BASE)
STARTER = STARTER_BASE

DATASETS = {
    'replogle2022': {
        'gex_h5ad': '/data/yunzliu/Replogle2022_K562_Day6_benchmark/02_gex/post_decontx/TruSeq_decontx_merged_48lanes.h5ad',
        'guide_csv': '/data/yunzliu/references/raw_guides_k562_essential.csv',
        'guide_design': 'dual', 'min_cells_per_construct': 10,
        'kd_v2_dir': '12_kd_efficiency/replogle2022',
    },
    'papalexi2021': {
        'gex_h5ad': '/data/yunzliu/papalexi_2021_benchmark/01_reference/papalexi_2021_rna.h5ad',
        'assignment_ref': '/data/yunzliu/papalexi_2021_benchmark/01_reference/papalexi_2021_assignment.csv',
        'guide_design': 'single', 'min_cells_per_construct': 10,
        'kd_v2_dir': '11_papalexi_benchmark/02_results/benchmark',
    },
}

LOADERS = {
    'standard': load_standard, 'crispat_2beta': load_crispat_2beta,
    'crispat_umi': load_crispat_umi, 'fishash_topk': load_fishash_topk,
}

NL = '\n'


def load_replogle_gex():
    """Replogle Nextera GEX (anndata sparse CSC, 643k x 116k ENSG genes)."""
    import anndata as ad
    print("Loading Replogle GEX (Nextera, 8.4 GB) ...", end=' ', flush=True)
    t0 = time.time()
    sc = ad.read_h5ad(DATASETS['replogle2022']['gex_h5ad'])
    X_full = sc.X  # sparse CSC
    nc, ng_full = X_full.shape

    # HVG: sample 30k cells, per-column variance (CSC)
    from scipy.sparse import csc_matrix, issparse
    sample_n = min(30000, nc)
    idx_s = np.sort(np.random.RandomState(42).choice(nc, sample_n, replace=False))
    X_sample = X_full[idx_s, :]
    if not issparse(X_sample):
        X_sample = csc_matrix(X_sample)
    col_mean = np.array(X_sample.mean(axis=0)).flatten()
    col_mean_sq = np.array(X_sample.power(2).mean(axis=0)).flatten()
    gene_vars = col_mean_sq - col_mean ** 2
    top_idx = np.argsort(gene_vars)[-5000:]; top_idx = np.sort(top_idx)

    # Extract top-5000 HVG as dense numpy (for MW-U)
    X = X_full[:, top_idx].toarray().astype(np.float32)
    # Gene names: use ENSG IDs stripped (labels only, MW-U doesn't use them)
    gene_names = []
    for i in top_idx:
        v = sc.var_names[i]
        s = v.decode('utf-8') if isinstance(v, bytes) else str(v)
        gene_names.append(s.replace('_S', ''))

    # Cell lookup (Nextera: barcode_16mer + lane)
    cell_lookup = {}
    for i in range(nc):
        seq = str(sc.obs.barcode_16mer.iloc[i])
        lane = int(sc.obs.lane.iloc[i])
        cell_lookup[(lane, seq)] = i
    del sc

    # Guide map
    sg2gene = {}; nt_sgrnas = set(); guide_info = {}
    with open(DATASETS['replogle2022']['guide_csv']) as gf:
        gf.readline()
        for line in gf:
            p = line.strip().split(',')
            if len(p) < 8: continue
            gene, sgA, sgB, pid = p[1], p[4], p[6], p[0]
            if gene == 'non-targeting':
                if sgA: nt_sgrnas.add(sgA)
                if sgB: nt_sgrnas.add(sgB)
            sg2gene[sgA] = gene; sg2gene[sgB] = gene
            if sgA: guide_info[sgA] = (pid, gene)
            if sgB: guide_info[sgB] = (pid, gene)

    print(f"{nc:,}x{ng_full} -> {X.shape[1]:,} genes, {len(guide_info):,} pairs [{time.time()-t0:.0f}s]")
    return X, cell_lookup, gene_names, sg2gene, nt_sgrnas, guide_info


def load_papalexi_gex():
    """Papalexi: sparse CSC h5ad (20k x 18k) — compute variance on CSC columns."""
    print("Loading Papalexi GEX (70M nnz) ...", end=' ', flush=True)
    t0 = time.time()
    f = h5py.File(DATASETS['papalexi2021']['gex_h5ad'], 'r')
    X_g = f['X']
    data = X_g['data'][:]; indices = X_g['indices'][:]; indptr = X_g['indptr'][:]
    ng_full = len(indptr) - 1
    nc = int(indices.max()) + 1
    from scipy.sparse import csc_matrix
    X_csc = csc_matrix((data, indices, indptr), shape=(nc, ng_full))

    # Variance directly on CSC columns (fast — column-major)
    gene_vars = np.zeros(ng_full, dtype=np.float64)
    for j in range(ng_full):
        col = X_csc[:, j].toarray().flatten()
        gene_vars[j] = col.var()

    top_idx = np.argsort(gene_vars)[-5000:]; top_idx = np.sort(top_idx)
    # Load selected columns as numpy
    X = np.zeros((nc, 5000), dtype=np.float32)
    for k, gi in enumerate(top_idx):
        X[:, k] = X_csc[:, gi].toarray().flatten()

    # Gene names
    if 'var' in f and 'index' in f['var']:
        var_idx = f['var']['index'][:]
        all_gene_names = [x.decode('utf-8') if isinstance(x, bytes) else str(x) for x in var_idx]
    else:
        all_gene_names = [str(i) for i in range(ng_full)]
    gene_names = [all_gene_names[i] for i in top_idx]

    # Cell barcodes
    cbs = f['obs']['index'][:]
    cbs_dec = [x.decode('utf-8') if isinstance(x, bytes) else str(x) for x in cbs]
    cell_lookup = {}
    bc_pat = re.compile(r'^l(\d+)_([ACGT]{16})$')
    for i, bc in enumerate(cbs_dec):
        m = bc_pat.match(bc)
        if m: cell_lookup[(int(m.group(1)), m.group(2))] = i
    f.close()

    # Guide map (from Papalexi assignment CSV)
    sg2gene = {}; nt_sgrnas = set(); guide_info = {}
    with open(DATASETS['papalexi2021']['assignment_ref']) as gf:
        for row in csv.DictReader(gf):
            gid = row.get('guide_ID', '').strip()
            gene = row.get('gene_target', '').strip()
            nt = row.get('NT', '').strip(); pert = row.get('perturbation', '').strip()
            if not gid: continue
            sg2gene[gid] = gene; guide_info[gid] = gene
            if nt.startswith('NT') or pert == 'NT': nt_sgrnas.add(gid)

    print(f"{nc:,}x{ng_full} -> {X.shape[1]:,} genes, {len(cell_lookup):,} cell_lookup [{time.time()-t0:.0f}s]")
    return X, cell_lookup, gene_names, sg2gene, nt_sgrnas, guide_info


def _de_one(args):
    """Mann-Whitney U on construct cells vs NT cells."""
    X_cells, X_nt, cid = args
    nc, n_nt = X_cells.shape[0], X_nt.shape[0]
    if nc < 5 or n_nt < 5: return (cid, 0, X_cells.shape[1], nc)
    ng = X_cells.shape[1]
    pvals = np.ones(ng)
    for gi in range(ng):
        try:
            _, p = mannwhitneyu(X_cells[:, gi], X_nt[:, gi], alternative='two-sided')
            pvals[gi] = p
        except ValueError:
            pvals[gi] = 1.0
    _, p_adj, _, _ = multipletests(pvals, method='fdr_bh')
    return (cid, int((p_adj < 0.05).sum()), ng, nc)


def main():
    p = argparse.ArgumentParser(description='Phase 2')
    p.add_argument('--dataset', required=True, choices=['replogle2022', 'papalexi2021'])
    p.add_argument('--method', default=None); p.add_argument('--tool', default=None)
    p.add_argument('--workers', type=int, default=16); p.add_argument('--no-fpr', action='store_true')
    args = p.parse_args()

    ds = DATASETS[args.dataset]
    out_dir = os.path.join(STARTER, ds['kd_v2_dir'], 'discovery')
    os.makedirs(out_dir, exist_ok=True)

    # Load GEX once
    if args.dataset == 'replogle2022':
        X, cell_lookup, gene_names, sg2gene, nt_sgrnas, guide_info = load_replogle_gex()
    else:
        X, cell_lookup, gene_names, sg2gene, nt_sgrnas, guide_info = load_papalexi_gex()
    ng = X.shape[1]

    # Scan specs
    sys.path.insert(0, os.path.join(STARTER, '03_scripts'))
    from benchmark_kd_efficiency import build_specs_k562, build_specs_papalexi
    all_specs = build_specs_k562() if args.dataset == 'replogle2022' else build_specs_papalexi()
    keep = {'pgmm_em', 'crispat_pgmm_umi0', 'crispat_2beta',
            'umi_threshold_t3', 'umi_threshold_t5', 'umi_threshold_t10', 'fishash'}
    specs = [(n, t, ln, kw) for n, t, ln, kw in all_specs if n in keep
             and (not args.method or n == args.method) and (not args.tool or t == args.tool)]
    seen = set(); uniq = []
    for s in specs:
        if (s[0], s[1]) not in seen: seen.add((s[0], s[1])); uniq.append(s)
    specs = uniq

    print(f"{NL}{'='*80}{NL}  PHASE 2: {args.dataset}  ({len(specs)} combos, {ng} genes, {args.workers} workers){NL}{'='*80}{NL}")

    for method_name, tool, loader_name, loader_kw in specs:
        print(f"{'='*60}  {method_name} / {tool} {'='*60}")
        t_start = time.time()
        csv_path = loader_kw.get('fpath', '')
        if not os.path.exists(csv_path):
            print(f"  SKIP: {csv_path}"); continue
        loader_fn = LOADERS.get(loader_name)
        if not loader_fn: print(f"  SKIP: unknown loader"); continue
        sort_kw = {k: v for k, v in loader_kw.items() if k in ('sort_key', 'sort_desc', 'fpath')}
        pgmm = loader_fn(**sort_kw)
        n_assigns = sum(len(v) for v in pgmm.values()); n_cells = len(pgmm)
        print(f"  {n_assigns:,} rows, {n_cells:,} cells [{time.time()-t_start:.0f}s]")

        # Build per-construct cell index + NT pool
        construct_cells = defaultdict(set); nt_set = set(); n_unmatched = 0
        for key in pgmm:
            if not pgmm[key]: continue
            tg = pgmm[key][0][0]; idx = cell_lookup.get(key)
            if idx is None: n_unmatched += 1; continue
            if tg in nt_sgrnas: nt_set.add(idx)
            elif ds['guide_design'] == 'dual':
                info = guide_info.get(tg)
                if info: construct_cells[info[0]].add(idx)
            else:
                gene = sg2gene.get(tg, tg)
                if gene != 'non-targeting': construct_cells[gene].add(idx)

        nt_list = sorted(nt_set); n_nt = len(nt_list); n_cons = len(construct_cells)
        print(f"  {n_cons} constructs, {n_nt} NT cells, {n_unmatched} unmatched")
        if n_nt < 5: print(f"  SKIP: <5 NT cells"); continue

        X_nt = X[nt_list, :]
        cells_per = [len(v) for v in construct_cells.values()]
        median_cells = float(np.median(cells_per)) if cells_per else 0

        # ---- Discovery ----
        print(f"  Discovery: {n_cons} constructs ...", end=' ', flush=True)
        t1 = time.time()
        tasks = [(X[sorted(cells), :], X_nt, cid) for cid, cells in construct_cells.items()
                 if len(cells) >= ds['min_cells_per_construct']]
        discoveries = []
        if tasks:
            with Pool(args.workers) as pool:
                for cid, n_sig, ng_t, ncells in pool.map(_de_one, tasks):
                    discoveries.append(n_sig)
        n_tested = len(discoveries); skipped = n_cons - n_tested
        median_disc = float(np.median(discoveries)) if discoveries else 0
        total_disc = int(np.sum(discoveries))
        print(f"[{time.time()-t1:.0f}s] {n_tested} tested, median={median_disc:.1f}, total={total_disc}")

        # ---- FPR (separate pool) ----
        fpr_result = None
        if not args.no_fpr and n_nt > 0:
            print(f"  FPR: ...", end=' ', flush=True)
            t2 = time.time()
            nt_guide_cells = defaultdict(set)
            for key in pgmm:
                if not pgmm[key]: continue
                tg = pgmm[key][0][0]
                if tg not in nt_sgrnas: continue
                idx = cell_lookup.get(key)
                if idx is None: continue
                nt_guide_cells[tg].add(idx)
            fp_tasks = [(X[sorted(cells), :], X_nt, g_nt) for g_nt, cells in nt_guide_cells.items()
                        if len(cells) >= 5]
            fp_counts = []
            if fp_tasks:
                with Pool(min(args.workers, 8)) as pool:
                    for cid, n_sig, ng_t, ncells in pool.map(_de_one, fp_tasks):
                        fp_counts.append(n_sig)
            total_fp = int(np.sum(fp_counts)) if fp_counts else 0
            n_nt_tested = len(fp_tasks); denom = n_nt_tested * ng
            fpr = total_fp / denom if denom > 0 else 0
            print(f"[{time.time()-t2:.0f}s] {n_nt_tested} NT guides x {ng}={denom:,}, FP={total_fp}, FPR={fpr:.6f}")
            fpr_result = {'n_nt_guides_tested': n_nt_tested, 'total_false_discoveries': total_fp,
                          'total_tests': denom, 'fpr': fpr}

        # KD from Phase 1
        kd_path = os.path.join(STARTER, ds['kd_v2_dir'], f'{method_name}__{tool}.json')
        kd_median = None; kd_frac = None
        if os.path.exists(kd_path):
            with open(kd_path) as f: kd_d = json.load(f)
            kd_median = kd_d.get('kd_efficiency_median')
            kd_frac = kd_d.get('fraction_expected_direction')

        result = {'method': method_name, 'tool': tool, 'dataset': args.dataset,
                  'wall_s': round(time.time()-t_start, 1), 'n_cells_assigned': n_cells,
                  'median_cells_per_construct': median_cells, 'n_constructs_tested': n_tested,
                  'kd_median': kd_median, 'kd_frac_expected': kd_frac,
                  'discovery_n_tested': n_tested, 'discovery_median': median_disc,
                  'discovery_total': total_disc, 'n_nt_cells': n_nt}
        if fpr_result: result.update(fpr_result)

        out_path = os.path.join(out_dir, f'{method_name}__{tool}.json')
        with open(out_path, 'w') as f: json.dump(result, f, indent=2)
        print(f"  -> {out_path}")

    # Summary
    summary = []
    for fn in sorted(os.listdir(out_dir)):
        if fn.endswith('.json') and not fn.startswith('_'):
            with open(os.path.join(out_dir, fn)) as f: summary.append(json.load(f))
    summary.sort(key=lambda x: (x.get('discovery_median', 0), x.get('kd_median', 0) or 0), reverse=True)
    with open(os.path.join(out_dir, '_phase2_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"{NL}{'='*80}{NL}  Done: {len(summary)} profiles{NL}{'='*80}")


if __name__ == '__main__':
    main()
