#!/usr/bin/env python3
"""
Assignment Benchmark — 6 methods × 3 extraction tools vs ground truth.

Follows the proven logic from:
  /data/yunzliu/results/guide_extraction/benchmark_collection/scripts/plot_assignment_figures.py

Key design decisions carried forward:
  - Cell matching by (gem_group or lane, 16mer) — NOT bare 16mer
  - Score-based per-cell ranking using method-specific sort column
  - T2/T3 independent of T1 (any top-k guide pair hits true pair)
  - guide name normalisation via to_dot()

Usage:
  python benchmark_assignments.py --all
"""

import os, sys, json, re, time, csv, warnings
from collections import defaultdict, Counter
import numpy as np
import h5py
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────
H5AD_PATH   = '/data/yunzliu/references/published/K562_essential_raw_singlecell_01.h5ad'
GUIDE_CSV   = '/data/yunzliu/references/raw_guides_k562_essential.csv'
TRANS_TABLE = '/data/yunzliu/Replogle2022_K562_Day6_benchmark/01_references/from_10x/cellranger_whitelist_translation_3v3.txt'
STARTER_BASE = '/data/yunzliu/assignment_benchmark_starter'

BC_STD  = re.compile(r'^([ACGT]{16})-(\d+)$')       # ground truth: 16mer-gem_group
BC_LANE = re.compile(r'^([ACGT]{16})-L(\d+)$')      # assignment:   16mer-LNN

def to_dot(g):
    """Normalise guide name: _23- → .23- for mapping consistency."""
    return g.replace('_23-', '.23-')


# ══════════════════════════════════════════════════════════════════════════
# Load reference data
# ══════════════════════════════════════════════════════════════════════════
def load_ground_truth():
    """Return dict: (gem_group, 16mer) → 'sgA|sgB' pair label."""
    print("Loading ground truth …", end=' ', flush=True)
    t0 = time.time()
    f = h5py.File(H5AD_PATH, 'r')
    cats = f['obs']['__categories']['sgID_AB']
    gt = {}
    for i in range(len(f['obs']['cell_barcode'])):
        bc = f['obs']['cell_barcode'][i]
        s = bc.decode() if isinstance(bc, bytes) else str(bc)
        m = BC_STD.match(s)
        if m:
            key = (int(m.group(2)), m.group(1))  # (gem_group, 16mer)
            sg = cats[f['obs']['sgID_AB'][i]]
            label = sg.decode() if isinstance(sg, bytes) else str(sg)
            gt[key] = label
    f.close()
    print(f"{len(gt):,} cells  [{time.time()-t0:.1f}s]")
    return gt


def load_guide_mapping():
    """Return (sg2pair, pair2guides, label2pair)."""
    print("Loading guide mapping …", end=' ', flush=True)
    sg2pair, pair2guides = {}, {}
    with open(GUIDE_CSV) as f:
        f.readline()
        for line in f:
            p = line.strip().split(',')
            if len(p) < 8: continue
            pid, sgA, sgB = p[0], p[4], p[6]
            sg2pair[sgA] = pid; sg2pair[sgB] = pid
            pair2guides[pid] = [sgA, sgB]
    # label translation: sgA|sgB → pair_id (either order)
    label2pair = {}
    for pid, (sa, sb) in pair2guides.items():
        label2pair[f'{sa}|{sb}'] = pid
        label2pair[f'{sb}|{sa}'] = pid
    print(f"{len(sg2pair):,} sgIDs → {len(pair2guides):,} pairs")
    return sg2pair, pair2guides, label2pair


def load_barcode_translation():
    """Feature barcode → GEX 16mer, for Cell Ranger protospacer."""
    print("Loading barcode translation …", end=' ', flush=True)
    feat2gex = {}
    with open(TRANS_TABLE) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                feat2gex[parts[0]] = parts[1]
    print(f"{len(feat2gex):,} entries")
    return feat2gex


# ══════════════════════════════════════════════════════════════════════════
# Load assignments — each method has its own loader
# ══════════════════════════════════════════════════════════════════════════

def load_standard(fpath, sort_key='UMI_counts', sort_desc=True):
    """
    Load PGMM EM / crispat PGMM style CSV.
    Format: cell, gRNA, UMI_counts, [prob_gaussian, ...]

    When sort_key='prob_gaussian' AND the column exists, uses compound sort:
      (prob_gaussian DESC, UMI_counts DESC) — matching sgbenchmark's
      AssignmentLevel behaviour with prob_column configured.
    Otherwise sorts by sort_key alone.

    Returns dict: (lane, 16mer) → [(guide, prob_or_score, umi), …]
    """
    pgmm = defaultdict(list)
    has_prob = (sort_key == 'prob_gaussian')
    with open(fpath) as f:
        for row in csv.DictReader(f):
            cell = row.get('cell', '').strip()
            guide = row.get('gRNA', '').strip()
            if not cell or not guide: continue
            m = BC_LANE.match(cell)
            if not m: continue
            key = (int(m.group(2)), m.group(1))  # (lane, 16mer)
            umi = int(float(row.get('UMI_counts', 0) or 0))
            if has_prob:
                try:
                    prob = float(row.get('prob_gaussian', 0))
                except (ValueError, TypeError):
                    prob = 0.0
                pgmm[key].append((guide, prob, umi))
            else:
                try:
                    score = float(row.get(sort_key, umi))
                except (ValueError, TypeError):
                    score = float(umi)
                pgmm[key].append((guide, score, umi))
    # Sort
    if has_prob:
        # Compound: prob DESC, then UMI DESC (sgbenchmark AssignmentLevel behaviour)
        for k in pgmm:
            pgmm[k].sort(key=lambda x: (-x[1], -x[2]))
    else:
        for k in pgmm:
            pgmm[k].sort(key=lambda x: -x[1] if sort_desc else x[1])
    n = sum(len(v) for v in pgmm.values())
    print(f"    {n:,} rows  {len(pgmm):,} cells")
    return pgmm


def load_crispat_2beta(fpath):
    """
    Load 07_2beta style CSV.
    Format: cell, percent_counts, gRNA, batch, UMI_counts
    """
    pgmm = defaultdict(list)
    with open(fpath) as f:
        for row in csv.DictReader(f):
            cell = row.get('cell', '').strip()
            guide = row.get('gRNA', '').strip()
            if not cell or not guide: continue
            m = BC_LANE.match(cell)
            if not m: continue
            key = (int(m.group(2)), m.group(1))
            umi = int(float(row.get('UMI_counts', 0) or 0))
            score = float(row.get('percent_counts', 0) or 0)
            pgmm[key].append((guide, score, umi))
    for k in pgmm:
        pgmm[k].sort(key=lambda x: -x[1])  # higher percent_counts = better
    n = sum(len(v) for v in pgmm.values())
    print(f"    {n:,} rows  {len(pgmm):,} cells")
    return pgmm


def load_crispat_umi(fpath):
    """
    Load 08_umi style CSV.
    Format: cell, gRNA, UMI_counts
    """
    pgmm = defaultdict(list)
    with open(fpath) as f:
        for row in csv.DictReader(f):
            cell = row.get('cell', '').strip()
            guide = row.get('gRNA', '').strip()
            if not cell or not guide: continue
            m = BC_LANE.match(cell)
            if not m: continue
            key = (int(m.group(2)), m.group(1))
            umi = int(float(row.get('UMI_counts', 0) or 0))
            pgmm[key].append((guide, float(umi), umi))
    for k in pgmm:
        pgmm[k].sort(key=lambda x: -x[1])
    n = sum(len(v) for v in pgmm.values())
    print(f"    {n:,} rows  {len(pgmm):,} cells")
    return pgmm


def load_fishash(fpath, sort_key='odds_ratio_regularized', sort_desc=True):
    """
    Load 09_fishash CSV.
    Format: cell, gRNA, UMI_counts, log_pval, odds_ratio_regularized
    """
    pgmm = defaultdict(list)
    with open(fpath) as f:
        for row in csv.DictReader(f):
            cell = row.get('cell', '').strip()
            guide = row.get('gRNA', '').strip()
            if not cell or not guide: continue
            m = BC_LANE.match(cell)
            if not m: continue
            key = (int(m.group(2)), m.group(1))
            umi = int(float(row.get('UMI_counts', 0) or 0))
            try:
                score = float(row.get(sort_key, 0))
            except (ValueError, TypeError):
                score = 0.0
            pgmm[key].append((guide, score, umi))
    for k in pgmm:
        pgmm[k].sort(key=lambda x: -x[1] if sort_desc else x[1])
    n = sum(len(v) for v in pgmm.values())
    print(f"    {n:,} rows  {len(pgmm):,} cells")
    return pgmm


def load_cr_protospacer(dirpath):
    """
    Merge all 48 lane protospacer CSVs.
    Barcodes are already GEX format (16mer-gem_group) — NO translation needed.
    """
    all_files = sorted(
        [f for f in os.listdir(dirpath) if f.startswith('lane_') and f.endswith('.csv')],
        key=lambda x: int(re.search(r'lane_(\d+)', x).group(1))
    )
    pgmm = defaultdict(list)
    total_rows = 0
    for fn in all_files:
        fpath = os.path.join(dirpath, fn)
        with open(fpath) as f:
            for row in csv.DictReader(f):
                bc_full = row.get('cell_barcode', '').strip()  # GEX format: 16mer-{gem}
                fc = row.get('feature_call', '').strip()
                umis_str = row.get('num_umis', '').strip()
                if not bc_full or not fc: continue
                m = BC_STD.match(bc_full)
                if not m: continue
                key = (int(m.group(2)), m.group(1))  # (gem_group, GEX 16mer)
                guides = fc.split('|')
                umi_vals = [int(u) for u in umis_str.split('|')]
                for g, u in zip(guides, umi_vals):
                    pgmm[key].append((g, float(u), u))
                total_rows += len(guides)
    for k in pgmm:
        pgmm[k].sort(key=lambda x: -x[2])  # UMI descending
    print(f"    {total_rows:,} rows  {len(pgmm):,} cells  ({len(all_files)} lanes)")
    return pgmm


# ══════════════════════════════════════════════════════════════════════════
# Metrics (faithful to plot_assignment_figures.py logic)
# ══════════════════════════════════════════════════════════════════════════

def compute_metrics(pgmm, gt, sg2pair, label2pair):
    """Return dict of metrics matching the previous assignment_metrics.json schema."""
    pgmm_keys = set(pgmm.keys())
    gt_keys   = set(gt.keys())
    shared    = pgmm_keys & gt_keys
    n_shared  = len(shared)
    n_gt      = len(gt_keys)

    rec      = n_shared / max(n_gt, 1)
    jac      = n_shared / max(len(pgmm_keys | gt_keys), 1)

    t1 = 0; t2 = 0; t3 = 0
    per_construct_pred = Counter()
    per_construct_true = Counter()
    ari_pairs = []

    for key in shared:
        assigns = pgmm[key]          # already sorted by score desc
        sl = gt[key]                  # ground truth label 'sgA|sgB'
        sp = label2pair.get(sl, '')   # true pair_id

        if len(assigns) == 0:
            continue

        # T1: top-1 guide → pair_id == ground truth pair
        g1 = to_dot(assigns[0][0])
        p1 = sg2pair.get(g1, '')
        is_correct = bool(p1 and p1 == sp)
        if is_correct:
            t1 += 1

        # T2/T3: any of top-k guides → pair_id matches ground truth
        # (independent of T1 — correct in old script)
        for k in [2, 3]:
            tp = set()
            for j in range(min(k, len(assigns))):
                gj = to_dot(assigns[j][0])
                pj = sg2pair.get(gj, '')
                if pj:
                    tp.add(pj)
            if sp in tp:
                if k == 2:
                    t2 += 1
                else:
                    t3 += 1

        # Per-construct tracking (top-1 only)
        if sp:
            per_construct_pred[p1] += 1   # p1 is the predicted pair (top-1)
            per_construct_true[sp] += 1   # sp is the true pair

        # ARI
        ari_pairs.append((p1 if p1 else f"UNK_{key}", sp if sp else f"UNK_{key}"))

    # Per-construct Pearson r
    all_pids = set(per_construct_pred.keys()) | set(per_construct_true.keys())
    x_pc = [per_construct_true.get(pid, 0) for pid in all_pids]
    y_pc = [per_construct_pred.get(pid, 0) for pid in all_pids]
    if len(x_pc) > 2:
        r_pc = np.corrcoef(x_pc, y_pc)[0, 1]
    else:
        r_pc = 0.0

    # ARI
    ari_val = adjusted_rand_index(ari_pairs)

    # Descriptive
    gpc = [len(v) for v in pgmm.values()]
    all_umis = []
    for v in pgmm.values():
        for x in v:
            all_umis.append(x[2])

    return {
        'cell_recovery_rate':      round(rec, 6),
        'jaccard_cells':           round(jac, 6),
        'n_shared_cells':          n_shared,
        't1_pair_accuracy':        round(t1 / max(n_shared, 1), 6),
        't2_pair_accuracy':        round(t2 / max(n_shared, 1), 6),
        't3_pair_accuracy':        round(t3 / max(n_shared, 1), 6),
        'ari':                     round(ari_val, 6),
        'per_construct_pearson_r': round(float(r_pc), 6),
        'total_assignments':       sum(gpc),
        'cells_assigned':          len(pgmm),
        'guides_detected':         len(set(x[0] for v in pgmm.values() for x in v)),
        'guides_per_cell_median':  float(np.median(gpc)) if gpc else 0.0,
        'guides_per_cell_mean':    float(np.mean(gpc)) if gpc else 0.0,
        'guides_per_cell_max':     int(max(gpc)) if gpc else 0,
        'umi_median':              float(np.median(all_umis)) if all_umis else 0.0,
    }


def adjusted_rand_index(pairs):
    """ARI from list of (pred_label, true_label)."""
    if len(pairs) < 2:
        return 0.0
    pred_map = {}; true_map = {}; pi = 0; ti = 0
    clean = []
    for p, t in pairs:
        if p not in pred_map: pred_map[p] = pi; pi += 1
        if t not in true_map: true_map[t] = ti; ti += 1
        clean.append((pred_map[p], true_map[t]))
    n = len(clean)
    if n < 2: return 0.0
    contingency = defaultdict(lambda: defaultdict(int))
    for p, t in clean:
        contingency[p][t] += 1
    ps = Counter(); ts = Counter()
    for p, row in contingency.items():
        ps[p] = sum(row.values())
        for t, v in row.items():
            ts[t] += v
    sum_comb = sum(sum(v*(v-1)//2 for v in row.values()) for row in contingency.values())
    sum_pred = sum(v*(v-1)//2 for v in ps.values())
    sum_true = sum(v*(v-1)//2 for v in ts.values())
    total = n*(n-1)//2
    if total == 0: return 0.0
    expected = sum_pred * sum_true / total
    max_ij = (sum_pred + sum_true) / 2
    return (sum_comb - expected) / (max_ij - expected) if max_ij != expected else 0.0


# ══════════════════════════════════════════════════════════════════════════
# Method spec factory
# ══════════════════════════════════════════════════════════════════════════

def build_specs():
    """Return list of (method, tool, loader_fn, loader_kwargs)."""
    specs = []
    base = STARTER_BASE

    # 1. Cell Ranger protospacer — barcodes are already GEX format, no translation
    protospacer_dir = os.path.join(base, '01_cellranger_protospacer')
    specs.append(('cellranger_protospacer', 'cellranger',
                  'cr_protospacer', {'dirpath': protospacer_dir}))

    # 2. PGMM EM — 3 tools — UMI≥1 pre-filter, prob≥0.75 gate, sort by UMI DESC
    for tool in ['cellranger', 'ham', 'simpleaf_k15']:
        fpath = os.path.join(base, f'05_pgmm_em_assignment/{tool}/assignments.csv')
        if os.path.exists(fpath):
            specs.append(('pgmm_em', tool, 'standard',
                          {'fpath': fpath, 'sort_key': 'UMI_counts', 'sort_desc': True}))

    # 3. PGMM SVI (crispat) — 3 tools × UMI {0,3} — sorted by UMI_counts (no prob col)
    for tool in ['cellranger', 'ham', 'simpleaf_k15']:
        for umi in [0, 3]:
            fpath = os.path.join(base, f'06_pgmm_crispat/{tool}/UMI_{umi}/assignments.csv')
            if os.path.exists(fpath):
                specs.append((f'crispat_pgmm_umi{umi}', tool, 'standard',
                              {'fpath': fpath, 'sort_key': 'UMI_counts', 'sort_desc': True}))

    # 4. 2-Beta — 3 tools — sorted by percent_counts
    for dir_name, tool_label in [('cellranger_sparse', 'cellranger'), ('ham', 'ham'), ('simpleaf_k15', 'simpleaf_k15')]:
        fpath = os.path.join(base, f'07_2beta_crispat/{dir_name}/assignments.csv')
        if os.path.exists(fpath):
            specs.append(('crispat_2beta', tool_label, 'crispat_2beta', {'fpath': fpath}))

    # 5. UMI threshold (standalone) — 3 tools × t={3,5,10} — sorted by UMI_counts
    for tool in ['cellranger', 'ham', 'simpleaf_k15']:
        for t in [3, 5, 10]:
            fpath = os.path.join(base, f'08_umi_crispat/{tool}/t{t}/assignments.csv')
            if os.path.exists(fpath):
                specs.append((f'umi_threshold_t{t}', tool, 'crispat_umi', {'fpath': fpath}))

    # 6. Fishash (top-K post-processed) — 3 tools × top-1/2/3
    # Fishash: full assignment CSV sorted by log_pval ASC
    for tool in ['cellranger', 'ham', 'simpleaf_k15']:
        fpath = os.path.join(base, f'09_fishash/{tool}/assignments.csv')
        if os.path.exists(fpath):
            specs.append(('fishash', tool, 'fishash_topk', {'fpath': fpath}))

    return specs


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def load_fishash_topk(fpath, n_guides=None):
    """Fishash full assignment CSV. Sorts by log_pval ASC per cell.
    Consistent with scprocess-perturb standardize_assignment.py."""
    pgmm = defaultdict(list)
    with open(fpath) as f:
        for row in csv.DictReader(f):
            cell = row.get('cell','').strip(); guide = row.get('gRNA','').strip()
            if not cell or not guide: continue
            m = BC_LANE.match(cell)
            if not m: continue
            key = (int(m.group(2)), m.group(1))
            umi = int(float(row.get('UMI_counts',0) or 0))
            lp  = float(row.get('log_pval', 0) or 0)
            pgmm[key].append((guide, lp, umi))
    for k in pgmm: pgmm[k].sort(key=lambda x: x[1])  # log_pval ASC
    n = sum(len(v) for v in pgmm.values())
    print(f"    {n:,} rows  {len(pgmm):,} cells")
    return pgmm

LOADERS = {
    'standard':          load_standard,
    'crispat_2beta':     load_crispat_2beta,
    'crispat_umi':       load_crispat_umi,
    'fishash':           load_fishash,
    'fishash_topk':      load_fishash_topk,
    'cr_protospacer':    load_cr_protospacer,
}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--list', action='store_true')
    args = parser.parse_args()

    specs = build_specs()

    if args.list:
        for m, t, lf, lk in specs:
            print(f"{m:<30s} {t:<15s} {lf:<15s} {lk}")
        return

    if not args.all:
        print("Use --all to run all specs, or --list to see them.")
        sys.exit(1)

    # Load references
    gt = load_ground_truth()
    sg2pair, pair2guides, label2pair = load_guide_mapping()
    feat2gex = load_barcode_translation()

    out_dir = os.path.join(STARTER_BASE, '10_benchmark_results')
    os.makedirs(out_dir, exist_ok=True)
    all_results = []

    for method, tool, loader_name, lk in specs:
        label = f"{method}__{tool}"
        out_json = os.path.join(out_dir, f"{label}.json")

        print(f"\n{'='*80}")
        print(f"  {label}")
        print(f"  Loader: {loader_name}  kwargs: {lk}")
        print(f"{'='*80}")

        t0 = time.time()

        # Call loader — no feat2gex needed (protospacer barcodes are GEX)
        pgmm = LOADERS[loader_name](**lk)

        metrics = compute_metrics(pgmm, gt, sg2pair, label2pair)
        wall = round(time.time() - t0, 1)

        result = {
            'method': method, 'tool': tool,
            '_sorted_by': (
                lk.get('sort_key') or
                {'crispat_2beta': 'percent_counts',
                 'crispat_umi': 'UMI_counts',
                 'cr_protospacer': 'UMI_counts'}.get(loader_name, 'UMI_counts')
            ),
            '_label': f"{method} ({tool})",
            'wall_s': wall,
            **metrics,
        }

        with open(out_json, 'w') as f:
            json.dump(result, f, indent=2, default=str)

        all_results.append(result)

        print(f"  Recovery: {metrics['cell_recovery_rate']:.4f}  "
              f"T1: {metrics['t1_pair_accuracy']:.4f}  "
              f"T2: {metrics['t2_pair_accuracy']:.4f}  "
              f"T3: {metrics['t3_pair_accuracy']:.4f}  "
              f"Pearson_r: {metrics['per_construct_pearson_r']:.4f}  "
              f"ARI: {metrics['ari']:.4f}  "
              f"gpC med: {metrics['guides_per_cell_median']:.0f}  [{wall}s]")

    # Summary table
    if len(all_results) > 1:
        summary_json = os.path.join(out_dir, '_all_methods_summary.json')
        with open(summary_json, 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"\n{'='*100}")
        print(f"  CROSS-METHOD SUMMARY  ({len(all_results)} combinations)")
        print(f"{'='*100}")
        print(f"{'Method/Tool':<45s} {'Rec':>6s} {'T1':>8s} {'T2':>8s} {'T3':>8s} {'ARI':>8s} {'PearsonR':>8s} {'gpC':>5s}")
        print("-" * 100)
        for r in sorted(all_results, key=lambda r: -r['t1_pair_accuracy']):
            print(f"{r['method']+'__'+r['tool']:<45s} "
                  f"{r['cell_recovery_rate']:6.4f} "
                  f"{r['t1_pair_accuracy']:8.4f} "
                  f"{r['t2_pair_accuracy']:8.4f} "
                  f"{r['t3_pair_accuracy']:8.4f} "
                  f"{r['ari']:8.4f} "
                  f"{r['per_construct_pearson_r']:8.4f} "
                  f"{r['guides_per_cell_median']:5.1f}")
        print(f"{'='*100}\nSaved: {summary_json}")


if __name__ == '__main__':
    main()
