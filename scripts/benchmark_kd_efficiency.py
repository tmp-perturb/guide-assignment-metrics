#!/usr/bin/env python3
"""
Knockdown Efficiency Benchmark — orthogonal to ground-truth-based accuracy.

Computes per-guide log2 fold-change of target gene expression in
assigned cells vs. non-targeting control cells.  Answers:
  "Do the cells this method assigned actually show reduced target gene
   expression — regardless of whether the label was 'correct'?"

Supports CRISPRi / CRISPRko (knockdown: expect log2FC < 0) and
CRISPRa (activation: expect log2FC > 0).

Usage:
  # All methods for Replogle 2022
  python benchmark_kd_efficiency.py --dataset replogle2022

  # Single method
  python benchmark_kd_efficiency.py --dataset replogle2022 \
      --method pgmm_em --tool ham

  # Papalexi
  python benchmark_kd_efficiency.py --dataset papalexi2021
"""

import os, sys, csv, re, json, time, gzip, argparse, warnings
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────
STARTER_BASE = '/data/yunzliu/assignment_benchmark_starter'

DATASET_CONFIGS = {
    'replogle2022': {
        'gex_h5ad': '/data/yunzliu/Replogle2022_K562_Day6_benchmark/02_gex/post_decontx/TruSeq_decontx_merged_48lanes.h5ad',
        'guide_csv': '/data/yunzliu/references/raw_guides_k562_essential.csv',
        'guide_map_format': 'pair_id',
        'perturbation_type': 'knockdown',
        'nt_gene': 'non-targeting',
        'gex_barcode_from': 'h5ad_obs',  # obs['cell_barcode'] → (gem_group, 16mer)
        'spec_key': 'k562',
    },
    'papalexi2021': {
        'gex_h5ad': '/data/yunzliu/papalexi_2021_benchmark/01_reference/papalexi_2021_rna.h5ad',
        'guide_map_csv': '/data/yunzliu/papalexi_2021_benchmark/01_reference/papalexi_2021_assignment.csv',
        'guide_map_format': 'papalexi_gt',
        'perturbation_type': 'knockdown',
        'nt_prefix': 'NTg',
        'gex_barcode_from': 'obs_index',
        'spec_key': 'papalexi',
        'out_dir': '11_papalexi_benchmark/02_results/benchmark',
    },
}

BC_LANE = re.compile(r'^([ACGT]{16})-L(\d+)$')
BC_GT   = re.compile(r'^l(\d+)_([ACGT]{16})$')


# ══════════════════════════════════════════════════════════════════════════
# Loaders — identical to benchmark_assignments.py loaders
# ══════════════════════════════════════════════════════════════════════════

def load_standard(fpath, sort_key='UMI_counts', sort_desc=True):
    pgmm = defaultdict(list)
    has_prob = (sort_key == 'prob_gaussian')
    with open(fpath) as f:
        for row in csv.DictReader(f):
            cell = row.get('cell','').strip(); guide = row.get('gRNA','').strip()
            if not cell or not guide: continue
            m = BC_LANE.match(cell)
            if not m: continue
            key = (int(m.group(2)), m.group(1))
            umi = int(float(row.get('UMI_counts',0) or 0))
            if has_prob:
                prob = float(row.get('prob_gaussian',0) or 0)
                pgmm[key].append((guide, prob, umi))
            else:
                try: score = float(row.get(sort_key, umi))
                except: score = float(umi)
                pgmm[key].append((guide, score, umi))
    if has_prob:
        for k in pgmm: pgmm[k].sort(key=lambda x: (-x[1], -x[2]))
    else:
        for k in pgmm: pgmm[k].sort(key=lambda x: -x[1] if sort_desc else x[1])
    return pgmm

def load_crispat_2beta(fpath):
    pgmm = defaultdict(list)
    with open(fpath) as f:
        for row in csv.DictReader(f):
            cell = row.get('cell','').strip(); guide = row.get('gRNA','').strip()
            if not cell or not guide: continue
            m = BC_LANE.match(cell)
            if not m: continue
            key = (int(m.group(2)), m.group(1))
            umi = int(float(row.get('UMI_counts',0) or 0))
            score = float(row.get('percent_counts',0) or 0)
            pgmm[key].append((guide, score, umi))
    for k in pgmm: pgmm[k].sort(key=lambda x: -x[1])
    return pgmm

def load_crispat_umi(fpath):
    pgmm = defaultdict(list)
    with open(fpath) as f:
        for row in csv.DictReader(f):
            cell = row.get('cell','').strip(); guide = row.get('gRNA','').strip()
            if not cell or not guide: continue
            m = BC_LANE.match(cell)
            if not m: continue
            key = (int(m.group(2)), m.group(1))
            umi = int(float(row.get('UMI_counts',0) or 0))
            pgmm[key].append((guide, float(umi), umi))
    for k in pgmm: pgmm[k].sort(key=lambda x: -x[1])
    return pgmm

def load_fishash(fpath, sort_key='odds_ratio_regularized', sort_desc=True):
    pgmm = defaultdict(list)
    with open(fpath) as f:
        for row in csv.DictReader(f):
            cell = row.get('cell','').strip(); guide = row.get('gRNA','').strip()
            if not cell or not guide: continue
            m = BC_LANE.match(cell)
            if not m: continue
            key = (int(m.group(2)), m.group(1))
            umi = int(float(row.get('UMI_counts',0) or 0))
            try: score = float(row.get(sort_key, 0))
            except: score = 0.0
            pgmm[key].append((guide, score, umi))
    for k in pgmm: pgmm[k].sort(key=lambda x: -x[1] if sort_desc else x[1])
    return pgmm

def load_fishash_topk(fpath):
    """Fishash full assignment CSV. 5 columns: cell,gRNA,UMI_counts,log_pval,odds_ratio_regularized.
    Sorts by log_pval ASC per cell (smaller = more significant). Top-1 = most significant.
    Consistent with scprocess-perturb standardize_assignment.py (sort_keys=[('log_pval','asc')]).
    Now reads FULL output (no postprocess top-1 filter)."""
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
    # Sort by log_pval ASC (smaller = more significant), consistent with postprocess_fishash.py
    for k in pgmm: pgmm[k].sort(key=lambda x: x[1])
    return pgmm

LOADERS = {
    'standard': load_standard,
    'crispat_2beta': load_crispat_2beta,
    'crispat_umi': load_crispat_umi,
    'fishash': load_fishash,
    'fishash_topk': load_fishash_topk,
}


# ══════════════════════════════════════════════════════════════════════════
# Spec builders
# ══════════════════════════════════════════════════════════════════════════

def build_specs_k562():
    specs = []; base = STARTER_BASE
    for tool in ['cellranger', 'ham', 'simpleaf_k15']:
        f = os.path.join(base, f'05_pgmm_em_assignment/{tool}/assignments.csv')
        if os.path.exists(f): specs.append(('pgmm_em', tool, 'standard', {'fpath': f, 'sort_key': 'UMI_counts', 'sort_desc': True}))
    for tool in ['cellranger', 'ham', 'simpleaf_k15']:
        for umi in [0, 3]:
            f = os.path.join(base, f'06_pgmm_crispat/{tool}/UMI_{umi}/assignments.csv')
            if os.path.exists(f): specs.append((f'crispat_pgmm_umi{umi}', tool, 'standard', {'fpath': f, 'sort_key': 'UMI_counts', 'sort_desc': True}))
    for dir_name, tool_label in [('cellranger_sparse', 'cellranger'), ('ham', 'ham'), ('simpleaf_k15', 'simpleaf_k15')]:
        f = os.path.join(base, f'07_2beta_crispat/{dir_name}/assignments.csv')
        if os.path.exists(f): specs.append(('crispat_2beta', tool_label, 'crispat_2beta', {'fpath': f}))
    for tool in ['cellranger', 'ham', 'simpleaf_k15']:
        for t in [3, 5, 10]:
            f = os.path.join(base, f'08_umi_crispat/{tool}/t{t}/assignments.csv')
            if os.path.exists(f): specs.append((f'umi_threshold_t{t}', tool, 'crispat_umi', {'fpath': f}))
    for tool in ['cellranger', 'ham', 'simpleaf_k15']:
        f = os.path.join(base, f'09_fishash/{tool}/assignments.csv')
        if os.path.exists(f):
            specs.append(('fishash', tool, 'fishash_topk', {'fpath': f}))
    return specs

def build_specs_papalexi():
    specs = []; base = os.path.join(STARTER_BASE, '11_papalexi_benchmark', '02_results')
    for tool in ['ham', 'simpleaf']:
        f = os.path.join(base, f'pgmm_em/{tool}/assignments.csv')
        if os.path.exists(f): specs.append(('pgmm_em', tool, 'standard', {'fpath': f, 'sort_key': 'UMI_counts', 'sort_desc': True}))
    for tool in ['ham', 'simpleaf']:
        for umi in [0, 3]:
            f = os.path.join(base, f'crispat_pgmm/UMI_{umi}/{tool}/assignments.csv')
            if os.path.exists(f): specs.append((f'crispat_pgmm_umi{umi}', tool, 'standard', {'fpath': f, 'sort_key': 'UMI_counts', 'sort_desc': True}))
    for tool in ['ham', 'simpleaf']:
        f = os.path.join(base, f'crispat_2beta/{tool}/assignments.csv')
        if os.path.exists(f): specs.append(('crispat_2beta', tool, 'crispat_2beta', {'fpath': f}))
    for tool in ['ham', 'simpleaf']:
        for t in [3, 5, 10]:
            f = os.path.join(base, f'crispat_umi/{tool}/t{t}/assignments.csv')
            if os.path.exists(f): specs.append((f'umi_threshold_t{t}', tool, 'crispat_umi', {'fpath': f}))
    for tool in ['ham', 'simpleaf']:
        f = os.path.join(base, f'fishash/{tool}/assignments.csv')
        if os.path.exists(f):
            specs.append(('fishash', tool, 'fishash_topk', {'fpath': f}))
    return specs


# ══════════════════════════════════════════════════════════════════════════
# GEX + guide mapping loaders
# ══════════════════════════════════════════════════════════════════════════

def load_gex_k562(h5ad_path, sg2gene, nt_sgrnas):
    """Return (X_csr, cell_lookup, col_gene_names).
    Nextera h5ad (anndata sparse + ENSG var_names). Builds symbol→ENSG→col mapping.
    cell_lookup: (lane_int, 16mer) → row index.
    """
    import anndata as ad
    print("  Loading Replogle 2022 GEX (Nextera, selective columns) …", end=' ', flush=True)
    t0 = time.time()

    # Determine which gene symbols we need
    needed_genes = set()
    for sg, gene in sg2gene.items():
        if sg not in nt_sgrnas:
            needed_genes.add(gene)

    # Build symbol→ENSG mapping (cached)
    import json as _json
    _map_json = os.path.join(STARTER_BASE, "benchmark_output/_gene_symbol_to_ensg.json")
    with open(_map_json) as _f:
        _sym2ensg = _json.load(_f)

    sc = ad.read_h5ad(h5ad_path)
    # Build ENSG→col index from var_names (ENSG00000197530_S → strip _S)
    ensg_to_col = {}
    for i, v in enumerate(sc.var_names):
        s = v.decode('utf-8') if isinstance(v, bytes) else str(v)
        ensg_to_col[s.replace('_S', '')] = i

    # Find columns for needed gene symbols
    col_gene_names = []  # gene symbols, order = column order in X_subset
    needed_cols = []
    for sym in sorted(needed_genes):
        ensg = _sym2ensg.get(sym)
        if ensg is not None and ensg in ensg_to_col:
            needed_cols.append(ensg_to_col[ensg])
            col_gene_names.append(sym)

    X_subset = sc.X[:, needed_cols].tocsr()  # CSR for row access
    ncells = sc.shape[0]

    # Build cell_lookup from barcode_16mer + lane (Nextera format)
    cell_lookup = {}
    for i in range(ncells):
        seq = str(sc.obs.barcode_16mer.iloc[i])
        lane = int(sc.obs.lane.iloc[i])
        cell_lookup[(lane, seq)] = i
    del sc

    print(f"{len(cell_lookup):,} cells × {len(col_gene_names)} genes  [{time.time()-t0:.1f}s]")
    return X_subset, cell_lookup, col_gene_names

def load_gex_papalexi(h5ad_path):
    """Return (gex_csr, cell_lookup, gene_list).
    cell_lookup: 16mer → row index (Papalexi is single-lane).
    """
    import anndata as ad
    print("  Loading Papalexi GEX …", end=' ', flush=True)
    t0 = time.time()
    adata = ad.read_h5ad(h5ad_path)
    cell_lookup = {}
    for i, bc in enumerate(adata.obs_names):
        m = BC_GT.match(bc)  # l1_16mer
        if m:
            cell_lookup[m.group(2)] = i
    print(f"{len(cell_lookup):,} cells  [{time.time()-t0:.1f}s]")
    return adata.X.tocsr(), cell_lookup, adata.var_names.tolist()

def load_guide_map_k562(csv_path):
    """Return (sg2gene, gene_set, nt_sgrnas).
    sg2gene: sgID → gene_symbol (string, for mapping)
    nt_sgrnas: set of all non-targeting sgRNAs
    """
    sg2gene = {}; nt_sgrnas = set()
    with open(csv_path) as f:
        f.readline()
        for line in f:
            p = line.strip().split(',')
            if len(p) < 8: continue
            gene = p[1]; sgA = p[4]; sgB = p[6]
            if gene == 'non-targeting':
                nt_sgrnas.add(sgA); nt_sgrnas.add(sgB)
            sg2gene[sgA] = gene; sg2gene[sgB] = gene
    print(f"  Guide map: {len(sg2gene):,} sgRNAs → {len(set(sg2gene.values())):,} genes, {len(nt_sgrnas):,} NT sgRNAs")
    return sg2gene, nt_sgrnas

def load_guide_map_papalexi(csv_path):
    """Return (sg2gene, gene_set, nt_sgrnas).
    sg2gene: guide_ID → gene_target
    nt_sgrnas: guides starting with 'NT' or where perturbation='NT'
    """
    sg2gene = {}; nt_sgrnas = set()
    with open(csv_path) as f:
        f.readline()
        for line in f:
            p = line.strip().split(',')
            if len(p) < 6: continue
            gid = p[1]; gene = p[2]; pert = p[3]; nt_val = p[4]
            sg2gene[gid] = gene
            if pert == 'NT' or nt_val.startswith('NT'):
                nt_sgrnas.add(gid)
    print(f"  Guide map: {len(sg2gene):,} guides → {len(set(sg2gene.values())):,} genes, {len(nt_sgrnas):,} NT guides")
    return sg2gene, nt_sgrnas


# ══════════════════════════════════════════════════════════════════════════
# Core KD computation
# ══════════════════════════════════════════════════════════════════════════

def compute_kd_metrics(pgmm, gex_csr, cell_lookup, gene_list,
                       sg2gene, nt_sgrnas, perturbation_type='knockdown',
                       min_cells=5, pseudocount_frac=0.01):
    """
    Compute per-guide log2FC of target gene expression in assigned cells.

    For each guide g:
      - C_g = cells with top-1 assignment = g
      - If |C_g| < min_cells → skip
      - If g is NT → accumulate NT expression for baseline
      - If g maps to target gene T:
        - expr_Cg = mean(GEX[C_g, T])
        - expr_NT = mean(GEX[NT_cells, T])
        - log2FC = log2(expr_Cg + ε) - log2(expr_NT + ε)
        - ε = pseudocount_frac * global_mean_expr

    Returns dict with per-guide and aggregate metrics.
    """
    # NOTE: v2 uses per-gene pseudocount.
    # ε_g = pseudocount_frac × mean_expression of gene g across all loaded cells.
    # This replaces the global scalar ε, preventing false weak-KD artifacts
    # for low-expression genes whose magnitude is comparable to global ε.

    # Pre-compute per-gene mean expression for per-gene ε
    if hasattr(gex_csr, 'toarray'):  # sparse CSR
        gene_means = np.array(gex_csr.mean(axis=0)).flatten()
    else:
        gene_means = gex_csr.mean(axis=0).flatten()

    # Collect NT cell indices and guide indices
    nt_cell_indices = set()
    guide_to_cells = defaultdict(set)
    guide_to_target_gene = {}

    for key in pgmm:
        if not pgmm[key]: continue
        top_guide = pgmm[key][0][0]
        lane, seq16 = key
        cell_idx = cell_lookup.get((lane, seq16))
        if cell_idx is None:
            # Try lane=0 (Papalexi single-lane)
            if lane == 1:
                cell_idx = cell_lookup.get(seq16)
        if cell_idx is None: continue

        guide_to_cells[top_guide].add(cell_idx)
        if top_guide in nt_sgrnas:
            nt_cell_indices.add(cell_idx)
        else:
            target_gene = sg2gene.get(top_guide)
            if target_gene:
                guide_to_target_gene[top_guide] = target_gene

    # Gene index lookup
    gene_to_idx = {g: i for i, g in enumerate(gene_list)}

    # Compute per-guide KD
    nt_cells_list = sorted(nt_cell_indices)
    per_guide = []
    per_gene_grp = defaultdict(list)

    if len(nt_cells_list) < min_cells:
        print(f"    WARNING: only {len(nt_cells_list)} NT cells — KD computation may be unreliable")

    for guide, cells in guide_to_cells.items():
        target_gene = guide_to_target_gene.get(guide)
        if target_gene is None: continue
        g_idx = gene_to_idx.get(target_gene)
        if g_idx is None: continue

        cells_list = sorted(cells)
        if len(cells_list) < min_cells: continue

        # Mean expression in assigned cells
        if hasattr(gex_csr, 'toarray'):  # sparse CSR
            expr_assigned = np.array(gex_csr[cells_list, g_idx].toarray()).flatten()
        else:  # dense numpy
            expr_assigned = gex_csr[cells_list, g_idx].flatten()
        mean_assigned = expr_assigned.mean()

        # Mean expression in NT cells for this gene
        if hasattr(gex_csr, 'toarray'):
            expr_nt = np.array(gex_csr[nt_cells_list, g_idx].toarray()).flatten()
        else:
            expr_nt = gex_csr[nt_cells_list, g_idx].flatten()
        mean_nt = expr_nt.mean()

        eps_g = pseudocount_frac * max(float(gene_means[g_idx]), 1e-8)

        if mean_nt < eps_g and mean_assigned < eps_g: continue

        log2fc = np.log2(max(mean_assigned + eps_g, eps_g)) - np.log2(max(mean_nt + eps_g, eps_g))

        per_guide.append({
            'guide': guide,
            'target_gene': target_gene,
            'n_cells': len(cells_list),
            'mean_expr_assigned': float(mean_assigned),
            'mean_expr_nt': float(mean_nt),
            'log2fc': round(float(log2fc), 6),
        })
        per_gene_grp[target_gene].append(log2fc)

    if not per_guide:
        return None

    log2fcs = np.array([g['log2fc'] for g in per_guide])

    # ── NT-NT Baseline ────────────────────────────────────────────────
    # For NT guides, compute KD against random non-NT genes — should be ≈0
    nt_nt_results = _compute_nt_nt_baseline(
        guide_to_cells, nt_cell_indices, gene_to_idx,
        gene_list, gex_csr, gene_means, pseudocount_frac,
        min_cells, sg2gene, nt_sgrnas)

    # (pair consistency is computed later in main() using the guide CSV)

    # Direction-aware metrics
    if perturbation_type == 'knockdown':
        frac_expected = float((log2fcs < 0).mean())
        frac_strong   = float((log2fcs < -0.5).mean())
    elif perturbation_type == 'activation':
        frac_expected = float((log2fcs > 0).mean())
        frac_strong   = float((log2fcs > 0.5).mean())
    else:
        frac_expected = float((np.abs(log2fcs) > 0.5).mean())
        frac_strong   = float((np.abs(log2fcs) > 1.0).mean())

    # Per-gene summary
    gene_summary = {}
    for gene, vals in per_gene_grp.items():
        vals_a = np.array(vals)
        gene_summary[gene] = {
            'n_guides': len(vals),
            'kd_median': round(float(np.median(vals_a)), 4),
            'kd_mean': round(float(vals_a.mean()), 4),
        }

    return {
        'n_guides_tested': len(per_guide),
        'n_nt_cells': len(nt_cells_list),
        'n_target_genes': len(gene_summary),
        'kd_efficiency_median': round(float(np.median(log2fcs)), 4),
        'kd_efficiency_mean': round(float(log2fcs.mean()), 4),
        'kd_efficiency_std': round(float(log2fcs.std()), 4),
        'fraction_expected_direction': round(frac_expected, 4),
        'fraction_strong_perturbation': round(frac_strong, 4),
        'frac_log2fc_lt_neg1': round(float((log2fcs < -1.0).mean()), 4),
        'frac_log2fc_lt_neg2': round(float((log2fcs < -2.0).mean()), 4),
        'per_guide': per_guide,
        'per_gene_summary': gene_summary,
        'perturbation_type': perturbation_type,
        'nt_nt_baseline': nt_nt_results,
    }


def _compute_nt_nt_baseline(guide_to_cells, nt_cell_indices, gene_to_idx,
                            gene_list, gex_csr, gene_means, pseudocount_frac,
                            min_cells, sg2gene, nt_sgrnas):
    """NT guide KD against random non-target genes. Should return ~0."""
    import random
    random.seed(42)

    nt_guides = [g for g in guide_to_cells if g in nt_sgrnas]
    if len(nt_guides) < 1: return None

    # Get all non-NT genes we can test against
    non_nt_genes = sorted(set(
        sg2gene.get(g) for g in guide_to_cells
        if g not in nt_sgrnas and sg2gene.get(g)
    ))

    if len(non_nt_genes) < 5: return None

    nt_cells_list = sorted(nt_cell_indices)
    k_genes = min(20, len(non_nt_genes))
    nt_nt_kds = []

    for g_nt in nt_guides:
        cells = guide_to_cells[g_nt]
        if len(cells) < min_cells: continue
        cells_list = sorted(cells)
        # Random sample of non-NT genes
        sampled = random.sample(non_nt_genes, k_genes)
        for T_rand in sampled:
            g_idx = gene_to_idx.get(T_rand)
            if g_idx is None: continue
            if hasattr(gex_csr, 'toarray'):
                expr_cells = np.array(gex_csr[cells_list, g_idx].toarray()).flatten()
                expr_nt = np.array(gex_csr[nt_cells_list, g_idx].toarray()).flatten()
            else:
                expr_cells = gex_csr[cells_list, g_idx].flatten()
                expr_nt = gex_csr[nt_cells_list, g_idx].flatten()
            mean_c = expr_cells.mean(); mean_n = expr_nt.mean()
            eps_g = pseudocount_frac * max(float(gene_means[g_idx]), 1e-8)
            if mean_n < eps_g and mean_c < eps_g: continue
            lfc = np.log2(max(mean_c+eps_g,eps_g)) - np.log2(max(mean_n+eps_g,eps_g))
            nt_nt_kds.append(float(lfc))

    if not nt_nt_kds: return None
    arr = np.array(nt_nt_kds)
    return {
        'kd_median': round(float(np.median(arr)), 4),
        'kd_mean': round(float(arr.mean()), 4),
        'kd_std': round(float(arr.std()), 4),
        'kd_iqr': round(float(np.percentile(arr,75)-np.percentile(arr,25)), 4),
        'n_tests': len(nt_nt_kds),
        'kd_list': [float(x) for x in nt_nt_kds],  # raw values for downstream plotting
    }


def _compute_pair_consistency_from_per_guide(per_guide, guide_csv_path):
    """From per-guide KD data + original guide CSV, compare sgA/sgB KD per pair."""
    # Build pair mapping from CSV: sgID → (pair_id, role='A'|'B')
    sg_to_pair = {}
    pair_to_info = {}
    with open(guide_csv_path) as f:
        f.readline()
        for line in f:
            p = line.strip().split(',')
            if len(p) < 8: continue
            pid, gene, sgA, sgB = p[0], p[1], p[4], p[6]
            if gene == 'non-targeting': continue
            sg_to_pair[sgA] = (pid, 'A')
            sg_to_pair[sgB] = (pid, 'B')
            pair_to_info[pid] = {'sgA': sgA, 'sgB': sgB, 'gene': gene}

    # Build guide→KD lookup
    guide_kd = {}
    for g in per_guide:
        guide_kd[g['guide']] = g['log2fc']

    # Find pairs where both guides have KD
    deltas = []
    discordant = []
    for pid, info in pair_to_info.items():
        sgA, sgB = info['sgA'], info['sgB']
        if sgA in guide_kd and sgB in guide_kd:
            d = abs(guide_kd[sgA] - guide_kd[sgB])
            deltas.append(d)
            if d > 0.5:
                discordant.append({
                    'pair_id': pid, 'gene': info['gene'],
                    'sgA': sgA, 'kd_A': guide_kd[sgA],
                    'sgB': sgB, 'kd_B': guide_kd[sgB],
                    'delta': round(d, 4),
                })

    if not deltas: return None
    arr = np.array(deltas)
    return {
        'n_pairs_tested': len(deltas),
        'concordance_rate': round(float((arr <= 0.5).mean()), 4),
        'delta_median': round(float(np.median(arr)), 4),
        'delta_mean': round(float(arr.mean()), 4),
        'delta_std': round(float(arr.std()), 4),
        'n_discordant': len(discordant),
        'delta_list': [float(x) for x in deltas],  # raw deltas for density plot
        'discordant_pairs': discordant[:20],  # top 20
    }


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='KD Efficiency Benchmark')
    parser.add_argument('--dataset', required=True, choices=['replogle2022','papalexi2021'])
    parser.add_argument('--method', default=None, help='Specific method name')
    parser.add_argument('--tool', default=None, help='Specific tool (with --method)')
    parser.add_argument('--list', action='store_true', help='List all method-tool specs')
    parser.add_argument('--min-cells', type=int, default=5, help='Min cells per guide for KD calc')
    parser.add_argument('--list-specs', action='store_true', help='Same as --list')
    args = parser.parse_args()
    if args.list_specs: args.list = True

    cfg = DATASET_CONFIGS[args.dataset]

    # Build specs
    if cfg['spec_key'] == 'k562':
        all_specs = build_specs_k562()
    else:
        all_specs = build_specs_papalexi()

    if args.list:
        print(f"{'Method':<28s} {'Tool':<15s} {'Loader':<14s} {'kwargs'}")
        print("-"*90)
        for m, t, lf, lk in all_specs:
            print(f"{m:<28s} {t:<15s} {lf:<14s} {str(lk)[:60]}")
        return

    # Filter
    if args.method:
        tool_filt = args.tool
        all_specs = [(m,t,lf,lk) for m,t,lf,lk in all_specs
                     if m == args.method and (t == tool_filt if tool_filt else True)]
    if not all_specs:
        print("No specs matched.")
        return

    perturbation_type = cfg['perturbation_type']

    # Load guide map first (K562 needs it for selective GEX column loading)
    print(f"\n{'='*70}")
    print(f"KD Efficiency Benchmark — {args.dataset}")
    print(f"Perturbation type: {perturbation_type}")
    print(f"{'='*70}\n")

    if cfg['spec_key'] == 'k562':
        sg2gene, nt_sgrnas = load_guide_map_k562(cfg['guide_csv'])
        X_dense, cell_lookup, gene_list = load_gex_k562(cfg['gex_h5ad'], sg2gene, nt_sgrnas)
    else:
        sg2gene, nt_sgrnas = load_guide_map_papalexi(cfg['guide_map_csv'])
        X_dense, cell_lookup, gene_list = load_gex_papalexi(cfg['gex_h5ad'])

    out_dir = os.path.join(STARTER_BASE, cfg.get('out_dir', f'12_kd_efficiency/{args.dataset}'))
    os.makedirs(out_dir, exist_ok=True)
    all_results = []

    for method, tool, loader_name, lk in all_specs:
        label = f"{method}__{tool}"
        out_json = os.path.join(out_dir, f"{label}.json")

        print(f"\n  {label}")
        print(f"  Loader: {loader_name}  kwargs: {{k: lk.get(k) for k in ['sort_key','sort_desc','fpath'] if k in lk}}")

        t0 = time.time()

        # Load assignment
        pgmm = LOADERS[loader_name](**lk)
        n_cells = len(pgmm)
        n_assigns = sum(len(v) for v in pgmm.values())
        print(f"    {n_assigns:,} rows  {n_cells:,} cells  [{time.time()-t0:.1f}s]")

        # Compute KD
        t1 = time.time()
        metrics = compute_kd_metrics(
            pgmm, X_dense, cell_lookup, gene_list,
            sg2gene, nt_sgrnas,
            perturbation_type=perturbation_type,
            min_cells=args.min_cells,
        )

        if metrics is None:
            print(f"    SKIP — no valid guides (all NT or <{args.min_cells} cells)")
            continue

        # ── Pair consistency (Replogle 2022 only) ────────────────────
        if cfg['spec_key'] == 'k562' and 'per_guide' in metrics:
            pair_info = _compute_pair_consistency_from_per_guide(
                metrics['per_guide'], cfg['guide_csv'])
            if pair_info:
                metrics['pair_consistency'] = pair_info
            else:
                metrics['pair_consistency'] = None

        kd_wall = time.time() - t1
        wall = time.time() - t0

        # Combine without per-guide list (too large for summary)
        summary = {
            'method': method, 'tool': tool,
            'dataset': args.dataset,
            'wall_s': round(wall, 1),
            'kd_wall_s': round(kd_wall, 1),
            'n_assignment_cells': n_cells,
            'n_assignment_rows': n_assigns,
            **{k: metrics[k] for k in metrics if k not in ('per_guide','per_gene_summary')},
        }
        # Keep per_gene_summary (compact)
        summary['per_gene_summary'] = metrics['per_gene_summary']

        with open(out_json, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        all_results.append(summary)

        print(f"    KD median: {metrics['kd_efficiency_median']:.4f}  "
              f"expected_dir: {metrics['fraction_expected_direction']:.3f}  "
              f"strong: {metrics['fraction_strong_perturbation']:.3f}  "
              f"guides: {metrics['n_guides_tested']}  [{wall:.1f}s]")

    # ── Cross-method summary ──
    if len(all_results) > 1:
        summary_json = os.path.join(out_dir, '_kd_summary.json')
        with open(summary_json, 'w') as f:
            json.dump(all_results, f, indent=2)

        # Quick table: HAM tools only for readability
        priority_methods = ['pgmm_em','crispat_pgmm_umi0','crispat_pgmm_umi3',
                            'crispat_2beta','umi_threshold_t3','umi_threshold_t5',
                            'umi_threshold_t10','fishash']
        print(f"\n{'='*80}")
        print(f"KD Efficiency Summary — {args.dataset}")
        print(f"{'='*80}")
        print(f"{'Method':<28s} {'Tool':<14s} {'KD med':>8s} {'KD mean':>8s} {'FracExp':>8s} {'FracStrong':>10s} {'Guides':>8s}")
        print("-"*85)
        for r in all_results:
            if r['tool'] in ('ham', 'simpleaf_k15'):
                print(f"{r['method']:<28s} {r['tool']:<14s} "
                      f"{r['kd_efficiency_median']:8.4f} {r['kd_efficiency_mean']:8.4f} "
                      f"{r['fraction_expected_direction']:8.4f} {r['fraction_strong_perturbation']:10.4f} "
                      f"{r['n_guides_tested']:>8d}")
        print(f"\nSaved: {summary_json}")

    print(f"\nDone — {len(all_results)} KD profiles written to {out_dir}/\n")


if __name__ == '__main__':
    main()
