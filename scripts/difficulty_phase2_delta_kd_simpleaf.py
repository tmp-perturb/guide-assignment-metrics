#!/usr/bin/env python3
"""
Phase 2 — delta–KD validation (simpleaf, umi_t3).
Full NT negative control + within-guide Spearman.  Same protocol as HAM version.

Fixed 2026-08-07: Added NT control and kd_std (were missing from previous version).
"""
import csv, json, os, re, time, numpy as np
from collections import defaultdict
from scipy.stats import spearmanr
import anndata as ad

ST = "/data/yunzliu/assignment_benchmark_starter"
OUT = os.path.join(ST, "benchmark_output/difficulty_stratification")
TRUSEQ = "/data/yunzliu/Replogle2022_K562_Day6_benchmark/02_gex/post_decontx/TruSeq_decontx_merged_48lanes.h5ad"
GUIDE_CSV = "/data/yunzliu/references/raw_guides_k562_essential.csv"
BC = re.compile(r'^([ACGT]{16})-L(\d+)$')
EPS = 0.01
DECILES = [0,10,20,30,40,50,60,70,80,90,100]
ts = time.time()
print("Phase 2 — simpleaf delta-KD")
print("=" * 50)

# ═══ 1. Difficulty table ═══
print("[1/6] Difficulty table …")
tsv = os.path.join(OUT, "cell_difficulty_simpleaf.tsv")
delta_per_cell = {}
with open(tsv) as f:
    for row in csv.DictReader(f, delimiter='\t'):
        c = row['cell_id']; m = BC.match(c)
        if m: delta_per_cell[(int(m.group(2)), m.group(1))] = float(row['delta'])
print(f"  {len(delta_per_cell):,} cells [{time.time()-ts:.0f}s]")

# ═══ 2. Guide → gene mapping + NT list ═══
print("[2/6] Guide map …")
sg2gene = {}; nt_sgrnas = set()
with open(GUIDE_CSV) as gf:
    for line in csv.reader(gf):
        if line[0].startswith('unique'): continue  # header
        if len(line) < 8: continue
        gene, sgA, sgB = line[1], line[4], line[6]
        if gene == 'non-targeting':
            if sgA: nt_sgrnas.add(sgA)
            if sgB: nt_sgrnas.add(sgB)
        sg2gene[sgA] = gene; sg2gene[sgB] = gene
print(f"  {len(sg2gene):,} sgIDs, {len(nt_sgrnas)} NT [{time.time()-ts:.0f}s]")

# ═══ 3. Load assignment (umi_t3 simpleaf) ═══
print("[3/6] Assignment (umi_t3 simpleaf) …")
top1_guide = {}
with open(f"{ST}/08_umi_crispat/simpleaf_k15/t3/assignments.csv") as f:
    for row in csv.DictReader(f):
        cell = row.get('cell','').strip(); guide = row.get('gRNA','').strip()
        if not cell or not guide: continue
        m = BC.match(cell)
        if not m: continue
        k = (int(m.group(2)), m.group(1))
        score = int(float(row.get('UMI_counts', 0) or 0))
        if k not in top1_guide or score > top1_guide[k][1]:
            top1_guide[k] = (guide, score)
for k in top1_guide: top1_guide[k] = top1_guide[k][0]
print(f"  {len(top1_guide):,} cells assigned [{time.time()-ts:.0f}s]")

# ═══ 4. GEX: column-extract needed genes ═══
print("[4/6] GEX column extraction …")

# Collect needed genes (targeting)
needed_genes = set()
guide_to_cells = defaultdict(list)
nt_cell_indices = []
for k, g in top1_guide.items():
    delta = delta_per_cell.get(k)
    if delta is None: continue
    gene = sg2gene.get(g)
    if g in nt_sgrnas:
        continue  # NT guides handled separately
    if gene: needed_genes.add(gene); guide_to_cells[(g, gene)].append((k, delta))

# Also need NT guide genes for negative control
for g_nt in nt_sgrnas:
    needed_genes.add(g_nt)

print(f"  {len(needed_genes)} genes needed [{time.time()-ts:.0f}s]")

with open(os.path.join(ST, "benchmark_output/gene_symbol_to_ensg.json")) as f: sym2ensg = json.load(f)

sc = ad.read_h5ad(TRUSEQ)
nc = sc.shape[0]

ensg2col = {}
for i, v in enumerate(sc.var_names):
    s = v.decode('utf-8') if isinstance(v, bytes) else str(v)
    ensg2col[s.replace('_S', '')] = i

gene2col = {}
for sym in sorted(needed_genes):
    es = sym2ensg.get(sym)
    if es and es in ensg2col: gene2col[sym] = ensg2col[es]
needed_cols = sorted(gene2col.values())
col2loc = {c: i for i, c in enumerate(needed_cols)}

X_small = sc.X[:, needed_cols].toarray().astype(np.float32)
print(f"  Extracted {len(needed_cols)} gene columns → {X_small.nbytes/1024**3:.1f} GB [{time.time()-ts:.0f}s]")

# Cell lookup
cell_lookup = {}
for i in range(nc):
    seq = str(sc.obs.barcode_16mer.iloc[i])
    lane = int(sc.obs.lane.iloc[i])
    cell_lookup[(lane, seq)] = i
del sc
print(f"  {len(cell_lookup):,} cells in GEX [{time.time()-ts:.0f}s]")

# NT pool
nt_set = set()
for k, g in top1_guide.items():
    if g in nt_sgrnas:
        idx = cell_lookup.get(k)
        if idx is not None: nt_set.add(idx)
nt_list = sorted(nt_set)
print(f"  NT pool: {len(nt_list):,} cells [{time.time()-ts:.0f}s]")

# Per-gene means + NT means
gene_means = {}; nt_means = {}
for sym in gene2col:
    idx = col2loc[gene2col[sym]]
    gene_means[sym] = float(X_small[:, idx].mean())
    nt_means[sym] = float(X_small[nt_list, idx].mean()) if nt_list else gene_means[sym]

# ═══ 5. Per-guide delta-stratified KD ═══
print("[5/6] Delta-stratified KD …")
all_delta_kd = []
per_gene_corr = []
nt_delta_kd = []

# Targeting guides
n_sk_cons = 0; n_sk_few = 0
for (guide, gene), cells_deltas in guide_to_cells.items():
    if len(cells_deltas) < 20: n_sk_few += 1; continue
    g_col = gene2col.get(gene)
    if g_col is None: n_sk_few += 1; continue
    g_idx = col2loc[g_col]

    cells_deltas.sort(key=lambda x: x[1])
    n = len(cells_deltas)
    deltas = np.array([d for _, d in cells_deltas])

    bounds = [np.percentile(deltas, p) for p in DECILES]
    for di in range(10):
        lo, hi = bounds[di], bounds[di+1]
        mask = (deltas >= lo) & (deltas <= hi) if di < 9 else (deltas >= lo)
        bin_cells = [(k, d) for (k, d), m in zip(cells_deltas, mask) if m]
        if len(bin_cells) < 5: continue
        exprs = []
        for k, d in bin_cells:
            idx = cell_lookup.get(k)
            if idx is None: continue
            exprs.append(X_small[idx, g_idx])
        if not exprs: continue
        mean_expr = np.mean(exprs)
        em = gene_means.get(gene, 1e-8); enm = nt_means.get(gene, em)
        epsg = EPS * max(em, 1e-8)
        kd = float(np.log2(max(mean_expr + epsg, epsg)) - np.log2(max(enm + epsg, epsg)))
        all_delta_kd.append((di, kd))

    # Within-guide rank correlation
    cell_kds = []
    for k, d in cells_deltas:
        idx = cell_lookup.get(k)
        if idx is None: continue
        expr = float(X_small[idx, g_idx])
        em = gene_means.get(gene, 1e-8); enm = nt_means.get(gene, em)
        epsg = EPS * max(em, 1e-8)
        kd_v = np.log2(max(expr + epsg, epsg)) - np.log2(max(enm + epsg, epsg))
        cell_kds.append((d, kd_v))
    if len(cell_kds) >= 10:
        d_rank = np.array([d for d, _ in cell_kds]); kd_arr = np.array([v for _, v in cell_kds])
        if np.std(d_rank) < 1e-12 or np.std(kd_arr) < 1e-12: n_sk_cons += 1; continue
        r, p = spearmanr(d_rank, kd_arr)
        if not np.isnan(r): per_gene_corr.append((r, p))
        else: n_sk_cons += 1

print(f"  {len(all_delta_kd)} delta-KD pairs, {len(per_gene_corr)} genes with within-guide ρ [{time.time()-ts:.0f}s]")

# ═══ 6. NT negative control ═══
print("[6/6] NT negative control …")
non_nt_genes = sorted(set(sg2gene.values()) - {'non-targeting'})
rng = np.random.RandomState(42)

# Get NT cells by guide from assignment
nt_guide_cells = defaultdict(list)
for k, g in top1_guide.items():
    if g in nt_sgrnas:
        delta = delta_per_cell.get(k)
        if delta is not None:
            nt_guide_cells[g].append((k, delta))

for g_nt, cells_deltas in nt_guide_cells.items():
    if len(cells_deltas) < 10: continue
    cells_deltas.sort(key=lambda x: x[1])
    deltas = np.array([d for _, d in cells_deltas])
    if np.max(deltas) == np.min(deltas): continue
    bounds = [np.percentile(deltas, p) for p in DECILES]
    t_rand = rng.choice(non_nt_genes)
    g_col = gene2col.get(t_rand)
    if g_col is None: continue
    g_idx = col2loc[g_col]
    for di in range(10):
        lo, hi = bounds[di], bounds[di+1]
        mask = (deltas >= lo) & (deltas <= hi) if di < 9 else (deltas >= lo)
        bin_cells = [(k, d) for (k, d), m in zip(cells_deltas, mask) if m]
        if len(bin_cells) < 5: continue
        exprs = []
        for k, d in bin_cells:
            idx = cell_lookup.get(k)
            if idx is None: continue
            exprs.append(X_small[idx, g_idx])
        if not exprs: continue
        mean_expr = np.mean(exprs)
        em = gene_means.get(t_rand, 1e-8); enm = nt_means.get(t_rand, em)
        epsg = EPS * max(em, 1e-8)
        kd = float(np.log2(max(mean_expr + epsg, epsg)) - np.log2(max(enm + epsg, epsg)))
        nt_delta_kd.append((di, kd))

# ═══ Aggregate + save ═══
print(f"\nAggregating …")
target_by_decile = defaultdict(list)
nt_by_decile = defaultdict(list)
for di, kd in all_delta_kd: target_by_decile[di].append(kd)
for di, kd in nt_delta_kd: nt_by_decile[di].append(kd)

results = {"target": {}, "nt_control": {}, "within_guide_spearman": {}}
for di in range(10):
    if target_by_decile[di]:
        arr = np.array(target_by_decile[di])
        results["target"][f"D{di}"] = {
            "n": len(arr),
            "kd_median": round(float(np.median(arr)), 4),
            "kd_mean": round(float(arr.mean()), 4),
            "kd_std": round(float(arr.std()), 4),
        }
    if nt_by_decile[di]:
        arr = np.array(nt_by_decile[di])
        results["nt_control"][f"D{di}"] = {
            "n": len(arr),
            "kd_median": round(float(np.median(arr)), 4),
            "kd_mean": round(float(arr.mean()), 4),
            "kd_std": round(float(arr.std()), 4),
        }

# Spearman summary
if per_gene_corr:
    rs = [r for r, p in per_gene_corr]
    ps = [p for r, p in per_gene_corr]
    results["within_guide_spearman"] = {
        "n_genes_tested": len(guide_to_cells),
        "n_valid": len(rs),
        "n_skipped_constant": n_sk_cons,
        "n_skipped_few_cells": n_sk_few,
        "r_median": round(float(np.median(rs)), 4),
        "r_mean": round(float(np.mean(rs)), 4),
        "frac_p_lt_005": round(float((np.array(ps) < 0.05).mean()), 4),
    }

out_path = os.path.join(OUT, "phase2_delta_kd_simpleaf.json")
with open(out_path, 'w') as f: json.dump(results, f, indent=2)
print(f"→ {out_path}")

# Print summary
print(f"\nDelta-KD (targeting):")
print(f"{'Decile':<10s} {'n':>8s} {'KD median':>10s} {'KD mean':>10s} {'KD std':>10s}")
for di in range(10):
    if f"D{di}" in results["target"]:
        r = results["target"][f"D{di}"]
        print(f"  D{str(di):<7s} {r['n']:>8,} {r['kd_median']:10.4f} {r['kd_mean']:10.4f} {r['kd_std']:10.4f}")

print(f"\nDelta-KD (NT control):")
print(f"{'Decile':<10s} {'n':>8s} {'KD median':>10s} {'KD mean':>10s}")
for di in range(10):
    if f"D{di}" in results["nt_control"]:
        r = results["nt_control"][f"D{di}"]
        print(f"  D{di:<7s} {r['n']:>8,} {r['kd_median']:10.4f} {r['kd_mean']:10.4f}")

if per_gene_corr:
    print(f"\nWithin-guide Spearman: median r={results['within_guide_spearman']['r_median']:.4f}, "
          f"P<0.05 fraction={results['within_guide_spearman']['frac_p_lt_005']:.4f}")

print(f"\nDone [{time.time()-ts:.0f}s]")
