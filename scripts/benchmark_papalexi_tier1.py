#!/usr/bin/env python3
"""Papalexi 2021 Tier-1 benchmark — single-guide, no construct mapping."""
import csv, json, os, re, sys, time
from collections import Counter
import numpy as np

STARTER = "/data/yunzliu/assignment_benchmark_starter"
GT_CSV = "/data/yunzliu/papalexi_2021_benchmark/01_reference/papalexi_2021_assignment.csv"
OUT_DIR = os.path.join(STARTER, "11_papalexi_benchmark/02_results/benchmark/tier1")
os.makedirs(OUT_DIR, exist_ok=True)

BC_GT = re.compile(r'^l(\d+)_([ACGT]{16})$')
BC_AS = re.compile(r'^([ACGT]{16})-L(\d+)$')

# ═══ GT load ═══
# POOL COLLAPSE (see DATA_INDEX.md / handoff): Papalexi is 8 sub-pools (l1-l8); GT keeps
# pool identity (l{N}_16mer, 20,729 rows), but the extraction collapsed all pools into a
# single `-L01` lane, so the assignment CSVs carry only 20,441 unique bare 16mers.
# Matching MUST be on the bare 16mer -> we map every pool to (1, 16mer) so it aligns with
# the assignment loader's `-L01` -> (1, 16mer). 285 barcodes recur across pools with
# different guides; this is an irreducible ambiguity of the collapse -> last-pool-wins
# (matches published `03_benchmark/_papalexi_summary.json`, n_gt=20,441, rec=0.985).
# An earlier version keyed on (pool, 16mer), intersecting only pool l1 -> rec=0.105 (bug).
gt_label = {}   # (1, 16mer) → guide_ID   (pool collapsed)
gt_gene  = {}   # (1, 16mer) → target_gene
nt_guides = set()
with open(GT_CSV) as f:
    for row in csv.DictReader(f):
        idx = row.get('index','').strip()
        guide = row.get('guide_ID','').strip()
        gene = row.get('gene_target','').strip()
        m = BC_GT.match(idx)
        if not m: continue
        key = (1, m.group(2))     # collapse all pools -> bare 16mer; last-pool-wins
        gt_label[key] = guide
        gt_gene[key] = gene
        if row.get('NT','').startswith('NT'):
            nt_guides.add(guide)

n_gt = len(gt_label)
print(f"GT: {n_gt:,} cells, {len(nt_guides)} NT guides")

# ═══ Method specs ═══
METHODS = {
    "pgmm_em":       ("standard",   {"fpath": f"{STARTER}/11_papalexi_benchmark/02_results/pgmm_em/{{tool}}/assignments.csv",       "sort_key": "UMI_counts", "sort_desc": True}),
    "umi_t3":        ("standard",   {"fpath": f"{STARTER}/11_papalexi_benchmark/02_results/crispat_umi/{{tool}}/t3/assignments.csv","sort_key": "UMI_counts", "sort_desc": True}),
    "umi_t5":        ("standard",   {"fpath": f"{STARTER}/11_papalexi_benchmark/02_results/crispat_umi/{{tool}}/t5/assignments.csv","sort_key": "UMI_counts", "sort_desc": True}),
    "umi_t10":       ("standard",   {"fpath": f"{STARTER}/11_papalexi_benchmark/02_results/crispat_umi/{{tool}}/t10/assignments.csv","sort_key": "UMI_counts","sort_desc": True}),
    "crispat_pgmm":  ("standard",   {"fpath": f"{STARTER}/11_papalexi_benchmark/02_results/crispat_pgmm/UMI_0/{{tool}}/assignments.csv", "sort_key": "UMI_counts", "sort_desc": True}),
    "crispat_2beta": ("crispat_2beta", {"fpath": f"{STARTER}/11_papalexi_benchmark/02_results/crispat_2beta/{{tool}}/assignments.csv"}),
    "fishash":       ("fishash_topk",  {"fpath": f"{STARTER}/11_papalexi_benchmark/02_results/fishash/{{tool}}/assignments.csv"}),
}

# ═══ Loaders ═══
sys.path.insert(0, os.path.join(STARTER, "03_scripts"))
from benchmark_assignments import load_standard, load_crispat_2beta, load_crispat_umi, load_fishash_topk
LOADERS = {"standard": load_standard, "crispat_2beta": load_crispat_2beta,
           "crispat_umi": load_crispat_umi, "fishash_topk": load_fishash_topk}

ts = time.time()

for tool, tl in [("ham","ham"),("simpleaf","simpleaf")]:
    print(f"\n{'='*50}\n  Papalexi 2021 — {tl}\n{'='*50}")
    for method_name, (loader_name, kw) in METHODS.items():
        t0 = time.time()
        csv_path = kw["fpath"].format(tool=tool)
        if not os.path.exists(csv_path):
            print(f"  {method_name:16s}: SKIP ({csv_path})")
            continue

        # Load assignment
        loader_fn = LOADERS[loader_name]
        resolved_path = csv_path  # already has {tool} substituted above
        if loader_name == 'crispat_2beta':
            pgmm = load_crispat_2beta(fpath=resolved_path)
        elif loader_name == 'fishash_topk':
            pgmm = load_fishash_topk(fpath=resolved_path)
        else:
            sort_key = kw.get('sort_key', 'UMI_counts')
            sort_desc = kw.get('sort_desc', True)
            pgmm = load_standard(fpath=resolved_path, sort_key=sort_key, sort_desc=sort_desc)

        # Per-cell top-1
        top1 = {}
        for key in pgmm:
            if not pgmm[key]: continue
            top1[key] = pgmm[key][0][0]

        # Compute metrics
        shared = set(top1.keys()) & set(gt_label.keys())
        n_shared = len(shared)
        n_rec = n_shared

        n_t1 = 0
        ari_pairs = []
        per_gt = Counter()
        per_pred = Counter()
        for key in shared:
            g = top1[key]
            gt_g = gt_label[key]
            if g == gt_g:
                n_t1 += 1
            ari_pairs.append((g, gt_g))
            per_gt[gt_label[key]] += 1
            per_pred[g] += 1

        rec = n_shared / max(n_gt, 1)
        t1_acc = n_t1 / max(n_shared, 1)
        eff_t1 = n_t1 / max(n_gt, 1)

        # ARI
        from sklearn.metrics import adjusted_rand_score
        yt = []; yp = []
        for k in shared:
            yt.append(gt_label[k]); yp.append(top1[k])
        ari = adjusted_rand_score(yt, yp) if len(yt) > 1 else 0.0

        # Per-gene Pearson r
        all_genes = set(list(per_gt.keys()) + list(per_pred.keys()))
        x = [per_gt.get(g,0) for g in all_genes]
        y = [per_pred.get(g,0) for g in all_genes]
        r = float(np.corrcoef(x,y)[0,1]) if len(x) > 2 else 0.0

        gpc = [len(v) for v in pgmm.values()]

        result = {
            "method": method_name, "tool": tl, "dataset": "papalexi2021",
            "cell_recovery_rate": round(rec, 6),
            "t1_pair_accuracy": round(t1_acc, 6),
            "effective_t1": round(eff_t1, 6),
            "ari": round(ari, 6),
            "per_construct_pearson_r": round(float(r), 6),
            "n_shared_cells": n_shared,
            "cells_assigned": len(pgmm),
            "guides_per_cell_median": float(np.median(gpc)) if gpc else 0.0,
            "guides_per_cell_mean": float(np.mean(gpc)) if gpc else 0.0,
        }
        out = os.path.join(OUT_DIR, f"{method_name}__{tl}.json")
        with open(out,'w') as f: json.dump(result, f, indent=2)

        print(f"  {method_name:16s} Rec={rec:.4f} T1={t1_acc:.4f} EffT1={eff_t1:.4f} ARI={ari:.4f} r={r:.4f} [{time.time()-t0:.0f}s]")

# Summary
all_summary = []
for fn in sorted(os.listdir(OUT_DIR)):
    if fn.endswith('.json') and not fn.startswith('_'):
        with open(os.path.join(OUT_DIR, fn)) as f: all_summary.append(json.load(f))
all_summary.sort(key=lambda x: x.get('effective_t1',0), reverse=True)
with open(os.path.join(OUT_DIR, '_summary.json'), 'w') as f: json.dump(all_summary, f, indent=2)
print(f"\nDone — {len(all_summary)} profiles [{time.time()-ts:.0f}s]")
