#!/usr/bin/env python3
"""
Phase 1 B1: Construct-level set evaluation.
No GEX dependency. Reads GT h5ad obs + guide CSV + assignment CSVs.

Per shared cell:
  GT:   construct_id, {sgA, sgB}
  Method: top-1 guide, ALL guides set

Metrics:
  Pair recall distribution: P(both) / P(one) / P(zero)
  Set Precision / Recall / F1 (per cell, then mean)

Covers 7 methods × 2 extraction tools = 14 combos.
"""
import csv, json, os, sys, re, time, h5py, numpy as np
from collections import defaultdict, Counter

STARTER = "/data/yunzliu/assignment_benchmark_starter"
GUIDE_CSV = "/data/yunzliu/references/raw_guides_k562_essential.csv"
GT_H5AD = "/data/yunzliu/references/published/K562_essential_raw_singlecell_01.h5ad"
OUT_B1 = os.path.join(STARTER, "benchmark_output/_B1_construct_set_eval.json")
OUT_B1_MD = os.path.join(STARTER, "benchmark_output/_B1_construct_set_table.md")

BC_LANE = re.compile(r'^([ACGT]{16})-L(\d+)$')
BC_GT   = re.compile(r'^([ACGT]{16})-(\d+)$')

# ── Assignment CSV paths ──
ASSIGNMENT_SPECS = {
    "pgmm_em": {
        "ham": f"{STARTER}/05_pgmm_em_assignment/ham/assignments.csv",
        "simpleaf_k15": f"{STARTER}/05_pgmm_em_assignment/simpleaf_k15/assignments.csv",
    },
    "umi_t3": {
        "ham": f"{STARTER}/08_umi_crispat/ham/t3/assignments.csv",
        "simpleaf_k15": f"{STARTER}/08_umi_crispat/simpleaf_k15/t3/assignments.csv",
    },
    "umi_t5": {
        "ham": f"{STARTER}/08_umi_crispat/ham/t5/assignments.csv",
        "simpleaf_k15": f"{STARTER}/08_umi_crispat/simpleaf_k15/t5/assignments.csv",
    },
    "umi_t10": {
        "ham": f"{STARTER}/08_umi_crispat/ham/t10/assignments.csv",
        "simpleaf_k15": f"{STARTER}/08_umi_crispat/simpleaf_k15/t10/assignments.csv",
    },
    "crispat_pgmm": {
        "ham": f"{STARTER}/06_pgmm_crispat/ham/UMI_0/assignments.csv",
        "simpleaf_k15": f"{STARTER}/06_pgmm_crispat/simpleaf_k15/UMI_0/assignments.csv",
    },
    "crispat_2beta": {
        "ham": f"{STARTER}/07_2beta_crispat/ham/assignments.csv",
        "simpleaf_k15": f"{STARTER}/07_2beta_crispat/simpleaf_k15/assignments.csv",
    },
    "fishash": {
        "ham": f"{STARTER}/09_fishash/ham/assignments.csv",
        "simpleaf_k15": f"{STARTER}/09_fishash/simpleaf_k15/assignments.csv",
    },
}

# ── Sort keys (for top-1 determination) ──
SORT_CONFIG = {
    "pgmm_em":      ("UMI_counts", True),    # UMI DESC
    "umi_t3":        ("UMI_counts", True),
    "umi_t5":        ("UMI_counts", True),
    "umi_t10":       ("UMI_counts", True),
    "crispat_pgmm":  ("UMI_counts", True),
    "crispat_2beta": ("percent_counts", True),
    "fishash":       ("log_pval", False),        # ASC
}

# ── Column name for secondary sort key ──
SORT_COL = {
    "pgmm_em": "prob_gaussian",
    "umi_t3": "UMI_counts",
    "umi_t5": "UMI_counts",
    "umi_t10": "UMI_counts",
    "crispat_pgmm": "UMI_counts",
    "crispat_2beta": "percent_counts",
    "fishash": "log_pval",
}

ts = time.time()

# ═══════════════════════════════════════════════════════════════
# 1. Build sgID → construct mapping (with ENST normalization)
# ═══════════════════════════════════════════════════════════════
print("[1/4] Building sgID → construct mapping ...")

def normalize_guide_name(g):
    """Normalize guide name for lookup: _ENST..._ENST... → ,ENST...,ENST..."""
    # The guide CSV uses commas between ENST IDs
    # The assignment CSVs use underscores
    # Strategy: store BOTH forms in the lookup dict
    return g

sgid_to_construct = {}
sgid_to_gene = {}
construct_to_sgids = defaultdict(set)
# Track which guides have multi-ENST names for normalization
enst_aliases = {}  # underscore_form → comma_form

# Regex pattern for multi-ENST: .23-ENST..._ENST...
ENST_PAT = re.compile(r'^(.*\.23-)(ENST\d+\.\d+)_(ENST[\d._]+)$')

with open(GUIDE_CSV) as f:
    r = csv.DictReader(f)
    for row in r:
        pid = row['unique sgRNA pair ID'].strip()
        gene = row['gene'].strip()
        for sg_col in ['sgID_A', 'sgID_B']:
            sgid = row[sg_col].strip()
            if not sgid:
                continue
            sgid_to_construct[sgid] = pid
            sgid_to_gene[sgid] = gene
            construct_to_sgids[pid].add(sgid)

            # Check for comma-ENST pattern and add underscore alias
            if ',ENST' in sgid:
                alias = sgid.replace(',ENST', '_ENST')
                enst_aliases[alias] = sgid
                sgid_to_construct[alias] = pid
                sgid_to_gene[alias] = gene

print(f"  {len(sgid_to_construct):,} sgID→construct entries "
      f"(+{len(enst_aliases):,} ENST aliases added)")

# Verify the 16 missing guides are now covered
missing_check = [
    'BRD4_-_15391126.23-ENST00000263377.2_ENST00000371835.4',
    'ZNF718_+_124420.23-ENST00000510175.1_ENST00000511079.1_ENST00000513304.1_ENST00000513889.1',
]
for g in missing_check:
    found = g in sgid_to_construct
    print(f"  Verify: {g[:60]}... → {'OK' if found else 'STILL MISSING'}")

# ═══════════════════════════════════════════════════════════════
# 2. Build GT per-cell: construct_id, {sgA, sgB}
# ═══════════════════════════════════════════════════════════════
print("\n[2/4] Building GT per-cell ...")

f = h5py.File(GT_H5AD, 'r')
sgID_codes = f['obs']['sgID_AB'][:]
sgID_cats = f['obs']['__categories']['sgID_AB'][:]
cbs = f['obs']['cell_barcode'][:]
f.close()

gt_construct = {}   # (lane, 16mer) → construct_id
gt_set = {}         # (lane, 16mer) → {sgA, sgB}
n_no_construct = 0
n_no_sgid = 0

for i in range(len(cbs)):
    cb = cbs[i]
    if isinstance(cb, bytes):
        cb = cb.decode()
    m = BC_GT.match(cb)
    if not m:
        continue
    key = (int(m.group(2)), m.group(1))

    sg_code = sgID_codes[i]
    sgab = sgID_cats[sg_code]
    if isinstance(sgab, bytes):
        sgab = sgab.decode()

    # Parse "sgA|sgB"
    parts = sgab.split('|')
    if len(parts) != 2:
        n_no_sgid += 1
        continue
    sgA, sgB = parts[0], parts[1]

    # Find construct from sgA (either sg should give same construct)
    pid = sgid_to_construct.get(sgA)
    if pid is None:
        n_no_construct += 1
        continue

    gt_construct[key] = pid
    gt_set[key] = {sgA, sgB}

n_gt = len(gt_set)
print(f"  {n_gt:,} GT cells (construct: {len(gt_construct):,}, "
      f"missing construct: {n_no_construct}, missing sgID: {n_no_sgid})")

# ═══════════════════════════════════════════════════════════════
# 3. Load assignments + compute metrics per method × tool
# ═══════════════════════════════════════════════════════════════
print("\n[3/4] Computing construct-level set metrics ...")

all_results = {}

for method_name in ["pgmm_em", "umi_t3", "umi_t5", "umi_t10",
                     "crispat_pgmm", "crispat_2beta", "fishash"]:
    sort_col = SORT_COL[method_name]
    sort_desc = SORT_CONFIG[method_name][1]

    for tool in ["ham", "simpleaf_k15"]:
        csv_path = ASSIGNMENT_SPECS[method_name][tool]
        key = f"{method_name}__{tool}"
        t0 = time.time()

        print(f"  {key:35s} ...", end=' ', flush=True)

        # Load assignment CSV: per-cell all guides
        per_cell = defaultdict(list)
        with open(csv_path) as f:
            r = csv.DictReader(f)
            for row in r:
                cell = row.get('cell', '').strip()
                guide = row.get('gRNA', '').strip()
                if not cell or not guide:
                    continue
                m = BC_LANE.match(cell)
                if not m:
                    continue
                cell_key = (int(m.group(2)), m.group(1))

                # Parse sort value
                if sort_col == 'prob_gaussian':
                    score = float(row.get('prob_gaussian', 0) or 0)
                    umi = int(float(row.get('UMI_counts', 0) or 0))
                    per_cell[cell_key].append((guide, score, umi))
                elif sort_col == 'log_pval':
                    lp = float(row.get('log_pval', 0) or 0)
                    umi = int(float(row.get('UMI_counts', 0) or 0))
                    per_cell[cell_key].append((guide, lp, umi))
                elif sort_col == 'percent_counts':
                    pct = float(row.get('percent_counts', 0) or 0)
                    umi = int(float(row.get('UMI_counts', 0) or 0))
                    per_cell[cell_key].append((guide, pct, umi))
                else:  # UMI_counts
                    umi = int(float(row.get('UMI_counts', 0) or 0))
                    per_cell[cell_key].append((guide, umi, umi))

        # Sort per cell and get top-1 + all guides set
        n_shared = 0
        n_both = 0
        n_one = 0
        n_zero = 0
        n_unmapped_top1 = 0
        sum_precision = 0.0
        sum_recall = 0.0
        sum_f1 = 0.0
        n_evaluated = 0  # cells with all guides mapped
        sum_gpc_on_shared = 0  # gpC on shared cells only

        for cell_key, guides in per_cell.items():
            # Must be in GT
            if cell_key not in gt_set:
                continue
            n_shared += 1
            sum_gpc_on_shared += len(guides)

            # Sort guides
            guides.sort(key=lambda x: x[1], reverse=sort_desc)

            # Top-1
            top1_guide = guides[0][0]

            # All guides set (convert to construct-aware set)
            method_set = set()
            all_mapped = True
            for g, _, _ in guides:
                pid = sgid_to_construct.get(g)
                if pid is None:
                    all_mapped = False
                else:
                    method_set.add(pid)  # Use construct ID as set element

            if not all_mapped:
                # This cell has at least one guide that can't be mapped to construct
                # Still evaluate what we can, but flag it
                pass

            # GT set
            gt_s = gt_set[cell_key]
            gt_construct_id = gt_construct[cell_key]

            # Pair recall: how many of {sgA, sgB} are in method_set?
            # But method_set contains construct IDs, not sgIDs.
            # We need sgID-level comparison for pair recall.
            # Reconstruct method sgID set
            method_sgid_set = set()
            for g, _, _ in guides:
                method_sgid_set.add(g)

            n_intersect = len(method_sgid_set & gt_s)
            if n_intersect == 2:
                n_both += 1
            elif n_intersect == 1:
                n_one += 1
            else:
                n_zero += 1

            # Set precision / recall / F1 using sgID-level sets
            # |method_set ∩ gt_set| / |method_set|
            # |method_set ∩ gt_set| / |gt_set| = |method_set ∩ gt_set| / 2
            intersect = method_sgid_set & gt_s
            n_inter = len(intersect)
            n_meth = len(method_sgid_set)
            n_gt = 2

            if n_meth > 0:
                prec = n_inter / n_meth
                rec = n_inter / n_gt
                if prec + rec > 0:
                    f1 = 2 * prec * rec / (prec + rec)
                else:
                    f1 = 0.0
                sum_precision += prec
                sum_recall += rec
                sum_f1 += f1
                n_evaluated += 1

        # Aggregate
        precision_mean = sum_precision / n_evaluated if n_evaluated > 0 else 0
        recall_mean = sum_recall / n_evaluated if n_evaluated > 0 else 0
        f1_mean = sum_f1 / n_evaluated if n_evaluated > 0 else 0
        n_total = n_both + n_one + n_zero
        p_both = n_both / n_total if n_total > 0 else 0
        p_one = n_one / n_total if n_total > 0 else 0
        p_zero = n_zero / n_total if n_total > 0 else 0

        result = {
            "method": method_name,
            "tool": tool,
            "n_shared": n_shared,            # GT ∩ method cells
            "n_evaluated": n_evaluated,       # cells with valid sgID sets
            "pair_recall_both": n_both,
            "pair_recall_one": n_one,
            "pair_recall_zero": n_zero,
            "pair_recall_p_both": round(p_both, 6),
            "pair_recall_p_one": round(p_one, 6),
            "pair_recall_p_zero": round(p_zero, 6),
            "set_precision_mean": round(precision_mean, 6),
            "set_recall_mean": round(recall_mean, 6),
            "set_f1_mean": round(f1_mean, 6),
            "gpC_mean": round(sum(len(v) for v in per_cell.values()) / max(len(per_cell), 1), 2),
            "gpC_mean_on_shared": round(sum_gpc_on_shared / max(n_shared, 1), 2) if n_shared > 0 else 0,
            "wall_s": round(time.time() - t0, 1),
        }
        all_results[key] = result

        print(f"shared={n_shared:,}  both={n_both:,}({p_both:.4f})  "
              f"one={n_one:,}({p_one:.4f})  zero={n_zero:,}({p_zero:.4f})  "
              f"F1={f1_mean:.4f}  gpC(all)={sum(len(v) for v in per_cell.values()) / max(len(per_cell), 1):.2f}"
              f"  gpC(shared)={sum_gpc_on_shared / max(n_shared, 1):.2f}  [{time.time()-t0:.0f}s]")

# ═══════════════════════════════════════════════════════════════
# 4. Save outputs
# ═══════════════════════════════════════════════════════════════
print(f"\n[4/4] Saving outputs [{time.time()-ts:.0f}s]")

with open(OUT_B1, 'w') as f:
    json.dump(all_results, f, indent=2)

# Generate markdown table (HAM)
md_lines = [
    "# Construct-Level Set Evaluation — Replogle 2022",
    "",
    "Per-cell metrics on GT ∩ method cells. Pair recall = fraction of GT {sgA, sgB} pair recovered by method's guide set. Set F1 = per-cell precision/recall harmonic mean.",
    "",
    "## HAM extraction",
    "",
    "| Method | n_shared | P(both) | P(one) | P(zero) | Set Prec | Set Recall | Set F1 | gpC (all) | gpC (shared) |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]

for method_name in ["pgmm_em", "umi_t3", "umi_t5", "umi_t10",
                     "crispat_pgmm", "crispat_2beta", "fishash"]:
    key = f"{method_name}__ham"
    r = all_results[key]
    md_lines.append(
        f"| {method_name} | {r['n_shared']:,} | {r['pair_recall_p_both']:.4f} | "
        f"{r['pair_recall_p_one']:.4f} | {r['pair_recall_p_zero']:.4f} | "
        f"{r['set_precision_mean']:.4f} | {r['set_recall_mean']:.4f} | "
        f"{r['set_f1_mean']:.4f} | {r['gpC_mean']:.2f} | {r['gpC_mean_on_shared']:.2f} |"
    )

md_lines.append("")
md_lines.append("## simpleaf extraction")
md_lines.append("")
md_lines.append("| Method | n_shared | P(both) | P(one) | P(zero) | Set Prec | Set Recall | Set F1 | gpC (all) | gpC (shared) |")
md_lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")

for method_name in ["pgmm_em", "umi_t3", "umi_t5", "umi_t10",
                     "crispat_pgmm", "crispat_2beta", "fishash"]:
    key = f"{method_name}__simpleaf_k15"
    r = all_results[key]
    md_lines.append(
        f"| {method_name} | {r['n_shared']:,} | {r['pair_recall_p_both']:.4f} | "
        f"{r['pair_recall_p_one']:.4f} | {r['pair_recall_p_zero']:.4f} | "
        f"{r['set_precision_mean']:.4f} | {r['set_recall_mean']:.4f} | "
        f"{r['set_f1_mean']:.4f} | {r['gpC_mean']:.2f} | {r['gpC_mean_on_shared']:.2f} |"
        f"{r['set_f1_mean']:.4f} | {r['gpC_mean']:.2f} |"
    )

with open(OUT_B1_MD, 'w') as f:
    f.write('\n'.join(md_lines))

print(f"  → {OUT_B1}")
print(f"  → {OUT_B1_MD}")

# Key observation: pgmm_em vs umi_t3
print(f"\n{'='*70}")
print("Key: pgmm_em vs umi_t3 comparison")
print(f"{'='*70}")
for tool in ["ham", "simpleaf_k15"]:
    pg = all_results[f"pgmm_em__{tool}"]
    um = all_results[f"umi_t3__{tool}"]
    print(f"\n  {tool}:")
    print(f"    pgmm_em:   pair_recall_both={pg['pair_recall_p_both']:.6f}, "
          f"set_prec={pg['set_precision_mean']:.6f}, set_recall={pg['set_recall_mean']:.6f}, F1={pg['set_f1_mean']:.6f}")
    print(f"    umi_t3:    pair_recall_both={um['pair_recall_p_both']:.6f}, "
          f"set_prec={um['set_precision_mean']:.6f}, set_recall={um['set_recall_mean']:.6f}, F1={um['set_f1_mean']:.6f}")
    # Check if any difference
    for metric in ['pair_recall_p_both', 'set_precision_mean', 'set_recall_mean', 'set_f1_mean']:
        if abs(pg[metric] - um[metric]) > 1e-6:
            print(f"    *** {metric}: Δ = {pg[metric] - um[metric]:.6f}")

# UMI threshold trend
print(f"\n{'='*70}")
print("Key: UMI threshold trend (t3 → t5 → t10)")
print(f"{'='*70}")
for tool in ["ham", "simpleaf_k15"]:
    print(f"\n  {tool}:")
    for method_name in ["umi_t3", "umi_t5", "umi_t10"]:
        r = all_results[f"{method_name}__{tool}"]
        print(f"    {method_name}: pair_recall_both={r['pair_recall_p_both']:.4f}, "
              f"one={r['pair_recall_p_one']:.4f}, zero={r['pair_recall_p_zero']:.4f}, "
              f"prec={r['set_precision_mean']:.4f}, recall={r['set_recall_mean']:.4f}, F1={r['set_f1_mean']:.4f}")

print(f"\nDone [{time.time()-ts:.0f}s]")
