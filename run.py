#!/usr/bin/env python3
"""Omnibenchmark metrics module: guide_assignment_metrics (per-lineage scorer).

Scores ONE assignment lineage. Metric selection via --metric (each value = its
own node/output). Re-orchestration only: numeric logic is the vendored
`scripts/*` reused verbatim — either imported (benchmark_assignments,
benchmark_kd_efficiency, which expose parameterised functions) or copied
line-for-line where the source is a top-level monolith (Papalexi Tier-1,
construct set). This entrypoint parameterises the hard-coded reference paths with
injected inputs and picks the per-method loader by auto-detecting the CSV schema.

Implemented + parity-verified: tier1 (dual + single), kd (+nt_nt +pair), construct_set.
Pending per-lineage refactor (vendored, raise at runtime): discovery, mismatch,
strat_tier1, strat_mismatch_loc, capacity — see OMNIBENCHMARK_CONVERSION_PLAN.md.

Contract:
    --output_dir <dir> --name <node_id>
    --guide_assignment.assignments <assignments.csv>
    --data.gt_labels <gt.h5ad|gt.csv>  --data.guide_map <guide_map.csv>  --data.spec <spec.json>
    [--data.gex <gex.h5ad>] [--data.difficulty_table <cell_difficulty.tsv>]
    --metric <tier1|construct_set|kd|discovery|mismatch|strat_tier1|strat_mismatch_loc|capacity>
Output: <output_dir>/<metric>.scores.json
"""
import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "scripts"))

import benchmark_assignments as BA          # noqa: E402  (Tier-1 dual + loaders + compute_metrics)
import benchmark_kd_efficiency as KD        # noqa: E402  (KD loaders + compute_kd_metrics)

BC_STD = re.compile(r'^([ACGT]{16})-(\d+)$')       # Replogle GT: 16mer-gem_group
BC_LANE = re.compile(r'^([ACGT]{16})-L(\d+)$')     # assignment:  16mer-LNN
BC_GT_P = re.compile(r'^l(\d+)_([ACGT]{16})$')     # Papalexi GT: l{N}_16mer


def load_spec(path):
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------
# assignment loader — schema auto-detect reproduces the original per-method sort
# --------------------------------------------------------------------------
def load_assignment_auto(fpath):
    with open(fpath) as f:
        cols = set(next(csv.reader(f)))
    if 'percent_counts' in cols:
        return BA.load_crispat_2beta(fpath)
    if 'log_pval' in cols:
        return BA.load_fishash_topk(fpath)
    if 'prob_gaussian' in cols:
        return BA.load_standard(fpath, sort_key='UMI_counts', sort_desc=True)
    return BA.load_crispat_umi(fpath)


# --------------------------------------------------------------------------
# reference loaders (parameterised copies of the vendored logic)
# --------------------------------------------------------------------------
def load_gt_dual(gt_h5ad):
    """(gem_group,16mer)->'sgA|sgB'. Verbatim from benchmark_assignments.load_ground_truth."""
    import h5py
    f = h5py.File(gt_h5ad, 'r')
    cats = f['obs']['__categories']['sgID_AB']
    gt = {}
    for i in range(len(f['obs']['cell_barcode'])):
        bc = f['obs']['cell_barcode'][i]
        s = bc.decode() if isinstance(bc, bytes) else str(bc)
        m = BC_STD.match(s)
        if m:
            key = (int(m.group(2)), m.group(1))
            sg = cats[f['obs']['sgID_AB'][i]]
            gt[key] = sg.decode() if isinstance(sg, bytes) else str(sg)
    f.close()
    return gt


def load_guide_mapping(guide_csv):
    """sg2pair, label2pair. Verbatim from benchmark_assignments.load_guide_mapping."""
    sg2pair, pair2guides = {}, {}
    with open(guide_csv) as f:
        f.readline()
        for line in f:
            p = line.strip().split(',')
            if len(p) < 8:
                continue
            pid, sgA, sgB = p[0], p[4], p[6]
            sg2pair[sgA] = pid
            sg2pair[sgB] = pid
            pair2guides[pid] = [sgA, sgB]
    label2pair = {}
    for pid, (sa, sb) in pair2guides.items():
        label2pair[f'{sa}|{sb}'] = pid
        label2pair[f'{sb}|{sa}'] = pid
    return sg2pair, label2pair


# ==========================================================================
# metric: tier1
# ==========================================================================
def metric_tier1(args, spec, out_json):
    assignments = getattr(args, "guide_assignment.assignments")
    gt_path = getattr(args, "data.gt_labels")
    guide_map = getattr(args, "data.guide_map")
    t0 = time.time()
    pgmm = load_assignment_auto(assignments)

    if spec["guide_design"] == "single":
        result = _tier1_single(pgmm, gt_path, spec)
    else:
        gt = load_gt_dual(gt_path)
        sg2pair, label2pair = load_guide_mapping(guide_map)
        metrics = BA.compute_metrics(pgmm, gt, sg2pair, label2pair)
        result = {"metric": "tier1", "dataset": spec["dataset"],
                  "guide_design": "dual", **metrics}
    result["wall_s"] = round(time.time() - t0, 1)
    _dump(result, out_json)
    print(f"tier1: rec={result.get('cell_recovery_rate')} "
          f"T1={result.get('t1_pair_accuracy')}")


def _tier1_single(pgmm, gt_csv, spec):
    """Papalexi single-guide Tier-1. Copied verbatim from benchmark_papalexi_tier1.py
    (pool-collapse: GT keyed on (1,16mer), last-pool-wins)."""
    from sklearn.metrics import adjusted_rand_score
    import numpy as np
    gt_label = {}
    with open(gt_csv) as f:
        for row in csv.DictReader(f):
            idx = row.get('index', '').strip()
            guide = row.get('guide_ID', '').strip()
            m = BC_GT_P.match(idx)
            if not m:
                continue
            gt_label[(1, m.group(2))] = guide          # collapse pools; last wins
    n_gt = len(gt_label)
    top1 = {k: pgmm[k][0][0] for k in pgmm if pgmm[k]}
    shared = set(top1) & set(gt_label)
    n_shared = len(shared)
    n_t1 = sum(1 for k in shared if top1[k] == gt_label[k])
    per_gt, per_pred = Counter(), Counter()
    for k in shared:
        per_gt[gt_label[k]] += 1
        per_pred[top1[k]] += 1
    yt = [gt_label[k] for k in shared]
    yp = [top1[k] for k in shared]
    ari = adjusted_rand_score(yt, yp) if len(yt) > 1 else 0.0
    genes = set(per_gt) | set(per_pred)
    x = [per_gt.get(g, 0) for g in genes]
    y = [per_pred.get(g, 0) for g in genes]
    r = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else 0.0
    gpc = [len(v) for v in pgmm.values()]
    return {"metric": "tier1", "dataset": spec["dataset"], "guide_design": "single",
            "cell_recovery_rate": round(n_shared / max(n_gt, 1), 6),
            "t1_pair_accuracy": round(n_t1 / max(n_shared, 1), 6),
            "effective_t1": round(n_t1 / max(n_gt, 1), 6),
            "ari": round(ari, 6), "per_construct_pearson_r": round(r, 6),
            "n_shared_cells": n_shared, "cells_assigned": len(pgmm),
            "guides_per_cell_median": float(np.median(gpc)) if gpc else 0.0,
            "guides_per_cell_mean": float(np.mean(gpc)) if gpc else 0.0}


# ==========================================================================
# metric: construct_set   (dual only; copied from construct_level_eval.py)
# ==========================================================================
def metric_construct_set(args, spec, out_json):
    if spec["guide_design"] != "dual":
        _dump({"metric": "construct_set", "dataset": spec["dataset"],
               "skipped": "single-guide design"}, out_json)
        print("construct_set: skipped (single-guide)")
        return
    assignments = getattr(args, "guide_assignment.assignments")
    gt_path = getattr(args, "data.gt_labels")
    guide_csv = getattr(args, "data.guide_map")
    t0 = time.time()
    import h5py

    # sgID -> construct (+ ENST underscore aliases), verbatim from construct_level_eval.py
    sgid_to_construct = {}
    with open(guide_csv) as f:
        for row in csv.DictReader(f):
            pid = row['unique sgRNA pair ID'].strip()
            for sg_col in ['sgID_A', 'sgID_B']:
                sgid = row[sg_col].strip()
                if not sgid:
                    continue
                sgid_to_construct[sgid] = pid
                if ',ENST' in sgid:
                    sgid_to_construct[sgid.replace(',ENST', '_ENST')] = pid

    # GT per-cell {sgA,sgB}
    f = h5py.File(gt_path, 'r')
    sgID_codes = f['obs']['sgID_AB'][:]
    sgID_cats = f['obs']['__categories']['sgID_AB'][:]
    cbs = f['obs']['cell_barcode'][:]
    f.close()
    gt_set = {}
    for i in range(len(cbs)):
        cb = cbs[i].decode() if isinstance(cbs[i], bytes) else cbs[i]
        m = BC_STD.match(cb)
        if not m:
            continue
        sgab = sgID_cats[sgID_codes[i]]
        sgab = sgab.decode() if isinstance(sgab, bytes) else sgab
        parts = sgab.split('|')
        if len(parts) != 2:
            continue
        if sgid_to_construct.get(parts[0]) is None:
            continue
        gt_set[(int(m.group(2)), m.group(1))] = {parts[0], parts[1]}

    # per-cell all guides (schema-aware sort col)
    with open(assignments) as fh:
        cols = set(next(csv.reader(fh)))
    if 'percent_counts' in cols:
        sort_col = 'percent_counts'
    elif 'log_pval' in cols:
        sort_col = 'log_pval'
    else:
        sort_col = 'UMI_counts'
    per_cell = defaultdict(list)
    with open(assignments) as fh:
        for row in csv.DictReader(fh):
            cell = row.get('cell', '').strip()
            guide = row.get('gRNA', '').strip()
            if not cell or not guide:
                continue
            m = BC_LANE.match(cell)
            if not m:
                continue
            key = (int(m.group(2)), m.group(1))
            umi = int(float(row.get('UMI_counts', 0) or 0))
            sc = float(row.get(sort_col, umi) or 0) if sort_col != 'UMI_counts' else umi
            per_cell[key].append((guide, sc, umi))

    n_shared = n_both = n_one = n_zero = n_evaluated = 0
    sum_p = sum_r = sum_f1 = 0.0
    sum_gpc_shared = 0
    for key, guides in per_cell.items():
        if key not in gt_set:
            continue
        n_shared += 1
        sum_gpc_shared += len(guides)
        method_sgid_set = {g for g, _, _ in guides}
        gt_s = gt_set[key]
        n_inter = len(method_sgid_set & gt_s)
        if n_inter == 2:
            n_both += 1
        elif n_inter == 1:
            n_one += 1
        else:
            n_zero += 1
        n_meth = len(method_sgid_set)
        if n_meth > 0:
            prec = n_inter / n_meth
            rec = n_inter / 2
            f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
            sum_p += prec
            sum_r += rec
            sum_f1 += f1
            n_evaluated += 1
    n_total = n_both + n_one + n_zero
    result = {
        "metric": "construct_set", "dataset": spec["dataset"],
        "n_shared": n_shared, "n_evaluated": n_evaluated,
        "pair_recall_both": n_both, "pair_recall_one": n_one, "pair_recall_zero": n_zero,
        "pair_recall_p_both": round(n_both / n_total, 6) if n_total else 0,
        "pair_recall_p_one": round(n_one / n_total, 6) if n_total else 0,
        "pair_recall_p_zero": round(n_zero / n_total, 6) if n_total else 0,
        "set_precision_mean": round(sum_p / n_evaluated, 6) if n_evaluated else 0,
        "set_recall_mean": round(sum_r / n_evaluated, 6) if n_evaluated else 0,
        "set_f1_mean": round(sum_f1 / n_evaluated, 6) if n_evaluated else 0,
        "gpC_mean": round(sum(len(v) for v in per_cell.values()) / max(len(per_cell), 1), 2),
        "gpC_mean_on_shared": round(sum_gpc_shared / max(n_shared, 1), 2) if n_shared else 0,
        "wall_s": round(time.time() - t0, 1),
    }
    _dump(result, out_json)
    print(f"construct_set: F1={result['set_f1_mean']} both={result['pair_recall_p_both']}")


# ==========================================================================
# metric: kd (+ nt_nt baseline + pair consistency) — imports vendored KD funcs
# ==========================================================================
def metric_kd(args, spec, out_json):
    assignments = getattr(args, "guide_assignment.assignments")
    gex = getattr(args, "data.gex")
    guide_map = getattr(args, "data.guide_map")
    if gex is None:
        sys.exit("kd metric requires --data.gex")
    t0 = time.time()
    if spec["guide_design"] == "dual":
        sg2gene, nt = KD.load_guide_map_k562(guide_map)
        X, cell_lookup, gene_list = KD.load_gex_k562(gex, sg2gene, nt)
    else:
        sg2gene, nt = KD.load_guide_map_papalexi(getattr(args, "data.gt_labels"))
        X, cell_lookup, gene_list = KD.load_gex_papalexi(gex)
    pgmm = load_assignment_auto(assignments)
    n_cells = len(pgmm)
    n_rows = sum(len(v) for v in pgmm.values())
    t1 = time.time()
    metrics = KD.compute_kd_metrics(pgmm, X, cell_lookup, gene_list, sg2gene, nt,
                                    perturbation_type='knockdown', min_cells=5)
    if metrics is None:
        sys.exit("kd: no valid guides")
    if spec["guide_design"] == "dual":
        metrics['pair_consistency'] = KD._compute_pair_consistency_from_per_guide(
            metrics['per_guide'], guide_map)
    summary = {"metric": "kd", "dataset": spec["dataset"],
               "wall_s": round(time.time() - t0, 1), "kd_wall_s": round(time.time() - t1, 1),
               "n_assignment_cells": n_cells, "n_assignment_rows": n_rows,
               **{k: metrics[k] for k in metrics if k not in ('per_guide', 'per_gene_summary')}}
    summary['per_gene_summary'] = metrics['per_gene_summary']
    _dump(summary, out_json)
    print(f"kd: median={summary['kd_efficiency_median']} "
          f"expected_dir={summary['fraction_expected_direction']} guides={summary['n_guides_tested']}")


# ==========================================================================
# stratified metrics (Phase 1a/1c/3) — consume difficulty.table + assignment + GT
# Logic copied verbatim from difficulty_fix_all.py.
# ==========================================================================
STRATA = ["easy", "noise", "ambig", "gray"]


def _load_strata(diff_table):
    """Replicates difficulty_fix_all.py lines 56-96: extraction-specific tertile
    cutoffs -> per-cell stratum_hard, hard_sets, key->idx, k80 array."""
    import numpy as np
    rows = list(csv.DictReader(open(diff_table), delimiter='\t'))
    ents = np.array([float(r['entropy_lib']) for r in rows])
    dlts = np.array([float(r['delta']) for r in rows])
    libs = np.array([float(r['libsize_pctl_in_lane']) for r in rows])
    ent_t = [float(np.percentile(ents, 33.33)), float(np.percentile(ents, 66.67))]
    dlt_t = [float(np.percentile(dlts, 33.33)), float(np.percentile(dlts, 66.67))]
    n = len(rows)
    h = np.array(["gray"] * n, dtype=object)
    for i in range(n):
        e, d, lp = ents[i], dlts[i], libs[i]
        if lp > 50 and e < ent_t[0]:
            h[i] = 'easy'
        elif lp < 50 and e > ent_t[1]:
            h[i] = 'noise'
        elif lp > 50 and e > ent_t[0] and d < dlt_t[0]:
            h[i] = 'ambig'
        else:
            h[i] = 'gray'
    hard_sets = {st: set() for st in STRATA}
    key_to_idx = {}
    k80s = np.array([int(r['k80']) for r in rows])
    for i in range(n):
        m = BC_LANE.match(rows[i]['cell_id'])
        k = (int(m.group(2)), m.group(1)) if m else (0, rows[i]['cell_id'])
        key_to_idx[k] = i
        hard_sets[h[i]].add(k)
    return rows, h, hard_sets, key_to_idx, k80s


def _load_gt_construct(gt_h5ad, guide_csv):
    """sgid_to_c, gt_keys, gt_construct, gt_set_d — verbatim from difficulty_fix_all lines 103-125."""
    import h5py
    sgid_to_c = {}
    with open(guide_csv) as f:
        for row in csv.DictReader(f):
            pid = row['unique sgRNA pair ID'].strip()
            for sg_col in ['sgID_A', 'sgID_B']:
                sg = row[sg_col].strip()
                if sg:
                    sgid_to_c[sg] = pid
                if ',ENST' in sg:
                    sgid_to_c[sg.replace(',ENST', '_ENST')] = pid
    f = h5py.File(gt_h5ad, 'r')
    cbs = f['obs']['cell_barcode'][:]
    sg_codes = f['obs']['sgID_AB'][:]
    sg_cats = f['obs']['__categories']['sgID_AB'][:]
    f.close()
    gt_keys, gt_construct, gt_set_d = set(), {}, {}
    for i in range(len(cbs)):
        cb = cbs[i].decode() if isinstance(cbs[i], bytes) else str(cbs[i])
        m = BC_STD.match(cb)
        if not m:
            continue
        k = (int(m.group(2)), m.group(1))
        gt_keys.add(k)
        sgab = sg_cats[sg_codes[i]]
        sgab = sgab.decode() if isinstance(sgab, bytes) else str(sgab)
        parts = sgab.split('|')
        if len(parts) == 2:
            pid = sgid_to_c.get(parts[0])
            if pid:
                gt_construct[k] = pid
                gt_set_d[k] = {parts[0], parts[1]}
    return sgid_to_c, gt_keys, gt_construct, gt_set_d


def _load_top1_ag(assignments):
    """top1{key:guide}, ag{key:set} — verbatim streaming from difficulty_fix_all lines 137-156."""
    with open(assignments) as fh:
        cols = set(next(csv.reader(fh)))
    if 'log_pval' in cols:
        sort_col = 'log_pval'
    elif 'percent_counts' in cols:
        sort_col = 'percent_counts'
    else:
        sort_col = 'UMI_counts'
    top1, ag = {}, {}
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
            if sort_col == 'log_pval':
                s = float(row.get(sort_col, 0) or 0)
                if k not in top1 or s < top1[k][1]:
                    top1[k] = (guide, s)
            elif sort_col == 'percent_counts':
                s = float(row.get(sort_col, 0) or 0)
                if k not in top1 or s > top1[k][1]:
                    top1[k] = (guide, s)
            else:
                s = int(float(row.get('UMI_counts', 0) or 0))
                if k not in top1 or s > top1[k][1]:
                    top1[k] = (guide, s)
            if k not in ag:
                ag[k] = set()
            ag[k].add(guide)
    for k in top1:
        top1[k] = top1[k][0]
    return top1, ag


def _require_strat_inputs(args, spec):
    if spec["guide_design"] != "dual":
        sys.exit("stratified metrics wired for dual-guide (Replogle) only in this cut.")
    dt = getattr(args, "data.difficulty_table")
    if not dt:
        sys.exit("stratified metric requires --data.difficulty_table")
    return dt


def metric_strat_tier1(args, spec, out_json):
    dt = _require_strat_inputs(args, spec)
    assignments = getattr(args, "guide_assignment.assignments")
    gt_path = getattr(args, "data.gt_labels")
    guide_csv = getattr(args, "data.guide_map")
    _, _, hard_sets, _, _ = _load_strata(dt)
    sgid_to_c, gt_keys, gt_construct, gt_set_d = _load_gt_construct(gt_path, guide_csv)
    top1, ag = _load_top1_ag(assignments)
    result = {}
    for st in STRATA:
        shared = hard_sets[st] & gt_keys
        n_gt = len(shared)
        if n_gt < 10:
            continue
        n_t1 = n_assn = nb = no = nz = 0
        sp = sr = sf = ne = 0.0
        for k in shared:
            g1 = top1.get(k)
            if g1 is None:
                continue
            n_assn += 1
            if sgid_to_c.get(g1, g1) == gt_construct.get(k):
                n_t1 += 1
            mg = ag.get(k, set())
            gs = gt_set_d.get(k, set())
            ni = len(mg & gs)
            if ni == 2:
                nb += 1
            elif ni == 1:
                no += 1
            else:
                nz += 1
            nm = len(mg)
            if nm > 0:
                p = ni / nm
                r = ni / 2
                sp += p
                sr += r
                if p + r > 0:
                    sf += 2 * p * r / (p + r)
                    ne += 1
        nt = nb + no + nz
        rec = n_assn / max(n_gt, 1)
        t1_acc = n_t1 / max(n_assn, 1) if n_assn > 0 else 0
        result[f"{spec['dataset']}__{st}"] = {
            "stratum": st, "n_gt": n_gt, "n_assigned": n_assn,
            "rec": round(rec, 6), "t1": round(t1_acc, 6), "eff_t1": round(rec * t1_acc, 6),
            "p_both": round(nb / max(nt, 1), 6), "p_one": round(no / max(nt, 1), 6),
            "p_zero": round(nz / max(nt, 1), 6),
            "set_prec": round(sp / max(ne, 1), 6), "set_rec": round(sr / max(ne, 1), 6),
            "set_f1": round(sf / max(ne, 1), 6)}
    _dump(result, out_json)
    print(f"strat_tier1: {len(result)} strata")


def metric_strat_mismatch_loc(args, spec, out_json):
    dt = _require_strat_inputs(args, spec)
    assignments = getattr(args, "guide_assignment.assignments")
    gt_path = getattr(args, "data.gt_labels")
    guide_csv = getattr(args, "data.guide_map")
    _, h, _, key_to_idx, _ = _load_strata(dt)
    sgid_to_c, gt_keys, gt_construct, _ = _load_gt_construct(gt_path, guide_csv)
    top1, _ = _load_top1_ag(assignments)
    mm_c, corr_c = [], []
    for k in gt_keys:
        g = top1.get(k)
        if g is None:
            continue
        if sgid_to_c.get(g, g) == gt_construct.get(k):
            corr_c.append(k)
        else:
            mm_c.append(k)
    mm_s = Counter()
    for k in mm_c:
        si = key_to_idx.get(k)
        if si is not None:
            mm_s[h[si]] += 1
    total = len(mm_c)
    result = {"dataset": spec["dataset"], "n_mismatch": total, "n_correct": len(corr_c),
              "mm_pct": {s: round(mm_s.get(s, 0) / max(total, 1) * 100, 1) for s in STRATA}}
    _dump(result, out_json)
    print(f"strat_mismatch_loc: n_mismatch={total}")


def metric_capacity(args, spec, out_json):
    import numpy as np
    dt = _require_strat_inputs(args, spec)
    assignments = getattr(args, "guide_assignment.assignments")
    _, h, _, key_to_idx, k80s = _load_strata(dt)
    gpc = {}
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
            gpc[k] = gpc.get(k, 0) + 1
    hi_k80, hi_gpc = [], []
    for k, n in gpc.items():
        si = key_to_idx.get(k)
        if si is None:
            continue
        if h[si] in ('easy', 'ambig'):
            hi_k80.append(int(k80s[si]))
            hi_gpc.append(n)
    slope = intercept = 0
    if len(hi_k80) > 10:
        Xk = np.column_stack([hi_k80, np.ones(len(hi_k80))])
        beta, _, _, _ = np.linalg.lstsq(Xk, hi_gpc, rcond=None)
        slope = round(float(beta[0]), 3)
        intercept = round(float(beta[1]), 2)
    ag_all = np.array(list(gpc.values()))
    result = {"dataset": spec["dataset"], "mean_gpc": round(float(ag_all.mean()), 2),
              "n_cells": len(gpc), "slope_k80": slope, "intercept": intercept}
    _dump(result, out_json)
    print(f"capacity: mean_gpc={result['mean_gpc']} slope_k80={slope}")


# ==========================================================================
# discovery (Phase 2 crispat-style) — reuses vendored phase2_discovery functions.
# GEX-heavy (~160 GB HVG dense); NT pool is per-method (archive-consistent).
# ==========================================================================
def metric_discovery(args, spec, out_json):
    from multiprocessing import Pool
    import phase2_discovery as PD
    assignments = getattr(args, "guide_assignment.assignments")
    gex = getattr(args, "data.gex")
    guide_map = getattr(args, "data.guide_map")
    if gex is None:
        sys.exit("discovery requires --data.gex")
    dual = spec["guide_design"] == "dual"
    if dual:
        PD.DATASETS['replogle2022']['gex_h5ad'] = gex
        PD.DATASETS['replogle2022']['guide_csv'] = guide_map
        X, cell_lookup, gene_names, sg2gene, nt_sgrnas, guide_info = PD.load_replogle_gex()
        min_cells = PD.DATASETS['replogle2022']['min_cells_per_construct']
    else:
        PD.DATASETS['papalexi2021']['gex_h5ad'] = gex
        PD.DATASETS['papalexi2021']['assignment_ref'] = getattr(args, "data.gt_labels")
        X, cell_lookup, gene_names, sg2gene, nt_sgrnas, guide_info = PD.load_papalexi_gex()
        min_cells = PD.DATASETS['papalexi2021']['min_cells_per_construct']
    ng = X.shape[1]
    pgmm = load_assignment_auto(assignments)
    n_cells = len(pgmm)
    t0 = time.time()
    construct_cells = defaultdict(set)
    nt_set = set()
    for key in pgmm:
        if not pgmm[key]:
            continue
        tg = pgmm[key][0][0]
        idx = cell_lookup.get(key)
        if idx is None:
            continue
        if tg in nt_sgrnas:
            nt_set.add(idx)
        elif dual:
            info = guide_info.get(tg)
            if info:
                construct_cells[info[0]].add(idx)
        else:
            gene = sg2gene.get(tg, tg)
            if gene != 'non-targeting':
                construct_cells[gene].add(idx)
    nt_list = sorted(nt_set)
    n_nt = len(nt_list)
    if n_nt < 5:
        sys.exit("discovery: <5 NT cells")
    X_nt = X[nt_list, :]
    tasks = [(X[sorted(cells), :], X_nt, cid) for cid, cells in construct_cells.items()
             if len(cells) >= min_cells]
    discoveries = []
    if tasks:
        with Pool(int(getattr(args, "workers", 8) or 8)) as pool:
            for cid, n_sig, ng_t, ncells in pool.map(PD._de_one, tasks):
                discoveries.append(n_sig)
    n_tested = len(discoveries)
    median_disc = float(np.median(discoveries)) if discoveries else 0
    total_disc = int(np.sum(discoveries))
    # FPR
    nt_guide_cells = defaultdict(set)
    for key in pgmm:
        if not pgmm[key]:
            continue
        tg = pgmm[key][0][0]
        if tg not in nt_sgrnas:
            continue
        idx = cell_lookup.get(key)
        if idx is not None:
            nt_guide_cells[tg].add(idx)
    fp_tasks = [(X[sorted(cells), :], X_nt, g) for g, cells in nt_guide_cells.items()
                if len(cells) >= 5]
    fp_counts = []
    if fp_tasks:
        with Pool(min(int(getattr(args, "workers", 8) or 8), 8)) as pool:
            for cid, n_sig, ng_t, ncells in pool.map(PD._de_one, fp_tasks):
                fp_counts.append(n_sig)
    total_fp = int(np.sum(fp_counts)) if fp_counts else 0
    denom = len(fp_tasks) * ng
    fpr = total_fp / denom if denom > 0 else 0
    cells_per = [len(v) for v in construct_cells.values()]
    result = {"metric": "discovery", "dataset": spec["dataset"], "wall_s": round(time.time() - t0, 1),
              "n_cells_assigned": n_cells, "median_cells_per_construct": float(np.median(cells_per)) if cells_per else 0,
              "n_constructs_tested": n_tested, "discovery_n_tested": n_tested,
              "discovery_median": median_disc, "discovery_total": total_disc, "n_nt_cells": n_nt,
              "n_nt_guides_tested": len(fp_tasks), "total_false_discoveries": total_fp,
              "total_tests": denom, "fpr": fpr}
    _dump(result, out_json)
    print(f"discovery: median={median_disc} total={total_disc} fpr={fpr:.6f}")


import numpy as np  # noqa: E402  (used by discovery/mismatch)


def metric_mismatch(args, spec, out_json):
    """Per-lineage mismatch arbitration (analyze_mismatch.py, single method).
    NT pool is this method's NT cells (per-lineage); archive used the 5-method
    union pool, so win_rate may differ marginally — documented in CONSISTENCY."""
    import h5py
    import anndata as ad
    if spec["guide_design"] != "dual":
        _dump({"metric": "mismatch", "skipped": "dual-guide only"}, out_json)
        return
    EPS = 0.01
    assignments = getattr(args, "guide_assignment.assignments")
    gex = getattr(args, "data.gex")
    gt_h5ad = getattr(args, "data.gt_labels")
    guide_csv = getattr(args, "data.guide_map")
    sym_map = getattr(args, "data.gene_symbol_to_ensg")
    if not sym_map:
        sys.exit("metric 'mismatch' requires --data.gene_symbol_to_ensg")
    t0 = time.time()

    s2p, p2g = {}, {}
    sg2gene, nt_sgrnas = {}, set()
    with open(guide_csv) as g:
        g.readline()
        for line in g:
            p = line.strip().split(',')
            if len(p) < 8:
                continue
            pid, gene, sgA, sgB = p[0], p[1], p[4], p[6]
            s2p[sgA] = pid
            s2p[sgB] = pid
            if pid not in p2g:
                p2g[pid] = gene
            if gene == 'non-targeting':
                if sgA:
                    nt_sgrnas.add(sgA)
                if sgB:
                    nt_sgrnas.add(sgB)
            sg2gene[sgA] = gene
            sg2gene[sgB] = gene
    f = h5py.File(gt_h5ad, 'r')
    sgID_c = f['obs']['sgID_AB'][:]
    sgID_t = f['obs']['__categories']['sgID_AB'][:]
    cbs = f['obs']['cell_barcode'][:]
    f.close()
    gt = {}
    for i in range(len(cbs)):
        s = cbs[i].decode() if isinstance(cbs[i], bytes) else str(cbs[i])
        m = BC_STD.match(s)
        if not m:
            continue
        sgab = sgID_t[sgID_c[i]]
        sgab = sgab.decode() if isinstance(sgab, bytes) else str(sgab)
        sgA = sgab.split('|')[0] if '|' in sgab else sgab
        pid = s2p.get(sgA, sgA)
        gt[(int(m.group(2)), m.group(1))] = (pid, p2g.get(pid, sgA))
    with open(sym_map) as fh:
        sym2ensg = json.load(fh)

    top1 = {k: v[0][0] for k, v in load_assignment_auto(assignments).items() if v}
    needed = set()
    cells = []
    for ck, gm in top1.items():
        tv = gt.get(ck)
        if tv is None:
            continue
        gp, gg = tv
        if s2p.get(gm, gm) == gp:
            continue
        Tm = sg2gene.get(gm)
        if Tm and Tm != 'non-targeting':
            needed.add(Tm)
        if gg and gg != 'non-targeting':
            needed.add(gg)
        cells.append((ck, gm, Tm, gp, gg))

    sc = ad.read_h5ad(gex)
    nc = sc.shape[0]
    ensg2col = {}
    for i, v in enumerate(sc.var_names):
        s = v.decode('utf-8') if isinstance(v, bytes) else str(v)
        ensg2col[s.replace('_S', '')] = i
    gene2col = {}
    for sym in sorted(needed):
        es = sym2ensg.get(sym)
        if es and es in ensg2col:
            gene2col[sym] = ensg2col[es]
    needed_cols = sorted(gene2col.values())
    col2loc = {c: i for i, c in enumerate(needed_cols)}
    X_small = sc.X[:, needed_cols].toarray().astype(np.float32)
    cell_lookup = {}
    for i in range(nc):
        seq = str(sc.obs['barcode_16mer'].iloc[i])
        lane = int(sc.obs['lane'].iloc[i])
        cell_lookup[(lane, seq)] = i
    del sc

    nt_cells = set()
    for ck, gm in top1.items():
        if gm in nt_sgrnas:
            rw = cell_lookup.get(ck)
            if rw is not None:
                nt_cells.add(rw)
    nt_list = sorted(nt_cells)
    gene_means, nt_means = {}, {}
    for sym in gene2col:
        loc = col2loc[gene2col[sym]]
        gene_means[sym] = float(X_small[:, loc].mean())
        nt_means[sym] = float(X_small[nt_list, loc].mean()) if nt_list else gene_means[sym]

    n_skip = n_mw = n_gw = n_tie = 0
    for ck, gm, Tm, gp, Tg in cells:
        rw = cell_lookup.get(ck)
        if rw is None or Tm is None or Tg is None or Tm == 'non-targeting' or Tg == 'non-targeting':
            n_skip += 1
            continue
        lm = col2loc.get(gene2col.get(Tm, -1))
        lg = col2loc.get(gene2col.get(Tg, -1))
        if lm is None or lg is None:
            n_skip += 1
            continue
        em = float(X_small[rw, lm])
        eg = float(X_small[rw, lg])
        enm = nt_means.get(Tm, em)
        eng = nt_means.get(Tg, eg)
        epsm = EPS * max(gene_means.get(Tm, 1e-8), 1e-8)
        epsg = EPS * max(gene_means.get(Tg, 1e-8), 1e-8)
        kdm = float(np.log2(max(em + epsm, epsm)) - np.log2(max(enm + epsm, epsm)))
        kdg = float(np.log2(max(eg + epsg, epsg)) - np.log2(max(eng + epsg, epsg)))
        if kdm < kdg - 0.01:
            n_mw += 1
        elif kdg < kdm - 0.01:
            n_gw += 1
        else:
            n_tie += 1
    nv = n_mw + n_gw + n_tie
    result = {"metric": "mismatch", "dataset": spec["dataset"], "n_mismatch": len(cells),
              "n_skip": n_skip, "n_valid": nv, "n_method_wins": n_mw, "n_gt_wins": n_gw,
              "n_tie": n_tie, "win_rate": round(n_mw / nv, 4) if nv else 0,
              "wall_s": round(time.time() - t0, 1)}
    _dump(result, out_json)
    print(f"mismatch: n={len(cells)} win_rate={result['win_rate']}")


def metric_strat_delta_kd(args, spec, out_json):
    """Delta-KD binned by difficulty STRATUM (easy/noise/ambig/gray).
    Single-lineage port of benchmark_framework.analyses.run_difficulty_phase2_stratum
    (the new stratum-binned delta-KD). Reuses this module's _load_strata (identical to
    the framework's _compute_strata) + the KD gex/guide-map loaders. Target KD per
    stratum + an NT negative control (NT guides vs a random non-NT gene)."""
    import numpy as np
    if spec["guide_design"] != "dual":
        _dump({"metric": "strat_delta_kd", "dataset": spec["dataset"],
               "skipped": "dual-guide only"}, out_json)
        print("strat_delta_kd: skipped (single-guide)")
        return
    dt = getattr(args, "data.difficulty_table")
    gex = getattr(args, "data.gex")
    guide_map = getattr(args, "data.guide_map")
    assignments = getattr(args, "guide_assignment.assignments")
    if gex is None or dt is None:
        sys.exit("strat_delta_kd requires --data.gex and --data.difficulty_table")
    eps_frac = 0.01
    # strata (same tertile logic as the framework's _compute_strata)
    _rows, h, _hs, key_to_idx, _k = _load_strata(dt)
    stratum_per_cell = {k: h[i] for k, i in key_to_idx.items()}
    # target-gene column subset + guide map
    sg2gene, nt = KD.load_guide_map_k562(guide_map)
    X, cell_lookup, gene_list = KD.load_gex_k562(gex, sg2gene, nt)
    g2i = {g: i for i, g in enumerate(gene_list)}
    col_cache = {}

    def gcol(gi):
        if gi not in col_cache:
            c = X[:, gi]
            col_cache[gi] = np.asarray(c.todense()).flatten() if hasattr(c, "todense") else np.asarray(c).flatten()
        return col_cache[gi]

    needed_gi = {g2i[gene] for gene in set(sg2gene.values()) if gene in g2i}
    gene_eps = {gi: max(gcol(gi).mean() * eps_frac, 1e-6) for gi in needed_gi}
    pgmm = load_assignment_auto(assignments)
    top1 = {k: v[0][0] for k, v in pgmm.items() if v}
    nt_list = sorted({cell_lookup[k] for k, g in top1.items() if g in nt and k in cell_lookup})
    if len(nt_list) < 5:
        sys.exit("strat_delta_kd: <5 NT cells")
    ntm_cache = {}

    def ntmean(gi):
        if gi not in ntm_cache:
            ntm_cache[gi] = float(gcol(gi)[nt_list].mean())
        return ntm_cache[gi]

    guide_cells, nt_guide_cells = defaultdict(list), defaultdict(list)
    for k, g in top1.items():
        st = stratum_per_cell.get(k)
        idx = cell_lookup.get(k)
        if st is None or idx is None:
            continue
        if g in nt:
            nt_guide_cells[g].append((idx, st))
        else:
            gene = sg2gene.get(g)
            if gene and gene != 'non-targeting' and gene in g2i:
                guide_cells[(g, gene)].append((idx, st))
    non_nt_genes = sorted({gene for gene in sg2gene.values() if gene != 'non-targeting' and gene in g2i})
    rng = np.random.RandomState(42)

    def stratum_kd(groups, pick_gene):
        by_st = defaultdict(list)
        for key, cs in groups.items():
            if len(cs) < 20:
                continue
            gene = pick_gene(key)
            gi = g2i.get(gene)
            if gi is None:
                continue
            col = gcol(gi)
            eps = gene_eps[gi]
            ntm = ntmean(gi)
            for st in STRATA:
                sel = [idx for (idx, s) in cs if s == st]
                if len(sel) < 5:
                    continue
                me = col[sel].mean()
                by_st[st].append(float(np.log2(max(me + eps, eps)) - np.log2(max(ntm + eps, eps))))
        return by_st

    tgt = stratum_kd(guide_cells, lambda key: key[1])
    ntc = stratum_kd(nt_guide_cells,
                     lambda key: non_nt_genes[rng.randint(len(non_nt_genes))]) if non_nt_genes else {}
    res = {"metric": "strat_delta_kd", "dataset": spec["dataset"], "target": {}, "nt_control": {}}
    for st in STRATA:
        if tgt.get(st):
            a = tgt[st]
            res["target"][st] = {"n": len(a), "kd_median": round(float(np.median(a)), 4),
                                 "kd_mean": round(float(np.mean(a)), 4), "kd_std": round(float(np.std(a)), 4)}
        if ntc.get(st):
            a = ntc[st]
            res["nt_control"][st] = {"n": len(a), "kd_median": round(float(np.median(a)), 4),
                                     "kd_mean": round(float(np.mean(a)), 4), "kd_std": round(float(np.std(a)), 4)}
    _dump(res, out_json)
    print(f"strat_delta_kd: {len(res['target'])} target strata")


def _pending(name):
    def _fn(args, spec, out_json):
        sys.exit(f"metric '{name}': vendored in scripts/ but per-lineage refactor pending.")
    return _fn


def _dump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


DISPATCH = {
    "tier1": metric_tier1,
    "construct_set": metric_construct_set,
    "kd": metric_kd,
    "discovery": metric_discovery,
    "mismatch": metric_mismatch,
    "strat_tier1": metric_strat_tier1,
    "strat_mismatch_loc": metric_strat_mismatch_loc,
    "capacity": metric_capacity,
    "strat_delta_kd": metric_strat_delta_kd,
}


def main():
    p = argparse.ArgumentParser(description="Omnibenchmark module: guide_assignment_metrics")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--name", default="node")
    p.add_argument("--guide_assignment.assignments", required=True)
    p.add_argument("--data.gt_labels", required=True)
    p.add_argument("--data.guide_map", required=True)
    p.add_argument("--data.spec", required=True)
    p.add_argument("--data.gex", default=None)
    p.add_argument("--data.difficulty_table", default=None)
    p.add_argument("--data.gene_symbol_to_ensg", default=None)
    p.add_argument("--metric", required=True)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    spec = load_spec(getattr(args, "data.spec"))
    out_json = os.path.join(args.output_dir, f"{args.metric}.scores.json")
    fn = DISPATCH.get(args.metric)
    if fn is None:
        sys.exit(f"unknown metric '{args.metric}' (have {sorted(DISPATCH)})")
    fn(args, spec, out_json)
    print("guide_assignment_metrics: wrote", os.path.basename(out_json))


if __name__ == "__main__":
    main()
