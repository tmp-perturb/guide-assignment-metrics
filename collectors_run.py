#!/usr/bin/env python3
"""Omnibenchmark metric_collector: guide_assignment_collectors.

Cross-lineage fan-in aggregators. Re-orchestration only: logic copied verbatim
from the vendored scripts (compute_jaccard.py, difficulty_fix_all.py Phases 1b/1e).

    jaccard          -> guide-assignment Jaccard across all lineages (compute_jaccard.py)
    strat_jaccard    -> Phase 1b: per-stratum pairwise Jaccard (difficulty_fix_all.py)
    extraction_shift -> Phase 1e: delta/entropy shift between two extractions' tables

Inputs are passed as label=path tokens (--assignments label=path ...) and
--difficulty_table label=path; OB fan-in paths are also accepted and the
label (data-lineage + method) is inferred from the path.
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

BC_LANE = re.compile(r'^([ACGT]{16})-L(\d+)$')
STRATA = ["easy", "noise", "ambig", "gray"]
# canonical 5-method Jaccard pairs (difficulty_fix_all.JACCARD_PAIRS)
JACCARD_PAIRS = [("pgmm_em", "umi_t3"), ("pgmm_em", "crispat_pgmm"), ("pgmm_em", "crispat_2beta"),
                 ("pgmm_em", "fishash"), ("umi_t3", "crispat_pgmm"), ("umi_t3", "crispat_2beta"),
                 ("umi_t3", "fishash"), ("crispat_pgmm", "crispat_2beta"), ("crispat_pgmm", "fishash"),
                 ("crispat_2beta", "fishash")]


def _sort_col(path):
    with open(path) as f:
        cols = set(next(csv.reader(f)))
    if 'log_pval' in cols:
        return 'log_pval', False
    if 'percent_counts' in cols:
        return 'percent_counts', True
    return 'UMI_counts', True


def load_top1_guide(path):
    """{cell -> top1_guide}, schema-aware. (compute_jaccard.load_top1_guide)"""
    sort_col, sort_desc = _sort_col(path)
    per_cell = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            cell = row.get("cell", "").strip()
            guide = row.get("gRNA", row.get("guide", "")).strip()
            if not cell or not guide:
                continue
            per_cell[cell].append((float(row.get(sort_col, 0) or 0), guide))
    out = {}
    for cell, guides in per_cell.items():
        guides.sort(key=lambda x: x[0], reverse=sort_desc)
        out[cell] = guides[0][1]
    return out


def load_top1_key(path):
    """{(lane,16mer) -> top1_guide}, streaming max (difficulty_fix_all Phase 1b)."""
    sort_col, _ = _sort_col(path)
    t1 = {}
    with open(path) as f:
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
                if k not in t1 or s < t1[k][1]:
                    t1[k] = (guide, s)
            elif sort_col == 'percent_counts':
                s = float(row.get(sort_col, 0) or 0)
                if k not in t1 or s > t1[k][1]:
                    t1[k] = (guide, s)
            else:
                s = int(float(row.get('UMI_counts', 0) or 0))
                if k not in t1 or s > t1[k][1]:
                    t1[k] = (guide, s)
    return {k: v[0] for k, v in t1.items()}


def load_strata(diff_table):
    """key -> stratum, from difficulty.table tertiles (difficulty_fix_all lines 56-96)."""
    rows = list(csv.DictReader(open(diff_table), delimiter='\t'))
    ents = np.array([float(r['entropy_lib']) for r in rows])
    dlts = np.array([float(r['delta']) for r in rows])
    libs = np.array([float(r['libsize_pctl_in_lane']) for r in rows])
    ent_t = [float(np.percentile(ents, 33.33)), float(np.percentile(ents, 66.67))]
    dlt_t = [float(np.percentile(dlts, 33.33)), float(np.percentile(dlts, 66.67))]
    hard_sets = {st: set() for st in STRATA}
    for i, r in enumerate(rows):
        e, d, lp = ents[i], dlts[i], libs[i]
        if lp > 50 and e < ent_t[0]:
            st = 'easy'
        elif lp < 50 and e > ent_t[1]:
            st = 'noise'
        elif lp > 50 and e > ent_t[0] and d < dlt_t[0]:
            st = 'ambig'
        else:
            st = 'gray'
        m = BC_LANE.match(r['cell_id'])
        k = (int(m.group(2)), m.group(1)) if m else (0, r['cell_id'])
        hard_sets[st].add(k)
    return hard_sets


# ---------------------------------------------------------------------------
# Lineage resolution for OB fan-in.
# OB writes a `parameters.json` next to every node output, so the exact params
# (variant / threshold / crispat_method) are recoverable — the path hash alone
# is opaque and cannot distinguish crispat pgmm/2beta or umi t3/t5/t10.
# ---------------------------------------------------------------------------
def _params_for(path):
    pj = os.path.join(os.path.dirname(path), "parameters.json")
    if os.path.exists(pj):
        try:
            return json.load(open(pj))
        except Exception:
            return {}
    return {}


def _norm_method(module, params):
    """(OB module id, params) -> normalized method name (or None if unmapped)."""
    if module == "pgmm":
        return {"mle": "pgmm_em", "map_e2": "pgmm_em_e2"}.get(str(params.get("variant", "mle")))
    if module == "umi":
        return f"umi_t{params.get('threshold')}" if params.get("threshold") is not None else None
    if module == "crispat":
        return {"pgmm": "crispat_pgmm", "2beta": "crispat_2beta"}.get(str(params.get("crispat_method")))
    if module == "fishash":
        return "fishash"
    return module


def _dataset_of(path):
    """The first-stage lineage id (dir after 'data'), which encodes the extraction tool."""
    parts = os.path.normpath(path).split(os.sep)
    if "data" in parts:
        i = parts.index("data")
        if i + 1 < len(parts):
            return parts[i + 1]
    # fallback: {dataset}_<suffix> filename prefix (e.g. replogle_ham_cell_difficulty.tsv)
    base = os.path.basename(path)
    for suf in ("_cell_difficulty.tsv", "_gex.h5ad", "_gt.h5ad", "_guide_map.csv"):
        if base.endswith(suf):
            return base[: -len(suf)]
    return "default"


def _lineage(path):
    """assignments.csv path -> (dataset, method_norm). Reads sibling parameters.json."""
    parts = os.path.normpath(path).split(os.sep)
    module = None
    if "guide_assignment" in parts:
        module = parts[parts.index("guide_assignment") + 1]
    method = _norm_method(module, _params_for(path)) if module else None
    return _dataset_of(path), (method or module or os.path.basename(os.path.dirname(path)))


def _parse_assignments(tokens):
    """-> list of (dataset, method, path). 'label=path' forces (default, label)."""
    out = []
    for t in tokens:
        if "=" in t and not os.path.exists(t):
            lab, p = t.split("=", 1)
            out.append(("default", lab, p))
        else:
            ds, m = _lineage(t)
            out.append((ds, m, t))
    return out


def _parse_tables(tokens):
    """difficulty tables / data inputs -> {dataset: path}. 'label=path' forces label."""
    out = {}
    for t in tokens:
        if "=" in t and not os.path.exists(t):
            lab, p = t.split("=", 1)
            out[lab] = p
        else:
            out[_dataset_of(t)] = t
    return out


# canonical 5 methods for stratified Jaccard / mismatch (difficulty_fix_all naming)
CANON5 = ["pgmm_em", "umi_t3", "crispat_pgmm", "crispat_2beta", "fishash"]


def collect_jaccard(entries, out):
    """entries: [(dataset, method, path)]. Full cross-lineage matrix."""
    labels = [f"{m}__{ds}" for ds, m, _ in entries]
    assignments = [load_top1_guide(p) for _, _, p in entries]
    n = len(labels)
    matrix = [[1.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            a, b = assignments[i], assignments[j]
            shared = set(a) & set(b)
            matrix[i][j] = matrix[j][i] = round(sum(1 for c in shared if a[c] == b[c]) / len(shared), 6) if shared else 0.0
    json.dump({"labels": labels, "jaccard": matrix}, open(out, "w"), indent=2)
    print(f"jaccard: {n} lineages")


def _strat_jaccard_one(method2path, diff_table):
    hard_sets = load_strata(diff_table)
    top1s = {m: load_top1_key(p) for m, p in method2path.items() if m in CANON5}
    key_sets_m = {m: set(t.keys()) for m, t in top1s.items()}
    result = {}
    for st in STRATA:
        st_set = hard_sets[st]
        pr = {}
        for mi, mj in JACCARD_PAIRS:
            if mi not in top1s or mj not in top1s:
                continue
            sc = key_sets_m[mi] & key_sets_m[mj] & st_set
            if len(sc) < 5:
                continue
            agree = sum(1 for c in sc if top1s[mi].get(c) == top1s[mj].get(c))
            pr[f"{mi}_{mj}"] = {"J": round(agree / len(sc), 6), "n": len(sc)}
        result[st] = pr
    return result


def collect_strat_jaccard(entries, diff_by_ds, out):
    """Phase 1b, grouped by dataset (tool). One difficulty table per dataset."""
    by_ds = defaultdict(dict)
    for ds, m, p in entries:
        by_ds[ds][m] = p
    result = {}
    for ds, m2p in by_ds.items():
        dt = diff_by_ds.get(ds) or (list(diff_by_ds.values())[0] if len(diff_by_ds) == 1 else None)
        if dt is None:
            continue
        result[ds] = _strat_jaccard_one(m2p, dt)
    # single-dataset call -> flatten to {stratum:...} for back-compat with standalone tests
    payload = result[list(result)[0]] if len(result) == 1 else result
    json.dump(payload, open(out, "w"), indent=2)
    print(f"strat_jaccard: {len(result)} dataset(s)")


def collect_extraction_shift(diff_tables, out):
    """Phase 1e: delta/entropy shift between two extractions (difficulty_fix_all lines 280-300)."""
    tabs = list(diff_tables.values())
    if len(tabs) != 2:
        sys.exit(f"extraction_shift needs exactly 2 difficulty tables, got {len(tabs)}")
    def keyed(p):
        d = {}
        for r in csv.DictReader(open(p), delimiter='\t'):
            m = BC_LANE.match(r['cell_id'])
            k = (int(m.group(2)), m.group(1)) if m else (0, r['cell_id'])
            d[k] = r
        return d
    a, b = keyed(tabs[0]), keyed(tabs[1])
    common = set(a) & set(b)
    sd = [float(b[k]["delta"]) - float(a[k]["delta"]) for k in common]
    se = [float(b[k]["entropy_lib"]) - float(a[k]["entropy_lib"]) for k in common]
    res = {"n_common": len(common),
           "delta_shift_median": round(float(np.median(sd)), 4),
           "entropy_shift_median": round(float(np.median(se)), 4)}
    json.dump(res, open(out, "w"), indent=2)
    print(f"extraction_shift: n_common={len(common)}")


M5_CANON = ["pgmm_em", "crispat_pgmm_umi0", "crispat_2beta", "fishash", "umi_threshold_t3"]


def _mismatch_one(entries, gex, gt_h5ad, guide_csv):
    """Phase-3 mismatch arbitration for ONE dataset/tool — union-NT across the
    canonical methods present (analyze_mismatch.py, verbatim). entries: {method: path}."""
    import h5py
    import anndata as ad
    EPS = 0.01
    BC_STD = re.compile(r'^([ACGT]{16})-(\d+)$')
    sym_map = "/data/yunzliu/assignment_benchmark_starter/benchmark_output/gene_symbol_to_ensg.json"
    T1 = {lab: {k: v for k, v in _first_guide(p).items()} for lab, p in entries.items()}

    s2p, p2g, sg2gene, nt = {}, {}, {}, set()
    with open(guide_csv) as g:
        g.readline()
        for line in g:
            p = line.strip().split(',')
            if len(p) < 8:
                continue
            pid, gene, sgA, sgB = p[0], p[1], p[4], p[6]
            s2p[sgA] = pid
            s2p[sgB] = pid
            p2g.setdefault(pid, gene)
            if gene == 'non-targeting':
                nt.add(sgA)
                nt.add(sgB)
            sg2gene[sgA] = gene
            sg2gene[sgB] = gene
    f = h5py.File(gt_h5ad, 'r')
    sc_, st_ = f['obs']['sgID_AB'][:], f['obs']['__categories']['sgID_AB'][:]
    cbs = f['obs']['cell_barcode'][:]
    f.close()
    gt = {}
    for i in range(len(cbs)):
        s = cbs[i].decode() if isinstance(cbs[i], bytes) else str(cbs[i])
        m = BC_STD.match(s)
        if not m:
            continue
        sg = st_[sc_[i]]
        sg = sg.decode() if isinstance(sg, bytes) else str(sg)
        sgA = sg.split('|')[0] if '|' in sg else sg
        pid = s2p.get(sgA, sgA)
        gt[(int(m.group(2)), m.group(1))] = (pid, p2g.get(pid, sgA))
    sym2ensg = json.load(open(sym_map))

    needed, mm = set(), {}
    for mth, tt in T1.items():
        cells = []
        for ck, gm in tt.items():
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
        mm[mth] = cells
    sc = ad.read_h5ad(gex)
    nc = sc.shape[0]
    e2c = {}
    for i, v in enumerate(sc.var_names):
        ss = v.decode() if isinstance(v, bytes) else str(v)
        e2c[ss.replace('_S', '')] = i
    g2c = {s: e2c[sym2ensg[s]] for s in sorted(needed) if s in sym2ensg and sym2ensg[s] in e2c}
    cols = sorted(g2c.values())
    c2l = {c: i for i, c in enumerate(cols)}
    X = sc.X[:, cols].toarray().astype(np.float32)
    cl = {}
    for i in range(nc):
        cl[(int(sc.obs['lane'].iloc[i]), str(sc.obs['barcode_16mer'].iloc[i]))] = i
    del sc
    ntc = set()
    for mth, tt in T1.items():
        for ck, g in tt.items():
            if g in nt:
                rw = cl.get(ck)
                if rw is not None:
                    ntc.add(rw)
    ntl = sorted(ntc)
    gm_ = {s: float(X[:, c2l[g2c[s]]].mean()) for s in g2c}
    nm_ = {s: float(X[ntl, :][:, c2l[g2c[s]]].mean()) for s in g2c} if ntl else dict(gm_)
    result = {}
    for mth in T1:
        n_mw = n_gw = n_tie = n_skip = 0
        for ck, gmg, Tm, gp, Tg in mm[mth]:
            rw = cl.get(ck)
            if rw is None or Tm is None or Tg is None or Tm == 'non-targeting' or Tg == 'non-targeting':
                n_skip += 1
                continue
            lm = c2l.get(g2c.get(Tm, -1))
            lg = c2l.get(g2c.get(Tg, -1))
            if lm is None or lg is None:
                n_skip += 1
                continue
            em, eg = float(X[rw, lm]), float(X[rw, lg])
            enm, eng = nm_.get(Tm, em), nm_.get(Tg, eg)
            epsm = EPS * max(gm_.get(Tm, 1e-8), 1e-8)
            epsg = EPS * max(gm_.get(Tg, 1e-8), 1e-8)
            kdm = float(np.log2(max(em + epsm, epsm)) - np.log2(max(enm + epsm, epsm)))
            kdg = float(np.log2(max(eg + epsg, epsg)) - np.log2(max(eng + epsg, epsg)))
            if kdm < kdg - 0.01:
                n_mw += 1
            elif kdg < kdm - 0.01:
                n_gw += 1
            else:
                n_tie += 1
        nv = n_mw + n_gw + n_tie
        result[mth] = {"n_mismatch": len(mm[mth]), "n_skip": n_skip, "n_valid": nv,
                       "n_method_wins": n_mw, "n_gt_wins": n_gw, "n_tie": n_tie,
                       "win_rate": round(n_mw / nv, 4) if nv else 0}
    print(f"  mismatch group: {len(result)} methods (union-NT {len(ntl)} cells)")
    return result


def collect_mismatch(entries, gex_by_ds, gt, guide_map, out):
    """Phase-3 mismatch, grouped by dataset (tool). entries: [(dataset, method, path)]."""
    by_ds = defaultdict(dict)
    for ds, m, p in entries:
        if m in CANON5:
            by_ds[ds][m] = p
    result = {}
    for ds, m2p in by_ds.items():
        gx = gex_by_ds.get(ds) or (list(gex_by_ds.values())[0] if gex_by_ds else None)
        if gx is None:
            continue
        result[ds] = _mismatch_one(m2p, gx, gt, guide_map)
    payload = result[list(result)[0]] if len(result) == 1 else result
    json.dump(payload, open(out, "w"), indent=2)
    print(f"mismatch: {len(result)} dataset(s)")


def _first_guide(path):
    sort_col, sort_desc = _sort_col(path)
    t1 = {}
    for row in csv.DictReader(open(path)):
        cell = row.get('cell', '').strip()
        guide = row.get('gRNA', '').strip()
        if not cell or not guide:
            continue
        m = BC_LANE.match(cell)
        if not m:
            continue
        k = (int(m.group(2)), m.group(1))
        s = float(row.get(sort_col, 0) or 0)
        if k not in t1 or (s < t1[k][1] if not sort_desc else s > t1[k][1]):
            t1[k] = (guide, s)
    return {k: v[0] for k, v in t1.items()}


def main():
    p = argparse.ArgumentParser(description="Omnibenchmark collector: guide_assignment_collectors")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--name", default="node")
    p.add_argument("--collector", required=True,
                   choices=["jaccard", "strat_jaccard", "extraction_shift", "mismatch"])
    p.add_argument("--gex", default=None)
    p.add_argument("--gt", default=None)
    p.add_argument("--guide_map", default=None)
    # OB injects input-id flags; accept them as aliases (fan-in / single)
    p.add_argument("--guide_assignment.assignments", action="append", default=[])
    p.add_argument("--data.gex", default=None)
    p.add_argument("--data.gt_labels", default=None)
    p.add_argument("--data.guide_map", default=None)
    p.add_argument("--data.difficulty_table", action="append", default=[])
    p.add_argument("--assignments", action="append", default=[])
    p.add_argument("--assignments_dir", action="append", default=[])
    p.add_argument("--difficulty_table", action="append", default=[])
    args, extra = p.parse_known_args()
    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, f"{args.collector}.json")

    toks = list(args.assignments) + list(getattr(args, "guide_assignment.assignments"))
    for d in args.assignments_dir:
        toks += sorted(glob.glob(os.path.join(d, "**", "assignments.csv"), recursive=True))
    toks += [t for t in extra if t.endswith(".csv") or "=" in t]
    entries = _parse_assignments(toks)            # [(dataset, method, path)]
    dt_toks = list(args.difficulty_table) + list(getattr(args, "data.difficulty_table"))
    diff_by_ds = _parse_tables(dt_toks)           # {dataset: table_path}
    # gex/gt/guide_map: fan-in gives one per dataset; gt/guide_map identical across lineages
    gex_by_ds = _parse_tables([g for g in [args.gex, getattr(args, "data.gex")] if g] +
                              [t for t in extra if t.endswith("_gex.h5ad")])
    gt = args.gt or getattr(args, "data.gt_labels")
    guide_map = args.guide_map or getattr(args, "data.guide_map")

    if args.collector == "jaccard":
        if len(entries) < 2:
            sys.exit(f"jaccard needs >=2 assignments, got {len(entries)}")
        collect_jaccard(entries, out)
    elif args.collector == "strat_jaccard":
        if not diff_by_ds or not entries:
            sys.exit("strat_jaccard needs difficulty tables and assignments")
        collect_strat_jaccard(entries, diff_by_ds, out)
    elif args.collector == "mismatch":
        if not gex_by_ds or not gt or not guide_map:
            sys.exit("mismatch needs gex + gt + guide_map")
        collect_mismatch(entries, gex_by_ds, gt, guide_map, out)
    else:
        collect_extraction_shift(diff_by_ds, out)


if __name__ == "__main__":
    main()
