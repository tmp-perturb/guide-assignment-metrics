#!/usr/bin/env python3
"""
Fix ALL difficulty stratification phases:
1. Rebuild difficulty TSVs with extraction-specific strata (correct cutoffs per extraction)
2. Phase 1a: stratified metrics from TSV precomputed strata
3. Phase 1b: all method-pair Jaccard per stratum (HAM + simpleaf)
4. Phase 1c: mismatch localization (HAM + simpleaf)
5. Phase 1e: extraction shift
6. Phase 3: capacity calibration (HAM + simpleaf)
7. Generate properly formatted markdown with BOTH extractions everywhere
"""
import csv, json, os, re, time, h5py, numpy as np
from collections import defaultdict, Counter

ST = "/data/yunzliu/assignment_benchmark_starter"
OUT = os.path.join(ST, "benchmark_output/difficulty_stratification")
os.makedirs(OUT, exist_ok=True)
BC_LANE = re.compile(r'^([ACGT]{16})-L(\d+)$')
BC_GT   = re.compile(r'^([ACGT]{16})-(\d+)$')

SORT_CFG = {
    "pgmm_em": ("UMI_counts",True), "umi_t3": ("UMI_counts",True),
    "crispat_pgmm": ("UMI_counts",True), "crispat_2beta": ("percent_counts",True),
    "fishash": ("log_pval",False),
}
CSV_P = {
    "pgmm_em": {"ham": f"{ST}/05_pgmm_em_assignment/ham/assignments.csv",
                 "simpleaf": f"{ST}/05_pgmm_em_assignment/simpleaf_k15/assignments.csv"},
    "umi_t3": {"ham": f"{ST}/08_umi_crispat/ham/t3/assignments.csv",
               "simpleaf": f"{ST}/08_umi_crispat/simpleaf_k15/t3/assignments.csv"},
    "crispat_pgmm": {"ham": f"{ST}/06_pgmm_crispat/ham/UMI_0/assignments.csv",
                      "simpleaf": f"{ST}/06_pgmm_crispat/simpleaf_k15/UMI_0/assignments.csv"},
    "crispat_2beta": {"ham": f"{ST}/07_2beta_crispat/ham/assignments.csv",
                       "simpleaf": f"{ST}/07_2beta_crispat/simpleaf_k15/assignments.csv"},
    "fishash": {"ham": f"{ST}/09_fishash/ham/assignments.csv",
                 "simpleaf": f"{ST}/09_fishash/simpleaf_k15/assignments.csv"},
}
METHODS = list(SORT_CFG.keys())
TOOLS = ["ham","simpleaf"]
STRATA = ["easy","noise","ambig","gray"]
JACCARD_PAIRS = [("pgmm_em","umi_t3"),("pgmm_em","crispat_pgmm"),("pgmm_em","crispat_2beta"),
                 ("pgmm_em","fishash"),("umi_t3","crispat_pgmm"),("umi_t3","crispat_2beta"),
                 ("umi_t3","fishash"),("crispat_pgmm","crispat_2beta"),("crispat_pgmm","fishash"),
                 ("crispat_2beta","fishash")]

ts = time.time()
print("=" * 60)
print("Difficulty Fix-All — extraction-specific throughout")
print("=" * 60)

# ══════════════════════════════════════════════════════════
# 1. Rebuild difficulty TSVs with extraction-specific strata
# ══════════════════════════════════════════════════════════
print("\n[1/7] Rebuilding difficulty TSVs with correct strata …")
all_data = {}
for tool in TOOLS:
    tsv_path = os.path.join(OUT, f"cell_difficulty_{tool}.tsv")
    rows = []
    with open(tsv_path) as f:
        for row in csv.DictReader(f, delimiter='\t'):
            rows.append(row)
    ents = np.array([float(r['entropy_lib']) for r in rows])
    dlts = np.array([float(r['delta']) for r in rows])
    libs = np.array([float(r['libsize_pctl_in_lane']) for r in rows])
    # EXTRACTION-SPECIFIC tertiles
    ent_t = [float(np.percentile(ents,33.33)), float(np.percentile(ents,66.67))]
    dlt_t = [float(np.percentile(dlts,33.33)), float(np.percentile(dlts,66.67))]
    n = len(rows)
    h_arr = np.array(["gray"]*n, dtype=object)
    for i in range(n):
        e=ents[i]; d=dlts[i]; lp=libs[i]
        if lp>50 and e<ent_t[0]: h_arr[i]='easy'
        elif lp<50 and e>ent_t[1]: h_arr[i]='noise'
        elif lp>50 and e>ent_t[0] and d<dlt_t[0]: h_arr[i]='ambig'
        else: h_arr[i]='gray'
    # Write back with corrected strata
    for i in range(n): rows[i]['stratum_hard'] = h_arr[i]
    with open(tsv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys(), delimiter='\t')
        w.writeheader(); w.writerows(rows)
    # Build key sets
    hard_sets = {}
    for st in STRATA:
        idx = np.where(h_arr==st)[0]
        c = rows[0]['cell_id']; m = BC_LANE.match(c)
        hard_sets[st] = set()
        for i in idx:
            m2 = BC_LANE.match(rows[i]['cell_id'])
            if m2: hard_sets[st].add((int(m2.group(2)), m2.group(1)))
    all_data[tool] = {"rows": rows, "strata": h_arr, "hard_sets": hard_sets,
                       "ent_t": ent_t, "dlt_t": dlt_t,
                       "ents": ents, "dlts": dlts, "libs": libs,
                       "perps": np.array([float(r['perplexity']) for r in rows]),
                       "k80s": np.array([int(r['k80']) for r in rows])}
    hc = Counter(h_arr)
    print(f"  {tool}: easy={hc['easy']:,} noise={hc['noise']:,} ambig={hc['ambig']:,} gray={hc['gray']:,}")
    print(f"    cutoffs: ent_t=[{ent_t[0]:.4f},{ent_t[1]:.4f}] dlt_t=[{dlt_t[0]:.4f},{dlt_t[1]:.4f}]")

# ══════════════════════════════════════════════════════════
# 2. GT
# ══════════════════════════════════════════════════════════
print("\n[2/7] Loading GT …")
gt_keys = set(); gt_construct = {}; gt_set_d = {}
GUIDE_CSV = "/data/yunzliu/references/raw_guides_k562_essential.csv"
sgid_to_c = {}
with open(GUIDE_CSV) as f:
    for row in csv.DictReader(f):
        pid = row['unique sgRNA pair ID'].strip()
        for sg_col in ['sgID_A','sgID_B']:
            sg = row[sg_col].strip()
            if sg: sgid_to_c[sg] = pid
            if ',ENST' in sg: sgid_to_c[sg.replace(',ENST','_ENST')] = pid
f = h5py.File("/data/yunzliu/references/published/K562_essential_raw_singlecell_01.h5ad",'r')
cbs = f['obs']['cell_barcode'][:]; sg_codes = f['obs']['sgID_AB'][:]
sg_cats = f['obs']['__categories']['sgID_AB'][:]; f.close()
for i in range(len(cbs)):
    cb = cbs[i].decode() if isinstance(cbs[i],bytes) else str(cbs[i])
    m = BC_GT.match(cb)
    if not m: continue
    k = (int(m.group(2)), m.group(1)); gt_keys.add(k)
    sgab = sg_cats[sg_codes[i]]; sgab = sgab.decode() if isinstance(sgab,bytes) else str(sgab)
    parts = sgab.split('|')
    if len(parts)==2:
        pid = sgid_to_c.get(parts[0])
        if pid: gt_construct[k] = pid; gt_set_d[k] = {parts[0], parts[1]}
print(f"  GT: {len(gt_keys):,}")

# ══════════════════════════════════════════════════════════
# 3. Phase 1a — Stratified metrics
# ══════════════════════════════════════════════════════════
print("\n[3/7] Phase 1a — Stratified metrics …")
all_metrics = {}
for tool in TOOLS:
    for method in METHODS:
        csv_path = CSV_P[method][tool]; sort_col, sort_desc = SORT_CFG[method]
        if not os.path.exists(csv_path): continue
        top1 = {}; ag = {}
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                cell = row.get('cell','').strip(); guide = row.get('gRNA','').strip()
                if not cell or not guide: continue
                m = BC_LANE.match(cell)
                if not m: continue
                k = (int(m.group(2)), m.group(1))
                if sort_col == 'log_pval':
                    s = float(row.get(sort_col,0) or 0)
                    if k not in top1 or s < top1[k][1]: top1[k] = (guide, s)
                elif sort_col == 'percent_counts':
                    s = float(row.get(sort_col,0) or 0)
                    if k not in top1 or s > top1[k][1]: top1[k] = (guide, s)
                else:
                    s = int(float(row.get('UMI_counts',0) or 0))
                    if k not in top1 or s > top1[k][1]: top1[k] = (guide, s)
                if k not in ag: ag[k] = set()
                ag[k].add(guide)
        for k in top1: top1[k] = top1[k][0]
        for st in STRATA:
            shared = all_data[tool]["hard_sets"][st] & gt_keys
            n_gt = len(shared)
            if n_gt < 10: continue
            n_t1 = 0; n_assn = 0; nb = no = nz = 0; sp = sr = sf = ne = 0.0
            for k in shared:
                g1 = top1.get(k)
                if g1 is None: continue
                n_assn += 1
                if sgid_to_c.get(g1,g1) == gt_construct.get(k): n_t1 += 1
                mg = ag.get(k,set()); gs = gt_set_d.get(k,set()); ni = len(mg & gs)
                if ni == 2: nb += 1
                elif ni == 1: no += 1
                else: nz += 1
                nm = len(mg)
                if nm > 0:
                    p = ni/nm; r = ni/2; sp += p; sr += r
                    if p+r > 0: sf += 2*p*r/(p+r); ne += 1
            nt = nb+no+nz
            rec = n_assn/max(n_gt,1)
            t1_acc = n_t1/max(n_assn,1) if n_assn>0 else 0
            all_metrics[f"{method}__{tool}__{st}"] = {
                "method":method,"tool":tool,"stratum":st,"n_gt":n_gt,"n_assigned":n_assn,
                "rec":round(rec,6),"t1":round(t1_acc,6),"eff_t1":round(rec*t1_acc,6),
                "p_both":round(nb/max(nt,1),6),"p_one":round(no/max(nt,1),6),"p_zero":round(nz/max(nt,1),6),
                "set_prec":round(sp/max(ne,1),6),"set_rec":round(sr/max(ne,1),6),"set_f1":round(sf/max(ne,1),6)}
        del top1, ag
    print(f"  {tool}: {len([k for k in all_metrics if tool in k])} entries")
with open(os.path.join(OUT,"phase1a_stratum_metrics.json"),'w') as f: json.dump(all_metrics,f,indent=2)

# ══════════════════════════════════════════════════════════
# 4. Phase 1b — All-pair Jaccard per stratum
# ══════════════════════════════════════════════════════════
print("\n[4/7] Phase 1b — All-pair Jaccard per stratum …")
jaccard_stratum = {}
for tool in TOOLS:
    jaccard_stratum[tool] = {}
    top1s = {}
    for method in METHODS:
        sort_col, sort_desc = SORT_CFG[method]
        t1 = {}
        with open(CSV_P[method][tool]) as f:
            for row in csv.DictReader(f):
                cell = row.get('cell','').strip(); guide = row.get('gRNA','').strip()
                if not cell or not guide: continue
                m = BC_LANE.match(cell)
                if not m: continue
                k = (int(m.group(2)), m.group(1))
                if sort_col == 'log_pval':
                    s = float(row.get(sort_col,0) or 0)
                    if k not in t1 or s < t1[k][1]: t1[k] = (guide, s)
                elif sort_col == 'percent_counts':
                    s = float(row.get(sort_col,0) or 0)
                    if k not in t1 or s > t1[k][1]: t1[k] = (guide, s)
                else:
                    s = int(float(row.get('UMI_counts',0) or 0))
                    if k not in t1 or s > t1[k][1]: t1[k] = (guide, s)
        for k in t1: t1[k] = t1[k][0]
        top1s[method] = t1
    key_sets_m = {m: set(t1s.keys()) for m, t1s in top1s.items()}
    for st in STRATA:
        st_set = all_data[tool]["hard_sets"][st]
        pair_results = {}
        for mi, mj in JACCARD_PAIRS:
            sc = key_sets_m[mi] & key_sets_m[mj] & st_set
            if len(sc) < 5: continue
            agree = sum(1 for c in sc if top1s[mi].get(c) == top1s[mj].get(c))
            pair_results[f"{mi}_{mj}"] = {"J": round(agree/len(sc),6), "n": len(sc)}
        jaccard_stratum[tool][st] = pair_results
with open(os.path.join(OUT,"phase1b_jaccard_stratum.json"),'w') as f: json.dump(jaccard_stratum,f,indent=2)

# ══════════════════════════════════════════════════════════
# 5. Phase 1c — Mismatch localization (BOTH extractions)
# ══════════════════════════════════════════════════════════
print("\n[5/7] Phase 1c — Mismatch localization …")
mismatch_loc = {}
for tool in TOOLS:
    h = all_data[tool]["strata"]
    rows = all_data[tool]["rows"]
    key_to_idx = {}
    for i,r in enumerate(rows):
        m = BC_LANE.match(r['cell_id'])
        k = (int(m.group(2)), m.group(1)) if m else (0, r['cell_id'])
        key_to_idx[k] = i
    for method in METHODS:
        sort_col, sort_desc = SORT_CFG[method]
        t1 = {}
        with open(CSV_P[method][tool]) as f:
            for row in csv.DictReader(f):
                cell = row.get('cell','').strip(); guide = row.get('gRNA','').strip()
                if not cell or not guide: continue
                m = BC_LANE.match(cell)
                if not m: continue
                k = (int(m.group(2)), m.group(1))
                if sort_col == 'log_pval':
                    s = float(row.get(sort_col,0) or 0)
                    if k not in t1 or s < t1[k][1]: t1[k] = (guide, s)
                elif sort_col == 'percent_counts':
                    s = float(row.get(sort_col,0) or 0)
                    if k not in t1 or s > t1[k][1]: t1[k] = (guide, s)
                else:
                    s = int(float(row.get('UMI_counts',0) or 0))
                    if k not in t1 or s > t1[k][1]: t1[k] = (guide, s)
        for k in t1: t1[k] = t1[k][0]
        mm_c = []; corr_c = []
        for k in gt_keys:
            g = t1.get(k)
            if g is None: continue
            if sgid_to_c.get(g,g) == gt_construct.get(k): corr_c.append(k)
            else: mm_c.append(k)
        mm_s = Counter()
        for k in mm_c:
            si = key_to_idx.get(k)
            if si is not None: mm_s[h[si]] += 1
        total = len(mm_c)
        mismatch_loc[f"{method}__{tool}"] = {
            "method":method,"tool":tool,"n_mismatch":total,"n_correct":len(corr_c),
            "mm_pct": {s: round(mm_s.get(s,0)/max(total,1)*100,1) for s in STRATA}}
with open(os.path.join(OUT,"phase1c_mismatch_loc.json"),'w') as f: json.dump({"mismatch_localization":mismatch_loc},f,indent=2)

# ══════════════════════════════════════════════════════════
# 6. Phase 1e — Extraction shift
# ══════════════════════════════════════════════════════════
print("\n[6/7] Phase 1e — Extraction shift …")
d_ham = all_data["ham"]; d_sim = all_data["simpleaf"]
key_h = {}
for i,r in enumerate(d_ham["rows"]):
    m = BC_LANE.match(r['cell_id'])
    key_h[(int(m.group(2)),m.group(1)) if m else (0,r['cell_id'])] = i
key_s = {}
for i,r in enumerate(d_sim["rows"]):
    m = BC_LANE.match(r['cell_id'])
    key_s[(int(m.group(2)),m.group(1)) if m else (0,r['cell_id'])] = i
common = set(key_h.keys()) & set(key_s.keys())
shifts_d = []; shifts_e = []
for k in common:
    shifts_d.append(float(d_sim["rows"][key_s[k]]["delta"]) - float(d_ham["rows"][key_h[k]]["delta"]))
    shifts_e.append(float(d_sim["rows"][key_s[k]]["entropy_lib"]) - float(d_ham["rows"][key_h[k]]["entropy_lib"]))
extraction_shift = {
    "n_common": len(common),
    "delta_shift_median": round(float(np.median(shifts_d)),4),
    "entropy_shift_median": round(float(np.median(shifts_e)),4),
}
with open(os.path.join(OUT,"phase1e_extraction_shift.json"),'w') as f: json.dump(extraction_shift,f,indent=2)

# ══════════════════════════════════════════════════════════
# 7. Phase 3 — Capacity calibration (BOTH extractions)
# ══════════════════════════════════════════════════════════
print("\n[7/7] Phase 3 — Capacity calibration …")
capacity = {}
for tool in TOOLS:
    h = all_data[tool]["strata"]; rows = all_data[tool]["rows"]
    ki = {}
    for i,r in enumerate(rows):
        m = BC_LANE.match(r['cell_id'])
        ki[(int(m.group(2)),m.group(1)) if m else (0,r['cell_id'])] = i
    for method in METHODS:
        gpc = {}
        with open(CSV_P[method][tool]) as f:
            for row in csv.DictReader(f):
                cell = row.get('cell','').strip(); guide = row.get('gRNA','').strip()
                if not cell or not guide: continue
                m = BC_LANE.match(cell)
                if not m: continue
                k = (int(m.group(2)), m.group(1))
                gpc[k] = gpc.get(k,0) + 1
        hi_k80 = []; hi_gpc = []
        for k, n in gpc.items():
            si = ki.get(k)
            if si is None: continue
            if h[si] in ('easy','ambig'):
                hi_k80.append(int(all_data[tool]["k80s"][si]))
                hi_gpc.append(n)
        slope = 0; intercept = 0
        if len(hi_k80) > 10:
            Xk = np.column_stack([hi_k80, np.ones(len(hi_k80))])
            beta, _, _, _ = np.linalg.lstsq(Xk, hi_gpc, rcond=None)
            slope = round(float(beta[0]),3); intercept = round(float(beta[1]),2)
        ag_all = np.array(list(gpc.values()))
        capacity[f"{method}__{tool}"] = {
            "mean_gpc": round(float(ag_all.mean()),2), "n_cells": len(gpc),
            "slope_k80": slope, "intercept": intercept,
        }
with open(os.path.join(OUT,"phase3_capacity.json"),'w') as f: json.dump(capacity,f,indent=2)

# ══════════════════════════════════════════════════════════
# 8. Generate properly formatted markdown
# ══════════════════════════════════════════════════════════
print("\n[8] Generating markdown …")
gt_comp = {}
for tool in TOOLS:
    h = all_data[tool]["strata"]; rows = all_data[tool]["rows"]
    gt_in = Counter()
    for i,r in enumerate(rows):
        m = BC_LANE.match(r['cell_id'])
        k = (int(m.group(2)),m.group(1)) if m else (0,r['cell_id'])
        if k in gt_keys: gt_in[h[i]] += 1
    gt_comp[tool] = {s: gt_in.get(s,0) for s in STRATA}

md = []
md.append("# Cell-Level Difficulty Stratification — Replogle 2022\n")
md.append(f"**Generated:** 2026-08-06 (extraction-specific cutoffs throughout)  \n\n")

md.append("## 0. Data Files\n")
md.append(f"All files in `benchmark_output/difficulty_stratification/`:  \n")
for fn in sorted(os.listdir(OUT)):
    if not fn.startswith('.'): md.append(f"- `{fn}`\n")
md.append("\n")

# --- Section 1: Phase 0 ---
md.append("## 1. Phase 0 — Per-Cell Difficulty Table\n\n")
md.append("**Algorithm.** For each cell in the raw guide count matrix (CSR sparse):\n\n")
md.append("```\n")
md.append("libsize      = Σ UMI\n")
md.append("n_detected   = |{g : UMI_g > 0}|\n")
md.append("delta        = (top1 − top2) / libsize\n")
md.append("H            = −Σ p_i · log₂(p_i)\n")
md.append("entropy_lib  = H / log₂(N_guides_in_library)\n")
md.append("entropy_det  = H / log₂(max(n_detected, 2))\n")
md.append("perplexity   = 2^H\n")
md.append("k80          = min k s.t. Σ_{i=1..k} UMI_{(i)} ≥ 0.80 × libsize\n")
md.append("```\n\n")

md.append("### Summary statistics\n\n")
md.append("| Metric | HAM-p25 | HAM-p50 | HAM-p75 | simpleaf-p25 | simpleaf-p50 | simpleaf-p75 |\n")
md.append("|---|---:|---:|---:|---:|---:|---:|\n")
for metric, ham_fn, sim_fn in [
    ("libsize", lambda d: d["libs"], lambda d: d["libs"]),
    ("delta", lambda d: d["dlts"], lambda d: d["dlts"]),
    ("entropy_lib", lambda d: d["ents"], lambda d: d["ents"]),
    ("perplexity", lambda d: d["perps"], lambda d: d["perps"]),
    ("k80", lambda d: d["k80s"], lambda d: d["k80s"]),
]:
    hv = ham_fn(all_data["ham"]); sv = sim_fn(all_data["simpleaf"])
    md.append(f"| {metric} | {np.percentile(hv,25):.1f} | {np.percentile(hv,50):.1f} | {np.percentile(hv,75):.1f} | {np.percentile(sv,25):.1f} | {np.percentile(sv,50):.1f} | {np.percentile(sv,75):.1f} |\n")
md.append("\n")

# --- Section 2: Stratum Definitions ---
md.append("## 2. Phase 1A — Stratum Definitions\n\n")
md.append("**Extraction-specific cutoffs** (tertiles computed per extraction):\n\n")
md.append("| Extraction | ent_lower | ent_upper | dlt_lower | dlt_upper | easy | noise | ambig | gray |\n")
md.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
for tool in TOOLS:
    d = all_data[tool]; hc = Counter(d["strata"])
    md.append(f"| {tool} | {d['ent_t'][0]:.4f} | {d['ent_t'][1]:.4f} | {d['dlt_t'][0]:.4f} | {d['dlt_t'][1]:.4f} | {hc['easy']:,} | {hc['noise']:,} | {hc['ambig']:,} | {hc['gray']:,} |\n")

md.append(f"\n**stratum_hard rules** (identical logic per extraction, different cutoffs):\n\n")
md.append("| Stratum | Rule |\n|---|---|\n")
md.append("| `easy` | libsize_pct > 50 AND entropy_lib < lower_tertile |\n")
md.append("| `noise` | libsize_pct < 50 AND entropy_lib > upper_tertile |\n")
md.append("| `ambig` | libsize_pct > 50 AND entropy_lib > lower_tertile AND delta < lower_tertile |\n")
md.append("| `gray` | everything else |\n\n")

# GT composition per stratum
md.append("### GT composition per stratum\n\n")
md.append("| Extraction | Stratum | All cells | GT cells | GT% |\n")
md.append("|---|---:|---:|---:|\n")
for tool in TOOLS:
    hc = Counter(all_data[tool]["strata"])
    for st in STRATA:
        n_all = hc[st]; n_gt = gt_comp[tool][st]
        md.append(f"| {tool} | {st} | {n_all:,} | {n_gt:,} | {n_gt/max(n_all,1)*100:.1f}% |\n")
md.append("\n")

# --- Section 3: Stratified Metrics ---
md.append("## 3. Phase 1C — Stratified Tier-1 Metrics\n\n")
for tool in TOOLS:
    md.append(f"### {tool}\n\n")
    md.append("| Method | Stratum | n_GT | Rec | T1 | EffT1 | Set F1 |\n")
    md.append("|---|---:|---:|---:|---:|---:|\n")
    for method in METHODS:
        for st in STRATA:
            k = f"{method}__{tool}__{st}"
            if k in all_metrics:
                r = all_metrics[k]
                md.append(f"| {method} | {st} | {r['n_gt']:,} | {r['rec']:.4f} | {r['t1']:.4f} | {r['eff_t1']:.4f} | {r['set_f1']:.4f} |\n")
    md.append("\n")

# --- Section 4: Jaccard ---
md.append("## 4. Phase 1C — Cross-Method Jaccard per Stratum\n\n")
for tool in TOOLS:
    md.append(f"### {tool}\n\n")
    md.append("| Pair | easy | noise | ambig | gray |\n")
    md.append("|---|---:|---:|---:|---:|\n")
    for mi, mj in JACCARD_PAIRS:
        pk = f"{mi}_{mj}"
        vals = [f"{jaccard_stratum[tool][st].get(pk,{}).get('J',0):.4f}" if pk in jaccard_stratum[tool][st] else "—" for st in STRATA]
        md.append(f"| {mi}–{mj} | " + " | ".join(vals) + " |\n")
    md.append("\n")

# --- Section 5: Mismatch ---
md.append("## 5. Phase 1D — Mismatch Cell Localization\n\n")
for tool in TOOLS:
    md.append(f"### {tool}\n\n")
    md.append("| Method | n | easy% | noise% | ambig% | gray% |\n")
    md.append("|---|---:|---:|---:|---:|---:|\n")
    for method in METHODS:
        k = f"{method}__{tool}"
        if k in mismatch_loc:
            r = mismatch_loc[k]
            md.append(f"| {method} | {r['n_mismatch']:,} | {r['mm_pct']['easy']:.0f}% | {r['mm_pct']['noise']:.0f}% | {r['mm_pct']['ambig']:.0f}% | {r['mm_pct']['gray']:.0f}% |\n")
    md.append("\n")

# --- Section 6: Extraction shift ---
md.append("## 6. Phase 1E — Extraction Shift\n\n")
md.append(f"- Common cells: {extraction_shift['n_common']:,}  \n")
md.append(f"- delta shift median: {extraction_shift['delta_shift_median']}  \n")
md.append(f"- entropy shift median: {extraction_shift['entropy_shift_median']}  \n\n")

# --- Section 7: Delta-KD ---
md.append("## 7. Phase 2 — Delta-KD Validation (HAM)\n\n")
d2 = json.load(open(os.path.join(OUT,"phase2_delta_kd.json")))
md.append("| Decile | n (targeting) | KD median (targeting) | KD median (NT control) |\n")
md.append("|---|---:|---:|---:|\n")
for di in range(10):
    k = f"D{di}"; t = d2["target"][k]; nt = d2.get("nt_control",{}).get(k,{})
    md.append(f"| D{di} | {t['n']:,} | {t['kd_median']:.4f} | {nt.get('kd_median',0):.4f} |\n")

# Phase 2 simpleaf
if os.path.exists(os.path.join(OUT,"phase2_delta_kd_simpleaf.json")):
    d2s = json.load(open(os.path.join(OUT,"phase2_delta_kd_simpleaf.json")))
    md.append("\n### simpleaf\n\n")
    md.append("| Decile | n (targeting) | KD median (targeting) |\n")
    md.append("|---|---:|---:|\n")
    for di in range(10):
        k = f"D{di}"; t = d2s["target"][k]
        md.append(f"| D{di} | {t['n']:,} | {t['kd_median']:.4f} |\n")

# --- Section 8: Capacity ---
md.append("\n## 8. Phase 3 — Capacity Calibration\n\n")
md.append("| Method | Tool | Mean gpC | Slope (k80) | Intercept |\n")
md.append("|---|---:|---:|---:|\n")
for k, r in capacity.items():
    parts = k.split("__"); md.append(f"| {parts[0]} | {parts[1]} | {r['mean_gpc']} | {r['slope_k80']} | {r['intercept']} |\n")

out_md = os.path.join(OUT, "new_dimension_results.md")
with open(out_md, 'w') as f: f.write("\n".join(md))

print(f"\n{'='*60}")
print(f"Done [{time.time()-ts:.0f}s]")
print(f"All outputs → {OUT}/")
print(f"Markdown  → {out_md}")
