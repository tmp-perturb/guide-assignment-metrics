"""
Mismatch Biology Arbitration final — Replogle 2022 HAM.
Standard anndata load (verified, same as all benchmark scripts).
Extracts only ~1925 target-gene columns → dense numpy in ONE step.
Peak RSS ~30GB for ~90s (load + extract), then 4.6GB thereafter.
Wall time ~2 min. Algorithm identical to v1 (verified correct).
"""
import csv, json, os, sys, time, re, h5py, numpy as np
from collections import defaultdict
import anndata as ad

STARTER = "/data/yunzliu/assignment_benchmark_starter"
TRUSEQ = "/data/yunzliu/Replogle2022_K562_Day6_benchmark/02_gex/post_decontx/TruSeq_decontx_merged_48lanes.h5ad"
GT_H5AD = "/data/yunzliu/references/published/K562_essential_raw_singlecell_01.h5ad"
GUIDE_CSV = "/data/yunzliu/references/raw_guides_k562_essential.csv"
MAPPING = os.path.join(STARTER, "benchmark_output/_gene_symbol_to_ensg.json")
OUT = os.path.join(STARTER, "12_kd_efficiency/replogle2022/_mismatch_arbitration.json")
OUT_TMP = os.path.join(STARTER, "benchmark_output/_mismatch_results.md")
EPS = 0.01

sys.path.insert(0, os.path.join(STARTER, "03_scripts"))
from benchmark_kd_efficiency import (load_standard, load_crispat_2beta,
    load_crispat_umi, load_fishash_topk)

M5_I = ["pgmm_em","crispat_pgmm_umi0","crispat_2beta","fishash","umi_threshold_t3"]
M5_D = ["pgmm_em","crispat_pgmm","crispat_2beta","fishash","umi_t3"]
TOOL = "ham"
ts = time.time()

# ═══ 1. GT + guide maps ═══
print("[1/5] GT + guide maps …")
f = h5py.File(GT_H5AD, 'r')
sgID_c = f['obs']['sgID_AB'][:]; sgID_t = f['obs']['__categories']['sgID_AB'][:]
cbs = f['obs']['cell_barcode'][:]; f.close()
bcre = re.compile(r'^([ACGT]{16})-(\d+)$')
s2p={}; p2g={}
with open(GUIDE_CSV) as g:
    g.readline()
    for line in g:
        p=line.strip().split(',');
        if len(p)<8: continue
        pid,gene,sgA,sgB=p[0],p[1],p[4],p[6]
        s2p[sgA]=pid; s2p[sgB]=pid
        if pid not in p2g: p2g[pid]=gene
gt={}
for i in range(len(cbs)):
    s=cbs[i].decode() if isinstance(cbs[i],bytes) else str(cbs[i])
    m=bcre.match(s)
    if not m: continue
    k=(int(m.group(2)),m.group(1))
    sgab=sgID_t[sgID_c[i]]; sgab=sgab.decode() if isinstance(sgab,bytes) else str(sgab)
    sgA=sgab.split('|')[0] if '|' in sgab else sgab
    pid=s2p.get(sgA,sgA); gene=p2g.get(pid,sgA)
    gt[k]=(pid,gene)
print(f"  {len(gt):,} GT [{time.time()-ts:.0f}s]")

sg2gene={}; nt_sgrnas=set()
with open(GUIDE_CSV) as g:
    g.readline()
    for line in g:
        p=line.strip().split(',');
        if len(p)<8: continue
        gene,sgA,sgB=p[1],p[4],p[6]
        if gene=='non-targeting':
            if sgA: nt_sgrnas.add(sgA)
            if sgB: nt_sgrnas.add(sgB)
        sg2gene[sgA]=gene; sg2gene[sgB]=gene
with open(MAPPING) as f: sym2ensg=json.load(f)

# ═══ 2. Assignments + mismatch scan ═══
print("[2/5] Assignments + mismatch scan …")
def load_t1(ci):
    if ci=="pgmm_em":
        pg=load_standard(fpath=f"{STARTER}/05_pgmm_em_assignment/{TOOL}/assignments.csv", sort_key='UMI_counts',sort_desc=True)
    elif ci=="crispat_pgmm_umi0":
        pg=load_standard(fpath=f"{STARTER}/06_pgmm_crispat/{TOOL}/UMI_0/assignments.csv", sort_key='UMI_counts',sort_desc=True)
    elif ci=="crispat_2beta":
        pg=load_crispat_2beta(fpath=f"{STARTER}/07_2beta_crispat/{TOOL}/assignments.csv")
    elif ci=="umi_threshold_t3":
        pg=load_crispat_umi(fpath=f"{STARTER}/08_umi_crispat/{TOOL}/t3/assignments.csv")
    elif ci=="fishash":
        pg=load_fishash_topk(fpath=f"{STARTER}/09_fishash/{TOOL}/assignments.csv")
    return {k:v[0][0] for k,v in pg.items() if v}

needed=set(); mm_data={}
for ci,di in zip(M5_I,M5_D):
    t1=load_t1(ci); cells=[]
    for ck,gm in t1.items():
        tv=gt.get(ck)
        if tv is None: continue
        gp,gg=tv; mp=s2p.get(gm,gm)
        if mp==gp: continue
        Tm=sg2gene.get(gm); Tg=gg
        if Tm and Tm!='non-targeting': needed.add(Tm)
        if Tg and Tg!='non-targeting': needed.add(Tg)
        cells.append((ck,gm,Tm,gp,Tg))
    mm_data[ci]=cells
    print(f"  {di}: {len(cells):,} mismatches")
print(f"  Needs {len(needed)} genes [{time.time()-ts:.0f}s]")

# ═══ 3. GEX load + extract + cell_lookup (STANDARD anndata) ═══
print("[3/5] Load GEX (standard anndata) + extract needed columns …")
sc = ad.read_h5ad(TRUSEQ)  # ← standard, same as all benchmark scripts
nc = sc.shape[0]

# Gene mapping (symbol → ENSG → column)
ensg2col={}
for i,v in enumerate(sc.var_names):
    s=v.decode('utf-8') if isinstance(v,bytes) else str(v)
    ensg2col[s.replace('_S','')]=i
gene2col={}
for sym in sorted(needed):
    es=sym2ensg.get(sym)
    if es and es in ensg2col: gene2col[sym]=ensg2col[es]
needed_cols=sorted(gene2col.values())
col2loc={c:i for i,c in enumerate(needed_cols)}

# ONE extract – then free CSC
X_small = sc.X[:, needed_cols].toarray().astype(np.float32)
print(f"  X_small: {X_small.shape} ({X_small.nbytes/1024**3:.1f} GB)")

# Cell lookup
cell_lookup={}
for i in range(nc):
    seq=str(sc.obs['barcode_16mer'].iloc[i])
    lane=int(sc.obs['lane'].iloc[i])
    cell_lookup[(lane,seq)]=i
del sc  # FREE 25GB CSC
print(f"  Cell lookup: {len(cell_lookup):,} [{time.time()-ts:.0f}s]")

# ═══ 4. NT pool + per-gene means (dense numpy) ═══
print("[4/5] NT pool + per-gene means …")
nt_cells=set()
for ci in M5_I:
    t1=load_t1(ci)
    for ck in t1:
        if t1[ck] in nt_sgrnas:
            rw=cell_lookup.get(ck)
            if rw is not None: nt_cells.add(rw)
nt_list=sorted(nt_cells)
print(f"  NT: {len(nt_list):,} cells")

gene_list=sorted(gene2col.keys())
gene_means={}
for sym in gene_list:
    gene_means[sym]=float(X_small[:, col2loc[gene2col[sym]]].mean())

nt_sub=X_small[nt_list,:]; nt_means={}
for sym in gene_list:
    nt_means[sym]=float(nt_sub[:, col2loc[gene2col[sym]]].mean())
del nt_sub
print(f"  Done [{time.time()-ts:.0f}s]")

# ═══ 5. Per-cell KD + save + compare ═══
print("[5/5] Per-cell KD …")
all_results={}
for ci,di in zip(M5_I,M5_D):
    cells=mm_data[ci]; n_skip=0; n_mw=0; n_gw=0; n_tie=0; det=[]
    for ck,gm,Tm,gp,Tg in cells:
        rw=cell_lookup.get(ck)
        if rw is None or Tm is None or Tg is None or Tm=='non-targeting' or Tg=='non-targeting':
            n_skip+=1; continue
        lm=col2loc.get(gene2col.get(Tm,-1)); lg=col2loc.get(gene2col.get(Tg,-1))
        if lm is None or lg is None: n_skip+=1; continue
        em=float(X_small[rw,lm]); eg=float(X_small[rw,lg])
        enm=nt_means.get(Tm,em); eng=nt_means.get(Tg,eg)
        epsm=EPS*max(gene_means.get(Tm,1e-8),1e-8); epsg=EPS*max(gene_means.get(Tg,1e-8),1e-8)
        kdm=float(np.log2(max(em+epsm,epsm))-np.log2(max(enm+epsm,epsm)))
        kdg=float(np.log2(max(eg+epsg,epsg))-np.log2(max(eng+epsg,epsg)))
        if   kdm<kdg-0.01:    n_mw+=1; win="method"
        elif kdg<kdm-0.01:    n_gw+=1; win="gt"
        else:                  n_tie+=1; win="tie"
        if len(det)<100: det.append({"cell":f"{ck[1]}-L{ck[0]:02d}","kd_method":round(kdm,4),"kd_gt":round(kdg,4),"winner":win})
    nm=len(cells); nv=n_mw+n_gw+n_tie; wr=n_mw/nv if nv else 0
    print(f"  {di:15s} n={nm} skip={n_skip} method={n_mw} gt={n_gw} tie={n_tie} rate={wr:.4f} [{time.time()-ts:.0f}s]")
    all_results[ci]={"display_name":di,"n_mismatch":nm,"n_skip":n_skip,"n_valid":nv,"n_method_wins":n_mw,"n_gt_wins":n_gw,"n_tie":n_tie,"win_rate":round(wr,4),"details":det}

with open(OUT,'w') as f: json.dump(all_results,f,indent=2)
print(f"  {OUT}")
md=["# Mismatch — Replogle 2022 HAM\n","| Method | Mismatches | Skipped | Valid | Method wins | GT wins | Ties | Win rate |","| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
for ci in M5_I:
    r=all_results[ci]
    md.append(f"| {r['display_name']} | {r['n_mismatch']:,} | {r['n_skip']:,} | {r['n_valid']:,} | {r['n_method_wins']:,} | {r['n_gt_wins']:,} | {r['n_tie']:,} | {r['win_rate']:.4f} |")
with open(OUT_TMP,'w') as f: f.write("\n".join(md))

v1f=os.path.join(STARTER,"archive/_mismatch_arbitration_v1_81h.json")
if os.path.exists(v1f):
    with open(v1f) as f: v1=json.load(f)
    print("\nv1 vs final:"); ok=0
    for ci in M5_I:
        r1=v1[ci]; r7=all_results[ci]; dm=r1["n_method_wins"]-r7["n_method_wins"]; dg=r1["n_gt_wins"]-r7["n_gt_wins"]
        l="OK" if abs(r1["n_mismatch"]-r7["n_mismatch"])<10 and abs(r1["win_rate"]-r7["win_rate"])<0.02 else "DIFF"
        print(f"  {r7['display_name']:15s} v1={r1['win_rate']:.4f} final={r7['win_rate']:.4f}  dm={dm:+d} dg={dg:+d}  {l}")
        if l=="OK": ok+=1
    print(f"  {ok}/5 OK" if ok==5 else f"  {ok}/5 differ")

print(f"\nDone [{time.time()-ts:.0f}s]")
