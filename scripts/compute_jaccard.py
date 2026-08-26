"""
Compute cross-method guide-assignment Jaccard similarity.

Jaccard (guide-assignment): For each pair (A, B) of method×tool combinations:
  J(A,B) = |{c in S_A ∩ S_B : top_guide_A(c) == top_guide_B(c)}| / |S_A ∩ S_B|

where S_A = cells with a top-1 assignment from method A.

This replaces the incorrect cell-set Jaccard that only checked whether the
same cells were assigned (ignoring which guide was assigned).
"""
import csv, json, os, itertools

# ── Replogle 2022 ──────────────────────────────────────────────────────

BASE_R = "/data/yunzliu/assignment_benchmark_starter"

REPLOGLE_SOURCES = {
    # method_label -> {tool: csv_path}
    "pgmm_em": {
        "cellranger":     f"{BASE_R}/05_pgmm_em_assignment/cellranger/assignments.csv",
        "ham":            f"{BASE_R}/05_pgmm_em_assignment/ham/assignments.csv",
        "simpleaf_k15":   f"{BASE_R}/05_pgmm_em_assignment/simpleaf_k15/assignments.csv",
    },
    "crispat_pgmm_umi0": {
        "cellranger":     f"{BASE_R}/06_pgmm_crispat/cellranger/UMI_0/assignments.csv",
        "ham":            f"{BASE_R}/06_pgmm_crispat/ham/UMI_0/assignments.csv",
        "simpleaf_k15":   f"{BASE_R}/06_pgmm_crispat/simpleaf_k15/UMI_0/assignments.csv",
    },
    "crispat_pgmm_umi3": {
        "cellranger":     f"{BASE_R}/06_pgmm_crispat/cellranger/UMI_3/assignments.csv",
        "ham":            f"{BASE_R}/06_pgmm_crispat/ham/UMI_3/assignments.csv",
        "simpleaf_k15":   f"{BASE_R}/06_pgmm_crispat/simpleaf_k15/UMI_3/assignments.csv",
    },
    "crispat_2beta": {
        "cellranger":     f"{BASE_R}/07_2beta_crispat/cellranger/assignments.csv",
        "ham":            f"{BASE_R}/07_2beta_crispat/ham/assignments.csv",
        "simpleaf_k15":   f"{BASE_R}/07_2beta_crispat/simpleaf_k15/assignments.csv",
    },
    "umi_threshold_t3": {
        "cellranger":     f"{BASE_R}/08_umi_crispat/cellranger/t3/assignments.csv",
        "ham":            f"{BASE_R}/08_umi_crispat/ham/t3/assignments.csv",
        "simpleaf_k15":   f"{BASE_R}/08_umi_crispat/simpleaf_k15/t3/assignments.csv",
    },
    "umi_threshold_t5": {
        "cellranger":     f"{BASE_R}/08_umi_crispat/cellranger/t5/assignments.csv",
        "ham":            f"{BASE_R}/08_umi_crispat/ham/t5/assignments.csv",
        "simpleaf_k15":   f"{BASE_R}/08_umi_crispat/simpleaf_k15/t5/assignments.csv",
    },
    "umi_threshold_t10": {
        "cellranger":     f"{BASE_R}/08_umi_crispat/cellranger/t10/assignments.csv",
        "ham":            f"{BASE_R}/08_umi_crispat/ham/t10/assignments.csv",
        "simpleaf_k15":   f"{BASE_R}/08_umi_crispat/simpleaf_k15/t10/assignments.csv",
    },
    "fishash": {
        "cellranger":     f"{BASE_R}/09_fishash/cellranger/assignments.csv",
        "ham":            f"{BASE_R}/09_fishash/ham/assignments.csv",
        "simpleaf_k15":   f"{BASE_R}/09_fishash/simpleaf_k15/assignments.csv",
    },
}

# ── Papalexi 2021 ──────────────────────────────────────────────────────

BASE_P = f"{BASE_R}/11_papalexi_benchmark/02_results"

PAPALEXI_SOURCES = {
    "pgmm_em": {
        "ham":      f"{BASE_P}/pgmm_em/ham/assignments.csv",
        "simpleaf": f"{BASE_P}/pgmm_em/simpleaf/assignments.csv",
    },
    "crispat_pgmm_umi0": {
        "ham":      f"{BASE_P}/crispat_pgmm/UMI_0/ham/assignments.csv",
        "simpleaf": f"{BASE_P}/crispat_pgmm/UMI_0/simpleaf/assignments.csv",
    },
    "crispat_pgmm_umi3": {
        "ham":      f"{BASE_P}/crispat_pgmm/UMI_3/ham/assignments.csv",
        "simpleaf": f"{BASE_P}/crispat_pgmm/UMI_3/simpleaf/assignments.csv",
    },
    "crispat_2beta": {
        "ham":      f"{BASE_P}/crispat_2beta/ham/assignments.csv",
        "simpleaf": f"{BASE_P}/crispat_2beta/simpleaf/assignments.csv",
    },
    "umi_threshold_t3": {
        "ham":      f"{BASE_P}/crispat_umi/ham/t3/assignments.csv",
        "simpleaf": f"{BASE_P}/crispat_umi/simpleaf/t3/assignments.csv",
    },
    "umi_threshold_t5": {
        "ham":      f"{BASE_P}/crispat_umi/ham/t5/assignments.csv",
        "simpleaf": f"{BASE_P}/crispat_umi/simpleaf/t5/assignments.csv",
    },
    "umi_threshold_t10": {
        "ham":      f"{BASE_P}/crispat_umi/ham/t10/assignments.csv",
        "simpleaf": f"{BASE_P}/crispat_umi/simpleaf/t10/assignments.csv",
    },
    "fishash": {
        "ham":      f"{BASE_P}/fishash/ham/assignments.csv",
        "simpleaf": f"{BASE_P}/fishash/simpleaf/assignments.csv",
    },
}


def load_top1_guide(path, method_key):
    """Return {cell_barcode: top1_guide_string} using method-specific sort key.

    ALL methods must aggregate per cell then sort — CSV rows are NOT globally
    sorted per cell after multiprocessing chunk concatenation.
    """
    from collections import defaultdict

    if 'fishash' in method_key:
        sort_col, sort_desc = 'log_pval', False   # ASC
    elif '2beta' in method_key:
        sort_col, sort_desc = 'percent_counts', True
    else:
        sort_col, sort_desc = 'UMI_counts', True

    per_cell = defaultdict(list)
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cell = row.get("cell", "").strip()
            guide = row.get("gRNA", row.get("guide", "")).strip()
            if not cell or not guide:
                continue
            score = float(row.get(sort_col, 0) or 0)
            per_cell[cell].append((score, guide))

    result = {}
    for cell, guides in per_cell.items():
        guides.sort(key=lambda x: x[0], reverse=sort_desc)
        result[cell] = guides[0][1]
    return result


def compute_jaccard_matrix(sources):
    """Compute guide-assignment Jaccard for all method×tool combos.

    sources: {method_label: {tool: csv_path}}

    Returns: {"labels": [...], "jaccard": [[...], ...]}
    """
    # Build ordered list of (label, csv_path)
    entries = []  # [(label, path)]
    for method in sorted(sources):
        for tool in sorted(sources[method]):
            label = f"{method}__{tool}"
            entries.append((label, sources[method][tool]))

    n = len(entries)
    print(f"  Loading {n} assignment files...")
    assignments = {}  # label -> {cell: top1_guide}
    for label, path in entries:
        assignments[label] = load_top1_guide(path, label)
        print(f"    {label}: {len(assignments[label])} cells")

    # Compute pairwise Jaccard
    matrix = [[1.0] * n for _ in range(n)]
    labels = [e[0] for e in entries]

    total_pairs = n * (n - 1) // 2
    done = 0
    for i in range(n):
        for j in range(i + 1, n):
            a = assignments[labels[i]]
            b = assignments[labels[j]]
            shared = set(a.keys()) & set(b.keys())
            if not shared:
                matrix[i][j] = matrix[j][i] = 0.0
            else:
                agree = sum(1 for c in shared if a[c] == b[c])
                jac = agree / len(shared)
                matrix[i][j] = matrix[j][i] = round(jac, 6)
            done += 1
            if done % 50 == 0:
                print(f"    {done}/{total_pairs} pairs done")

    return {"labels": labels, "jaccard": matrix}


# ── Main ────────────────────────────────────────────────────────────────

for dataset, sources, out_dir in [
    ("replogle2022", REPLOGLE_SOURCES,
     f"{BASE_R}/12_kd_efficiency/replogle2022/discovery"),
    ("papalexi2021", PAPALEXI_SOURCES,
     f"{BASE_R}/12_kd_efficiency/papalexi2021/discovery"),
]:
    print(f"\n{'='*60}")
    print(f"Computing guide-assignment Jaccard: {dataset}")
    print(f"{'='*60}")

    result = compute_jaccard_matrix(sources)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "_jaccard.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {out_path}")

    # Quick summary
    mat = result["jaccard"]
    labels = result["labels"]
    off_diag = [mat[i][j] for i in range(len(mat)) for j in range(i + 1, len(mat))]
    print(f"  Off-diagonal range: [{min(off_diag):.4f}, {max(off_diag):.4f}]")
    print(f"  Off-diagonal mean:  {sum(off_diag)/len(off_diag):.4f}")

print("\nDone.")
