"""python select_ds.py [--cand CSV] [--h5ad-dir DIR | --h5ad FILE --dataset ID] --out DIR [--only ID,ID] : R1 eligibility and (coarse, fine) pair per dataset, METADATA ONLY (backed h5ad obs read).
Rules (PLAN_1012 R1): human; fine/coarse columns picked mechanically (no method output); >=97% of each fine type inside one coarse type; coarse has fewer categories than fine;
pair = most whole-dataset target pairs, ties -> fewer coarse categories -> alphabetical (coarse, fine); >=3 targets, >=8,000 cells after drops, >=2 donors.
Amendment 2026-10-12 (before any R1 method output; decided from kidney metadata alone, applies to all datasets): NOTLAB also excludes model-prediction columns (label transfer, Azimuth, CellTypist, scANVI, scOMM, confidence/score) and cluster-id columns by name, because a prediction is not curated truth and favours the arm it came from. NOTLAB already goes beyond the PLAN list (library, institute, preservation, age, is_*, reported, sequenced, ...) and DROP now matches substrings (doublet, low quality) anywhere in a label.
Amendment 2 2026-10-12 ~20:45 EDT (column names only, before any R1 method output on that dataset): 72a69193 got (patient_group, region), which are not cell-type columns.
Label columns must now also have a cell-type-annotation name (CTLAB below; cell_type is exempt). Applies to all datasets; the selection was rerun and every change is logged in OUT/selection_amend2.txt.
Optional OUT/not_zero_shot.csv (dataset_id,reason) overrides the zero-shot flag (study-level overlap the id lists miss).
Writes OUT/selection.csv (one row per dataset) and OUT/<dataset_id>/selection.json (chosen pair, all valid pairs, donors, zero-shot status)."""
import os, re, sys, json, argparse, glob, numpy as np, pandas as pd
HOME = os.path.expanduser("~"); BAD = {"None", "nan", "Unknown"}
DROP = re.compile(r"^(|unknown|unassigned|unclassified|unannotated|undefined|na|n/a|nan|none|empty|-)$|doublet|low[ _.-]?quality", re.I)   # labels dropped first
# columns that are never labels: spec list (donor, sample, batch, assay, sex, tissue, disease, development stage, ethnicity) + ontology ids + CELLxGENE technical columns
NOTLAB = re.compile(r"_ontology_term_id$|donor|sample|batch|assay|^sex|tissue|disease|development|ethnic|library|suspension|software|reference_genome|gene_annotation|preservation|institute|primary_data|organism|intronic|sequenced|^age|^is_|reported|predict|pred_|transfer|scomm|azimuth|celltypist|singler|scanvi|popv|majority_voting|confidence|_score|_prob|cluster|leiden|louvain|seurat_clusters", re.I)
CTLAB = re.compile(r"cell.?type|annot|class|type|lineage|subset|sub(?!ject)|major|minor|level|(^|_)l\d", re.I)   # amendment 2: label columns must be named as cell-type annotations
NUMLIKE = re.compile(r"^-?\d+(\.\d+)?$")

def targets(df):
    """locked target rule (verbatim logic of r5/code/score_hlca.targets): (coarse L, fine T, n) with T != L, T not in BAD, n >= 10, n / n(L) < 0.10."""
    nl = df.label.value_counts(); out = []
    for (L, T), n in df.groupby(["label", "true_type"], observed=True).size().items():
        if T != L and T not in BAD and n >= 10 and n / nl[L] < 0.10: out.append((L, T, int(n)))
    return out

def label_cols(obs, don):
    """candidate label columns: categorical/string, 2..300 non-numeric categories, not numeric ids, not donor-specific, not in NOTLAB (cell_type exempt from the name filter)."""
    out = []
    for c in obs.columns:
        s = obs[c]
        if not (isinstance(s.dtype, pd.CategoricalDtype) or s.dtype == object or pd.api.types.is_string_dtype(s)): continue
        if c != "cell_type" and NOTLAB.search(c): continue
        if c != "cell_type" and not CTLAB.search(c): continue    # amendment 2
        if c == "cell_type_ontology_term_id": continue
        cats = pd.Index(s.astype(str).unique()); nn = [x for x in cats if not NUMLIKE.match(x)]
        if not (2 <= len(nn) <= 300) or len(nn) < 0.5 * len(cats): continue    # numeric cluster ids out
        if don is not None and per_donor(s.astype(str), don): continue
        out.append(c)
    return out

def per_donor(s, don):
    """per-donor clusters: >= 50% of categories have >= 95% of their cells in one donor (only meaningful with >= 2 donors)."""
    if don.nunique() < 2: return False
    t = pd.crosstab(s.values, don.values); return bool(((t.max(axis=1) / t.sum(axis=1)) >= 0.95).mean() >= 0.5)

def best_pair(obs, don):
    """returns (chosen dict or None, list of all valid pairs). Drops unknown labels per pair first."""
    cols = label_cols(obs, don); valid = []
    S = {c: obs[c].astype(str) for c in cols}
    for f in cols:
        for c in cols:
            if c == f or c == "cell_type": continue      # cell_type is fine only
            ok = ~(S[c].str.contains(DROP) | S[f].str.contains(DROP)).values
            if ok.sum() == 0: continue
            lab, tt = S[c].values[ok], S[f].values[ok]
            nc, nf = len(set(lab)), len(set(tt))
            if nc >= nf: continue
            t = pd.crosstab(tt, lab).values
            if not ((t.max(axis=1) / t.sum(axis=1)) >= 0.97).all(): continue
            tg = targets(pd.DataFrame({"label": lab, "true_type": tt}))
            valid.append(dict(coarse=c, fine=f, n_targets=len(tg), n_coarse=nc, n_fine=nf, n_cells=int(ok.sum())))
    valid.sort(key=lambda v: (-v["n_targets"], v["n_coarse"], v["coarse"], v["fine"]))
    return (valid[0] if valid else None), valid

def zero_shot(did, cid, out=None):
    """dataset/collection in r5/used_datasets.csv or the atlas manifests (atlas = 8-char id prefix of a dataset or collection) -> flags."""
    ov = f"{out}/not_zero_shot.csv" if out else ""
    if ov and os.path.exists(ov):
        o = pd.read_csv(ov).set_index("dataset_id").reason
        if did in o.index: return f"not zero-shot on study: {o[did]}"
    uf = f"{HOME}/work/r5/used_datasets.csv"
    if not os.path.exists(uf): return "zero-shot status unknown (used_datasets.csv missing)"
    u = pd.read_csv(uf); ds = set(u.dataset_id) | {a for f in glob.glob(f"{HOME}/work/r5/data/atlas_manifest_*.csv") + glob.glob(f"{HOME}/work/ablation_donor/data/atlas_manifest_*.csv") for a in pd.read_csv(f).atlas.astype(str)}
    d_used = did in set(u.dataset_id) or did[:8] in ds; c_used = cid in set(u.collection_id) or cid[:8] in ds
    return "not zero-shot on dataset" if d_used else "not zero-shot on study" if c_used else "zero-shot"

def select_one(row, h5, out):
    """row: candidate csv row (dataset_id, collection_id, organism, assay, ...). Returns dict logged to selection.csv."""
    import anndata as ad
    did = row["dataset_id"]; r = dict(dataset_id=did, organism=row["organism"], assay=row["assay"], multiome="multiome" in str(row["assay"]).lower(), zero_shot=zero_shot(did, row["collection_id"], out), eligible=False, reason="")
    if row["organism"] != "Homo sapiens": r["reason"] = "not human"; return r
    if not os.path.exists(h5): r["reason"] = "h5ad missing"; return r
    a = ad.read_h5ad(h5, backed="r"); obs = a.obs.copy(); a.file.close()
    don = obs["donor_id"].astype(str) if "donor_id" in obs else None
    r.update(n_cells_all=len(obs), n_donors=int(don.nunique()) if don is not None else 0, reference_genome=";".join(map(str, obs["reference_genome"].unique())) if "reference_genome" in obs else "NA")
    ch, valid = best_pair(obs, don)
    if ch is None: r["reason"] = "no valid (coarse, fine) pair"
    else:
        r.update(coarse=ch["coarse"], fine=ch["fine"], n_targets=ch["n_targets"], n_cells=ch["n_cells"], n_coarse=ch["n_coarse"], n_fine=ch["n_fine"])
        why = [m for m, bad in (("<3 targets", ch["n_targets"] < 3), ("<8000 cells after drops", ch["n_cells"] < 8000), ("<2 donors", r["n_donors"] < 2)) if bad]
        r["reason"] = "; ".join(why); r["eligible"] = not why
    os.makedirs(f"{out}/{did}", exist_ok=True); json.dump(dict(row=r, chosen=ch, valid_pairs=valid, label_columns=label_cols(obs, don)), open(f"{out}/{did}/selection.json", "w"), indent=1, default=str)
    return r

def demo():
    n = 300; rng = np.random.default_rng(0); fine = np.array(["A1"] * 200 + ["A2"] * 20 + ["B1"] * 60 + ["B2"] * 20); coarse = np.array([f[0] for f in fine]); don = pd.Series(rng.choice(["d1", "d2", "d3"], n))
    obs = pd.DataFrame(dict(cell_type=fine, coarse_class=coarse, clu=rng.integers(0, 5, n).astype(str), donor_id=don.values, other=np.where(rng.random(n) < .5, "x", "y")), index=[f"c{i}" for i in range(n)])
    ch, valid = best_pair(obs, don)
    assert ch["coarse"] == "coarse_class" and ch["fine"] == "cell_type" and ch["n_targets"] == 1, ch   # only A2 (20/220 < 10%); B2 is 25% of B
    obs.loc[obs.cell_type == "A1", "cell_type"] = "unknown"; assert best_pair(obs, don)[0]["n_cells"] == 100       # unknown dropped first
    assert label_cols(obs, don) == ["cell_type", "coarse_class"] or "clu" not in label_cols(obs, don)         # numeric ids excluded
    assert DROP.search("Low quality") and DROP.search("T cell doublet") and DROP.search("") and not DROP.search("T cell")
    assert not CTLAB.search("patient_group") and not CTLAB.search("region") and not CTLAB.search("subject") and all(CTLAB.search(x) for x in ("Manual_Annotation_L1_RNA_Integration", "LASP_sub", "Class", "majorclass", "celltype_l2", "sub_celltype"))
    assert NOTLAB.search("scOMM_HCA_l1") and NOTLAB.search("predicted_label") and NOTLAB.search("leiden_3") and not NOTLAB.search("Manual_Annotation_L1")

if __name__ == "__main__":
    if len(sys.argv) == 1: demo(); print("demo ok"); sys.exit(0)
    p = argparse.ArgumentParser(); p.add_argument("--cand", default=f"{HOME}/work/jiaqi1011/out/w6_annot_atac/atac21_candidates.csv"); p.add_argument("--h5ad-dir"); p.add_argument("--h5ad"); p.add_argument("--dataset")
    p.add_argument("--out", required=True); p.add_argument("--only"); A = p.parse_args(); os.makedirs(A.out, exist_ok=True)
    C = pd.read_csv(A.cand); C = C[C.dataset_id.isin(A.only.split(","))] if A.only else C
    C = C[C.dataset_id == A.dataset] if A.dataset else C; rows = []
    for _, row in C.iterrows():
        h5 = A.h5ad or f"{A.h5ad_dir}/{row.dataset_id}.h5ad"; r = select_one(row, h5, A.out); rows.append(r); print(r["dataset_id"][:8], r["eligible"], r.get("coarse"), r.get("fine"), r.get("n_targets"), r["reason"], flush=True)
    pd.DataFrame(rows).to_csv(f"{A.out}/selection.csv", index=False)
