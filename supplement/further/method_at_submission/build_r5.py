"""Test-pool builder for the second set of atlases. Reuses build_atlas5.pools and its pilot / slim-h5ad format unchanged.
Structure only: reads obs, raw counts and truth labels (targets), never any score. Env R5 = data root (outputs), R5_WORK = static inputs (default ~/work/r5).
  python build_r5.py build NAME [NAME ...]      NAME = 8-char dataset id of an ATL entry; atlas file $R5/atlas/<dataset_id>.h5ad
  python build_r5.py r4 luad                    regression check against the first set's config (no drops, 24 pools max, split test4)
  python build_r5.py promote                    BC1-BC4 statuses -> primary / alternate decision (atlas5_final.csv)
  python build_r5.py donors NAME                donor checks from the API cache only (no download)
Outputs ($R5): slim_atlas/<name>_p<k>.h5ad, hmod/pilots/atl_<name>_p<k>.parquet, atlas_manifest_<name>.csv, leak5/ (log_<name>.csv, labels_<name>.csv,
targets_<name>.csv, ../leak5_log.csv), build5/status_<name>.json."""
import os, sys, re, json, glob, time, numpy as np, pandas as pd, anndata as ad, h5py, scipy.sparse as sp
HR_WORK = os.environ.get("HR_WORK", os.path.expanduser("~/work"))
SUPP_DATA = os.environ.get("SUPP_DATA", os.path.expanduser("~/data"))
from anndata.io import sparse_dataset
os.environ.setdefault("SI_H", os.path.join(os.path.dirname(os.path.abspath(__file__)), "hmod"))   # hlib / rprec for score_hlca without copying them to $R5/hmod
import build_atlas5 as b          # pool engine: b.pools, b.N, b.OUT, b.PIL, b.AT, b.SI

R5, N, MAXP = b.R5, b.N, 8
W = os.environ.get("R5_WORK") or HR_WORK + "/r5"
LK, ST = f"{R5}/leak5", f"{R5}/build5"
SPLIT = "test5"

# ---- atlas config. name = dataset id[:8]; (dataset_id, coarse = visible label col, fine = truth col, stratum, role, order)
_T = [("6f9de485-58cd-4342-bfc4-b3d3dd223aa8", "author_cell_type", "cell_state", "tumour", "primary", 1),
      ("1c53ef1e-45c7-4498-be59-6797f3950875", "broad_anno", "fine_anno", "tumour", "primary", 2),
      ("fe067858-1282-46ea-9553-b8cd19167266", "author_cell_type", "author_sub_cell_type", "tumour", "primary", 3),
      ("0bf30b02-db97-4560-b9cd-0fe3588892af", "lv1_annot", "lv2_annot", "tumour", "primary", 4),
      ("624d92e2-d006-4241-9fd8-03535f850301", "cell_type", "atlas_classification", "tumour", "primary", 5),
      ("1cb09bb9-bb9d-4264-870d-c1cab521f697", "cell_type", "Label_fine", "tumour", "alternate", 1),
      ("cff99df2-4904-44f7-9173-ff837f95606e", "Major_celltypes", "Author_Annotation", "non-tumour", "primary", 1),
      ("228438db-5728-4e10-b660-56a54d9a24dd", "High_Lvl_Annotation", "Annotation", "non-tumour", "primary", 2),
      ("b58c19c6-bc2d-4461-b9bf-60fa2ac91479", "broad_celltype", "author_cell_type", "non-tumour", "primary", 3),
      ("b57462e3-1df5-463a-a3fa-4d67b93ef087", "PrimaryAnnotation", "ThirdAnnotation", "non-tumour", "primary", 4),
      ("c3bf8bcf-abb7-4d71-8491-bd4901b474cf", "main_cell_type", "subtype", "non-tumour", "primary", 5),
      ("551ad2bc-2e82-4541-8aab-849988d43752", "annotation_level0", "annotation_level2", "non-tumour", "alternate", 1)]
ATL = {t[0][:8]: dict(zip(("id", "lab", "fine", "stratum", "role", "order"), t)) for t in _T}
R4 = {"luad": ("author_cell_type_level_1", "author_cell_type_level_2")}     # regression config for luad, no drops

# donors dropped by decision before the test; every one has an exact string match in the reference set
DROP_PRE = {
    "fe067858": "CV0902 CV0904 CV0911 CV0915 CV0917 CV0926 CV0929 CV0934 CV0939 CV0940 CV0944 "
                "MH8919176 MH8919177 MH8919178 MH8919179 MH8919226 MH8919227 MH8919282 MH8919283 MH8919332 MH8919333 newcastle65 newcastle74 A35 A36".split(),
    "551ad2bc": "582C 583B 591C 621B 640C 398B 343B 367C 411C".split()}      # 411C added: norm match to psccolon 411c, alias match to "A32 (411C)" in panel file 80a2c5b6 and the CBTM siblings
# mechanical rule: all non-primary cells of these atlases are dropped before pooling (meta-atlas with generic-id donors whose source cannot be found)
NPDROP = {"fe067858"}
# CELLDROP: visible OR truth label, lower-cased and stripped, equals one of these
LABDROP = re.compile(r"^(doublets?|unassigned|unclassified|unknown|undefined|low[ _]?quality|removed|nan|none|na|-)$")
GEN = {"p", "h", "d", "s", "donor", "patient", "subject", "sample", "pt", "ctrl", "control", "case", "individual"}
GENID = re.compile(r"^(?:%s)?[-_ ]?\d+$" % "|".join(sorted(GEN, key=len, reverse=True)))
STUDIES = "zilionis qian azizi karaayvaz hlca jerby ma durante sole gestilinger zheng kurten choi nieto".split()
TAG = re.compile(r"^([A-Za-z]+?)(?:LUAD|LUSC|BRCA|CRC|Ov|OV|PBMC|MEL|HCC|UVM|SKCM|GBM|PAAD|HNSC|RCC|BLCA)_")


# ---------------------------------------------------------------- donor / leak helpers
def cohort_norm(d):
    s = d.strip().lower(); m = re.match(r"^([a-z]+)(?=[0-9_\-]|$)", s); pre = ""
    if m and len(m.group(1)) >= 3 and m.group(1) not in GEN: pre = m.group(1); s = s[len(pre):]
    return pre, re.sub(r"[^a-z0-9]", "", s)


def aliases(d):
    """token-level alias rule (check 4): 'A32 (411C)' -> ['A32', '411C'] (text outside and inside one trailing parenthesis, each at least 2 characters)."""
    m = re.match(r"^(.*?)\s*\(([^()]+)\)\s*$", d.strip())
    return [t.strip() for t in m.groups() if len(t.strip()) >= 2] if m else []


def study_tag(d):
    m = TAG.match(d)
    if m and len(m.group(1)) >= 2 and m.group(1).lower() not in GEN: return m.group(1).lower()
    s = d.lower()
    for k in STUDIES:
        if s.startswith(k) and (len(s) == len(k) or not d[len(k)].islower()) and (len(s) == len(k) or not s[len(k)].isalpha() or d[len(k)].isupper()): return k
    return None


def read_overrides():
    fn = f"{LK}/overrides.csv"; cols = ["atlas", "action", "item", "reason"]
    if not os.path.exists(fn): return pd.DataFrame(columns=cols)
    d = pd.read_csv(fn, dtype=str); [d.__setitem__(c, np.nan) for c in cols if c not in d]
    return d[cols]


def _jload(p):
    return json.load(open(p))


def reference(refresh=False):
    """{'donors': [(donor, source)], 'corpus': lowercase text of donor ids + titles + citations + collection names of the used collections}.
    Sources: API donor lists of the 672 datasets, the panel files' obs donor_id, the manifest pool names of the earlier atlas sets. Cached in $R5/leak5/ref.json."""
    cp = f"{LK}/ref.json"
    if os.path.exists(cp) and not refresh: return _jload(cp)
    don, text = [], []
    used = {l.split()[0] for l in open(f"{W}/tmp/opus_check/used_colls.txt") if l.strip()}; colls = _jload(f"{W}/tmp/opus_check/colls.json")
    inused = {d["dataset_id"] for c in colls if c["collection_id"][:8] in used for d in c["datasets"]}     # datasets of the used collections only (the ds cache also holds candidates)
    for f in glob.glob(f"{W}/tmp/opus_check/ds/*.json"):
        x = _jload(f); sid = os.path.basename(f)[:-5]
        if sid not in inused: continue
        for d in x.get("donor_id") or []: don.append((str(d), "api:" + sid))
        text += [str(x.get("citation", "")), str(x.get("title", ""))]
    for c in colls:
        if c["collection_id"][:8] in used: text += [c.get("name") or "", c.get("doi") or ""]
    pf = glob.glob(SUPP_DATA + "/cellxgene/panel/*.h5ad")
    for f in pf:                                   # obs only; skipped (and logged) if unreadable
        try:
            with h5py.File(f, "r") as h:
                from anndata.io import read_elem
                v = read_elem(h["obs/donor_id"]); v = v.astype(str) if hasattr(v, "astype") else v
                for d in pd.unique(np.asarray(v)): don.append((str(d), "panel:" + os.path.basename(f)[:-5]))
        except Exception as e: don.append(("", f"panel_unreadable:{os.path.basename(f)}:{type(e).__name__}"))
    for f in glob.glob(HR_WORK + "/cfgen_natrare/j1004/h/si_v3/vm3_m12/r_all/atlas_manifest_*.csv"):
        for nm in pd.read_csv(f, usecols=["block"]).block.astype(str):
            t = nm.split("+")
            if len(nm) >= 200: t = t[:-1]          # the last name may be cut at 200 characters
            for d in t: don.append((d, "manifest:" + os.path.basename(f)[15:-4]))
    don = [x for x in don if x[0]]; text += [d for d, _ in don]
    r = {"donors": don, "corpus": "\n".join(text).lower()}; os.makedirs(LK, exist_ok=True); json.dump(r, open(cp, "w")); return r


def donor_checks(donors, ref):
    """-> rows (donor, kind, ref, generic). kind: exact | norm | study. Pure function of the donor list and the reference."""
    ex, nm = {}, {}
    for d0, s0 in ref["donors"]:
        for d, s in [(d0, s0)] + [(t, f"{s0}[alias of {d0}]") for t in aliases(d0)]:     # reference ids with a parenthetical also enter as their tokens
            ex.setdefault(d, []).append(s); pre, rem = cohort_norm(d)
            if rem: nm.setdefault(rem, []).append((pre, d, s))
    rows, corp = [], ref["corpus"]
    for d in sorted(set(donors)):
        gen = bool(GENID.match(d.strip().lower()))
        if d in ex: rows.append((d, "exact", ";".join(sorted(set(ex[d]))[:2]) + f" (+{max(0, len(set(ex[d])) - 2)} more)", gen)); continue
        pre, rem = cohort_norm(d)
        hit = [(p, d2, s) for p, d2, s in nm.get(rem, []) if rem and (p == pre or not p or not pre)]
        if hit: rows.append((d, "norm", ";".join(sorted({f"{d2}@{s}" for _, d2, s in hit})[:4]), gen or (bool(GENID.match(rem)) and not (pre and any(p == pre for p, _, _ in hit)))))   # generic remainder (p1, 07): logged, kept
        t = study_tag(d)
        if t and re.search(r"(?<![a-z])%s(?![a-z])" % re.escape(t), corp): rows.append((d, "study", t, False))
    return rows


def label_rule_counts(o, cols):
    out = []
    for c in cols:
        for v, n in o[c].value_counts().items(): out.append((c, v, int(n), bool(LABDROP.match(str(v).strip().lower()))))
    return pd.DataFrame(out, columns=["column", "label", "n_cells", "celldrop"])


def raw_ok(f, xk):
    with h5py.File(f, "r") as h:
        g = h[xk]; v = np.asarray(g["data"][:200000] if "data" in g else g[:200].ravel()[:200000], dtype=np.float64)
    return bool(len(v) and np.allclose(v, np.round(v)) and v.min() >= 0)


# ---------------------------------------------------------------- prep: obs + drops (shared with the twin builder)
def prep(name, log=None, ref=None):
    """Load obs of the atlas, apply donor drops and CELLDROP. Returns dict(o, pos, f, xk, var, on, cfg, ...). `log(check,item,detail)` collects section-3 evidence."""
    r4 = name in R4; log = log or (lambda *a: None)
    if r4: lab, fine = R4[name]; f = f"{b.AT}/{name}.h5ad"
    else: c = ATL[name]; lab, fine = c["lab"], c["fine"]; f = f"{b.AT}/{c['id']}.h5ad"
    a = ad.read_h5ad(f, backed="r"); cols = list(dict.fromkeys([lab, fine, "donor_id"]))
    miss = [c for c in cols if c not in a.obs.columns]
    if miss: raise KeyError(f"BC4 missing obs columns {miss}")
    o = a.obs[cols].astype(str).copy(); on = a.obs_names.copy()
    assay = a.obs["assay"].astype(str).values if "assay" in a.obs.columns else None
    prim = (a.obs["is_primary_data"].astype(str) == "True").values if "is_primary_data" in a.obs.columns else np.ones(len(o), bool)
    vv = a.raw.var if a.raw is not None else a.var
    var = vv.index.copy(); xk = "raw/X" if a.raw is not None else "X"
    gm = pd.DataFrame({"gene_id": vv.index.astype(str), "feature_name": (vv["feature_name"].astype(str).values if "feature_name" in vv else (a.var["feature_name"].reindex(vv.index).astype(str).values if "feature_name" in a.var else vv.index.astype(str)))})
    a.file.close()
    assert on.is_unique, "obs_names not unique"
    lr = label_rule_counts(o, [lab, fine])
    P = dict(name=name, f=f, xk=xk, var=var, genemap=gm, on=on, lab=lab, fine=fine, r4=r4, n_raw=len(o), donors_raw=sorted(o.donor_id.unique()), labels=lr)
    keep = np.ones(len(o), bool)
    if r4: return _finish(P, o, keep)
    # (i) assay rule (a)
    if assay is not None:
        non10 = ~pd.Series(assay).str.startswith("10x").values; keep &= ~non10
        log("celldrop_assay", "non-10x", json.dumps(pd.Series(assay[non10]).value_counts().to_dict()))
    # donor drops: pre-declared + human overrides in $R5/leak5/overrides.csv (atlas,action,item[,reason]); a line may be written only before this atlas's pools are built
    ov = read_overrides()
    dd = set(DROP_PRE.get(name, [])) | set(ov[(ov.atlas == name) & (ov.action == "drop_donor")].item.astype(str))
    absent = sorted(dd - set(o.donor_id)); keep &= ~o.donor_id.isin(dd).values
    log("donor_drop", f"{len(dd)} declared", f"present {len(dd) - len(absent)}; absent from obs: {absent}")
    if ref is not None:
        rows = donor_checks(P["donors_raw"], ref); strong = {d for d, k, _, g in rows if (k == "study") or not g}
        for d, k, s, g in rows: log("donor_" + k, d, f"{'generic' if g else 'STRONG'} {s}" + (" [declared drop]" if d in dd else ""))
        acc = set(ov[(ov.atlas == name) & (ov.action == "accept_donor") & ov.reason.notna()].item.astype(str))
        acc_study = bool(((ov.atlas == name) & (ov.action == "accept_study") & ov.reason.notna()).any())
        P["study_hits"] = sorted({d for d, k, _, _ in rows if k == "study"} - dd)
        # undeclared strong match = BC1 FAIL until a human line (drop_donor, or accept_donor / accept_study with a reason) exists; nothing is dropped silently
        P["undeclared_strong"] = sorted(strong - dd - acc - (set(P["study_hits"]) if acc_study else set()))
        # (iv) 4b: is_primary_data. Non-primary cells are logged per donor; mechanical rule NPDROP drops ALL non-primary cells of the listed atlases
        npm = pd.Series(~prim).groupby(o.donor_id.values).sum(); npm = npm[npm > 0]
        log("is_primary_data", "obs", f"non-primary cells {int((~prim).sum())} of {len(prim)}; donors carrying them {len(npm)}")
        for d, n in npm.items(): log("is_primary_data_donor", d, f"{int(n)} non-primary cells; strong match={d in strong}; generic-only match={d in {x[0] for x in rows} - strong}")
        sd = ~prim if name in NPDROP else np.zeros(len(o), bool); keep &= ~sd; P["np_dropped"] = int(sd.sum())
        log("celldrop_4b", "non-primary cells dropped by NPDROP rule" if name in NPDROP else "no NPDROP rule for this atlas", str(int(sd.sum())))
    # (ii) label rule on visible OR truth label
    bad = np.zeros(len(o), bool)
    for c in (lab, fine): bad |= o[c].astype(object).fillna("nan").astype(str).str.strip().str.lower().map(lambda s: bool(LABDROP.match(s))).values
    P["labdrop"] = int((bad & keep).sum()); keep &= ~bad
    return _finish(P, o, keep)


def _finish(P, o, keep):
    P["o"] = o[keep]; P["pos"] = np.where(keep)[0]; P["n_cells"] = int(keep.sum()); P["donors"] = sorted(P["o"].donor_id.unique())
    return P


def packs(P):
    """Donor packs in the seeded order of the first atlas set: [(donor list, local row indices)]; trailing partial pack discarded. Same order / accumulation as b.pools."""
    o = P["o"]; order = np.random.RandomState(0).permutation(sorted(o.donor_id.unique())); by = o.groupby("donor_id").indices
    out, cur, names = [], [], []
    for d in order:
        cur.append(by[d]); names.append(str(d))
        if sum(map(len, cur)) >= N: out.append((names, np.concatenate(cur))); cur, names = [], []
    return out, order


def write_pool(P, pn, pid, r, split, extra_src=None):
    """Pool writer (build_atlas5.build), unchanged format. r = local row indices into P['o'] (sorted)."""
    rows = P["pos"][r]
    with h5py.File(P["f"], "r") as h: X = sparse_dataset(h[P["xk"]])[rows]
    X = sp.csr_matrix(X).astype(np.float32)
    assert X.shape[0] == len(rows) and np.allclose(X.data, np.round(X.data)), "BC1 check 5: non-integer counts in pool"
    pp = P["o"].iloc[r]; os.makedirs(b.OUT, exist_ok=True); os.makedirs(b.PIL, exist_ok=True)
    ad.AnnData(X=X, obs=pd.DataFrame(index=pp.index), var=pd.DataFrame(index=P["var"])).write_h5ad(f"{b.OUT}/{pn}.h5ad")
    pd.DataFrame({"obs_name": pp.index.values, "label": pp[P["lab"]].values, "true_type": pp[P["fine"]].values, "is_rare": 0,
                  "donor": pp.donor_id.values, "source_path": f"{pn}.h5ad"}).to_parquet(f"{b.PIL}/{pid}.parquet", index=False)
    return pp


# ---------------------------------------------------------------- build
def build(name, force=False):
    t0 = time.time(); r4 = name in R4; L = []; log = lambda c, i, d: L.append((name, c, i, d))
    os.makedirs(LK, exist_ok=True); os.makedirs(ST, exist_ok=True)
    cfg = {} if r4 else ATL[name]; st = dict(name=name, **{k: v for k, v in cfg.items()}); bc = {}
    try:
        ref = None if r4 else reference()
        if not r4: _coll_checks(name, cfg, log, bc)
        P = prep(name, log, ref)
        if not r4:
            api = _jload(f"{W}/tmp/cand_api.json")[name]; ad_ = set(map(str, api["donor_id"]))
            log("api_record", "donor_id", f"API {len(ad_)} donors, obs {len(P['donors_raw'])}; only-in-API {sorted(ad_ - set(P['donors_raw']))[:5]}, only-in-obs {sorted(set(P['donors_raw']) - ad_)[:5]}")
            log("api_record", "is_primary_data / assay / cell_count", f"{api['is_primary_data']} / {[x['label'] for x in api['assay']]} / {api['cell_count']} (primary {api.get('primary_cell_count')})")
    except Exception as e:
        st.update(status="FAIL", BC4=f"{type(e).__name__}: {e}", wall_s=round(time.time() - t0, 1)); return _save(name, st, L)
    o = P["o"]; lab, fine = P["lab"], P["fine"]
    P["labels"][P["labels"].celldrop].to_csv(f"{LK}/labels_{name}.csv", index=False)       # realised CELLDROP list: fixed rule intersected with the atlas's labels (pre-drop counts)
    st.update(n_cells_raw=P["n_raw"], n_cells=P["n_cells"], donors_raw=len(P["donors_raw"]), donors=len(P["donors"]), labdrop=P.get("labdrop", 0),
              np_dropped=P.get("np_dropped", 0), undeclared_strong=P.get("undeclared_strong", []))
    os.makedirs(f"{R5}/genemap", exist_ok=True); P["genemap"].to_csv(f"{R5}/genemap/{name}.csv", index=False)      # Ensembl id -> symbol, structure only, so the atlas download can be deleted
    log("summary", "cells/donors", f"{P['n_cells']} cells of {P['n_raw']}; {len(P['donors'])} donors of {len(P['donors_raw'])}")
    # BC1 additions: check 3 study flags, check 5 raw counts, donors >= 8
    if not r4:
        if not raw_ok(P["f"], P["xk"]): bc["BC1"] = "FAIL check 5: non-integer or negative values in the first 200k stored values"
        if P.get("undeclared_strong"):
            bc["BC1"] = (f"FAIL check 3/4: {len(P['undeclared_strong'])} undeclared strong donor match(es) {P['undeclared_strong'][:8]} "
                         "(human read: overrides.csv drop_donor, or accept_donor / accept_study with a reason, written before this atlas is built)")
        if len(P["donors"]) < 8: bc["BC1"] = bc.get("BC1") or f"FAIL F4/check 4: {len(P['donors'])} donors after drops (<8)"
    pk, _ = packs(P); maxp = 24 if r4 else MAXP
    pools = [(("+".join(n))[:200], r) for n, r in pk]
    chk = b.pools(o, "pack", "donor_id", np.random.RandomState(0))           # pool engine: sampled rows of every pack
    assert [x[0] for x in chk] == [p[0] for p in pools], "pack bookkeeping differs from build_atlas5.pools"
    sel = chk[:maxp]; st.update(n_packs=len(chk), n_pools=len(sel))
    if not r4 and (len(sel) < 2 or P["n_cells"] < 60000): bc["BC2"] = f"FAIL: {len(chk)} packs, {P['n_cells']} cells (need >=2 pools of {N} and >=60000 cells)"
    import score_hlca as sh
    tg, man, tg_rows = [], [], []
    for k, (blk, r) in enumerate(sel):
        pp = o.iloc[r]; df = pd.DataFrame({"label": pp[lab].values, "true_type": pp[fine].values}); t = sh.targets(df)
        tg.append(t); man.append((f"atl_{name}_p{k}", name, SPLIT if not r4 else "test4", blk, k, len(r), pp[lab].nunique()))
        tg_rows += [(f"atl_{name}_p{k}", L_, T_, n_) for L_, T_, n_ in t]
    pd.DataFrame(tg_rows, columns=["pilot_id", "label", "target", "n"]).to_csv(f"{LK}/targets_{name}.csv", index=False)
    st["targets_per_pool"] = [len(t) for t in tg]; st["pools_with_target"] = sum(1 for t in tg if t)
    if not r4 and st["pools_with_target"] < 2: bc["BC3"] = f"FAIL: {st['pools_with_target']} pools with a target"
    st.update({k: v for k, v in bc.items()})
    ok = not any(str(v).startswith("FAIL") for v in bc.values())
    if ok or force:
        try:
            for k, (blk, r) in enumerate(sel): write_pool(P, f"{name}_p{k}", f"atl_{name}_p{k}", r, SPLIT); print(f"atl_{name}_p{k}", len(r), flush=True)
            pd.DataFrame(man, columns=["pilot_id", "atlas", "split", "block", "pool", "n_cells", "n_labels"]).to_csv(f"{R5}/atlas_manifest_{name}.csv", index=False)
        except AssertionError as e: st["BC1"] = f"FAIL check 5: {e}"; ok = False
        except Exception as e: st["BC4"] = f"FAIL {type(e).__name__}: {e}"; ok = False
    st.update(status="PASS" if ok else "FAIL", wall_s=round(time.time() - t0, 1)); return _save(name, st, L)


def _coll_checks(name, cfg, log, bc):
    """checks 1-3 (dataset id, collection id, title / DOI / description) against used_datasets.csv and the used collections."""
    u = pd.read_csv(f"{W}/used_datasets.csv"); cand = pd.read_csv(f"{W}/CANDIDATES.csv").set_index("dataset_id").loc[cfg["id"]]
    cid = cand.collection_id
    if cfg["id"] in set(u.dataset_id): bc["BC1"] = "FAIL check 1: dataset id is used"
    if cid in set(u.collection_id): bc["BC1"] = "FAIL check 2: collection id is used"
    log("check1_2", cfg["id"], f"dataset used={cfg['id'] in set(u.dataset_id)}; collection {cid} used={cid in set(u.collection_id)}")
    cs = _jload(f"{W}/tmp/opus_check/colls.json"); used = {l.split()[0] for l in open(f"{W}/tmp/opus_check/used_colls.txt") if l.strip()}
    nrm = lambda s: re.sub(r"[^a-z0-9 ]", "", str(s or "").lower()).strip()
    U = [c for c in cs if c["collection_id"][:8] in used]; me = [c for c in cs if c["collection_id"] == cid]
    if not me: log("check3", cid, "candidate collection record not in colls.json: needs API fetch (not done)"); return
    me = me[0]; ut = {nrm(c.get("name")) for c in U}; ud = {nrm(c.get("doi")) for c in U if c.get("doi")}
    hit = nrm(me.get("name")) in ut or (me.get("doi") and nrm(me["doi"]) in ud)
    blob = nrm(me.get("description")) + " " + nrm(json.dumps(me.get("links"))) + " " + nrm(cand.title)
    named = sorted(d for d in ud if d and d in re.sub(r"[^a-z0-9]", "", blob))
    log("check3", me.get("doi") or "", f"title/doi match to a used collection: {bool(hit)}; used DOIs named in description/links: {named}")
    if hit or named: bc["BC1"] = "FAIL check 3: same study as a used collection (see leak5_log.csv)"


def _save(name, st, L):
    json.dump(st, open(f"{ST}/status_{name}.json", "w"), indent=1, default=str)
    pd.DataFrame(L, columns=["atlas", "check", "item", "detail"]).to_csv(f"{LK}/log_{name}.csv", index=False)
    parts = [pd.read_csv(p) for p in sorted(glob.glob(f"{LK}/log_*.csv"))]; pd.concat(parts).to_csv(f"{R5}/leak5_log.csv", index=False)
    print(json.dumps({k: v for k, v in st.items() if k != "undeclared_strong"}, default=str)); return st


# ---------------------------------------------------------------- alternates (build-check rule)
def decide(status):
    """status: {name: 'PASS'|'FAIL'|None}. Per stratum: primaries in order; a failed primary is replaced by the next unused alternate, at most 2 per stratum;
    underpowered if < 4 atlases remain. Returns (final dict stratum -> names, need_build list, notes)."""
    final, need, notes = {}, [], []
    for s in ("tumour", "non-tumour"):
        pr = sorted([n for n, c in ATL.items() if c["stratum"] == s and c["role"] == "primary"], key=lambda n: ATL[n]["order"])
        al = sorted([n for n, c in ATL.items() if c["stratum"] == s and c["role"] == "alternate"], key=lambda n: ATL[n]["order"]); out, subs = [], 0
        for p in pr:
            if status.get(p) is None: need.append(p); continue
            if status[p] == "PASS": out.append(p); continue
            while al and subs < 2:
                a = al.pop(0); subs += 1; notes.append(f"{s}: {p} failed, substitute {a}")
                if status.get(a) is None: need.append(a); break
                if status[a] == "PASS": out.append(a); break
                notes.append(f"{s}: substitute {a} failed")
            else: notes.append(f"{s}: {p} failed, no substitute left")
        final[s] = out
        if len(out) < 4 and not need: notes.append(f"{s}: UNDERPOWERED ({len(out)} atlases)")
    return final, need, notes


def promote():
    st = {n: (_jload(f"{ST}/status_{n}.json")["status"] if os.path.exists(f"{ST}/status_{n}.json") else None) for n in ATL}
    ov = read_overrides()
    for n in ATL:       # BC4 catches exceptions, including the lab's own setup: a BC4 failure counts only after a human line `confirm_bc4,<atlas>,<name>,<reason>`
        j = _jload(f"{ST}/status_{n}.json") if os.path.exists(f"{ST}/status_{n}.json") else {}
        if st[n] == "FAIL" and str(j.get("BC4", "")).startswith("FAIL") and not ((ov.atlas == n) & (ov.action == "confirm_bc4") & ov.reason.notna()).any():
            print(f"HUMAN CONFIRMATION NEEDED: {n} failed BC4 ({str(j['BC4'])[:160]}); not substituted until overrides.csv has confirm_bc4"); st[n] = None
    final, need, notes = decide(st); print("statuses:", st); print("final:", final); print("need_build:", need); [print(x) for x in notes]
    pd.DataFrame([(s, n, ATL[n]["id"]) for s, v in final.items() for n in v], columns=["stratum", "name", "dataset_id"]).to_csv(f"{R5}/atlas5_final.csv", index=False)


def demo():
    assert sum(map(len, DROP_PRE.values())) == 34 and len(DROP_PRE["fe067858"]) == 25 and len(DROP_PRE["551ad2bc"]) == 9
    assert cohort_norm("KNUH02") == ("knuh", "02") and cohort_norm("P01")[0] == "" and cohort_norm("Donor_3")[0] == ""
    assert study_tag("ZilionisLUAD_p1") == "zilionis" and study_tag("HLCALUAD_1") == "hlca" and study_tag("QianOv_3") == "qian" and study_tag("MaHCC_1") == "ma" and study_tag("P1") is None
    assert study_tag("kurten12") == "kurten" and study_tag("Maria") is None and GENID.match("p01") and GENID.match("91") and not GENID.match("CV0902")
    ref = {"donors": [("Zilionis_Klein_2019_patient_1", "api:x"), ("CV0902", "api:y"), ("P1", "api:z"), ("ABC_07", "panel:q")], "corpus": "zilionis_klein_2019_patient_1\nqian\n"}
    rr = {(d, k): g for d, k, _, g in donor_checks(["CV0902", "P1", "xyz07", "abc07", "ZilionisLUAD_p1", "Q9"], ref)}
    assert rr == {("CV0902", "exact"): False, ("P1", "exact"): True, ("abc07", "norm"): False, ("ZilionisLUAD_p1", "norm"): True, ("ZilionisLUAD_p1", "study"): False}, rr
    A = lambda **k: {n: k.get(n) for n in ATL}
    ok = {n: "PASS" for n in ATL}
    assert decide(ok)[0]["tumour"] == [n for n in sorted(ATL, key=lambda n: ATL[n]["order"]) if ATL[n]["stratum"] == "tumour" and ATL[n]["role"] == "primary"]
    f, nd, nt = decide({**ok, "fe067858": "FAIL"}); assert "1cb09bb9" in f["tumour"] and len(f["tumour"]) == 5 and not nd
    f, nd, nt = decide({**ok, "fe067858": "FAIL", "1c53ef1e": "FAIL"}); assert len(f["tumour"]) == 4 and any("no substitute" in x for x in nt) and any("UNDERPOWERED" not in x for x in nt) is not None
    f, nd, nt = decide({**ok, "cff99df2": None}); assert nd == ["cff99df2"]
    f, nd, nt = decide({**ok, "cff99df2": "FAIL", "551ad2bc": None}); assert nd == ["551ad2bc"]


if __name__ == "__main__":
    demo(); cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "build": [build(n) for n in args]
    elif cmd == "r4": [build(n) for n in args]
    elif cmd == "promote": promote()
    elif cmd == "donors":                      # donor checks from the API cache only
        x = _jload(f"{W}/tmp/cand_api.json")[args[0]]; ref = reference()
        for d, k, s, g in donor_checks(x["donor_id"], ref): print(args[0], d, k, "generic" if g else "STRONG", s)
