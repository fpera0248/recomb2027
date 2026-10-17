"""python build_ga.py --h5ad H5 --frag FRAGMENTS.tsv.bgz --gtf ensdb110_genes_idkey.gtf --data DIR : fragments of the WHITELISTED cells (DIR/whitelist.txt, from pools.py prep) -> gene-activity counts DIR/ga_counts.h5ad.
Same recipe as the kidney pilot (atac_gene_activity.py in the parent folder, earlier name p3_build.py): Ensembl 110 gene body + 2 kb upstream, paired insertion, GRCh38 chromosome sizes, gene_name = Ensembl id. Resumable (ga_counts.h5ad = done; frag.h5ad kept).
New vs the pilot: barcode and contig style are detected from the fragment file head (identity / strip '-N' / sample prefix removal; chr-prefixed or not) and the cell names are mapped back to the h5ad obs names. Venv: ~/work/jiaqi1011/venvs/atac (snapATAC2)."""
import os, re, sys, gzip, argparse, itertools, numpy as np
sys.dont_write_bytecode = True
LEN = [248956422, 242193529, 198295559, 190214555, 181538259, 170805979, 159345973, 145138636, 138394717, 133797422, 135086622, 133275309, 114364328, 107043718, 101991189, 90338345, 83257441, 80373285, 58617616, 64444167, 46709983, 50818468, 156040895, 57227415, 16569]
NAMES = [str(i) for i in range(1, 23)] + ["X", "Y", "M"]   # GRCh38 primary assembly lengths (chrM 16569)
TRANSFORMS = {"identity": lambda x: x, "strip_suffix": lambda x: re.sub(r"-\d+$", "", x), "after_sep": lambda x: re.split(r"[_#:]", x)[-1], "after_sep_strip": lambda x: re.sub(r"-\d+$", "", re.split(r"[_#:]", x)[-1])}

def head(frag, n=200000):
    """first n fragment lines: (contigs set, barcodes set)."""
    ch, bc = set(), set()
    with gzip.open(frag, "rt") as f:
        for l in itertools.islice((l for l in f if not l.startswith("#")), n): t = l.split("\t", 4); ch.add(t[0]); bc.add(t[3])
    return ch, bc

def pick_transform(obs_names, bcs, min_match=0.5):
    """transform of obs names with unique images that explains most fragment barcodes; returns (name, match fraction, map obs -> barcode)."""
    best = None
    for k, fn in TRANSFORMS.items():
        im = [fn(x) for x in obs_names]; T = set(im)
        if len(T) != len(im): continue                                    # ambiguous (barcode shared by several samples): cannot map
        h = len(T & bcs); m = h / max(len(bcs), 1)
        if best is None or h > best[3]: best = (k, m, dict(zip(obs_names, im)), h)
    # share of fragment barcodes explained, or (ARC-style files full of non-cell barcodes) >= 20% of the cells seen in the head; the post-import n_obs check is the real gate
    assert best and (best[1] >= min_match or best[3] >= 0.2 * len(obs_names)), f"no barcode transform explains the fragment barcodes (best {best[:2] if best else None}); check obs_names vs fragment barcodes"
    return best[:3]

def gtf_for(gtf, chr_style, out):
    """GTF seqnames must match the fragment contigs: the idkey GTF is chr-prefixed (chr1.., chrM); derive an unprefixed copy when fragments use 1..22,X,Y,MT."""
    if chr_style: return gtf
    if not os.path.exists(out):
        with open(gtf) as f, open(out + ".tmp", "w") as o:
            for l in f: o.write(re.sub(r"^chr", "", re.sub(r"^chrM\t", "MT\t", l), count=1))
        os.rename(out + ".tmp", out)
    return out

def demo():
    import tempfile
    d = tempfile.mkdtemp(); fr = f"{d}/f.tsv.gz"
    with gzip.open(fr, "wt") as o: o.write("# comment\n"); [o.write(f"chr1\t{i}\t{i+50}\tACGT{i%7}-1\t1\n") for i in range(100)]
    ch, bc = head(fr); assert ch == {"chr1"} and len(bc) == 7
    obs = [f"s1_ACGT{i}-1" for i in range(7)]; k, m, mp = pick_transform(obs, bc); assert k == "after_sep" and m == 1.0 and mp[obs[2]] == "ACGT2-1"
    obs2 = [f"ACGT{i}-1" for i in range(7)]; assert pick_transform(obs2, bc)[0] == "identity"
    dup = [f"s{j}_ACGT{i}-1" for j in range(2) for i in range(7)]
    try: pick_transform(dup, bc); raise SystemExit("ambiguous barcodes must fail")
    except AssertionError: pass
    g = f"{d}/g.gtf"; open(g, "w").write('chr1\te\tgene\t1\t10\t.\t+\t.\tgene_id "ENSG1"; gene_name "ENSG1";\nchrM\te\tgene\t1\t10\t.\t+\t.\tgene_id "ENSG2"; gene_name "ENSG2";\n')
    o = gtf_for(g, False, f"{d}/n.gtf"); t = open(o).read().split("\n"); assert t[0].startswith("1\t") and t[1].startswith("MT\t") and gtf_for(g, True, "x") == g
    assert len(LEN) == len(NAMES) == 25

if __name__ == "__main__":
    if len(sys.argv) == 1: demo(); print("demo ok"); sys.exit(0)
    p = argparse.ArgumentParser(); [p.add_argument(f"--{x}", required=True) for x in ("h5ad", "frag", "gtf", "data")]; A = p.parse_args(); D = A.data; os.makedirs(f"{D}/tmp", exist_ok=True)
    if os.path.exists(f"{D}/ga_counts.h5ad"): print("BUILD_DONE (cached)"); sys.exit(0)
    import anndata as ad, pandas as pd, scipy.sparse as sp, snapatac2 as snap
    obs_names = list(ad.read_h5ad(A.h5ad, backed="r").obs_names); ch, bc = head(A.frag); chr_style = all(c.startswith("chr") for c in ch if len(c) < 6 and not c.startswith("GL"))
    k, m, mp = pick_transform(obs_names, bc); wl = [x.strip() for x in open(f"{D}/whitelist.txt") if x.strip()]; inv = {mp[x]: x for x in wl}; assert len(inv) == len(wl)
    print("barcode transform", k, "match", m, "chr_style", chr_style, "n whitelist", len(wl), flush=True); open(f"{D}/whitelist_frag.txt", "w").write("\n".join(inv) + "\n")
    CS = {("chr" if chr_style else "") + n if n != "M" else ("chrM" if chr_style else "MT"): l for n, l in zip(NAMES, LEN)}; gtf = gtf_for(A.gtf, chr_style, f"{D}/ensdb110_genes_idkey_nochr.gtf"); frag = f"{D}/frag.h5ad"
    if not os.path.exists(frag): snap.pp.import_fragments(A.frag, CS, file=frag + ".tmp", whitelist=f"{D}/whitelist_frag.txt", min_num_fragments=0, sorted_by_barcode=False, chrM=["chrM" if chr_style else "MT"], n_jobs=int(os.environ.get("BUILD_JOBS", 4)), tempdir=f"{D}/tmp").close(); os.rename(frag + ".tmp", frag)
    a = snap.read(frag, backend="hdf5"); print(a, flush=True)
    if os.path.exists(f"{D}/ga_tmp.h5ad"): os.remove(f"{D}/ga_tmp.h5ad")
    g = snap.pp.make_gene_matrix(a, gene_anno=gtf, file=f"{D}/ga_tmp.h5ad", counting_strategy="paired-insertion", upstream=2000, include_gene_body=True); g.close(); a.close()
    g = snap.read(f"{D}/ga_tmp.h5ad", backend="hdf5"); X = sp.csr_matrix(g.X[:]); names = [inv[x] for x in g.obs_names]
    x = ad.AnnData(X=X, obs=pd.DataFrame(index=names), var=pd.DataFrame(index=list(g.var_names))); g.close()
    print(x, float(X.sum(1).mean()), flush=True); assert X.sum() > 0 and np.allclose(X.data, np.round(X.data)) and x.n_obs >= 0.5 * len(wl), f"too few cells with fragments: {x.n_obs} of {len(wl)}"
    x.write_h5ad(f"{D}/ga_counts.h5ad"); print("BUILD_DONE")
