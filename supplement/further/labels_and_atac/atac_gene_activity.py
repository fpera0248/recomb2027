"""python atac_gene_activity.py : fragments -> gene-activity counts for the renal-cortex multiome (CELLxGENE 09b518f9). Venv: ~/work/jiaqi1011/venvs/atac. Resumable.
Gene annotation = Ensembl 110 EnsDb (Bioconductor AnnotationHub) exported by ensembl_gene_annotation.R; gene_name key = Ensembl ID so the matrix columns are ENSG ids."""
import os, re, sys, anndata as ad, snapatac2 as snap
HR_WORK = os.environ.get("HR_WORK", os.path.expanduser("~/work"))
D = HR_WORK + "/jiaqi1011/data/w6_atac"; FR = f"{D}/e037f157-49b6-4a1b-b34f-51a928778a27-fragment.tsv.bgz"; H5 = f"{D}/46aada24-d005-4486-9e21-866da4fdbd7a.h5ad"
CS = dict(zip(["chr" + str(i) for i in range(1, 23)] + ["chrX", "chrY", "chrM"], [248956422, 242193529, 198295559, 190214555, 181538259, 170805979, 159345973, 145138636, 138394717, 133797422, 135086622, 133275309,
          114364328, 107043718, 101991189, 90338345, 83257441, 80373285, 58617616, 64444167, 46709983, 50818468, 156040895, 57227415, 16569]))   # GRCh38 primary assembly lengths
if os.path.exists(f"{D}/ga_counts.h5ad"): sys.exit(0)
os.makedirs(f"{D}/tmp", exist_ok=True)
gtf = f"{D}/ensdb110_genes_idkey.gtf"
if not os.path.exists(gtf):
    with open(f"{D}/ensdb110_genes.gtf") as f, open(gtf, "w") as o:
        for l in f:   # fragment contigs are UCSC-style (chr1..chrM): prefix the Ensembl seqnames; gene_name := Ensembl id
            l = "chrM" + l[2:] if l.startswith("MT\t") else "chr" + l; assert re.match(r"chr(\d+|X|Y|M)\t", l), l; o.write(re.sub(r'gene_name "[^"]*"', lambda m: 'gene_name "' + re.search(r'gene_id "([^"]*)"', l).group(1) + '"', l))
wl = f"{D}/whitelist.txt"; open(wl, "w").write("\n".join(ad.read_h5ad(H5, backed="r").obs_names) + "\n")
frag = f"{D}/frag.h5ad"
if not os.path.exists(frag):
    snap.pp.import_fragments(FR, CS, file=frag + ".tmp", whitelist=wl, min_num_fragments=0, sorted_by_barcode=False, chrM=["chrM"], n_jobs=4, tempdir=f"{D}/tmp").close(); os.rename(frag + ".tmp", frag)
a = snap.read(frag, backend="hdf5"); print(a, flush=True)
if os.path.exists(f"{D}/ga_tmp.h5ad"): os.remove(f"{D}/ga_tmp.h5ad")
g = snap.pp.make_gene_matrix(a, gene_anno=gtf, file=f"{D}/ga_tmp.h5ad", counting_strategy="paired-insertion", upstream=2000, include_gene_body=True); g.close(); a.close()
import numpy as np, pandas as pd, scipy.sparse as sp
g = snap.read(f"{D}/ga_tmp.h5ad", backend="hdf5"); X = sp.csr_matrix(g.X[:]); x = ad.AnnData(X=X, obs=pd.DataFrame(index=list(g.obs_names)), var=pd.DataFrame(index=list(g.var_names))); g.close()
print(x, float(X.sum(1).mean()), flush=True); assert X.sum() > 0 and np.allclose(X.data, np.round(X.data)); x.write_h5ad(f"{D}/ga_counts.h5ad"); print("BUILD_DONE")
