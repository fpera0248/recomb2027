# gen16m: gen16.py + SAVE_GENE hook that saves every arm's added cells in gene space (for marker recovery).
# Writes gen16m*.csv so gen16's own outputs are never touched.
"""THE HEAD-TO-HEAD: does duplicating real cells beat every generative model?

One donor-disjoint split, one dose, one readout, every generator family we can run:

    real           no augmentation
    upsample       copies of that class's real cells            <-- the claim under test
    nb             per-class negative binomial, no copula        (scDesign3 stripped bare)
    sd3            scDesign3, all 31 classes in one call
    sd3_nosingle   scDesign3 with the 1-cell classes removed
    synth_w1       CFGen flow matching, as shipped
    synthbal_w1    CFGen with class-balanced minibatches         (its best configuration)
    scvi           scVI, cell type through batch_key
    scvi_cond      CondSCVI, cell type as a first-class label
    sdiff          scDiffusion, classifier-guided latent diffusion
    lat_boot       real latents through the shared decoder       (the upper reference)
    lat_gauss      per-class Gaussian in latent space

The summary is written for the claim, not for the generators: per band we report the delta
against upsampling with a CLASS-CLUSTERED t (8 cell types, not 8*seeds observations), the
WIN RATE of upsampling over each arm, and the wall-clock seconds each arm costs to produce
its cells. A method that ties duplication and costs an hour of training has lost.

(inherited from gen9)
Donor-disjoint rare-type augmentation: the regime where scDesign3 cannot run.

Same AIDA data, same dose matching, same recall@200 readout as gen6/gen8, with two changes
that together make the setting honest AND hostile to a per-class parametric fit:

  * the split is DONOR-DISJOINT. Whole donors go to eval; no train donor contributes an eval
    cell. Our earlier per-cell split leaked donor identity and inflated rare-type accuracy.
  * the rung ladder starts at 1. A one-cell class has no within-class covariance to estimate,
    which is exactly where scDesign3's Gaussian copula (corr_by = cell_type) has nothing to fit.

Arms:
    real        no augmentation
    upsample    copies of that class's real cells      (the honest baseline)
    sd3         scDesign3, all 31 classes in one call  (may abort; failure is the result)
    synth_w1    CFGen flow matching, as shipped        (the deep-generator reference)
    lat_boot    the class's REAL latents resampled through the shared decoder
    lat_gauss   a per-class Gaussian in latent space through that same decoder  (the graft)

sd3 is attempted twice per seed: once with every class, and once with the singleton classes
removed. If the first aborts and the second succeeds, the singleton is the cause and one
1-cell class kills synthesis for the other 30 classes too.
"""
import sys, os, subprocess, tempfile, numpy as np, pandas as pd, torch, anndata as ad, scanpy as sc, time, warnings, pathlib
HR_WORK = os.environ.get("HR_WORK", os.path.expanduser("~/work"))
SUPP_DATA = os.environ.get("SUPP_DATA", os.path.expanduser("~/data"))
import scipy.io, scipy.sparse
import scvi
from sdiff_mod import build_sdiff, sample_sdiff
from harmonypy import compute_lisi
warnings.filterwarnings('ignore')
import logging; logging.getLogger('lightning.pytorch').setLevel(logging.ERROR)
sys.path.insert(0,'CFGen')
from cfgen.models.base.encoder_model import EncoderModel
from cfgen.models.fm.denoising_model import MLPTimeStep
from cfgen.models.fm.fm import FM
from cfgen.models.featurizers.category_featurizer import CategoricalFeaturizer
torch.set_num_threads(8)

SMOKE = '--smoke' in sys.argv
HVG=400 if '--smoke' in sys.argv else 2000
NEV=50 if '--smoke' in sys.argv else 200
LAT=50; EDIM=20; BS=256; NS=128
CAP=150 if '--smoke' in sys.argv else 500   # control classes capped so scDesign3 can run at all
RS_BIN=HR_WORK + '/envs/scdesign3/bin/Rscript'
SD3_CORES=8   # mem_patch.R (SD3_MEM=1) strips the per-fit env capture, so 8 forks fit in RAM: 2000 genes in 1086s, cache 31 MB
AE_EPOCHS = 2 if SMOKE else 30
SCVI_EPOCHS = 2 if SMOKE else 60
SDIFF_EP = (2,2,2) if SMOKE else (40,60,30)   # VAE, diffusion, classifier
VAE_STEPS = 200 if SMOKE else 5000            # their AdamW at their bs=128
FM_EPOCHS = 2 if SMOKE else 60
RUNGS=np.array([1,2,4,8]) if '--smoke' in sys.argv else np.array([1,2,4,8,16,32,64,128])
WGRID=[1.0]
TARGET=64 if '--smoke' in sys.argv else 256   # every augmented arm reaches exactly this many cells
SEEDS=[0] if SMOKE else [0,1,2]
PROBE=['double negative T regulatory cell','CD141-positive myeloid dendritic cell',
       'innate lymphoid cell','pre-conventional dendritic cell','plasma cell',
       'plasmacytoid dendritic cell','mature B cell',
       'CD16-negative, CD56-bright natural killer cell, human']   # the 8 rarest types in AIDA
if '--smoke' in sys.argv: PROBE=PROBE[:4]
D=SUPP_DATA + '/pilots/aida/'
t0=time.time()

# ---- data: union of the two pools, counts, HVG ----
A=ad.concat([ad.read_h5ad(D+'AIDA_AbundantPool_120k.h5ad'), ad.read_h5ad(D+'AIDA_RarePool.h5ad')],
            join='inner', index_unique='-')
A.obs['ct']=A.obs.cell_type.astype(str)
dall_raw=A.obs.donor_id.astype(str).values
A=A[:, np.asarray(A.X.sum(0)).ravel()>0].copy()
sc.pp.highly_variable_genes(A, n_top_genes=HVG, flavor='seurat_v3', subset=True)
Xc=np.asarray(A.X.todense() if hasattr(A.X,'todense') else A.X, dtype=np.float32)
assert np.allclose(Xc, np.round(Xc)), 'X must be raw counts'
cts=sorted(A.obs.ct.unique()); ci={c:i for i,c in enumerate(cts)}; NC=len(cts); UNLAB=NC; NCAT=NC+1
yall=A.obs.ct.map(ci).values.astype(np.int64)
dall=A.obs.donor_id.astype(str).values; UD=np.unique(dall)
sf=Xc.sum(1,keepdims=True); Xn=np.log1p(Xc/sf*1e4).astype(np.float32)
print(f"{Xc.shape[0]:,} cells x {Xc.shape[1]} HVGs, {NC} types, sizes "
      f"{np.bincount(yall).min()}..{np.bincount(yall).max()}  {time.time()-t0:.0f}s",flush=True)

from scvi.distributions import NegativeBinomial

def decode_latents(ae,Z,L):
    """Push latents through the SHARED AE decoder and its NB noise. L is the raw library size."""
    with torch.no_grad():
        mu=ae.decode({"rna":Z},{"rna":L})["rna"]
        return NegativeBinomial(mu=mu,theta=torch.exp(ae.theta)).sample().numpy().astype(np.float32)

def mk_batch(idx):
    return {"X":{"rna":torch.from_numpy(Xc[idx])}, "X_norm":{"rna":torch.from_numpy(Xn[idx])},
            "y":{"cell_type":torch.from_numpy(yall[idx])}}

def balanced_perm(g,ytr,n):
    """n positions with every class equally likely, sampled with replacement.

    A class with 2 cells contributes 2/12010 of an ordinary epoch's gradient. Here it
    contributes 1/NCAT of it, which is the strongest form of 'train harder on the rare class'."""
    cls=np.unique(ytr); per=n//len(cls)+1
    out=np.concatenate([g.choice(np.where(ytr==c)[0],per,replace=True) for c in cls])
    return g.permutation(out)[:n]

def build_ae(tr_idx,seed,bal=False):
    torch.manual_seed(seed)
    ae=EncoderModel(in_dim={"rna":HVG}, encoder_kwargs={"rna":{"dims":[512,256,LAT],"batch_norm":True,
                    "dropout":False,"dropout_p":0.0}}, learning_rate=1e-3, weight_decay=1e-5,
                    covariate_specific_theta=False, conditioning_covariate="cell_type", n_cat=NCAT)
    ae.log=lambda *a,**k: None
    opt=torch.optim.Adam(ae.parameters(),lr=1e-3,weight_decay=1e-5)
    g=np.random.default_rng(seed)
    ya=yall[tr_idx]
    for ep in range(AE_EPOCHS):
        p=tr_idx[balanced_perm(g,ya,len(tr_idx))] if bal else g.permutation(tr_idx)
        for i in range(0,len(p),BS):
            b=p[i:i+BS]
            if len(b)<2: continue                      # BatchNorm needs >1 sample
            opt.zero_grad(); ae._step(mk_batch(b),'train').backward(); opt.step()
    ae.eval(); return ae

def build_fm(ae,INIT,tr_idx,ytr,seed,sfs,bal=False):
    torch.manual_seed(seed+1000)
    fe={"cell_type":CategoricalFeaturizer(NCAT,False,'cpu',embedding_dimensions=EDIM)}
    if INIT=='zero': torch.nn.init.zeros_(fe["cell_type"].embeddings.weight)
    dn=MLPTimeStep(in_dim=LAT, hidden_dim=32, dropout_prob=0, n_blocks=3,
                   size_factor_min={"rna":sfs['mn']}, size_factor_max={"rna":sfs['mx']},
                   embed_size_factor=False, covariate_list=["cell_type"], embedding_dim=EDIM,
                   normalization="none", conditional=True, is_binarized=False, modality_list=["rna"],
                   conditioning_probability=0.8, guided_conditioning=True)
    fm=FM(encoder_model=ae, denoising_model=dn, feature_embeddings=fe, plotting_folder=pathlib.Path('.'),
          in_dim={"rna":LAT}, size_factor_statistics={"mean":{"rna":{"cell_type":sfs['mu']}},"sd":{"rna":{"cell_type":sfs['sd']}}},
          covariate_list=["cell_type"], theta_covariate="cell_type", size_factor_covariate="cell_type",
          learning_rate=1e-4, weight_decay=1e-6, sigma=1e-4, use_ot=False, modality_list=["rna"],
          guidance_weights={"cell_type":1})
    fm.log_dict=lambda *a,**k: None
    opt=torch.optim.Adam(list(dn.parameters())+list(fe["cell_type"].parameters()),lr=1e-4,weight_decay=1e-6)
    g=np.random.default_rng(seed+7)
    for ep in range(FM_EPOCHS):
        p=balanced_perm(g,ytr,len(tr_idx)) if bal else g.permutation(len(tr_idx))
        for i in range(0,len(p),BS):
            j=p[i:i+BS]
            if len(j)<2: continue
            b=mk_batch(tr_idx[j]); b["y"]["cell_type"]=torch.from_numpy(ytr[j])
            opt.zero_grad(); fm._step(b,'train').backward(); opt.step()
    fm.eval(); return fm

def metrics(G,R,c,CENT):
    dgr=np.linalg.norm(G[:,None]-R[None],axis=2)
    dgg=np.linalg.norm(G[:,None]-G[None],axis=2); drr=np.linalg.norm(R[:,None]-R[None],axis=2)
    e=2*dgr.mean()-dgg.mean()-drr.mean()
    rr=np.sort(drr,1)[:,K]; gg=np.sort(dgg,1)[:,K]
    return dict(energy=float(e),
        purity=float((np.linalg.norm(G[:,None]-CENT[None],axis=2).argmin(1)==c).mean()),
        precision=float((dgr<=rr[None,:]).any(1).mean()),
        recall=float((dgr<=gg[:,None]).any(0).mean()))
K=5

def energy(Ga,Rb):
    ab=np.linalg.norm(Ga[:,None]-Rb[None],axis=2).mean(); aa=np.linalg.norm(Ga[:,None]-Ga[None],axis=2).mean()
    bb=np.linalg.norm(Rb[:,None]-Rb[None],axis=2).mean(); return 2*ab-aa-bb

rows=[]; drows=[]; lrows=[]; srows=[]; crows=[]
def cost(seed,arm,sec): crows.append(dict(seed=seed,arm=arm,sec=float(sec)))
for seed in SEEDS:
    rng=np.random.default_rng(seed)
    # hold out NEV eval cells per type, then PHYSICALLY DELETE the probe class's surplus training
    # cells. In cfgen_aug.py those cells stayed in the trunk under an UNLAB token, so the autoencoder
    # and the unconditional flow still saw the manifold and only the label index was withheld. Here
    # nothing sees them. That is what a genuinely rare cell type looks like to the generator.
    # DONOR-DISJOINT: take whole donors for eval until every probe class has NEV eval cells.
    # The remaining donors are the only source of training cells, so no donor is in both sides.
    perm=rng.permutation(UD); cnt={c:0 for c in PROBE}; ev_d=[]
    for d in perm:
        if all(v>=NEV for v in cnt.values()): break
        ev_d.append(d); m=dall==d
        for c in PROBE: cnt[c]+=int((A.obs.ct.values[m]==c).sum())
    ev_mask=np.isin(dall,ev_d); tr_mask=~ev_mask
    assert not (set(dall[ev_mask]) & set(dall[tr_mask])), 'donor appears on both sides'
    ev_idx=[]; full_tr=[]; keep=[]
    probe=np.array(sorted(ci[c] for c in PROBE))                       # fixed: the 8 rarest AIDA types
    rung_of=dict(zip(probe.tolist(), rng.permutation(RUNGS).tolist())) # rung<->type reshuffled per seed
    for c in range(NC):
        ep=rng.permutation(np.where((yall==c)&ev_mask)[0]); ev_idx.append(ep[:NEV])
        rest=rng.permutation(np.where((yall==c)&tr_mask)[0])
        full_tr.append(rest)                       # pre-drop pool: fixes the centroid yardstick only
        if c in rung_of:
            r=rung_of[c]
            assert len(rest)>=r, f'{cts[c]}: {len(rest)} train cells, rung {r} does not fit'
            rest=rest[:r]
        else:
            rest=rest[:CAP]                        # cap controls so scDesign3 is feasible at all
        keep.append(rest)
    ev_idx=np.concatenate(ev_idx); full_tr=np.concatenate(full_tr); tr_idx=np.concatenate(keep)
    ytr=yall[tr_idx].copy()
    n_c=np.array([int((ytr==c).sum()) for c in range(NC)])
    assert all(n_c[c]==r for c,r in rung_of.items()), 'physical drop did not land on the rung exactly'
    print(f"  seed {seed}: {len(ev_d)}/{len(UD)} eval donors, {len(tr_idx):,} train cells "
          f"(deleted {len(full_tr)-len(tr_idx):,}), probe rungs {sorted(n_c[probe])}",flush=True)
    print(f"    donor overlap train/eval: {len(set(dall[tr_idx]) & set(dall[ev_idx]))} donors",flush=True)
    lsf=np.log(Xc[tr_idx].sum(1))
    sfs={'mn':float(lsf.min()),'mx':float(lsf.max()),
         'mu':torch.tensor([lsf[ytr==c].mean() if (ytr==c).any() else lsf.mean() for c in range(NCAT)],dtype=torch.float32),
         'sd':torch.tensor([max(lsf[ytr==c].std(),1e-2) if (ytr==c).sum()>1 else lsf.std() for c in range(NCAT)],dtype=torch.float32)}
    CENT=np.stack([Xn[full_tr[yall[full_tr]==c]].mean(0) for c in range(NC)])  # pre-drop, never trained on
    if seed==SEEDS[0]:
        hit=[(np.linalg.norm(Xn[ev_idx[yall[ev_idx]==c]][:,None]-CENT[None],axis=2).argmin(1)==c).mean()
             for c in range(NC)]
        print(f"  real held-out cell purity ceiling {np.mean(hit):.3f} (chance {1/NC:.3f})",flush=True)
    _tae=time.time(); ae=build_ae(tr_idx,seed); _ae_sec=time.time()-_tae
    print(f"  seed {seed} AE done {time.time()-t0:.0f}s",flush=True)
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    _tc=time.time(); fm=build_fm(ae,'rand',tr_idx,ytr,seed,sfs); _fm_fit=time.time()-_tc
    print(f"  seed {seed} FM done {time.time()-t0:.0f}s",flush=True)

    lab=np.arange(len(tr_idx))                     # no UNLAB here: every surviving cell is labelled
    pca=PCA(50,random_state=seed).fit(Xn[tr_idx[rng.choice(len(tr_idx),min(30000,len(tr_idx)),replace=False)]])
    Ztr=pca.transform(Xn[tr_idx[lab]]); ytr_l=ytr[lab]
    Zev=pca.transform(Xn[ev_idx]); yev=yall[ev_idx]
    SAVE_GENE=os.environ.get('SAVE_GENE')
    if SAVE_GENE:   # remember each generator block's gene-space input, keyed by the PCA output object arm() receives
        os.makedirs(SAVE_GENE,exist_ok=True); REC={}; _pt=pca.transform
        def _rec(X,_pt=_pt,REC=REC):
            out=_pt(X)
            if len(X)<=TARGET: REC[id(out)]=np.asarray(X,np.float32).copy()   # live objects never share an id
            return out
        pca.transform=_rec
        pm=np.isin(yall[full_tr],probe)
        np.savez_compressed(f'{SAVE_GENE}/s{seed}_base.npz',Xtr=Xn[tr_idx[lab]],ytr=ytr_l,Ztr=Ztr,Xev=Xn[ev_idx],yev=yev,
            Xfull=Xn[full_tr[pm]],yfull=yall[full_tr[pm]],probe=probe,n_c=n_c,cts=np.array(cts),
            comp=pca.components_,pmean=pca.mean_,genes=np.array(A.var_names).astype(str))

    # ---- scDesign3 arm: same training cells, same targets, conditioned on cell_type only ----
    # temp dir lives beside the script, NOT under the system temp tree, which can be wiped while a run is in progress.
    os.makedirs('sd3_tmp',exist_ok=True)
    def run_sd3(drop, tag):
        """Fit scDesign3 on the training cells whose class is not in `drop`. Returns (arm, err)."""
        sel=~np.isin(ytr,list(drop)) if drop else np.ones(len(ytr),bool)
        want=pd.DataFrame([dict(cell_type=cts[c],n=TARGET-int(n_c[c]))
                           for c in probe if TARGET-int(n_c[c])>0 and c not in drop])
        if not len(want): return None,'nothing to generate'
        wd=tempfile.mkdtemp(prefix=f'sd3_{seed}_{tag}_', dir='sd3_tmp')
        scipy.io.mmwrite(os.path.join(wd,'train_counts.mtx'),
                         scipy.sparse.csr_matrix(Xc[tr_idx[sel]].T))     # genes x cells
        pd.DataFrame(dict(cell_type=[cts[i] for i in ytr[sel]])).to_csv(
            os.path.join(wd,'train_meta.csv'),index=False)
        want.to_csv(os.path.join(wd,'want.csv'),index=False)
        t=time.time()
        r=subprocess.run([RS_BIN,'sd3_arm.R',wd,str(SD3_CORES),str(seed)],capture_output=True,text=True)
        cost(seed, 'sd3' if tag=='all' else 'sd3_nosingle', time.time()-t)
        if r.returncode!=0:
            err=r.stderr.strip().replace(chr(10),' | ')[-500:]
            print(f"  seed {seed} sd3[{tag}] FAILED after {time.time()-t:.0f}s: {err}",flush=True)
            return None,err
        S=scipy.io.mmread(os.path.join(wd,'sim_counts.mtx')).toarray().T.astype(np.float32)
        sm=pd.read_csv(os.path.join(wd,'sim_meta.csv')).cell_type.values
        assert len(sm)==len(S)==int(want.n.sum()), (len(sm),len(S),int(want.n.sum()))
        # guard against the ASI fallback trap: synthetic cells must not BE real cells
        assert not any(np.allclose(S[0],Xc[i]) for i in tr_idx[:300]), 'sd3 returned real cells'
        Sn=np.log1p(S/np.maximum(S.sum(1,keepdims=True),1)*1e4).astype(np.float32)
        print(f"  seed {seed} sd3[{tag}] ok {time.time()-t:.0f}s, {len(S)} cells",flush=True)
        return {int(ci[c]):pca.transform(Sn[sm==c]) for c in want.cell_type}, ''

    single=[int(c) for c in range(NC) if n_c[c]<2]        # classes the copula cannot fit at all
    sd3,err_all=run_sd3(set(), 'all')
    sd3_drop,err_drop=(None,'not attempted')
    if sd3 is None and single:
        # isolate the cause: same data minus the singleton classes
        sd3_drop,err_drop=run_sd3(set(single), 'nosingle')
    srows.append(dict(seed=seed, n_singleton=len(single),
                      singleton_types='; '.join(cts[c] for c in single),
                      all_classes_ok=sd3 is not None, err_all=err_all,
                      nosingle_ok=sd3_drop is not None, err_nosingle=err_drop))

    gen={}                                          # synthetic cells per probe class, per w
    for w in WGRID:
        for c in probe:
            k=TARGET-int(n_c[c])
            if k<=0: continue
            out=[]
            for i in range(0,k,128):
                b=min(128,k-i)
                sm=fm.sample(batch_size=b,n_sample_steps=100,theta_covariate="cell_type",
                    size_factor_covariate="cell_type",conditioning_covariates=["cell_type"],
                    covariate_indices={"cell_type":torch.full((b,),int(c),dtype=torch.long)},
                    guidance_weights={"cell_type":float(w)})["rna"].numpy()
                out.append(np.log1p(sm/np.maximum(sm.sum(1,keepdims=True),1)*1e4).astype(np.float32))
            gen[(w,int(c))]=pca.transform(np.vstack(out))
    cost(seed,'synth_w1',time.time()-_tc)   # FM fit + sampling
    print(f"  seed {seed} generated {time.time()-t0:.0f}s",flush=True)

    # ---- the graft arms: bypass the flow, keep the decoder ----------------------------
    def flow_sample(model,c,k,w=1.0):
        out=[]
        for i in range(0,k,128):
            b=min(128,k-i)
            out.append(model.sample(batch_size=b,n_sample_steps=100,theta_covariate="cell_type",
                size_factor_covariate="cell_type",conditioning_covariates=["cell_type"],
                covariate_indices={"cell_type":torch.full((b,),int(c),dtype=torch.long)},
                guidance_weights={"cell_type":float(w)})["rna"].numpy())
        return np.vstack(out).astype(np.float32)

    def to_pca(S):
        return pca.transform(np.log1p(S/np.maximum(S.sum(1,keepdims=True),1)*1e4).astype(np.float32))

    lb={}; lg={}
    # Both bypass the flow entirely: real latents, then the SHARED decoder.
    # lat_boot puts the latents exactly where the real cells are, so it isolates the decoder.
    # lat_gauss estimates mean and diagonal sd from that class's own n_c cells and nothing else,
    # which is scDesign3's per-class marginal moved into the deep model's latent space.
    for c in probe:
        k=TARGET-int(n_c[c])
        if k<=0: continue
        idx_c=tr_idx[ytr==c]
        ll=np.log(Xc[idx_c].sum(1))
        L=torch.from_numpy(np.exp(rng.normal(ll.mean(), max(ll.std(),1e-2) if len(ll)>1 else 0.1,
                                             k)).astype(np.float32))[:,None]
        boot=torch.from_numpy(rng.integers(0,len(idx_c),k))
        for enc,dst in ((ae,lb),):
            with torch.no_grad(): z=enc.encode(mk_batch(idx_c))["rna"]
            dst[int(c)]=to_pca(decode_latents(enc, z[boot], L))
        with torch.no_grad(): zr=ae.encode(mk_batch(idx_c))["rna"]
        mu_z=zr.mean(0,keepdim=True)
        sd_z=zr.std(0,keepdim=True).clamp_min(1e-3) if len(zr)>1 else torch.full_like(mu_z,1e-2)
        znew=mu_z+sd_z*torch.from_numpy(rng.normal(0,1,(k,zr.shape[1])).astype(np.float32))
        lg[int(c)]=to_pca(decode_latents(ae, znew, L))
    print(f"  seed {seed} latent arms done {time.time()-t0:.0f}s",flush=True)

    def _lib(c,k):
        """Per-class lognormal library draw, shared by every arm so dose AND depth match."""
        ic=tr_idx[ytr==c]; ll=np.log(Xc[ic].sum(1))
        return np.exp(rng.normal(ll.mean(), max(ll.std(),1e-2) if len(ll)>1 else 0.1,
                                 k)).astype(np.float32)

    # ---- CFGen at its best configuration: class-balanced minibatches --------------
    # gen15 measured this at +0.319 recall@200 over the shipped sampler. It is the only
    # knob on CFGen that moves the rare classes, so the comparison has to include it.
    _tc=time.time()
    fmb=build_fm(ae,'rand',tr_idx,ytr,seed,sfs,bal=True)
    genb={}
    for c in probe:
        k=TARGET-int(n_c[c])
        if k<=0: continue
        genb[int(c)]=to_pca(flow_sample(fmb,int(c),k,1.0))
    del fmb
    cost(seed,'synthbal_w1',time.time()-_tc)
    print(f"  seed {seed} CFGen balanced done {time.time()-t0:.0f}s",flush=True)

    # ---- scVI: cell type through batch_key ----------------------------------------
    _tc=time.time()
    scvi.settings.seed=seed
    Atr=ad.AnnData(Xc[tr_idx].copy(),
                   obs=pd.DataFrame({'cell_type':pd.Categorical([cts[i] for i in ytr])}))
    Ab=Atr.copy(); scvi.model.SCVI.setup_anndata(Ab, batch_key='cell_type')
    vae=scvi.model.SCVI(Ab, n_latent=LAT, n_hidden=256, n_layers=2)
    vae.train(max_epochs=SCVI_EPOCHS, batch_size=BS, accelerator='cpu',
              enable_model_summary=False, enable_progress_bar=False)
    vae.module.eval()
    bmap=list(vae.adata_manager.get_state_registry('batch').categorical_mapping)
    scv={}
    for c in probe:
        k=TARGET-int(n_c[c])
        if k<=0: continue
        bi=bmap.index(cts[c]); L=_lib(c,k)
        torch.manual_seed(seed+31+int(c))
        with torch.no_grad():
            # SCVI's generative exponentiates: it wants the LOG library.
            out=vae.module.generative(z=torch.randn(k,LAT),
                                      library=torch.from_numpy(np.log(L))[:,None],
                                      batch_index=torch.full((k,1),bi,dtype=torch.long))
            scv[int(c)]=to_pca(out['px'].sample().numpy().astype(np.float32))
    del vae
    cost(seed,'scvi',time.time()-_tc)
    print(f"  seed {seed} scVI done {time.time()-t0:.0f}s",flush=True)

    # ---- CondSCVI: cell type as a first-class label, per-class bias and prior -----
    _tc=time.time()
    scvi.settings.seed=seed
    Ac=Atr.copy(); scvi.model.CondSCVI.setup_anndata(Ac, labels_key='cell_type')
    cvae=scvi.model.CondSCVI(Ac, n_latent=LAT, n_hidden=256, n_layers=2)
    cvae.train(max_epochs=SCVI_EPOCHS, batch_size=BS, accelerator='cpu',
               enable_model_summary=False, enable_progress_bar=False)
    cvae.module.eval()
    lmap=list(cvae.adata_manager.get_state_registry('labels').categorical_mapping)
    scc={}
    for c in probe:
        k=TARGET-int(n_c[c])
        if k<=0: continue
        li=lmap.index(cts[c]); L=_lib(c,k)
        torch.manual_seed(seed+61+int(c))
        with torch.no_grad():
            # CondSCVI multiplies: px_rate = library * px_scale, so it wants the RAW library.
            out=cvae.module.generative(z=torch.randn(k,LAT),
                                       library=torch.from_numpy(L)[:,None],
                                       y=torch.full((k,1),li,dtype=torch.long),
                                       batch_index=torch.zeros(k,1,dtype=torch.long))
            scc[int(c)]=to_pca(out['px'].sample().numpy().astype(np.float32))
    del cvae
    cost(seed,'scvi_cond',time.time()-_tc)
    print(f"  seed {seed} CondSCVI done {time.time()-t0:.0f}s",flush=True)

    # ---- per-class NB: no decoder, no latent space, no gene-gene dependence -------
    # Per-gene rate from that class's n_c cells only; dispersion pooled across all training
    # cells because theta is unestimable from 2 cells. scDesign3 with the copula removed.
    _tc=time.time()
    Ltr=Xc[tr_idx].sum(1,keepdims=True); med=np.median(Ltr)
    Xs=Xc[tr_idx]/Ltr*med
    m_g=Xs.mean(0); v_g=Xs.var(0)
    th_g=np.clip(np.where(v_g>m_g*1.001, m_g**2/np.maximum(v_g-m_g,1e-9), 1e6), 1e-2, 1e6)
    nb={}
    for c in probe:
        k=TARGET-int(n_c[c])
        if k<=0: continue
        sel=(ytr==c)
        rate=(Xc[tr_idx][sel]/Ltr[sel]).mean(0)
        L=_lib(c,k)[:,None].astype(np.float64)
        mu=np.maximum(rate[None,:]*L,1e-12)
        S=rng.negative_binomial(th_g[None,:], th_g[None,:]/(th_g[None,:]+mu)).astype(np.float32)
        nb[int(c)]=to_pca(S)
    cost(seed,'nb',time.time()-_tc)
    print(f"  seed {seed} per-class NB done {time.time()-t0:.0f}s",flush=True)

    # ---- scDiffusion: a third family, and a third conditioning mechanism ----------
    # CFGen conditions by label embedding in the flow, scVI by a decoder covariate,
    # scDiffusion by a CLASSIFIER GRADIENT on noisy latents. Its own optimizer is used, because
    # Adam with coupled weight decay cripples their VAE.
    _tc=time.time()
    SM=build_sdiff(Xn[tr_idx], ytr, NCAT, seed, *SDIFF_EP, bs=BS,
                   vae_steps=VAE_STEPS, vae_ckpt=f'vae_ckpt/g16_s{seed}_{VAE_STEPS}.pt')
    sdf={}
    for c in probe:
        k=TARGET-int(n_c[c])
        if k<=0: continue
        sdf[int(c)]=pca.transform(sample_sdiff(SM, int(c), k))   # already log1p-CPM
    del SM
    cost(seed,'sdiff',time.time()-_tc)
    print(f"  seed {seed} scDiffusion done {time.time()-t0:.0f}s",flush=True)

    def arm(name):
        Z,y=[Ztr],[ytr_l]; GB=[]
        for c in probe:
            k=TARGET-int(n_c[c])
            if k<=0: continue
            if name=='real': continue
            if name=='upsample':
                pos=np.where(ytr_l==c)[0]; sel=rng.choice(pos,k,replace=True)
                Z.append(Ztr[sel])
            elif name=='sd3_nosingle':
                if sd3_drop is None or int(c) not in sd3_drop: continue
                Z.append(sd3_drop[int(c)])
            elif name=='sd3':
                Z.append(sd3[int(c)])
            elif name=='lat_boot':
                Z.append(lb[int(c)])
            elif name=='lat_gauss':
                Z.append(lg[int(c)])
            elif name=='synthbal_w1':
                Z.append(genb[int(c)])
            elif name=='scvi':
                Z.append(scv[int(c)])
            elif name=='scvi_cond':
                Z.append(scc[int(c)])
            elif name=='nb':
                Z.append(nb[int(c)])
            elif name=='sdiff':
                Z.append(sdf[int(c)])
            else:
                Z.append(gen[(float(name.split('_w')[1]),int(c))])
            if SAVE_GENE: GB.append(Xn[tr_idx[lab]][sel] if name=='upsample' else REC[id(Z[-1])])
            y.append(np.full(k,c,dtype=np.int64))
        if SAVE_GENE and GB:
            np.savez_compressed(f'{SAVE_GENE}/s{seed}_{name}.npz',X=np.vstack(GB),y=np.concatenate(y[1:]))
        Z=np.vstack(Z); y=np.concatenate(y)
        clf=LogisticRegression(max_iter=2000,n_jobs=8,random_state=seed).fit(Z,y)
        P=clf.predict_proba(Zev); cls=list(clf.classes_); pred=clf.classes_[P.argmax(1)]
        for c in range(NC):
            true=(yev==c); j=cls.index(c) if c in cls else None
            r200=np.nan if j is None else float(true[np.argsort(-P[:,j])[:NEV]].mean())
            rows.append(dict(seed=seed,arm=name,c=c,name_=cts[c],n=int(n_c[c]),
                             probe=bool(c in set(probe.tolist())),
                             rec_argmax=float((pred[true]==c).mean()),rec200=r200,
                             called=int((pred==c).sum())))

    for a in ('real','upsample','nb','synth_w1','synthbal_w1','scvi','scvi_cond','sdiff',
              'lat_boot','lat_gauss')+(('sd3',) if sd3 else ())+(('sd3_nosingle',) if sd3_drop else ()):
        arm(a); print(f"  seed {seed} {a} {time.time()-t0:.0f}s",flush=True)
    cost(seed,'real',0.0); cost(seed,'upsample',0.0)
    cost(seed,'lat_boot',_ae_sec); cost(seed,'lat_gauss',_ae_sec)
    cost(seed,'ae_shared',_ae_sec)   # CFGen and both lat arms all pay this once
    pd.DataFrame(crows).to_csv('gen16m_cost.csv',index=False)
    # ---- displacement geometry + iLISI, PCA-50, the space the probe sees ---------------
    pset=set(probe.tolist()); ab=np.array([c for c in range(NC) if c not in pset])
    CEV=np.stack([Zev[yev==c].mean(0) for c in range(NC)])
    c_glob=Zev[np.isin(yev,ab)].mean(0)
    Rpool=Ztr[rng.choice(len(Ztr),min(3000,len(Ztr)),replace=False)]   # real backdrop for pooled iLISI
    for c in probe:
        k=TARGET-int(n_c[c])
        if k<=0: continue
        Rev=Zev[yev==c]; c_true=Rev.mean(0)
        sigma=float(np.linalg.norm(Rev-c_true,axis=1).mean())
        a_star=int(ab[np.linalg.norm(CEV[ab]-c_true,axis=1).argmin()]); c_near=CEV[a_star]
        pos=np.where(ytr_l==c)[0]
        sets={'train':Ztr[pos], 'upsample':Ztr[rng.choice(pos,k,replace=True)],
              'synth_w1':gen[(1.0,int(c))], 'synthbal_w1':genb[int(c)],
              'scvi':scv[int(c)], 'scvi_cond':scc[int(c)], 'nb':nb[int(c)], 'sdiff':sdf[int(c)],
              'lat_boot':lb[int(c)], 'lat_gauss':lg[int(c)]}
        if sd3: sets['sd3']=sd3[int(c)]
        if sd3_drop and int(c) in sd3_drop: sets['sd3_nosingle']=sd3_drop[int(c)]
        for nm,G in sets.items():
            delta=G.mean(0)-c_true
            drows.append(dict(seed=seed,c=int(c),name_=cts[c],n=int(n_c[c]),arm=nm,sigma=sigma,near=cts[a_star],
                d_norm=float(np.linalg.norm(delta)/sigma),
                pull_glob=float(delta@(c_glob-c_true)/np.linalg.norm(c_glob-c_true)**2),
                pull_near=float(delta@(c_near -c_true)/np.linalg.norm(c_near -c_true)**2)))
        # iLISI of synthetic-vs-real, two scopings. 1 = fully segregated, 2 = perfectly mixed.
        #   pooled  backdrop is ALL real cells, so landing anywhere dense scores well
        #   within   backdrop is the held-out real cells OF THAT CLASS, so only the right place scores
        for nm in ('upsample','nb','sd3','sd3_nosingle','synth_w1','synthbal_w1',
                   'scvi','scvi_cond','sdiff','lat_boot','lat_gauss'):
            if nm not in sets: continue
            G=sets[nm]
            for scope,R in (('pooled',Rpool),('within',Rev)):
                Z=np.vstack([R,G])
                md=pd.DataFrame({'src':['real']*len(R)+['syn']*len(G)})
                v=compute_lisi(Z,md,['src'],perplexity=min(30,(len(Z)-2)//3))[:,0]
                lrows.append(dict(seed=seed,c=int(c),name_=cts[c],n=int(n_c[c]),arm=nm,scope=scope,
                                  ilisi=float(v[len(R):].mean())))
    pd.DataFrame(drows).to_csv('gen16m_disp.csv',index=False)
    pd.DataFrame(lrows).to_csv('gen16m_ilisi.csv',index=False)
    print(f"  seed {seed} geometry + iLISI done {time.time()-t0:.0f}s",flush=True)
    pd.DataFrame(rows).to_csv('gen16m.csv',index=False)   # partial results survive a kill
    pd.DataFrame(srows).to_csv('gen16m_sd3status.csv',index=False)

S3=pd.DataFrame(srows); S3.to_csv('gen16m_sd3status.csv',index=False)
print('\n=== can scDesign3 run at all on a donor-disjoint split with a 1-cell class?')
for _,r in S3.iterrows():
    print(f"  seed {r.seed}: {r.n_singleton} singleton class(es) [{r.singleton_types}]")
    print(f"    all 31 classes  : {'RAN' if r.all_classes_ok else 'ABORTED'}"
          + ('' if r.all_classes_ok else f"  {r.err_all[-220:]}"))
    print(f"    singletons drop : {'RAN' if r.nosingle_ok else r.err_nosingle[:60]}")

T=pd.DataFrame(rows); T.to_csv('gen16m.csv',index=False)
ARMS=tuple(a for a in ('real','upsample','nb','sd3','sd3_nosingle','synth_w1','synthbal_w1','scvi','scvi_cond','sdiff','lat_boot','lat_gauss') if a in set(T.arm))
P=T[T.probe]
print("\n=== recall@200 on held-out cells, 8 NATURALLY RARE probe classes, mean over seeds")
print(f"{'n_c':>6} " + ' '.join(f'{a:>10}' for a in ARMS))
for r in sorted(P.n.unique()):
    d=P[P.n==r]
    print(f"{r:>6} " + ' '.join(f"{d[d.arm==a].rec200.mean():>10.3f}" for a in ARMS))
print("\n=== gain over the real-only arm, paired per (seed, class)")
from scipy import stats
piv=P.pivot_table(index=['seed','c','n'],columns='arm',values='rec200').reset_index()
for lo,hi in [(0,8),(8,64),(64,257)]:
    d=piv[(piv.n>lo)&(piv.n<=hi)].dropna()
    if len(d)<3: continue
    out=[]
    for a in ARMS[1:]:
        dl=d[a]-d['real']; t=stats.ttest_rel(d[a],d['real']).statistic
        out.append(f"{a} {dl.mean():+.3f} (t {t:+.2f})")
    print(f"  n_c {lo+1}-{hi} (N={len(d)}): " + '  '.join(out))
print("\n=== does synthesis beat the dose-matched upsampling baseline?")
for lo,hi in [(0,8),(8,64),(64,257)]:
    d=piv[(piv.n>lo)&(piv.n<=hi)].dropna()
    if len(d)<3: continue
    o=[]
    for a in ARMS[2:]:
        dl=d[a]-d['upsample']; o.append(f"{a} vs upsample {dl.mean():+.3f} (t {stats.ttest_rel(d[a],d['upsample']).statistic:+.2f})")
    print(f"  n_c {lo+1}-{hi}: " + '  '.join(o))
print("\n=== control classes (never augmented) -- sanity, should not move")
C=T[~T.probe]
print('  ' + '  '.join(f"{a} {C[C.arm==a].rec200.mean():.3f}" for a in ARMS))


D=pd.DataFrame(drows); D.to_csv('gen16m_disp.csv',index=False)
L=pd.DataFrame(lrows); L.to_csv('gen16m_ilisi.csv',index=False)
GARMS=[a for a in ('train','upsample','nb','sd3','sd3_nosingle','synth_w1','synthbal_w1',
                   'scvi','scvi_cond','sdiff','lat_boot','lat_gauss') if a in set(D.arm)]
BINS=((1,8,'1-8'),(9,64,'9-64'),(65,257,'65+'))
for stat,lab in (('d_norm','distance from the true class centroid, in units of class spread'),
                 ('pull_glob','fraction of the way toward the abundant-cell centroid')):
    print(f"\n=== {stat}: {lab}")
    print('  n_c     '+''.join(f'{a:>11s}' for a in GARMS))
    for lo,hi,nm in BINS:
        d=D[(D.n>=lo)&(D.n<=hi)]
        print(f'  {nm:6s}  '+''.join(f'{d[d.arm==a][stat].mean():11.3f}' for a in GARMS))

print("\n=== paired vs the upsampling null, mean and t")
for stat in ('d_norm','pull_glob'):
    print(f'  {stat}')
    for lo,hi,nm in BINS:
        out=[]
        p=D[(D.n>=lo)&(D.n<=hi)].pivot_table(index=['seed','c'],columns='arm',values=stat)
        for a in GARMS[2:]:
            if a not in p.columns or 'upsample' not in p.columns: continue
            v=(p[a]-p['upsample']).dropna()
            if len(v)<2: continue
            out.append(f'{a} {v.mean():+.3f} (t {v.mean()/(v.std(ddof=1)/np.sqrt(len(v))):+.2f})')
        print(f'    n_c {nm:5s}: '+'  '.join(out))

LARMS=[a for a in ('upsample','nb','sd3','sd3_nosingle','synth_w1','synthbal_w1',
                   'scvi','scvi_cond','sdiff','lat_boot','lat_gauss') if a in set(L.arm)]
print("\n=== iLISI of synthetic vs real cells (1 = segregated, 2 = perfectly mixed)")
for scope,note in (('pooled','backdrop = all real cells, so landing anywhere dense scores well'),
                   ('within','backdrop = held-out real cells OF THAT CLASS')):
    print(f'  {scope}: {note}')
    print('    n_c   '+''.join(f'{a:>11s}' for a in LARMS))
    for lo,hi,nm in BINS:
        d=L[(L.scope==scope)&(L.n>=lo)&(L.n<=hi)]
        print(f'    {nm:5s} '+''.join(f'{d[d.arm==a].ilisi.mean():11.3f}' for a in LARMS))


# =====================================================================================
# THE CLAIM: does duplicating real cells beat every generative model?
# =====================================================================================
# Three numbers per arm, because no single one answers it:
#   delta     mean recall@200 minus upsampling, paired per (seed, class)
#   t_CLUST   averaged within cell type FIRST, then t over the 8 types. With only 8 distinct
#             types, reusing them across seeds does not create independent observations;
#             the naive t runs 2-3x larger on identical data.
#   win       fraction of (seed, class) pairs where upsampling is at least as good
#   cost      median wall-clock seconds to produce one seed's synthetic cells
C=pd.DataFrame(crows); C.to_csv('gen16m_cost.csv',index=False)
COST=C.groupby('arm').sec.median().to_dict()

def clust_t(d, a, b):
    """Average the paired delta within cell type, then t over cell types."""
    v=d.assign(_dl=d[a]-d[b]).groupby('name_')._dl.mean().dropna()
    if len(v)<2: return np.nan, np.nan, np.nan, 0
    se=v.std(ddof=1)/np.sqrt(len(v))
    t=v.mean()/se if se>0 else np.nan
    return float(v.mean()), float(t), float(2*stats.t.sf(abs(t),len(v)-1)), len(v)

piv2=P.pivot_table(index=['seed','c','n','name_'],columns='arm',values='rec200').reset_index()
BANDS=[('rare  n_c<=8',0,8),('abund n_c>=16',15,10**6),('ALL rungs',0,10**6)]
print('\n'+'='*100)
print('THE CLAIM: upsampling (duplicating real cells) against every generative model')
print('='*100)
for lab,lo,hi in BANDS:
    d=piv2[(piv2.n>lo)&(piv2.n<=hi)]
    if not len(d): continue
    print(f'\n### {lab}   (N={len(d)} seed x class pairs)')
    print(f"  {'arm':<14}{'rec200':>8}{'delta':>9}{'t_CLUST':>9}{'p':>8}{'ups wins':>10}{'cost s':>9}")
    for a in ARMS:
        if a=='upsample':
            v=d['upsample'].dropna()
            print(f"  {a:<14}{v.mean():>8.3f}{'--':>9}{'':>9}{'':>8}{'':>10}{COST.get(a,float('nan')):>9.0f}")
            continue
        dd=d[[a,'upsample','name_']].dropna()
        if not len(dd): continue
        m,t,p,nt=clust_t(dd,a,'upsample')
        win=float((dd['upsample']>=dd[a]).mean())
        print(f"  {a:<14}{dd[a].mean():>8.3f}{m:>+9.4f}{t:>+9.2f}{p:>8.3f}{win:>10.2f}{COST.get(a,float('nan')):>9.0f}")

print('\n### whole matrix, all %d classes (macro over classes, then over seeds)'%NC)
for met in ('rec200','rec_argmax'):
    print('  '+met+'  '+'  '.join(f"{a} {T[T.arm==a].groupby(['seed','c'])[met].mean().groupby('seed').mean().mean():.4f}" for a in ARMS))

print('\n### per rung, recall@200')
print(f"{'n_c':>5} " + ' '.join(f'{a:>12}' for a in ARMS))
for r in sorted(P.n.unique()):
    d=P[P.n==r]
    print(f"{r:>5} " + ' '.join(f"{d[d.arm==a].rec200.mean():>12.3f}" for a in ARMS))

print('\n### wall-clock seconds to produce one seed of synthetic cells (median over seeds)')
for a in ARMS:
    if a in COST: print(f'  {a:<14}{COST[a]:>9.0f}')
