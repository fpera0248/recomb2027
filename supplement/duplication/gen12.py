"""gen12: scVI is also given the cell type label. Why is it worse than upsampling?

Jiaqi's question. scVI takes a label, scDesign3 takes a label, so on paper they ask the model
for the same thing. scDesign3 lands near the class and scVI does not. Three candidate reasons,
and CondSCVI happens to ship exactly the three fixes, so they can be switched on one at a time
on one identical donor-disjoint split at one identical dose:

    scvi_batch      label handed in through batch_key. This is what gen5 and gen7 measured.
    scvi_cond       CondSCVI. Encoder and decoder are label conditioned and there is a per
                    label, per gene additive bias in gene space. The prior is still a shared
                    N(0, I), so z itself carries no class information.
    scvi_cond_mog   + a per label mixture prior, so the latent draw is class specific too.
    scvi_cond_w     + per class loss weighting (weight_obs). A 2 cell class otherwise gets
                    about 1/250th of the gradient share of a 500 cell class, because the loss
                    is a per cell mean.
    scvi_cond_boot  CondSCVI's decoder fed REAL latents. Separates a bad sampler from a bad
                    decoder, the same control that closed the CFGen question in gen9.

Reference lines carried over unchanged from gen12: upsample, lat_rt, lat_boot, lat_gauss,
lat_gauss_pool. 6 seeds, because four scVI models are trained per seed. Every contrast is
reported naively AND class clustered, since there are only 8 distinct probe types.
"""
import sys, os, subprocess, tempfile, numpy as np, pandas as pd, torch, anndata as ad, scanpy as sc, time, warnings, pathlib
HR_WORK = os.environ.get("HR_WORK", os.path.expanduser("~/work"))
HR_DATA = os.environ.get("HR_DATA", os.path.expanduser("~/data"))
import scipy.io, scipy.sparse
import scvi
# sdiff/scvi arms dropped here: gen6 decomposes the CFGen stack, gen5 has the cross-family table
from harmonypy import compute_lisi
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_recall_curve
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
FM_EPOCHS = 2 if SMOKE else 60
RUNGS=np.array([1,2,4,8]) if '--smoke' in sys.argv else np.array([1,2,4,8,16,32,64,128])
WGRID=[1.0]
TARGET=64 if '--smoke' in sys.argv else 256   # every augmented arm reaches exactly this many cells
SEEDS=[0] if SMOKE else list(range(6))
RF_TREES=100 if SMOKE else 500        # cscGAN used 1000; 500 is within noise here and half the cost
PROBE=['double negative T regulatory cell','CD141-positive myeloid dendritic cell',
       'innate lymphoid cell','pre-conventional dendritic cell','plasma cell',
       'plasmacytoid dendritic cell','mature B cell',
       'CD16-negative, CD56-bright natural killer cell, human']   # the 8 rarest types in AIDA
if '--smoke' in sys.argv: PROBE=PROBE[:4]
D=HR_DATA + '/pilots/aida/'
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

def _f1_best(true, p):
    """Best F1 over all thresholds. An optimistic bound on what a free operating point can buy."""
    pr,rc,_=precision_recall_curve(true.astype(int), p)
    f=2*pr*rc/np.clip(pr+rc,1e-12,None)
    return float(np.nanmax(f))

rows=[]; drows=[]; lrows=[]; srows=[]
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
    ae=build_ae(tr_idx,seed); print(f"  seed {seed} AE done {time.time()-t0:.0f}s",flush=True)
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression

    lab=np.arange(len(tr_idx))                     # no UNLAB here: every surviving cell is labelled
    pca=PCA(50,random_state=seed).fit(Xn[tr_idx[rng.choice(len(tr_idx),min(30000,len(tr_idx)),replace=False)]])
    Ztr=pca.transform(Xn[tr_idx[lab]]); ytr_l=ytr[lab]
    Zev=pca.transform(Xn[ev_idx]); yev=yall[ev_idx]

    # ---- scDesign3 arm: same training cells, same targets, conditioned on cell_type only ----
    # temp dir lives beside the script, NOT under /private/var/folders: that tree gets wiped
    # on session teardown and took the last run's scratch with it.
    sd3=None; sd3_drop=None   # scDesign3 arms dropped, see docstring


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

    # pooled within-class sd in latent space: the "borrow the second moment" arm
    with torch.no_grad(): Zae=ae.encode(mk_batch(tr_idx))["rna"].numpy()
    Zctr=Zae.copy()
    for cc in range(NC):
        mm=ytr==cc
        if mm.sum()>1: Zctr[mm]-=Zae[mm].mean(0)
    SD_POOL=torch.from_numpy(np.maximum(Zctr.std(0),1e-3).astype(np.float32))[None]

    lb={}; lg={}; lr={}; lgp={}
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
        # lat_rt: pure autoencoder round trip. Each real cell decoded back with ITS OWN library
        # size, then duplicated to dose. Zero added variance, so at n_c=1 it is k copies of one
        # reconstruction. This is upsampling with the AE in the middle and nothing else.
        Lown=torch.from_numpy(Xc[idx_c].sum(1).astype(np.float32))[:,None]
        rt=to_pca(decode_latents(ae, zr, Lown))
        lr[int(c)]=rt[rng.integers(0,len(rt),k)]
        # lat_gauss_pool: own mean, second moment BORROWED from the pooled within-class sd.
        zp=mu_z+SD_POOL*torch.from_numpy(rng.normal(0,1,(k,zr.shape[1])).astype(np.float32))
        lgp[int(c)]=to_pca(decode_latents(ae, zp, L))
    print(f"  seed {seed} latent arms done {time.time()-t0:.0f}s",flush=True)

    # ---- scVI arms: Jiaqi's #2 ------------------------------------------------------
    Atr=ad.AnnData(Xc[tr_idx].copy(),
                   obs=pd.DataFrame({'cell_type':pd.Categorical([cts[i] for i in ytr])}))
    sb={}; sc1={}; scm={}; scw={}; sbo={}

    def _lib(c,k):
        """Same per-class lognormal library draw the latent arms use, so dose AND depth match."""
        ic=tr_idx[ytr==c]; ll=np.log(Xc[ic].sum(1))
        return np.exp(rng.normal(ll.mean(), max(ll.std(),1e-2) if len(ll)>1 else 0.1,
                                 k)).astype(np.float32)

    scvi.settings.seed=seed
    Ab=Atr.copy(); scvi.model.SCVI.setup_anndata(Ab, batch_key='cell_type')
    vae=scvi.model.SCVI(Ab, n_latent=LAT, n_hidden=256, n_layers=2)
    vae.train(max_epochs=SCVI_EPOCHS, batch_size=BS, accelerator='cpu',
              enable_model_summary=False, enable_progress_bar=False)
    vae.module.eval()
    bmap=list(vae.adata_manager.get_state_registry('batch').categorical_mapping)
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
            sb[int(c)]=to_pca(out['px'].sample().numpy().astype(np.float32))
    print(f"  seed {seed} scvi_batch done {time.time()-t0:.0f}s",flush=True)

    def cond_fit(weight_obs, prior):
        scvi.settings.seed=seed
        Ac=Atr.copy(); scvi.model.CondSCVI.setup_anndata(Ac, labels_key='cell_type')
        m=scvi.model.CondSCVI(Ac, n_latent=LAT, n_hidden=256, n_layers=2,
                              weight_obs=weight_obs, prior=prior)
        m.train(max_epochs=SCVI_EPOCHS, batch_size=BS, accelerator='cpu',
                enable_model_summary=False, enable_progress_bar=False)
        m.module.eval(); return m

    def cond_sample(m, c, k, real_latents=False):
        lmap=list(m.adata_manager.get_state_registry('labels').categorical_mapping)
        li=lmap.index(cts[c]); L=_lib(c,k)
        y=torch.full((k,1),li,dtype=torch.long)
        torch.manual_seed(seed+61+int(c))
        with torch.no_grad():
            if real_latents:
                ic=tr_idx[ytr==c]
                inf=m.module.inference(torch.from_numpy(Xc[ic]),
                                       torch.full((len(ic),1),li,dtype=torch.long))
                zr=inf['qz'].loc
                z=zr[torch.from_numpy(rng.integers(0,len(zr),k))]
            elif getattr(m.module,'prior','normal')=='mog':
                mu=m.module.prior_means[li]; sd=torch.exp(m.module.prior_log_std[li])+1e-4
                comp=torch.distributions.Categorical(logits=m.module.prior_logits[li]).sample((k,))
                z=mu[comp]+sd[comp]*torch.randn(k,LAT)
            else:
                z=torch.randn(k,LAT)
            # CondSCVI's generative multiplies: px_rate = library * px_scale, so RAW library.
            out=m.module.generative(z=z, library=torch.from_numpy(L)[:,None], y=y,
                                    batch_index=torch.zeros(k,1,dtype=torch.long))
            return to_pca(out['px'].sample().numpy().astype(np.float32))

    for tag,dst,kw in (('scvi_cond',sc1,dict(weight_obs=False,prior='normal')),
                       ('scvi_cond_mog',scm,dict(weight_obs=False,prior='mog')),
                       ('scvi_cond_w',scw,dict(weight_obs=True,prior='normal'))):
        m=cond_fit(**kw)
        for c in probe:
            k=TARGET-int(n_c[c])
            if k<=0: continue
            dst[int(c)]=cond_sample(m,c,k)
            if tag=='scvi_cond': sbo[int(c)]=cond_sample(m,c,k,real_latents=True)
        print(f"  seed {seed} {tag} done {time.time()-t0:.0f}s",flush=True)
        del m


    def build(name):
        """Assemble one arm's training matrix. Identical to gen9; only the readout changes."""
        Z,y=[Ztr],[ytr_l]
        for c in probe:
            k=TARGET-int(n_c[c])
            if k<=0: continue
            if name=='real': continue
            if name=='upsample':
                pos=np.where(ytr_l==c)[0]
                Z.append(Ztr[rng.choice(pos,k,replace=True)])
            elif name=='lat_boot':       Z.append(lb[int(c)])
            elif name=='lat_gauss':      Z.append(lg[int(c)])
            elif name=='lat_rt':         Z.append(lr[int(c)])
            elif name=='lat_gauss_pool': Z.append(lgp[int(c)])
            elif name=='scvi_batch':     Z.append(sb[int(c)])
            elif name=='scvi_cond':      Z.append(sc1[int(c)])
            elif name=='scvi_cond_mog':  Z.append(scm[int(c)])
            elif name=='scvi_cond_w':    Z.append(scw[int(c)])
            elif name=='scvi_cond_boot': Z.append(sbo[int(c)])
            else: raise ValueError(name)
            y.append(np.full(k,c,dtype=np.int64))
        return np.vstack(Z), np.concatenate(y)

    def score(name, cfg, c, true, p_c, pred_c):
        """One (arm, readout, class) row. p_c ranks cells, pred_c is the hard call."""
        return dict(seed=seed, arm=name, cfg=cfg, c=c, name_=cts[c], n=int(n_c[c]),
                    probe=bool(c in set(probe.tolist())),
                    rec200=float(true[np.argsort(-p_c)[:NEV]].mean()),
                    rec_argmax=float(pred_c[true].mean()),
                    prec=float(true[pred_c].mean()) if pred_c.sum() else 0.0,
                    f1=float(f1_score(true, pred_c, zero_division=0)),
                    f1_best=_f1_best(true, p_c),
                    called=int(pred_c.sum()))

    def arm(name):
        Z,y=build(name)
        # --- 31-class, logistic regression: ours -------------------------------
        clf=LogisticRegression(max_iter=2000,n_jobs=8,random_state=seed).fit(Z,y)
        P=clf.predict_proba(Zev); cls=list(clf.classes_); pred=clf.classes_[P.argmax(1)]
        for c in range(NC):
            j=cls.index(c) if c in cls else None
            if j is None: continue
            rows.append(score(name,'mc_lr',c,(yev==c),P[:,j],(pred==c)))
    for a in ('real','upsample','lat_rt','lat_boot','lat_gauss','lat_gauss_pool','scvi_batch','scvi_cond','scvi_cond_mog','scvi_cond_w','scvi_cond_boot'):
        arm(a); print(f"  seed {seed} {a} {time.time()-t0:.0f}s",flush=True)
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
              'lat_rt':lr[int(c)], 'lat_boot':lb[int(c)],
              'lat_gauss':lg[int(c)], 'lat_gauss_pool':lgp[int(c)],
              'scvi_batch':sb[int(c)], 'scvi_cond':sc1[int(c)],
              'scvi_cond_mog':scm[int(c)], 'scvi_cond_w':scw[int(c)],
              'scvi_cond_boot':sbo[int(c)]}
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
        for nm in ('upsample','lat_boot','lat_gauss','scvi_batch','scvi_cond',
                   'scvi_cond_mog','scvi_cond_w','scvi_cond_boot'):
            if nm not in sets: continue
            G=sets[nm]
            for scope,R in (('pooled',Rpool),('within',Rev)):
                Z=np.vstack([R,G])
                md=pd.DataFrame({'src':['real']*len(R)+['syn']*len(G)})
                v=compute_lisi(Z,md,['src'],perplexity=min(30,(len(Z)-2)//3))[:,0]
                lrows.append(dict(seed=seed,c=int(c),name_=cts[c],n=int(n_c[c]),arm=nm,scope=scope,
                                  ilisi=float(v[len(R):].mean())))
    pd.DataFrame(drows).to_csv('gen12_disp.csv',index=False)
    pd.DataFrame(lrows).to_csv('gen12_ilisi.csv',index=False)
    print(f"  seed {seed} geometry + iLISI done {time.time()-t0:.0f}s",flush=True)
    pd.DataFrame(rows).to_csv('gen12.csv',index=False)   # partial results survive a kill
    pd.DataFrame(srows).to_csv('gen12_sd3status.csv',index=False)

T=pd.DataFrame(rows); T.to_csv('gen12.csv',index=False)
P=T[T.probe & (T.cfg=='mc_lr')]
ARMS=[a for a in ('real','lat_rt','lat_boot','lat_gauss','lat_gauss_pool','scvi_batch',
                  'scvi_cond','scvi_cond_mog','scvi_cond_w','scvi_cond_boot') if a in set(P.arm)]
from scipy import stats

def contrast(mask, a, b, met='rec200'):
    """Paired a-minus-b on the probe rows selected by mask, two ways.

    naive  : every (seed, class) row is an independent observation. This is the number
             everyone reports and it is pseudoreplicated, because the same 8 cell types
             recur in every seed.
    clust  : average within CELL TYPE first, then test over the distinct types. With 8
             probe types the effective n is 8 no matter how many seeds we run, which is
             the honest denominator.
    """
    q = P[mask].pivot_table(index=['seed','name_'], columns='arm', values=met)
    if a not in q or b not in q: return None
    d = (q[a]-q[b]).dropna()
    if len(d) < 3: return None
    tn, pn = stats.ttest_1samp(d, 0)
    g = d.groupby(level='name_').mean()
    tc, pc = (stats.ttest_1samp(g, 0) if len(g) >= 3 else (float('nan'),)*2)
    return dict(mean=d.mean(), t_naive=tn, n_naive=len(d),
                t_clust=tc, p_clust=pc, n_clust=len(g))

BANDS=[('n_c<=2',P.n<=2),('n_c<=4',P.n<=4),('n_c<=8',P.n<=8),('n_c>=16',P.n>=16),('all rungs',P.n>0)]
print("\n"+"="*104)
print("EVERY ARM AGAINST DOSE-MATCHED DUPLICATION, including the four scVI conditioning designs")
print("Every cell is the arm minus dose-matched UPSAMPLING, recall@200 at a matched operating point.")
print("="*104)
for lab, m in BANDS:
    print(f"\n### {lab}")
    for a in ARMS:
        r = contrast(m, a, 'upsample')
        if r: print(f"  {a:16s} {r['mean']:+.4f}   naive t {r['t_naive']:+5.2f} (n {r['n_naive']:3d})"
                    f"   CLASS-CLUSTERED t {r['t_clust']:+5.2f} p {r['p_clust']:.3f} (n {r['n_clust']})")
    u = P[m & (P.arm=='upsample')].rec200.mean(); rl = P[m & (P.arm=='real')].rec200.mean()
    print(f"  {'[absolute]':16s} upsample {u:.4f}   real {rl:.4f}")

print("\n"+"="*104)
print("THE DECISIVE CONTRASTS: scVI gets the same label, so which part of it is failing")
print("="*104)
for lab, m in BANDS:
    print(f"\n### {lab}")
    for a, b, why in [('scvi_cond','scvi_batch','conditioning design: CondSCVI vs label-as-batch'),
                      ('scvi_cond_mog','scvi_cond','+ a per-label prior on z'),
                      ('scvi_cond_w','scvi_cond','+ per-class loss weighting (gradient share)'),
                      ('scvi_cond_boot','scvi_cond','REAL latents, same decoder: sampler or decoder'),
                      ('lat_boot','scvi_cond_boot','our decoder vs scVI decoder, both fed real latents'),
                      ('lat_gauss','scvi_cond','the whole method against the best scVI')]:
        r = contrast(m, a, b)
        if r: print(f"  {a:15s} - {b:15s} {r['mean']:+.4f}  clust t {r['t_clust']:+5.2f} p {r['p_clust']:.3f}   {why}")

print("\n"+"="*104); print("WHOLE MATRIX, all 31 classes (macro over classes, then over seeds)"); print("="*104)
W=T[T.cfg=='mc_lr']
for met in ('rec200','rec_argmax','f1'):
    g=W.groupby(['arm','seed'])[met].mean().groupby('arm').mean()
    print(f"  {met:11s} " + "  ".join(f"{a} {g.get(a,float('nan')):.4f}" for a in ['real','upsample']+ARMS[1:]))
print("\n  control (never-augmented) classes only, rec_argmax:")
g=W[~W.probe].groupby(['arm','seed']).rec_argmax.mean().groupby('arm').mean()
print("    " + "  ".join(f"{a} {g.get(a,float('nan')):.4f}" for a in ['real','upsample']+ARMS[1:]))

print("\n"+"="*104); print("PER-RUNG, recall@200"); print("="*104)
q=P.pivot_table(index='n',columns='arm',values='rec200')
print(q[[c for c in ['real','upsample']+ARMS[1:] if c in q]].round(3).to_string())

D=pd.DataFrame(drows)
print("\n=== pull_glob: fraction of the way toward the abundant centroid (0 good, 1 = landed on it)")
D['band']=pd.cut(D.n,[0,8,64,10**6],labels=['1-8','9-64','65+'])
print(D.pivot_table(index='band',columns='arm',values='pull_glob',observed=True).round(3).to_string())
print("\n=== d_norm: distance from the true class centroid in units of class spread")
print(D.pivot_table(index='band',columns='arm',values='d_norm',observed=True).round(3).to_string())
print("\nEXIT=0")
