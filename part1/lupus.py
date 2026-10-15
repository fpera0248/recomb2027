# lupus: aida.py's exact protocol ported to a second dataset
"""The AIDA head-to-head protocol on the lupus dataset (H5AD)."""
import sys, os, subprocess, tempfile, numpy as np, pandas as pd, torch, anndata as ad, scanpy as sc, time, warnings, pathlib
import scipy.io, scipy.sparse
import scvi
from diffusion import build_sdiff, sample_sdiff
from harmonypy import compute_lisi
warnings.filterwarnings('ignore')
import logging; logging.getLogger('lightning.pytorch').setLevel(logging.ERROR)
sys.path.insert(0,'CFGen')
from cfgen.models.base.encoder_model import EncoderModel
from cfgen.models.fm.denoising_model import MLPTimeStep
from cfgen.models.fm.fm import FM
from cfgen.models.featurizers.category_featurizer import CategoricalFeaturizer

torch.set_num_threads(8)
NJ = 8

SMOKE = '--smoke' in sys.argv
HVG=400 if SMOKE else 2000
NEV=50 if SMOKE else 200
LAT=50; EDIM=20; BS=256; NS=128
CAP=150 if SMOKE else 500
RS_BIN=os.environ.get('RSCRIPT','Rscript')
SD3_CORES=8   # shared-machine cap (aida used 8)
SD3_TIMEOUT=180 if SMOKE else int(os.environ.get('SD3_TIMEOUT',0))
AE_EPOCHS = 2 if SMOKE else 30
SCVI_EPOCHS = 2 if SMOKE else 60
SDIFF_EP = (2,2,2) if SMOKE else (40,60,30)
VAE_STEPS = 200 if SMOKE else 5000
FM_EPOCHS = 2 if SMOKE else 60
SIZES=np.array([1,2,4,8]) if SMOKE else np.array([1,2,4,8,16,32,64,128])
WGRID=[1.0]
TARGET=64 if SMOKE else 256
SEEDS=[0] if SMOKE else [0,1,2]
PROBE=['progenitor cell','plasmablast','plasmacytoid dendritic cell',
       'non-classical monocyte','conventional dendritic cell','lymphocyte']
if SMOKE: PROBE=PROBE[:4]
LUPUS=os.environ['H5AD']
CTRL_CAP = 2000 if SMOKE else 12000    # per-control-type subsample cap (5 controls); probes: keep every cell
t0=time.time()

# ---- data: sequential chunked scan (contiguous h5 reads only), raw counts from .raw, then HVG ----
_bk=ad.read_h5ad(LUPUS, backed='r')
_ct=_bk.obs.cell_type.astype(str).values
_don=_bk.obs.donor_id.astype(str).values
is_probe=np.isin(_ct,PROBE)
keep=is_probe.copy()
rng0=np.random.default_rng(12345)
for c in sorted(set(_ct)-set(PROBE)):
    ix=np.where(_ct==c)[0]
    sel=rng0.choice(ix,min(len(ix),CTRL_CAP),replace=False)
    keep[sel]=True
print(f"lupus keep-mask: {int(keep.sum()):,}/{_bk.n_obs:,} rows selected  {time.time()-t0:.0f}s",flush=True)

CH=100_000
pieces=[]; seen={c:0 for c in PROBE}; ntot=0
for i in range(0,_bk.n_obs,CH):
    j=min(i+CH,_bk.n_obs)
    m=keep[i:j]
    if not m.any(): continue
    chunk=_bk[i:j].to_memory()
    assert chunk.raw is not None, 'raw counts missing on this slice'
    piece=ad.AnnData(X=chunk.raw.X[m].copy(),
                      obs=chunk.obs[['donor_id','cell_type']].iloc[np.where(m)[0]].copy(),
                      var=chunk.raw.var.copy())
    pieces.append(piece); ntot+=piece.n_obs
    for c in PROBE: seen[c]+=int((piece.obs.cell_type.astype(str).values==c).sum())
    del chunk
    print(f"  scanned rows {j:,}/{_bk.n_obs:,}, kept {ntot:,} so far  {time.time()-t0:.0f}s",flush=True)
    if SMOKE and all(v>=80 for v in seen.values()) and ntot>=4000: break
del _bk
A=ad.concat(pieces, join='outer'); del pieces
A.obs['ct']=A.obs.cell_type.astype(str)
A.var_names_make_unique()
A=A[:, np.asarray(A.X.sum(0)).ravel()>0].copy()
sc.pp.highly_variable_genes(A, n_top_genes=HVG, flavor='seurat_v3', subset=True)
Xc=np.asarray(A.X.todense() if hasattr(A.X,'todense') else A.X, dtype=np.float32)
assert np.allclose(Xc, np.round(Xc)), 'X must be raw counts'
cts=sorted(A.obs.ct.unique()); ci={c:i for i,c in enumerate(cts)}; NC=len(cts); UNLAB=NC; NCAT=NC+1
yall=A.obs.ct.map(ci).values.astype(np.int64)
dall=A.obs.donor_id.astype(str).values; UD=np.unique(dall)
sf=Xc.sum(1,keepdims=True); Xn=np.log1p(Xc/sf*1e4).astype(np.float32)
print(f"{Xc.shape[0]:,} cells x {Xc.shape[1]} HVGs, {NC} types ({len(PROBE)} probe), sizes "
      f"{np.bincount(yall).min()}..{np.bincount(yall).max()}, {len(UD)} donors  {time.time()-t0:.0f}s",flush=True)
for c in PROBE:
    assert (yall==ci[c]).sum()>=(SIZES.max() if not SMOKE else 8)+10, f'{c}: too few cells survived the subsample for the top size'

from scvi.distributions import NegativeBinomial

def decode_latents(ae,Z,L):
    with torch.no_grad():
        mu=ae.decode({"rna":Z},{"rna":L})["rna"]
        return NegativeBinomial(mu=mu,theta=torch.exp(ae.theta)).sample().numpy().astype(np.float32)

def mk_batch(idx):
    return {"X":{"rna":torch.from_numpy(Xc[idx])}, "X_norm":{"rna":torch.from_numpy(Xn[idx])},
            "y":{"cell_type":torch.from_numpy(yall[idx])}}

def balanced_perm(g,ytr,n):
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
            if len(b)<2: continue
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

rows=[]; drows=[]; lrows=[]; srows=[]; crows=[]
def cost(seed,method,sec): crows.append(dict(seed=seed,method=method,sec=float(sec)))
SDIFF_BUDGET_S = 1800   # keep scDiffusion only if < 30 min/seed
sdiff_ok=True
for seed in SEEDS:
    rng=np.random.default_rng(seed)
    perm=rng.permutation(UD); cnt={c:0 for c in PROBE}; ev_d=[]
    for d in perm:
        if all(v>=NEV for v in cnt.values()): break
        ev_d.append(d); m=dall==d
        for c in PROBE: cnt[c]+=int((A.obs.ct.values[m]==c).sum())
    ev_mask=np.isin(dall,ev_d); tr_mask=~ev_mask
    assert not (set(dall[ev_mask]) & set(dall[tr_mask])), 'donor appears on both sides'
    ev_idx=[]; full_tr=[]; keep_=[]
    probe=np.array(sorted(ci[c] for c in PROBE))
    size_of=dict(zip(probe.tolist(), rng.permutation(SIZES)[:len(probe)].tolist()))
    for c in range(NC):
        ep=rng.permutation(np.where((yall==c)&ev_mask)[0]); ev_idx.append(ep[:NEV])
        rest=rng.permutation(np.where((yall==c)&tr_mask)[0])
        full_tr.append(rest)
        if c in size_of:
            r=size_of[c]
            assert len(rest)>=r, f'{cts[c]}: {len(rest)} train cells, size {r} does not fit'
            rest=rest[:r]
        else:
            rest=rest[:CAP]
        keep_.append(rest)
    ev_idx=np.concatenate(ev_idx); full_tr=np.concatenate(full_tr); tr_idx=np.concatenate(keep_)
    ytr=yall[tr_idx].copy()
    n_c=np.array([int((ytr==c).sum()) for c in range(NC)])
    assert all(n_c[c]==r for c,r in size_of.items()), 'physical drop did not land on the size exactly'
    print(f"  seed {seed}: {len(ev_d)}/{len(UD)} eval donors, {len(tr_idx):,} train cells "
          f"(deleted {len(full_tr)-len(tr_idx):,}), probe sizes {sorted(n_c[probe])}",flush=True)
    print(f"    donor overlap train/eval: {len(set(dall[tr_idx]) & set(dall[ev_idx]))} donors",flush=True)
    lsf=np.log(Xc[tr_idx].sum(1))
    sfs={'mn':float(lsf.min()),'mx':float(lsf.max()),
         'mu':torch.tensor([lsf[ytr==c].mean() if (ytr==c).any() else lsf.mean() for c in range(NCAT)],dtype=torch.float32),
         'sd':torch.tensor([max(lsf[ytr==c].std(),1e-2) if (ytr==c).sum()>1 else lsf.std() for c in range(NCAT)],dtype=torch.float32)}
    CENT=np.stack([Xn[full_tr[yall[full_tr]==c]].mean(0) for c in range(NC)])
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

    lab=np.arange(len(tr_idx))
    pca=PCA(50,random_state=seed).fit(Xn[tr_idx[rng.choice(len(tr_idx),min(30000,len(tr_idx)),replace=False)]])
    Ztr=pca.transform(Xn[tr_idx[lab]]); ytr_l=ytr[lab]
    Zev=pca.transform(Xn[ev_idx]); yev=yall[ev_idx]
    n_ev=np.array([int((yev==c).sum()) for c in range(NC)]); NEVc=np.minimum(NEV,n_ev)
    short=[(cts[c],int(n_ev[c])) for c in probe if n_ev[c]<NEV]
    if short: print(f"  seed {seed} NEV shortfall (per-class NEV used instead of {NEV}): {short}",flush=True)

    SAVE_GENE=os.environ.get('SAVE_GENE')
    if SAVE_GENE:
        os.makedirs(SAVE_GENE,exist_ok=True); REC={}; _pt=pca.transform
        def _rec(X,_pt=_pt,REC=REC):
            out=_pt(X)
            if len(X)<=TARGET: REC[id(out)]=np.asarray(X,np.float32).copy()
            return out
        pca.transform=_rec
        pm=np.isin(yall[full_tr],probe)
        np.savez_compressed(f'{SAVE_GENE}/s{seed}_base.npz',Xtr=Xn[tr_idx[lab]],ytr=ytr_l,Ztr=Ztr,Xev=Xn[ev_idx],yev=yev,
            Xfull=Xn[full_tr[pm]],yfull=yall[full_tr[pm]],probe=probe,n_c=n_c,cts=np.array(cts),
            comp=pca.components_,pmean=pca.mean_,genes=np.array(A.var_names).astype(str),NEVc=NEVc)

    os.makedirs('sd3_tmp',exist_ok=True)
    def run_sd3(drop, tag):
        sel=~np.isin(ytr,list(drop)) if drop else np.ones(len(ytr),bool)
        want=pd.DataFrame([dict(cell_type=cts[c],n=TARGET-int(n_c[c]))
                           for c in probe if TARGET-int(n_c[c])>0 and c not in drop])
        if not len(want): return None,'nothing to generate'
        wd=tempfile.mkdtemp(prefix=f'sd3_{seed}_{tag}_', dir='sd3_tmp')
        scipy.io.mmwrite(os.path.join(wd,'train_counts.mtx'),
                         scipy.sparse.csr_matrix(Xc[tr_idx[sel]].T))
        pd.DataFrame(dict(cell_type=[cts[i] for i in ytr[sel]])).to_csv(
            os.path.join(wd,'train_meta.csv'),index=False)
        want.to_csv(os.path.join(wd,'want.csv'),index=False)
        t=time.time()
        # SD3_TIMEOUT: observed a lupus-specific fit_copula hang (20+ min, <5% CPU, no AIDA precedent
        # -- AIDA's own nosingle method ran 8-45s) that a bare subprocess.run would wait on forever.
        try:
            r=subprocess.run([RS_BIN,'scdesign3.R',wd,str(SD3_CORES),str(seed)],capture_output=True,text=True,timeout=SD3_TIMEOUT or None)
        except subprocess.TimeoutExpired:
            cost(seed, 'sd3' if tag=='all' else 'sd3_nosingle', time.time()-t)
            print(f"  seed {seed} sd3[{tag}] TIMED OUT after {SD3_TIMEOUT}s (killed)",flush=True)
            return None,f'timeout after {SD3_TIMEOUT}s'
        cost(seed, 'sd3' if tag=='all' else 'sd3_nosingle', time.time()-t)
        if r.returncode!=0:
            err=r.stderr.strip().replace(chr(10),' | ')[-500:]
            print(f"  seed {seed} sd3[{tag}] FAILED after {time.time()-t:.0f}s: {err}",flush=True)
            return None,err
        S=scipy.io.mmread(os.path.join(wd,'sim_counts.mtx')).toarray().T.astype(np.float32)
        sm=pd.read_csv(os.path.join(wd,'sim_meta.csv')).cell_type.values
        assert len(sm)==len(S)==int(want.n.sum()), (len(sm),len(S),int(want.n.sum()))
        assert not any(np.allclose(S[0],Xc[i]) for i in tr_idx[:300]), 'sd3 returned real cells'
        Sn=np.log1p(S/np.maximum(S.sum(1,keepdims=True),1)*1e4).astype(np.float32)
        print(f"  seed {seed} sd3[{tag}] ok {time.time()-t:.0f}s, {len(S)} cells",flush=True)
        return {int(ci[c]):pca.transform(Sn[sm==c]) for c in want.cell_type}, ''

    single=[int(c) for c in range(NC) if n_c[c]<2]
    sd3,err_all=run_sd3(set(), 'all')
    sd3_drop,err_drop=(None,'not attempted')
    if sd3 is None and single:
        sd3_drop,err_drop=run_sd3(set(single), 'nosingle')
    srows.append(dict(seed=seed, n_singleton=len(single),
                      singleton_types='; '.join(cts[c] for c in single),
                      all_classes_ok=sd3 is not None, err_all=err_all,
                      nosingle_ok=sd3_drop is not None, err_nosingle=err_drop))

    gen={}
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
    cost(seed,'synth_w1',time.time()-_tc)
    print(f"  seed {seed} generated {time.time()-t0:.0f}s",flush=True)

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
    print(f"  seed {seed} latent methods done {time.time()-t0:.0f}s",flush=True)

    def _lib(c,k):
        ic=tr_idx[ytr==c]; ll=np.log(Xc[ic].sum(1))
        return np.exp(rng.normal(ll.mean(), max(ll.std(),1e-2) if len(ll)>1 else 0.1,
                                 k)).astype(np.float32)

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
            out=vae.module.generative(z=torch.randn(k,LAT),
                                      library=torch.from_numpy(np.log(L))[:,None],
                                      batch_index=torch.full((k,1),bi,dtype=torch.long))
            scv[int(c)]=to_pca(out['px'].sample().numpy().astype(np.float32))
    del vae
    cost(seed,'scvi',time.time()-_tc)
    print(f"  seed {seed} scVI done {time.time()-t0:.0f}s",flush=True)

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
            out=cvae.module.generative(z=torch.randn(k,LAT),
                                       library=torch.from_numpy(L)[:,None],
                                       y=torch.full((k,1),li,dtype=torch.long),
                                       batch_index=torch.zeros(k,1,dtype=torch.long))
            scc[int(c)]=to_pca(out['px'].sample().numpy().astype(np.float32))
    del cvae
    cost(seed,'scvi_cond',time.time()-_tc)
    print(f"  seed {seed} CondSCVI done {time.time()-t0:.0f}s",flush=True)

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

    _tc=time.time()
    sdf={}
    if sdiff_ok:
        SM=build_sdiff(Xn[tr_idx], ytr, NCAT, seed, *SDIFF_EP, bs=BS,
                       vae_steps=VAE_STEPS, vae_ckpt=f'vae_ckpt/lupus_s{seed}_{VAE_STEPS}.pt')
        for c in probe:
            k=TARGET-int(n_c[c])
            if k<=0: continue
            sdf[int(c)]=pca.transform(sample_sdiff(SM, int(c), k))
        del SM
    _sdiff_sec=time.time()-_tc
    cost(seed,'sdiff',_sdiff_sec)
    print(f"  seed {seed} scDiffusion done ({_sdiff_sec:.0f}s this seed) {time.time()-t0:.0f}s",flush=True)
    if _sdiff_sec>SDIFF_BUDGET_S:
        sdiff_ok=False
        print(f"  seed {seed} scDiffusion cost {_sdiff_sec:.0f}s > {SDIFF_BUDGET_S}s budget: DROPPING for remaining seeds",flush=True)

    def method(name):
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
                if int(c) not in sdf: continue
                Z.append(sdf[int(c)])
            else:
                Z.append(gen[(float(name.split('_w')[1]),int(c))])
            if SAVE_GENE: GB.append(Xn[tr_idx[lab]][sel] if name=='upsample' else REC[id(Z[-1])])
            y.append(np.full(k,c,dtype=np.int64))
        if SAVE_GENE and GB:
            np.savez_compressed(f'{SAVE_GENE}/s{seed}_{name}.npz',X=np.vstack(GB),y=np.concatenate(y[1:]))
        Z=np.vstack(Z); y=np.concatenate(y)
        clf=LogisticRegression(max_iter=2000,n_jobs=NJ,random_state=seed).fit(Z,y)
        P=clf.predict_proba(Zev); cls=list(clf.classes_); pred=clf.classes_[P.argmax(1)]
        for c in range(NC):
            true=(yev==c); j=cls.index(c) if c in cls else None
            r200=np.nan if j is None else float(true[np.argsort(-P[:,j])[:int(NEVc[c])]].mean())
            rows.append(dict(seed=seed,method=name,c=c,name_=cts[c],n=int(n_c[c]),
                             probe=bool(c in set(probe.tolist())),nev_used=int(NEVc[c]),
                             rec_argmax=float((pred[true]==c).mean()),rec200=r200,
                             called=int((pred==c).sum())))

    method_list=['real','upsample','nb','synth_w1','synthbal_w1','scvi','scvi_cond']
    if sdiff_ok or sdf: method_list.append('sdiff')
    method_list+= ['lat_boot','lat_gauss']
    method_list+= (['sd3'] if sd3 else []) + (['sd3_nosingle'] if sd3_drop else [])
    for a in method_list:
        method(a); print(f"  seed {seed} {a} {time.time()-t0:.0f}s",flush=True)
    cost(seed,'real',0.0); cost(seed,'upsample',0.0)
    cost(seed,'lat_boot',_ae_sec); cost(seed,'lat_gauss',_ae_sec)
    cost(seed,'ae_shared',_ae_sec)
    pd.DataFrame(crows).to_csv('lupus_cost.csv',index=False)

    pset=set(probe.tolist()); ab=np.array([c for c in range(NC) if c not in pset])
    CEV=np.stack([Zev[yev==c].mean(0) for c in range(NC)])
    c_glob=Zev[np.isin(yev,ab)].mean(0)
    Rpool=Ztr[rng.choice(len(Ztr),min(3000,len(Ztr)),replace=False)]
    for c in probe:
        k=TARGET-int(n_c[c])
        if k<=0: continue
        Rev=Zev[yev==c]; c_true=Rev.mean(0)
        sigma=float(np.linalg.norm(Rev-c_true,axis=1).mean()) if len(Rev)>0 else np.nan
        a_star=int(ab[np.linalg.norm(CEV[ab]-c_true,axis=1).argmin()]); c_near=CEV[a_star]
        pos=np.where(ytr_l==c)[0]
        sets={'train':Ztr[pos], 'upsample':Ztr[rng.choice(pos,k,replace=True)],
              'synth_w1':gen[(1.0,int(c))], 'synthbal_w1':genb[int(c)],
              'scvi':scv[int(c)], 'scvi_cond':scc[int(c)], 'nb':nb[int(c)],
              'lat_boot':lb[int(c)], 'lat_gauss':lg[int(c)]}
        if int(c) in sdf: sets['sdiff']=sdf[int(c)]
        if sd3: sets['sd3']=sd3[int(c)]
        if sd3_drop and int(c) in sd3_drop: sets['sd3_nosingle']=sd3_drop[int(c)]
        for nm,G in sets.items():
            delta=G.mean(0)-c_true
            drows.append(dict(seed=seed,c=int(c),name_=cts[c],n=int(n_c[c]),method=nm,sigma=sigma,near=cts[a_star],
                d_norm=float(np.linalg.norm(delta)/sigma) if sigma==sigma and sigma>0 else np.nan,
                pull_glob=float(delta@(c_glob-c_true)/np.linalg.norm(c_glob-c_true)**2),
                pull_near=float(delta@(c_near -c_true)/np.linalg.norm(c_near -c_true)**2)))
        if len(Rev)>=3:
            for nm in ('upsample','nb','sd3','sd3_nosingle','synth_w1','synthbal_w1',
                       'scvi','scvi_cond','sdiff','lat_boot','lat_gauss'):
                if nm not in sets: continue
                G=sets[nm]
                for scope,R in (('pooled',Rpool),('within',Rev)):
                    Z=np.vstack([R,G])
                    md=pd.DataFrame({'src':['real']*len(R)+['syn']*len(G)})
                    v=compute_lisi(Z,md,['src'],perplexity=min(30,(len(Z)-2)//3))[:,0]
                    lrows.append(dict(seed=seed,c=int(c),name_=cts[c],n=int(n_c[c]),method=nm,scope=scope,
                                      ilisi=float(v[len(R):].mean())))
    pd.DataFrame(drows).to_csv('lupus_disp.csv',index=False)
    pd.DataFrame(lrows).to_csv('lupus_ilisi.csv',index=False)
    print(f"  seed {seed} geometry + iLISI done {time.time()-t0:.0f}s",flush=True)
    pd.DataFrame(rows).to_csv('lupus.csv',index=False)
    pd.DataFrame(srows).to_csv('lupus_sd3status.csv',index=False)

S3=pd.DataFrame(srows); S3.to_csv('lupus_sd3status.csv',index=False)
print('\n=== can scDesign3 run at all on the lupus donor-disjoint split?')
for _,r in S3.iterrows():
    print(f"  seed {r.seed}: {r.n_singleton} singleton class(es) [{r.singleton_types}]")
    print(f"    all classes    : {'RAN' if r.all_classes_ok else 'ABORTED'}"
          + ('' if r.all_classes_ok else f"  {r.err_all[-220:]}"))
    print(f"    singletons drop : {'RAN' if r.nosingle_ok else r.err_nosingle[:60]}")

T=pd.DataFrame(rows); T.to_csv('lupus.csv',index=False)
METHODS=tuple(a for a in ('real','upsample','nb','sd3','sd3_nosingle','synth_w1','synthbal_w1','scvi','scvi_cond','sdiff','lat_boot','lat_gauss') if a in set(T.method))
P=T[T.probe]
print("\n=== recall@200 (per-class NEV, see nev_used) on held-out cells, lupus probe classes, mean over seeds")
print(f"{'n_c':>6} " + ' '.join(f'{a:>10}' for a in METHODS))
for r in sorted(P.n.unique()):
    d=P[P.n==r]
    print(f"{r:>6} " + ' '.join(f"{d[d.method==a].rec200.mean():>10.3f}" for a in METHODS))
from scipy import stats
print("\n=== does synthesis beat the dose-matched upsampling baseline?")
piv=P.pivot_table(index=['seed','c','n'],columns='method',values='rec200').reset_index()
for lo,hi in [(0,8),(8,64),(64,257)]:
    d=piv[(piv.n>lo)&(piv.n<=hi)].dropna()
    if len(d)<3: continue
    o=[]
    for a in METHODS[2:]:
        if a not in d.columns: continue
        dl=d[a]-d['upsample']; o.append(f"{a} vs upsample {dl.mean():+.3f} (t {stats.ttest_rel(d[a],d['upsample']).statistic:+.2f})")
    print(f"  n_c {lo+1}-{hi}: " + '  '.join(o))
print("\n=== control classes (never augmented) -- sanity, should not move")
C=T[~T.probe]
print('  ' + '  '.join(f"{a} {C[C.method==a].rec200.mean():.3f}" for a in METHODS))

C=pd.DataFrame(crows); C.to_csv('lupus_cost.csv',index=False)
COST=C.groupby('method').sec.median().to_dict()

def clust_t(d, a, b):
    v=d.assign(_dl=d[a]-d[b]).groupby('name_')._dl.mean().dropna()
    if len(v)<2: return np.nan, np.nan, np.nan, 0
    se=v.std(ddof=1)/np.sqrt(len(v))
    t=v.mean()/se if se>0 else np.nan
    return float(v.mean()), float(t), float(2*stats.t.sf(abs(t),len(v)-1)), len(v)

piv2=P.pivot_table(index=['seed','c','n','name_'],columns='method',values='rec200').reset_index()
NPROBE=len(PROBE)
BANDS=[(f'rare  n_c<=8',0,8),(f'abund n_c>=16',15,10**6),('ALL sizes',0,10**6)]
print('\n'+'='*100)
print(f'THE CLAIM ON LUPUS: upsampling against every generative model ({NPROBE} probe types)')
print('='*100)
for lab,lo,hi in BANDS:
    d=piv2[(piv2.n>lo)&(piv2.n<=hi)]
    if not len(d): continue
    print(f'\n### {lab}   (N={len(d)} seed x class pairs)')
    print(f"  {'method':<14}{'rec200':>8}{'delta':>9}{'t_CLUST':>9}{'p':>8}{'ups wins':>10}{'cost s':>9}")
    for a in METHODS:
        if a=='upsample':
            v=d['upsample'].dropna()
            print(f"  {a:<14}{v.mean():>8.3f}{'--':>9}{'':>9}{'':>8}{'':>10}{COST.get(a,float('nan')):>9.0f}")
            continue
        if a not in d.columns: continue
        dd=d[[a,'upsample','name_']].dropna()
        if not len(dd): continue
        m,t,p,nt=clust_t(dd,a,'upsample')
        win=float((dd['upsample']>=dd[a]).mean())
        print(f"  {a:<14}{dd[a].mean():>8.3f}{m:>+9.4f}{t:>+9.2f}{p:>8.3f}{win:>10.2f}{COST.get(a,float('nan')):>9.0f}")

print('\n### wall-clock seconds to produce one seed of synthetic cells (median over seeds)')
for a in METHODS:
    if a in COST: print(f'  {a:<14}{COST[a]:>9.0f}')
