"""zero_support_task: gen13 said the zero-support task does not exist. gen13 tested the EASY version.

In gen13 the target types still kept 300 to 700 cells in the other two ancestries, so the
classifier sat at 0.92 and putting the real deleted cells back bought 0.0035. That rules out
zero support as a problem for a well supported cell type. It does not rule out the condition
the census actually shows, which is a type that is RARE and has zero cells in one group at
the same time.

So here the surviving cells are capped too. Every target type is cut to N_KEEP training cells
in total, and the only thing that differs between the two conditions is WHERE those cells come
from. Counts are identical, so nothing is confounded with n:

    blind_n        N_KEEP cells of T drawn from all three ancestries    B is represented
    zeroB_n        N_KEEP cells of T drawn from the other two only      B has zero support
    zeroB_up       + duplicate those up to TARGET                       all upsampling CAN do
    zeroB_shift    + the additive ancestry contrast, from the other     a linear repair,
                     30 cell types, applied to those copies            no model training
    zeroB_oracle   + REAL (T,B) cells up to TARGET                      what a PERFECT
                                                                        generator of (T,B) buys
    blind_up       blind_n duplicated up to TARGET                      upsampling's own best

The decisive number is zeroB_oracle minus zeroB_up. That is the entire headroom any generator
could ever capture on this task. If it is zero, no generator can win it and the fork closes.
"""
import os, sys, time, warnings, numpy as np, pandas as pd, anndata as ad, scanpy as sc
SUPP_DATA = os.environ.get("SUPP_DATA", os.path.expanduser("~/data"))
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from scipy import stats
warnings.filterwarnings('ignore')

SMOKE='--smoke' in sys.argv
HVG=400 if SMOKE else 2000
SEEDS=[0] if SMOKE else list(range(6))
EVAL_FRAC=0.30
N_KEEP=8 if SMOKE else 16      # every target type is cut to this many TRAIN cells, both conditions
TARGET=32 if SMOKE else 128    # every augmented arm is dosed back up to exactly this many
ANC=['singaporean chinese','singaporean indian','singaporean malay']
TARGETS=['mature B cell','CD16-negative, CD56-bright natural killer cell, human','naive B cell',
         'memory B cell','gamma-delta T cell','plasmacytoid dendritic cell',
         'mucosal invariant T cell','plasma cell']
if SMOKE: TARGETS=TARGETS[:3]
D=SUPP_DATA + '/pilots/aida/'
t0=time.time()

A=ad.concat([ad.read_h5ad(D+'AIDA_AbundantPool_120k.h5ad'), ad.read_h5ad(D+'AIDA_RarePool.h5ad')],
            join='inner', index_unique='-')
A=A[A.obs.institute.values=='Genome Institute of Singapore'].copy()   # one site: no site confound
A=A[:, np.asarray(A.X.sum(0)).ravel()>0].copy()
sc.pp.highly_variable_genes(A, n_top_genes=HVG, flavor='seurat_v3', subset=True)
Xc=np.asarray(A.X.todense() if hasattr(A.X,'todense') else A.X, dtype=np.float32)
Xn=np.log1p(Xc/Xc.sum(1,keepdims=True)*1e4).astype(np.float32)
cts=sorted(A.obs.cell_type.astype(str).unique()); ci={c:i for i,c in enumerate(cts)}; NC=len(cts)
y=A.obs.cell_type.astype(str).map(ci).values.astype(np.int64)
anc=A.obs.self_reported_ethnicity.astype(str).values
don=A.obs.donor_id.astype(str).values
assert set(anc)==set(ANC), sorted(set(anc))
print(f"{len(y):,} SG cells x {HVG} HVGs, {NC} types, {len(np.unique(don))} donors  {time.time()-t0:.0f}s",flush=True)

rows=[]
for seed in SEEDS:
    rng=np.random.default_rng(seed)
    # donor-disjoint, stratified by ancestry so every (type, ancestry) eval stratum keeps cells
    ev_d=[]
    for a in ANC:
        d=np.unique(don[anc==a]); ev_d += list(rng.permutation(d)[:max(1,int(round(EVAL_FRAC*len(d))))])
    ev=np.isin(don,ev_d); tr=~ev
    assert not (set(don[ev]) & set(don[tr]))
    tr_idx=np.where(tr)[0]; ev_idx=np.where(ev)[0]
    pca=PCA(50,random_state=seed).fit(Xn[tr_idx[rng.choice(len(tr_idx),min(30000,len(tr_idx)),replace=False)]])
    Ztr=pca.transform(Xn[tr_idx]); Zev=pca.transform(Xn[ev_idx])
    ytr=y[tr_idx]; yev=y[ev_idx]; atr=anc[tr_idx]; aev=anc[ev_idx]

    # target (type, ancestry) pairs: the ancestry rotates with the seed so no group is special
    pairs=[(ci[t], ANC[(i+seed)%3]) for i,t in enumerate(TARGETS)]
    # two keep-sets of IDENTICAL size per type. Only the ancestry composition differs.
    K_blind={}; K_zero={}; REAL_B={}
    for c,B in pairs:
        allc=np.where(ytr==c)[0]; notB=np.where((ytr==c)&(atr!=B))[0]
        assert len(notB)>=N_KEEP, f'{cts[c]} {B}: only {len(notB)} non-B train cells'
        K_blind[c]=rng.choice(allc,N_KEEP,replace=False)
        K_zero[c]=rng.choice(notB,N_KEEP,replace=False)
        REAL_B[c]=np.where((ytr==c)&(atr==B))[0]
    tgt=np.isin(ytr,[c for c,_ in pairs])
    print(f"  seed {seed}: {len(ev_d)} eval donors, {len(tr_idx):,} train cells, "
          f"{len(pairs)} types cut to {N_KEEP} cells each",flush=True)

    # additive ancestry contrast in PCA space, estimated ONLY from non-target cell types and
    # centred per cell type so composition differences cannot leak in
    tgt_c={c for c,_ in pairs}
    def delta(B):
        ds=[]
        for c in range(NC):
            if c in tgt_c: continue
            i=(ytr==c)&(atr==B); j=(ytr==c)&(atr!=B)
            if i.sum()>=10 and j.sum()>=10: ds.append(Ztr[i].mean(0)-Ztr[j].mean(0))
        return np.mean(ds,0)
    DEL={B:delta(B) for B in ANC}

    def build(name):
        """Non-target cells untouched. Target types get N_KEEP real cells, then dose to TARGET."""
        Z=[Ztr[~tgt]]; yy=[ytr[~tgt]]
        for c,B in pairs:
            base=K_blind[c] if name.startswith('blind') else K_zero[c]
            Z.append(Ztr[base]); yy.append(np.full(len(base),c,dtype=np.int64))
            k=TARGET-len(base)
            if k<=0 or name in ('blind_n','zeroB_n'): continue
            if name.startswith('zeroB_oracle'):
                # oracle       dose up with REAL ancestry-B cells: the stratum we deleted
                # oracle_notB  dose up with MORE REAL cells from the same two other ancestries.
                # If these two are equal, the gain is sample size and has nothing to do with
                # ancestry, and no generator can capture it either.
                src=REAL_B[c] if name=='zeroB_oracle' else np.setdiff1d(
                        np.where((ytr==c)&(atr!=B))[0], base)
                if len(src)==0: continue
                Z.append(Ztr[rng.choice(src,k,replace=True)])
            else:
                S=Ztr[rng.choice(base,k,replace=True)]
                Z.append(S+DEL[B] if name=='zeroB_shift' else S)
            yy.append(np.full(k,c,dtype=np.int64))
        return np.vstack(Z), np.concatenate(yy)

    for name in ('blind_n','zeroB_n','zeroB_up','zeroB_shift','zeroB_oracle',
                 'zeroB_oracle_notB','blind_up'):
        Z,yy=build(name)
        clf=LogisticRegression(max_iter=2000,n_jobs=8,random_state=seed).fit(Z,yy)
        P=clf.predict_proba(Zev); cls=list(clf.classes_)
        for c,B in pairs:
            j=cls.index(c); n_tot=int((yev==c).sum())
            top=np.zeros(len(yev),bool); top[np.argsort(-P[:,j])[:n_tot]]=True   # matched operating point
            mB=(yev==c)&(aev==B); mO=(yev==c)&(aev!=B)
            rows.append(dict(seed=seed,arm=name,c=c,type_=cts[c],B=B,n_ev_B=int(mB.sum()),
                             n_tr_B=len(REAL_B[c]),
                             rec_B=float(top[mB].mean()), rec_other=float(top[mO].mean()),
                             rec_all=float(top[yev==c].mean())))
        print(f"    {name:14s} {time.time()-t0:.0f}s",flush=True)
    pd.DataFrame(rows).to_csv('gen14.csv',index=False)

T=pd.DataFrame(rows); T.to_csv('gen14.csv',index=False)
print("\n"+"="*96)
print("RARE + ZERO SUPPORT: at a FIXED 16 training cells per type, does it matter whether")
print("any of them come from ancestry B? And can anything buy back what it costs?")
print("="*96)
def con(a,b,met):
    q=T.pivot_table(index=['seed','type_'],columns='arm',values=met)
    d=(q[a]-q[b]).dropna()
    g=d.groupby(level='type_').mean()
    tc,pc=stats.ttest_1samp(g,0)
    return d.mean(), stats.ttest_1samp(d,0)[0], tc, pc, len(g)
print(f"\n{'contrast':34s} {'mean':>8s} {'t_naive':>8s} {'t_CLUST':>8s} {'p':>7s}  n_types")
for a,b,why in [('zeroB_n','blind_n','zero support, at identical n'),
                ('zeroB_up','zeroB_n','what duplication buys back'),
                ('zeroB_shift','zeroB_n','what the linear shift buys back'),
                ('zeroB_oracle','zeroB_n','REAL ancestry-B cells: the ceiling'),
                ('zeroB_oracle','zeroB_up','headroom over duplication: real cells vs copies'),
                ('zeroB_oracle_notB','zeroB_up','SAME, but the real cells are NOT from B'),
                ('zeroB_oracle','zeroB_oracle_notB','THE TEST: is the gain ancestry, or just n?'),
                ('zeroB_shift','zeroB_up','the shift against duplication'),
                ('blind_up','blind_n','duplication when B IS represented')]:
    m,tn,tc,pc,n=con(a,b,'rec_B')
    print(f"{a:14s}-{b:14s} {m:+8.4f} {tn:+8.2f} {tc:+8.2f} {pc:7.3f}  {n}   {why}")
print("\nabsolute recall of the held-out (type, ancestry B) eval cells, matched operating point")
print(T.pivot_table(index='arm',values=['rec_B','rec_other','rec_all']).round(4).to_string())
print("\nper target type, rec_B")
print(T.pivot_table(index='type_',columns='arm',values='rec_B')
       [['blind_n','zeroB_n','zeroB_up','zeroB_shift','zeroB_oracle','zeroB_oracle_notB','blind_up']].round(3).to_string())
print("\ntrain cells deleted per (type, ancestry) pair")
print(T[T.arm=='zeroB_n'].pivot_table(index='type_',values=['n_tr_B','n_ev_B']).round(0).to_string())
print("\nEXIT=0")
