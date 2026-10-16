"""m27 Step 3: unify the latent representation across every arm and test whether the
duplication-vs-generator verdict depends on which encoder produced the latent code.

Three encoders, each fit ONLY on real training cells (identical encoder for every arm):
  pca   existing PCA-50 (m27_base's own comp/pmean, fit on log1p-CPM)
  scvi  scVI (n_latent=30, default arch), fit on real RAW counts; posterior mean latent
  mlp   plain autoencoder (2000-256-50) on log1p-CPM, MSE-trained

A duplicated cell's latent is EXACTLY the encoder output of the real cell it copies: encoders
are deterministic functions of input, and upsample's added rows are copies of real rows, so no
special-casing is needed -- this script just runs every arm's saved cells through each encoder.

Same processing for every arm (Jiaqi 2026-09-28): every arm's saved cells go through the shared
normalizer (stat16.renorm: expm1 -> rows to 1e4 -> log1p; a no-op to ~1e-6 for arms already made from
counts, a real fix for sdiff, whose decoder output was never renormalized). jitter_s1.0 / smote_k5 are
built ONCE per seed in gene space on the real training cells (stat16.gene_arm), so all three encoders
see identical cells. Arms without raw counts (sdiff, jitter, smote) reach scVI as PSEUDO-counts:
expm1(X)/1e4 * L, L drawn from that TYPE's real training library sizes (Ltr). That approximation is
flagged in the output.

Evaluates every arm in every latent with stat16.readout for lr/knn/rf/mlp, rec200 on probe types,
paired vs duplication (polh.signflip_p by type), then reports sign flips and Kendall-tau rank
agreement of arms across latents.
  python3 m27_latent.py > m27_latent_report.txt   (writes m27_latent.csv)
"""
import glob, os, sys, warnings, numpy as np, pandas as pd, torch, anndata as ad
warnings.filterwarnings('ignore')
import logging; logging.getLogger('lightning.pytorch').setLevel(logging.ERROR)
import scvi
from scipy.stats import kendalltau
from stat16 import readout, NEV, gene_arm, renorm
import stat16
import polh
torch.set_num_threads(int(os.environ.get('M27_THREADS', 8)))
# M27_PSEUDO_ALL=1: for scVI, EVERY arm's added cells (upsample included) go through the same pseudo-count path,
# so no arm gets real counts while another gets resampled library sizes (verifier 2026-09-28: the path alone moves rf +0.03-0.05)
PSEUDO_ALL = os.environ.get('M27_PSEUDO_ALL') == '1'
# M27_LIB=anchor: cells built from a real cell (jitter/jitternz/smote) enter scVI with THAT cell's library,
# so upsample (real counts) and its noised variants share one count path; sdiff has no source cell (random draw)
ANCHOR_LIB = os.environ.get('M27_LIB') == 'anchor'
TAG = os.environ.get('M27_TAG', '')
ONLY = set(os.environ['M27_ARMS'].split(',')) if os.environ.get('M27_ARMS') else None   # partial rerun, e.g. sdiff,upsample
D = os.environ.get('M27_DIR', 'm27_base')   # m27_hlc = HLCA with counts rebuilt by hl_counts.py
SCVI_LAT, SCVI_EPOCHS = 30, int(os.environ.get('M27_SCVI_EPOCHS', 60))
AE_EPOCHS = int(os.environ.get('M27_AE_EPOCHS', 30))

def fit_mlp_ae(Xtr, seed, hidden=256, lat=50, epochs=AE_EPOCHS, bs=256):
    torch.manual_seed(seed)
    enc = torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1], hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, lat))
    dec = torch.nn.Sequential(torch.nn.Linear(lat, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, Xtr.shape[1]))
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-3)
    X = torch.from_numpy(Xtr.astype(np.float32)); rng = np.random.default_rng(seed)
    for _ in range(epochs):
        p = rng.permutation(len(X))
        for i in range(0, len(p), bs):
            b = X[p[i:i + bs]]
            if len(b) < 2: continue
            opt.zero_grad(); torch.nn.functional.mse_loss(dec(enc(b)), b).backward(); opt.step()
    enc.eval()
    return lambda Xn: enc(torch.from_numpy(np.asarray(Xn, np.float32))).detach().numpy()

def fit_scvi(Ctr, seed):
    scvi.settings.seed = seed
    A = ad.AnnData(np.asarray(Ctr, np.float32).copy()); scvi.model.SCVI.setup_anndata(A)
    vae = scvi.model.SCVI(A, n_latent=SCVI_LAT, n_hidden=256, n_layers=2)
    vae.train(max_epochs=SCVI_EPOCHS, batch_size=256, accelerator='cpu',
               enable_model_summary=False, enable_progress_bar=False)
    vae.module.eval()
    def encode(Cn):
        # No setup_anndata() here on purpose: a fresh independent registration would give Aq its own
        # manager, unconnected to vae's. get_latent_representation's _validate_anndata auto-transfers
        # the trained model's field registry onto an unregistered adata of matching shape (the
        # standard scvi-tools "apply model to new data" path); we never trained on any categorical
        # covariate, so there is nothing that transfer can get wrong.
        Aq = ad.AnnData(np.asarray(Cn, np.float32).copy())
        return vae.get_latent_representation(Aq)
    return encode

rows = []
for bf in sorted(glob.glob(f'{D}/s*_base.npz')):
    seed = int(os.path.basename(bf)[1:].split('_')[0]); b = np.load(bf)
    Xtr, ytr, Xev, yev = b['Xtr'], b['ytr'], b['Xev'], b['yev']
    Ctr, Cev, Ltr = b['Ctr'], b['Cev'], b['Ltr']
    comp, pmean, cts, n_c, probe = b['comp'], b['pmean'], b['cts'], b['n_c'], b['probe']
    NC = len(cts); pset = set(probe.tolist())
    NEVc = b['NEVc'] if 'NEVc' in b.files else np.full(NC, NEV)   # HLCA scores top-NEVc per class, as m27_hlclf does
    arms = sorted({os.path.basename(f)[len(f's{seed}_'):-4] for f in glob.glob(f'{D}/s{seed}_*.npz')} - {'base'})
    if ONLY: arms = [x for x in arms if x in ONLY]
    print(f'seed {seed}: fitting mlp + scvi encoders on {len(Xtr)} real training cells, arms={arms}', flush=True)

    pca_enc = lambda X: (np.asarray(X, np.float32) - pmean) @ comp.T
    mlp_enc = fit_mlp_ae(Xtr, seed)
    scvi_enc = fit_scvi(Ctr, seed)
    rng = np.random.default_rng(seed + 999)
    lib_by_c = {c: Ltr[ytr == c] for c in range(NC) if (ytr == c).any()}

    ENC = {'pca': (pca_enc, 'X', pca_enc(Xtr), pca_enc(Xev)),
           'mlp': (mlp_enc, 'X', mlp_enc(Xtr), mlp_enc(Xev)),
           'scvi': (scvi_enc, 'C', scvi_enc(Ctr), scvi_enc(Cev))}

    # every arm's added cells, built once per seed so all three encoders see the identical cells
    up = np.load(f'{D}/s{seed}_upsample.npz'); T = int(n_c[probe[0]] + (up['y'] == probe[0]).sum())
    CELLS = {'real': (Xtr[:0], ytr[:0], None)}
    for a in arms:
        d = np.load(f'{D}/s{seed}_{a}.npz'); CELLS[a] = (renorm(d['X']), d['y'], d['C'] if 'C' in d.files else None)
    for a in [x for x in ('jitter_s1.0', 'jitternz_s1.0', 'smote_k5') if not ONLY or x in ONLY]:
        stat16.ANCHORS = []; A = gene_arm(a, Xtr, ytr, probe, n_c, seed, T); anc = np.concatenate(stat16.ANCHORS); stat16.ANCHORS = None
        Xa = np.vstack([v[0] for v in A.values()]); assert len(anc) == len(Xa), a
        CELLS[a] = (Xa, np.concatenate([v[1] for v in A.values()]),
                    np.round(np.expm1(Xa.astype(np.float64)) / 1e4 * Ltr[anc][:, None]).astype(np.float32) if ANCHOR_LIB else None)
    for a, (Xa, ya, Ca) in CELLS.items():
        if PSEUDO_ALL and len(ya): rng = np.random.default_rng([seed, 999, sum(map(ord, a))])
        if (Ca is None or PSEUDO_ALL) and len(ya):   # pseudo-counts from log1p-CPM + real per-type library sizes
            Ca = np.zeros_like(Xa, dtype=np.float32)
            for c in np.unique(ya):
                m = ya == c; pool = lib_by_c.get(int(c))
                if pool is None or len(pool) == 0: pool = Ltr
                L = rng.choice(pool, m.sum(), replace=True)
                Ca[m] = np.round(np.clip(np.expm1(Xa[m]) / 1e4 * L[:, None], 0, None))
            CELLS[a] = (Xa, ya, Ca)
            print(f'  seed {seed} {a}: no raw counts, scVI gets pseudo-counts', flush=True)

    for lat_name, (enc, src, Zb, Zevb) in ENC.items():
        for a, (Xa, ya, Ca) in CELLS.items():
            Za = np.zeros((0, Zb.shape[1]), Zb.dtype) if len(ya) == 0 else enc(Ca if src == 'C' else Xa)
            Z = np.vstack([Zb, Za]); y = np.concatenate([ytr, ya])
            for k in ('lr', 'knn', 'rf', 'mlp'):
                P, pred = readout(k, Z, y, np.ones(len(y)), Zevb, seed)
                cls = sorted(np.unique(y).tolist())
                for c in range(NC):
                    if c not in cls: continue
                    j = cls.index(c); t = yev == c
                    rows.append(dict(seed=seed, latent=lat_name, arm=a, clf=k, c=c, name_=str(cts[c]),
                        n=int(n_c[c]), probe=bool(c in pset),
                        rec200=float(t[np.argsort(-P[:, j])[:int(NEVc[c])]].mean()),
                        rec_argmax=float((pred[t] == c).mean())))
            print(f'  seed {seed} {lat_name} {a} done', flush=True)

R = pd.DataFrame(rows); R.to_csv(f'm27_latent{TAG}.csv', index=False)
if not len(R):
    print(f'no {D}/s*_base.npz found yet -- run m27_gen_base.py first'); raise SystemExit

P = R[R.probe]
print(f"\n{P.seed.nunique()} seed(s) x {P.latent.nunique()} latents x {P.clf.nunique()} classifiers x {P.name_.nunique()} probe types\n")

print('=== rec200 vs duplication, per latent x classifier (paired by type, polh.signflip_p)')
sign_table = {}
for (lat, clf), g in P.groupby(['latent', 'clf']):
    w = g.pivot_table(index=['seed', 'name_', 'n'], columns='arm', values='rec200')
    if 'upsample' not in w.columns: continue
    print(f'\n  {lat} / {clf}')
    for a in sorted(w.columns, key=lambda a: -w[a].mean()):
        dser = (w[a] - w['upsample']).dropna()          # dropna first, THEN pull matching group labels --
        dd = dser.values                                  # slicing grp[:len(dd)] would misalign whichever
        gg = dser.index.get_level_values('name_').values   # rows dropna happened to drop.
        p, dt = polh.signflip_p(dd, gg) if a != 'upsample' else (1.0, np.array([0.0]))
        sign_table[(lat, clf, a)] = float(dd.mean())
        print(f'    {a:14s} rec200 {w[a].mean():.3f}  d_vs_dup {dd.mean():+.4f}  p {p:.3f}')

print('\n=== does any arm flip SIGN (vs duplication) across the three latents? (lr classifier)')
arms_ = sorted({a for (_, clf, a) in sign_table if clf == 'lr' and a != 'upsample'})
for a in arms_:
    ds = {lat: sign_table.get((lat, 'lr', a)) for lat in ('pca', 'mlp', 'scvi')}
    signs = {k: (np.sign(v) if v is not None else None) for k, v in ds.items()}
    flips = len({s for s in signs.values() if s is not None}) > 1
    print(f"  {a:14s} " + '  '.join(f'{k}={v:+.4f}' if v is not None else f'{k}=NA' for k, v in ds.items())
          + ('   <-- SIGN FLIPS ACROSS LATENTS' if flips else ''))

print('\n=== Kendall tau rank agreement of arms across latent pairs (mean rec200 per arm, lr classifier)')
means = {}
for lat in ('pca', 'mlp', 'scvi'):
    w = P[(P.latent == lat) & (P.clf == 'lr')].groupby('arm').rec200.mean()
    means[lat] = w
for i, l1 in enumerate(('pca', 'mlp', 'scvi')):
    for l2 in ('pca', 'mlp', 'scvi')[i + 1:]:
        common = means[l1].index.intersection(means[l2].index)
        if len(common) < 3: continue
        tau, p = kendalltau(means[l1][common], means[l2][common])
        print(f'  {l1} vs {l2}: tau={tau:+.3f} p={p:.3f} (n={len(common)} arms)')
