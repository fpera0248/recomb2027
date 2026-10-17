"""Main driver for the one-versus-rest metrics. For one dataset (aida/hlca/lupus): every cached arm x 4 classifiers x
{multiclass, ovr(probe types only)} x {default, matched} threshold, full metric suite.
  python3 run_e1.py aida     -> aida_full.csv, next to this script
"""
import sys, os, time, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from common import load_dataset, n_seeds, arm_list, build_Zy, score_class

DATASET = sys.argv[1]
OUT = os.path.join(os.path.dirname(__file__), f'{DATASET}_full.csv')
D, H, readout, gene_arm = load_dataset(DATASET)
ARMS = arm_list(H)
SEEDS = list(range(n_seeds(D)))
print(f'{DATASET}: dir={D} seeds={SEEDS} arms={ARMS}', flush=True)


def do_seed_arm(seed, arm):
    t0 = time.time()
    Z, y, b, Zev, yev = build_Zy(D, H, gene_arm, seed, arm)
    if Z is None:
        return []
    cts, n_c, probe = b['cts'], b['n_c'], b['probe']
    NEVc = b['NEVc'] if 'NEVc' in b.files else np.full(len(cts), 200)
    pset = set(probe.tolist())
    rows = []
    for k in ('lr', 'knn', 'rf', 'mlp'):
        # --- multiclass: one fit covers every class ---
        P, pred = readout(k, Z, y, np.ones(len(y)), Zev, seed)
        if P.shape[1] < len(cts):
            Pf = np.full((len(Zev), len(cts)), -np.inf); Pf[:, np.unique(y)] = P; P = Pf
        for c in range(len(cts)):
            t = yev == c
            row = dict(dataset=DATASET, seed=seed, arm=arm, clf=k, mode='multiclass', c=int(c),
                       name_=str(cts[c]), n=int(n_c[c]), probe=bool(c in pset), nev_used=int(NEVc[c]))
            row.update(score_class(t, P[:, c], pred == c, int(NEVc[c])))
            rows.append(row)
        # --- one-vs-rest: a dedicated binary fit per probe (rare) type ---
        for c in probe.tolist():
            ybin = (y == c).astype(int)
            Pb, predb = readout(k, Z, ybin, np.ones(len(ybin)), Zev, seed)
            classes1 = np.unique(ybin)
            if 1 in classes1:
                col1 = int(np.where(classes1 == 1)[0][0]) if Pb.shape[1] == len(classes1) else 1
                s = Pb[:, col1] if Pb.shape[1] > col1 else np.zeros(len(Zev))
            else:
                s = np.zeros(len(Zev))
            t = yev == c
            row = dict(dataset=DATASET, seed=seed, arm=arm, clf=k, mode='ovr', c=int(c),
                       name_=str(cts[c]), n=int(n_c[c]), probe=True, nev_used=int(NEVc[c]))
            row.update(score_class(t, s, predb == 1, int(NEVc[c])))
            rows.append(row)
    print(f'  seed={seed} arm={arm} done in {time.time()-t0:.1f}s ({len(rows)} rows)', flush=True)
    return rows


if __name__ == '__main__':
    from joblib import Parallel, delayed
    jobs = [delayed(do_seed_arm)(s, a) for s in SEEDS for a in ARMS]
    out = Parallel(n_jobs=int(os.environ.get('NJ', 2)), verbose=0)(jobs)
    df = pd.DataFrame([r for o in out for r in o])
    df.to_csv(OUT, index=False)
    print(f'wrote {OUT} ({len(df)} rows)', flush=True)
