"""Shared loading + metrics for AIDA/HLCA/lupus, same readout path as hlca_classifier_readout.py.
Imports two modules unmodified from the locked tree HR_WORK/cfgen_natrare, under their original names: m27_hlclf (the file released here as
hlca_classifier_readout.py) and stat16 (released here as statistical_augmentations.py). It sets M27_HL before the import so the ARMS/_fn/_base
of m27_hlclf pick up the right cache dir. Does NOT edit either file.
"""
import os, sys, glob, numpy as np
HR_WORK = os.environ.get("HR_WORK", os.path.expanduser("~/work"))
sys.path.insert(0, HR_WORK + '/cfgen_natrare')
os.chdir(HR_WORK + '/cfgen_natrare')   # hlca_classifier_readout._base/_fn use paths relative to this dir
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')

DIRS = {'aida': 'm27_base', 'hlca': 'm27_hl', 'lupus': 'm27_mk'}

from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef, cohen_kappa_score


def load_dataset(dataset):
    """Sets M27_HL then imports hlca_classifier_readout fresh (per-process import, so the module-level HL/ARMS/_fn
    bind to this dataset's cache dir). Returns the module plus gene_arm/readout from stat16 (statistical_augmentations.py)."""
    D = DIRS[dataset]
    os.environ['M27_HL'] = D
    os.environ['M27_CSV'] = f'/tmp/_unused_{dataset}.csv'
    import importlib
    if 'm27_hlclf' in sys.modules:
        del sys.modules['m27_hlclf']   # force re-read of M27_HL for this process
    import m27_hlclf as H
    from stat16 import readout, gene_arm
    return D, H, readout, gene_arm


def n_seeds(D):
    return len(glob.glob(f'{D}/s*_base.npz'))


NOISE_ARMS = ('jitter_s1.0', 'jitternz_s1.0', 'smote_k5')   # built on the fly (gene_arm), never saved as npz,
# so they never show up in H.ARMS' glob of s*_*.npz -- hlca_classifier_readout.py's own run block adds them as a
# separate hardcoded job list; same fix needed here or they'd silently be missing.

def arm_list(H):
    return sorted(set(H.ARMS) | {'real'} | set(NOISE_ARMS))


def build_Zy(D, H, gene_arm, seed, arm):
    """Reproduces hlca_classifier_readout.fit / fit_noise's Z,y construction exactly (copied, not imported, since
    those functions also score). arm='real' = no augmentation (Ztr,ytr alone)."""
    b, comp, pmean, Ztr, ytr, Zev, yev = H._base(seed)
    if arm == 'real':
        return Ztr, ytr, b, Zev, yev
    if arm in ('jitter_s1.0', 'jitternz_s1.0', 'smote_k5'):
        n_c, probe = b['n_c'], b['probe']
        up = np.load(f'{D}/s{seed}_upsample.npz')
        T = int(n_c[probe[0]] + (up['y'] == probe[0]).sum())
        A = gene_arm(arm, b['Xtr'], ytr, probe, n_c, seed, T)
        Z = np.vstack([Ztr] + [(v[0] - pmean) @ comp.T for v in A.values()])
        y = np.concatenate([ytr] + [v[1] for v in A.values()])
    else:
        f = H._fn(seed, arm)
        if f is None:
            return None, None, b, Zev, yev
        a = np.load(f)
        Z = np.vstack([Ztr, (a['X'] - pmean) @ comp.T])
        y = np.concatenate([ytr, a['y']])
    return Z, y, b, Zev, yev


def top_k_mask(s, k, n):
    """Exactly np.argsort(-s)[:k] as a boolean mask -- same tie-break as hlca_classifier_readout's rec200."""
    k = min(k, n)
    idx = np.argsort(-s)[:k]
    m = np.zeros(n, bool)
    m[idx] = True
    return m


def conf_metrics(t, called):
    t = np.asarray(t, bool); called = np.asarray(called, bool)
    TP = int((t & called).sum()); FP = int((~t & called).sum())
    FN = int((t & ~called).sum()); TN = int((~t & ~called).sum())
    rec = TP / (TP + FN) if (TP + FN) else np.nan
    fpr = FP / (FP + TN) if (FP + TN) else np.nan
    prec = TP / (TP + FP) if (TP + FP) else np.nan
    f1 = 2 * prec * rec / (prec + rec) if (prec == prec and rec == rec and (prec + rec) > 0) else np.nan
    spec = TN / (TN + FP) if (TN + FP) else np.nan
    bal_acc = np.nanmean([rec, spec]) if not (rec != rec and spec != spec) else np.nan
    gmean = np.sqrt(rec * spec) if (rec == rec and spec == spec) else np.nan
    mcc = matthews_corrcoef(t, called) if len(np.unique(t)) > 1 and len(np.unique(called)) > 1 else np.nan
    kappa = cohen_kappa_score(t, called) if len(np.unique(t)) > 1 else np.nan
    return dict(recall=rec, tpr=rec, fpr=fpr, precision=prec, f1=f1, specificity=spec,
                balanced_accuracy=bal_acc, mcc=mcc, gmean=gmean, kappa=kappa,
                TP=TP, FP=FP, FN=FN, TN=TN)


def score_class(t, s, default_called, nevc):
    n = len(t)
    matched_called = top_k_mask(s, nevc, n)
    row = {}
    if len(np.unique(t)) > 1:
        row['auroc'] = roc_auc_score(t, s); row['ap'] = average_precision_score(t, s)
    else:
        row['auroc'] = np.nan; row['ap'] = np.nan
    for pfx, called in (('default', default_called), ('matched', matched_called)):
        for kk, vv in conf_metrics(t, called).items():
            row[f'{pfx}_{kk}'] = vv
    return row
