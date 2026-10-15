"""31-type probe run: recall at the matched operating point (rec200) of each augmentation against duplication ('upsample').

Unit = cell type (mean over its 8 appearances); two-sided sign-flip over the 31 types (seeded Monte-Carlo above 16 types).
BH over the 20 primary tests (5 augmentations x 4 classifiers). 'real' is no augmentation, so its d is (no augmentation) minus duplication.
"""
import pandas as pd
from statsmodels.stats.multitest import multipletests

from .stats import signflip_p

PRIMARY = ["jitter_s0.5", "jitter_s1.0", "jitternz_s1.0", "smote_k5", "real"]
CLF = ["lr", "knn", "rf", "mlp"]


def table(csv):
    S = pd.read_csv(csv)
    rows = []
    for k in CLF:
        P = S[(S.clf == k) & S.probe]
        for part, m in (("all", P.n > 0), ("n<=8", P.n <= 8), ("n>=16", P.n >= 16)):
            w = P[m].pivot_table(index=["seed", "name_"], columns="arm", values="rec200")
            for a in w.columns:
                if a == "upsample":
                    continue
                d = (w[a] - w["upsample"]).values
                p, dt = signflip_p(d, w.index.get_level_values("name_").values)
                rows.append(dict(clf=k, part=part, arm=a, arm_rec=w[a].mean(), up_rec=w["upsample"].mean(), d=d.mean(), types_pos=int((dt > 0).sum()), types=len(dt), p=p))
    R = pd.DataFrame(rows)
    prim = (R.part == "all") & R.arm.isin(PRIMARY)
    assert prim.sum() == 20, prim.sum()
    R.loc[prim, "q_primary"] = multipletests(R.loc[prim, "p"], method="fdr_bh")[1]
    return R
