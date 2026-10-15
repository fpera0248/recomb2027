"""Compare a rebuilt-from-scratch run with the numbers of the paper.

    python -m eval.full_check RESULT.json [--no-sccad]     

RESULT.json is written by eval.analyze (eval/run_eval.sh). PAPER holds the atlas-equal means and difference of the 10-atlas test
(ours minus Leiden subclustering = T1, ours minus scCAD with labels = T2). Tolerance: pool union AP moves by about 0.01 between numerical
stacks (the Leiden partitions change), so means and differences must be within 0.03, the sign must agree and the atlas-level exact test must
stay below 0.05. A rebuild cannot match the paper to the digit. Exit status 1 when a check fails. T2 needs the scCAD run (--no-sccad skips it).
"""
import json
import sys

TOL = 0.03
PAPER = {"union_AP": {"T1": (0.570, 0.446, 0.124), "T2": (0.570, 0.427, 0.143)}, "E1_union": {"T1": (None, None, 0.087), "T2": (None, None, 0.111)}}


def check(res, tests=("T1", "T2")):
    bad = 0
    for met, per in PAPER.items():
        for t in tests:
            ours, other, diff = per[t]
            r = res[met][t]
            ok = abs(r["diff"] - diff) <= TOL and r["diff"] > 0 and r["p"] < 0.05
            for got, want in ((r["a_mean"], ours), (r["b_mean"], other)):
                ok &= want is None or abs(got - want) <= TOL
            bad += not ok
            print(f"{met:9s} {t}: ours {r['a_mean']:.3f} other {r['b_mean']:.3f} diff {r['diff']:+.3f} (paper {diff:+.3f}) ahead {r['atlas_ahead']}/{r['n_atlases']} exact p {r['p']:.4f} {'OK' if ok else 'OUT OF TOLERANCE'}")
    return bad


def selftest():
    fake = {m: {t: dict(a_mean=v[0] or 0.5, b_mean=v[1] or 0.4, diff=v[2], p=0.002, atlas_ahead=10, n_atlases=10) for t, v in per.items()} for m, per in PAPER.items()}
    assert check(fake) == 0
    fake["union_AP"]["T1"]["diff"] = 0.2
    assert check(fake) == 1


if __name__ == "__main__":
    if sys.argv[1] == "--selftest":
        selftest()
    else:
        sys.exit(1 if check(json.load(open(sys.argv[1])), ("T1",) if "--no-sccad" in sys.argv else ("T1", "T2")) else 0)
