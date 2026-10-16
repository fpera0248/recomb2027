"""Compare a rebuilt-from-scratch run with the numbers of the paper.

    python -m eval.full_check RESULT.json [--no-sccad] [--set8]

RESULT.json is written by eval.analyze (eval/run_eval.sh). PAPER holds the atlas-equal means and difference of the 10-atlas test
(ours minus Leiden subclustering = T1, ours minus scCAD with labels = T2). Tolerance: Leiden partitions change between numerical
stacks by an amount not measured, so means and differences must be within 0.03 (a chosen bound), the sign must agree and the atlas-level exact test must
stay below 0.05. The E1_union differences (0.087, 0.111) are from the 10-atlas analysis, not printed in the paper. The run must have 57 pools in 10 atlases.
Exit status 1 when a check fails. T2 needs the scCAD run (--no-sccad skips it).

--set8 checks the 8-atlas test set instead (PAPER8: 69 pools in 8 atlases, union_AP only; E1_union is not checked because its paper p
values, 0.086 against Leiden and 0.18 against scCAD with labels, are above 0.05). Same tolerance. The exact p must be below 0.05 for T1
only: the paper gives +0.090 with p = 0.14 for ours minus scCAD with labels. A rebuild on identical pools gave ours 0.312 and Leiden 0.207.
"""

import json
import sys

TOL = 0.03
PAPER = {
    "union_AP": {"T1": (0.570, 0.446, 0.124), "T2": (0.570, 0.427, 0.143)},
    "E1_union": {"T1": (None, None, 0.087), "T2": (None, None, 0.111)},
}
PAPER8 = {"union_AP": {"T1": (0.321, 0.220, 0.101), "T2": (0.321, 0.231, 0.090)}}


def check(res, tests=("T1", "T2"), paper=PAPER, n=(57, 10), p_tests=("T1", "T2")):
    bad = 0
    for met, per in paper.items():
        for t in tests:
            ours, other, diff = per[t]
            r = res[met][t]
            assert (r["n_pools"], r["n_atlases"]) == n, (
                met,
                t,
                r["n_pools"],
                r["n_atlases"],
            )
            ok = abs(r["diff"] - diff) <= TOL and r["diff"] > 0 and (t not in p_tests or r["p"] < 0.05)
            for got, want in ((r["a_mean"], ours), (r["b_mean"], other)):
                ok &= want is None or abs(got - want) <= TOL
            bad += not ok
            print(
                f"{met:9s} {t}: ours {r['a_mean']:.3f} other {r['b_mean']:.3f} diff {r['diff']:+.3f} (paper {diff:+.3f}) ahead {r['atlas_ahead']}/{r['n_atlases']} exact p {r['p']:.4f} {'OK' if ok else 'OUT OF TOLERANCE'}"
            )
    return bad


def selftest():
    fake = {
        m: {
            t: dict(
                a_mean=v[0] or 0.5,
                b_mean=v[1] or 0.4,
                diff=v[2],
                p=0.002,
                atlas_ahead=10,
                n_atlases=10,
                n_pools=57,
            )
            for t, v in per.items()
        }
        for m, per in PAPER.items()
    }
    assert check(fake) == 0
    fake["union_AP"]["T1"]["diff"] = 0.2
    assert check(fake) == 1
    fake8 = {"union_AP": {t: dict(a_mean=v[0], b_mean=v[1], diff=v[2], p=0.03, atlas_ahead=6, n_atlases=8, n_pools=69) for t, v in PAPER8["union_AP"].items()}}
    assert check(fake8, paper=PAPER8, n=(69, 8), p_tests=("T1",)) == 0
    fake8["union_AP"]["T2"]["p"] = 0.14  # T2 needs no p on the 8-atlas set
    assert check(fake8, paper=PAPER8, n=(69, 8), p_tests=("T1",)) == 0
    fake8["union_AP"]["T1"]["p"] = 0.14
    assert check(fake8, paper=PAPER8, n=(69, 8), p_tests=("T1",)) == 1


if __name__ == "__main__":
    if sys.argv[1] == "--selftest":
        selftest()
    else:
        s8 = {"paper": PAPER8, "n": (69, 8), "p_tests": ("T1",)} if "--set8" in sys.argv else {}
        sys.exit(
            1
            if check(
                json.load(open(sys.argv[1])),
                ("T1",) if "--no-sccad" in sys.argv else ("T1", "T2"),
                **s8,
            )
            else 0
        )
