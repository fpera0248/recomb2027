"""Independent check of the exact sign-flip tests in an analysis result file.

    python -m eval.signflip RESULT.json [...]

Recomputes p = #{s in {+1,-1}^A : |sum s_a d_a| >= |sum d_a| - 1e-12} / 2^A by plain enumeration
(not the numpy code that wrote the file) and checks that the reported p, difference and number of
atlases ahead match, that p is at least the two-sided floor 2/2^A, and that the pass flag is
never looser than mean > 0 and p < 0.05. Exit status 1 on any mismatch.
"""

import json
import sys


def p_exact(d):
    tot, n, hits = abs(sum(d)), len(d), 0
    for m in range(2**n):
        if abs(sum(x if m >> i & 1 else -x for i, x in enumerate(d))) >= tot - 1e-12:
            hits += 1
    return hits / 2**n


def check(path):
    r, bad = json.load(open(path)), 0
    for metric in ("union_AP", "E1_union"):
        for t, v in (r.get(metric) or {}).items():
            if not isinstance(v, dict) or "per_atlas_diff" not in v:
                continue
            d = [x for x in v["per_atlas_diff"].values() if x == x]
            A = len(d)
            if not A:
                continue
            p, mean = p_exact(d), sum(d) / A
            ok = (
                abs(p - v["p"]) < 1e-12
                and abs(mean - v["diff"]) < 1e-9
                and sum(x > 0 for x in d) == v["atlas_ahead"]
                and p >= 2 / 2**A - 1e-15
            )
            ok_pass = (
                v["pass"] == (mean > 0 and p < 0.05) or not v["pass"]
            )  # the file may add stricter conditions, never looser
            bad += not (ok and ok_pass)
            print(
                f"{metric:9s} {t}: A={A:2d} diff={mean:+.4f} ahead={sum(x > 0 for x in d)}/{A} p={p:.5f} (file {v['p']:.5f}) pass={v['pass']} {'OK' if ok and ok_pass else 'MISMATCH'}"
            )
    return bad


if __name__ == "__main__":
    assert (
        p_exact([1, 1, 1]) == 0.25
        and p_exact([1, -1]) == 1.0
        and p_exact([1, 1, 1, 1, 1]) == 0.0625
    )
    sys.exit(1 if sum(check(f) for f in sys.argv[1:]) else 0)
