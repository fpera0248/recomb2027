"""Check the rebuilt pools of the 8-atlas test set against eval/expected/test8_pools.tsv.

    python -m eval.check_pools8 DATA_DIR            compare (DATA_DIR is $HR_DATA written by data/build_pools8.py)
    python -m eval.check_pools8 DATA_DIR --print    print the table of the pools found (to regenerate the expected file from pools.tgz)
    python -m eval.check_pools8 --selftest

Per pool: number of cells, number of targets (eval/targets.py), sha1 of the sorted cell ids joined by newlines and the
donor pack (block of the manifest). The expected values are those of the original pools (pools.tgz of the release).
A TODO cell in the expected table is not compared and is counted as unchecked. Exit status 1 on a mismatch, a
missing pool or a pool that is not in the table.
"""

import glob
import hashlib
import os
import sys

import pandas as pd

from .targets import targets

EXPECTED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "expected", "test8_pools.tsv")


def observed(data):
    man = [pd.read_csv(f, dtype=str) for f in glob.glob(f"{data}/manifests/atlas_manifest_*.csv")]
    block = pd.concat(man).set_index("pilot_id").block if man else pd.Series(dtype=str)
    out = {}
    for f in sorted(glob.glob(f"{data}/pilots/atl_*.parquet")):
        df = pd.read_parquet(f, columns=["obs_name", "label", "true_type"]).astype(str)
        ids = "\n".join(sorted(df.obs_name)).encode()
        pid = os.path.basename(f)[:-8]
        out[pid] = (len(df), len(targets(df)), hashlib.sha1(ids).hexdigest(), block.get(pid, "no manifest row"))
    return out


def compare(obs, exp):
    bad = todo = 0
    for pid in sorted(set(obs) | set(exp.index)):
        if pid not in obs or pid not in exp.index:
            print(f"{pid}: {'missing from the build' if pid not in obs else 'not in the expected table'}")
            bad += 1
            continue
        for col, got in zip(("n_cells", "n_targets", "sha1_cell_ids", "block"), obs[pid]):
            want = exp.at[pid, col]
            if want == "TODO":
                todo += 1
            elif str(got) != want:
                print(f"{pid}: {col} {got} != expected {want}")
                bad += 1
    print(f"{len(obs)} pools built, {len(exp)} expected, {bad} mismatches, {todo} cells unchecked (TODO)")
    return bad


def selftest():
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        os.makedirs(f"{d}/pilots")
        os.makedirs(f"{d}/manifests")
        pd.DataFrame({"pilot_id": ["atl_t_p0"], "block": ["d1+d2"]}).to_csv(f"{d}/manifests/atlas_manifest_t.csv", index=False)
        ids = [f"c{i}" for i in range(210)][::-1]
        lab = ["a"] * 200 + ["b"] * 10
        typ = ["a"] * 185 + ["x"] * 15 + ["b"] * 10
        pd.DataFrame({"obs_name": ids, "label": lab, "true_type": typ}).to_parquet(f"{d}/pilots/atl_t_p0.parquet")
        obs = observed(d)
        sha = hashlib.sha1("\n".join(sorted(ids)).encode()).hexdigest()
        assert obs["atl_t_p0"] == (210, 1, sha, "d1+d2"), obs
        exp = pd.DataFrame([("atl_t_p0", "210", "1", sha, "d1+d2")], columns=["pilot_id", "n_cells", "n_targets", "sha1_cell_ids", "block"]).set_index("pilot_id")
        assert compare(obs, exp) == 0
        exp.loc["atl_t_p0", "sha1_cell_ids"] = "0" * 40
        assert compare(obs, exp) == 1
        exp.loc["atl_t_p0", "sha1_cell_ids"] = "TODO"
        assert compare(obs, exp) == 0


if __name__ == "__main__":
    if sys.argv[1] == "--selftest":
        selftest()
        print("selftest ok")
    elif "--print" in sys.argv:
        print("pilot_id\tatlas\tn_cells\tn_targets\tsha1_cell_ids\tblock")
        for pid, v in observed(sys.argv[1]).items():
            print(f"{pid}\t{pid[4:].rsplit('_p', 1)[0]}\t{v[0]}\t{v[1]}\t{v[2]}\t{v[3]}")
    else:
        sys.exit(1 if compare(observed(sys.argv[1]), pd.read_csv(EXPECTED, sep="\t", dtype=str).set_index("pilot_id")) else 0)
