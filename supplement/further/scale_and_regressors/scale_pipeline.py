"""W1c: time + memory of each pipeline step vs pool size. Calls the LOCKED code in ~/work/r5/code (never copied/edited).
  $R5_PY scale_pipeline.py run          driver (needs anndata for the atlas pick, so run it with the venv python): ladder of sizes, one /usr/bin/time -l subprocess per step -> $OUT/scale.csv
  $R5_PY  scale_pipeline.py build N SEED builds one pool (called by the driver) into $R5 (scale data root)
  python3 scale_pipeline.py selftest
Env: R5 (scale data root, REQUIRED, never ~/work/r5/data), R5_PY, R5_MODEL, OUT, ATLAS_DIR, SCALE_NAME (atlas name, default = largest set-B atlas by n_obs),
SIZES ("n:seed,..."), MEM_FREE_MIN (25), RSS_CAP_GB (18), CLEAN (1 = delete slim pool after its size)."""
import os, sys, re, csv, time, subprocess, math, glob
H = os.path.expanduser("~/work/r5"); CODE = f"{H}/code"
E = os.environ.get; R5 = E("R5", ""); OUT = E("OUT", os.path.expanduser("~/work/jiaqi1011/out/w1_scale"))
ATLAS_DIR = E("ATLAS_DIR", f"{H}/data/atlas")
SIZES = [(int(a), int(b)) for a, b in (x.split(":") for x in E("SIZES", "5000:0,10000:0,20000:0,20000:1,20000:2,50000:0,100000:0,200000:0").split(","))]
MEM_MIN, CAP_MB = float(E("MEM_FREE_MIN", "25")), float(E("RSS_CAP_GB", "18")) * 1024
STEPS = ["build", "sr3", "sr3m", "sr3h", "m1_leiden_sub", "learn14_predict", "frozen4"]
BASE_MB = {"build": 3000, "sr3": 3209, "sr3m": 3209, "sr3h": 3209, "m1_leiden_sub": 3053, "learn14_predict": 312, "frozen4": 115}   # 20k medians, ~/work/r5/runtime_all.csv; first-size prediction only
COLS = ["n", "seed", "step", "status", "wall_s", "cpu_s", "load1", "max_rss_mb", "out_bytes", "exit", "pred_rss_mb", "machine", "n_cells", "note"]


def parse_time(txt):
    """/usr/bin/time -l (macOS) stderr -> (wall_s, rss_mb, cpu_s = user+sys)"""
    w = re.search(r"^\s*([\d.]+)\s+real\s+([\d.]+)\s+user\s+([\d.]+)\s+sys", txt, re.M); r = re.search(r"^\s*(\d+)\s+maximum resident set size", txt, re.M)
    return (float(w.group(1)) if w else None, int(r.group(1)) / 2**20 if r else None, float(w.group(2)) + float(w.group(3)) if w else None)


def predict_mb(hist, n):
    """hist = [(n, rss_mb)] of this step. Linear from the largest-n point; exponent from the last two distinct n if available (clamped 1..2)."""
    if not hist: return None
    by = {}
    for k, v in hist: by[k] = max(by.get(k, 0), v)
    ks = sorted(by); n2 = ks[-1]; e = 1.0
    if len(ks) > 1 and by[ks[-2]] > 0: e = min(2.0, max(1.0, math.log(by[n2] / by[ks[-2]]) / math.log(n2 / ks[-2])))
    return by[n2] * (n / n2) ** e


def free_pct():
    out = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout; m = re.search(r"free percentage:\s*(\d+)%", out)
    if not m: print("memory_pressure output not parsed (gate closed):", out[-300:], flush=True)
    return int(m.group(1)) if m else 0


def fsize(paths): return sum(os.path.getsize(p) for p in paths if os.path.exists(p))


def build(n, seed):
    sys.path.insert(0, CODE); sys.path.insert(0, f"{CODE}/hmod"); os.environ.setdefault("SI_H", f"{CODE}/hmod")
    import numpy as np, build_r5 as br                  # R5 env set before import -> br.b.OUT / PIL point at the scale root
    br.b.AT = ATLAS_DIR; br.LK = f"{H}/data/leak5"      # atlas files + read-only overrides.csv from the real tree; nothing is written there
    name = E("SCALE_NAME") or pick_atlas()[0]
    P = br.prep(name, None, None)
    if name in br.NPDROP:                               # prep(ref=None) skips the builder's NPDROP rule; apply it here (is_primary_data)
        import anndata as ad; a = ad.read_h5ad(f"{ATLAS_DIR}/{br.ATL[name]['id']}.h5ad", backed="r"); prim = (a.obs["is_primary_data"].astype(str) == "True").values; a.file.close()
        k = prim[P["pos"]]; P["o"] = P["o"][k]; P["pos"] = P["pos"][k]; P["n_cells"] = int(k.sum()); P["donors"] = sorted(P["o"].donor_id.unique())
    N = P["n_cells"]
    if n > N: print(f"ATLAS_SMALL {N} usable cells < pool {n}", flush=True); sys.exit(3)
    o = P["o"]; by = o.groupby("donor_id").indices; cur = []   # builder logic: donors in seeded permutation (seed 0 = builder order), accumulate to >= n, sample n
    for d in np.random.RandomState(seed).permutation(sorted(by)):
        cur.append(by[d])
        if sum(map(len, cur)) >= n: break
    r = np.sort(np.random.RandomState(seed).choice(np.concatenate(cur), n, replace=False))
    pn = f"scale_n{n}_s{seed}"; br.write_pool(P, pn, "atl_" + pn, r, "scale")
    print("built", pn, n, "donors", len(cur), "of", len(by), flush=True)


def pick_atlas():
    """largest set-B atlas by n_obs (backed read); set B = build_r5.ATL ids present in ATLAS_DIR; SCALE_NAME overrides (laptop test: luad)."""
    sys.path.insert(0, CODE); import anndata as ad, build_r5 as br
    if E("SCALE_NAME"): n = E("SCALE_NAME"); return n, None
    best = None
    for k, c in br.ATL.items():
        if c["role"] != "primary": continue             # set B = the 10 primaries; alternates excluded
        f = f"{ATLAS_DIR}/{c['id']}.h5ad"
        if os.path.exists(f):
            a = ad.read_h5ad(f, backed="r"); no = a.n_obs; a.file.close()
            if best is None or no > best[1]: best = (k, no)
    return best


def run():
    assert R5 and os.path.realpath(R5) != os.path.realpath(f"{H}/data"), "R5 must be the scale data root"
    os.makedirs(OUT, exist_ok=True); os.makedirs(f"{OUT}/logs", exist_ok=True); csvf = f"{OUT}/scale.csv"; mach = E("MACHINE", "mini")
    rows = list(csv.DictReader(open(csvf))) if os.path.exists(csvf) else []
    if not rows: csv.writer(open(csvf, "w")).writerow(COLS)
    done = {(int(r["n"]), int(r["seed"]), r["step"]) for r in rows if r["status"] == "ok"}
    hist = {s: [(int(r["n"]), float(r["max_rss_mb"])) for r in rows if r["step"] == s and r["status"] == "ok"] for s in STEPS}
    py = E("R5_PY", "python3"); model = E("R5_MODEL", os.path.expanduser("~/work/r5/refit_dev_nat.pkl")); me = os.path.abspath(__file__)
    atl, natl = (E("SCALE_NAME"), None) if E("SCALE_NAME") else pick_atlas()
    print("atlas", atl, "n_obs", natl, flush=True)
    base = {**os.environ, "SCALE_NAME": atl, "SI_H": f"{CODE}/hmod", "SLIM_DIR": f"{R5}/slim_atlas", "PYTHONWARNINGS": "ignore"}
    sr3env = {**base, **{v: "2" for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")}}   # workers use 2 threads
    failed = False
    def log(n, s, step, status, wall=None, rss=None, ob=None, ex=None, pred=None, nc=None, note="", cpu=None, load=None):
        r = dict(n=n, seed=s, step=step, status=status, wall_s=wall, cpu_s=cpu, load1=load, max_rss_mb=rss, out_bytes=ob, exit=ex, pred_rss_mb=pred, machine=mach, n_cells=nc, note=note)
        csv.DictWriter(open(csvf, "a"), COLS).writerow(r); print(r, flush=True)
    for n, s in SIZES:
        pid = f"atl_scale_n{n}_s{s}"; C, Hc = f"{R5}/cands/{pid}", f"{R5}/preds_m14p/m14p_frozen/{pid}.parquet"
        ids = f"{OUT}/ids_{pid}.txt"; open(ids, "w").write(pid + "\n")
        cmds = {"build": ([py, me, "build", str(n), str(s)], base, [f"{R5}/slim_atlas/scale_n{n}_s{s}.h5ad", f"{R5}/hmod/pilots/{pid}.parquet"]),
                "sr3": ([py, f"{CODE}/sr3.py", pid], sr3env, [f"{C}.groups.parquet", f"{C}.cells.parquet"]),
                "sr3m": ([py, f"{CODE}/sr3m.py", pid], sr3env, [f"{C}.markers.parquet"]),
                "sr3h": ([py, f"{CODE}/sr3h.py", pid], sr3env, [f"{C}.halo.parquet"]),
                "m1_leiden_sub": ([py, f"{CODE}/run_m1_only.py", pid], base, [f"{R5}/preds_base/m1_leiden_sub/{pid}.parquet"]),
                "learn14_predict": ([py, f"{CODE}/learn14.py", "predict", model, pid, Hc, "P4_s1"], {**base, "FEATS": "mkh"}, [Hc]),
                "frozen4": ([py, f"{CODE}/frozen4.py", ids], base, [f"{R5}/preds_frozen4/frozen4/{pid}.parquet"])}
        stop = False
        for step in STEPS:
            if (n, s, step) in done and (step != "build" or os.path.exists(f"{R5}/slim_atlas/scale_n{n}_s{s}.h5ad")): continue   # rebuild if CLEAN removed the pool
            pr = predict_mb(hist[step], n) or BASE_MB[step] * n / 20000
            if pr > CAP_MB: log(n, s, step, "DNF", pred=round(pr), note=f"predicted RSS {pr/1024:.1f} GB > cap {CAP_MB/1024:.0f} GB"); stop = True; break
            for _ in range(10):
                if free_pct() >= MEM_MIN: break
                time.sleep(60)
            else: log(n, s, step, "DNF", pred=round(pr), note=f"memory_pressure free < {MEM_MIN}% after 10 min"); stop = True; break
            lf = f"{OUT}/logs/{pid}.{step}.log"; t0 = time.time(); load = round(os.getloadavg()[0], 2)
            with open(lf, "w") as f: p = subprocess.run(["/usr/bin/time", "-l", *cmds[step][0]], env=cmds[step][1], stdout=f, stderr=f)
            wall, rss, cpu = parse_time(open(lf).read()); wall = wall if wall is not None else time.time() - t0
            if p.returncode == 3 and step == "build": log(n, s, step, "DNF", wall, rss, None, 3, round(pr), note="atlas has fewer usable cells than this size (ladder stops)", load=load); stop = True; break
            if p.returncode: log(n, s, step, "FAIL", wall, rss, None, p.returncode, round(pr), note=f"see {lf}", load=load); stop = failed = True; break
            log(n, s, step, "ok", round(wall, 2), round(rss, 1), fsize(cmds[step][2]), 0, round(pr), n if step == "build" else None, cpu=round(cpu, 2) if cpu is not None else None, load=load)
            hist[step].append((n, rss))
        if E("CLEAN", "1") == "1" and not stop: [os.remove(f) for f in glob.glob(f"{R5}/slim_atlas/scale_n{n}_s{s}.h5ad")]
        if stop: break
    open(f"{OUT}/{'FAILED' if failed else 'DONE'}", "w").write(time.ctime() + "\n")


def selftest():
    assert parse_time("        1.50 real   1.0 user  0.2 sys\n  123456789  maximum resident set size\n") == (1.5, 123456789 / 2**20, 1.2)
    assert abs(predict_mb([(1, 10)], 4) - 40) < 1e-9
    assert abs(predict_mb([(1, 10), (2, 40)], 4) - 160) < 1e-6 and predict_mb([(1, 10), (2, 12)], 4) == 24   # exponent clamped to >= 1
    print("selftest ok")


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "build": build(int(sys.argv[2]), int(sys.argv[3]))
    elif c == "run": run()
    else: selftest()
