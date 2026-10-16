"""hlib.load_pilot, with source h5ads redirected to SLIM_DIR (si/slim: raw counts of the needed cells only)."""
import os, sys
SI = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.environ.get("SI_H") or os.path.join(os.environ.get("R5") or os.path.expanduser("~/work/r5/data"), "hmod"))
import hlib, scanpy as sc
_read = sc.read_h5ad
def _r(path, *a, **k):
    return _read(os.path.join(os.environ["SLIM_DIR"], os.path.basename(str(path))) if os.environ.get("SLIM_DIR") else path, *a, **k)
hlib.sc = type("S", (), {"read_h5ad": staticmethod(_r), "__getattr__": lambda s, n: getattr(sc, n)})()
load_pilot = hlib.load_pilot
