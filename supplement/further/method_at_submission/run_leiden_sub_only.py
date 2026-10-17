"""python run_leiden_sub_only.py PILOT_ID : run_base.py restricted to m1_leiden_sub (run_base.py itself is unchanged)."""
import os, sys, runpy
HR_WORK = os.environ.get("HR_WORK", os.path.expanduser("~/work"))
sys.path.insert(0, os.environ.get("SI_H") or os.path.join(os.environ.get("R5") or HR_WORK + "/r5/data", "hmod"))
import methods as M
M.METHODS = {"m1_leiden_sub": M.METHODS["m1_leiden_sub"]}
runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_base.py"), run_name="__main__")
