"""python list_units.py MAXPOOL pools|units : pool ids (set B, pool index <= MAXPOOL) or 'variant pid' lines, pool-index major."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from w6common import *
P = pools(); P = P[P.pool <= int(sys.argv[1])].sort_values(["pool", "atlas"])
for pid in P.pilot_id:
    if sys.argv[2] == "pools": print(pid)
    else:
        for v in VARIANTS: print(v, pid)
