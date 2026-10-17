# sourced by the embedding queue scripts. Same paths on laptop and mini (same home folder on both).
W=${HR_WORK:-$HOME/work}/jiaqi1011; C=${C:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}; O=${W2_OUT:-$W/out/w2_fm}; LG=$W/logs; mkdir -p "$O" "$LG"
PY=${PY:-$HOME/venv/bin/python}; PYT=${PYT:-${HR_WORK:-$HOME/work}/r5/venvs/tools/bin/python}      # mini: pinned pipeline venv / tools venv (scgpt)
export W2_OUT=$O OMP_NUM_THREADS=${OMP_NUM_THREADS:-3} MKL_NUM_THREADS=${MKL_NUM_THREADS:-3} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-3} NUMBA_NUM_THREADS=${NUMBA_NUM_THREADS:-3} PYTHONWARNINGS=ignore PYTHONDONTWRITEBYTECODE=1
POOLS=${W2_POOLS:-$C/pools_phase1.txt}; export W2_POOLS=$POOLS
memfree() { memory_pressure 2>/dev/null | awk '/free percentage/{gsub("%","",$NF); print $NF}' | head -1; }
gate() { local f; while f=$(memfree); [ -n "$f" ] && [ "$f" -lt 25 ]; do echo "$(date +%T) memory free ${f}% < 25%, waiting"; sleep 60; done; }
[ -n "${MARK:-}" ] && exec >> "$LG/w2_fm_$MARK.log" 2>&1
fin() { if [ -n "${MARK:-}" ]; then rm -f "$O/DONE_$MARK" "$O/FAILED_$MARK"; touch "$O/$1_$MARK"; if [ -z "${W2_NOQ:-}" ]; then d=$W/out/w2_fm_$MARK; mkdir -p "$d"; rm -f "$d/DONE" "$d/FAILED"; touch "$d/$1"; fi; fi; }
trap 'fin FAILED' ERR
