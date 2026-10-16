# sourced by every run/ script. R5 = data root, R5RUN = state dir (claims, done, runtime.csv, did_not_fit.csv, STATUS), MACHINE = label in runtime.csv.
RUNDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); CODE=$HOME/work/r5/code
export PATH=$HOME/.local/bin:$PATH                              # uv lives here on the mini; without it the pip-freeze md5 (parity marker, queue gate) is unknown/empty
export R5=${R5:-$HOME/work/r5/data} R5RUN=${R5RUN:-$RUNDIR/state}
export R5_PY=${R5_PY:-$HOME/venv/bin/python}                     # pipeline python = the PINNED parity venv (mini ~/venv, freeze md5 below; NOT the MINI_SETUP tools venv, which has scikit-learn 1.9.1). Enforced by mini_queue.sh. Laptop tests: /usr/local/bin/python3
export PYT=${PYT:-$HOME/work/r5/venvs/tools/bin/python}           # tool venv (same path on mini and laptop)
export INFRA_PY=${INFRA_PY:-python3}                              # runs timed.py (stdlib only)
export MACHINE=${MACHINE:-mini} CAPS_FILE=${CAPS_FILE:-$RUNDIR/caps.csv}
export R5_MODEL=${R5_MODEL:-$HOME/work/r5/refit_dev_nat.pkl} SI_H=$R5/hmod
export SCGPT_CKPT=${SCGPT_CKPT:-$HOME/work/r5/tools/scgpt_ckpt}
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 NUMBA_NUM_THREADS=3 PYTHONWARNINGS=ignore
export ADMIT=${ADMIT:-1}                                          # mini admission rules (timed.py); 0 = off
mkdir -p "$R5RUN"
export EXPECT_FREEZE_MD5=${EXPECT_FREEZE_MD5:-a8bfe7750e3af2ac5933be8168ce3437}   # uv pip freeze of the mini ~/venv (MACHINE_CHECK.md, PREREG section 11); laptop tests override it
