#!/bin/bash
# statistical_arms family rerun for the arms the renorm ceiling clip changes (jitter*, jitternz*, li2025), then merge + reports.
# Pre-clip copies: pcaspace_bak/preclip_stat16/. Then lupus sdiff seed 2 + lupus readers.
cd ${HR_WORK:-$HOME/work}/cfgen_natrare
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NJ=4
st(){ echo "=== $1 $(date +%H:%M:%S)"; }
J=jitter_s0.25,jitter_s0.5,jitter_s1.0,jitter_s2.0,jitter_s4.0,li2025; NZ=jitternz_s0.5,jitternz_s1.0,jitternz_s2.0
st base;  ARMS=$J CLF=lr TAG=_cj python3 statistical_arms.py run 0 20 > clipj_base.log 2>&1 && python3 clip_jitter_merge.py '' && python3 statistical_arms.py report > stat16_report.txt 2>&1
st clf;   ARMS=$J CLF=knn,rf,mlp TAG=_clfcj python3 statistical_arms.py run 0 20 > clipj_clf.log 2>&1 && python3 clip_jitter_merge.py _clf && TAG=_clf python3 statistical_arms.py report > stat16_clf_report.txt 2>&1
st nz;    ARMS=$NZ CLF=lr,knn,rf,mlp TAG=_nzcj python3 statistical_arms.py run 0 20 > clipj_nz.log 2>&1 && python3 clip_jitter_merge.py _nz && TAG=_nz python3 statistical_arms.py report > stat16_nz_report.txt 2>&1
st trhvg; REP=rep16t ARMS=$J CLF=lr,knn,rf,mlp TAG=_trhvgcj python3 statistical_arms.py run 0 20 > clipj_trhvg.log 2>&1 && python3 clip_jitter_merge.py _trhvg && REP=rep16t TAG=_trhvg python3 statistical_arms.py report > stat16_trhvg_report.txt 2>&1
st sd3clf16_report; python3 sd3clf16.py report > sd3clf16_report.txt 2>&1
st lupus_s2; python3 scdiffusion_clipped_regen.py m27_mk 2 > clipj_regen_lupus2.log 2>&1
st lupusclf; NJ=3 python3 m27_sd3clf.py run > m27_lupusclf.log 2>&1; python3 m27_sd3clf.py report > m27_lupusclf_report.txt 2>&1
st lupusmk; python3 m27_mk.py m27_mk > m27_lupusmk_report.txt 2>&1
st ALL_DONE
