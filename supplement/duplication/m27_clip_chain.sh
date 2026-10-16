#!/bin/bash
# Re-derive every scDiffusion number after the log1p(1e4) ceiling was added to stat16.renorm (2026-09-28).
# Pre-clip outputs are in pcaspace_bak/preclip/. Lane A: AIDA readers (mk16 cells already rewritten).
# Lane B: regenerate sdiff for the hard / lupus sets, then their readers.
cd /Users/fperaltacastro/work/cfgen_natrare
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 NJ=3 M27_THREADS=3
st(){ echo "=== $1 $(date +%H:%M:%S)"; }
laneA(){
  st sd3clf16; python3 sd3clf16.py run > clip_sd3clf16.log 2>&1; python3 sd3clf16.py report > sd3clf16_report.txt 2>&1
  st mk16; python3 mk16.py mk16 > mk16_report.txt 2>&1
  st de; python3 m27_de.py run mk16 > clip_de.log 2>&1; python3 m27_de.py report > m27_de_report.txt 2>&1
  st repl; python3 m27_repl_fix.py > m27_repl_fix_report.txt 2>&1
  st step1; M27_FORCE=1 python3 m27_step1.py > clip_step1.log 2>&1
  st latent_sdiff; M27_PSEUDO_ALL=1 M27_ARMS=sdiff,upsample M27_TAG=_pseudo_sdiff python3 m27_latent.py > m27_latent_pseudo_sdiff.log 2>&1
  st A_DONE
}
laneB(){
  st regen_hard; python3 m27_sdiff_regen.py m27_hard 0 1 2 > clip_regen_hard.log 2>&1
  st hardclf; python3 m27_hardclf.py run > m27_hardclf.log 2>&1; python3 m27_hardclf.py report > m27_hardclf_report.txt 2>&1
  st hardmk; python3 m27_hardmk.py m27_hard > m27_hardmk_report.txt 2>&1
  st regen_lupus; python3 m27_sdiff_regen.py m27_mk 0 1 > clip_regen_lupus.log 2>&1
  st lupusclf; python3 m27_sd3clf.py run > m27_lupusclf.log 2>&1; python3 m27_sd3clf.py report > m27_lupusclf_report.txt 2>&1
  st lupusmk; python3 m27_mk.py m27_mk > m27_lupusmk_report.txt 2>&1
  st B_DONE
}
laneA > m27_clip_A.log 2>&1 &
laneB > m27_clip_B.log 2>&1 &
wait
st ALL_DONE >> m27_clip_A.log
