#!/bin/bash
# usage: laptop_lock_run.sh CMD... : wait for ~/work/jiaqi1011/.laptop_lock, run CMD under it, release (only if we took it).
L=$HOME/work/jiaqi1011/.laptop_lock
until mkdir "$L" 2>/dev/null; do sleep 30; done
trap 'rmdir "$L"' EXIT
"$@"
