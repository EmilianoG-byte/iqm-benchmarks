#!/bin/zsh
# T3 refits of ptm_design_pipeline_2q.ipynb: 10 refits per design, then script 18; outputs data/18_<design>_output.txt
set -e
cd "$(dirname $0)/../scripts"
A="--noise-strength 0.3 --N 600 --m 2200 --L-max 28"
B="--noise-strength 0.05 --N 400 --m 150 --L-max 32"
python 17_ptm_refit_runner.py ${=A} --first 0 --last 10
python 17_ptm_refit_runner.py ${=B} --first 0 --last 10
python 17_ptm_refit_runner.py ${=A} --first 0 --last 2 --tag tight --absolute-tolerance 0.02 --num-outer 60
python 18_ptm_refit_check.py ${=A} 2>&1 | grep -v "INFO\|xla\|Complex\|x_bar" > ../data/18_default_s0.3_N600_m2200_L28_output.txt
python 18_ptm_refit_check.py ${=B} 2>&1 | grep -v "INFO\|xla\|Complex\|x_bar" > ../data/18_default_s0.05_N400_m150_L32_output.txt
echo T3 DONE
