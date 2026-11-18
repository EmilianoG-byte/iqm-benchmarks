#!/bin/zsh
# Design A of T3 with fully converged fits (settings of script 07 "converged"), then script 18 for both tags
set -e
cd "$(dirname $0)"
A="--noise-strength 0.3 --N 600 --m 2200 --L-max 28"
python 17_ptm_refit_runner.py ${=A} --first 0 --last 10 --tag conv --absolute-tolerance 1e-4 --num-outer 60 --relative-precision 1e-8
python 18_ptm_refit_check.py ${=A} --tag conv --reference-tag conv 2>&1 | grep -v "INFO\|xla\|Complex\|x_bar" > ../data/18_conv_s0.3_N600_m2200_L28_output.txt
python 18_ptm_refit_check.py ${=A} --tag default --reference-tag conv 2>&1 | grep -v "INFO\|xla\|Complex\|x_bar" > ../data/18_default_s0.3_N600_m2200_L28_output.txt
echo T3 CONVERGED DONE
