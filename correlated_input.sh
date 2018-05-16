#!/bin/bash
infile_path=/hiskp4/helmes/projects/analysis-code/ini/pi_K/I_32/gathering_correlators/
#for i in A40.24; do
for i in A60.24 A80.24 A100.24; do
#for i in A40.20; do
#for i in B85.24; do
#for i in D45.32; do
#for i in A30.32 A40.32; do
#for i in B25.32t; do
#for i in D30.48; do
#for i in B35.32 B55.32; do
  infile=gather_corr_${i}.ini
  echo ${infile_path}
  echo ${infile}
  ./strange_all_in.py ${infile_path}/${infile}
  ./omit_outliers ${infile_path}/${infile}
  ./plot_corrs.py ${infile_path}/${infile}
done
