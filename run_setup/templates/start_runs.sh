#!/bin/bash

#for i in D30.48; do
#for i in A40.20; do
#for i in D45.32; do
for  i in A40.24; do
#for i in A40.24 A60.24 A80.24 A100.24; do
#for i in A30.32 A40.32; do
#for i in B35.32 B55.32; do
#for i in B85.24; do
  for s in amu_s_185 amu_s_225 amu_s_2464; do
  #for s in pi; do
  #for s in amu_s_16 amu_s_186 amu_s_21 pi; do
  #for s in amu_s_13 amu_s_15 amu_s_18; do
  #for s in amu_s_115 amu_s_15 amu_s_18; do
    echo "start fit on ensemble $i/$s"
    cd $i/$s
    #sbatch job_script.slurm
    sbatch fit_c4.slurm
  cd ../../
  done
done
