#!/bin/bash -x

#SBATCH --job-name==N= 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output==N=.out
#SBATCH --error==N=.err
#SBATCH --time=01:00:00
#SBATCH --mail-user=user@provider
#SBATCH --mail-type=FAIL
#SBATCH --mem=5G

LAT==E=
STRANGE==S=
WORKDIR=/path/to/analysis/runs/${LAT}/${STRANGE}
EXECDIR=/path/to/analysis-code

cd ${EXECDIR}
date
# Fit correlation functions including thermal states
./executable1 ${WORKDIR}/${LAT}.ini 
date
# calculate scattering length
./executable2 ${WORKDIR}/${LAT}.ini
date



