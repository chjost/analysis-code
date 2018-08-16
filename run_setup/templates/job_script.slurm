#!/bin/bash -x

#SBATCH --job-name==N= 
#SBATCH --nodes=1
#SBATCH --partition==PARTITION=
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output==N=.out
#SBATCH --error==N=.err
#SBATCH --time==RUNTIME=
#SBATCH --mail-user=user@provider
#SBATCH --mail-type=FAIL
#SBATCH --mem=5G

LAT==E=
STRANGE==S=
WORKDIR==WRKDIR=/runs/${LAT}/${STRANGE}
EXECDIR==CODEDIR=

cd ${EXECDIR}



