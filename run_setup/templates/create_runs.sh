#!/bin/bash
boot=1500
# spatial, temporal and half temporal extent
l=24
t=48
t2=24

# Rundir
wrkdir=/hiskp4/helmes/analysis/scattering/pi_k/I_32_final
codedir=/hiskp4/helmes/projects/analysis-code
# slurm partition
partition=devel
# Fitrange settings
# minimal mass interval
mmin_pi=5
mmin_k=5
# minimal combined interval
cmin=7
# mass boundaries
frmi_pi=11
frmf_pi=24
frmi_k=11
frmf_k=24
# e_tot boundaries
frei=11
fref=22

# switches for job script and infiles

for i in A40.24 ; do
  #A60.24 A80.24 A100.24 
#for i in A40.20; do
#for i in B85.24; do
#for i in D45.32; do
#for i in A30.32 A40.32; do
#for i in B25.32t; do
#for i in D30.48; do
#for i in B35.32 B55.32; do
  #for s in amu_s_16 amu_s_186 amu_s_21; do
  #for s in amu_s_115 amu_s_18 amu_s_15; do
  #for s in amu_s_13 amu_s_18 amu_s_15; do
  for s in amu_s_185 amu_s_225 amu_s_2464; do
    echo "creating ensemble $i/$s"
    
    mkdir -p $i/$s
    mkdir -p ../plots/$i/$s
    cd $i/$s
    pwd

    # Job_script for kaon fits
    cp ../../templates/job_script.slurm ./fit_kaon.slurm
    sed -i "s@=N=@${i}_fit@g" fit_kaon.slurm
    sed -i "s@=E=@${i}@g" fit_kaon.slurm
    sed -i "s@=S=@${s}@g" fit_kaon.slurm
    sed -i "s@=PARTITION=@${partition}@g" fit_kaon.slurm
    sed -i "s@=WRKDIR=@${wrkdir}@g" fit_kaon.slurm
    sed -i "s@=CODEDIR=@${codedir}@g" fit_kaon.slurm
    sed -i '$ a date' fit_kaon.slurm
    sed -i "$ a ./pik_32_fit_k.py \${WORKDIR}/${i}_k.ini" fit_kaon.slurm
    sed -i '$ a date' fit_kaon.slurm

    # job script for 4point scripts
    cp ../../templates/job_script.slurm ./fit_c4.slurm
    sed -i "s@=N=@${i}_fit_c4@g" fit_c4.slurm
    sed -i "s@=E=@${i}@g" fit_c4.slurm
    sed -i "s@=S=@${s}@g" fit_c4.slurm
    sed -i "s@=PARTITION=@${partition}@g" fit_c4.slurm
    sed -i "s@=WRKDIR=@${wrkdir}@g" fit_c4.slurm
    sed -i "s@=CODEDIR=@${codedir}@g" fit_c4.slurm
    sed -i '$ a date' fit_c4.slurm
    #sed -i "$ a ./pik_32_fit_pik.py \${WORKDIR}/${i}_pik.ini" fit_c4.slurm
    sed -i "$ a ./trials/compare_epik_extractions.py \${WORKDIR}/${i}_pik.ini" fit_c4.slurm
    sed -i '$ a date' fit_c4.slurm
    
    # job script for scattering length
    cp ../../templates/job_script.slurm ./scat_len.slurm
    sed -i "s@=N=@${i}_scat_len@g" scat_len.slurm
    sed -i "s@=E=@${i}@g" scat_len.slurm
    sed -i "s@=S=@${s}@g" scat_len.slurm
    sed -i "s@=PARTITION=@${partition}@g" scat_len.slurm
    sed -i "s@=WRKDIR=@${wrkdir}@g" scat_len.slurm
    sed -i "s@=CODEDIR=@${codedir}@g" scat_len.slurm
    sed -i '$ a date' scat_len.slurm
    sed -i "$ a ./scat_len_pik \${WORKDIR}/${i}_pik.ini" scat_len.slurm
    sed -i '$ a date' scat_len.slurm

    # infile for strange mass dependent things
    cp ../../templates/ens_pik.ini ${i}_pik.ini
    sed -i "s@=E=@${i}@g" ${i}_pik.ini
    sed -i "s@=L=@${l}@g" ${i}_pik.ini
    sed -i "s@=T=@${t}@g" ${i}_pik.ini
    sed -i "s@=T2=@${t}@g" ${i}_pik.ini
    sed -i "s@=S=@${s}@g" ${i}_pik.ini
    sed -i "s@=WRKDIR=@${wrkdir}@g" ${i}_pik.ini 
    sed -i "s@=NSAM=@${boot}@g" ${i}_pik.ini
    sed -i "s@=IMIN_E=@${cmin}@g" ${i}_pik.ini
    sed -i "s@=IC4=@${frei},${fref}@g" ${i}_pik.ini

    # infile for strange mass dependent things
    cp ../../templates/ens_k.ini ${i}_k.ini
    sed -i "s@=E=@${i}@g" ${i}_k.ini
    sed -i "s@=L=@${l}@g" ${i}_k.ini
    sed -i "s@=T=@${t}@g" ${i}_k.ini
    sed -i "s@=T2=@${t}@g" ${i}_k.ini
    sed -i "s@=S=@${s}@g" ${i}_k.ini
    sed -i "s@=WRKDIR=@${wrkdir}@g" ${i}_k.ini 
    sed -i "s@=NSAM=@${boot}@g" ${i}_k.ini
    sed -i "s@=IMIN_K=@${mmin_k}@g" ${i}_k.ini
    sed -i "s@=IMASS_K=@${frmi_k},${frmf_k}@g" ${i}_k.ini
    cd ../../
  done
    cd ${i}/pi
    pwd
    
    # job script for pion fits
    cp ../../templates/job_script.slurm ./fit_pion.slurm
    sed -i "s@=N=@${i}_pion_fit@g" fit_pion.slurm
    sed -i "s@=E=@${i}@g" fit_pion.slurm
    sed -i "s@=S=@pi@g" fit_pion.slurm
    sed -i "s@=PARTITION=@${partition}@g" fit_pion.slurm
    sed -i "s@=WRKDIR=@${wrkdir}@g" fit_pion.slurm
    sed -i "s@=CODEDIR=@${codedir}@g" fit_pion.slurm
    sed -i '$ a date' fit_pion.slurm
    sed -i "$ a ./pik_32_fit_pi.py \${WORKDIR}/${i}_pi.ini" fit_pion.slurm
    sed -i '$ a date' fit_pion.slurm
    
    # infile for pion
    cp ../../templates/ens_pi.ini ${i}_pi.ini
    sed -i "s@=E=@${i}@g" ${i}_pi.ini
    sed -i "s@=L=@${l}@g" ${i}_pi.ini
    sed -i "s@=T=@${t}@g" ${i}_pi.ini
    sed -i "s@=T2=@${t}@g" ${i}_pi.ini
    sed -i "s@=WRKDIR=@${wrkdir}@g" ${i}_pi.ini 
    sed -i "s@=NSAM=@${boot}@g" ${i}_pi.ini
    sed -i "s@=IMIN_PI=@${mmin_pi}@g" ${i}_pi.ini
    sed -i "s@=IMASS_PI=@${frmi_pi},${frmf_pi}@g" ${i}_pi.ini
    cd ../../
done
