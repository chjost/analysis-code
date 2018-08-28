#!/usr/bin/python
import pandas as pd
import os
import subprocess as sub

def main():
    jobs = ['pion','kaon']
    wrkdir ='/hiskp4/helmes/analysis/scattering/pi_k/I_32_publish/runs'
    # Loop over runfolders from parameters
    options = pd.read_csv('./run_parameters.txt',sep='\s+',header=0)
    #options = pd.read_csv('./run_parameters_test.txt',sep='\s+',header=0)
    pion_info = options[['Ensemble','pi_dir']].drop_duplicates()
    print(pion_info)
    for i,p in pion_info.iterrows():
        slurm_dir = wrkdir+'/'+p['Ensemble']+'/'+p['pi_dir']
        print("Starting pion from: %s" %slurm_dir)
        os.chdir(slurm_dir)
        sub.check_call(['sbatch','fit_pion.slurm'])

    strange_info = options[['Ensemble','mu_s_dir']]
    for i,p in strange_info.iterrows():
        slurm_dir = wrkdir+'/'+p['Ensemble']+'/'+p['mu_s_dir']
        print("Starting kaon from: %s" %slurm_dir)
        os.chdir(slurm_dir)
        sub.check_call(['sbatch','fit_kaon.slurm'])
        #sub.check_call(['sbatch','fit_c4.slurm'])
        #sub.check_call(['sbatch','scat_len.slurm'])
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
