#!/usr/bin/python
import pandas as pd
import os
import subprocess as sub

from infile_inserts import *
from jobfile_inserts import *

def create_paths(opt):
    wrkdir ='/hiskp4/helmes/analysis/scattering/pi_k/I_32_publish'
    subdirs = ['data','runs','results','plots']
    for sd in subdirs:
        sd_path = wrkdir+'/'+sd
        sd_mu_s = sd_path+'/'+opt['Ensemble']+'/'+opt['mu_s_dir'] 
        sd_pi = sd_path+'/'+opt['Ensemble']+'/'+opt['pi_dir']
        sub.check_call(['mkdir','-p', sd_mu_s])
        sub.check_call(['mkdir','-p', sd_pi])

def create_infile(correlator,options):
    dictionary = {"pion":infile_pion,
                  "kaon":infile_kaon,
                  "pik":infile_pik}
    dictionary[correlator](options)

def create_jobfile(correlator,options):
    dictionary = {"pion":jobfile_pion,
                  "kaon":jobfile_kaon,
                  "pik":jobfile_pik,
                  "scat_len":jobfile_scat_len}
    dictionary[correlator](options)

def main():
    #correlators = ['pion', 'kaon', 'pik']
    correlators = ['pion', 'kaon']
    #correlators = ['pik']
    options = pd.read_csv('./run_parameters.txt',sep='\s+',header=0)
    print(options.index)
    for i in options.index:
        ensemble_options = options.xs(i).to_dict()
        #create_paths(ensemble_options)
        print(ensemble_options)
        for corr in correlators:
            create_infile(corr,ensemble_options)
            create_jobfile(corr,ensemble_options)
        #create_jobfile("scat_len",ensemble_options)
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt")


