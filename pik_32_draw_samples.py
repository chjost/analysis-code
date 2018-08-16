#!/usr/bin/python

# Script to draw bootstrapsamples from Correlation functions 
# 
# 
# 
# 

import sys
import numpy as np
import itertools
# Christian's packages
sys.path.append('/hiskp4/helmes/projects/analysis-code/')

import analysis2 as ana
def main():
    # parse infile
    if len(sys.argv) < 2:
        ens = ana.LatticeEnsemble.parse("kk_I1_TP0_A40.24.ini")
    else:
        ens = ana.LatticeEnsemble.parse(sys.argv[1])
    # get data from input file
    prefix = ens.get_data("path")
    print prefix
    lat = ens.name()
    nboot = 10000 
    bs_bl = ens.get_data("boot_bl")
    readdata = False 
    read_kfit = False
    corr_pi_in ="pi_charged_p0"
    corr_pi_out = "corr_pi_bs10000"
    corr_k_in ="k_charged_p0"
    corr_k_out = "corr_k_bs10000"
    corr_pik_in = "pik_charged_A1_TP0_00" 
    corr_pik_out = "corr_pik_bs10000"
    # get data from input file
    prefix = ens.get_data("path")
    print prefix
    lat = ens.name()
    datadir = ens.get_data("datadir")
    datadir_pi = ens.get_data("datadir_pi")
    gmax = ens.get_data("gmax")
    d2 = ens.get_data("d2")
    try:
        debug = ens.get_data("debug")
    except KeyError:
        debug = 0
    L = ens.L()
    T = ens.T()
    T2 = ens.T2()
    addT = np.ones((nboot,)) * T
    addT2 = np.ones((nboot,)) * T2

    # Bootstrap pion
    print("read single particle corrs")
    files = ["%s/%s.dat" % (datadir_pi,corr_pi_in)]
    pi_corr = ana.Correlators(files, matrix=False)
    pi_corr.sym_and_boot(nboot,bl=bs_bl,method='stationary')
    print(pi_corr.shape)
    pi_corr.save("%s/%s_%s.npy" % (datadir,corr_pi_out , lat))

    # Bootstrap kaon
    print("read single particle corrs")
    files = ["%s/%s.dat" % (datadir,corr_k_in)]
    k_corr = ana.Correlators(files, matrix=False)
    k_corr.sym_and_boot(nboot,bl=bs_bl,method='stationary')
    print(k_corr.shape)
    k_corr.save("%s/%s_%s.npy" % (datadir,corr_k_out , lat))

    # Bootstrap pik
    files = ["%s/%s.dat" % (datadir,corr_pik_in)]
    pik_corr = ana.Correlators(files, matrix=False,conf_col=3)
    # symmetrize and bootstrap
    pik_corr.sym_and_boot(nboot,bl=bs_bl,method='stationary')
    pik_corr.save("%s/%s_%s.npy" % (datadir,corr_pik_out, lat))

if __name__ == '__main__':
    try:
        print("starting")
        main()
    except KeyboardInterrupt:
        pass

