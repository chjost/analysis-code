#!/usr/bin/python

import sys
import numpy as np
import analysis2 as ana
import re

def main():
####################################################
# parse the input file and setup parameters
#####################################################
   # if len(sys.argv) < 2:
   #     ens = ana.LatticeEnsemble.parse("kk_I1_TP0_A40.24.ini")
   # else:
   #     ens = ana.LatticeEnsemble.parse(sys.argv[1])

   # # read settings
   # readsingledata = True
   # readsinglefit = True
   # plotsingle = False 
   # readtwodata = True
   # readtwofit = False
   # plottwo = False

   # # get data from input file
   # prefix = ens.get_data("path")
   # print prefix
   # lat = ens.name()
   # nboot = ens.get_data("nboot")
   # datadir = ens.get_data("datadir")
   # plotdir = ens.get_data("plotdir")
   # gmax = ens.get_data("gmax")
   # d2 = ens.get_data("d2")
   # try:
   #     debug = ens.get_data("debug")
   # except KeyError:
   #     debug = 0
   # T = ens.T()
   # T2 = ens.T2()
   # addT = np.ones((nboot,)) * T
   # addT2 = np.ones((nboot,)) * T2

   # # set up fit ranges (t_m is for mass, t_r for ratio)
   # # initialize as a list with numbers of correlators as length
   # t_mass = [np.int_(ens.get_data("fitmass_k")) for i in range(4)]
   # #t_ratio = [np.int_(ens.get_data("fitratio")) for i in range(4)]
   # min_size_mass = ens.get_data("tmin_mass_k")
   # #min_size_ratio = ens.get_data("tmin_ratio")
#######################################################################
# Begin calculation
#######################################################################
    # single particle correlator
    print("read single particle corrs")
    datadir = '/hiskp4/helmes/analysis/scattering/pi_k/I_32_final/data/A30.32'
    mu_s = ['amu_s_185','amu_s_225','amu_s_2464']
    #files = ["%s/pi_corr_p%d.dat" % (prefix, d) for d in range(4)]
    files = ["%s/%s/k_charged_p0.dat" % (datadir, s) for s in mu_s]
    samples = [2000,1000,500,333,250,200]
    bl = [1,2,4,6,8,10]
    for nb in zip(samples,bl):
        corr = ana.Correlators(files, matrix=False,conf_col = 2)
        corr.sym_and_boot(2000,blocking=True,bl=nb[1])
        print(corr.data.shape)
        corr.mass()
        meff_fit = ana.LatticeFit(2,-1,-1,10)
        meff = meff_fit.fit([0.1],corr,[20,29])
        #meff.print_data()
        data_arr = np.asarray([meff.data[i][:,0,0] for i in range(len(meff.data))])
        print("\nblocklength: %d\tsamples: %d\n" %(nb[1],nb[0]))
        print("\ncorrelation coefficients:\n")
        np.set_printoptions(formatter={'float':lambda x: '%06f'%x})
        print(re.sub('[ ]+', ' ', re.sub(' *[\\[\\]] *', '', np.array_str(np.corrcoef(data_arr)))))
        print("\ncovariance matrix\n")
        np.set_printoptions()
        print(re.sub('[ ]+', ' ', re.sub(' *[\\[\\]] *', '', np.array_str(np.cov(data_arr)))))
        print("\n\n")
if __name__ == '__main__':
    try:
        print("starting")
        main()
    except KeyboardInterrupt:
        pass
