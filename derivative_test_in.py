#!/usr/bin/python
# A sript to get all correlationfunctions as ascii to the right position without
# loosing correlation in MC-time
import sys
import numpy as np
import os
import re
import analysis2 as ana

###############################################################################
############################### Start Main ####################################
###############################################################################
def main():
    ens = ana.LatticeEnsemble.parse(sys.argv[1])
    lat = ens.get_data("name")
    T = ens.get_data("T")
    rawdir = ens.get_data("rawdir")
    datadir = ens.get_data("datadir") 
    plotdir = ens.get_data("plotdir") 
    c = ['c%d' %i for i in np.arange(0,9)]
    Corrs = ana.inputnames(sys.argv[1],c,h5=False)
    print(Corrs[0])
    conf_feed = ["/C2+_cnfg%04d" %i for i in np.arange(714,2394,48)] 
    # Correlation function is 
    # read gamma 13 forward
    for i in range(9):
        Corr_list = []
        for cname in Corrs[i]:
            Corr_list.append(ana.read_confs(rawdir,cname,conf_feed,T,h5=True,verb=False))
    # build total correlation function
        Ctot = np.zeros_like(Corr_list[0])
        for c in Corr_list:
            Ctot += c
        ana.write_data_ascii(Ctot,datadir+'/C2+_corr%d.dat'%(i),
                         conf=["conf%4d/"%i for i in np.arange(714,2394,48)])
    #
if __name__ == "__main__":
    main()
