#!/hadron/knippsch/Enthought/Canopy_64bit/User/bin/python

# Traverse over init files of runs and retrieve fit range information

import sys
import os
from analysis2 import ensemble as ensemble
import numpy as np
from collections import namedtuple as ntp

def fitranges_from_ens(init,ens,mu_s,corr):
    if corr is "pi":
        ti = init.get_data("fitmass_pi")[0]
        tf = init.get_data("fitmass_pi")[1]
    elif corr is "k":
        ti = init.get_data("fitmass_k")[0] 
        tf = init.get_data("fitmass_k")[1] 
    else:
        ti = init.get_data("fitetot")[0] 
        tf = init.get_data("fitetot")[1] 

    fitranges=ntp("fitranges","ens mu_s corr ti tf")
    ranges=fitranges(ens,mu_s,corr,ti,tf)
    return ranges

def to_float(string):
    d=string.rsplit("_")[-1]
    return float("0.0%s"%d)

def main():
    base="/hiskp2/helmes/analysis/scattering/test/pi_k/I_32/runs"
    ens =["A30.32", "A40.24", "A40.32", "A60.24",
          "A80.24", "A100.24", "B25.32", "B35.32", "B55.32",
          "B85.24", "D45.32", "D30.48"]
    mus_a_fld = ["amu_s_185","amu_s_225","amu_s_2464"]
    mus_b_fld = ["amu_s_16","amu_s_186","amu_s_21"]
    mus_d_fld = ["amu_s_13","amu_s_15","amu_s_18"]
    mus_d_fld_var = ["amu_s_115","amu_s_15","amu_s_18"]
    header = "#Ensemble\tmu_s\tcorrelator\tt_i\tt_f"
    print(header)
    # Loop over ensembles
    for e in ens:
        if e[0] is "A":
            mus_fld=mus_a_fld
        if e[0] is "B":
            mus_fld=mus_b_fld
        if e[0] is "D":
            mus_fld=mus_d_fld
            if e is "D30.48":
                mus_fld = mus_d_fld_var
    # loop over strange quark masses
        for mu in mus_fld:
            filename = base+"/"+e+"/"+mu+"/"+e+"_pik.ini"
            ini=ensemble.LatticeEnsemble.parse(filename)
    # loop over particles
            for corr in ["pi", "k", "pik"]:
                mu_s=to_float(mu)
                fr=fitranges_from_ens(ini,e,mu_s,corr)
                print("%s\t%s\t\t\t%s\t\t\t%d\t%d" %(fr.ens, fr.mu_s, fr.corr, fr.ti, fr.tf))
            

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
