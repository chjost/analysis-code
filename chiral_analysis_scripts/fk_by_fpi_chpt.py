#!/usr/bin/python2
import sys
from scipy import stats
from scipy import interpolate as ip
import numpy as np
from numpy.polynomial import polynomial as P
import pandas as pd
import math
import matplotlib
matplotlib.use('Agg') # has to be imported before the next lines
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages

# Christian's packages
sys.path.append('/home/christopher/programming/analysis-code/')
import analysis2 as ana

def main():
################################################################################
#                   set up objects                                             #
################################################################################
    # Get parameters from initfile
    if len(sys.argv) < 2:
        ens = ana.LatticeEnsemble.parse("A40.24.ini")
    else:
        ens = ana.LatticeEnsemble.parse(sys.argv[1])
    # second system argument is fixing for ms
    ms_fixing=sys.argv[2]
    # get data from input file
    lat = ens.name()
    latA = ens.get_data("namea")
    latB = ens.get_data("nameb")
    latD = ens.get_data("named")
    strangeA = ens.get_data("strangea")
    strangeB = ens.get_data("strangeb")
    strangeD = ens.get_data("stranged")
    strange_eta_A = ens.get_data("strange_alt_a")
    strange_eta_B = ens.get_data("strange_alt_b")
    strange_eta_D = ens.get_data("strange_alt_d")
    space=['A','B','D']
   # keep seeds per zp method fixed
    zp_meth=ens.get_data("zp_meth")
    external_seeds=ens.get_data("external_seeds_%s"%(ms_fixing.lower()))
    continuum_seeds=ens.get_data("continuum_seeds_%s"%(ms_fixing.lower()))
    lat_dict = {'A':latA,'B':latB,'D':latD}
    amulA = ens.get_data("amu_l_a")
    amulB = ens.get_data("amu_l_b")
    amulD = ens.get_data("amu_l_d")
    amu_l_dict = {'A': amulA,'B': amulB, 'D': amulD}

    #dictionary of strange quark masses
    mu_s_dict = {'A': strangeA,'B': strangeB, 'D': strangeD}
    mu_s_eta_dict = {'A': strange_eta_A,'B': strange_eta_B, 'D': strange_eta_D}
    amusA = ens.get_data("amu_s_a")
    amusB = ens.get_data("amu_s_b")
    amusD = ens.get_data("amu_s_d")
    amu_s_dict = {'A': amusA,'B': amusB, 'D': amusD}
    print(amu_s_dict)
    #quark = ens.get_data("quark")
    datadir = ens.get_data("datadir") 
    plotdir = ens.get_data("plotdir") 
    resdir = ens.get_data("resultdir") 
    nboot = ens.get_data("nboot")
    # Prepare external data
    ext_data = ana.ExtDat(external_seeds,space,zp_meth)
    cont_data = ana.ContDat(continuum_seeds,zp_meth=zp_meth)
    fpi_raw = ana.read_extern("../plots2/data/fpi.dat",(1,2))
    dummies=np.loadtxt("./dummy_data_fk_fpi.txt")
    # set up dummy data for experiments
    observables = ['beta','mu_l','mu_s','sample','f_k','f_pi',
                   'M_pi','M_K','M_eta']
    results_fix_ms = pd.DataFrame(columns=observables)
    beta_vals = [1.90,1.95,2.1]
    for i,a in enumerate(space):
        for j,m in enumerate(amu_l_dict[a]):
            beta = np.full(nboot,beta_vals[i])
            mu_light = np.full(nboot,m)
            value_list = [beta,mu_light, amu_s_dict[a][0],np.arange(nboot),
                ana.draw_gauss_distributed(dummies[i+j,11],dummies[i+j,12],
                    (nboot,),origin=True),
                ana.draw_gauss_distributed(dummies[i+j,5],dummies[i+j,6],
                    (nboot,),origin=True),
                ana.draw_gauss_distributed(dummies[i+j,1],dummies[i+j,2],
                    (nboot,),origin=True),
                ana.draw_gauss_distributed(dummies[i+j,3],dummies[i+j,4],
                    (nboot,),origin=True),
                ana.draw_gauss_distributed(dummies[i+j,7],dummies[i+j,8],
                    (nboot,),origin=True)]
            tmp_frame=pd.DataFrame({key:values for key,
                                    values in zip(observables,value_list)})
            results_fix_ms = results_fix_ms.append(tmp_frame)
    print(results_fix_ms)
    
if __name__=="__main__":
    try:
        main()
    except(KeyboardInterrupt):
        print("KeyboardInterrupt")
