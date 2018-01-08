#!/hiskp2/werner/libraries/Python-2.7.12/python

# Check of the GMOR-relation in the end we would like to have a plot of the eta
# masses calculated with the GMOR relation and the interpolated Eta masses as a
# function of the light quark mass.
# system imports
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
sys.path.append('/hiskp2/helmes/projects/analysis-code/')
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
################### Setup chiral analysis ######################################
    extrapol = ana.ChirAna("gell_mann_okubo_check_%s%d"%(ms_fixing.upper(),zp_meth),
                           correlated=False,gamma=False,match=True)
    # have five x-values: mk,mpi,meta,fpi,r0/a
    ens_shape_chirana = (len(latA),len(latB),len(latD))
    print(ens_shape_chirana)
    lyt_xvals = (len(space),ens_shape_chirana,1,5,1500)
    lyt_yvals = (len(space),ens_shape_chirana,1,1,1500)
    extrapol.create_empty(lyt_xvals,lyt_yvals,lat_dict=lat_dict)
    print("\nSetup complete, begin chiral analysis")
    try:
        extrapol.load(resdir)
    except:
        print("Could not load chiral analysis!")
    #Convert interpolated data to pandas dataframe (place that in an own script?)
    observables = ['beta','mu_l','mu_s','M_pi','M_K','M_eta','sample']
    results_fix_ms = pd.DataFrame(columns=observables)
    beta_vals = [1.90,1.95,2.1]
    for i,a in enumerate(space):
        for j,m in enumerate(amu_l_dict[i]):
            beta = np.full(nboot,beta_vals[i])
            mu_light = np.full(nboot,m)
            value_list = [beta,mu_light,
                          extrapol.amu_matched_to[i,j,0,0],
                          extrapol.x_data[i][j,0,0],
                          extrapol.x_data[i][j,0,1],
                          extrapol.x_data[i][j,0,4],
                          np.arange(nboot)]
            tmp_frame=pd.DataFrame({key:values for key,
                                    values in zip(observables,value_list)})
            results_fix_ms.append(tmp_frame)
    results_fix_ms.sample(n=20)
    #filename=plotdir+'/check_gell_mann_okubo_M%d%s.pdf'%(zp_meth,ms_fixing.upper())
    #chiral_plot = ana.LatticePlot(filename, join=False,debug=4)
    #chiral_plot.plot_gell_man_okubo()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
