#!/usr/bin/python
################################################################################
#
# Author: Christopher Helmes (helmes@hiskp.uni-bonn.de)
# Date:   May 2018
#
# Copyright (C) 2018 Christopher Helmes
# 
# This program is free software: you can redistribute it and/or modify it under 
# the terms of the GNU General Public License as published by the Free Software 
# Foundation, either version 3 of the License, or (at your option) any later 
# version.
# 
# This program is distributed in the hope that it will be useful, but WITHOUT 
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS 
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with tmLQCD. If not, see <http://www.gnu.org/licenses/>.
#
################################################################################
#
# system imports
import sys
from scipy import stats
from scipy import interpolate as ip
import numpy as np
from numpy.polynomial import polynomial as P
import math
import matplotlib
matplotlib.use('Agg') # has to be imported before the next lines
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

# Christian's packages
sys.path.append('/hiskp4/helmes/projects/analysis-code/')
import analysis2 as ana

#def mk_global():
#
#def errfunc_mk_global():

def main():
################################################################################
#                   set up objects                                             #
################################################################################
    # Get parameters from initfile
    if len(sys.argv) < 2:
        ens = ana.LatticeEnsemble.parse("A40.24.ini")
    else:
        ens = ana.LatticeEnsemble.parse(sys.argv[1])

    # get data from input file
    #TODO: Could we hide that in a function?
    #print(ens)
    #lat = ens.name()
    #space=ens.get_data("beta")
    #latA = ens.get_data("namea")
    #latB = ens.get_data("nameb")
    #latD = ens.get_data("named")
    #strangeA = ens.get_data("strangea")
    #strangeB = ens.get_data("strangeb")
    #strangeD = ens.get_data("stranged")
    #strange_eta_A = ens.get_data("strange_alt_a")
    #strange_eta_B = ens.get_data("strange_alt_b")
    #strange_eta_D = ens.get_data("strange_alt_d")
    zp_meth=ens.get_data("zp_meth")
    #external_seeds=ens.get_data("external_seeds_a")
    #continuum_seeds=ens.get_data("continuum_seeds_b")
    #amulA = ens.get_data("amu_l_a")
    #amulB = ens.get_data("amu_l_b")
    #amulD = ens.get_data("amu_l_d")
    ##dictionary of strange quark masses
    #amusA = ens.get_data("amu_s_a")
    #amusB = ens.get_data("amu_s_b")
    #amusD = ens.get_data("amu_s_d")
    ## dictionaries for chiral analysis
    #lat_dict = ana.make_dict(space,[latA,latB,latD])
    #amu_l_dict = ana.make_dict(space,[amulA,amulB,amulD])
    #mu_s_dict = ana.make_dict(space,[strangeA,strangeB,strangeD])
    #mu_s_eta_dict = ana.make_dict(space,[strange_eta_A,strange_eta_B,strange_eta_D])
    #amu_s_dict = ana.make_dict(space,[amusA,amusB,amusD])
    #print(amu_s_dict)
    datadirens.get_data("datadir") 
    resdir = ens.get_data("resultdir") 
    # Load theata from the resultdir
    proc_id = 'piK_I32_unfixed_data_B%d'%(zp_meth) 
    unfixed_data_path = resdir+'/'+proc_id+'.h5' 
    unfixed_data = pd.read_hdf(unfixed_data_path,key=proc_id)
    print(unfixed_data.sample(n=20))
    # Fits take place per bootstrapsample need an errorfunction and a function
    # We also need a covariance matrix that gets inverted
    obs = ['M_K']
    cov_matrix = build_cov_matrix(obs,unfixed_data)
    #hdfstorer = pd.HDFStore(unfixed_data_path)
    #hdfstorer['raw_data'] = unfixed_data
    #hdfstorer['covariancematrix'] = cov_matrix
    #hdfstorer['fitresults']
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt")

