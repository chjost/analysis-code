#!/usr/bin/python
################################################################################
#
# Author: Christopher Helmes (helmes@hiskp.uni-bonn.de)
# Date:   June 2018
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
# Datatables for chpt input data 
# 
# 
# 
# 
################################################################################

# system imports
import itertools as it
import sys
from scipy import stats
from scipy import interpolate as ip
import pandas as pd
import numpy as np
from numpy.polynomial import polynomial as P
import math
import matplotlib
matplotlib.use('Agg') # has to be imported before the next lines
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages

# Christian's packages
sys.path.append('/hiskp4/helmes/projects/analysis-code/')
import analysis2 as ana
import chiron as chi

def main():
    resdir = "/hiskp4/helmes/analysis/scattering/pi_k/I_32_blocked/results"
    filename = resdir+'/pi_K_I32_overview.h5'
    keyname = 'fse_true/data_collection'
    data = pd.read_hdf(filename,key=keyname)
    print(data.sample(n=20))
    chptdata = data.loc[data['ChPT']=='nlo_chpt']
    for tp in it.product((1,2),('A','B')):
        for poll in ('E1', 'E3'):
            subdata = chptdata.loc[(chptdata['RC'] == tp[0]) & (chptdata['ms_fix']==tp[1]) &
                    (chptdata['poll']==poll)]
            print("\n\nInput for Branch M%d%s %s" %(tp[0],tp[1],poll))
            # print a datatable for every branch
            groups = ['beta','L','mu_l']
            obs = ['mu_piK/fpi','mu_piK_a32','M_K','M_eta','M_pi']
            print(chi.print_si_format(chi.bootstrap_means(subdata,groups,obs)))
            #print(chi.bootstrap_means(subdata,groups,obs))

            
    chptdata = data.loc[data['ChPT']=='gamma']
    for tp in it.product((1,2),('A','B')):
        for poll in ('E1', 'E3'):
            subdata = chptdata.loc[(chptdata['RC'] == tp[0]) & (chptdata['ms_fix']==tp[1]) &
                    (chptdata['poll']==poll)]
            print("\n\nInput for Branch M%d%s %s" %(tp[0],tp[1],poll))
            # print a datatable for every branch
            groups = ['beta','L','mu_l']
            obs = ['M_K/M_pi','Gamma','M_K','M_eta','M_pi']
            print(chi.print_si_format(chi.bootstrap_means(subdata,groups,obs)))
            #print(chi.bootstrap_means(subdata,groups,obs))

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard Interrupt")

