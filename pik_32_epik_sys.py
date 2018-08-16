#!/usr/bin/python
################################################################################
#
# Author: Christian Jost/Christopher Helmes
# (jost@hiskp.uni-bonn.de/helmes@hiskp.uni-bonn.de)
# Date:   August 2018
#
# Copyright (C) 2015 Christian Jost
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
# Script that collects epik results and stores them as a dataframe
# them in a LaTeX compatible table format
#
#
# For informations on input parameters see the description of the function.
#
################################################################################

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg') # has to be imported before the next lines
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

import analysis2 as ana
def folded_error(error):
  return np.sqrt(np.square(error[1][0]) + np.square(np.amax(error[2][0])))

def print_table_header(col_names):
  """ Print the header for a latex table"""
  print(' & '.join(col_names)+'\\\\')
  print('\midrule')

def fitres_stats(filename,par):
    # load fitresult and calculate error
    fitresult = ana.FitResult.read(filename)
    fitresult.calc_error()
    res_mean = fitresult.error[par][0][0][0]
    res_std = fitresult.error[par][1][0]
    res_sys_dn = fitresult.error[par][2][0][0]
    res_sys_up = fitresult.error[par][2][0][1]
    stats_list = [res_mean,res_std,res_sys_dn,res_sys_up]
    return stats_list

def main():

    #ens =["A30.32", 
    #      "B85.24", "D45.32"]
    ens =["A30.32", "A40.24", "A40.32", "A60.24", "A80.24", "A100.24",
          "B35.32","B55.32","B85.24", 
          "D45.32", "D30.48"]
    mus_a_fld = ["amu_s_185","amu_s_225","amu_s_2464"]
    mus_b_fld = ["amu_s_16","amu_s_186","amu_s_21"]
    mus_d_fld = ["amu_s_13","amu_s_15","amu_s_18"]
    mus_d_fld_var = ["amu_s_115","amu_s_15","amu_s_18"]
    mus_eta_a_fld =     ["strange_1850","strange_2250","strange_2464"]
    mus_eta_b_fld =     ["strange_1600","strange_1860","strange_2100"]
    mus_eta_d_fld =     ["strange_1300","strange_1500","strange_1800"]
    mus_eta_d_fld_var = ["strange_1150","strange_1500","strange_1800"]

    # Lowest mu_s values:
    #mus_a_fld = ["amu_s_185"]
    #mus_b_fld = ["amu_s_16"]
    #mus_d_fld = ["amu_s_13","amu_s_15","amu_s_18"]
    #mus_d_fld_var = ["amu_s_115"]

    # Medium mu_s values:
    #mus_a_fld = ["amu_s_225"]
    #mus_b_fld = ["amu_s_186"]
    #mus_d_fld = ["amu_s_15"]
    #mus_d_fld_var = ["amu_s_15"]

    # Highest mu_s values:
    #mus_a_fld = ["amu_s_2464"]
    #mus_b_fld = ["amu_s_21"]
    #mus_d_fld = ["amu_s_18"]
    #mus_d_fld_var = ["amu_s_18"]
    mass_fld = {"A":mus_a_fld,"B":mus_b_fld,"D":mus_d_fld}
    data = '/hiskp4/helmes/analysis/scattering/pi_k/I_32_publish/data'
    res_array = []
    for e in ens:
        ms=mass_fld[e[0]]
        if e == "D30.48":
            ms = mus_d_fld_var
        for s in ms:
            filename = '%s/%s/%s/fit_pik_%s_E1.npz'%(data,e,s,e)
            print(filename)
            tmp = [e,s]
            tmp+=fitres_stats(filename,1)
            res_array.append(tmp)
    res_df = pd.DataFrame(res_array,columns = ['ensemble','mu_s','epik','d(epik)','sdn(epik)','sup(epik)'])
    storename = "%s/%s"%(data,'epik_overview.txt')
    res_df.to_csv(storename,sep='\t')
# make this script importable, according to the Google Python Style Guide
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
