#!/usr/bin/python
################################################################################
#
# Author: Christopher Helmes (helmes@hiskp.uni-bonn.de)
# Date:   December 2017
#
# Copyright (C) 2017 Christopher Helmes
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
# Collect the data from all analysis branches in one dataframe. Then calculate
# derived observables: m_pi a_32, m_pi a_12, mu_pik a_12 and tau_piK 
# The data produced is stored as a binary object. 
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
def method_df(df,tp):
    _df = df.loc[(df['ChPT']==tp[0]) & (df['poll']==tp[1]) 
           & (df['RC']==tp[2]) &(df['ms_fix']==tp[3])]
    return _df

def fitrange_averages(df,observables):
    """Calculate a weighted average over the fitranges for a given
    observable
    
    """
    chpt = ['nlo_chpt','gamma']
    epik_meth = ['E1','E3']
    zp_meth = [1, 2]
    ms_fixing = ['A', 'B']
    weighted_df = pd.DataFrame()
    # loop over all methods for weight computation
    for tp in it.product(chpt,epik_meth,zp_meth,ms_fixing):
      # choose method
      fr_col = 'fr_end'
      if tp[0] == 'gamma':
        fr_col = 'fr_bgn'
      tmp_df = method_df(df,tp)
      for o in observables:
        weighted_average_method(tmp_df,o,fr_col=fr_col)
    return weighted_df

def weighted_average_method(df,obs,fr_col='fr_end'):
    """Compute the weights of a fit from the pvalue and standard deviation of
    the fit results.

    The weights for each fitrange are calculated using the formula from
    arxiv:1506.
    w_X = [(1-2*|0.5-p|)min(\Delta X)/\Delta X]^2

    Parameters
    ----------
    df : pd.DataFrame in long format
    groups : labels of fit method

    Returns
    -------
    data frame augmented with the weights
    """
    # to use the implemented method need to reshape the results
    # pivot out fitresult with fitranges as column
    data_for_weights = df.pivot(columns='fr_end',
                                          values=obs)
    data_for_weights.info()
    pvals_for_weights = df.pivot(columns='fr_end',values = 'p-val')
    weights = ana.compute_weight(data_for_weights.values,pvals_for_weights.values)
    print(weights)
    weight_df = pd.DataFrame(data=np.atleast_2d(weights),columns=data_for_weights.columns)
    weighted_mean = pd.DataFrame()
    for fw in zip(data_for_weights.columns.values,weights):
      weighted['weight'].loc[_df['fr_end']==fw[0]] = fw[1]

    # broadcast computed weights to fitranges
    #_df['weight'] = 1.
    #print(_df.sample(n=20))
    #return _df[[obs,'fr_bgn','fr_end']]
    


def main():
    pd.set_option('display.width',1000)
    # keys for the hdf datasets
    resultdir = '/hiskp4/helmes/analysis/scattering/pi_k/I_32_blocked/results/'
    
    filename = 'pi_K_I32_overview.h5'
    keyname = '/physical_results' 
    final_results = pd.read_hdf(resultdir+filename,key=keyname)
    final_results.info()
    pd.read_hdf(resultdir+filename,key=keyname)
    fr_means = fitrange_averages(final_results,['mu_piK_a32_phys']) 
    #compute_weight_method(final_results,['ChPT','poll','RC'],['mu_piK_a32_phys'])
    print(final_results.sample(n=20))



if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard Interrupt")

