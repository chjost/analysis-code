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
      # HEuristic way to slim the dataframe
      averaged_df = tmp_df[['ChPT',
          'poll','RC','ms_fix','sample']].iloc[0:1500]
      averaged_df['RC'] = averaged_df['RC'].apply(lambda x: str(x))
      # create a frame per method for the averaged observables
      for o in observables:
        o_weighted = weighted_average_method(tmp_df,o,fr_col=fr_col)
        averaged_df = averaged_df.merge(o_weighted,on='sample')
      weighted_df = weighted_df.append(averaged_df)
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
    data_for_weights = df.pivot(columns=fr_col,
                                          values=obs)
    pvals_for_weights = df.pivot(columns=fr_col,values = 'p-val')
    weights = ana.compute_weight(data_for_weights.values,pvals_for_weights.values)
    weight_df = pd.DataFrame(data=np.atleast_2d(weights),columns=data_for_weights.columns)
    # after having calculated the weights we want to replace the fitrange
    # dimension by their weighted average
    weighted_mean = pd.DataFrame()
    # the weighted mean is given by 
    # \bar{x} = \frac{\sum_{i=1}^N w_i*x_i}/\sum_{i}^N w_i
    weighted_mean[obs] = np.sum(np.asarray([ data_for_weights.values[:,i]*weights[i] for i in
            range(data_for_weights.shape[1])]),axis=0)/np.sum(weights)
    weighted_mean['sample'] = np.arange(weighted_mean.shape[0])
    return weighted_mean 

def main():
    pd.set_option('display.width',1000)
    # keys for the hdf datasets
    resultdir = '/hiskp4/helmes/analysis/scattering/pi_k/I_32_blocked/results/'
    
    filename = 'pi_K_I32_overview.h5'
    keyname = '/interp_corr_false/physical_results' 
    #keyname = '/physical_results' 
    final_results = pd.read_hdf(resultdir+filename,key=keyname)
    final_results.info()
    pd.read_hdf(resultdir+filename,key=keyname)
    final_results['chi^2/dof'] = final_results['chi^2']/final_results['dof']
    observables = ['mu_piK_a32_phys','L_piK','mu_piK_a12_phys','M_pi_a32_phys',
                   'M_pi_a12_phys','tau_piK','chi^2/dof']
    #observables = ['mu_piK_a32_phys','L_piK','chi^2/dof']
    groups_fr = ['ChPT','poll','RC','ms_fix','fr_bgn','fr_end']
    groups_wofr = ['ChPT','poll','RC','ms_fix']
    fr_means = fitrange_averages(final_results,observables)
    print(chi.print_si_format(chi.bootstrap_means(final_results,groups_fr,observables)))
    print(chi.print_si_format(chi.bootstrap_means(fr_means,groups_wofr,observables)))
    #compute_weight_method(final_results,['ChPT','poll','RC'],['mu_piK_a32_phys'])
    sources = ['poll','ms_fix','RC','ChPT']
    observables = ['L_piK','mu_piK_a32_phys',
                   'M_pi_a32_phys','M_pi_a12_phys','tau_piK']
    #observables = ['L_piK','mu_piK_a32_phys']
    final_result = chi.get_systematics(fr_means,sources,observables)
    print(final_result)



if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard Interrupt")

