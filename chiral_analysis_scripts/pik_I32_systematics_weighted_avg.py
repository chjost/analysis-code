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
    #_df = df.loc[(df['ChPT']==tp[0]) & (df['poll']==tp[1]) 
    #       & (df['RC']==tp[2]) &(df['ms_fix']==tp[3])]
    _df = df.loc[(df['ChPT']==tp[0]) & (df['poll']==tp[1]) ]
    return _df

def fitrange_averages(df,observables):
    """Calculate a weighted average over the fitranges for a given
    observable
    
    """
    chpt = ['nlo_chpt','gamma']
    epik_meth = ['E1','E2']
    #zp_meth = [1, 2]
    zp_meth = [1,]
    #ms_fixing = ['A', 'B']
    ms_fixing = ['A',]
    weighted_df = pd.DataFrame()
    # loop over all methods for weight computation
    for tp in it.product(chpt,epik_meth):
        # choose method
        fr_col = 'fr_end'
        if tp[0] == 'gamma':
          fr_col = 'fr_bgn'
        tmp_df = method_df(df,tp)
        # HEuristic way to slim the dataframe
        averaged_df = tmp_df[['ChPT', 'poll','sample']].iloc[0:1500]
        # create a frame per method for the averaged observables
        for o in observables:
            #print("\ncalculate weighted average for observable %s"%o)
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
    #weighted_mean[obs] = np.sum(np.asarray([ data_for_weights.values[:,i]*weights[i] for i in
    #        range(data_for_weights.shape[1])]),axis=0)/np.sum(weights)
    weighted_mean['sample'] = np.arange(weighted_mean.shape[0])
    return weighted_mean 

def systematics_per_method(obs,col,vals,dataframe,glob_mean,asym=False):
    """For a given source get the mean of the deviations from the weighted
    median
    
    Parameters
    ----------
    dataframe:  Dataframe of the observable including weights
    glob_mean:  Global mean for distance calculation
    obs:        string, observable under investigation
    col:        string, Source of systematic uncertainty
    vals:       list of strings, Values of Source

    Returns
    -------
    mean:       numerical mean over the distances to the weighted average
    """

    dst=[]
    sys_dn,sys_up=0.,0.
    for v in vals:
        partial = dataframe.loc[dataframe[col]==v]
        partial_avg_df = partial[[obs,'fr','sample','poll','weights']]
        partial_avg = partial_avg_df.groupby(['sample']).agg(chi.weighted_mean_sample,
            (obs)).reset_index()
        #dst.append(np.abs(partial_avg[obs].iloc[0]-glob_mean))
        if partial_avg[obs].iloc[0] < glob_mean:
            sys_dn = glob_mean - partial_avg[obs].iloc[0]
        elif glob_mean < partial_avg[obs].iloc[0]: 
            sys_up = partial_avg[obs].iloc[0] - glob_mean
    dst = {'sys_dn':sys_dn,'sys_up':sys_up}
    print(dst)
    print(sys_dn,sys_up)
    if asym is not False:
        mean = dst 
    else:
        mean = (np.mean(dst.values()))
    return mean

def weighted_systematic(frame,obs,asym=False):
    """Fill a dictionary of observables with a weighted average and estimates of
    systematic uncertainties

    Parameters
    ----------
    frame:  Dataframe holdng the bootstrapsamples of the observables. Besides the
            observables the following columns are obligatory:
            'ChPT','poll', 'fr_bgn','fr_end','sample','p-val
    obs:    string, observable for which to calculate the systematic
            uncertainties
    asym:   Bool, if False the mean over the systematic error is taken

    Returns
    -------
    result: Dictionary of observables and systematic uncertainties for different
            sources
    """
    # Return a dictionary with global mean, statistical error and systematics
    # average

    test_df = frame[[obs,'ChPT','poll', 'fr_bgn','fr_end','sample','p-val']]
    #print(test_df.sample(n=20))
    # Get weights for all methods
    weight_df = pd.DataFrame()
    method_tuples = it.product(('gamma','nlo_chpt'),('E1','E2')) 
    for tp in method_tuples:
        df = method_df(test_df,tp)
        #print(df.sample(n=20))
        weight_df = weight_df.append(chi.get_weights(df,obs,
                                                    rel=True))
    print(weight_df.loc[weight_df['sample']==0])
    glob_avg_df = weight_df[[obs,'fr','sample','ChPT',
                          'poll','weights']]
    glob_avg = glob_avg_df.groupby(['sample']).agg(chi.weighted_mean_sample,
            (obs)).reset_index()
    print("Global Average of %s" %obs)
    #print(chi.bootstrap_means(glob_avg,None,obs))
    #fitrange spread
    fr_means = np.sort(weight_df.loc[weight_df['sample']==0][obs].values)[(0,-1),]
    global_mean=chi.bootstrap_means(glob_avg,None,obs).values[0,0]
    global_err=chi.bootstrap_means(glob_avg,None,obs).values[0,1]

    dst = {}
    fr_dn,fr_up = 0.,0.
    #fill dst dictionary with systematic errors
    for frm in fr_means:
        if frm < global_mean:
            fr_dn = np.abs(global_mean - frm)
        elif global_mean < frm:
            fr_up = np.abs(frm-global_mean)
    #print("Fit range systematic of %s" %obs)
    #print(fr_systematic)
    dst = {'sys_dn':fr_dn,'sys_up':fr_up}
    if asym is not False:
       fr_val = dst
    else:
        fr_val = np.mean(dst.values())

   # fr_val = 
    chpt_val = systematics_per_method(obs,'ChPT',['gamma','nlo_chpt'],
                           weight_df,global_mean,asym)
    poll_val = systematics_per_method(obs,'poll',['E1','E2'],
                                      weight_df,global_mean,asym) 
    if asym is not False:
        result = {"obs":[obs],"weighted_mean":[global_mean],"std":[global_err],
                "fr_dn":fr_val['sys_dn'],"fr_up":fr_val['sys_up'],
                "chpt_dn":chpt_val['sys_dn'],"chpt_up":chpt_val['sys_up'],
                "poll_dn":poll_val['sys_dn'],"poll_up":poll_val['sys_up']}
    else:
        result = {"obs":[obs],"weighted_mean":[global_mean],"std":[global_err],
                "fr":[fr_val],"chpt":[chpt_val],"poll":[poll_val]}
    return result
def main():
    pd.set_option('display.width',1000)
    # keys for the hdf datasets
    resultdir = '/hiskp4/helmes/analysis/scattering/pi_k/I_32_cov_false/results/'
    
    filename = 'pi_K_I32_overview.h5'
    #keyname = '/interp_corr_false/physical_results' 
    #keyname = '/fse_false/physical_results' 
    keyname = '/fse_true/physical_results' 
    #keyname = '/physical_results' 
    final_results = pd.read_hdf(resultdir+filename,key=keyname)
    final_results.info()
    pd.read_hdf(resultdir+filename,key=keyname)
    final_results['chi^2/dof'] = final_results['chi^2']/final_results['dof']
    #observables = ['mu_piK_a32_phys','L_piK','mu_piK_a12_phys','M_pi_a32_phys',
    #               'M_pi_a12_phys','tau_piK','chi^2/dof']
    #observables = ['mu_piK_a32_phys','L_piK','mu_piK_a12_phys','M_pi_a32_phys',
    #               'M_pi_a12_phys','tau_piK']
    # Acceptance test needs less observables
    observables = ['mu_piK_a32_phys','L_piK']
    # Improvise an acceptance test for refactoring systematic_spaghetti
    systematics = pd.DataFrame()
    for o in observables:
            tmp = pd.DataFrame(data=weighted_systematic(final_results,o,asym=True))
            print(tmp)
            systematics=systematics.append(tmp)
    try:
        final_systematics = systematics[['obs','weighted_mean','std','fr',
                                         'chpt','poll']]
    except:
        final_systematics = systematics[['obs','weighted_mean','std',
                                         'fr_dn','fr_up', 'chpt_dn','chpt_up',
                                         'poll_dn','poll_up']]
    print(final_systematics)
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard Interrupt")

