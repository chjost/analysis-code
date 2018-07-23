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
# Interpolate the observables $M_K^2$, $M_{\eta}^2$, $\mu_{\pi K}a_0$ and
# $\mu_{\pi K}$ for one ensemblke and method
# 
# Load a dataframe from a fixing fit (hdf5 pandas dataframe)
# Calculate the bare strange quark mass
# interpolate in all ensembles and observables to that strange quark mass
# save the result as a new pandas dataframe
################################################################################

# system imports
import argparse
import sys
from scipy import stats
from scipy import interpolate as ip
from scipy import optimize as opt
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
sys.path.append('/hiskp4/helmes/projects/analysis-code/')
import analysis2 as ana
import chiron as chi
def amk_sq(p,x):
    # Parameters are equal for each beta, input data is not
    return p[0]+p[1]*x

def linear_errfunc(p,x,y,cov_iu):
    # cov_u is the upper triangular matrix of cov
    f_vector = np.r_[amk_sq(p,x)]
    return np.dot(cov_iu,y-f_vector)
def get_covariance(dataframe):
    return dataframe.cov().values
def pivot_dataframe(dataframe,obs_name,ordering,bname):
    """Pivot a dataframe from long format for calculation of a covariance matrix

    The dataframe gets cut based on the conjunction of observable name, ordering
    keys and smaples. The result is a dataframe with observable values for each
    ensemble as columns and
    samples as rows

    Parameters
    ----------
    dataframe: pd-dataframe, Observable data containing at least one observable
                column and one sample column in long data format
    obs_name: str, name of the observable that gets pivoted
    ordering: list of str, parameters by which the values get ordered
    bname: name of the sample column

    Returns
    -------
    _cut_pivot: pd dataframe as described above
    """
    # cut data TODO there has to be a better solution this has advantage of a
    # generalized ordering (beta,mu_l) or (beta, mu_l, mu_s) 
    _l = list((obs_name,ordering,bname))
    _cut_indices = [obs_name,]
    for o in ordering:
        _cut_indices.append(o)
    _cut_indices.append(bname)
    _df_cut = dataframe[_cut_indices]
    _df_cut['ensemble'] = list(_df_cut[ordering].itertuples(index=False,
                                                                   name=None))
    _cut_pivot = _df_cut.pivot_table(index = bname,columns='ensemble',values=obs_name)
    return _cut_pivot
def get_dataframe_fit(dataframe,y_obs,order,bname,priors=None):
    """Get inverse covariance matrix from nbootd data
    
    The dataframe containing x and y nboots is split up. For each y-observable it
    gets pivoted accordingly and then concatenated to a dataframe with the
    nboots as rows. Afterwards the inverse covariance matrix is calculated as
    np.linalg.inv(pd.DataFrame.cov())

    Parameters
    ----------
    dataframe: pd-dataframe containing the nboots for all ensembles and
               observables
    y_obs: str, y-observables for covariance matrix
    order: list of str, ordering parameters for the covariance matrix
    bname: str, name of the nboot column

    Returns
    -------
    _cov_inv: nd-array, the inverted covariance matrix
    """
    _cov_df_y = pivot_dataframe(dataframe,y_obs,order,bname)
    if priors is not None:
        _covariance_df = pd.concat([_cov_df_y,priors],axis=1)
    else:
        _covariance_df = _cov_df_y
    return _covariance_df
def pick_ensemble(df,columns,values):
    if len(columns) != len(values):
        print("Columns and values have different lengths!")
    else:
        picked = df.loc[(df[columns[0]]==values[0]) &
                        (df[columns[1]]==values[1]) &
                        (df[columns[2]]==values[2])]
        return picked

def main():
    # choose fitrange ms_fixing and epik-extraction per command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile",help="infile for paths",type=str,
                        required=True)
    parser.add_argument("--zp", help="Method of RC Z_P",type=int, required=True)
    parser.add_argument("--epik", help="Which method of fitting E_piK",type=int,
                        required=True)
    parser.add_argument("--msfix",help="Method for fixing ms",type=str,required=True)
    args = parser.parse_args()
    # Get presets from analysis input files
    ens = ana.LatticeEnsemble.parse(args.infile)
    nboot = ens.get_data("nboot")
    plotdir = ens.get_data("plotdir") 
    resdir = ens.get_data("resultdir") 
    filename = resdir + '/piK_I32_unfixed_data_B1.h5'
    unfixed_data = pd.read_hdf(filename)
    pd.read_hdf(filename)
    # Pick the data based on (beta,L,mu_l)
    #obs = 'mu_piK_a32'
    obs = 'M_eta^2'
    ens_id = [1.90,24,0.0080]
    # range for the plot
    mu_s = [0.0185,0.02464]
    fit_df = pick_ensemble(unfixed_data,['beta','L','mu_l'],ens_id)
    print("Picked data")
    fit_df.info()
    # To fit a function we need x and y data and a covariance matrix
    # get the xdata without any errors
    #xdata = fit_df[['beta','L','mu_l','M_K/M_pi']].where(fit_df['nboot']==0).dropna()
    #xdata = xdata.set_index(['beta','L','mu_l'],drop=True).sort_index()
    xdata = fit_df[['mu_s']].where(fit_df['nboot']==0).dropna()
    # get the ydata
    ydata = get_dataframe_fit(fit_df,obs,['beta','L','mu_l','mu_s'],
                              'nboot')
    ydata.info()
    data_for_cov = ydata 
                  
    data_for_cov.info()
    cov = get_covariance(data_for_cov)
    # Our fit is correlated
    cov = np.diag(np.diagonal(cov))
    print(cov)
    cov_iu = np.linalg.cholesky(np.linalg.inv(cov)).T
    print(cov_iu)
    ## Fitresults dataframe by beta
    col_names = ['slope','intercept']
    fitres = pd.DataFrame(columns = col_names)
    xp = np.arange(xdata.shape[0])
    start = (1.,0.1)
    p = np.r_[start]
    # we have x uncertainties in play, set degrees of freedom manually, 2
    # parameters and 1 prior
    dof_data = fit_df.where(fit_df['nboot']==0).dropna().shape[0]
    dof =  dof_data - len(start)
    print('degrees of freedom are: %d' %dof)
    for b in np.arange(nboot):
        _tmp_fitres = opt.least_squares(linear_errfunc, p,
                args=(xdata.values[:,0], ydata.iloc[b].values, cov_iu))
        _chisq = 2*_tmp_fitres.cost
        _pval = 1-stats.chi2.cdf(_chisq,dof)
        _tmp_pars = dict(zip(col_names,_tmp_fitres.x[-2:]))
        _res_dict = {'nboot':b, 'chi^2':_chisq,'dof':dof,'p-val':_pval}
        _res_dict.update(_tmp_pars)
        _tmpdf = pd.DataFrame(data = _res_dict,index=[b])
        fitres = fitres.append(_tmpdf)
        #if b%100 == 0:
        #    print(_res_dict)
    fitres.info()
    fit_df=fit_df.merge(fitres,on='nboot')
    fit_df.info()
    #print(fitres.nboot(n=20))
    plot_df = fit_df[['beta','L','mu_l','mu_s',obs]]
    print(chi.print_si_format(chi.bootstrap_means(fit_df,['beta','L','mu_l'],
                        ['intercept','slope','chi^2','dof'])))
    groups = ['beta','L','mu_l','mu_s']
    observables = [obs]
    plot_means = chi.bootstrap_means(plot_df,groups,observables)
    # make a plot
    plotname = plotdir+'/pi_K_I32_ens_interp_M%d%s_E%d.pdf'%(args.zp,
                        args.msfix,args.epik) 
    with PdfPages(plotname) as pdf:
        plt.xlabel(r'$a\mu_s$')
        plt.ylabel('Observable')
        x = xdata.values[:,0]
        y = plot_means.loc[:,[(obs,'own_mean')]].values
        yerr = plot_means.loc[:,[(obs,'own_std')]].values[:,0]
        print(x,y,yerr)
        plt.errorbar(x,y,yerr=yerr,fmt='xb',
                     label = 'data')
        # an errorband is derived from filling the maximal and minimal
        # curve
        p = fit_df[['slope','intercept']].values[0:1500].T
        print(p.shape)
        x = np.linspace(mu_s[0],mu_s[1],num=300)
        #original sample is at 0
        y = amk_sq(p[:,0],x)
        yext = np.asarray([amk_sq(p,m) for m in x])
        print(yext.shape)
        ymin = yext.min(axis=1)
        ymax = yext.max(axis=1)
        print(x.shape)
        print(y.shape)
        print(ymin.shape)
        print(ymax.shape)
        plt.errorbar(x,y,fmt='--b',label='linear fit')
        plt.fill_between(x,ymin,ymax,color='blue',alpha=0.4)
        plt.legend()
        pdf.savefig()
        plt.clf()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard Interrupt")


