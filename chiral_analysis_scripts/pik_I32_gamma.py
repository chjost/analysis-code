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
# Extrapolate the data for $mu_{\pi K} a_0$ to the physical point in terms of
# linearised NLO ChPT.
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
################################################################################

# system imports
import sys
from scipy import stats
from scipy import interpolate as ip
from scipy import optimize as opt
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
sys.path.append('/home/christopher/programming/analysis-code/')
import analysis2 as ana
import chiron as chi

def get_dataframe_fit_xy(dataframe,x_obs,y_obs,order,bname,priors=None):
    """Get inverse covariance matrix from sampled data
    
    The dataframe containing x and y samples is split up. For each observable it
    gets pivoted accordingly and then concatenated to a dataframe with the
    samples as rows. Afterwards the inverse covariance matrix is calculated as
    np.linalg.inv(pd.DataFrame.cov())

    Parameters
    ----------
    dataframe: pd-dataframe containing the samples for all ensembles and
               observables
    x_obs: str, x-observable for covariance matrix
    y_obs: str, y-observables for covariance matrix
    order: list of str, ordering parameters for the covariance matrix
    bname: str, name of the sample column

    Returns
    -------
    _cov_inv: nd-array, the inverted covariance matrix
    """
    _cov_df_x = pivot_dataframe(dataframe,x_obs,order,bname)
    _cov_df_y = pivot_dataframe(dataframe,y_obs,order,bname)
    if priors is not None:
        _covariance_df = pd.concat([_cov_df_x,_cov_df_y,priors],axis=1)
    else:
        _covariance_df = pd.concat([_cov_df_x,_cov_df_y],axis=1)
    return _covariance_df

def get_dataframe_fit(dataframe,y_obs,order,bname,priors=None):
    """Get inverse covariance matrix from sampled data
    
    The dataframe containing x and y samples is split up. For each y-observable it
    gets pivoted accordingly and then concatenated to a dataframe with the
    samples as rows. Afterwards the inverse covariance matrix is calculated as
    np.linalg.inv(pd.DataFrame.cov())

    Parameters
    ----------
    dataframe: pd-dataframe containing the samples for all ensembles and
               observables
    y_obs: str, y-observables for covariance matrix
    order: list of str, ordering parameters for the covariance matrix
    bname: str, name of the sample column

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

def get_inverse_covariance(dataframe):
    _cov =dataframe.cov().values 
    return np.linalg.cholesky(np.linalg.inv(_cov)).T
def get_uncorrelated_inverse_covariance(dataframe):
    _cov =dataframe.cov().values
    _cov_diag =np.diag(np.diagonal(_cov)) 
    return np.linalg.cholesky(np.linalg.inv(_cov_diag)).T

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

def get_priors(dataframe,obs,ordering,bname):
    _priors = []
    for o in obs:
        _p = pivot_dataframe(dataframe,o,ordering,bname)
        _priors.append(_p)
    return pd.concat(_priors,axis=1)

def gamma_errfunc(p,x,y,cov):
    #print("In NLO-Errfunc shape of x-values is:")
    #print(x.shape)
    # expect two priors
    if y.shape[0] > x.shape[0]:
        #_res = pik_I32_chipt_nlo(x[:,0],x[:,1],x[:,2],x[:,3],p)-y[:-2]
        _res = p[1]-2*p[0]*x[:,0]-y[:-1]
        #_res = np.r_[_res,p[1]-y[-2],p[2]-y[-1]]
        _res = np.r_[_res,p[1]-y[-1]]
    else: 
      _res = np.r_[p[1]-2*p[0]*x]-y
    # calculate the chi values weighted with inverse covariance matrix
    _chi = np.dot(cov,_res)
    return _chi

def df_cols_to_list(dataframe,names):
    l1 = list(dataframe[names[0]].values)
    l2 = list(dataframe[names[1]].values)
    l3 = list(dataframe[names[2]].values)
    return l1,l2,l3 

def cut_data(dataframe,obs,interval=None):
    """Based on a given observable limit the dataframe to contain the values
    inside the observables range

    Based on the original data of the observable the dataframe gets filtered
    accordingly.
    """
    if interval is not None:
        # get the 0th samples of the dataframe
        cut_decide = dataframe[['beta','L','mu_l',obs]].loc[dataframe['sample']==0]
        # Obtain a list of values for beta,L and mu_l that we like to keep
        choice = cut_decide[['beta','L','mu_l']].loc[(cut_decide[obs]>interval[0]) &
                                                     (cut_decide[obs]<interval[1])]
        # need lists
        ix_b_lst, ix_mu_lst, ix_l_lst = df_cols_to_list(choice,['beta','mu_l','L'])
        # Select the data
        cut = dataframe.loc[dataframe.beta.isin(ix_b_lst) &
                            dataframe.mu_l.isin(ix_mu_lst) & 
                            dataframe.L.isin(ix_l_lst)]
    else:
        cut = dataframe
    return cut

def main():
    pd.set_option('display.width',1000)
    # Get parameters from initfile
    if len(sys.argv) < 2:
        ens = ana.LatticeEnsemble.parse("A40.24.ini")
    else:
        ens = ana.LatticeEnsemble.parse(sys.argv[1])
################################################################################
#                   set up objects                                             #
################################################################################
    # get data from input file
    ms_fixing = sys.argv[2]
    epik_meth = ens.get_data("epik_meth") 
    zp_meth=ens.get_data("zp_meth")
    datadir = ens.get_data("datadir") 
    plotdir = ens.get_data("plotdir") 
    resdir = ens.get_data("resultdir") 
    nboot = ens.get_data("nboot")
    # Load the data from the resultdir
    proc_id = 'pi_K_I32_interpolate_M%d%s'%(zp_meth,ms_fixing.upper()) 
    data_path = resdir+'/'+proc_id+'.h5'
    if ms_fixing.upper() == 'B':
        key = 'Interpolate_sigma_%s'%epik_meth
    else:
        key = 'Interpolate_%s'%epik_meth
    interpolated_data = pd.read_hdf(data_path, key=key)
    interpolated_data.info()
    print(chi.bootstrap_means(interpolated_data,['beta','L','mu_l'],['mu_piK_a32']))
    # A few of the data are squared, we need the unsquared data
    extrapol_df = pd.DataFrame(index=interpolated_data.index,
                               data= interpolated_data[['beta','L','mu_l','sample',
                                                        'fpi','M_pi',
                                                        'mu_piK_a32']])
    extrapol_df['M_K'] = interpolated_data['M_K^2'].pow(1./2)
    extrapol_df['M_eta'] = interpolated_data['M_eta^2'].pow(1./2)
    extrapol_df['M_K/M_pi'] = extrapol_df['M_K']/extrapol_df['M_pi']
    extrapol_df['Gamma'] = ana.gamma_pik(extrapol_df['M_pi'].values,
                                         extrapol_df['M_K'].values,
                                         extrapol_df['mu_piK_a32'].values,
                                         extrapol_df['fpi'].values,
                                         extrapol_df['M_eta'].values)
    groups = ['beta','L','mu_l']
    obs = ['fpi','M_pi','M_K','M_eta','Gamma','M_K/M_pi']
    means = chi.bootstrap_means(extrapol_df,groups,obs)
    chi.print_si_format(means)
    fit_ranges=[[0.,2.5],[0.,1.41],[0.,1.35]]
    for i,fr in enumerate(fit_ranges):
        fit_df = cut_data(extrapol_df,'M_K/M_pi',fr)
        # To fit a function we need x and y data and a covariance matrix
        # get the xdata without any errors
        xdata = fit_df[['beta','L','mu_l','M_K/M_pi']].where(fit_df['sample']==0).dropna()
        xdata = xdata.set_index(['beta','L','mu_l'],drop=True).sort_index()
        # get the priors needed for y vector and  covariance matrix
        l5samples = ana.draw_gauss_distributed(5.41e-3,3e-5,(nboot,),origin=True)
        idx = np.arange(nboot)
        L5 = pd.DataFrame(data=l5samples, index=idx,columns=['L_5']) 
        L5.info()
        # get the ydata
        ydata = get_dataframe_fit(fit_df,'Gamma',['beta','L','mu_l'],
                                  'sample',priors=L5)
        ydata.info()
        print(xdata)
        print(ydata.sample(n=20))
        chi.print_si_format(means)
        data_for_cov = get_dataframe_fit(fit_df,'Gamma',
                                       ['beta','L','mu_l'],'sample',priors=L5)
        data_for_cov.info()
        cov = get_covariance(data_for_cov)
        # Our fit is uncorrelated
        cov = np.diag(np.diagonal(cov))
        chi.print_si_format(means)
        cov_iu = np.linalg.cholesky(np.linalg.inv(cov))
        ## Fitresults dataframe by beta
        col_names = ['sample','chi^2','L_piK','L_5']
        fitres = pd.DataFrame(columns = col_names)
        p = np.array((1.,0.1))
        for b in np.arange(nboot):
            _tmp_fitres = opt.least_squares(gamma_errfunc, p, args=(xdata.values,
                                            ydata.iloc[b].values, cov_iu))
            _chisq = 2*_tmp_fitres.cost
            _tmp_pars = dict(zip(col_names[2:],_tmp_fitres.x))
            _res_dict = {'fr_bgn':fr[0],'fr_end':fr[1],'sample':b,'chi^2':_chisq}
            _res_dict.update(_tmp_pars)
            _tmpdf = pd.DataFrame(data = _res_dict,index=[b])
            fitres = fitres.append(_tmpdf)
            #if b%100 == 0:
            #    print(xdata.values)
            #    print(ydata.iloc[b].values)
            #    print(_res_dict)
        fitres.info()
        fit_df=fit_df.merge(fitres,on='sample')
        chi.print_si_format(chi.bootstrap_means(fit_df,['beta','L','mu_l'],
                            ['M_K/M_pi','Gamma','L_piK','L_5','chi^2']))

        # Store Fit dataframe with parameters and fitrange
        result_id = 'pi_K_I32_gamma_M%d%s'%(zp_meth,ms_fixing.upper())
        hdf_filename = resdir+'/'+result_id+'.h5'
        hdfstorer = pd.HDFStore(hdf_filename)
        hdfstorer.put('nlo_chpt/%s/fr_%d'%(epik_meth,i),fit_df)
        del hdfstorer

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt")

