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
# $\mu_{\pi K}/f_\pi$ using NLO-ChPT.
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

def amk_sq(p,x):
    # Parameters are equal for each beta, input data is not
    #return p[0]/(p[4]*p[3])*(x[:,0]+x[:,1])*(1+p[1]*p[3]/p[4]*x[:,0]+p[2]/p[3]**2)
    return p[0]*(p[1]+x)

def errfunc_mk(p,x,y,cov_iu):
    # cov_u is the upper triangular matrix of cov
    f_vector = np.r_[amk_sq(p,x)]
    return np.dot(cov_iu,y-f_vector)
def mu_a32_errfunc(p,x,y,cov):
    #print("In NLO-Errfunc shape of x-values is:")
    #print(x.shape)
    # expect two priors
    if y.shape[0] > x.shape[0]:
        #_res = pik_I32_chipt_nlo(x[:,0],x[:,1],x[:,2],x[:,3],p)-y[:-2]
        _res = ana.pik_I32_chipt_nlo(x[:,0],x[:,1],x[:,2],x[:,3],p,meta=x[:,4])-y[:-1]
        #_res = np.r_[_res,p[1]-y[-2],p[2]-y[-1]]
        _res = np.r_[_res,p[1]-y[-1]]
    else: 
      _res = ana.pik_I32_chipt_nlo(x[:,0],x[:,1],x[:,2],x[:,3],p,meta=x[:,4])-y
    # calculate the chi values weighted with inverse covariance matrix
    _chi = np.dot(cov,_res)
    return _chi
def main():
    pd.set_option('display.width',1000)



################################################################################
#                   set up objects                                             #
################################################################################
    # Get parameters from initfile
    if len(sys.argv) < 2:
        ens = ana.LatticeEnsemble.parse("A40.24.ini")
    else:
        ens = ana.LatticeEnsemble.parse(sys.argv[1])

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
    interpolated_data = pd.read_hdf(data_path, key='Interpolate_sigma_%s'%epik_meth)
    interpolated_data.info()
    print(chi.bootstrap_means(interpolated_data,['beta','L','mu_l'],['mu_piK_a32']))
    # To fit a function we need x and y data and a covariance matrix
    # get the xdata without any errors
    xdata = interpolated_data[['M_pi','M_K^2','M_eta^2','mu_piK_a32']].where(interpolated_data['sample']==0).dropna()
    print(xdata)
    # get the ydata
    ydata = get_dataframe_fit(interpolated_data,'mu_piK_a32',['beta','L','mu_l'],'sample')
    ydata.info()
    # get the priors and a covariance matrix
    l5samples = ana.draw_gauss_distributed(5.41e-3,3e-5,(nboot,),origin=True)
    idx = np.arange(nboot)
    L5 = pd.DataFrame(data=l5samples, index=idx,columns=['L_5']) 
    L5.info()
    data_for_cov = get_dataframe_fit(interpolated_data,'mu_piK_a32',
                                   ['beta','L','mu_l'],'sample',priors=L5)
    data_for_cov.info()
    cov = get_covariance(data_for_cov)
    # Our fit is uncorrelated
    cov = np.diag(np.diagonal(cov))
    #print(np.dot(np.linalg.inv(cov),cov))
    cov_iu = np.linalg.cholesky(np.linalg.inv(cov))
    print(cov_iu.T)
    ## Fitresults dataframe by beta
    col_names = ['nboot','chi^2','L_piK','L_5']
    fitres = pd.DataFrame(columns = col_names)
    p = np.array((1.,0.1))
    for b in np.arange(nboot):
        _tmp_fitres = opt.least_squares(mu_a32_errfunc, p, args=(xdata.values,
                                        ydata.iloc[b], cov_iu.T))
        _chisq = 2*_tmp_fitres.cost
        _tmp_pars = dict(zip(col_names[3:],_tmp_fitres.x))
        _res_dict = {'beta':beta,'nboot':b,'chi^2':_chisq}
        _res_dict.update(_tmp_pars)
        _tmpdf = pd.DataFrame(data = _res_dict,index=[b])
        fitres = fitres.append(_tmpdf)
        if b%100 == 0:
            print(_res_dict)
    fitres.info()
   
    #
    ## We need a plot of the difference between the data and the fitfunction
    ## Calculate difference
    ## Errorbar Plot
    ##with PdfPages(plotdir+'/D30.48_mksq_fit.pdf') as pdf
    ##    plt.xlabel()
    ##    pdf.savefig()
    ## Can be done by pandas
    #print("Fit Result Summary:")
    #means = chi.bootstrap_means(fitres,['beta',],['chi^2','P_0','P_1']) 
    #chi.print_si_format(means)
    
    #leastsquares takes callable function and arguments plus start estimate
    # One lattice spacing
    #hdfstorer = pd.HDFStore(unfixed_data_path)
    #hdfstorer['raw_data'] = unfixed_data
    #hdfstorer['covariancematrix'] = cov_matrix
    #hdfstorer['fitresults']
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
