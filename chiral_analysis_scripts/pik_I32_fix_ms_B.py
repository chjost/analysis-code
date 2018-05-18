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
sys.path.append('/hiskp4/helmes/projects/analysis-code/')
import analysis2 as ana
import chiron as chi

#def mk_global():
#
#def errfunc_mk_global():
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
#def get_data_vector_beta(dataframe,ordering,beta,priors=None):
#def get_function_vector_beta(dataframe,ordering,beta):
#def get_chi_difference(dataframe,):
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

def mute(cov):
    _cov = np.zeros_like(cov)
    for i in range((_cov.shape[0]-2)/3):
      _cov[3*i:3*i+3,3*i:3*i+3]=cov[3*i:3*i+3,3*i:3*i+3]
    for i in range(_cov.shape[0]-2,_cov.shape[0]):
      _cov[i,i]=cov[i,i]
    return _cov
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
    datadir = ens.get_data("datadir") 
    resdir = ens.get_data("resultdir") 
    # Load the data from the resultdir
    proc_id = 'piK_I32_unfixed_data_B%d'%(zp_meth) 
    unfixed_data_path = resdir+'/'+proc_id+'.h5' 
    unfixed_data = pd.read_hdf(unfixed_data_path,key=proc_id)
    unfixed_data.info()
    unfixed_A = unfixed_data.where(unfixed_data['beta']==1.90).dropna()
    unfixed_A.info()
    xdata_Ab = unfixed_A[unfixed_A.nboot==0]
    xdata_A = xdata_Ab.set_index(['L','mu_l','mu_s'],drop=False)[['mu_l','mu_s']].sort_index()
    #xdata_A = xdata_A[xdata_A.index.get_level_values('nboot') == 0]
    A_priors = get_priors(unfixed_A,['r_0','Z_P'],['beta'],'nboot')
    A_data = get_dataframe_fit(unfixed_A,'M_K^2_FSE',                       
                            ['L','mu_l','mu_s'],'nboot',priors=A_priors)
    A_data.info()
    data_for_cov = get_dataframe_fit(unfixed_A,'M_K^2_FSE',
                                   ['beta','L','mu_l','mu_s'],'nboot',priors=A_priors)
    data_for_cov.info()
    cov = get_covariance(data_for_cov)
    cov = mute(cov)
    cov_iu = np.linalg.cholesky(np.linalg.inv(cov))
    print(cov_iu)
    print(A_data.iloc[0])
    print(xdata_A)
    def amk_sq(p,x):
        # Parameters are equal for each beta, input data is not
        return p[0]/(p[4]*p[3])*(x[:,0]+x[:,1])*(1+p[1]/p[3]*x[:,0]+p[2]/p[3]**2)
    def errfunc_ms(p,x,y,cov_iu):
        # cov_u is the upper triangular matrix of cov
        f_vector = np.r_[amk_sq(p,x),p[3:5]]
        return np.dot(cov_iu,y-f_vector)
    p = np.array((5.558,0.143,4.845,5.193,0.524))
    chi_vec_A = (errfunc_ms(p,xdata_A.values,A_data.iloc[0],cov_iu))
    print(np.square(chi_vec_A))
    print(np.nansum(np.square(chi_vec_A)))

    # do a fit to the original data
    amk_sq_fitresult = opt.least_squares(errfunc_ms,p, args=(xdata_A.values,
                                         A_data.iloc[0], cov_iu))
    print(amk_sq_fitresult.x)
    # leastsquares only gives half of the answer
    print(2*amk_sq_fitresult.cost)
    print(amk_sq_fitresult.fun)
    print(np.sum(np.square(amk_sq_fitresult.fun)))
    print(amk_sq_fitresult.status,amk_sq_fitresult.message)

    
    #print(unfixed_data.sample(n=20))
    # Fits take place per bootstrapsample need an errorfunction and a function
    # Get beta dependent priors r0 and Pz as dataframe
    #r0 = pivot_dataframe(unfixed_data,'r_0',['beta'],'nboot')
    #r0.info()
    #zp = pivot_dataframe(unfixed_data,'Z_P',['beta'],'nboot')
    #zp.info()
    #priors = pd.concat([r0,zp],axis=1)
    #priors.info()
    ## We also need a covariance matrix that gets inverted
    #data_for_cov = get_dataframe_fit(unfixed_data,'M_K^2_FSE',
    #                               ['beta','L','mu_l','mu_s'],'nboot',priors=priors)
    #data_for_cov.info()
    #cov = get_covariance(data_for_cov)
    #get_measurements(unfixed_data,'M_K^2_FSE',['beta','L','mu_l','mu_s'],'nboot',beta)
    #print(data_for_cov.sample(n=20))
    #obs_of_interest = ['M_K^2_FSE']
    #result_means = chi.bootstrap_means(unfixed_data,['beta','L','mu_l','mu_s'],obs_of_interest)
    #chi.print_si_format(result_means)
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

