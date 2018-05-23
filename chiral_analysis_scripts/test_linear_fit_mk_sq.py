#!/usr/bin/python2.7
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
# Fix the strange quark mass in a chiral lQCD analysis. As a fixing parameter
# the Difference
# M_s^2 = r_0^2( M_K^2 - 0.5M_{\pi}^2 ),
# is introduced and the lattice results are interpolated to the corresponding 
# continuum value.
# 
# The input varies with the choice of a value for Z_P the multiplicative
# renormalization of quark masses. This is done via inputfiles which in addition
# state values for necessary random_seeds
# 
# The data produced is stored as a binary object. 
#
# 
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

def mute(cov):
    _cov = np.zeros_like(cov)
    for i in range((_cov.shape[0]-2)/3):
      _cov[3*i:3*i+3,3*i:3*i+3]=cov[3*i:3*i+3,3*i:3*i+3]
    for i in range(_cov.shape[0]-2,_cov.shape[0]):
      _cov[i,i]=cov[i,i]
    return _cov
def amk_sq(p,x):
    # Parameters are equal for each beta, input data is not
    #return p[0]/(p[4]*p[3])*(x[:,0]+x[:,1])*(1+p[1]*p[3]/p[4]*x[:,0]+p[2]/p[3]**2)
    return p[0]*(p[1]+x)
def errfunc_mk(p,x,y,cov_iu):
    # cov_u is the upper triangular matrix of cov
    f_vector = np.r_[amk_sq(p,x)]
    return np.dot(cov_iu,y-f_vector)
def main():
    pd.set_option('display.width',1000)
    beta = float(sys.argv[2])

################################################################################
#                   set up objects                                             #
################################################################################
    # Get parameters from initfile
    if len(sys.argv) < 2:
        ens = ana.LatticeEnsemble.parse("A40.24.ini")
    else:
        ens = ana.LatticeEnsemble.parse(sys.argv[1])

    # get data from input file
    zp_meth=ens.get_data("zp_meth")
    datadir = ens.get_data("datadir") 
    plotdir = ens.get_data("plotdir") 
    resdir = ens.get_data("resultdir") 
    # Load the data from the resultdir
    proc_id = 'piK_I32_unfixed_data_B%d'%(zp_meth) 
    unfixed_data_path = resdir+'/'+proc_id+'.h5' 
    unfixed_data = pd.read_hdf(unfixed_data_path,key=proc_id)
    unfixed_data.info()
    unfixed_A = unfixed_data.where(unfixed_data['beta']==beta).dropna()
    unfixed_A = unfixed_A.where((unfixed_A['mu_l']==0.0040)).dropna()
    xdata_Ab = unfixed_A[unfixed_A.nboot==0]
    xdata_A = xdata_Ab.set_index(['mu_s'],drop=False)['mu_s'].sort_index()
    #xdata_A = xdata_A[xdata_A.index.get_level_values('nboot') == 0]
    A_data = get_dataframe_fit(unfixed_A,'M_K^2_FSE',                       
                            ['mu_s'],'nboot')
    A_data.info()
    data_for_cov = get_dataframe_fit(unfixed_A,'M_K^2_FSE',
                                   ['mu_s'],'nboot')
    data_for_cov.info()
    cov = get_covariance(data_for_cov)
    #cov = np.diag(np.diagonal(cov))
    print(np.dot(np.linalg.inv(cov),cov))
    cov_iu = np.linalg.cholesky(np.linalg.inv(cov))
    print(cov_iu.T)
    # Fitresults dataframe by beta
    col_names = ['beta','nboot','chi^2','P_0','P_1']
    fitres = pd.DataFrame(columns = col_names)
    p = np.array((1.,0.1))
    for b in np.arange(2000):
        _tmp_fitres = opt.least_squares(errfunc_mk,p, args=(xdata_A.values,
                                         A_data.iloc[b], cov_iu.T))
        _chisq = 2*_tmp_fitres.cost
        _tmp_pars = dict(zip(col_names[3:],_tmp_fitres.x))
        _res_dict = {'beta':beta,'nboot':b,'chi^2':_chisq}
        _res_dict.update(_tmp_pars)
        _tmpdf = pd.DataFrame(data = _res_dict,index=[b])
        fitres = fitres.append(_tmpdf)
        if b%100 == 0:
            print(_res_dict)
    fitres.info()
   
    
    # We need a plot of the difference between the data and the fitfunction
    # Calculate difference
    # Errorbar Plot
    #with PdfPages(plotdir+'/D30.48_mksq_fit.pdf') as pdf
    #    plt.xlabel()
    #    pdf.savefig()
    # Can be done by pandas
    print("Fit Result Summary:")
    means = chi.bootstrap_means(fitres,['beta',],['chi^2','P_0','P_1']) 
    chi.print_si_format(means)
    
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

