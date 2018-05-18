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
    zp_meth=ens.get_data("zp_meth")
    datadir = ens.get_data("datadir") 
    plotdir = ens.get_data("plotdir") 
    resdir = ens.get_data("resultdir") 
    # Load theata from the resultdir
    proc_id = 'piK_I32_unfixed_data_B%d'%(zp_meth) 
    unfixed_data_path = resdir+'/'+proc_id+'.h5' 
    unfixed_data = pd.read_hdf(unfixed_data_path,key=proc_id)
    unfixed_data.info()
    unfixed_A = unfixed_data.where(unfixed_data['beta']==1.90).dropna()
    unfixed_A.info()
    A_priors = get_priors(unfixed_A,['r_0','Z_P'],['beta'],'nboot')
    A_data = get_dataframe_fit(unfixed_A,'M_K^2_FSE',                       
                            ['L','mu_l','mu_s'],'nboot',priors=A_priors)
    data_for_cov = get_dataframe_fit(unfixed_A,'M_K^2_FSE',
                                   ['beta','L','mu_l','mu_s'],'nboot',priors=A_priors)
    xdata_Ab = unfixed_A[unfixed_A.nboot==0]
    xdata_A = xdata_Ab.set_index(['L','mu_l','mu_s'],drop=False)[['mu_l','mu_s']].sort_index()
    print(xdata_A.values)
    data_for_cov.info()
    cov = get_covariance(data_for_cov)
    cov = mute(cov)
    cov_iu = np.cholesky(np.linalg.inv(cov))
    corrcoef = np.corrcoef(data_for_cov.values.T)
    # plot correlation coefficients in heatmap
    tickmarks = np.r_[xdata_A.values,np.asarray([[0,3],[0,4]])]
    with PdfPages(plotdir+'/correlation_fixms_beta19.pdf') as pdf:
        if corrcoef.shape[0] != corrcoef.shape[1]:
          raise ValueError("data not symmetric")
        plt.figure(figsize=(13,12))
        plt.xticks(np.arange(0.5,corrcoef.shape[0]+0.5),tickmarks,rotation=90)
        plt.yticks(np.arange(0.5,corrcoef.shape[0]+0.5),tickmarks)
        plt.pcolor(np.corrcoef(corrcoef), cmap=matplotlib.cm.bwr,
                vmin=np.amin(corrcoef), vmax=np.amax(corrcoef))
        plt.colorbar()
        pdf.savefig()
        plt.close()
    # For every ensemble we want the corresponding matrix of chi^2 contributions
    # as a stacked 3d plot
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# setup the figure and axes
fig = plt.figure(figsize=(8, 3))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

# fake data
_x = np.arange(4)
_y = np.arange(5)
_xx, _yy = np.meshgrid(_x, _y)
x, y = _xx.ravel(), _yy.ravel()

top = x + y
bottom = np.zeros_like(top)
width = depth = 1

ax1.bar3d(x, y, bottom, width, depth, top, shade=True)
ax1.set_title('Shaded')

ax2.bar3d(x, y, bottom, width, depth, top, shade=False)
ax2.set_title('Not Shaded')

plt.show()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt")

