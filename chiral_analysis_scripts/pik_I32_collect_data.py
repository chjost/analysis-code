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

def mua32_phys(df,cont):
    """Calculate the continuum value of mu_piK * a_32
    """
    meta = cont.get('meta')
    mpi = cont.get('mpi_0')
    mk = cont.get('mk')
    fpi = cont.get('fpi')
    #TODO: let that look nicer
    p = df[['L_piK','L_5']].head(n=meta.shape[0]).values.T
    return ana.pik_I32_chipt_nlo_cont(mpi,mk,fpi,p,meta=meta)

def mua12_phys(df,cont):
    """Calculate the continuum value of mu_piK * a_32
    """
    meta = cont.get('meta')
    mpi = cont.get('mpi_0')
    mk = cont.get('mk')
    fpi = cont.get('fpi')
    #TODO: let that look nicer
    p = df[['L_piK','L_5']].head(n=meta.shape[0]).values.T
    lpik=p[0]
    l5=p[1]
    return ana.mu_aI12(fpi,mpi,mk,fpi,l5,lpik)

def tau_pik(df,cont):
    """
    Calculate pik atom lifetime input parameters taken from arxiv:1707.02184
    """
    mpi = cont.get('mpi_0')
    mk = cont.get('mk')
    mpia12 = df['M_pi_a12_phys'].values
    mpia32 = df['M_pi_a32_phys'].values
    # hbar in Mev*s
    hbar = 6.58211899e-22
    hbarc = 197.33
    # bootstraps:
    # \alpha
    delta_K = ana.draw_gauss_distributed(0.04,0.022,mk.shape,origin=True)
    alpha = np.full_like(delta_K,7.29735254e-3)
    p_star = np.full_like(delta_K,11.8)
    # reduced mass of piK atom
    mu_pik  = mpi*mk/(mpi+mk) 
    # isospin odd scattering length
    # a_neg only includes L_5 better to take difference involving L_piK?
    #_a_neg = a_neg*hbarc
    _a_neg = (mpia12-mpia32)*hbarc/(3.*mpi)
    _tau = hbar*hbarc**2/(8*alpha**3*_a_neg**2*mu_pik**2*p_star*(1+delta_K))
    return _tau

def main():
    pd.set_option('display.width',1000)
    # keys for the hdf datasets
    chpt = ['nlo_chpt','gamma']
    epik_meth = ['E1','E2']
    #zp_meth = [1, 2]
    #ms_fixing = ['A', 'B']
    fr_labels = [0,1,2]
    # construct filenames
    file_prefix='pi_K_I32'
    resultdir = '/hiskp4/helmes/analysis/scattering/pi_k/I_32_cov_false/results/'
    # for physical calculations get dictionary of continuum bootstrapsamples
    # seeds for M1A,M1B,M2A and M2B
    ini_path = '/hiskp4/helmes/projects/analysis-code/ini/pi_K/I_32_cov_false'
    ini1 = ini_path+'/'+'chiral_analysis_mua0.ini'
    #ini1 = ini_path+'/'+'chiral_analysis_mua0_zp1.ini'
    #ini2 = ini_path+'/'+'chiral_analysis_mua0_zp2.ini'
    ens1 = ana.LatticeEnsemble.parse(ini1)
    nboot = ens1.get_data('nboot')
    #cont={'M1A':ana.ContDat(ens1.get_data('continuum_seeds_a'),zp_meth=1),
    #      'M2A':ana.ContDat(ens2.get_data('continuum_seeds_a'),zp_meth=2),
    #      'M1B':ana.ContDat(ens1.get_data('continuum_seeds_b'),zp_meth=1),
    #      'M2B':ana.ContDat(ens2.get_data('continuum_seeds_b'),zp_meth=2)}
    cont={'M1A':ana.ContDat(ens1.get_data('continuum_seeds_a'),zp_meth=1)}
    # construct
    df_collect = pd.DataFrame()
    final_results = pd.DataFrame()
    #for tp in it.product(chpt,epik_meth,zp_meth,ms_fixing,fr_labels):
    for tp in it.product(chpt,epik_meth,fr_labels):
        filename = '%s_%s_M1A.h5'%(file_prefix,tp[0])
        #keyname = '%s/%s/fr_%d'%(tp[0],tp[1],tp[4])
        #keyname = 'interp_corr_false/%s/%s/fr_%d'%(tp[0],tp[1],tp[4])
        #keyname = 'fse_false/%s/%s/fr_%d'%(tp[0],tp[1],tp[4])
        keyname = 'fse_true/%s/%s/fr_%d'%(tp[0],tp[1],tp[2])
        print(filename,keyname)
        branch_result = pd.read_hdf(resultdir+filename,key=keyname)
        branch_result.info()
        # extend dataframe for description
        branch_result['ChPT'] = tp[0]
        branch_result['poll'] = tp[1]
        #branch_result['RC'] = tp[2]
        #branch_result['ms_fix'] = tp[3]
        groups=['beta','L','mu_l']
        obs = ['mu_piK_a32_scaled','L_piK']
        print(chi.print_si_format(chi.bootstrap_means(branch_result,groups,obs)))
        df_collect = pd.concat((df_collect,branch_result))
    # append what we want to calculate to an endresults dataframe
    # We would like to know L_5, L_piK, mu_pik_a32_phys, mu_pik_a12_phys,
    # mpi_a32, mpi_a12, tau_pik
    # To calculate anything we need 
        tmp_fin_res = pd.DataFrame()
        tmp_res_index = branch_result['sample'].unique()
        tmp_fin_res['sample'] = pd.Series(data=branch_result['sample'].unique(),
                                          index = tmp_res_index)
        tmp_fin_res['L_5'] = pd.Series(data = branch_result['L_5'].unique(),
                                       index = tmp_res_index)
        tmp_fin_res['L_piK'] = pd.Series(data = branch_result['L_piK'].unique(),
                                         index = tmp_res_index)
        tmp_fin_res['fr_bgn'] = branch_result['fr_bgn'].unique()[0]
        tmp_fin_res['fr_end'] = branch_result['fr_end'].unique()[0]
        tmp_fin_res['ChPT'] = tp[0]
        tmp_fin_res['poll'] = tp[1]
        #tmp_fin_res['RC'] = tp[2]
        #tmp_fin_res['ms_fix'] = tp[3]
        #tmp_cont=cont['M%d%s'%(tp[2],tp[3])]
        tmp_cont=cont['M1A']
        tmp_fin_res['chi^2'] = branch_result.loc[0:nboot,'chi^2']
        tmp_fin_res['dof'] = branch_result.loc[0:nboot,'dof']
        tmp_fin_res['p-val'] = branch_result.loc[0:nboot,'p-val']
        tmp_fin_res['mu_piK_a32_phys'] = mua32_phys(tmp_fin_res,tmp_cont)
        tmp_fin_res['mu_piK_a12_phys'] = mua12_phys(tmp_fin_res,tmp_cont)
        tmp_fin_res['M_pi_a32_phys'] = tmp_fin_res['mu_piK_a32_phys'] * (tmp_cont.get('mk')+tmp_cont.get('mpi_0')) / tmp_cont.get('mk')
        tmp_fin_res['M_pi_a12_phys'] = tmp_fin_res['mu_piK_a12_phys'] * (tmp_cont.get('mk')+tmp_cont.get('mpi_0')) / tmp_cont.get('mk')
        tmp_fin_res['tau_piK'] = tau_pik(tmp_fin_res,tmp_cont)
        final_results = pd.concat((final_results,tmp_fin_res))
                                                     

    df_collect.info()
    final_results.info()
    print(final_results.sample(n=20))

    result_id = 'pi_K_I32_overview'
    hdf_filename = resultdir+'/'+result_id+'.h5'
    storer = pd.HDFStore(hdf_filename)
    #keyname = 'data_collection'
    #keyname = 'interp_corr_false/data_collection'
    #keyname = 'fse_false/data_collection'
    keyname = 'fse_true/data_collection'
    storer.put(keyname,df_collect)
    #keyname = 'physical_results'
    #keyname = 'interp_corr_false/physical_results'
    #keyname = 'fse_false/physical_results'
    keyname = 'fse_true/physical_results'
    storer.put(keyname,final_results)
    del storer


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
