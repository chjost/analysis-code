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
# $\mu_{\pi K}$
# 
# Load a dataframe from a fixing fit (hdf5 pandas dataframe)
# Calculate the bare strange quark mass
# interpolate in all ensembles and observables to that strange quark mass
# save the result as a new pandas dataframe
################################################################################

# system imports
import sys
from scipy import stats
from scipy import interpolate as ip
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

def amu_s_ref(df,cont_data):
    """calculate the reference bare strange mass 
    The formula is given by 
       a\mu_s^{\text{ref}} =
       \frac{(r_0M_K^{\text{phys}})^2}{P_{\mu}(\mu_\sigma)P_0P_r\left[1+P_1r_0m_l^{\text{phys}}+P_2P_r^{-2}\right]}
                             -\frac{P_Z}{P_{\mu}(\mu_\sigma)P_r}r_0m_l^{\text{phys}}\,.

    """
    p0 = df['P_0'].values
    p1 = df['P_1'].values
    p2 = df['P_2'].values
    pr = df['P_r'].values
    pz = df['P_Z'].values
    pmu = df['P_mu'].values
    mul = df['mu_l'].values
    # need to reshape the continuum data
    r0 = cont_data.get('r0')
    reps = p0.shape[0]/r0.shape[0]
    r0 = np.tile(r0,reps)
    mk = np.tile(cont_data.get('mk'),reps)
    ml = np.tile(cont_data.get('m_l'),reps)
    hbarc = 197.37
    amus_ref_func = pz*(r0*mk/hbarc)**2/(pmu*p0*pr*(1+p1*r0*ml/hbarc+p2/pr**2))-pz/(pmu*pr)*r0*ml/hbarc
    amus_ref = pd.Series(amus_ref_func,index=df.index)
    return amus_ref

def main():
    # need the initializations of fixms again

################################################################################
#                   set up objects                                             #
################################################################################
    # Get parameters from initfile
    if len(sys.argv) < 2:
        ens = ana.LatticeEnsemble.parse("A40.24.ini")
    else:
        ens = ana.LatticeEnsemble.parse(sys.argv[1])

    # get data from input file
    print(ens)
    lat = ens.name()
    space=ens.get_data("beta")
    latA = ens.get_data("namea")
    latB = ens.get_data("nameb")
    latD = ens.get_data("named")
    latD45 = ens.get_data("named45")
    strangeA = ens.get_data("strangea")
    strangeB = ens.get_data("strangeb")
    strangeD = ens.get_data("stranged")
    strangeD45 = ens.get_data("stranged45")
    strange_eta_A = ens.get_data("strange_alt_a")
    strange_eta_B = ens.get_data("strange_alt_b")
    strange_eta_D = ens.get_data("strange_alt_d")
    strange_eta_D45 = ens.get_data("strange_alt_d45")
    zp_meth=ens.get_data("zp_meth")
    try:
        epik_meth = ens.get_data("epik_meth")
    except:
        epik_meth=""
    external_seeds=ens.get_data("external_seeds_b")
    continuum_seeds=ens.get_data("continuum_seeds_b")
    amulA = ens.get_data("amu_l_a")
    amulB = ens.get_data("amu_l_b")
    amulD = ens.get_data("amu_l_d")
    amulD45 = ens.get_data("amu_l_d45")
    #dictionary of strange quark masses
    amusA = ens.get_data("amu_s_a")
    amusB = ens.get_data("amu_s_b")
    amusD = ens.get_data("amu_s_d")
    amusD45 = ens.get_data("amu_s_d45")
    # dictionaries for chiral analysis
    #lat_dict = ana.make_dict(space,[latA,latB,latD])
    lat_dict = ana.make_dict(space,[latA,latB,latD,latD45])
    #lat_dict = {'A':latA,'B':latB,'D':latD}
    #amu_l_dict = ana.make_dict(space,[amulA,amulB,amulD])
    amu_l_dict = ana.make_dict(space,[amulA,amulB,amulD,amulD45])
    #amu_l_dict = {'A': amulA,'B': amulB, 'D': amulD}
    mu_s_dict = ana.make_dict(space,[strangeA,strangeB,strangeD,strangeD45])
    #mu_s_dict = {'A': strangeA,'B': strangeB, 'D': strangeD}
    mu_s_eta_dict = ana.make_dict(space,[strange_eta_A,strange_eta_B,strange_eta_D,strange_eta_D45])
    #mu_s_eta_dict = {'A': strange_eta_A,'B': strange_eta_B, 'D': strange_eta_D}
    amu_s_dict = ana.make_dict(space,[amusA,amusB,amusD,amusD45])
    #amu_s_dict = {'A': amusA,'B': amusB, 'D': amusD}
    print(amu_s_dict)
    #quark = ens.get_data("quark")
    datadir = ens.get_data("datadir") 
    plotdir = ens.get_data("plotdir") 
    resdir = ens.get_data("resultdir") 
    nboot = ens.get_data("nboot")
    # Prepare external data
    ext_data = ana.ExtDat(external_seeds,space,zp_meth,nboot=2000)
    cont_data = ana.ContDat(continuum_seeds,zp_meth=zp_meth,nboot=2000)

    # Read in the fixed Data from a saved hdf5 file
    # need the filename and the key of the dataset for the parameters
    # Have to match filename and key from fix_ms_B script
    hdf_readname = resdir+'pi_K_I32_fixms_M%dB'%zp_meth+'.h5'
    fixms_B_result = pd.read_hdf(hdf_readname,key='Fitresults_sigma')
    interp_cols = ['beta','mu_l','mu_s','sample']
    fixms_B_result['amu_s_ref'] = amu_s_ref(fixms_B_result,cont_data)
    data_to_interpolate = pd.DataFrame()
    # copy over needed data from fixms result (can take M_K^2_FSE directly)
    for p in interp_cols:
        data_to_interpolate[p] = fixms_B_result[p]
    # build a temporary dataframe to hold the observables to be interpolated
    # generate new dataframe that gets filled subsequently
    # Set the columnnames 
    interp_cols = ['beta','mu_l','sample']
    interpolated_B = pd.DataFrame()
    #Interpolation in terms of MatchResults
    beta_list = [1.90,1.95,2.10,2.10]
    for i,a in enumerate(space):
        sample = np.arange(2000)
        beta = np.full(2000,beta_list[i])
        print("\nWorking at lattice spacing %s" %a)
        for j,e in enumerate(lat_dict[a]):
            mul = np.full(2000,amu_l_dict[a][j])
            # if using D30.48 modify lowest amus to 0.0115
            if e == 'D30.48':
                print("modifying lowest mu_s values")
                mu_s_dict[a][0] ='amu_s_115' 
                amu_s_dict[a][0]=0.0115
                mu_s_eta_dict[a][0]='strange_1150'

################### read in M_K^FSE ############################################
            mksq_fse = ana.MatchResult("mksq_fse_M%dB_%s"%(zp_meth,e),save=datadir+'%s/'%e)
            ana.MatchResult.create_empty(mksq_fse,2000,3)
            mk_names = [datadir+'%s/' % (e) +s+'/fit_k_%s.npz' % (e) for s in mu_s_dict[a]]
            mksq_fse_meas = ana.init_fitreslst(mk_names)
            mksq_fse.load_data(mksq_fse_meas,1,amu_s_dict[a],square=True)
            mksq_fse.add_extern_data('../plots2/data/k_fse_mk.dat',e,square=True,
                                   read='fse_mk',op='mult')

################### read in M_pi^FSE ###########################################
            mpi_fse = ana.MatchResult("mpi_fse_M%dB_%s"%(zp_meth,e),save=datadir+'%s/'%e)
            ana.MatchResult.create_empty(mpi_fse,2000,3)
            mpi_names = [datadir+'%s/' % (e) +'pi'+'/fit_pi_%s.npz' % (e) for s in mu_s_dict[a]]
            mpi_fse_meas = ana.init_fitreslst(mpi_names)
            mpi_fse.load_data(mpi_fse_meas,1,amu_s_dict[a],square=False)
            mpi_fse.add_extern_data('../plots2/data/k_fse_mpi.dat',e,square=False,
                                   read='fse_mpi',op='div')

#################### read in M_eta ##############################################
            metasq = ana.MatchResult("metasq_M%dB_%s"%(zp_meth,e),save=datadir+'%s/'%e)
            ana.MatchResult.create_empty(metasq,2000,3)
            meta_names = ['/hiskp4/hiskp2/jost/eta_data/'+'%s/' % (e) +s+'/fit_eta_rm_TP0.npz' for s in mu_s_eta_dict[a]]
            meta_meas = ana.init_fitreslst(meta_names)
            metasq.load_data(meta_meas,1,amu_s_dict[a],square=True)

########################  read in mu_pik a_3/2 ################################
            mua32 = ana.MatchResult("mua32_M%dB_%s_%s" %(zp_meth,epik_meth,
                                    e),save=datadir+'%s/'%e)
            ana.MatchResult.create_empty(mua32,2000,3)
            mua32_names = [datadir+'%s/' % (e) +s+'/mu_a0_TP0_%s_%s.npz' 
                           % (e,epik_meth) for s in mu_s_dict[a]]
            mua32_meas=ana.init_fitreslst(mua32_names)
            mua32.load_data(mua32_meas,0,amu_s_dict[a],square=False)
################################################################################
#                   fix strange quark mass                                     #
################################################################################
            evl_x = fixms_B_result['amu_s_ref'].where((fixms_B_result['beta']==beta_list[i]) &
                    (fixms_B_result['mu_l']==amu_l_dict[a][j]) &
                    (fixms_B_result['mu_s']==amu_s_dict[a][0])).dropna().values 
################### interpolate M_K^FSE ########################################
            #mksq_fse.amu = mssq_fse.obs
            label = [r'$a\mu_s$',r'$(aM_{K})^2$',
                     r'$a\mu_s = (a\mu_s^{\mathrm{ref}})^2$']
            mksq_fse.eval_at(evl_x,plotdir=plotdir,
                           ens=e,plot=True,label=label, meth=2)

################### interpolate M_eta ##########################################
            #metasq.amu = mssq_fse.obs
            label = [r'$a\mu_s$',r'$(aM_{\eta})^2$',
                     r'$a\mu_s = (a\mu_s^{\mathrm{ref}})^2$']
            metasq.eval_at(evl_x,plotdir=plotdir,
                           ens=e,plot=True,label=label, meth=2,
                           y_lim = [0.065,0.095])

################### interpolate mu_piK a_3/2 ###################################
            #mua32_fse.amu = mssq_fse.obs
            label = [r'$a\mu_s$',r'$\mu_{\pi K}\, a_{0}$',
                     r'$a\mu_s = (a\mu_s^{\mathrm{ref}})^2$']
            mua32.eval_at(evl_x,plotdir=plotdir,
                           ens=e,correlated=False,plot=True,label=label,
                           meth=2,y_lim = [-0.145,-0.09])
            interp_dict = {'beta':beta,
                            'mu_l':mul,
                            'mu_s^fix':evl_x,
                            'sample':sample,
                            'M_K^2':mksq_fse.eval_obs[2],
                            #'M_eta':metasq.eval_obs[2],
                            'mu_piK_a32':mua32.eval_obs[2]
                            'M_pi':mpifse
                            }
            tmp_df = pd.DataFrame(data=interp_dict)
            interpolated_B = interpolated_B.append(tmp_df)
    interpolated_B.info()
    print(interpolated_B.sample(n=10))
    groups = ['beta','mu_l']
    obs = ['M_K^2','mu_piK_a32','mu_s^fix']
    print(chi.bootstrap_means(interpolated_B,groups,obs))
    #proc_id = 'pi_K_I32_interpolate_M%dB'%(zp_meth)
    #hdf_savename = resdir+proc_id+'.h5'
    #hdfstorer = pd.HDFStore(hdf_savename)
    #hdfstorer.put('Interpolate_sigma_%s'%epik_meth,interpolate_B)
    #del hdfstorer

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard Interrupt")


