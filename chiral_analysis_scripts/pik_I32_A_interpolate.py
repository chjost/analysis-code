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
################################################################################
# system imports
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
    space=ens.get_data("beta")
    # for interpolation only 3 lattice spacings are needed
    space = space[:-1]
    latA = ens.get_data("namea")
    latB = ens.get_data("nameb")
    latD = ens.get_data("named")
    strangeA = ens.get_data("strangea")
    strangeB = ens.get_data("strangeb")
    strangeD = ens.get_data("stranged")
    strange_eta_A = ens.get_data("strange_alt_a")
    strange_eta_B = ens.get_data("strange_alt_b")
    strange_eta_D = ens.get_data("strange_alt_d")

    # read seeds from input files
    zp_meth=ens.get_data("zp_meth")
    try:
        epik_meth = ens.get_data("epik_meth")
    except:
        epik_meth=""
    external_seeds = ens.get_data("external_seeds_a")
    continuum_seeds = ens.get_data("continuum_seeds_a")
    amulA = ens.get_data("amu_l_a")
    amulB = ens.get_data("amu_l_b")
    amulD = ens.get_data("amu_l_d")

    #dictionary of strange quark masses
    amusA = ens.get_data("amu_s_a")
    amusB = ens.get_data("amu_s_b")
    amusD = ens.get_data("amu_s_d")
    # dictionaries for chiral analysis
    lat_dict = ana.make_dict(space,[latA,latB,latD])
    amu_l_dict = ana.make_dict(space,[amulA,amulB,amulD])
    mu_s_dict = ana.make_dict(space,[strangeA,strangeB,strangeD])
    mu_s_eta_dict = ana.make_dict(space,[strange_eta_A,strange_eta_B,strange_eta_D])
    amu_s_dict = ana.make_dict(space,[amusA,amusB,amusD])
    print(amu_s_dict)
    #quark = ens.get_data("quark")
    datadir = ens.get_data("datadir") 
    plotdir = ens.get_data("plotdir") 
    resdir = ens.get_data("resultdir") 
    nboot = ens.get_data("nboot")
    # Prepare external data, saving it is at the moment not feasible
    ext_data = ana.ExtDat(external_seeds,space,zp_meth)
    #ana.save_dict(resdir+'/external_observables_A%d.json'%zp_meth,ext_data.data)
    cont_data = ana.ContDat(continuum_seeds,zp_meth=zp_meth)
    #ana.save_dict(resdir+'/continuum_observables_A%d.json'%zp_meth,cont_data.data)
    fpi_raw = ana.read_extern("../plots2/data/fpi.dat",(1,2))
    print(fpi_raw)
    read = False
    print("\nSetup complete, begin chiral analysis")
    if read is True:
        print("Nothing to be done")
    else:
        interpolated_A = pd.DataFrame()
################################################################################
#                   input data                                                 #
################################################################################
        beta_list = [1.90,1.95,2.10]
        for i,a in enumerate(space):
            sample = np.arange(nboot)
            beta = np.full(nboot,beta_list[i])
            print("\nWorking at lattice spacing %s" %a)
            for j,e in enumerate(lat_dict[a]):
                mul = np.full(nboot,amu_l_dict[a][j])
                L = np.full(nboot,int(e.split('.')[-1]))
                # if using D30.48 modify lowest amus to 0.0115
                if e == 'D30.48':
                    print("modifying lowest mu_s values")
                    mu_s_dict[a][0] ='amu_s_115' 
                    amu_s_dict[a][0]=0.0115
                    mu_s_eta_dict[a][0]='strange_1150'

####    ############### read in M_K^FSE ############################################
                mksq_fse = ana.MatchResult("mksq_fse_M%dA_%s"%(zp_meth, e),
                                           save=datadir+'%s/'%e)
                ana.MatchResult.create_empty(mksq_fse,nboot,3)
                mk_names = [datadir+'%s/' % (e) +s+'/fit_k_%s.npz' % (e) for s in mu_s_dict[a]]
                print(mk_names)
                mksq_fse_meas = ana.init_fitreslst(mk_names)
                mksq_fse.load_data(mksq_fse_meas,1,amu_s_dict[a],square=True)
                mksq_fse.add_extern_data('../plots2/data/k_fse_mk.dat',e,square=True,
                                       read='fse_mk',op='mult')
                print("\nM_K^2:")
                print(mksq_fse.obs[:,0])

####    ############### read in M_pi^FSE ###########################################
                mpi_fse = ana.MatchResult("mpi_fse_M%dA_%s"%(zp_meth,e),
                                          save=datadir+'%s/'%e)
                ana.MatchResult.create_empty(mpi_fse,nboot,3)
                mpi_names = [datadir+'%s/' % (e) +'/pi'+'/fit_pi_%s.npz' % (e) for s in mu_s_dict[a]]
                mpi_fse_meas = ana.init_fitreslst(mpi_names)
                mpi_fse.load_data(mpi_fse_meas,1,amu_s_dict[a],square=False)
                mpi_fse.add_extern_data('../plots2/data/k_fse_mpi.dat',e,square=False,
                                       read='fse_mpi',op='div')
                print("\nM_pi:")
                print(mpi_fse.obs[:,0])

####    ############### build M_s^{2,FSE} ##########################################
                mssq_fse = ana.MatchResult("mssq_fse_M%dA_%s" % (zp_meth,e),save = datadir+'%s/'%e)
                ana.MatchResult.create_empty(mssq_fse,nboot,3)
                mssq_fse.set_data(mksq_fse.obs,amu_s_dict[a])
                mssq_fse.add_data(np.square(mpi_fse.obs),idx=slice(0,3),op='min',
                              fac=0.5)
                print("\nM_s^2:")
                print(mssq_fse.obs[:,0])
                
####    ############### read in M_eta ##############################################
                metasq = ana.MatchResult("metasq_M%dA_%s"%(zp_meth,e),save=datadir+'%s/'%e)
                ana.MatchResult.create_empty(metasq,nboot,3)
                meta_names = ['/hiskp4/hiskp2/jost/eta_data/'+'%s/' % (e) +s+'/fit_eta_rm_TP0.npz' for s in mu_s_eta_dict[a]]
                meta_meas = ana.init_fitreslst(meta_names)
                metasq.load_data(meta_meas,1,amu_s_dict[a],square=True)
                print("\nM_eta^2:")
                print(metasq.obs[:,0])

########################  read in mu_pik a_3/2 ################################
                mua32 = ana.MatchResult("mua32_M%dA_%s_%s" %(zp_meth,epik_meth,
                                        e),save=datadir+'%s/'%e)
                ana.MatchResult.create_empty(mua32,nboot,3)
                mua32_names = [datadir+'%s/' % (e) +s+'/mu_a0_TP0_%s_%s.npz' 
                               % (e,epik_meth) for s in mu_s_dict[a]]
                mua32_meas=ana.init_fitreslst(mua32_names)
                mua32.load_data(mua32_meas,0,amu_s_dict[a],square=False)
#####    ############### read in M_pi^FSE ###########################################
                mpi_fse = ana.MatchResult("mpi_fse_M%dB_%s"%(zp_meth,e),save=datadir+'%s/'%e)
                ana.MatchResult.create_empty(mpi_fse,nboot,3)
                mpi_names = [datadir+'%s/' % (e) +'pi'+'/fit_pi_%s.npz' % (e) for s in mu_s_dict[a]]
                mpi_fse_meas = ana.init_fitreslst(mpi_names)
                mpi_fse.load_data(mpi_fse_meas,1,amu_s_dict[a],square=False)
                mpi_fse.add_extern_data('../plots2/data/k_fse_mpi.dat',e,square=False,
                                       read='fse_mpi',op='div')
####    ############################################################################
#                       fix strange quark mass                                     #
####    ############################################################################
                
                evl_x = ana.mk_mpi_diff_phys(a, nboot, cont_data, ext_data)
                print("\n(M^2_k-0.5*M^2_pi):")
                print(ana.compute_error(evl_x))
####    ############### interpolate M_K^FSE ########################################
                mksq_fse.amu = mssq_fse.obs
                print("\nM_K^2:")
                print(mksq_fse.obs[:,0])
                label = [r'$a^2(M_K^2-0.5M^2_{\pi})$',r'$(aM_{K})^2$',
                         r'$a^2(M_K^2-0.5M^2_{\pi}) = (aM_s^{\mathrm{ref}})^2$']
                mksq_fse.eval_at(evl_x,plotdir=plotdir,
                               ens=e,plot=True,label=label, meth=2)

####    ############### interpolate M_eta ##########################################
                metasq.amu = mssq_fse.obs
                label = [r'$a^2(M_K^2-0.5M^2_{\pi})$',r'$(aM_{\eta})^2$',
                         r'$a^2(M_K^2-0.5M^2_{\pi}) = (aM_s^{\mathrm{ref}})^2$']
                metasq.eval_at(evl_x,plotdir=plotdir,
                               ens=e,plot=True,label=label, meth=2,
                               y_lim = [0.065,0.095])

####    ############### interpolate mu_piK a_3/2 ###################################
                mua32.amu = mssq_fse.obs
                label = [r'$a^2(M_K^2-0.5M^2_{\pi})$',r'$\mu_{\pi K}\, a_{0}$',
                         r'$a^2(M_K^2-0.5M^2_{\pi}) = (aM_s^{\mathrm{ref}})^2$']
                mua32.eval_at(evl_x,plotdir=plotdir,
                               ens=e,correlated=False,plot=True,label=label,
                               meth=2,y_lim = [-0.145,-0.09])

####    ############################################################################
#                       copy to pandas dataframe                                   #
####    ############################################################################
                interp_dict = {'beta':beta,
                                'L':L,
                                'mu_l':mul,
                                'mu_s^fix':evl_x,
                                'sample':sample,
                                'M_K^2':mksq_fse.eval_obs[2],
                                'M_eta^2':metasq.eval_obs[2],
                                'mu_piK_a32':mua32.eval_obs[2],
                                'M_pi':mpi_fse.obs[1]
                                }
                tmp_df = pd.DataFrame(data=interp_dict)
                interpolated_A = interpolated_A.append(tmp_df)
        groups = ['beta','L','mu_l']
        #observables=['M_K^2_FSE','P_0','P_1','P_2','P_r','P_Z','P_mu','M_K^2_func']
        obs = ['M_K^2','mu_piK_a32','M_eta^2','mu_s^fix']
        print(chi.bootstrap_means(interpolated_A,groups,obs))
        proc_id = 'pi_K_I32_interpolate_M%dA'%(zp_meth)
        hdf_savename = resdir+proc_id+'.h5'
        hdfstorer = pd.HDFStore(hdf_savename)
        hdfstorer.put('Interpolate_%s'%epik_meth,interpolated_A)
        del hdfstorer
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard Interrupt")


