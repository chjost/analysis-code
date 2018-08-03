#!/usr/bin/python
################################################################################
#
# Author: Christopher Helmes (helmes@hiskp.uni-bonn.de)
# Date:   June 2018
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
# Collect the uninterpolated data from all analysis branches in one dataframe.
# The observables are already bootstrapped and finte size corrected before any
# interpolation takes place. We store everything in a huge dataframe that gets
# converted to an R Dataframe afterwards.
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
def get_sys_error(frlst,q='low'):
    quant = []
    ud=0
    if q=='high':
        ud=1
    for r in frlst:
        r.calc_error()
        quant.append(r.error[0][2][0][ud])
    return quant
def matchresult_to_df(match_obs,mu_s,samples,obs_name):
    """Convert data from a MatchResult to a dataframe for merging
    """
    df = pd.DataFrame()
    #match observables are stored as a list
    for om in zip(match_obs,mu_s):
        tmp_df = pd.DataFrame()
        _mu_s = np.full(samples.shape,om[1])
        tmp_df['sample'] = samples
        tmp_df[obs_name] = om[0]
        tmp_df['mu_s']= _mu_s
        df=pd.concat((df,tmp_df))
    return df

def qlst_to_df(qlst,mu_s,samples,obs_name):
    """Convert list of quantiles to a dataframe for merging
    """
    df = pd.DataFrame()
    #match observables are stored as a list
    for om in zip(qlst,mu_s):
        tmp_df = pd.DataFrame()
        _mu_s = np.full(samples.shape,om[1])
        tmp_df['sample'] = samples
        tmp_df[obs_name] = om[0]
        tmp_df['mu_s']= _mu_s
        df=pd.concat((df,tmp_df))
    return df
def main():
    pd.set_option('display.width',1000)
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
    # can add lists to get new lists
    latD = ens.get_data("named45")+ens.get_data("named")
    strangeA = ens.get_data("strangea")
    strangeB = ens.get_data("strangeb")
    strangeD = ens.get_data("stranged45")
    strange_eta_A = ens.get_data("strange_alt_a")
    strange_eta_B = ens.get_data("strange_alt_b")
    strange_eta_D = ens.get_data("strange_alt_d45")

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
    # Arrays get added elementwise
    amulD = np.r_[ens.get_data("amu_l_d45"),ens.get_data("amu_l_d")]
    #dictionary of strange quark masses
    amusA = ens.get_data("amu_s_a")
    amusB = ens.get_data("amu_s_b")
    amusD = ens.get_data("amu_s_d45")
    # dictionaries for chiral analysis
    lat_dict = ana.make_dict(space,[latA,latB,latD])
    amu_l_dict = ana.make_dict(space,[amulA,amulB,amulD])
    print(amu_l_dict)
    mu_s_dict = ana.make_dict(space,[strangeA,strangeB,strangeD])
    mu_s_eta_dict = ana.make_dict(space,[strange_eta_A,strange_eta_B,strange_eta_D])
    amu_s_dict = ana.make_dict(space,[amusA,amusB,amusD])
    fpi_raw = ana.read_extern("../plots2/data/fpi.dat",(1,2))
    print(amu_s_dict)
    #quark = ens.get_data("quark")
    datadir = '/hiskp4/helmes/analysis/scattering/pi_k/I_32_blocked/data/' 
    plotdir = ens.get_data("plotdir") 
    resdir = ens.get_data("resultdir") 
    nboot = ens.get_data("nboot")
    # keys for the hdf datasets
    epik_meth = ['E1','E3']
    zp_meth = [1, 2]
    # construct filenames
    file_prefix='pi_K_I32'
    # for physical calculations get dictionary of continuum bootstrapsamples
    # seeds for M1A,M1B,M2A and M2B
    ini_path = '/hiskp4/helmes/projects/analysis-code/ini/pi_K/I_32_publish'
    ini1 = ini_path+'/'+'chiral_analysis_mua0_zp1.ini'
    ini2 = ini_path+'/'+'chiral_analysis_mua0_zp2.ini'
    ens1 = ana.LatticeEnsemble.parse(ini1)
    ens2 = ana.LatticeEnsemble.parse(ini2)
    nboot = ens2.get_data('nboot')
    cont={'M1A':ana.ContDat(ens1.get_data('continuum_seeds_a'),zp_meth=1),
          'M2A':ana.ContDat(ens2.get_data('continuum_seeds_a'),zp_meth=2),
          'M1B':ana.ContDat(ens1.get_data('continuum_seeds_b'),zp_meth=1),
          'M2B':ana.ContDat(ens2.get_data('continuum_seeds_b'),zp_meth=2)}
    # construct
    df_collect = pd.DataFrame()
    for tp in epik_meth:
################################################################################
#                   input data                                                 #
################################################################################
        beta_list = [1.90,1.95,2.10]
        for i,a in enumerate(space):
            ext = ana.ExtDat(ens1.get_data('external_seeds_a'),a,zp_meth=1)
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
                elif e == 'D45.32':
                    print("modifying lowest mu_s values")
                    mu_s_dict[a][0] ='amu_s_13' 
                    amu_s_dict[a][0]=0.013
                    mu_s_eta_dict[a][0]='strange_1300'

####    ############### read in M_K^FSE ############################################
                mksq_fse = ana.MatchResult("mksq_fse_%s"%(e),
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
                mpi_fse = ana.MatchResult("mpi_fse_%s"%(e),
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
                mssq_fse = ana.MatchResult("mssq_fse_%s" % (e),save = datadir+'%s/'%e)
                ana.MatchResult.create_empty(mssq_fse,nboot,3)
                mssq_fse.set_data(mksq_fse.obs,amu_s_dict[a])
                mssq_fse.add_data(np.square(mpi_fse.obs),idx=slice(0,3),op='min',
                              fac=0.5)
                print("\nM_s^2:")
                print(mssq_fse.obs[:,0])
                
####    ############### read in M_eta ##############################################
                metasq = ana.MatchResult("metasq_%s"%(e),save=datadir+'%s/'%e)
                ana.MatchResult.create_empty(metasq,nboot,3)
                meta_names = ['/hiskp4/hiskp2/jost/eta_data/'+'%s/' % (e) +s+'/fit_eta_rm_TP0.npz' for s in mu_s_eta_dict[a]]
                meta_meas = ana.init_fitreslst(meta_names)
                metasq.load_data(meta_meas,1,amu_s_dict[a],square=True)
                print("\nM_eta^2:")
                print(metasq.obs[:,0])
########################  read in delta E ################################
                dE = ana.MatchResult("dE_%s_%s" %(tp,
                                        e),save=datadir+'%s/'%e)
                ana.MatchResult.create_empty(dE,nboot,3)
                dE_names = [datadir+'%s/' % (e) +s+'/dE_TP0_%s_%s.npz' 
                               % (e,tp) for s in mu_s_dict[a]]
                dE_meas=ana.init_fitreslst(dE_names)
                dE.load_data(dE_meas,0,amu_s_dict[a],square=False)

########################  read in mu_pik a_3/2 ################################
                mua32 = ana.MatchResult("mua32_%s_%s" %(tp,
                                        e),save=datadir+'%s/'%e)
                ana.MatchResult.create_empty(mua32,nboot,3)
                mua32_names = [datadir+'%s/' % (e) +s+'/mu_a0_TP0_%s_%s.npz' 
                               % (e,tp) for s in mu_s_dict[a]]
                mua32_meas=ana.init_fitreslst(mua32_names)
                mua32_q16 = get_sys_error(mua32_meas) 
                mua32_q84 = get_sys_error(mua32_meas,q='high')
                mua32.load_data(mua32_meas,0,amu_s_dict[a],square=False)
#####    ############### read in M_pi^FSE ###########################################
                mpi_fse = ana.MatchResult("mpi_fse_%s"%(e),save=datadir+'%s/'%e)
                ana.MatchResult.create_empty(mpi_fse,nboot,3)
                mpi_names = [datadir+'%s/' % (e) +'pi'+'/fit_pi_%s.npz' % (e) for s in mu_s_dict[a]]
                mpi_fse_meas = ana.init_fitreslst(mpi_names)
                mpi_fse.load_data(mpi_fse_meas,1,amu_s_dict[a],square=False)
                mpi_fse.add_extern_data('../plots2/data/k_fse_mpi.dat',e,square=False,
                                       read='fse_mpi',op='div')
####    ############################################################################
#                        pseudo bootstraps of fpi                                    #
####    ############################################################################
                dummy, fpi = ana.prepare_fk(fpi_raw,e,nboot)
                print(fpi)
####    ############################################################################
                branch_result=pd.DataFrame()
                # extend dataframe for description
                branch_result['beta']=np.tile(beta,3)
                branch_result['L']=np.tile(L,3)
                branch_result['mu_l']=np.tile(mul,3)
                branch_result['mu_s']=np.repeat(amu_s_dict[a],nboot)
                branch_result['sample']=np.tile(sample,3)\
                # since nothing is specified by now we take the M1A for the
                # external data
                branch_result['r_0']=np.tile(ext.get(a,'r0'),3)
                branch_result['Z_P']=np.tile(ext.get(a,'zp'),3)
                branch_result=branch_result.merge(matchresult_to_df(mpi_fse.obs,
                        amu_s_dict[a],sample,'M_pi_FSE'),on=['mu_s','sample'])
                branch_result=branch_result.merge(matchresult_to_df(np.sqrt(mksq_fse.obs),
                        amu_s_dict[a],sample,'M_K_FSE'),on=['mu_s','sample'])
                branch_result=branch_result.merge(matchresult_to_df(np.sqrt(metasq.obs),
                        amu_s_dict[a],sample,'M_eta'),on=['mu_s','sample'])
                branch_result=branch_result.merge(matchresult_to_df([fpi for i
                    in range(3)],
                        amu_s_dict[a],sample,'f_pi'),on=['mu_s','sample'])
                branch_result=branch_result.merge(matchresult_to_df(mua32.obs,
                        amu_s_dict[a],sample,'mu_piK_a32'),on=['mu_s','sample'])
                branch_result=branch_result.merge(qlst_to_df(mua32_q16,
                        amu_s_dict[a],sample,'mu_piK_a32_q16') ,on=['mu_s','sample'])
                branch_result=branch_result.merge(qlst_to_df(mua32_q84,
                        amu_s_dict[a],sample,'mu_piK_a32_q84') ,on=['mu_s','sample'])
                branch_result=branch_result.merge(matchresult_to_df(dE.obs,
                        amu_s_dict[a],sample,'deltaE'),on=['mu_s','sample'])
                branch_result['poll'] = tp
                df_collect = pd.concat((df_collect,branch_result))
    print(df_collect.sample(n=20))

    filename=resdir+'pik_I32_bsamples.h5'
    store = pd.HDFStore(filename)
    store.put('non_interpolated',df_collect)
    del store
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
