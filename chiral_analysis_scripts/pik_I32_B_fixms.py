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
# the strange quark mass from a global fit to the physical kaon mass is taken.
# The fit reads
# <M_K formula>
# and for the strange quark mass
# <strange quark mass formula>
# is introduced. The lattice results are interpolated to the corresponding 
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
def amk_sq(df):
    p0 = df['P_0'].values
    p1 = df['P_1'].values
    p2 = df['P_2'].values
    pr = df['P_r'].values
    pz = df['P_Z'].values
    pmu = df['P_mu'].values
    mul = df['mu_l'].values
    mus = df['mu_s'].values
    mksq_func = p0/(pr*pz)*(mul+pmu*mus)*(1+p1*pr*mul/pz+p2/pr**2)
    mksq = pd.Series(mksq_func,index = df.index)
    return mksq

def ms_phys(df,cont):
    """ Calculate physical strange quark mass from continuum data and
    fitresutlts
    """
    ms_phys = 

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
    lat_dict = ana.make_dict(space,[latA,latB,latD,latD45])
    amu_l_dict = ana.make_dict(space,[amulA,amulB,amulD,amulD45])
    mu_s_dict = ana.make_dict(space,[strangeA,strangeB,strangeD,strangeD45])
    mu_s_eta_dict = ana.make_dict(space,[strange_eta_A,strange_eta_B,strange_eta_D,strange_eta_D45])
    amu_s_dict = ana.make_dict(space,[amusA,amusB,amusD,amusD45])
    print(amu_s_dict)
    datadir = ens.get_data("datadir") 
    plotdir = ens.get_data("plotdir") 
    resdir = ens.get_data("resultdir") 
    nboot = ens.get_data("nboot")
    # Prepare external data
    ext_data = ana.ExtDat(external_seeds,space,zp_meth,nboot=nboot)
    cont_data = ana.ContDat(continuum_seeds,zp_meth=zp_meth,nboot=nboot)
    fpi_raw = ana.read_extern("../plots2/data/fpi.dat",(1,2))
    print(fpi_raw)

    read_ms_fix = False
    read_ext = False
################### Setup fixation of strange quark mass #############################
    fixms = ana.ChirAna("pi-K_I32_chipt_fixms_M%dB"%zp_meth,correlated=True,gamma=False,
                           match=True, fit_ms = True)
    ens_shape_chirana = (len(latA),len(latB),len(latD),len(latD45))
    print(ens_shape_chirana)
    # have 3 strange quark masses 
    lyt_xvals = ana.generate_chirana_shape(space,ens_shape_chirana,3,2,nboot)
    lyt_yvals = ana.generate_chirana_shape(space,ens_shape_chirana,3,1,nboot) 
    #lyt_xvals = (3,ens_shape_chirana,3,2,nboot)
    #lyt_yvals = (3,ens_shape_chirana,3,1,nboot)
    fixms.create_empty(lyt_xvals,lyt_yvals,lat_dict=lat_dict)
    print("\nSetup complete, begin chiral analysis")
    if read_ms_fix is True:
        fixms.load(resdir)
    else:

################################################################################
#                   input data                                                 #
################################################################################
        for i,a in enumerate(space):
            print("\nWorking at lattice spacing %s" %a)
            for j,e in enumerate(lat_dict[a]):
                for k,s in enumerate(amu_s_dict[a]):
####################### add r0 ml ##################################################
                    mq_tmp = np.full((nboot,),amu_l_dict[a][j])
                    fixms.add_data(mq_tmp,(i,j,k,0),dim='x')
####################### add r0 ms ##################################################
                    mq_tmp = np.full((nboot,),s)
                    fixms.add_data(mq_tmp,(i,j,k,1),dim='x')
####    ############### read in M_K^FSE ############################################
                mksq_fse = ana.MatchResult("mksq_fse_M%dB_%s"%(zp_meth,e),save=datadir+'%s/'%e)
                ana.MatchResult.create_empty(mksq_fse,nboot,3)
                mk_names = [datadir+'%s/' % (e) +s+'/fit_k_%s.npz' % (e) for s in mu_s_dict[a]]
                mksq_fse_meas = ana.init_fitreslst(mk_names)
                mksq_fse.load_data(mksq_fse_meas,1,amu_s_dict[a],square=True)
                mksq_fse.add_extern_data('../plots2/data/k_fse_mk.dat',e,square=True,
                                       read='fse_mk',op='mult')
                # Add data to chirana object
                for k,s in enumerate(amu_s_dict[a[0]]):
                    fixms.add_data(mksq_fse.obs[k],(i,j,k,0),'y')
        #fixms.save(resdir)
        #fixms.fit_strange_mass(datadir=datadir,ext=ext_data)
        # have as many values for r and z as lattice spacings
        r = np.r_[[1. for s in space[:-1]]]
        z = np.r_[[1. for s in space[:-1]]]
        mu = np.r_[[1. for s in space[:-1]]]
        p = np.r_[6.,0.1,1.]
        #start=np.r_[r,z,p]
        start=np.r_[r,z,p,mu,0.1]
        # Get the prior samples
        external_r0 = [ext_data.get(beta,'r0') for beta in space[:-1]]
        external_zp = [ext_data.get(beta,'zp') for beta in space[:-1]]
        prior = np.vstack(tuple(external_r0 + external_zp))
        #prior = np.vstack((ext_data.get('A','r0'), ext_data.get('B','r0'),
        #                   ext_data.get('D','r0'), ext_data.get('A','zp'), 
        #                   ext_data.get('B','zp'), ext_data.get('D','zp')))

        fixms.fit(ana.global_ms_errfunc,start,plotdir=plotdir,correlated=True,
                  prior=prior)
        header=[r'$r_0m_l$',r'$(r_0 M_K)^2$',r'Physical $m_s$ from physical $M_K^2$',
               r'$(r_0M_{\pi})_{phys}^{2}$']
        ana.print_summary(fixms,header,amu_l_dict,amu_s_dict)
        label = [r'$a\mu_l$',r'$(aM_K)^2$']
        #pick the correct arguments
        args = fixms.fitres.data[0] 
        plot_args = np.asarray([np.hstack((args[:,0+i],args[:,len(space)-1+i],
            args[:,2*len(space):,0])) for i in range(len(space)-1)])
        # build dataframe for function evaluation
        # Intermezzo: collect fitresults
        fitres_columns = ['beta','mu_l','L','mu_s','sample','M_K^2_FSE','P_0','P_1',
                          'P_2','P_r','P_Z','P_mu','chisq']
        fitres = pd.DataFrame(columns=fitres_columns)
        # loop over beta, mu_l and mu_s
        beta_list = [1.90,1.95,2.10,2.10]
        # reset from D30.48 values
        print("modifying lowest mu_s values")
        for i,a in enumerate(space):
            sample = np.arange(nboot)
            beta = np.full(nboot,beta_list[i])
            for j,e in enumerate(lat_dict[a]):
                mul = np.full(nboot,amu_l_dict[a][j])
                L = np.full(nboot,int(e.split('.')[-1]))
                for k,s in enumerate(amu_s_dict[a]):
                    mus = np.full(nboot,amu_s_dict[a][k])
                    if i < 3:
                        fitres_dict = {'beta': beta,
                                        'L': L,
                                        'mu_l': mul,
                                        'mu_s':mus,
                                        'sample':sample,
                                        'M_K^2_FSE':fixms.get_data((i,j,k,0),'y'),
                                        'P_0':args[:,2*(len(space)-1),0],
                                        'P_1':args[:,2*(len(space)-1)+1,0],
                                        'P_2':args[:,2*(len(space)-1)+2,0],
                                        'P_r':args[:,0+i,0],
                                        'P_Z':args[:,3+i,0],
                                        'P_mu':args[:,2*(len(space)-1)+3+i,0],
                                        'chisq':fixms.fitres.chi2[0][:,0]}
                                        
                        tmp_df = pd.DataFrame(data=fitres_dict)
                        fitres = fitres.append(tmp_df)
                    else:
                        fitres_dict = {'beta': beta,
                                        'L': L,
                                        'mu_l': mul,
                                        'mu_s':mus,
                                        'sample':sample,
                                        'M_K^2_FSE':fixms.get_data((i,j,k,0),'y'),
                                        'P_0':args[:,2*(len(space)-1),0],
                                        'P_1':args[:,2*(len(space)-1)+1,0],
                                        'P_2':args[:,2*(len(space)-1)+2,0],
                                        'P_r':args[:,0+i-1,0],
                                        'P_Z':args[:,3+i-1,0],
                                        'P_mu':args[:,-1,0],
                                        'chisq':fixms.fitres.chi2[0][:,0]}
                        tmp_df = pd.DataFrame(data=fitres_dict)
                        fitres = fitres.append(tmp_df)

        fitres.info()
        print(fitres.sample(n=20))
        fitres['M_K^2_func'] = amk_sq(fitres)
        fitres['rel.dev.'] = (fitres['M_K^2_FSE']-fitres['M_K^2_func'])/fitres['M_K^2_FSE']
        groups = ['beta','L','mu_l','mu_s']
        observables=['M_K^2_FSE','P_0','P_1','P_2','P_r','P_Z','P_mu','M_K^2_func','chisq']
        #observables=['M_K^2_FSE','M_K^2_func','rel.dev.']
        print(chi.bootstrap_means(fitres,groups,observables))
        proc_id = 'pi_K_I32_fixms_M%dB'%(zp_meth)
        hdf_filename = resdir+proc_id+'.h5'
        hdfstorer = pd.HDFStore(hdf_filename)
        hdfstorer.put('Fitresults_sigma_woA4024',fitres)
        del hdfstorer
        
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard Interrupt")

