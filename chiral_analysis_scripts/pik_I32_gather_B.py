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
# Gather the data necessary to fix the strange quark mass in method B and put it
# into a pandas dataframe
# 
# The data is read in into the long data format and stored as an hdf5 file
# 
################################################################################

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
    lat = ens.name()
    space=ens.get_data("beta")
    space=space[:-1]
    latA = ens.get_data("namea")
    latB = ens.get_data("nameb")
    latD = ens.get_data("named")
    strangeA = ens.get_data("strangea")
    strangeB = ens.get_data("strangeb")
    strangeD = ens.get_data("stranged")
    strange_eta_A = ens.get_data("strange_alt_a")
    strange_eta_B = ens.get_data("strange_alt_b")
    strange_eta_D = ens.get_data("strange_alt_d")
    zp_meth=ens.get_data("zp_meth")
    epik_meth = ens.get_data("epik_meth") 
    external_seeds=ens.get_data("external_seeds_a")
    continuum_seeds=ens.get_data("continuum_seeds_b")
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
    datadir = ens.get_data("datadir") 
    resdir = ens.get_data("resultdir") 
    nboot = ens.get_data("nboot")
    # Prepare external data
    ext_data = ana.ExtDat(external_seeds,space,zp_meth,nboot=nboot)
    # initialize an empty dataframe and get the corresponding data 
    observables = ['nboot','beta','mu_l','mu_s','L','M_K^2_FSE','M_eta^2',
                   'mu_piK_a32','r_0','Z_P']
    unfixed_data = pd.DataFrame(columns=observables)
    beta_vals = [1.90,1.95,2.10]
    samples = np.arange(nboot)
    # Loop over mul and mus and put data into subframe
    for i,a in enumerate(space):
        print("\nWorking at lattice spacing %s" %a)
        beta = np.full(nboot,beta_vals[i])
        # Pick samples of r0 and ZP
        r0 = ext_data.get(a,'r0') 
        zp = ext_data.get(a,'zp')
        for j,e in enumerate(lat_dict[a]):
            mu_l = np.full(nboot,amu_l_dict[a][j])
            # Assumes ensemblename
            length = np.full(nboot,int(lat_dict[a][j].split('.')[1]))
            # if using D30.48 modify lowest amus to 0.0115
            if e == 'D30.48':
                print("modifying lowest mu_s values")
                mu_s_dict[a][0] ='amu_s_115' 
                amu_s_dict[a][0]=0.0115
    ################### read in M_K^FSE ############################################
            mksq_fse = ana.MatchResult("mksq_fse_M%dB_%s"%(zp_meth,e),save=datadir+'%s/'%e)
            ana.MatchResult.create_empty(mksq_fse,nboot,3)
            mk_names = [datadir+'%s/' % (e) +s+'/fit_k_%s.npz' % (e) for s in mu_s_dict[a]]
            mksq_fse_meas = ana.init_fitreslst(mk_names)
            mksq_fse.load_data(mksq_fse_meas,1,amu_s_dict[a],square=True)
            mksq_fse.add_extern_data('../plots2/data/k_fse_mk.dat',e,square=True,
                                   read='fse_mk',op='mult')
    ################### read in M_eta^2 ############################################
            metasq = ana.MatchResult("metasq_M%dB_%s"%(zp_meth,e),save=datadir+'%s/'%e)
            ana.MatchResult.create_empty(metasq,nboot,3)
            meta_names = ['/hiskp4/hiskp2/jost/eta_data/'+'%s/' % (e) +s+'/fit_eta_rm_TP0.npz' for s in mu_s_eta_dict[a]]
            meta_meas = ana.init_fitreslst(meta_names)
            metasq.load_data(meta_meas,1,amu_s_dict[a],square=True)
########################  read in mu_pik a_3/2 ################################
            mua32 = ana.MatchResult("mua32_M%dB_%s_%s" %(zp_meth,epik_meth,
                                    e),save=datadir+'%s/'%e)
            ana.MatchResult.create_empty(mua32,nboot,3)
            mua32_names = [datadir+'%s/' % (e) +s+'/mu_a0_TP0_%s_%s.npz' 
                           % (e,epik_meth) for s in mu_s_dict[a]]
            mua32_meas=ana.init_fitreslst(mua32_names)
            mua32.load_data(mua32_meas,0,amu_s_dict[a],square=False)
            # Add data to chirana object
            for k,s in enumerate(amu_s_dict[a]):
                mu_s = np.full(nboot,s)
                mksq = mksq_fse.obs[k]
                meta_sq = metasq.obs[k]
                mua32_dat = mua32.obs[k]
                value_list = [samples,beta,mu_l,mu_s,length,mksq,meta_sq,mua32_dat,r0,zp]
                tmp_frame = pd.DataFrame({key:values for key, values in
                    zip(observables, value_list)})
                # append subframe to unfixed dataframe
                unfixed_data =unfixed_data.append(tmp_frame)
    unfixed_data.nboot=unfixed_data.nboot.astype(int)
    unfixed_data.L=unfixed_data.L.astype(int)
    print(unfixed_data.sample(n=20))
    unfixed_data.info()
    proc_id = 'piK_I32_unfixed_data_B%d'%(zp_meth)
    hdf_filename = resdir+'/'+proc_id+'.h5'
    hdfstorer = pd.HDFStore(hdf_filename)
    hdfstorer[proc_id] = unfixed_data
    del hdfstorer

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
