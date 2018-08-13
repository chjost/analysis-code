#!/usr/bin/python

# Cut fitrange of pik fit and calculate delta E

import sys
import numpy as np
import pandas as pd
import itertools
# Christian's packages
sys.path.append('/hiskp4/helmes/projects/analysis-code/')

import analysis2 as ana

def main():
    # parse the input file
    if len(sys.argv) < 2:
        ens = ana.LatticeEnsemble.parse("kk_I1_TP0_A40.24.ini")
    else:
        ens = ana.LatticeEnsemble.parse(sys.argv[1])
    # get data from input file
    prefix = ens.get_data("path")
    print prefix
    lat = ens.name()
    nboot = ens.get_data("nboot")
    datadir = ens.get_data("datadir")
    datadir_pi = ens.get_data("datadir_pi")
    plotdir = ens.get_data("plotdir")
    gmax = ens.get_data("gmax")
    d2 = ens.get_data("d2")
    fit_e_tot = np.int_(ens.get_data("fitetot"))
    print(fit_e_tot)
    corr_pik_out = "corr_pik"
    try:
        debug = ens.get_data("debug")
    except KeyError:
        debug = 0
    L = ens.L()
    T = ens.T()
    T2 = ens.T2()
    addT = np.ones((nboot,)) * T
    addT2 = np.ones((nboot,)) * T2
    mus_dir = datadir.split('/')[-3]
    print(mus_dir)
    # get initial and final time
    range_cut = pd.read_csv('/hiskp4/helmes/analysis/scattering/pi_k/I_32_blocked/runs/fit_range_cut.txt',
            sep='\s+')
    print(range_cut)
    t_i = range_cut.where((range_cut['#Ensemble']==lat) &
            (range_cut['mu_s_dir']==mus_dir)).dropna()['pik_bgn'].values
    t_f = range_cut.where((range_cut['#Ensemble']==lat) &
            (range_cut['mu_s_dir']==mus_dir)).dropna()['pik_end'].values
    print(t_i,t_f)
#--------------- Define filenames
    fit_k_out="fit_k"
    fit_pi_out = "fit_pi"
    fit_pik_out = "fit_pik"
# --------------- Load single particle fitresults   
# Kaon fit
    k_fit = ana.FitResult.read("%s/%s_%s.npz" % (datadir,fit_k_out, lat))
    k_fit.print_data(1)
    k_fit.print_details()
    k_median = k_fit.singularize()

# Pion fit
    pi_fit = ana.FitResult.read("%s/%s_%s.npz" % (datadir_pi,fit_pi_out, lat))
    pi_fit.print_data(1)
    pi_fit.print_details()
    pi_median = pi_fit.singularize()

    # combine them in an energy difference
    diff_pi_k = k_fit.add_mass(pi_fit,neg=True)
    print(diff_pi_k.derived)
    diff_pi_k.singularize()
    diff_pi_k.print_details()
    diff_pi_k.print_data()
    masses=pi_median.comb_fitres(k_median,1)
    masses.print_details()
# --------------- fit epik
    pik_corr = ana.Correlators.read("%s/%s_%s.npz" % (datadir, corr_pik_out, lat))
    #E1
    # E1
    pik_corr_e1 = ana.Correlators.create(pik_corr.data,T=T)
    # Before fitting shift the correlator
    # make matrix out of corr
    e1_corr_shift = ana.Correlators.create(pik_corr_e1.data)
    e1_corr_shift.matrix=True
    e1_corr_shift.shape = np.append(pik_corr_e1.data.shape,1)
    e1_corr_shift.data.reshape((e1_corr_shift.shape))
    e1_corr_shift.shift(1,mass = diff_pi_k.singularize().data[0][:,0,0], shift=1,d2=0)
    # Convert again to correlator for plotting ws denotes weighted and shifted
    e1_corr_ws = ana.Correlators.create(e1_corr_shift.data[...,0],T=T)
    e1_corr_ws.shape = e1_corr_ws.data.shape
    # fit e1
    print("\nFitting Branch E1")
    print("\n3 parameter fit")
    fit_pi_k_e1 = ana.LatticeFit(8, dt_f=-1, dt_i=1, dt=4,
                              correlated=True,debug=0)
    start = [5.,0.5,10.]
    pi_k_fitresult_e1 = fit_pi_k_e1.fit(start,e1_corr_ws,[[11,19]],add=addT,
            oldfit=masses, oldfitpar=slice(0,2))
          
    pi_k_fitresult_e1.print_details()
    print("\neffective mass")
    e1_corr_ws.mass(function=4,add=[masses.data[0][:,0,0],masses.data[0][:,1,0]])
    fit_pi_k_e1 = ana.LatticeFit(2, dt_f=-1, dt_i=1, dt=4,
                              correlated=True,debug=0)
    start = [0.5]
    pi_k_fitmass_e1 = fit_pi_k_e1.fit(start,e1_corr_ws,[[11,19]],add=addT)
          
    pi_k_fitmass_e1.print_details()
    #pi_k_fitresult_e1_cut.save("%s/%s_%s_E1.npz" % (datadir,fit_pik_out,lat))
    # E3
    print("\nFitting Branch E3")
    pik_corr_e3 = ana.Correlators.create(pik_corr.data,T=T)
    pik_corr_e3.divide_out_pollution(pi_fit,k_fit)
    e3_corr_shift = ana.Correlators.create(pik_corr_e3.data)
    e3_corr_shift.matrix=True
    e3_corr_shift.shape = np.append(pik_corr_e3.data.shape,1)
    e3_corr_shift.data.reshape((e3_corr_shift.shape))
    e3_corr_shift.shift(1, shift=1,d2=0)
    # Convert again to correlator for plotting ws denotes weighted and shifted
    e3_corr_ws = ana.Correlators.create(e3_corr_shift.data[...,0],T=T)
    e3_corr_ws.shape = e3_corr_ws.data.shape
    e3_corr_ws.multiply_pollution(pi_fit,k_fit)
    # TODO: Need to get mass first! PAck in one function to clutter script less
    epi = pi_median.data[0][:,1,-1] 
    ek = k_median.data[0][:,1,-1]
    # Calculate effective mass
    e3_corr_ws.mass(function=5,add=[epi,ek])
    print("\nFitting Branch E3")
    fit_pi_k_e3 = ana.LatticeFit(2, dt_f=-1, dt_i=1, dt=4,
                              correlated=True)
    start = [0.5]
    pi_k_fitresult_e3 = fit_pi_k_e3.fit(start,e3_corr_ws,[[11,24]],add=addT)
    pi_k_fitresult_e3.print_details()
    #pi_k_fitresult_e3.save("%s/%s_%s_E3.npz" % (datadir,fit_pik_out,lat))

# --------------- Calculate m_pi + m_K
    sum_m = k_median.add_mass(pi_median)
    sum_m.print_data(0)
    sum_m.print_details()

# --------------- Calculate delta E for all three methods
    dE_e1 = pi_k_fitresult_e1.calc_dE(sum_m,parself=1,parmass=0,flv_diff=True,
                          isdependend=True)
    dE_e1.print_details()
    dE_e1mass = pi_k_fitmass_e1.calc_dE(sum_m,parself=0,parmass=0,flv_diff=True,
                          isdependend=False)
    dE_e1mass.print_details()
    #dE_e1.save("%s/dE_cut_%s_E1.npz" % (datadir,lat))

    dE_e3 = pi_k_fitresult_e3.calc_dE(sum_m,parself=0,parmass=0,flv_diff=True,
                          isdependend=False)
    #dE_e3.save("%s/dE_cut_%s_E3.npz" % (datadir,lat))
    dE_e3.print_details()

if __name__ == '__main__':
    try:
        print("starting")
        main()
    except KeyboardInterrupt:
        pass

