#!/usr/bin/python

# Calculate delta E and pi-K scattering length for I=3/2

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
    try:
        debug = ens.get_data("debug")
    except KeyError:
        debug = 0
    L = ens.L()
    T = ens.T()
    T2 = ens.T2()
    addT = np.ones((nboot,)) * T
    addT2 = np.ones((nboot,)) * T2
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

    mus_dir = datadir.split('/')[-3]
# --------------- Load total energy fitresult
    #E1
    pi_k_fitresult_e1 = ana.FitResult.read("%s/%s_%s_E1.npz" % (datadir,fit_pik_out,lat))
    #pi_k_fitresult_e1.print_data(1)
    #E2
    pi_k_fitresult_e2 = ana.FitResult.read("%s/%s_%s_allfr_E2.npz" % (datadir,fit_pik_out,lat))
    #pi_k_fitresult_e2.print_data(1)
    cut=False
    if cut is True:
        range_cut = pd.read_csv('/hiskp4/helmes/analysis/scattering/pi_k/I_32_publish/runs/range_cuts_pik.txt',
                sep='\s+')
        E1t_i = range_cut.where((range_cut['Ensemble']==lat) &
                (range_cut['mu_s_dir']==mus_dir)).dropna()['E1_init'].values
        E1t_f = range_cut.where((range_cut['Ensemble']==lat) &
                (range_cut['mu_s_dir']==mus_dir)).dropna()['E1_final'].values
        E2t_i = range_cut.where((range_cut['Ensemble']==lat) &
                (range_cut['mu_s_dir']==mus_dir)).dropna()['E2_init'].values
        E2t_f = range_cut.where((range_cut['Ensemble']==lat) &
                (range_cut['mu_s_dir']==mus_dir)).dropna()['E2_final'].values
        print(E1t_i,E1t_f)
        print(E2t_i,E2t_f)
        pi_k_fitresult_e1 = pi_k_fitresult_e1.cut_data(E1t_i,E1t_f,par=1) 
        pi_k_fitresult_e2 = pi_k_fitresult_e2.cut_data(E2t_i,E2t_f,par=1)
        pi_k_fitresult_e1.print_data(par=1)
        pi_k_fitresult_e2.print_data(par=1)


# --------------- Calculate m_pi + m_K
    sum_m = k_median.add_mass(pi_median)
    sum_m.print_data(0)

# --------------- Calculate reduced mass
    calc_mu = True
    if calc_mu:
        mu = pi_median.reduced_mass(k_median)
        mu.save("%s/mu_pi_k_TP%d_%s.npz" % (datadir, d2, lat))
    else:
        print("%s/mu_pi_k_TP%d_%s.npz" % (datadir, d2, lat))
        mu = ana.FitResult.read("%s/mu_pi_k_TP%d_%s.npz" % (datadir, d2, lat))
    mu.print_data(0)

# --------------- Calculate delta E for all three methods
    calc_dE = True
    if calc_dE:
        dE_e1 = pi_k_fitresult_e1.calc_dE(sum_m,parself=1,parmass=0,flv_diff=True,
                              isdependend=True)
        dE_e1.save("%s/dE_TP0_%s_E1.npz" % (datadir,lat))

        dE_e2 = pi_k_fitresult_e2.calc_dE(sum_m,parself=1,parmass=0,flv_diff=True,
                              isdependend=True)
        dE_e2.save("%s/dE_TP0_%s_E2.npz" % (datadir,lat))

    else: 
        dE_e1 = ana.FitResult.read("%s/dE_TP0_%s_E1.npz" % (datadir,lat))
        dE_e2 = ana.FitResult.read("%s/dE_TP0_%s_E2.npz" % (datadir,lat))
    dE_e1.print_data()
    dE_e2.print_data()
# --------------- Calculate scattering length
    calca = True
    if calca:
        print("calculate scattering length for E1")
        a_32_e1 = dE_e1.calc_scattering_length(mu,parself=0,isratio=True,
                                    isdependend=True,L=L)
        a_32_e1.save("%s/scat_len_TP%d_%s_E1.npz" % (datadir, d2, lat))
        print("calculate scattering length for E2")
        a_32_e2 = dE_e2.calc_scattering_length(mu,parself=0,isratio=True,
                                    isdependend=False,L=L)
        a_32_e2.save("%s/scat_len_TP%d_%s_E2.npz" % (datadir, d2, lat))
    else:
        print("%s/scat_len_TP%d_%s_E1.npz" % (datadir, d2, lat))
        a_32_e1 = ana.FitResult.read("%s/scat_len_TP%d_%s_E1.npz" % (datadir, d2, lat))
        print("%s/scat_len_TP%d_%s_E2.npz" % (datadir, d2, lat))
        a_32_e2 = ana.FitResult.read("%s/scat_len_TP%d_%s_E2.npz" % (datadir, d2, lat))
    a_32_e1.print_data()
    a_32_e2.print_data()
    plotter = ana.LatticePlot("%s/scat_len_TP%d_%s.pdf" % (plotdir, d2, lat))
    label = ["scattering length", "a$_{\pi K}$", "a$_{\pi K}$","E1"]
    plotter.histogram(a_32_e1, label)
    label = ["scattering length", "a$_{\pi K}$", "a$_{\pi K}$","E2"]
    plotter.histogram(a_32_e2, label)
    plotter.new_file("%s/qq_a0_%s.pdf" % (plotdir, lat))
    label = [r'QQ-Plot $a_0$ %s' % lat, r'weighted $a_0$ E1']
    plotter.qq_plot(a_32_e1,label,par=0)
    label = [r'QQ-Plot $a_0$ %s' % lat, r'weighted $a_0$ E2']
    plotter.qq_plot(a_32_e2,label,par=0)
    del plotter

# --------------- Dimensionless product mu a_3/2
    mult_mu_a0 = True
    if mult_mu_a0:
        mult_obs_e1 = a_32_e1.mult_obs(mu, "mu_a32_e1", isdependend = True)
        mult_obs_e1.save("%s/mu_a0_TP%d_%s_E1.npz" % (datadir, d2, lat))
        mult_obs_e2 = a_32_e2.mult_obs(mu, "mu_a32_e2", isdependend = True)
        mult_obs_e2.save("%s/mu_a0_TP%d_%s_E2.npz" % (datadir, d2, lat))
    else:
        mult_obs_e1 = ana.FitResult.read("%s/mu_a0_TP%d_%s_E1.npz" % (datadir, d2, lat))
        mult_obs_e2 = ana.FitResult.read("%s/mu_a0_TP%d_%s_E2.npz" % (datadir, d2, lat))
    mult_obs_e1.print_data()
    mult_obs_e2.print_data()
    plotter = ana.LatticePlot("%s/mu_a0_TP%d_%s.pdf" % (plotdir, d2, lat))
    label = ["mu a0", "$\mu$ a$_{\pi K}$", "$\mu$ a$_{\pi K}$","E1"]
    plotter.histogram(mult_obs_e1, label)
    label = ["mu a0", "$\mu$ a$_{\pi K}$", "$\mu$ a$_{\pi K}$","E2"]
    plotter.histogram(mult_obs_e2, label)
    label = ["mu a0", "$\mu$ a$_{\pi K}$", "$\mu$ a$_{\pi K}$","E3"]
    label = [r'QQ-Plot $a_0$ %s' % lat, r'$\mu$ $a_0$ E1']
    plotter.qq_plot(mult_obs_e1,label,par=0)
    label = [r'QQ-Plot $a_0$ %s' % lat, r'$\mu$ $a_0$ E2']
    plotter.qq_plot(mult_obs_e2,label,par=0)
    del plotter

if __name__ == '__main__':
    try:
        print("starting")
        main()
    except KeyboardInterrupt:
        pass
