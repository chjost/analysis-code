#!/usr/bin/python

# This script is part of an analysis chain and depends on fitted pion and kaon
# masses. they can be calculated using pik_32_fit_pi.py and pik_32_fit_k.py
# Fit piK four point correlation function with three methods:
# E1: Remove leading pollutional states with weighting and shifting, fit
#     remaining one explicitly
# E2: Subtract pollution explicitly, calculate effective mass and fit a constant
#     to it
# E3: divide out pollutional state, calculate effective mass and fit a constant
#     to it. Effective mass formula gets complicated.
# 
# The fits happen by combining different fitranges of the two point functions
# and thus are derived fits. Histograms are in order and can be checked for a
# gaussian distribution of the fitted energies

import sys
import numpy as np
import itertools
# Christian's packages
#sys.path.append('/hiskp4/helmes/projects/analysis-code/')

import analysis2 as ana

def main():
    read_data = True 
    read_fit = False
    do_e1 = True
    do_e2 = False
    do_e3 = True 
    # parse infile
    if len(sys.argv) < 2:
        ens = ana.LatticeEnsemble.parse("kk_I1_TP0_A40.24.ini")
    else:
        ens = ana.LatticeEnsemble.parse(sys.argv[1])
    # get data from input file
    prefix = ens.get_data("path")
    print prefix
    lat = ens.name()
    nboot = ens.get_data("nboot")
    bs_bl = ens.get_data("boot_bl")
    datadir = ens.get_data("datadir")
    datadir_pi = ens.get_data("datadir_pi")
    plotdir = ens.get_data("plotdir")
    gmax = ens.get_data("gmax")
    d2 = ens.get_data("d2")
    fit_e_tot = np.int_(ens.get_data("fitetot"))
    min_size_etot = ens.get_data("tmin_etot")
    try:
        debug = ens.get_data("debug")
    except KeyError:
        debug = 0
    L = ens.L()
    T = ens.T()
    T2 = ens.T2()
    addT = np.ones((nboot,)) * T
    addT2 = np.ones((nboot,)) * T2
    # Set external names
    fit_k_out = "fit_k" 
    fit_pi_out = "fit_pi"
    fit_pik_out = "fit_pik"
    corr_pik_in = "pik_charged_A1_TP0_00" 
    fit_pik_out = "fit_pik"
    corr_pik_out = "corr_pik"
    # Load pion and kaon fitresults
    k_fitresult = ana.FitResult.read("%s/%s_%s.npz" % (datadir,fit_k_out, lat))
    k_fitresult.print_details()
    pi_fitresult = ana.FitResult.read("%s/%s_%s.npz" % (datadir_pi,fit_pi_out, lat))
    pi_fitresult.print_details()
    # combine them in an energy difference
    diff_pi_k = k_fitresult.add_mass(pi_fitresult,neg=True)
    print(diff_pi_k.derived)
    diff_pi_k.singularize()
    diff_pi_k.print_details()
    diff_pi_k.print_data()
    k_median = k_fitresult.singularize()
    pi_median = pi_fitresult.singularize()
    masses=pi_median.comb_fitres(k_median,1)
    masses.print_details()
    # load 4point correlator
    files = ["%s/%s.dat" % (datadir,corr_pik_in)]
    if read_data == False:
        pik_corr = ana.Correlators(files, matrix=False,conf_col=3)
    # symmetrize and bootstrap
        pik_corr.sym_and_boot(nboot,bl=bs_bl,method='stationary')
        #pik_corr.bootstrap(nboot)
        pik_corr.save("%s/%s_%s.npy" % (datadir,corr_pik_out, lat))
    else:
        pik_corr = ana.Correlators.read("%s/%s_%s.npz" % (datadir, corr_pik_out, lat))

    if do_e1 is True:
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
        #fit_pi_k_e1 = ana.LatticeFit(8, dt_f=1, dt_i=1, dt=min_size_etot,
        #                          correlated=True,debug=3)
        fit_pi_k_e1 = ana.LatticeFit(8, dt_f=1, dt_i=1, dt=4,
                                  correlated=False,debug=3)
        if read_fit==False:
            start = [5.,0.5,10.]
            #pi_k_fitresult_e1 = fit_pi_k_e1.fit(start,e1_corr_ws,fit_e_tot,add=addT,
            #        oldfit=masses, oldfitpar=slice(0,2))
            pi_k_fitresult_e1 = fit_pi_k_e1.fit(start,e1_corr_ws,[[12,19]],add=addT,
                    oldfit=masses, oldfitpar=slice(0,2))
            pi_k_fitresult_e1.save("%s/%s_%s_E1.npz" % (datadir,fit_pik_out,lat))
        else:
            pi_k_fitresult_e1.read("%s/%s_%s_E1.npz" % (datadir,fit_pik_out,lat))
        pi_k_fitresult_e1.print_details()
        plotter=ana.LatticePlot("%s/%s_%s_E1.pdf" %(plotdir,fit_pik_out,lat))
        plotter.set_env(ylog=True,grid=False,title=True)
        label=[r'$\pi K$ raw correlator','$t$',r'$C(t)$',r'E1']
        plotter.plot(e1_corr_ws,label,pi_k_fitresult_e1,fit_pi_k_e1,add=addT,
                     oldfit=masses,oldfitpar=slice(0,2))
        plotter.new_file("%s/%s_%s_E1_hist.pdf" %(plotdir,fit_pik_out,lat))
        label = ["pik energy", "E$_{\pi K}$/a", "E$_{\pi K}$",'E1']
        plotter.histogram(pi_k_fitresult_e1,label,par=1)
        del plotter
    
    if do_e2 is True:
        # E2
        print("\nFitting Branch E2")
        pik_corr_e2 = ana.Correlators.create(pik_corr.data,T=T)
        e2_corr_sub = ana.Correlators.create(pik_corr_e2.data,T=T)
        e2_corr_sub.subtract_pollution(pi_fitresult,k_fitresult)
        # Calculate effective mass
        e2_corr_sub.mass(function=0)
        fit_pi_k_e2 = ana.LatticeFit(2, dt_f=1, dt_i=1, dt=min_size_etot,
                                  correlated=True)
        if read_fit==False:
            start = [0.5]
            pi_k_fitresult_e2 = fit_pi_k_e2.fit(start,e2_corr_sub,fit_e_tot,add=addT)
            pi_k_fitresult_e2.save("%s/%s_%s_E2.npz" % (datadir,fit_pik_out,lat))
        else:
            pi_k_fitresult_e2.read("%s/%s_%s_E2.npz" % (datadir,fit_pik_out,lat))
        pi_k_fitresult_e2.print_details()
        plotter=ana.LatticePlot("%s/%s_%s_E2.pdf" %(plotdir,fit_pik_out,lat))
        plotter.set_env(ylog=False,grid=False,title=True)
        label=[r'$\pi K$ effective mass','$t$',r'$M_eff(t)$',r'E2']
        plotter.plot(e2_corr_sub,label,pi_k_fitresult_e2,fit_pi_k_e2,add=addT)
        plotter.new_file("%s/%s_%s_e2_hist.pdf" %(plotdir,fit_pik_out,lat))
        label = ["pik energy", "E$_{\pi K}$/a", "E$_{\pi K}$",'E2']
        plotter.histogram(pi_k_fitresult_e2,label,par=0)
        del plotter
    
    if do_e3 is True:
        # E3
        print("\nFitting Branch E3")
        pik_corr_e3 = ana.Correlators.create(pik_corr.data,T=T)
        pik_corr_e3.divide_out_pollution(pi_fitresult,k_fitresult)
        e3_corr_shift = ana.Correlators.create(pik_corr_e3.data)
        e3_corr_shift.matrix=True
        e3_corr_shift.shape = np.append(pik_corr_e3.data.shape,1)
        e3_corr_shift.data.reshape((e3_corr_shift.shape))
        e3_corr_shift.shift(1, shift=1,d2=0)
        # Convert again to correlator for plotting ws denotes weighted and shifted
        e3_corr_ws = ana.Correlators.create(e3_corr_shift.data[...,0],T=T)
        e3_corr_ws.shape = e3_corr_ws.data.shape
        e3_corr_ws.multiply_pollution(pi_fitresult,k_fitresult)
        # TODO: Need to get mass first! PAck in one function to clutter script less
        epi = pi_median.data[0][:,1,-1] 
        ek = k_median.data[0][:,1,-1]
        # Calculate effective mass
        e3_corr_ws.mass(function=5,add=[epi,ek])
        print("\nFitting Branch E3")
        #fit_pi_k_e3 = ana.LatticeFit(2, dt_f=1, dt_i=1, dt=min_size_etot,
        #                          correlated=True)
        fit_pi_k_e3 = ana.LatticeFit(2, dt_f=-1, dt_i=-1, dt=2,
                                  correlated=True)
        if read_fit==False:
            start = [0.5]
            #pi_k_fitresult_e3 = fit_pi_k_e3.fit(start,e3_corr_ws,fit_e_tot,add=addT)
            pi_k_fitresult_e3 = fit_pi_k_e3.fit(start,e3_corr_ws,[[14,18]],add=addT)
            pi_k_fitresult_e3.save("%s/%s_%s_E3.npz" % (datadir,fit_pik_out,lat))
        else:
            pi_k_fitresult_e3.read("%s/%s_%s_E3.npz" % (datadir,fit_pik_out,lat))
        pi_k_fitresult_e3.print_details()
        plotter=ana.LatticePlot("%s/%s_%s_E3.pdf" %(plotdir,fit_pik_out,lat))
        plotter.set_env(ylog=False,grid=False,title=True)
        label=[r'$\pi K$ effective mass','$t$',r'$M_eff(t)$',r'E3']
        plotter.plot(e3_corr_ws,label,pi_k_fitresult_e3,fit_pi_k_e3,add=addT)
        plotter.new_file("%s/%s_%s_E3_hist.pdf" %(plotdir,fit_pik_out,lat))
        label = ["pik energy", "E$_{\pi K}$/a", "E$_{\pi K}$",'E3']
        plotter.histogram(pi_k_fitresult_e3,label,par=0)
        del plotter

    # plot all 3 correlators in one file
    #plotter=ana.LatticePlot("%s/%s_%s.pdf" %(plotdir,corr_pik_out,lat),join=True)
    #plotter.set_env(ylog=True,grid=False)
    #label=[r'$\pi K$ raw correlator','$t$',r'$C(t)$',r'E1']
    #plotter.plot(e1_corr_ws,label)
    #label=[r'$\pi K$ raw correlator','$t$',r'$C(t)$',r'E2']
    #plotter.plot(e2_corr_sub,label)
    #label=[r'$\pi K$ raw correlator','$t$',r'$C(t)$',r'E3']
    #plotter.plot(e3_corr_ws,label)
    #plotter.save()
    #del plotter
    # Compare effective masses
    #e2_corr_sub.mass()

    #m = e2_corr_sub.data[0,T/4,0] 
    #ylim = [m-0.2*m,m+0.2*m]

    #plotter=ana.LatticePlot("%s/%s_%s_mass.pdf" %(plotdir,corr_pik_out,lat),join=True)
    #plotter.set_env(ylog=False,grid=False,ylim = ylim,title=True)
    #label=[r'$\pi K$ effective mass','$t$',r'$M_{eff}$',r'E2']
    #plotter.plot(e2_corr_sub,label)
    #label=[r'$\pi K$ effective mass','$t$',r'$M_{eff}$',r'E3']
    #plotter.plot(e3_corr_ws,label)
    #plotter.save()
    #del plotter

    #pi_k_fitresult_e1.print_data(1)
    #pi_k_fitresult_e3.print_data(0)
if __name__ == '__main__':
    try:
        print("starting")
        main()
    except KeyboardInterrupt:
        pass

