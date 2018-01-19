#!/usr/bin/python

import sys
import numpy as np
import itertools
# Christian's packages
sys.path.append('/hadron/helmes/projects/analysis-code/')

import analysis2 as ana

def main():
    # parse the input file
    if len(sys.argv) < 2:
        ens = ana.LatticeEnsemble.parse("kk_I1_TP0_A40.24.ini")
    else:
        ens = ana.LatticeEnsemble.parse(sys.argv[1])
    # define filenames depending on os flag
    # read settings
    readdata = False 
    read_kfit = False
    read_pifit = False
    read_pikfit = False
    read_massfit = False
    readfit = False
    plotdata = True
    plotfit = True
    corr_pi_in = "pi_charged_p0"
    corr_pi_out="corr_pi_unit"
    corr_k_in ="k_charged_p0"
    corr_k_out = "corr_k_unit"
    corr_c55_in ="pik_charged_A1_TP0_00"
    corr_c55_out = "corr_c55_unit"
    pik_raw_plot = "corr_c55_unit"
    fit_k_out="fit_k_unit"
    fit_pi_out = "fit_pi_unit"
    fit_pik_out = "fit_pik_weight_unit"
    fit_pik_plot ="fit_pik_weight_unit" 
    mass_pik_plot="m_eff_pik_weight_unit"
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
    t_mass_pi = np.int_(ens.get_data("fitmass_pi"))
    t_mass_k = np.int_(ens.get_data("fitmass_k"))
    fit_e_tot = np.int_(ens.get_data("fitetot"))
    min_size_mass_pi = ens.get_data("tmin_mass_pi")
    min_size_mass_k = ens.get_data("tmin_mass_k")
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
################################################################################
#
#       Read in correlators and symmetrize/bootstrap
#
################################################################################
    # single particle k_correlator
    print("read single particle corrs")
    files = ["%s/%s.dat" % (datadir,corr_k_in)]
    #files = ["%s/eta_charged_p0_old.dat" % datadir] 
    #files = ["%s/pi_k_corr_p0.dat" % datadir] 
    if readdata == False:
        k_corr = ana.Correlators(files, matrix=False)
        k_corr.sym_and_boot(nboot)
        #k_corr.bootstrap(nboot)
        print(k_corr.shape)
        k_corr.save("%s/%s_%s.npy" % (datadir,corr_k_out , lat))
    else:
        k_corr = ana.Correlators.read("%s/%s_%s.npz" % (datadir,corr_k_out, lat))
    # single particle correlator
    print("read single particle corrs")
    files = ["%s/%s.dat" % (datadir_pi,corr_pi_in)]
    #files = ["%s/eta_charged_p0_old.dat" % datadir] 
    #files = ["%s/pi_corr_p0.dat" % datadir] 
    if readdata == False:
        pi_corr = ana.Correlators(files, matrix=False)
        pi_corr.sym_and_boot(nboot)
        #pi_corr.bootstrap(nboot)
        print(pi_corr.shape)
        pi_corr.save("%s/%s_%s.npy" % (datadir, corr_pi_out, lat))
    else:
        pi_corr = ana.Correlators.read("%s/%s_%s.npz" % (datadir, corr_pi_out, lat))
    # single particle correlator
    print("read two particle corrs")
    files = ["%s/%s.dat" % (datadir,corr_c55_in)]
    #files = ["%s/eta_charged_p0_old.dat" % datadir] 
    #files = ["%s/pi_corr_p0.dat" % datadir] 
    if readdata == False:
        corr = ana.Correlators(files, matrix=False,conf_col=3)
        corr.sym_and_boot(nboot)
        #corr.bootstrap(nboot)
        print(corr.shape)
        corr.save("%s/%s_%s.npy" % (datadir,corr_c55_out, lat))
    else:
        corr = ana.Correlators.read("%s/%s_%s.npz" % (datadir, corr_c55_out, lat))
    print("plot C55")

    #plotter=ana.LatticePlot("%s/%s_%s.pdf" %(plotdir,pik_raw_plot,lat),join=True)
    #plotter.set_env(ylog=True)
    #label=["C_55", "t", "C(t)","data"]
    #plotter.plot(corr,label)
    #plotter.save()
    #corr_mass = ana.Correlators.create(corr.data)
    #label=["C_55", "t", "m_eff(t)","data"]
    #corr_mass.mass(usecosh=True)
    #plotter.set_env(ylog=False)
    #plotter.plot(corr_mass,label)
    #plotter.save()
    #del plotter
    #print(corr.data.shape)
################################################################################
#
#           Fit constant to 2pt correlation functions of effective mass
#
################################################################################
    # Get effective mass from numerical solve
    # k_corr.mass(usecosh=False,exp=True)
    fit_k_single = ana.LatticeFit(9,dt_f=-1, dt_i=-1,
                                  dt=min_size_mass_k, correlated=True)
    start_single = [2.,0.3]
    if read_kfit == False:
        print("fitting kaon")
        k_fit = fit_k_single.fit(start_single, k_corr, [t_mass_k],
            add=addT)
        k_fit.save("%s/%s_%s.npz" % (datadir,fit_k_out, lat))
    else:
        k_fit = ana.FitResult.read("%s/%s_%s.npz" % (datadir,fit_k_out, lat))
    k_fit.print_details()
    k_fit.calc_error()
    
    # pi_corr.mass(usecosh=False,exp=True)
    fit_pi_single = ana.LatticeFit(9,dt_f=-1, dt_i=-1,
                                   dt=min_size_mass_pi, correlated=True)
    start_single = [2.,0.3]
    if read_pifit == False:
        print("fitting kaon")
        pi_fit = fit_pi_single.fit(start_single, pi_corr, [t_mass_pi],
            add=addT)
        pi_fit.save("%s/%s_%s.npz" % (datadir,fit_pi_out, lat))
    else:
        pi_fit = ana.FitResult.read("%s/%s_%s.npz" % (datadir,fit_pi_out, lat))
    pi_fit.print_details()
    pi_fit.calc_error()

 # Before fitting shift the correlator
################################################################################
#
#           Get rid of pollution in 4pt function by subtraction
#
################################################################################
    corr.subtract_pollution(pi_fit,k_fit)

################################################################################
#                                                                              #
#           Fit constant to effective mass of subtracted 4pt function          #
#                                                                              #
################################################################################
    # Compare E_pik from correlation function... 
    fit_pi_k = ana.LatticeFit(9, dt_f=-1, dt_i=-1, dt=min_size_etot,
                              correlated=True)
    start = [10.,0.5]
    if read_pikfit == False:
        print("fitting correlator")
        pi_k_fit = fit_pi_k.fit(start,corr,fit_e_tot,add=addT)
        pi_k_fit.save("%s/%s_%s.npz" % (datadir,fit_pik_out,lat))
    else:
        pi_k_fit = ana.FitResult.read("%s/%s_%s.npz" % (datadir,fit_pik_out,lat))
    pi_k_fit.calc_error()
    pi_k_fit.print_details()
    # plot it
    plotter = ana.LatticePlot("%s/fit_c55_subtracted_%s.pdf"%(plotdir,lat))
    plotter.set_env(ylog=True,grid=False)
    label=["C_55_subtracted ", "t", "C(t)","fit"]
    plotter.plot(corr,label,pi_k_fit,fit_pi_k,add=addT)
    del plotter


    # ...and from effective mass
    corr.mass(usecosh=False,exp=True)
    fit_mass_pi_k = ana.LatticeFit(2, dt_f=-1, dt_i=-1, dt=min_size_etot,
                              correlated=True)
    start = [0.5]
    if read_pikfit == False:
        print("fitting correlator")
        pi_k_mass_fit = fit_mass_pi_k.fit(start,corr,fit_e_tot,add=addT)
        pi_k_mass_fit.save("%s/%s_meff_%s.npz" % (datadir,fit_pik_out,lat))
    else:
        pi_k_mass_fit = ana.FitResult.read("%s/%s_meff_%s.npz" % (datadir,fit_pik_out,lat))
    pi_k_mass_fit.calc_error()
    pi_k_mass_fit.print_details()
    plotter = ana.LatticePlot("%s/fit_meff_subtracted_%s.pdf"%(plotdir,lat))
    plotter.set_env(ylog=False,grid=False)
    label=["meff_subtracted ", "t", "m_eff(t)","fit"]
    plotter.plot(corr,label,pi_k_mass_fit,fit_mass_pi_k,add=addT)
    del plotter

################################################################################
#                                                                              #  
#           Plot the results                                                   #
#                                                                              #
################################################################################

    #plotter = ana.LatticePlot("%s/fit_c55_weight_%s.pdf"%(plotdir,lat))
    #plotter.set_env(ylog=True,grid=False)
    #label=["C_55 weighted", "t", "C(t)","fit"]
    #print("\n This is the weighted and shifted correlator:")
    #plotter.plot(corr_ws,label,pi_k_fit,fit_pi_k,add=addT,oldfit=masses,oldfitpar=slice(0,2))
    #del plotter

 #--#-------------- Subtract subleading term from C_{\pi K} --------------------
    ##corr_ws_sub = sub_subleading(corr_ws, pi_k_fit, masses, addT)
    #corr_ws_sub = corr_ws.sub_subleading(pi_k_fit, masses, addT)

 #--#------------- Plot the original correlator with fits and the mass
    #plotter = ana.LatticePlot("%s/c55_weight_%s.pdf"%(plotdir,lat),join=True)
    #plotter.set_env(ylog=True,grid=False)
    #label=["C_55 weighted", "t", "C(t)","bare"]
    #plotter.plot(corr,label)
    #label=["C_55 weighted", "t", "C(t)","weighted and shifted"]
    #plotter.plot(corr_ws,label)
    #plotter.save()
    #del plotter
    #plotter=ana.LatticePlot("%s/%s_%s.pdf" %(plotdir,mass_pik_plot,lat),join=True)
    #plotter.set_env(ylog=False,grid=False)
    #corr.mass(usecosh=True)
    #corr_ws.mass(usecosh=False,weight = e_k-e_pi,shift=1.)
    #corr_ws_sub.mass(usecosh=False,weight = e_k-e_pi,shift=1.)
    #label[2] = "m_eff(t)"
    #label[3] = "UW"
    #plotter.plot(corr,label)
    #label[3] = "WS"
    #plotter.plot(corr_ws,label)
    #label[3] = "WSS"
    #plotter.plot(corr_ws_sub,label)
    #plotter.save()
    #del plotter


if __name__ == '__main__':
    try:
        print("starting")
        main()
    except KeyboardInterrupt:
        pass


