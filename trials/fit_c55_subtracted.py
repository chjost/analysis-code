#!/usr/bin/python

import sys
import numpy as np
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
    fit_pik_out = "fit_pik_subtract_unit"
    fit_pik_one_fitrange = "fit_pik_pvalcut_e2"
    fit_pik_plot ="fit_pik_subtract" 
    mass_pik_plot="m_eff_pik_subtract_unit"
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

    plotter=ana.LatticePlot("%s/%s_%s.pdf" %(plotdir,pik_raw_plot,lat),join=True)
    plotter.set_env(ylog=True)
    label=["C_55", "t", "C(t)","data"]
    plotter.plot(corr,label)
    plotter.save()
    del plotter
    print(corr.data.shape)
    fit_k_single = ana.LatticeFit(9,dt_f=-1, dt_i=-1,
                                  dt=min_size_mass_k, correlated=True)
    start_single = [1., 0.3]
    if read_kfit == False:
        print("fitting kaon")
        k_fit = fit_k_single.fit(start_single, k_corr, [t_mass_k],
            add=addT)
        k_fit.save("%s/%s_%s.npz" % (datadir,fit_k_out, lat))
    else:
        k_fit = ana.FitResult.read("%s/%s_%s.npz" % (datadir,fit_k_out, lat))
    k_fit.print_data(1)
    k_fit.print_details()
    k_fit.calc_error()

    fit_pi_single = ana.LatticeFit(9,dt_f=-1, dt_i=-1,
                                   dt=min_size_mass_pi, correlated=True)
    start_single = [1., 0.3]
    if read_pifit == False:
        print("fitting kaon")
        pi_fit = fit_pi_single.fit(start_single, pi_corr, [t_mass_pi],
            add=addT)
        pi_fit.save("%s/%s_%s.npz" % (datadir,fit_pi_out, lat))
    else:
        pi_fit = ana.FitResult.read("%s/%s_%s.npz" % (datadir,fit_pi_out, lat))
    pi_fit.print_data(1)
    pi_fit.print_details()
    pi_fit.calc_error()
    corr_sub = ana.Correlators.create(corr.data,T=corr.T)
    corr_sub.subtract_pollution(pi_fit,k_fit)
    corr_sub.mass(usecosh=True)
    corr.mass(usecosh=True)
    fit_pi_k = ana.LatticeFit(2, dt_f=1, dt_i=1, dt=min_size_etot,
                              correlated=True)
    #start = [10.,10.5]
    if read_pikfit == False:
        print("fitting correlator")
        start = [0.5]
        pi_k_fit = fit_pi_k.fit(start,corr_sub,fit_e_tot,add=addT2)
        print(pi_k_fit.data[0])
        pi_k_fit.save("%s/%s_%s.npz" % (datadir,fit_pik_out,lat))
    else:
        pi_k_fit = ana.FitResult.read("%s/%s_%s.npz" % (datadir,fit_pik_out,lat))
    pi_k_fit.print_data(0)
    pi_k_fit.print_details()
    pi_k_fit.calc_error()
    #plotter = ana.LatticePlot("%s/fit_c55_sub_%s.pdf"%(plotdir,lat))
    #plotter.set_env(ylog=False,grid=False)
    #label=["M_eff subtracted", "t", "M_eff(t)","fit"]
    #print("\n This is the weighted and shifted correlator:")
    #plotter.plot(corr_sub,label,pi_k_fit,fit_pi_k,add=addT)
    #del plotter

    # plotranges
    ylow = pi_k_fit.data[0][0,0,0]-0.05*pi_k_fit.data[0][0,0,0]
    yhigh = pi_k_fit.data[0][0,0,0]+0.05*pi_k_fit.data[0][0,0,0]

    plotter=ana.LatticePlot("%s/%s_%s" %(plotdir,fit_pik_plot, lat))
    plotter.set_env(ylog=False, grid=False)
    label=["Effective Mass", "t", "m_eff(t)", "pollution divided"]
    plotter.plot(corr_sub,label,fitresult=pi_k_fit,
                 fitfunc=fit_pi_k,add=addT-1)
    plotter.set_env(ylog=False, grid=False, 
                    xlim=[fit_e_tot[0]-2, fit_e_tot[1]+2], ylim=[ylow,yhigh])
    plotter.plot(corr_sub,label,fitresult=pi_k_fit, fitfunc=fit_pi_k,
                 add=addT-1)
    del plotter

    range_mk, r_mk_shape = pi_k_fit.get_ranges()
    nbins1 = r_mk_shape[0][0]/3
    if nbins1 > 0:
        if nbins1 > 10:
          nbins = nbins1
        else:
          nbins = r_mk_shape[0][0]
    plotter=ana.LatticePlot("%s/hist_fit_pik_%s.pdf" % (plotdir, lat))
    label = ["pik energy", "E$_{\pi K}$/a", "E$_{\pi K}$"]
    plotter.histogram(pi_k_fit, label, nb_bins=nbins, par=0)
    plotter.new_file("%s/qq_fit_pik_%s.pdf" % (plotdir, lat))
    label = [r'QQ-Plot $E_{\pi K}$ %s' % lat, r'weighted $E_{\pi K}$']
    plotter.qq_plot(pi_k_fit,label,par=0)
    del plotter

    pik_mass_fit_cherry = pi_k_fit.pick_data(pval=0.4)
    pik_mass_fit_cherry.print_details()
    pik_mass_fit_cherry.save("%s/%s_%s.npz" % (datadir,fit_pik_one_fitrange,lat))
if __name__ == '__main__':
    try:
        print("starting")
        main()
    except KeyboardInterrupt:
        pass

