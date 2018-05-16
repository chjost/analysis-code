#!/usr/bin/python

# Script for fitting the single kaon correlation function for each ensemble and
# mu_s. An infile is parsed, correlator is loadedand symmetrized/bootstrapped.
# An exponential is fitted to the Correlator data with varying fitranges.
# Fits are plotted together with a histogram of the energy values and a qq-plot
# to verify gaussian distribution of the data

import sys
import numpy as np
import itertools
# Christian's packages
sys.path.append('/hiskp4/helmes/projects/analysis-code/')

import analysis2 as ana
def main():
    # parse infile
    if len(sys.argv) < 2:
        ens = ana.LatticeEnsemble.parse("kk_I1_TP0_A40.24.ini")
    else:
        ens = ana.LatticeEnsemble.parse(sys.argv[1])
    readdata = False 
    read_kfit = False
    corr_k_in ="k_charged_p0"
    corr_k_out = "corr_k"
    fit_k_out="fit_k"
    # get data from input file
    prefix = ens.get_data("path")
    print prefix
    lat = ens.name()
    nboot = ens.get_data("nboot")
    bs_bl = ens.get_data("boot_bl")
    datadir = ens.get_data("datadir")
    plotdir = ens.get_data("plotdir")
    gmax = ens.get_data("gmax")
    d2 = ens.get_data("d2")
    t_mass_k = np.int_(ens.get_data("fitmass_k"))
    min_size_mass_k = ens.get_data("tmin_mass_k")
    try:
        debug = ens.get_data("debug")
    except KeyError:
        debug = 0
    L = ens.L()
    T = ens.T()
    T2 = ens.T2()
    addT = np.ones((nboot,)) * T
    addT2 = np.ones((nboot,)) * T2

    # read correlation function
    print("read single particle corrs")
    files = ["%s/%s.dat" % (datadir,corr_k_in)]
    if readdata == False:
        k_corr = ana.Correlators(files, matrix=False)
        k_corr.sym_and_boot(nboot,bl=bs_bl,method='stationary')
        print(k_corr.shape)
        k_corr.save("%s/%s_%s.npy" % (datadir,corr_k_out , lat))
    else:
        k_corr = ana.Correlators.read("%s/%s_%s.npz" % (datadir,corr_k_out,lat))
    # fit kaon correlation function for multiple fitranges
    fit_k = ana.LatticeFit(9,dt_f=-1, dt_i=1,
                                  dt=min_size_mass_k, correlated=True)
    start = None
    if read_kfit == False:
        print("fitting kaon")
        k_fitresult = fit_k.fit(start, k_corr, [t_mass_k],
            add=addT)
        k_fitresult.save("%s/%s_%s.npz" % (datadir,fit_k_out, lat))
    else:
        k_fitresult = ana.FitResult.read("%s/%s_%s.npz" % (datadir,fit_k_out,lat))
    k_fitresult.calc_error()
    k_fitresult.print_details

    # plot results together with histograms and qq-plots
    # plotranges
    xlim = [int(T/6), t_mass_k[1]+2]
    ylow = k_fitresult.data[0][0,1,0]-0.05*k_fitresult.data[0][0,1,0]
    yhigh = k_fitresult.data[0][0,1,0]+0.05*k_fitresult.data[0][0,1,0]
    plotter=ana.LatticePlot("%s/%s_%s.pdf" % (plotdir,fit_k_out, lat))
    plotter.set_env(ylog=True, grid=False,
                   xlim=xlim)
    # plot fits
    label=[r'Kaon Fits %s' %lat, r'$C_{\pi K}(t)$', r'$t$', r'fit']
    plotter.plot(k_corr,label,fitresult=k_fitresult,fitfunc=fit_k,add=addT)

    # plot m_eff
    plotter.new_file("%s/%s_%s_meff.pdf" %(plotdir,fit_k_out,lat))
    plotter.set_env(ylog=False,grid=False,
                   xlim=xlim,ylim=[ylow,yhigh])
    label=[r'Kaon effective mass %s' %lat, r'$M_{eff,K}(t)$', r'$t$', r'data']
    k_corr.mass()
    plotter.plot(k_corr,label)

    # plot histogram
    plotter.new_file("%s/%s_%s_hist.pdf" %(plotdir,fit_k_out,lat))
    label = [r'$K$ energy', r'$E_{K}/a$', r'E$_{K}$']
    plotter.histogram(k_fitresult, label, nb_bins=None, par=1)

    #plot qq-plot
    plotter.new_file("%s/%s_%s_qq.pdf" % (plotdir, fit_k_out,lat))
    label = [r'QQ-Plot $E_{K}$ %s' % lat, r'weighted $E_{K}$']
    plotter.qq_plot(k_fitresult,label,par=1)

    del plotter

    k_fitresult.print_details()
    k_fitresult.print_data(0)
    k_fitresult.print_data(1)

if __name__ == '__main__':
    try:
        print("starting")
        main()
    except KeyboardInterrupt:
        pass
