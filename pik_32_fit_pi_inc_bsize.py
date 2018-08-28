#!/usr/bin/python

import sys
import numpy as np
import itertools
# Christian's packages
sys.path.append('/hiskp4/helmes/projects/analysis-code/')

import analysis2 as ana
# Script for fitting the single pion correlation function for each ensemble and
# mu_s. An infile is parsed, correlator is loadedand symmetrized/bootstrapped.
# An exponential is fitted to the Correlator data with varying fitranges.
# Fits are plotted together with a histogram of the energy values and a qq-plot
# to verify gaussian distribution of the data

def main():
    # parse infile
    if len(sys.argv) < 2:
        ens = ana.LatticeEnsemble.parse("kk_I1_TP0_A40.24.ini")
    else:
        ens = ana.LatticeEnsemble.parse(sys.argv[1])
    readdata = True 
    read_pifit = False
    corr_pi_in ="pi_charged_p0"
    corr_pi_out = "corr_pi_bs10000"
    fit_pi_out="fit_pi"
    # get data from input file
    prefix = ens.get_data("path")
    print prefix
    lat = ens.name()
    nboot = 10000
    bs_bl = ens.get_data("boot_bl")
    datadir = ens.get_data("datadir_pi")
    plotdir = ens.get_data("plotdir_pi")
    gmax = ens.get_data("gmax")
    d2 = ens.get_data("d2")
    t_mass_pi = np.int_(ens.get_data("fitmass_pi"))
    min_size_mass_pi = ens.get_data("tmin_mass_pi")
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
    files = ["%s/%s.dat" % (datadir,corr_pi_in)]
    if readdata == False:
        pi_corr = ana.Correlators(files, matrix=False)
        pi_corr.sym_and_boot(nboot,bl=bs_bl,method='stationary')
        print(pi_corr.shape)
        pi_corr.save("%s/%s_%s.npy" % (datadir,corr_pi_out , lat))
    else:
        pi_corr = ana.Correlators.read("%s/%s_%s.npz" % (datadir,corr_pi_out,lat))
    pi_corr.T = T
    print(pi_corr.data.shape)
    # fit pion correlation function for multiple fitranges
    fit_pi = ana.LatticeFit(9,dt_f=-1, dt_i=1,
                                  dt=min_size_mass_pi, correlated=True)
    start = None 
    if read_pifit == False:
        print("fitting pion")
        pi_fitresult = fit_pi.fit(start, pi_corr, [t_mass_pi],
            add=addT)
        pi_fitresult.save("%s/%s_%s.npz" % (datadir,fit_pi_out, lat))
    else:
        pi_fitresult = ana.FitResult.read("%s/%s_%s.npz" % (datadir,fit_pi_out,lat))

    # plot results together with histograms and qq-plots
    # plotranges
    xlim = [int(T/6), t_mass_pi[1]+2]
    ylow = pi_fitresult.data[0][0,1,0]-0.05*pi_fitresult.data[0][0,1,0]
    yhigh = pi_fitresult.data[0][0,1,0]+0.05*pi_fitresult.data[0][0,1,0]
    plotter = ana.LatticePlot("%s/%s_%s.pdf" % (plotdir,fit_pi_out, lat))
    plotter.set_env(ylog=True,grid=False,
                   xlim=xlim)
    # plot fits
    label=[r'pion Fits %s' %lat, r'$C_{\pi}(t)$', r'$t$', r'fit']
    plotter.plot(pi_corr,label,fitresult=pi_fitresult,fitfunc=fit_pi,add=addT)

    # plot m_eff
    plotter.new_file("%s/%s_%s_meff.pdf" %(plotdir,fit_pi_out,lat))
    plotter.set_env(ylog=False,grid=False,
                   xlim=xlim,ylim=[ylow,yhigh])
    label=[r'pion effective mass %s' %lat, r'$M_{eff,\pi}(t)$', r'$t$', r'data']
    pi_corr.mass()
    plotter.plot(pi_corr,label)

    # plot histogram
    plotter.new_file("%s/%s_%s_hist.pdf" %(plotdir,fit_pi_out,lat))
    label = [r'$\pi$ energy', r'$E_{\pi}/a$', r'E$_{\pi}$']
    plotter.histogram(pi_fitresult, label, nb_bins=None, par=1)

    #plot qq-plot
    plotter.new_file("%s/%s_%s_qq.pdf" % (plotdir, fit_pi_out,lat))
    label = [r'QQ-Plot $E_{\pi}$ %s' % lat, r'weighted $E_{\pi}$']
    plotter.qq_plot(pi_fitresult,label,par=1)

    del plotter

    pi_fitresult.print_details()
    pi_fitresult.print_data(0)
    pi_fitresult.print_data(1)

if __name__ == '__main__':
    try:
        print("starting")
        main()
    except KeyboardInterrupt:
        pass
