#!/usr/bin/python
import pprint
import sys
import numpy as np
from scipy.optimize import fsolve
import itertools

# Christian's packages
sys.path.append('/hiskp4/helmes/projects/analysis-code/')
import analysis2 as ana
# Takes much more arguments than foreseen in Correlator class, do not know
# where to place it
def mass_divided(data,epi,ek,T):
    _T = T
    mass = np.zeros_like(data[:,:-2])
    print("Exponential solve for symmetric correlator")
    print(mass.shape)
    for b, row in enumerate(data):
        for t in range(len(row)-2):
             mass[b, t] = fsolve(pik_div,0.5,
                                 args=(row[t],row[t+1],t,_T,epi[b],ek[b]))
    correlator_mass=ana.Correlators.create(mass,T=_T)
    return correlator_mass

def pik_pollution(t,T,epi,ek):
    pollution = np.exp(-epi*t) * np.exp(-ek*(T-t)) + np.exp(-ek*t) * np.exp(-epi*(T-t))
    return pollution

def pik_div(epik,row,row_shifted,t,T,epi,ek):
    num = np.exp(-epik*t)+np.exp(-epik*(T-t)) - pik_pollution(t,T,epi,ek)/pik_pollution(t+1,T,epi,ek) * (np.exp(-epik*(t+1))+np.exp(-epik*(T-(t+1))))
    den = np.exp(-epik*(t+1))+np.exp(-epik*(T-(t+1))) - pik_pollution(t+1,T,epi,ek)/pik_pollution(t+2,T,epi,ek) * (np.exp(-epik*(t+2))+np.exp(-epik*(T-(t+2))))
    return row/row_shifted - num/den

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
    fit_pik_out = "fit_pik_divided_unit"
    fit_pik_one_fitrange = "fit_pik_pvalcut_e3"
    fit_pik_plot ="fit_pik_divided_unit" 
    mass_pik_plot="m_eff_pik_divided_unit"
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
    corr_mass = ana.Correlators.create(corr.data,T=T)
    label=["C_55", "t", "m_eff(t)","data"]
    corr_mass.mass(usecosh=True)
    plotter.set_env(ylog=False)
    plotter.plot(corr_mass,label)
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
    corr.divide_out_pollution(pi_fit,k_fit)

    print("Plotting divided Correlator before shifting")
    plotter=ana.LatticePlot("%s/fit_c55_div_bshift_%s" %(plotdir, lat))
    plotter.set_env(ylog=True, grid=False)
    label=["C_unshifted", "t", "C(t)", "just divided"]
    plotter.plot(corr,label,add=addT-1)
    del plotter

 # Before fitting shift the correlator
    # make matrix out of corr
    corr_shift = ana.Correlators.create(corr.data)
    corr_shift.matrix=True
    corr_shift.shape = np.append(corr.data.shape,1)
    print(corr.shape)
    corr_shift.data.reshape((corr_shift.shape))
    e_k = k_fit.singularize().data[0][:,1,0]
    e_pi = pi_fit.singularize().data[0][:,1,0]

    pollution_samples = np.zeros((1500,T/2,1))
    for t in range(pollution_samples.shape[1]):
        pollution_samples[:,t,0] = pik_pollution(t,T,e_pi,e_k)
    poll = np.column_stack((ana.compute_error(pollution_samples)))
    print("Pollution is:")
    print(poll)
    print(e_k[0],e_pi[0])
    corr_shift.shift(1, shift=1,d2=0)
    # Convert again to correlator for plotting ws denotes weighted and shifted
    corr_ws = ana.Correlators.create(corr_shift.data[...,0],T=corr.T)
    corr_ws.shape = corr_ws.data.shape


    corr_ws.multiply_pollution(pi_fit,k_fit)
    plotter=ana.LatticePlot("%s/corr_c55_div_%s" %(plotdir, lat))
    plotter.set_env(ylog=False, grid=False)
    label=["C_div(t)", "t", "m_eff(t)", "pollution divided"]
    plotter.plot(corr_ws,label)
    e_pik_eff = mass_divided(corr_ws.data,e_pi,e_k,T)
    plotter.set_env(ylog=False, grid=False)
    label=["Effective Mass", "t", "m_eff(t)", "pollution divided"]
    plotter.plot(e_pik_eff,label)
    del plotter

    #-------- Fit mass plateau to WS mass -------------------------------------
    fit_pik_mass = ana.LatticeFit(2,dt_f=1, dt_i=1, dt=min_size_etot,
                                  correlated=True)
    if read_massfit == False:
        start = [0.6]
        pik_mass_fit = fit_pik_mass.fit(start,e_pik_eff,fit_e_tot,add=addT)
        pik_mass_fit.save("%s/%s_%s.npz" % (datadir,fit_pik_out,lat))
    else:
        pik_mass_fit = ana.FitResult.read("%s/%s_%s.npz" % (datadir,fit_pik_out,lat))

    pik_mass_fit.print_data()
    pik_mass_fit.print_details()

    # plotranges
    ylow = pik_mass_fit.data[0][0,0,0]-0.05*pik_mass_fit.data[0][0,0,0]
    yhigh = pik_mass_fit.data[0][0,0,0]+0.05*pik_mass_fit.data[0][0,0,0]

    plotter=ana.LatticePlot("%s/%s_%s" %(plotdir,fit_pik_plot, lat))
    plotter.set_env(ylog=False, grid=False)
    label=["Effective Mass", "t", "m_eff(t)", "pollution divided"]
    plotter.plot(e_pik_eff,label,fitresult=pik_mass_fit,
                 fitfunc=fit_pik_mass,add=addT-1)
    plotter.set_env(ylog=False, grid=False, 
                    xlim=[fit_e_tot[0]-2, fit_e_tot[1]+2], ylim=[ylow,yhigh])
    plotter.plot(e_pik_eff,label,fitresult=pik_mass_fit, fitfunc=fit_pik_mass,
                 add=addT-1)
    del plotter

    range_mk, r_mk_shape = pik_mass_fit.get_ranges()
    nbins1 = r_mk_shape[0][0]/3
    if nbins1 > 0:
        if nbins1 > 10:
          nbins = nbins1
        else:
          nbins = r_mk_shape[0][0]
    plotter=ana.LatticePlot("%s/hist_fit_pik_%s.pdf" % (plotdir, lat))
    label = ["pik energy", "E$_{\pi K}$/a", "E$_{\pi K}$"]
    plotter.histogram(pik_mass_fit, label, nb_bins=nbins, par=0)
    plotter.new_file("%s/qq_fit_pik_%s.pdf" % (plotdir, lat))
    label = [r'QQ-Plot $E_{\pi K}$ %s' % lat, r'weighted $E_{\pi K}$']
    plotter.qq_plot(pik_mass_fit,label,par=0)

    del plotter

    pik_mass_fit_cherry = pik_mass_fit.pick_data(pval=0.5)
    pik_mass_fit_cherry.print_details()
    pik_mass_fit_cherry.save("%s/%s_%s.npz" % (datadir,fit_pik_one_fitrange,lat))

if __name__ == '__main__':
    try:
        print("starting")
        main()
    except KeyboardInterrupt:
        pass


