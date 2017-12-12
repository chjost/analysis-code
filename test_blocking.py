#!/hadron/knippsch/Enthought/Canopy_64bit/User/bin/python

import sys
import numpy as np
import analysis2 as ana

def main():
####################################################
# parse the input file and setup parameters
#####################################################
    if len(sys.argv) < 2:
        ens = ana.LatticeEnsemble.parse("kk_I1_TP0_A40.24.ini")
    else:
        ens = ana.LatticeEnsemble.parse(sys.argv[1])

    # read settings
    readsingledata = True
    readsinglefit = True
    plotsingle = False 
    readtwodata = True
    readtwofit = False
    plottwo = False

    # get data from input file
    prefix = ens.get_data("path")
    print prefix
    lat = ens.name()
    nboot = ens.get_data("nboot")
    datadir = ens.get_data("datadir")
    plotdir = ens.get_data("plotdir")
    gmax = ens.get_data("gmax")
    d2 = ens.get_data("d2")
    try:
        debug = ens.get_data("debug")
    except KeyError:
        debug = 0
    T = ens.T()
    T2 = ens.T2()
    addT = np.ones((nboot,)) * T
    addT2 = np.ones((nboot,)) * T2

    # set up fit ranges (t_m is for mass, t_r for ratio)
    # initialize as a list with numbers of correlators as length
    t_mass = [np.int_(ens.get_data("fitmass_k")) for i in range(4)]
    #t_ratio = [np.int_(ens.get_data("fitratio")) for i in range(4)]
    min_size_mass = ens.get_data("tmin_mass_k")
    #min_size_ratio = ens.get_data("tmin_ratio")
#######################################################################
# Begin calculation
#######################################################################

    # single particle correlator
    print("read single particle corrs")
    #files = ["%s/pi_corr_p%d.dat" % (prefix, d) for d in range(4)]
    files = ["%s/k_charged_p%d.dat" % (datadir, d) for d in range(1)]
    picorr = ana.Correlators(files, matrix=False,conf_col = 2)
    picorr_b1 = ana.Correlators(files, matrix=False,conf_col = 2)
    picorr_b2 = ana.Correlators(files, matrix=False,conf_col = 2)
    picorr_b3 = ana.Correlators(files, matrix=False,conf_col = 2)
    print(picorr.data)
    picorr.bootstrap(nboot)
    picorr_b1.bootstrap(nboot,blocking=True,bl=2)
    picorr_b2.bootstrap(nboot,blocking=True,bl=4)
    picorr_b3.bootstrap(nboot,blocking=True,bl=8)
    data_arr = np.dstack((picorr.data,picorr_b1.data,picorr_b2.data,picorr_b3.data))
    twopoint = ana.Correlators.create(data_arr)
    print(twopoint.data.shape)
    #fit_single = ana.LatticeFit(0, dt_i=1, dt_f=-1, dt=min_size_mass, debug=0)
    #print("fitting")
    #start_single = [1.,0.3]
    #twofit = fit_single.fit(start_single, twopoint, t_mass, corrid="epi", add=addT)
    #twofit.print_data(1)
    #twofit.print_details()
    #print("comparing data with blocklengths 0, 1, 2, 3")
    #collapsed = twofit.singularize()
    #collapsed.print_data(1)

    # plot the relative error of all correlation functions into one file
    print("plotting")
    plotter = ana.LatticePlot("%s/test_blocking_k_%s.pdf" % (plotdir,
      lat),join=True)
    plotter.set_env(ylog=True,grid=False)
    datalbl = ["unblocked","blocklength=2","blocklength=4","blocklength=8"]
    label = ["single particle", "t", "C(t)", datalbl]
    plotter.plot(twopoint, label, add=addT, debug=debug)
    #plotter.plot(picorr, label, add=addT, debug=debug)
    #plotter.plot(picorr_b1, label, add=addT, debug=debug)
    #plotter.plot(picorr_b2, label, add=addT, debug=debug)
    #plotter.plot(picorr_b3, label, add=addT, debug=debug)
    plotter.save()
    # now plot the relative errors
    plotter.set_env(ylog=False,grid=False,xlim=[18,22],ylim=[0.004,0.008])
    label = ["single particle", "t", "dC(t)/C(t)", datalbl]
    plotter.plot(twopoint, label, add=addT, debug=debug, rel=True)
    plotter.save()
    twopoint.mass()
    label = ["single particle", "t", "M_eff(t)", datalbl]
    plotter.set_env(ylog=False,grid=False)
    plotter.plot(twopoint,label,add=addT,debug=debug)
    plotter.save()
    del plotter
    
    # two particle correlator
    #print("read two particle corrs")
    ##files = ["%s/pi_corr_p%d.dat" % (prefix, d) for d in range(4)]
    #files = ["%s/kk_charged_A1_TP0_00.dat" %datadir]
    #kkcorr = ana.Correlators(files, matrix=False,conf_col = 2)
    #kkcorr_b1 = ana.Correlators(files, matrix=False,conf_col = 2)
    #kkcorr_b2 = ana.Correlators(files, matrix=False,conf_col = 2)
    #kkcorr_b3 = ana.Correlators(files, matrix=False,conf_col = 2)
    #kkcorr.sym_and_boot(nboot)
    #kkcorr_b1.sym_and_boot(nboot,blocking=True,bl=2)
    #kkcorr_b2.sym_and_boot(nboot,blocking=True,bl=3)
    #kkcorr_b3.sym_and_boot(nboot,blocking=True,bl=4)

    #data_arr = np.dstack((kkcorr.data,kkcorr_b1.data,kkcorr_b2.data,kkcorr_b3.data))
    #fourpoint = ana.Correlators.create(data_arr)
    #print(fourpoint.data.shape)
    #
    ## build ratio
    #ratio = fourpoint.ratio(twopoint, ratio=2 )
    #fit_ratio = ana.LatticeFit(1, dt=min_size_ratio, dt_i=1, dt_f=1, xshift=0.5,
    #    debug=0)
    #start_ratio = [1.8, 0.002]
    ## ratiofit
    #print("fitting")
    #ratiofit = fit_ratio.fit(start_ratio, ratio, t_ratio,
    #        corrid="R", add=addT, oldfit=collapsed, oldfitpar=1)
    #ratiofit.print_data(0)
    #ratiofit.print_data(1)
    #print("plotting")
    #plotter = ana.LatticePlot("%s/test_blocking_kk_%s.pdf" % (plotdir,
    #  lat),join=True)
    #plotter.set_env(ylog=False)
    #datalbl = ["unblocked","blocklength=2","blocklength=3","blocklength=4"]
    #label = ["two particle", "t", "C(t)", datalbl]
    #plotter.plot(fourpoint, label, add=addT, debug=debug)
    #plotter.save()
    ## now plot the relative errors
    #label = ["two particle", "t", "C(t)", datalbl]
    #plotter.plot(fourpoint, label, add=addT, debug=debug, rel=True)
    #plotter.save()
    #del plotter
if __name__ == '__main__':
    try:
        print("starting")
        main()
    except KeyboardInterrupt:
        pass
