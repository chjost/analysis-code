#!/hadron/knippsch/Enthought/Canopy_64bit/User/bin/python

import sys
import numpy as np
import itertools

import analysis2 as ana

def analyse_two_point_corr(dnames, fnames, ens, func_id, fr_min, fr_int, add,
                           nboot, plot=True, read=False, save=True, debug=0):

    """ Function shortcutting single particle fits
  
    This function uses the analysis package to, read in the correlators,
    bootstrap them and make sensible fits to the two point correlator 
  
    Parameters
    ----------
    dnames : tuple of directory names for data and plots, respectively
    fnames : tuple of filenames, suffixes added automatically. 
    ens :    string, the short name for the ensemble
    func_id : int, Which function to use for fitting, fordetails see analysis2/fit.py
    fr_min : minimal fitsize range
    fr_int : tuple of ints begin and end of fitrange
    add : additional arguments to the fit function

    Returns
    -------
    fit_res : FitResult object, details: analysis2/fit.py
    """
    #################### Prepare correlator #################### 
    print("read single particle corrs")
    files = ["%s/%s.dat" % (dnames[0],fnames[0])]
    if read == False:
        corr = ana.Correlators(files, matrix=False)
        # Symmetrization in time and bootstrapping
        corr.sym_and_boot(nboot)
        if debug > 1:
            print(corr.shape)
        corr.save("%s/%s_%s.npy" % (dnames[0], fnames[0], ens))
    else:
        corr = ana.Correlators.read("%s/%s_%s.npz" % (dnames[0], fnames[0], ens))

    ################ Fit two point ####################
    print("fitting C2")
    fit_single = ana.LatticeFit(func_id,dt_f=1, dt_i=2,
                                  dt=fr_min, correlated=True)
    # sensible default values
    start_single = [1., 0.3]
    if read == False:
        fitres = fit_single.fit(start_single, corr, fr_int,
            add=add)
        fitres.save("%s/fit_%s_%s.npz" % (dnames[0],fnames[0], ens))
    else:
        fitres = ana.FitResult.read("%s/fit_%s_%s.npz" % (dnames[0],fnames[0], ens))
    fitres.print_data(1)
    if debug > 1:
        fitres.print_details()
    fitres.calc_error()
  
   #################### Plot two point ####################  
    if plot:
        print("Plot fit of C2")
        plotter = ana.LatticePlot("%s/%s_%s.pdf" %(dnames[1],fnames[1],ens))
        label=[fnames[0], "t", "C(t)","data"]
        plotter.plot(corr, label, fitresult = fitres, 
                    fitfunc = fit_single, add=add)
        del plotter
        
    return fitres

