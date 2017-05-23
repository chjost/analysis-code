#!/hadron/knippsch/Enthought/Canopy_64bit/User/bin/python

import sys
import numpy as np
import itertools
import analysis2 as ana

def weight_scalar_corr(data, weight_fac, shift):
    """ Wrapper for shifitng and weighting a scalar correlator
    
    Correlator data gets turned into a matrix, shifted and weighted and
    converted back to a scalar correlation function, again.
    """
    corr_shift = ana.Correlators.create(data)
    corr_shift.matrix = True
    corr_shift.shape = np.append(data.shape, 1)
    corr_shift.data.reshape(corr_shift.shape)
    corr_shift.shift(1, mass=weight_fac, shift=shift, d2=0)
    corr_ws = ana.Correlators.create(corr_shift.data[(Ellipsis, 0)])
    corr_ws.shape = corr_ws.data.shape
    return corr_ws


def analyse_four_point_corr(dnames, fnames, ens, func_id, fr_min, fr_int, add, nboot, weight=None, shift=1, oldfitres=None, plot=True, read=False, save=True, debug=0):
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
    corr : Correlator object, symmetrised and bootstrapped,
           details: analysis2/Correlators.py
    fitres : FitResult object, details: analysis2/fit.py
    """
    print 'read single particle corrs'
    files = ['%s/%s.dat' % (dnames[0], fnames[0])]
    if read == False:
        corr = ana.Correlators(files, matrix=False)
        corr.sym_and_boot(nboot)
        if debug > 1:
            print corr.shape
        corr.save('%s/%s_%s.npy' % (dnames[0], fnames[0], ens))
    else:
        corr = ana.Correlators.read('%s/%s_%s.npz' % (dnames[0], fnames[0], ens))
    if weight is not None:
        if corr.matrix is False:
            print corr.shape
            corr = weight_scalar_corr(corr.data, weight, shift)
        else:
            corr.shift(1, mass=weight, shift=shift, d2=0)
    if debug > 1:
        oldfitres.print_details()
    fit_four_pt = ana.LatticeFit(func_id, dt_f=2, dt=fr_min, correlated=True)
    if read == False:
        start = [
         5.0, 0.5, 10.0]
        fitres = fit_four_pt.fit(start, corr, fr_int, add=add, oldfit=oldfitres, oldfitpar=slice(0, 2))
        fitres.save('%s/%s_%s.npz' % (dnames[0], fnames[1], ens))
    else:
        fitres = ana.FitResult.read('%s/%s_%s.npz' % (dnames[0], fnames[1], ens))
    fitres.print_data(1)
    if debug > 1:
        fitres.print_data(0)
        fitres.print_data(2)
        fitres.print_details()
    fitres.calc_error()
    return (
     corr, fitres)
# okay decompiling four_point_corr.pyc
