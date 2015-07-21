################################################################################
#
# Author: Bastian Knippschild (b.knippschild@gmx.de)
# Date:   Februar 2015
#
# Copyright (C) 2015 Bastian Knippschild
# 
# This program is free software: you can redistribute it and/or modify it under 
# the terms of the GNU General Public License as published by the Free Software 
# Foundation, either version 3 of the License, or (at your option) any later 
# version.
# 
# This program is distributed in the hope that it will be useful, but WITHOUT 
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS 
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with tmLQCD. If not, see <http://www.gnu.org/licenses/>.
#
################################################################################
#
# Function: Functions to fit and plot.
#
# For informations on input parameters see the description of the function.
#
################################################################################

import sys
from scipy.optimize import leastsq
import scipy.stats
import numpy as np
import analyze_fcts as af
from .plot import *
from .input_output import read_fitresults, write_fitresults

def fitting(fitfunc, X, Y, start_parm, E_single=None, correlated=True, verbose=True):
    """A function that fits a correlation function.

    This function fits the given function fitfunc to the data given in X and Y.
    The function needs some start values, given in start_parm, and can use a
    correlated or an uncorrelated fit.

    Args:
        fitfunc: The function to fit to the data.
        X: The time slices.
        Y: The bootstrap samples of the data.
        start_parm: The starting parameters for the fit.
        E_single: single particle energies entering the ratio R
        correlated: Flag to use a correlated or uncorrelated fit.
        verbose: Controls the amount of information written to the screen.

    Returns:
        The function returns the fitting parameters, the chi^2 and the p-value
        for every bootstrap sample.
    """
    if E_single is None:
        errfunc = lambda p, x, y, error: np.dot(error, (y-fitfunc(p,x)).T)
    else:
        errfunc = lambda p, x, y, e, error: np.dot(error, (y-fitfunc(p,x,e)).T)
    # compute inverse, cholesky decomposed covariance matrix
    if not correlated:
        cov = np.diag(np.diagonal(np.cov(Y.T)))
    else:
        cov = np.cov(Y.T)
    cov = (np.linalg.cholesky(np.linalg.inv(cov))).T

    # degrees of freedom
    dof = float(Y.shape[1]-len(start_parm)) 
    # create results arrays
    res = np.zeros((Y.shape[0], len(start_parm)))
    chisquare = np.zeros(Y.shape[0])
    # The FIT to the boostrap samples
    if E_single is None:
        for b in range(0, Y.shape[0]):
            p,cov1,infodict,mesg,ier = leastsq(errfunc, start_parm, 
                args=(X, Y[b,:], cov), full_output=1, factor=0.1)
            chisquare[b] = float(sum(infodict['fvec']**2.))
            res[b] = np.array(p)
    else:
        for b in range(0, Y.shape[0]):
            p,cov1,infodict,mesg,ier = leastsq(errfunc, start_parm,
                                       args=(X, Y[b,:],E_single[b], cov ), 
                                       full_output=1, factor=0.1)
            chisquare[b] = float(sum(infodict['fvec']**2.))
            res[b] = np.array(p)
    # calculate mean and standard deviation
    res_mean, res_std = af.calc_error(res)
    chi2 = np.median(chisquare)
    # p-value calculated
    pvals = 1. - scipy.stats.chi2.cdf(chisquare, dof)

    # The fit to the mean value
    y = np.mean(Y, axis=0)
    if E_single is None:
        p,cov1,infodict,mesg,ier = leastsq(errfunc, start_parm, \
                                   args=(X, y, cov), full_output=1)
    else:
        e_single = np.mean(E_single)
        p,cov1,infodict,mesg,ier = leastsq(errfunc, start_parm, \
                                   args=(X, y, e_single, cov), full_output=1)
    # writing results to screen
    if verbose:
        if correlated:
            print("fit results for a correlated fit:")
        else:
            print("fit results for an uncorrelated fit:")
        print("degrees of freedom: %f\n" % dof)
        
        print("bootstrap fit:")
        for rm, rs in zip(res_mean, res_std):
            print("  %.6e +/- %.6e" % (rm, rs))
        print("Chi^2/dof: %.6e +/- %.6e\n" % (chi2/dof,
              np.std(chisquare)/dof))

        print("mean value fit:")
        for rm, rs in zip(p, res_std):
            print("  %.6e +/- %.6e" % (rm, rs))
        print("Chi^2/dof: %.6e +/- %.6e\n" % (float(sum(infodict['fvec']**2.) /
              dof), np.std(chisquare)/dof))

        print("original data fit:")
        for rm, rs in zip(res[0], res_std):
            print("  %.6e +/- %.6e" % (rm, rs))
        print("Chi^2/dof: %.6e +/- %.6e" % (chisquare[0]/dof, np.std(chisquare)
              /dof))
        print("p-value: %lf" % pvals[0]) 

    return res, chisquare, pvals

def quantile_1D(data, weights, quantile):
    ind_sort = np.argsort(data)
    sort_data = data[ind_sort]
    sort_weig = wheights[ind_sort]
    Sn = np.cumsum(sort_weig)
    Pn = (Sn - 0.5*sort_weig) / np.sum(sort_weig)
    return np.interp(quantile, Pn, sort_data)

#def fitting_range(fitfunc, X, Y, start_parm, correlated=True, verbose=True):
#    """A function that fits a correlation function for different fit ranges.
#
#    This function fits the given function fitfunc to the data given in X and Y.
#    The function needs some start values, given in start_parm, and can use a
#    correlated or an uncorrelated fit. Fits are performed for many different
#    fit ranges.
#
#    Args:
#        fitfunc: The function to fit to the data.
#        X: The time slices.
#        Y: The bootstrap samples of the data.
#        start_parm: The starting parameters for the fit.
#        correlated: Flag to use a correlated or uncorrelated fit.
#        verbose: Controls the amount of information written to the screen.
#
#    Returns:
#    """
#    # vary the lower and upper end of the fit range
#    for lo in range(int(Y.shape[1]/4), Y.shape[1]-5):
#        for up in range(lo+5, X.shape[1]):
#            # fit the data
#            res, chi2, pval=fitting(fitfunc, X[lo:up], Y[:,lo:up], start_params,
#                                    correlated=correlated, verbose=False)
#            # calculate the weight
#            weight = ((1. - 2*np.abs(pval - 0.5)) * (1.0))**2
#            # calculate weighted median
#            median = quantile_1D(res[:,1], weight, 0.5)
#
#            # print some result on screen
#            print("%2d-%2d: p-value %.7lf, chi2/dof %.7lf, E %.7lf" % (lo, up,
#                  pval[0], chi2[0]/(len(X[lo:up])-len(start_params)),median))

#def scan_fit_range(fitfunc, X, Y, start_params, correlated=True, verbose=False):
#    """Fits the fitfunction to the data for different fit ranges and prints the
#       result.
#
#       Args:
#           fitfunc: The function to fit.
#           X: The time slices.
#           Y: The bootstrap samples of the data.
#           start_params: The start parameters for the fit.
#           correlated: Correlated or uncorrelated fit.
#           verbose: Verbosity of the fit function.
#
#       Returns:
#           Nothing.
#    """
#    ## vary the lower end of the fit range
#    #for lo in range(int(Y.shape[1]/4), Y.shape[1]-5):
#    #    # vary the upper end of the fit range
#    #    for up in range(lo+5, Y.shape[1]):
#    # vary the lower end of the fit range
#    for lo in range(10, 16):
#        # vary the upper end of the fit range
#        for up in range(20, 24):
#            # fir the data
#            res, chi2, pval=fitting(fitfunc, X[lo:up], Y[:,lo:up], start_params,
#                                    correlated=correlated, verbose=verbose)
#            # print some result on screen
#            print("%2d-%2d: p-value %.7lf, chi2/dof %.7lf, E %.7lf" % (lo, up,
#                  pval[0], chi2[0]/(len(X[lo:up])-len(start_params)),res[0,-1]))
#
#    return

def set_fit_interval(_data, lolist, uplist, intervalsize):
    """Initialize intervals to fit in with borders given for every principal
    correlator

    Args: 
        data: The lattice results to fit to. Necessary to obtain the number of
              gevp-eigenvalues.
        lolist: List of lower interval borders for every gevp-eigenvalue.
        uplist: List of upper interval borders for every gevp-eigenvalue.
        intervallsize: Minimal number of points to be contained in the 
                interval

    Returns:
        fit_intervals: list of tuples (lo, up) for every gevp-eigenvalue.
    """
    data = np.atleast_3d(_data)
    ncorr = data.shape[2]
    fit_intervals = []
    for _l in range(ncorr):
        fit_intervals.append([])
        if uplist[_l] > data.shape[1] - 1:
            print("upper bound for fit greater than time extent of data")
            print("using data time extend!")
            uplist[_l] = data.shape[1] - 1
        for lo in range(lolist[_l], uplist[_l] + 1):
            for up in range(lolist[_l], uplist[_l] + 1):
                # the +2 comes from the fact that the interval contains one
                # more number than the difference between the boundaries and
                # because python would exclude the upper boundary but we
                # include it explicitly
                if (up - lo + 2) > intervalsize:
                    fit_intervals[_l].append((lo, up))

    return fit_intervals

def genfit_comb(data, fitint_data, fitint_par, fitfunc, start_params, 
                par, tmin, lattice, _label, path=".plots/", 
                plotlabel="corr", verbose=False):
    """Fit and plot a function. With varying parameter, determined in a previous
    fit
    
    Args:
        data: The correlation functions.
        fitint_data: List of intervals for the fit of the functions.
        fitint_par: List of intervals for the varying parameter
        fitfunc: The function to fit to the data.
        start_params: The starting parameters for the fit function.
        par: the varying parameter
        tmin: Lower bound of the plot.
        lattice: The name of the lattice, used for the output file.
        label: Labels for the title and the axis.
        path: Path to the saving place of the plot.
        plotlabel: Label for the plot file.
        verbose: Amount of information printed to screen.

    Returns:
        res: Result of the fit to each bootstrap sample for each combination of
            fit ranges.
        chi2: Chi^2 for every fit
        pval: p-value for every fit.
    """
    # ensure at least 3d 
    data = np.atleast_3d(data)
    label = _label
    # init variables
    # nboot: number of bootstrap samples
    # npar: number of parameters to fit to
    # ncorr: number of correlators
    # ncorr_par: number of correlators of the varying parameter
    # nint_data: number of fit ranges for data
    # nint_par: number of fit ranges of the varying parameter
    nboot = data.shape[0]
    T2 = data.shape[1]
    ncorr = data.shape[2]
    ncorr_par = len(par)
    npar = len(start_params)
    nint_data = [len(fitint) for fitint in fitint_data]
    nint_par = [len(fitint) for fitint in fitint_par]
    if (ncorr is not len(nint_data)) or (ncorr_par is not len(nint_par)):
         print("data and fit intervals do not fit together!!")
         print("Returning nothing")
         return None
    if verbose:
        print(ncorr)
        print(nint_data)
        print(ncorr_par)
        print(nint_par)
    # initialize empty arrays with correct shape
    res, chi2, pval = [], [], []
    # loop over the correlators of the parameter
    for k in range(ncorr_par):
        res.append([])
        chi2.append([])
        pval.append([])
        # loop over the correlators of the data
        for l in range(ncorr):
            res[k].append(np.zeros((nboot, npar, nint_data[l], nint_par[k])))
            chi2[k].append(np.zeros((nboot, nint_data[l], nint_par[k])))
            pval[k].append(np.zeros((nboot, nint_data[l], nint_par[k])))
    # set fit x data for fit
    tlist = np.linspace(0., float(T2), float(T2), endpoint=False)
    # outputfile for the plot
    corrplot = PdfPages("%s/fit_%s_%s.pdf" % (path,plotlabel,lattice))
    # check the labels
    if len(label) < 3:
        print("not enough labels, using standard labels.")
        label = ["fit", "time", "C(t)", "", ""]
    if len(label) < 4:
        label.append("data")
    if len(label) < 5:
        label.append("")
    label_save = label[0]
    # loop over the correlation functions of the data
    for l in range(ncorr):
        # setup
        mdata, ddata = af.calc_error(data[:,:,l])
        # loop over the fit intervals
        for i in range(nint_data[l]):
            lo, up = fitint_data[l][i]
            if verbose:
                print("Interval [%d, %d]" % (lo, up))
                print("correlator %d" % l)
            # loop over the varying parameter and its fit intervals
            for k in range(ncorr_par):
                for j in range(nint_par[k]):
                    # fit the energy and print information
                    if verbose:
                        print("fitting correlation function")
                    res[k][l][:,:,i,j], chi2[k][l][:,i,j], pval[k][l][:,i,j] = \
                        fitting(fitfunc, tlist[lo:up+1], data[:,lo:up+1,l],
                                start_params, E_single = par[k][:,:,j], 
                                verbose=False)
                    if verbose:
                        print("p-value %.7lf\nChi^2/dof %.7lf\nresults:"
                              % (pval[k][l][0,i,j], chi2[k][l][0,i,j]/(
                                 (up - lo + 1) - npar)))
                        for p in enumerate(res[k][l][0,:,i,j]):
                            print("\tpar %d = %lf" % p)
                        print(" ")
                    mres, dres = af.calc_error(res[k][l][:,:,i,j])

                    # set up the plot labels
                    fitlabel = r"fit %d:%d\nm$_{\pi}$ = %f" % (lo, up, par[k][0,0,j])
                    title = "%s, %s, pc %d, [%d, %d]" % (label_save, lattice, l, lo, up)
                    label[0] = title
                    label[4] = fitlabel

                    # plot the original data and the fit for every fit range
                    if verbose:
                        print("plotting")
                    corr_fct_with_fit(tlist, data[0,:,l], ddata, fitfunc,
                        (mres, par[k][0,0,j]), [tmin,T2], label, corrplot,
                        logscale=False,fitrange=fitint_data[l][i])
    corrplot.close()
    return res, chi2, pval

def genfit(_data, fit_intervals, fitfunc, start_params, tmin, lattice, d, label,
            path=".plots/", plotlabel="corr", olddata=None, verbose=True):
    """Fit and plot the correlation function.
    
    Args:
        _data: The correlation functions.
        fit_intervalls: List of intervalls for the fit for the different
              correlation functions.
        fitfunc: The function to fit to the data.
        start_params: The starting parameters for the fit function.
        tmin: Lower bound of the plot.
        lattice: The name of the lattice, used for the output file.
        d: The total momentum of the reaction.
        label: Labels for the title and the axis.
        path: Path to the saving place of the plot.
        plotlabel: Label for the plot file.
        olddata: if not None, reuses old data at the location specified, if possible
        verbose: Amount of information printed to screen.

    Returns:
        res: Result of the fit to each bootstrap sample.
        chi2: Chi^2 for every fit
        pval: p-value for every fit.
    """
    data = np.atleast_3d(_data)
    # init variables
    nboot = data.shape[0]
    T2 = data.shape[1]
    ncorr = data.shape[2]
    npar = len(start_params)
    d2 = np.dot(d,d)
    ninter = [len(fitint) for fitint in fit_intervals]
    # initialize empty arrays
    res = []
    chi2 = []
    pval = []
    # initialize array for every principal correlator
    for _l in range(ncorr):
        res.append(np.zeros((nboot, npar, ninter[_l])))
        chi2.append(np.zeros((nboot, ninter[_l])))
        pval.append(np.zeros((nboot, ninter[_l])))
    # read old data and incorporate if possible
    if olddata:
        try:
            print("reading old data from %s" % olddata)
            _ranges, _par, _chi2, _pvals = read_fitresults(olddata)
            # sanity checks
            if len(_par) is not ncorr:
                print("old data has different ncorr")
                sys.exit(-15)
            # TODO(CJ):assumes all corr have same nboot
            if _par[0].shape[0] != nboot:
                print("old data (%d) has different nboot (%d)" % (_par[0].shape[0],nboot))
                sys.exit(-15)
            if _par[0].shape[1] != npar:
                print("old data has different number of parameter")
                sys.exit(-15)
            if _par[0].shape[1] != npar:
                print("old data has different number of parameter")
                sys.exit(-15)
        except IOError as e:
            print("could not read old data: %s" % e.message)
            print("continuing")
            olddata=None
    # set fit data
    tlist = np.linspace(0., float(T2), float(T2), endpoint=False)
    # outputfile for the plot
    corrplot = PdfPages("%s/fit_%s_%s_TP%d.pdf" % (path,plotlabel,lattice,d2))
    # check the labels
    if len(label) < 3:
        print("not enough labels, using standard labels.")
        label = ["fit", "time", "C(t)", "", ""]
    if len(label) < 4:
        label.append("data")
        label.append("")
    if len(label) < 5:
        label.append("")
    label_save = label[0]
    for _l in range(ncorr):
        # setup
        mdata, ddata = af.calc_error(data[:,:,_l])
        for _i in range(ninter[_l]):
            lo, up = fit_intervals[_l][_i]
            dofit=True
            if verbose:
                print("Interval [%d, %d]" % (lo, up))
                print("correlator %d" % _l)

            # check if already in old data
            if olddata:
                #print(_ranges[_l])
                #print(fit_intervals[_l][_i])
                for ind, v in enumerate(_ranges[_l]):
                    if v[0] == lo and v[1] == up:
                        print("found match at index %d" % ind)
                        res[_l][:,:,_i] = _par[_l][:,:,ind]
                        chi2[_l][:,_i] = _chi2[_l][:,ind]
                        pval[_l][:,_i] = _pvals[_l][:,ind]
                dofit=False
            if dofit:
                # fit the energy and print information
                if verbose:
                    print("fitting correlation function")
                    print(tlist[lo:up+1])
                res[_l][:,:,_i], chi2[_l][:,_i], pval[_l][:,_i] = fitting(fitfunc, 
                        tlist[lo:up+1], data[:,lo:up+1,_l], start_params, verbose=False)
            if verbose:
                print("p-value %.7lf\nChi^2/dof %.7lf\nresults:"
                      % (pval[_l][ 0, _i], chi2[_l][0,_i]/( (up - lo + 1) -
                                                           len(start_params))))
                for p in enumerate(res[_l][0,:,_i]):
                    print("\tpar %d = %lf" % p)
                print(" ")

            mres, dres = af.calc_error(res[_l][:,:,_i])

            # set up the plot labels
            fitlabel = "fit %d:%d" % (lo, up)
            title="%s, %s, TP %d, pc %d, [%d, %d]" % (label_save, lattice, d2, 
                                                      _l, lo, up)
            label[0] = title
            label[4] = fitlabel

            # plot the data and the fit
            if verbose:
                print("plotting")
            plot_data_with_fit(tlist, data[0,:,_l], ddata, fitfunc, mres,
                                   [tmin,T2], label, corrplot, logscale=False,
                                   fitrange=fit_intervals[_l][_i], pval=pval[_l][0,_i])
    corrplot.close()
    return res, chi2, pval

#def fit_ratio_samples(X, Y, start_parm, E_single, correlated=True, verbose=True):
#    """Fit the ratio to every bootstrap sample using the sample specific single
#    particle energy
#
#    Args: 
#        X: The time slices.
#        Y: The bootstrap samples of the data.
#        start_parm: The starting parameters for the fit.
#        E_single: The single particle energies to be replaced in the fit
#                  function
#        correlated: Flag to use a correlated or uncorrelated fit.
#        verbose: Controls the amount of information written to the screen.
#    Returns:
#        The function returns the fitting parameters, the chi^2 and the p-value
#        for every bootstrap sample.
#    """
#    # Set up globla stuff
#    T = Y.shape[1]
#    res = []
#    chi2 = []
#    pval = []
#    # Fit the correlation function sample by sample with a modified fitfunction
#    if Y.shape[0] != len(E_single):
#        print("Bootstrap Samples do not match single Energy results")
#        return
#    else:
#        for i in range(0,len(E_single)):
#            ratio = lambda p, t : p[0]*(np.cosh(p[1]*(t-T/2))+np.sinh(p[1]*(t-T/2))/(np.tanh(2*E_single[i]*(t-T/2))))
#            res_tmp, chi2_tmp, pval_tmp = fitting(ratio, X, Y[i], start_parm,
#                    correlated, verbose)
#            append(res, res_tmp[1])
#            append(chi2, chi2_tmp)
#            append(pval, pval_tmp)
#    return res, chi2, pval

def compute_weight(corr, params):
    """compute the weights for the histogram

    Args:
        corr: the correlation function
        params: the p-values for each correlator

    Returns:
        list of weigths if len(params)!=0, else empty list
    """
    errors = np.std(corr, axis=1)
    max_err = np.amax(errors)
    weights = []
    if len(params) != 0:
        for i in range(0, params.shape[0]):
            w = (1.-2*abs(params[i,1]-0.5))*max_err/errors[i]
            weights.append(w**2)
    return weigths

