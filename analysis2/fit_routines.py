"""
Routines for fitting
"""

import os
import itertools
from scipy.optimize import leastsq
import scipy.stats
import numpy as np

from functions import compute_error
from fit import FitResult

def fit(fitfunc, start, corr, ranges):
    """Fits fitfunc to a Correlators object.

    The predefined functions describe a single particle correlation
    function, a ratio of single and two-particle correlation
    functions and a constant function.

    Parameters
    ----------
    fitfunc : callable
        Choose between three predefined functions or an own
        fit function.
    start : float or sequence of floats
        The start parameters for the fit.
    corr : Correlators
        A correlators object with the data.
    ranges : sequence of ints or sequence of sequences of int
        The ranges in which to fit, either one range for all or
        one range for each data set in corr.
    """
    if isinstance(ranges[0], (tuple, list)):
        raise NotImplementedError("different ranges for different correlators")
    shape = corr.shape
    # calculate the fit ranges
    fit_ranges = calculate_ranges(ranges, shape)
    nranges = [len(ran) for ran in fit_ranges]
    ncorr = len(nranges)
    # prepare X data
    X = np.linspace(0., float(shape[1]), shape[1], endpoint=False)

    # prepare storage for results
    fitres = FitResult()
    fitres.set_ranges(fit_ranges)
    # generate the shapes for the fit results
    fit_shapes = [(shape[0], len(start), nran) for nran in nranges]
    chi_shapes = [(shape[0], nran) for nran in nranges]
    fitres.create_empty(fit_shapes, chi_shapes, ncorr)


def fit_comb(self, fitfunc, start, corr, ranges, oldfit, oldfitpar=None):
    """Fits fitfunc to a Correlators object.

    The predefined functions describe a single particle correlation
    function, a ratio of single and two-particle correlation
    functions and a constant function.

    Parameters
    ----------
    fitfunc : callable
        Choose between three predefined functions or an own
        fit function.
    start : float or sequence of floats
        The start parameters for the fit.
    corr : Correlators
        A correlators object with the data.
    ranges : sequence of ints or sequence of sequences of int
        The ranges in which to fit, either one range for all or
        one range for each data set in corr.
    oldfit : None or FitResult, optional
        Reuse the fit results of an old fit for the new fit.
    oldfitpar : None, int or sequence of int, optional
        Which parameter of the old fit to use, if there is more than one.
    """
    oldranges = oldfit.get_ranges()
    fit_ranges = calculate_ranges(ranges, corr.shape, oldranges)

def calculate_ranges(ranges, shape, oldshape=None, step=2, min_size=4):
    """Calculates the fit ranges.

    Parameters
    ----------
    ranges : sequence of ints or sequence of sequences of int
        The ranges in which to fit, either one range for all or
        one range for each data set in corr.
    shape : tuple
        The shape of the data.
    oldshape : sequence, optional
        The fitranges of a fit before.
    step : int, optional
        The steps in the loops.
    min_size : int, optional
        The minimal size of the interval.

    Returns
    -------
    sequence
        The fit ranges.
    sequence
        The "shape" of the ranges.
    """
    ncorr = shape[-1]
    # check if ranges makes sense
    if isinstance(ranges[0], int):
        # assume one range for every correlator
        r_tmp = []
        # check if we exceed the time extent
        if ranges[1] > shape[1] - 1:
            ranges[1] = shape[1] - 1
        for lo in range(ranges[0], ranges[1]+1, step):
            for up in range(ranges[0], ranges[1]+1, step):
                # the +2 comes from the fact that the interval contains one
                # more number than the difference between the boundaries and
                # because python would exclude the upper boundary but we
                # include it explicitly
                if (up - lo + 2) > min_size:
                    r_tmp.append((lo, up))
        fit_ranges = [r_tmp for i in range(ncorr)]
        shape = [[len(r_tmp)] * ncorr]
    else:
        # one fitrange for every correlator
        if len(ranges) != ncorr:
            raise ValueError("number of ranges and correlators is incompatible")
        fit_ranges = []
        for ran in ranges:
            fit_ranges.append([])
            # check if we exceed the time extent
            if ran[1] > shape[1] - 1:
                ran[1] = shape[1] - 1
            for lo in range(ran[0], ran[1]+1, step):
                for up in range(ran[0], ran[1]+1, step):
                    # the +2 comes from the fact that the interval contains one
                    # more number than the difference between the boundaries and
                    # because python would exclude the upper boundary but we
                    # include it explicitly
                    if (up - lo + 2) > min_size:
                        fit_ranges[-1].append((lo, up))
        shape = [[len(ran) for ran in fit_ranges]]
    if oldshape is not None:
        r_tmp = []
        for item in itertools.product(*oldshape):
            r_tmp.append([])
            comb = [[x for x in range(n)] for n in item]
            for i in itertools.product(*comb):
                r_tmp[-1].append(fit_ranges)
        shape = oldshape + shape
        fit_ranges = r_tmp
    return fit_ranges, shape

def fitting(fitfunc, X, Y, start, add=None, correlated=True, verbose=True):
    """A function that fits a correlation function.

    This function fits the given function fitfunc to the data given in
    X and Y. The function needs some start values, given in start, and
    can use a correlated or an uncorrelated fit.

    Parameters
    ----------
    fitfunc : callable
            The function to fit to the data.
    X, Y : ndarrays
        The X and Y data.
    start : sequence
        The starting parameters for the fit.
    add : ndarray, optional
        The additional parameters for the fit.
    correlated : bool
        Flag to use a correlated or uncorrelated fit.
    verbose : bool
        Controls the amount of information written to the screen.

    Returns
    -------
    ndarray
        The fit parameters after the fit.
    ndarray
        The chi^2 values of the fit.
    ndarray
        The p-values of the fit
    """
    # define error function
    if add is None:
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
    dof = float(Y.shape[1]-len(start)) 
    samples = Y.shape[0]
    # create results arrays
    res = np.zeros((samples, len(start)))
    chisquare = np.zeros(samples)
    # The FIT to the boostrap samples
    if add is None:
        for b in range(samples):
            p,cov1,infodict,mesg,ier = leastsq(errfunc, start, args=(X, Y[b],
                cov), full_output=1, factor=0.1)
            chisquare[b] = float(sum(infodict['fvec']**2.))
            res[b] = np.array(p)
    else:
        for b in range(samples):
            p,cov1,infodict,mesg,ier = leastsq(errfunc, start, args=(X, Y[b],
                add[b], cov), full_output=1, factor=0.1)
            chisquare[b] = float(sum(infodict['fvec']**2.))
            res[b] = np.array(p)
    # calculate mean and standard deviation
    res_mean, res_std = compute_error(res)
    # p-value calculated
    pvals = 1. - scipy.stats.chi2.cdf(chisquare, dof)

    # writing summary to screen
    if verbose:
        if correlated:
            print("fit results for a correlated fit:")
        else:
            print("fit results for an uncorrelated fit:")
        print("degrees of freedom: %f\n" % dof)
        print("bootstrap samples: %d\n" % samples)
        
        print("fit results:")
        for rm, rs in zip(res[0], res_std):
            print("  %.6e +/- %.6e" % (rm, rs))
        print("Chi^2/dof: %.6e +/- %.6e" % (chisquare[0]/dof, np.std(chisquare)
              /dof))
        print("p-value: %lf" % pvals[0]) 

    return res, chisquare, pvals
