"""
Routines for fitting
"""

import os
import itertools
from scipy.optimize import leastsq
import scipy.stats
import numpy as np

from statistics import compute_error

def fit_single(fitfunc, start, corr, franges, add=None, debug=0):
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
    franges : sequence of ints or sequence of sequences of int
        The calculated fit ranges for the data.
    add : ndarray, optional
        Additional arguments to the fit function.
    debug : int, optional
        The amount of info printed.
    """
    dshape = corr.shape
    ncorr = dshape[-1]
    # prepare X data
    if debug > 0:
        print("Get X data")
    X = np.linspace(0., float(dshape[1]), dshape[1], endpoint=False)

    # fitting
    if debug > 0:
        print("fitting the data")
    for n in range(ncorr):
        for i, r in enumerate(franges[n]):
            res, chi, pva = fitting(fitfunc, X[r[0]:r[1]],
                corr.data[:,r[0]:r[1],n], start, add = add, correlated=True,
                debug=debug)
            yield (n, i), res, chi, pva
    #        fitres.add_data((n, i), res, chi, pva)
    #return fitres

def fit_comb(fitfunc, start, corr, franges, fshape, oldfit, add=None,
        oldfitpar=None, useall=False, debug=0):
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
    franges : sequence of ints or sequence of sequences of int
        The calculated fit ranges for the data.
    fshape : sequence of ints or sequence of sequences of int
        The calculated fit ranges for the data.
    oldfit : None or FitResult, optional
        Reuse the fit results of an old fit for the new fit.
    add : ndarray, optional
        Additional arguments to the fit function. This is stacked along
        the third dimenstion to the oldfit data.
    oldfitpar : None, int or sequence of int, optional
        Which parameter of the old fit to use, if there is more than one.
    useall : bool
        Using all correlators in the single particle correlator or
        use just the lowest.
    debug : int, optional
        The amount of info printed.
    """
    dshape = corr.shape
    ncorrs = [len(s) for s in fshape]
    if not useall:
        ncorrs[-2] = 1
    if ncorrs[-1] != dshape[-1]:
        raise RuntimeError("something wrong in fit_comb")
    # prepare X data
    if debug > 0:
        print("Get X data")
    X = np.linspace(0., float(dshape[1]), dshape[1], endpoint=False)

    if debug > 0:
        print("fitting the data")
    # iterate over the correlation functions
    ncorriter = [[x for x in range(n)] for n in ncorrs]
    for item in itertools.product(*ncorriter):
        if debug > 1:
            print("fitting correlators %s" % str(item))
        # create the iterator over the fit ranges
        tmp = [fshape[i][x] for i,x in enumerate(item)]
        rangesiter = [[x for x in range(n)] for n in tmp]
        # iterate over the fit ranges
        for ritem in itertools.product(*rangesiter):
            if debug > 1:
                print("fitting fit ranges %s" % str(ritem))
            # get fit interval
            r = franges[item[-1]][ritem[-1]]
            # get old data
            add_data = oldfit.get_data(item[:-1] + ritem[:-1]) 
            # get only the wanted parameter if oldfitpar is given
            if oldfitpar is not None:
                add_data = add_data[:,oldfitpar]
            # if there is additional stuff needed for the fit
            # function add it to the old data
            if add is not None:
                # get the shape right, atleast_2d adds the dimension
                # in front, we need it in the end
                if add.ndim == 1:
                    add.shape = (-1, 1)
                if add_data.ndim == 1:
                    add_data.shape = (-1, 1)
                add_data = np.hstack((add_data, add))
            res, chi, pva = fitting(fitfunc, X[r[0]:r[1]],
                    corr.data[:,r[0]:r[1],item[-1]], start, add=add_data,
                    correlated=True, debug=debug)
            yield item + ritem, res, chi, pva

def calculate_ranges(ranges, shape, oldshape=None, step=2, min_size=4, debug=0):
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
    debug : int
        The amount of info printed.

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
        # check if we exceed the time extent
        if ranges[1] > shape[1] - 1:
            if debug > 1:
                print("the upper range exceeds the data range")
            ranges[1] = shape[1] - 1
        r_tmp = get_ranges(ranges[0], ranges[1], step, min_size)
        fit_ranges = [r_tmp for i in range(ncorr)]
        shape = [[r_tmp.shape[0]] * ncorr]
    else:
        # one fitrange for every correlator
        if len(ranges) != ncorr:
            raise ValueError("number of ranges and correlators is incompatible")
        fit_ranges = []
        for ran in ranges:
            # check if we exceed the time extent
            if ran[1] > shape[1] - 1:
                ran[1] = shape[1] - 1
            fit_ranges.append(get_ranges(ran[0], ran[1], step, min_size))
        shape = [[ran.shape[0] for ran in fit_ranges]]
    if oldshape is not None:
        shape = oldshape + shape
    fit_ranges = np.asarray(fit_ranges)
    return fit_ranges, shape

def get_ranges(lower, upper, step, minsize):
    """Get the intervals given a lower and upper bound, step size and
    the minimal size of the interval.
    """
    ran = []
    for lo in range(lower, upper+1, step):
        for up in range(lower, upper+1, step):
            if (up - lo + 2) > minsize:
                # the +2 comes from the fact that the interval contains one
                # more number than the difference between the boundaries and
                # because python would exclude the upper boundary but we
                # include it explicitly
                ran.append((lo, up))
    return np.asarray(ran)

def fitting(fitfunc, X, Y, start, add=None, correlated=True, debug=0):
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
    debug : int
        The amount of info printed.

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
    if debug > 1:
        print("defining errfunc and computing covariance matrix")
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
    if debug > 1:
        print("fitting the data")
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
    if debug > 3:
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

if __name__ == "__main__":
    pass
