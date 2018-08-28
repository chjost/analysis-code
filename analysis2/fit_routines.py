"""
Routines for fitting
"""

import os
import itertools
from scipy.optimize import leastsq
import scipy.stats
import numpy as np

from covariance import custom_cov
from statistics import compute_error
from functions import compute_eff_mass
from utils import loop_iterator, eig_decomp, svd_inv, chol_inv

def fit_single(fitfunc, start, corr, franges, add=None, debug=0,
        correlated=True, xshift=0., npar=2):
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
    correlated : bool
        Use the full covariance matrix or just the errors.
    """
    dshape = corr.shape
    ncorr = dshape[-1]
    # prepare X data
    if debug > 0:
        print("Get X data")
    X = np.linspace(0., float(dshape[1]), dshape[1], endpoint=False) + xshift

    # fitting
    if debug > 0:
        print("fitting the data")
    for n in range(ncorr):
        if debug > 1:
            print("fitting correlator %d" % (n))
        if isinstance(start[0], (tuple, list)) and len(start[0]) == 1:
            for i, r in enumerate(franges[n]):
                if debug > 1:
                    print("fitting interval %d" % (i))
                res, chi, pva = fitting(fitfunc, X[r[0]:r[1]+1],
                    corr.data[:,r[0]:r[1]+1,n], start[n], add=add,
                        correlated=correlated, debug=debug)
                yield (n, i), res, chi, pva
        elif isinstance(start[0], (tuple, list)):
            for i, r in enumerate(franges[n]):
                if debug > 1:
                    print("fitting interval %d" % (i))
                res, chi, pva = fitting(fitfunc, X[r[0]:r[1]+1],
                    corr.data[:,r[0]:r[1]+1,n], start[n][i], add=add,
                        correlated=correlated, debug=debug)
                yield (n, i), res, chi, pva
        else:
            for i, r in enumerate(franges[n]):
                if debug > 1:
                    print("fitting interval %d" % (i))
                res, chi, pva = fitting(fitfunc, X[r[0]:r[1]+1],
                    corr.data[:,r[0]:r[1]+1,n], start, add=add,
                        correlated=correlated, debug=debug)
                yield (n, i), res, chi, pva

def fit_comb(fitfunc, start, corr, franges, fshape, oldfit, add=None,
        oldfitpar=None, useall=False, debug=0, xshift=0., correlated=True, npar=1):
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
    xshift: a scalar to shift the X-linspace for the fitting
    debug : int, optional
        The amount of info printed.
    correlated : bool
        Use the full covariance matrix or just the errors.
    """
    dshape = corr.shape
    ncorrs = [len(s) for s in fshape]
    if not useall:
        ncorrs[-2] = 1
    if ncorrs[-1] != dshape[-1]:
        print(fshape)
        print(ncorrs)
        print(dshape)
        raise RuntimeError("something wrong in fit_comb")
    # prepare X data
    if debug > 0:
        print("Get X data")
        print("xshift is %f" % xshift)
    X = np.linspace(0., float(dshape[1]), dshape[1], endpoint=False)+xshift

    if debug > 0:
        print("fitting the data")
    # iterate over the correlation functions
    for item in loop_iterator(ncorrs):
        if debug > 1:
            print("fitting correlators %s" % str(item))
        n = item[-1]
        # iterate over the fit ranges
        tmp = [fshape[i][x] for i,x in enumerate(item)]
        for ritem in loop_iterator(tmp):
            m = ritem[-1]
            if debug > 1:
                print("fitting fit ranges %s" % str(ritem))
            # get fit interval
            r = franges[n][m]
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
            # do the fitting
            if isinstance(start[0], (tuple, list)):
                res, chi, pva = fitting(fitfunc, X[r[0]:r[1]+1],
                    corr.data[:,r[0]:r[1]+1,item[-2],n], start[n],
                    add=add_data, correlated=correlated, debug=debug)
            else:
                res, chi, pva = fitting(fitfunc, X[r[0]:r[1]+1],
                    corr.data[:,r[0]:r[1]+1,n], start,
                    add=add_data, correlated=correlated, debug=debug)
            yield item + ritem, res, chi, pva

def calculate_ranges(ranges, shape, oldshape=None, dt_i=2, dt_f=2, dt=4, debug=0,
        lintervals=False):
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
    dt_i, dt_f : ints, optional
        The step size for the first and last time slice for the fitting.
    dt : int, optional
        The minimal size of the interval.
    fixend : bool, optional
        Fix the last point in the fit ranges
    debug : int, optional
        The amount of info printed.
    lintervals : bool, optional
        In ranges intervals for the upper and lower limit respectively are given.

    Returns
    -------
    sequence
        The fit ranges.
    sequence
        The "shape" of the ranges.
    """
    ncorr = shape[-1]
    if lintervals:
        if isinstance(ranges[0], int):
            # assume one range for every correlator
            # check if we exceed the time extent
            if ranges[1] > shape[1] - 1:
                if debug > 1:
                    print("the upper range exceeds the data range")
                ranges[1] = shape[1] - 1
            r_tmp = get_ranges2(ranges[:2], ranges[2:], dt_i=dt_i, dt_f=dt_f, dt=dt)
            fit_ranges = [r_tmp for i in range(ncorr)]
            shape = [[r_tmp.shape[0]] * ncorr]
        else:
            # one fitrange for every correlator
            if len(ranges) != ncorr:
                raise ValueError("number of ranges and correlators is incompatible")
            fit_ranges = []
            for ran in ranges:
                # check if we exceed the time extent
                if ran[-1] > shape[1] - 1:
                    ran[-1] = shape[1] - 1
                fit_ranges.append(get_ranges2(ran[:2], ran[2:], dt_i=dt_i, dt_f=dt_f, dt=dt))
            shape = [[ran.shape[0] for ran in fit_ranges]]
    else:
        # check if ranges makes sense
        if isinstance(ranges[0], int):
            # assume one range for every correlator
            # check if we exceed the time extent
            if ranges[1] > shape[1] - 1:
                if debug > 1:
                    print("the upper range exceeds the data range")
                ranges[1] = shape[1] - 1
            r_tmp = get_ranges(ranges[0], ranges[1], dt_i=dt_i, dt_f=dt_f, dt=dt)
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
                fit_ranges.append(get_ranges(ran[0], ran[1], dt_i=dt_i, dt_f=dt_f, dt=dt))
            shape = [[ran.shape[0] for ran in fit_ranges]]
    if oldshape is not None:
        shape = oldshape + shape
    fit_ranges = np.asarray(fit_ranges)
    return fit_ranges, shape

def get_ranges(lower, upper, dt_i=2, dt_f=2, dt=4):
    """Get the intervals given a lower and upper bound, step size and
    the minimal size of the interval.
    """
    ran = []
    # keep initial time slice fixed
    if dt_i < 0:
        # keep upper time slice fixed
        if dt_f < 0:
            #if (up - lo + 2) > dt:
            if (upper - lower + 2) > dt:
                ran.append((lower, upper))
        else:
            for up in range(upper, lower-1, -dt_f):
                if (up - lower + 2) > dt:
                    ran.append((lower, up))
    # keep upper time slice fixed
    elif dt_f < 0:
        for lo in range(lower, upper+1, dt_i):
            if (upper - lo + 2) > dt:
                ran.append((lo, upper))
    else:
        for lo in range(lower, upper+1, dt_i):
            for up in range(upper, lower-1, -dt_f):
                if (up - lo + 2) > dt:
                    # the +2 comes from the fact that the interval contains one
                    # more number than the difference between the boundaries and
                    # because python would exclude the upper boundary but we
                    # include it explicitly
                    ran.append((lo, up))
    return np.asarray(ran)

def get_ranges2(lower, upper, dt_i=2, dt_f=2, dt=4):
    """Get the intervals given an interval for lower and upper bound,
    step size and the minimal size of the interval.
    """
    ran = []
    _di = np.abs(dt_i)
    _df = np.abs(dt_f)
    # keep initial time slice fixed
    if _di < 0:
        if isinstance(lower, (tuple, list, np.ndarray)):
            _low = lower[0]
        else:
            _low = lower
        # keep upper time slice fixed
        if _df < 0:
            if isinstance(upper, (tuple, list, np.ndarray)):
                _upp = upper[-1]
            else:
                _upp = upper
            if (_upp - _low + 2) > dt:
                ran.append((_low, _upp))
        else:
            for up in range(upper[-1], upper[0], -_df):
                if (up - _low + 2) > dt:
                    ran.append((_low, up))
    # keep upper time slice fixed
    elif _df < 0:
        if isinstance(upper, (tuple, list, np.ndarray)):
            _upp = upper[-1]
        else:
            _upp = upper
        for lo in range(lower[0], lower[-1], _di):
            if (_upp - lo + 2) > dt:
                ran.append((lo, _upp))
    else:
        for lo in range(lower[0], lower[-1], _di):
            for up in range(upper[-1], upper[0], -_df):
                if (up - lo + 2) > dt:
                    # the +2 comes from the fact that the interval contains one
                    # more number than the difference between the boundaries and
                    # because python would exclude the upper boundary but we
                    # include it explicitly
                    ran.append((lo, up))
    return np.asarray(ran)

def fitting(fitfunc, X, Y, start, add=None, correlated=True, mute=None, debug=0):
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
    mute : callable, function to act on the covariance matrix
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
    if debug > 1:
        print correlated
    if not correlated:
        cov = np.diag(np.diagonal(np.cov(Y.T)))
        #print(np.diag(cov))
    else:
        cov = np.cov(Y.T)
        #print("Covariance matrix in correlated fit:")
        np.set_printoptions(linewidth=1000,threshold=1500)
        #print(cov)
        np.set_printoptions()
        #cov_inv = np.linalg.inv(cov)
        #print("Covariance matrix multiplied its inverse")
        #print(cov.dot(cov_inv))
        if mute is not None:
          #print("Mutilating Covariance Matrix")
          cov = mute(cov)
          #print("Covariance Matrix:")
          #print(cov)
    #cov = (np.linalg.cholesky(np.linalg.inv(cov))).T
    cov = (np.linalg.cholesky(chol_inv(cov))).T
    corr = np.corrcoef(Y.T)
    #if debug > 0:
        #print("In fitting correlation matrix is:")
        #print(corr)
    #print("Correlation matrix for fit is:\n %r" %corr)
    # Eigendecomposition of covariance matrix with screen output
    # eig_decomp(cov)

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
        # Enable multiple bootstrapsamples for x-values pair samplewise
        #print("x-shape in fitting is:")
        #print(X.shape)
        #print(Y.shape)
        # This is a nasty hack to enable fitting for several bootstrapsamples in
        # the xdata
        #TODO: Solve this another way, if possible, unify array layouts
        if len(X.shape) == 2 and X.shape[-1]==Y.shape[0]: 
            for b in range(samples):
                #if b == 0:
                #  print("Overview over the data used:")
                #  print("x-data for fit")
                #  print(X[:,b])
                #  print("y-data for fit")
                #  print(Y[b])
                #if b == (samples-1):
                #  print("arguments:")
                #  print(res[b-1])
                p,cov1,infodict,mesg,ier = leastsq(errfunc, start, args=(X[:,b], Y[b],
                    cov), full_output=1,factor=100.)
                chisquare[b] = float(sum(infodict['fvec']**2.))
                res[b] = np.array(p)
        else:
            for b in range(samples):
                #if b == 0:
                #  print("Overview over the data used:")
                #  print("x-data for fit")
                #  print(X)
                #  print("y-data for fit")
                #  print(Y[b])
                #  print("inverse covariance matrix:")
                #  print(cov)
                #if b == (samples-1):
                #  print("arguments:")
                #  print(res[b-1])
                p,cov1,infodict,mesg,ier = leastsq(errfunc, start, args=(X, Y[b],
                    cov), full_output=1,factor=100.)
                chisquare[b] = float(sum(infodict['fvec']**2.))
                res[b] = np.array(p)
                # check the results with original data
            chi = errfunc(res[0],X,Y[0],cov)
            if debug > 0:
                print("Check of errorfunction:")
                print(chi)
                print("Chi_squared manually")
                print(np.sum(np.square(chi)))
    else:
        for b in range(samples):
            #if b == 0:
            #  print("Overview over the data used:")
            #  print("x-data for fit")
            #  print(X)
            #  print("y-data for fit")
            #  print(Y[b])
            #  print("inverse covariance matrix:")
            #  print(cov)
            #if b == (samples-1):
            #  print("arguments:")
            #  print(res[b-1])
            p,cov1,infodict,mesg,ier = leastsq(errfunc, start, args=(X, Y[b],
                add[b], cov), full_output=1,factor=100.)
            chisquare[b] = float(sum(infodict['fvec']**2.))
            res[b] = np.array(p)
            if debug > 0 and b ==0:
                chi = errfunc(res[0],X,Y[0],add[0],cov)
                print("Check of errorfunction:")
                print(chi)
                print("Chi_squared manually")
                print(np.sum(np.square(chi)))
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
        print("Chi^2: %.6e" % chisquare[0])
        print("Chi^2/dof: %.6e +/- %.6e" % (chisquare[0]/dof, np.std(chisquare)
              /dof))
        print("p-value: %.3e" % pvals[0]) 

    return res, chisquare, pvals 

# At the moment this is only useful for KK
def globalfitting(errfunc,x,y, start, add=None, correlated=False,
    cov=None, parlim=None, debug=0):
    """A function that fits Lattice fitresults.

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
    cov : a custommade covariacne matrix
    parlim: arraylike, vector of errors on the fitparameters
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
    if cov is not None:
        _cov = cov
    else:
        _cov = np.cov(y)
    if debug > 0:
        print("In chiral fit: fit correlated? %r" %correlated)
        print("vector of inverse errors:")
        print(cov)
        print("vector of x-values:")
        print(x[0])
        print("vector of y-values:")
        print(y[0])
        print("shape of y-values:")
        print(len(y))
        print("shape of x-values:")
        print(len(x))
        print("shape of err-values:")
        print(_cov.shape)
    samples=len(x)
    chisquare=np.zeros((samples,))
    res = np.zeros((samples,len(start)))
    dof = float(_cov.shape[0]-len(start))
    for b in range(samples):
        # TODO: If clauses in for loop to slow?
        if b == 0 and debug > 0:
          print("Overview over the data used:")
          print("x-data for fit")
          print(x[b])
          print("y-data for fit")
          print(y[b])
          print("inverse covariance matrix used:")
          print(_cov)
        if b == (samples-1) and debug > 0:
          print("arguments:")
          print(res[0])
        p,cov1,infodict,mesg,ier = leastsq(errfunc, start,
            args=(x[b],y[b],_cov), full_output=1,factor=100)
        chisquare[b] = float(sum(infodict['fvec']**2.))
        res[b] = np.array(p)
        if debug > 3:
            print("Fit %d converged with reason %d, %s" %(b,ier,mesg))
    chi = errfunc(res[0],x[0],y[0],_cov)
    if debug > 0:
        print("Check of errorfunction:")
        try:
            x_data = np.r_[x[0].A,x[0].B,x[0].D]
        except:
            x_data = np.r_[x[0].A,x[0].B] 
        print(x_data.shape)
        #print(np.column_stack((x_data,chi[:-1])))
        print("Chi_squared manually")
        print(np.square(chi))
        print(np.sum(np.square(chi)))
        print(chisquare[0])
    # calculate mean and standard deviation
    res_mean, res_std = compute_error(res)
    # p-value calculated
    pvals = 1. - scipy.stats.chi2.cdf(chisquare, dof)

    # writing summary to screen
    if debug > 3:
        print("fit results for an uncorrelated fit:")
        print("degrees of freedom: %f\n" % dof)
        print("bootstrap samples: %d\n" % samples)
        
        print("fit results:")
        for rm, rs in zip(res[0], res_std):
            print("  %.3f +/- %.3f, rel. err: %.2f %%" % (rm, rs, rs/rm*100.))
        print("Chi^2/dof: %.3f" % (chisquare[0]/dof))
        print("p-value: %.3f" % pvals[0]) 

    return res, chisquare, pvals

def compute_dE(mass, mass_w, energy, energy_w, isdependend=False, flv_diff=False):
    print("Compute dE flv_diff = %s" % flv_diff)
    needed = np.zeros(mass.shape[0])
    # check if fitranges have overlap
    if isdependend:
        for i in range(mass.shape[-1]):
            for j in range(energy.shape[-1]):
                if flv_diff == False:
                    tmp = energy[:,i,j] - 2.*mass[:,i]
                else:
                    tmp = energy[:,i,j] - mass[:,i]
                tmp_w = mass_w[i] * energy_w[i,j]
                yield (0,0,i,j), tmp, needed, tmp_w
    else:
        for i in range(mass.shape[-1]):
            for j in range(energy.shape[-1]):
                if flv_diff == False:
                    tmp = energy[:,j] - 2.*mass[:,i]
                else:
                    tmp = energy[:,j] - mass[:,i]
                tmp_w = mass_w[i] * energy_w[j]
                yield (0,0,i,j), tmp, needed, tmp_w
        
def get_start_values(ncorr, ranges, data, npar=2):
    """Calculate start values for the fits.

    Parameters
    ----------
    ncorr : int
        The number of correlators used.
    ranges : list of ints
        The fit ranges for each correlator.
    data : ndarray
        The correlator data.
    npar : int
        The number of parameters.

    Returns
    -------
    start : list of list of floats
        The start values for each fit.
    """
    start = []
    for n in range(ncorr):
        start.append([])
        # calculate effective values for correlator
        c_eff = np.nanmean(data[...,n], axis=0)
        m_eff = np.nanmean(compute_eff_mass(data[...,n],data.shape[-2]), axis=0)
        # iterate over the ranges
        for r in ranges[n]:
            if npar == 2:
                start[-1].append([c_eff[r[0]], m_eff[r[0]]])
            elif npar == 1:
                start[-1].append([m_eff[r[0]]])
            elif npar == 3:
                start[-1].append([c_eff[r[0]], m_eff[r[0]], 1.])
            else:
                raise RuntimeError("Cannot set starting values for" +
                    "funtions with npar > 3.")
    return start

def get_start_values_comb(ncorr, ranges, data, npar=2):
    """Calculate start values for the combined fits.

    Parameters
    ----------
    ncorr : int
        The number of correlators used.
    ranges : list of ints
        The fit ranges for each correlator.
    data : ndarray
        The correlator data.
    npar : int
        The number of parameters.

    Returns
    -------
    start : list of list of floats
        The start values for each fit.
    """
    start = []
    for n in range(ncorr[-1]):
        # calculate effective values for correlator
        # select all axis except 1 and the last
        # needs to be a tuple for nanmean
        ax = tuple([i for i in range(data.ndim) if i != 1])
        c_eff = np.nanmean(data[...,n], axis=ax[:-1])
        m_eff = np.nanmean(compute_eff_mass(data[...,n]), axis=ax[:-1])
        # iterate over the ranges
        for r in ranges[n]:
            if npar == 2:
                start.append([c_eff[r[0]], m_eff[r[0]]])
            elif npar == 1:
                start.append([m_eff[r[0]]])
            elif npar == 3:
                start.append([c_eff[r[0]], m_eff[r[0]], 1.])
            else:
                raise RuntimeError("Cannot set starting values for" +
                    "funtions with npar > 3.")
    return start

if __name__ == "__main__":    pass
