"""
Routines for fitting
"""

import numpy as np

def fit(fitfunc, start, corr, ranges):
    """Fits fitfunc to a Correlators object.

    The predefined functions describe a single particle correlation
    function, a ratio of single and two-particle correlation
    functions and a constant function.

    Parameters
    ----------
    fitfunc : {0, 1, 2, callable}
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
    fit_ranges = calculate_ranges(ranges, corr.shape)

def fit_comb(self, fitfunc, start, corr, ranges, oldfit, oldfitpar=None):
    """Fits fitfunc to a Correlators object.

    The predefined functions describe a single particle correlation
    function, a ratio of single and two-particle correlation
    functions and a constant function.

    Parameters
    ----------
    fitfunc : {0, 1, 2, callable}
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

def calculate_ranges(ranges, shape, oldranges=None, step=2, min_size=4):
    """Calculates the fit ranges.

    The layout is (ncorr, nranges, 2), where ncorr is the number of
    correlators and nranges the number of fitranges. The two is due to
    the lower and upper index of the fitrange.

    If a further fitrange is given in, then the ncorr and nrange from
    these ranges is prepended to the current fit ranges to give
    (oldncorr, oldnranges, ncorr, nranges, 2).
    This can in principal be done arbitrarily long.

    Parameters
    ----------
    ranges : sequence of ints or sequence of sequences of int
        The ranges in which to fit, either one range for all or
        one range for each data set in corr.
    shape : tuple
        The shape of the data.
    oldranges : ndarray
        The fitranges of a fit before.
    step : int, optional
        The steps in the loops.
    min_size : int, optional
        The minimal size of the interval.

    Returns
    -------
    ndarray
        The fit ranges.
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
    fit_ranges = np.asarray(fit_ranges)
    if oldranges is not None:
        # from the old ranges we only need the number of correlators
        # (always first number) and the number of fitranges (always
        # the second number). 
        r_tmp = np.zeros(oldranges.shape[:-1] + fit_ranges.shape)
        for a in range(fit_ranges.shape[0]):
            r_tmp[...,a,:,:] = fit_ranges[a]
        fit_ranges = r_tmp
    return fit_ranges

