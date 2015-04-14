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
# Function: A function that fits a correlation function.
#
# For informations on input parameters see the description of the function.
#
################################################################################

from scipy.optimize import leastsq
import scipy.stats
import numpy as np

def fitting(fitfunc, X, Y, start_parm, correlated=True, verbose=True):
    """A function that fits a correlation function.

    This function fits the given function fitfunc to the data given in X and Y.
    The function needs some start values, given in start_parm, and can use a
    correlated or an uncorrelated fit.

    Args:
        fitfunc: The function to fit to the data.
        X: The time slices.
        Y: The bootstrap samples of the data.
        start_parm: The starting parameters for the fit.
        correlated: Flag to use a correlated or uncorrelated fit.
        verbose: Controls the amount of information written to the screen.

    Returns:
        The function returns the fitting parameters, the chi^2 and the p-value
        for every bootstrap sample.
    """
    errfunc = lambda p, x, y, error: np.dot(error, (y-fitfunc(p,x)).T)
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
    for b in range(0, Y.shape[0]):
        p,cov1,infodict,mesg,ier = leastsq(errfunc, start_parm,
                                   args=(X, Y[b,:], cov), full_output=1)
        chisquare[b] = float(sum(infodict['fvec']**2.))
        res[b] = np.array(p)
    # calculate mean and standard deviation
    res_mean, res_std = np.mean(res, axis=0), np.std(res, axis=0)
    chi2 = np.median(chisquare)
    # p-value calculated
    pvals = 1. - scipy.stats.chi2.cdf(chisquare, dof)

    # The fit to the mean value
    y = np.mean(Y, axis=0)
    p,cov1,infodict,mesg,ier = leastsq(errfunc, start_parm, \
                               args=(X, y, cov), full_output=1)

    # writing results to screen
    if verbose:
        if correlated:
            print("fit results for a correlated fit:")
        else:
            print("fit results for an uncorrelated fit:")
        print("degrees of freedom: %f\n" % dof)
        
        print("bootstrap fit:")
        for rm, rs in zip(res_mean, res_std):
            print("  %.6e +/- %.6e") % (rm, rs)
        print("Chi^2/dof: %.6e +/- %.6e\n" % (chi2/dof,
              np.std(chisquare)/dof))

        print("mean value fit:")
        for rm, rs in zip(p, res_std):
            print("  %.6e +/- %.6e") % (rm, rs)
        print("Chi^2/dof: %.6e +/- %.6e\n" % (float(sum(infodict['fvec']**2.) /
              dof), np.std(chisquare)/dof))

        print("original data fit:")
        for rm, rs in zip(res[0], res_std):
            print("  %.6e +/- %.6e") % (rm, rs)
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

def fitting_range(fitfunc, X, Y, start_parm, correlated=True, verbose=True):
    """A function that fits a correlation function for different fit ranges.

    This function fits the given function fitfunc to the data given in X and Y.
    The function needs some start values, given in start_parm, and can use a
    correlated or an uncorrelated fit. Fits are performed for many different
    fit ranges.

    Args:
        fitfunc: The function to fit to the data.
        X: The time slices.
        Y: The bootstrap samples of the data.
        start_parm: The starting parameters for the fit.
        correlated: Flag to use a correlated or uncorrelated fit.
        verbose: Controls the amount of information written to the screen.

    Returns:
    """
    # vary the lower and upper end of the fit range
    for lo in range(int(Y.shape[1]/4), Y.shape[1]-5):
        for up in range(lo+5, x.shape[1]):
            # fit the data
            res, chi2, pval=fitting(fitfunc, X[lo:up], Y[:,lo:up], start_params,
                                    correlated=correlated, verbose=verbose)
            # calculate the weight
            weight = ((1. - np.abs(pval - 0.5)) * (1.0))**2
            # calculate weighted median
            median = quantile_1D(res[:,1], weight, 0.5)

            # print some result on screen
            print("%2d-%2d: p-value %.7lf, chi2/dof %.7lf, E %.7lf" % (lo, up,
                  pval[0], chi2[0]/(len(X[lo:up])-len(start_params)),median))




def scan_fit_range(fitfunc, X, Y, start_params, correlated=True, verbose=False):
    """Fits the fitfunction to the data for different fit ranges and prints the
       result.

       Args:
           fitfunc: The function to fit.
           X: The time slices.
           Y: The bootstrap samples of the data.
           start_params: The start parameters for the fit.
           correlated: Correlated or uncorrelated fit.
           verbose: Verbosity of the fit function.

       Returns:
           Nothing.
    """
    ## vary the lower end of the fit range
    #for lo in range(int(Y.shape[1]/4), Y.shape[1]-5):
    #    # vary the upper end of the fit range
    #    for up in range(lo+5, Y.shape[1]):
    # vary the lower end of the fit range
    for lo in range(7, 13):
        # vary the upper end of the fit range
        for up in range(15, 19):
            # fir the data
            res, chi2, pval=fitting(fitfunc, X[lo:up], Y[:,lo:up], start_params,
                                    correlated=correlated, verbose=verbose)
            # print some result on screen
            print("%2d-%2d: p-value %.7lf, chi2/dof %.7lf, E %.7lf" % (lo, up,
                  pval[0], chi2[0]/(len(X[lo:up])-len(start_params)),res[0,-1]))
