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

def fitting(fitfunc, X, Y, start_parm, correlated=True, verbose = 1):
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
        of the fit.
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
    # p-value calculated on original data
    pvalue = 1.-scipy.stats.chi2.cdf(chisquare[0], dof)

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
        print("p-value: %lf" % pvalue) 

    return res, chi2, pvalue
