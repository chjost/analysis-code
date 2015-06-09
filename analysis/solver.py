################################################################################
#
# Author: Christian Jost (jost@hiskp.uni-bonn.de)
# Date:   April 2015
#
# Copyright (C) 2015 Christian Jost
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
# Function: A set of functions to find roots for the determinant equation.
#           ATM this file reimplements the root finding procedure by L. Liu.
#
# For informations on input parameters see the description of the function.
#
################################################################################

import os
import numpy as np
import scipy.optimize as opt

from _memoize import memoize
from ._calc_energies import (EfromMpi, WfromE)
from findroot import root

def chi2(a0, r0, a2, r2, N, data, mpi, cov, infolist):
    """Calculates the total chi^2 of the problem.

    Most things are hardcoded for the test data. This function might change
    a lot.

    Args:
        a0: scattering length for l=0
        r0: scattering radius for l=0
        a2: scattering length for l=2
        r2: scattering radius for l=2
        N: sample number
        data: the energy data
        mpi: the pion masses
        cov: the inverse covariance matrix of the different lattice sizes
        infolist: list with information about lattice size, momentum, etc.

    Returns:
        The total chi^2
    """
    verbose = False
    Wroot = np.zeros(np.sum(len(t[-1]) for t in infolist))
    # loop over all energy entries, there are 3 mpi entries at the start of
    # the array
    for s in infolist:
        index = s[-1]
        if len(index) == 1:
            croot = root(s[0], mpi[N,s[1]], a0, r0, a2, r2, s[3], s[2], 1)
            Eroot = EfromMpi(mpi[N,s[1]], np.sqrt(croot), s[0])
            Wtmp = WfromE(Eroot, s[3], s[0])
            Wroot[index] = Wtmp
            if verbose:
                print("croot")
                print(croot)
                print("Eroot")
                print(Eroot)
                print("Wroot")
                print(Wtmp)
                print("data")
                print(data[N,index])
        elif len(index) == 2:
            croot = root(s[0], mpi[N,s[1]], a0, r0, a2, r2, s[3], s[2], 3)
            Eroot = EfromMpi(mpi[N,s[1]], np.sqrt(croot), s[0])
            Wtmp = WfromE(Eroot, s[3], s[0])
            Wroot[index[0]] = Wtmp[0]
            if verbose:
                print("croot")
                print(croot)
                print("Eroot")
                print(Eroot)
                print("Wroot")
                print(Wtmp)
                print("data")
                print(data[N,index])
            if (np.fabs(Wtmp[0]-data[N,index[0]])> \
                np.fabs(Wtmp[1]-data[N,index[0]])):
                print("unusual first level match")
                print("iE: %i, W: %.8lf, Wroot1: %.8lf, Wroot2: %.8lf"\
                      % (index[0], data[N,index[0]], Wtmp[0], Wtmp[1]))
            if (np.fabs(Wtmp[1]-data[N,index[1]])< \
                np.fabs(Wtmp[2]-data[N,index[1]])):
                Wroot[index[1]] = Wtmp[1]
            else:
                Wroot[index[1]] = Wtmp[2]
            if (np.fabs(Wtmp[1]-data[N,index[1]])> \
                np.fabs(Wtmp[2]-data[N,index[1]])):
                print("unusual second level match")
                print("iE: %i, W: %.8lf, Wroot2: %.8lf, Wroot3: %.8lf"\
                      % (index[1], data[N,index[1]], Wtmp[1], Wtmp[2]))
        elif len(index) == 0:
            continue
        else:
            print("wrong number of masked entries (nE_in)")
            os.sys.exit(-5)
        print("")
    # print the roots
    for Edata, Ecalc in zip(data[N], Wroot):
        print("%.7lf, %.7lf, %.4e\n" % (Edata, Ecalc, abs(Edata - Ecalc)))

    # calculate chi^2
    chi = np.dot((data[N] - Wroot), np.dot(cov, (data[N] - Wroot)))
    return chi

def minimizer(data, mpi, cov, infolist, _a0, _r0, _a2, _r2, h, n):
    """Minimizer for the chi^2 problem.

    Args:
        data: The data to operate on
        mpi: the pion masses
        cov: The covariance matrix of the data
        infolist: list containing information about lattice size, momentum etc
        _a0: starting value for a0
        _r0: starting value for r0
        _a2: starting value for a2
        _r2: starting value for r2
        h: numpy array of length 3
        n: Sample number

    Returns:
        a0, r0, a2, chi_min
    """
    # was a global variable
    minimizer_tolerance = 10e-6
    calc_chi2 = lambda a0, r0, a2: chi2(a0, r0, a2, 0., n, data, mpi, \
                                            cov, infolist)
    # init first variables
    alpha = 1.
    beta = 1.
    gamma = 0.5
    para = np.zeros((4,3))
    para_c = np.zeros((4,3))
    para_r = np.zeros((3,))
    para_e = np.zeros((3,))
    para_co = np.zeros((3,))

    # set para
    para[:,0] = _a0
    para[:,1] = _r0
    para[:,2] = _a2
    for i in range(1,4):
        para[i, i-1] += h[i-1]
    para_c[0] = (para[1] + para[2] + para[3])/3.
    para_c[1] = (para[0] + para[2] + para[3])/3.
    para_c[2] = (para[1] + para[0] + para[3])/3.
    para_c[3] = (para[1] + para[2] + para[0])/3.

    # init next variables
    chi_test = np.zeros((4,))
    chi_mean = 0.
    chi_diff = 1000.
    chi_r = 0.
    chi_e = 0.
    chi_co = 0.
    chi_temp = 0.

    print("starting to minimize")

    # CJ: needed for array sorting, done by numpy automatically
    sort = np.arange(0, 4, dtype=int)

    for i in range(para.shape[0]):
        chi_test[i] = calc_chi2(para[i,0], para[i,1], para[i,2])
        print(chi_test[i])
    chi_mean = np.mean(chi_test)
    chi_diff = np.var(chi_test)
    print("chi_mean:")
    print(chi_mean)
    print("chi_diff:")
    print(chi_diff)
    os.sys.exit(-10)

    n_iter = 0
    while( chi_diff > minimizer_tolerance ):
        print("start loop %i" % n_iter)
        # sort chi_test
        sort = np.argsort(chi_test)
        para_r = (1. + alpha) * para_c[sort[3]] - alpha * para[sort[3]]
        chi_r = calc_chi2(para_r[0], para_r[1], para_r[2])
        print("chi_r %lf" % chi_r)
        if (chi_r >= chi_test[sort[0]]) and (chi_r <= chi_test[sort[2]]):
            print("case0")
            para[sort[3]] = para_r
            chi_temp = chi_r
        elif (chi_r < chi_test[sort[0]]):
            print("case1")
            para_e = beta * para_r + (1. - beta) * para_c[sort[3]]
            chi_e = calc_chi2(para_e[0], para_e[1], para_e[2])
            print("chi_e %lf" % chi_e)
            if (chi_e < chi_test[sort[0]]):
                print("case10")
                para[sort[3]] = para_e
                chi_temp = chi_e
            else:
                print("case11")
                para[sort[3]] = para_r
                chi_temp = chi_r
        elif (chi_r > chi_test[sort[2]]):
            print("case2")
            if (chi_r < chi_test[sort[3]]):
                para_co = gamma * para_r + (1. - gamma) * para_c[sort[3]]
            else:
                para_co = gamma * para[sort[3]] + (1. - gamma) * para_c[sort[3]]
            chi_co = calc_chi2(para_co[0], para_co[1], para_co[2])
            print("chi_co %lf" % chi_co)
            if (chi_co < chi_test[sort[3]]) and (chi_co < chi_r):
                print("case20")
                para[sort[3]] = para_co
                chi_temp = chi_co
            else:
                print("case21")
                h /= 2.
                for i in range(1,4):
                    para[i, i-1] += h[i-1]
                para[sort[3]] = para_r
                for i in range(para.shape[0]):
                    chi_test[i] = calc_chi2(para[i,0], para[i,1], para[i,2])
                chi_temp = chi_test[sort[3]]
        else:
            print("error in while loop")
            os.sys.exit(-10)

        chi_test[sort[3]] = chi_temp
        para_c[0] = (para[1] + para[2] + para[3])/3.
        para_c[1] = (para[0] + para[2] + para[3])/3.
        para_c[2] = (para[1] + para[0] + para[3])/3.
        para_c[3] = (para[1] + para[2] + para[0])/3.

        print("chi_test: %lf %lf %lf %lf" % (chi_test[sort[0]],
            chi_test[sort[1]], chi_test[sort[2]], chi_test[sort[3]]))

        chi_mean = np.mean(chi_test)
        chi_diff = np.var(chi_test)

        print("n_iter %i, chi_mean %lf, chi_diff %lf" % (n_iter, chi_mean, chi_diff))
        for _i in range(para.shape[0]):
            print("a0: %lf, r0: %lf, a2 %lf" % (para[_i,0], para[_i,1], para[_i,2]))
        n_iter += 1
    # end while loop
    a0 = np.mean(para[:,0])
    r0 = np.mean(para[:,1])
    a2 = np.mean(para[:,2])

    return a0, r0, a2, chi_mean

def min2(par, N, data, mpi, cov, infolist):
    """Minimizer based on scipy functions.

    Args:
        par: parameters

    Returns:
        par: final parameters
        chi2: final chi^2
    """
    # some variables
    min_tolerance = 1e-6

    # invoke minimizer
    # notes on the algorithms
    # L-BFGS-B very slow converging
    # BFGS no converging
    # Powell slow converging, one parameter at a time
    # CG no converging
    # Newton-CG - not possible
    # Anneal no converging
    # TNC not converging
    # SLSQP not converging
    res = opt.minimize(chi2_2, x0=par[0:3], args=(N, data, mpi, cov, infolist),\
                       tol=min_tolerance, method='SLSQP')
    print(res)
    return res.x

def min3(par, N, data, mpi, cov, infolist, h):
    """Minimizer based on scipy functions.

    Args:
        par: parameters

    Returns:
        par: final parameters
        chi2: final chi^2
    """
    # some variables
    min_tol = 1e-6
    verbose = False

    # invoke minimizer
    res,cov1,infodict,mesg,ier = opt.leastsq(chi2_3, x0=par[:3], args=(data[N],
        mpi[N], cov, infolist), ftol=min_tol, xtol=min_tol, diag=h, full_output=True)
    # the following works
    #res,cov1,infodict,mesg,ier = opt.leastsq(chi2_2, x0=par[:3], args=(N, data,
    #    mpi, cov, infolist), ftol=min_tol, xtol=min_tol, diag=h, full_output=True)
    chi2 = float(np.sum(infodict['fvec']**2.))
    if verbose:
        print(res)
        print(chi2)
    return res, chi2

def chi2_2(par, N, data, mpi, cov, infolist):
    """Calculates the total chi^2 of the problem.

    Most things are hardcoded for the test data. This function might change
    a lot.

    Args:
        par: parameters, array of length 3
        N: sample number
        data: the energy data
        mpi: the pion masses
        cov: the inverse covariance matrix of the different lattice sizes
        infolist: list with information about lattice size, momentum, etc.

    Returns:
        The total chi^2
    """
    verbose=False
    print(par)
    calc_root = lambda p, i: root(s[0], mpi[N,s[1]], p[0], p[1], p[2], 0., \
                                  s[3], s[2], i)
    Wroot = np.zeros(np.sum(len(t[-1]) for t in infolist))
    # loop over all energy entries, there are 3 mpi entries at the start of
    # the array
    for s in infolist:
        index = s[-1]
        if len(index) == 1:
            croot = calc_root(par, 1)
            Eroot = EfromMpi(mpi[N,s[1]], np.sqrt(croot), s[0])
            Wtmp = WfromE(Eroot, s[3], s[0])
            Wroot[index] = Wtmp
            if verbose:
                print("croot")
                print(croot)
                print("Eroot")
                print(Eroot)
                print("Wroot")
                print(Wtmp)
                print("data")
                print(data[N,index])
        elif len(index) == 2:
            croot = calc_root(par, 3)
            Eroot = EfromMpi(mpi[N,s[1]], np.sqrt(croot), s[0])
            Wtmp = WfromE(Eroot, s[3], s[0])
            Wroot[index[0]] = Wtmp[0]
            if verbose:
                print("croot")
                print(croot)
                print("Eroot")
                print(Eroot)
                print("Wroot")
                print(Wtmp)
                print("data")
                print(data[N,index])
            if (np.fabs(Wtmp[0]-data[N,index[0]])> \
                np.fabs(Wtmp[1]-data[N,index[0]])):
                print("unusual first level match")
                print("iE: %i, W: %.8lf, Wroot1: %.8lf, Wroot2: %.8lf"\
                      % (index[0], data[N,index[0]], Wtmp[0], Wtmp[1]))
            if (np.fabs(Wtmp[1]-data[N,index[1]])< \
                np.fabs(Wtmp[2]-data[N,index[1]])):
                Wroot[index[1]] = Wtmp[1]
            else:
                Wroot[index[1]] = Wtmp[2]
            if (np.fabs(Wtmp[1]-data[N,index[1]])> \
                np.fabs(Wtmp[2]-data[N,index[1]])):
                print("unusual second level match")
                print("iE: %i, W: %.8lf, Wroot2: %.8lf, Wroot3: %.8lf"\
                      % (index[1], data[N,index[1]], Wtmp[1], Wtmp[2]))
        elif len(index) == 0:
            continue
        else:
            print("wrong number of entries (nE_in)")
            os.sys.exit(-5)
        #print("")
    if verbose:
        for Edata, Ecalc in zip(data[N], Wroot):
            print("%.7lf, %.7lf, %.4e\n" % (Edata, Ecalc, abs(Edata - Ecalc)))

    # calculate chi^2
    #chi = np.dot((data[N] - Wroot), np.dot(cov, (data[N] - Wroot)))
    #print(chi)
    #return chi
    dx = np.dot(cov, (data[N] - Wroot))
    if verbose:
        print(np.sum(dx**2))
    return dx

@memoize(20)
def chi2_3(par, data, mpi, cov, infolist):
    """Calculates the total chi^2 of the problem.

    Most things are hardcoded for the test data. This function might change
    a lot.

    Args:
        par: parameters, array of length 3
        data: the energy data
        mpi: the pion masses
        cov: the inverse covariance matrix of the different lattice sizes
        infolist: list with information about lattice size, momentum, etc.

    Returns:
        The total chi^2
    """
    verbose=False
    print(par)
    calc_root = lambda p, i: root(s[0], mpi[s[1]], p[0], p[1], p[2], 0., \
                                  s[3], s[2], i)
    Wroot = np.zeros(np.sum(len(t[-1]) for t in infolist))
    # loop over all energy entries, there are 3 mpi entries at the start of
    # the array
    for s in infolist:
        index = s[-1]
        if len(index) == 1:
            croot = calc_root(par, 1)
            Eroot = EfromMpi(mpi[s[1]], np.sqrt(croot), s[0])
            Wtmp = WfromE(Eroot, s[3], s[0])
            Wroot[index] = Wtmp
            if verbose:
                print("croot")
                print(croot)
                print("Eroot")
                print(Eroot)
                print("Wroot")
                print(Wtmp)
                print("data")
                print(data[N,index])
        elif len(index) == 2:
            croot = calc_root(par, 3)
            Eroot = EfromMpi(mpi[s[1]], np.sqrt(croot), s[0])
            Wtmp = WfromE(Eroot, s[3], s[0])
            Wroot[index[0]] = Wtmp[0]
            if verbose:
                print("croot")
                print(croot)
                print("Eroot")
                print(Eroot)
                print("Wroot")
                print(Wtmp)
                print("data")
                print(data[N,index])
            if (np.fabs(Wtmp[0]-data[index[0]])> \
                np.fabs(Wtmp[1]-data[index[0]])):
                print("unusual first level match")
                print("iE: %i, W: %.8lf, Wroot1: %.8lf, Wroot2: %.8lf"\
                      % (index[0], data[index[0]], Wtmp[0], Wtmp[1]))
            if (np.fabs(Wtmp[1]-data[index[1]])< \
                np.fabs(Wtmp[2]-data[index[1]])):
                Wroot[index[1]] = Wtmp[1]
            else:
                Wroot[index[1]] = Wtmp[2]
            if (np.fabs(Wtmp[1]-data[index[1]])> \
                np.fabs(Wtmp[2]-data[index[1]])):
                print("unusual second level match")
                print("iE: %i, W: %.8lf, Wroot2: %.8lf, Wroot3: %.8lf"\
                      % (index[1], data[index[1]], Wtmp[1], Wtmp[2]))
        elif len(index) == 0:
            continue
        else:
            print("wrong number of entries (nE_in)")
            os.sys.exit(-5)
        #print("")
    if verbose:
        for Edata, Ecalc in zip(data, Wroot):
            print("%.7lf, %.7lf, %.4e\n" % (Edata, Ecalc, abs(Edata - Ecalc)))

    # calculate chi^2
    #chi = np.dot((data[N] - Wroot), np.dot(cov, (data[N] - Wroot)))
    #print(chi)
    #return chi
    dx = np.dot(cov, (data - Wroot))
    if verbose:
        print(np.sum(dx**2))
    return dx
