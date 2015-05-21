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
import numpy.ma as ma
import scipy.optimize as opt
import cPickle

import zeta

def memoize(function, limit=None):
    """Function decorator for caching results.
    
    The caching is implemented as a dictionary, the key is created from the 
    arguments to the function. Only any 'verbose' flag is skipped.
    The storage implements a LRU cache, if the size is limited. The least used
    element is dropped when the limit is reached.

    Args:
        function: The function to wrap.
        limit: The maximum number of results cached.

    Returns:
        The decorated function.
    """
    # return immediately if the function has no arguments
    if isinstance(function, int):
        def memoize_wrapper(f):
            return memoize(f, function)

        return memoize_wrapper
    # create the dictionary and a list of keys
    dict = {}
    list = []
    def memoize_wrapper(*args, **kwargs):
        # filter arguments that should not be hashed
        kwa = kwargs
        if 'verbose' in kwa:
            kwa.pop('verbose')
        key = cPickle.dumps((args, kwa))
        try:
            # see if key is in list and if so append it to the end
            list.append(list.pop(list.index(key)))
        except ValueError:
            # if key is not in list, create it
            dict[key] = function(*args, **kwargs)
            list.append(key)
            # if size is limited and the limit is reached, delete first element
            if limit is not None and len(list) > limit:
                del dict[list.pop(0)]
        return dict[key]
    # save the list and the dictionary to the function
    memoize_wrapper._memoize_dict = dict
    memoize_wrapper._memoize_list = list
    memoize_wrapper._memoize_limit = limit
    memoize_wrapper._memoize_origfunc = function
    memoize_wrapper.func_name = function.func_name
    return memoize_wrapper

@memoize(50)
def WfromE(E, d=np.array([0., 0., 0.]), L=24):
    """Calculates the CM energery from the energy.

    Args:
        E: the energy
        d: total momentum vector of the system
        L: lattice size

    Returns:
        The center of mass energy.
    """
    return np.sqrt(E*E + np.dot(d, d) * 4. * np.pi*np.pi / (float(L)*float(L)))

@memoize(50)
def EfromW(W, d=np.array([0., 0., 0.]), L=24):
    """Calculates the moving frame energy from the CM energy.

    Args:
        W: the energy
        d: total momentum vector of the system
        L: lattice size

    Returns:
        The energy.
    """
    return np.sqrt(W*W - np.dot(d, d) * 4. * np.pi*np.pi / (float(L)* float(L)))

@memoize(50)
def WfromE_lat(E, d=np.array([0., 0., 0.]), L=24):
    """Calculates the CM energery from the energy using the lattice dispersion
    relation.

    Args:
        E: the energy
        d: total momentum vector of the system
        L: lattice size

    Returns:
        The center of mass energy.
    """
    return np.arccosh(np.cosh(E) + 2. * np.sum(np.sin(d*np.pi/float(L))**2))

@memoize(50)
def EfromW_lat(W, d=np.array([0., 0., 0.]), L=24):
    """Calculates the moving frame energy from the CM energy using the lattice
    dispersion relation.

    Args:
        W: the energy
        d: total momentum vector of the system
        L: lattice size

    Returns:
        The energy.
    """
    return np.arccosh(np.cosh(E) - 2. * np.sum(np.sin(d*np.pi/float(L))**2))

@memoize(50)
def EfromMpi(mpi, q, L):
    """Calculates the center of mass energy for a pion with momentum q.

    Args:
        mpi: pion mass
        q: pion momentum
        L: lattice size

    Returns:
        The energy.
    """
    return 2.*np.sqrt(mpi*mpi + 4.*q*q*np.pi*np.pi/(float(L)*float(L)))

@memoize(50)
def EfromMpi_lat(mpi, q, L):
    """Calculates the center of mass energy for a pion with momentum q using
    the lattice dispersion relation.

    Args:
        mpi: pion mass
        q: pion momentum
        L: lattice size

    Returns:
        The energy.
    """
    return 2. * np.arccosh(np.cosh(mpi) + 2. * np.sin(q * np.pi / float(L))**2)

@memoize(50)
def calc_gamma(q2, mpi, L, d):
    """Calculates the Lorentz boost factor for the given energy and momentum.

    Args:
        q2: the momentum squared
        mpi: the pion mass
        L: the lattice size
        d: the total momentum vector of the system

    Returns:
        The Lorentz boost factor.
    """
    E = EfromMpi(mpi, np.sqrt(q2), L)
    return WfromE(E, d, L) / E
@memoize(200)
def SinglePoints(mpi, L, N, P=1):
    """Calculates the singular points for moving frames.
    Only calculates up to N=6 for MF1 and to N=7 for MF2 and MF3.
    Higher moving frames not available yet.

    Args:
        mpi: lattice pion mass
        L: lattice size
        N: the singular point to be calculated 
        P: The squared total momentum.

    Returns:
        Nth singular point for the Pth moving frame.
    """
    if P == 1 and N > 6:
        print("SinglePoints only calculates the lowest 7 points for MF1")
        os.sys.exit(-5)
    elif (P == 2 or P == 3) and (N > 7):
        print("SinglePoints only calculates the lowest 8 points for MF%1d" % P)
        os.sys.exit(-5)
    # lowest momenta squared for two free particles
    if P == 1:
        k1 = np.array([0. ,1., 1., 2., 2., 3., 4.])
        k2 = np.array([1. ,2., 4., 3., 5., 6., 5.])
    elif P == 2:
        k1 = np.array([0. ,1., 1., 2., 1., 2., 2., 3.])
        k2 = np.array([2. ,1., 3., 2., 5., 4., 6., 5.])
    elif P == 3:
        k1 = np.array([0. ,1., 1., 2., 3., 2., 3., 5.])
        k2 = np.array([3. ,2., 6., 5., 4., 9., 8., 6.])
    piL = np.pi*np.pi / (float(L) * float(L))
    mpi2 = mpi*mpi
    W = np.sqrt(k1[N] * 4. * piL + mpi2) + np.sqrt(k2[N] * 4. * piL + mpi2)
    E = np.sqrt(W*W - 4. * piL)
    q2 = ((E*E - 2. * mpi2)*(E*E - 2. * mpi2) - 4. * mpi2*mpi2) / (4. * E*E)
    return q2 * float(L)*float(L) / (4. * np.pi*np.pi)

@memoize(100)
def SinglePointsP1(mpi, L, N):
    """Calculates the singular points for first moving frame.
    Only calculates up to N=6.

    Args:
        mpi: lattice pion mass
        L: lattice size
        N: the singular point to be calculated 

    Returns:
        nth singular point
    """
    if (N > 6):
        print("SinglePointsP1 only calculates the lowest 7 points")
        os.sys.exit(-5)
    # lowest momenta squared for two free particles for total momentum 1
    k1 = np.array([0. ,1., 1., 2., 2., 3., 4.])
    k2 = np.array([1. ,2., 4., 3., 5., 6., 5.])
    piL = np.pi*np.pi / (float(L) * float(L))
    mpi2 = mpi*mpi
    W = np.sqrt(k1[N] * 4. * piL + mpi2) + np.sqrt(k2[N] * 4. * piL + mpi2)
    E = np.sqrt(W*W - 4. * piL)
    q2 = ((E*E - 2. * mpi2)*(E*E - 2. * mpi2) - 4. * mpi2*mpi2) / (4. * E*E)
    return q2 * float(L)*float(L) / (4. * np.pi*np.pi)

@memoize(100)
def SinglePointsP2(mpi, L, N):
    """Calculates the singular points for second moving frame
    Only calculates up to N=7.

    Args:
        mpi: lattice pion mass
        L: lattice size
        N: the singular point to be calculated 

    Returns:
        nth singular point
    """
    if (N > 7):
        print("SinglePointsP2 only calculates the lowest 8 points")
        os.sys.exit(-5)
    # lowest momenta squared for two free particles for total momentum 2
    k1 = np.array([0. ,1., 1., 2., 1., 2., 2., 3.])
    k2 = np.array([2. ,1., 3., 2., 5., 4., 6., 5.])
    piL = np.pi*np.pi / (float(L) * float(L))
    mpi2 = mpi*mpi
    W = np.sqrt(k1[N] * 4. * piL + mpi2) + np.sqrt(k2[N] * 4. * piL + mpi2)
    E = np.sqrt(W*W - 2. * 4. * piL)
    q2 = ((E*E - 2. * mpi2)*(E*E - 2. * mpi2) - 4. * mpi2*mpi2) / (4. * E*E)
    return q2 * float(L)*float(L) / (4. * np.pi*np.pi)

@memoize(100)
def SinglePointsP3(mpi, L, N):
    """Calculates the singular points for third moving frame
    Only calculates up to N=7.

    Args:
        mpi: lattice pion mass
        L: lattice size
        N: the singular point to be calculated 

    Returns:
        nth singular point
    """
    if (N > 7):
        print("SinglePointsP3 only calculates the lowest 8 points")
        os.sys.exit(-5)
    # lowest momenta squared for two free particles for total momentum 3
    k1 = np.array([0. ,1., 1., 2., 3., 2., 3., 5.])
    k2 = np.array([3. ,2., 6., 5., 4., 9., 8., 6.])
    piL = np.pi*np.pi / (float(L) * float(L))
    mpi2 = mpi*mpi
    W = np.sqrt(k1[N] * 4. * piL + mpi2) + \
        np.sqrt(k2[N] * 4. * piL + mpi2)
    E = np.sqrt(W*W - 3. * 4. * piL)
    q2 = ((E*E - 2. * mpi*mpi)*(E*E - 2. * mpi*mpi) - 4. * mpi2*mpi2) / (4.*E*E)
    return q2 * float(L)*float(L) / (4. * np.pi*np.pi)

@memoize(15000)
def omega(q2, gamma=None, l=0, m=0, d=np.array([0., 0., 0.]), m_split=1.,
         prec=10e-6, verbose=False):
    """Calculates the Zeta function including the some prefactor.

    Args:
        q2: The squared momentum transfer.
        gamma: The Lorentz boost factor.
        l, m: The quantum numbers.
        d: The total momentum vector of the system.
        m_split: The mass difference between the particles.
        prec: The calculation precision.
        verbose: The amount of info printed.

    Returns:
        The value of the Zeta function.
    """
    factor = gamma * np.power(np.sqrt(q2), l+1) * np.power(np.pi, 1.5) * np.sqrt(2*l+1)
    var =  zeta.Z(q2, gamma, l, m, d, m_split, prec, verbose)
    return var / factor

@memoize(50)
def det000(L, mpi, a0, r0, q2):
    """Calculates the determinant equation for CMF in A1 irrep.

    Args:
        mpi: lattice pion mass
        L: lattice size
        a0: scattering length for l=0
        r0: scattering radius for l=0
        q2: momentum squared

    Returns:
        The value of the determinant equation for the given parameters.
    """
    d = np.array([0., 0., 0.])
    omega00 = omega(q2, gamma=1., l=0, m=0, d=d)
    q = np.sqrt(q2) * 2. * np.pi / float(L)
    delta = a0 / q + 0.5 * r0 * q
    return (omega00 - delta).real

@memoize(50)
def det000_E(L, mpi, a2, r2, q2):
    """Calculates the determinant equation for CMF in E irrep.

    Args:
        mpi: lattice pion mass
        L: lattice size
        a2: scattering length for l=2
        r2: scattering radius for l=2
        q2: momentum squared

    Returns:
        The value of the determinant equation for the given parameters.
    """
    d = np.array([0., 0., 0.])
    omega00 = omega(q2, gamma=1., l=0, m=0, d=d)
    omega40 = omega(q2, gamma=1., l=4, m=0, d=d)
    q = np.sqrt(q2) * 2. * np.pi / float(L)
    delta = a2 / q**5 + 0.5 * r2 / q**3
    return (omega00 + 18. * omega40 / 7. - delta).real

@memoize(50)
def det000_T2(L, mpi, a2, r2, q2):
    """Calculates the determinant equation for CMF in T2 irrep.

    Args:
        mpi: lattice pion mass
        L: lattice size
        a2: scattering length for l=2
        r2: scattering radius for l=2
        q2: momentum squared

    Returns:
        The value of the determinant equation for the given parameters.
    """
    d = np.array([0., 0., 0.])
    omega00 = omega(q2, gamma=1., l=0, m=0, d=d)
    omega40 = omega(q2, gamma=1., l=4, m=0, d=d)
    q = np.sqrt(q2) * 2. * np.pi / float(L)
    delta = a2 / q**5 + 0.5 * r2 / q**3
    return (omega00 - 12. * omega40 / 7. - delta).real

@memoize(50)
def det001(L, mpi, a0, r0, a2, r2, q2):
    """Calculates the determinant equation for MF1 in A1 irrep.

    Args:
        mpi: lattice pion mass
        L: lattice size
        a0: scattering length for l=0
        r0: scattering radius for l=0
        a2: scattering length for l=2
        r2: scattering radius for l=2
        q2: momentum squared

    Returns:
        The value of the determinant equation for the given parameters.
    """
    d = np.array([0., 0., 1.])
    gamma = calc_gamma(q2, mpi, L, d)
    omega00 = omega(q2, gamma=gamma, l=0, m=0, d=d)
    omega20 = omega(q2, gamma=gamma, l=2, m=0, d=d)
    omega40 = omega(q2, gamma=gamma, l=4, m=0, d=d)
    q = np.sqrt(q2) * 2. * np.pi / float(L)
    delta0 = a0 / q + 0.5 * r0 * q
    delta2 = a2 / q**5 + 0.5 * r2 / q**3
    return ((omega00 - delta0) * (omega00 + 10. * omega20 / 7. + \
           18. * omega40 / 7. - delta2) - 5. * omega20**2).real

@memoize(50)
def det110(L, mpi, a0, r0, a2, r2, q2):
    """Calculates the determinant equation for MF2 in A1 irrep.

    Args:
        mpi: lattice pion mass
        L: lattice size
        a0: scattering length for l=0
        r0: scattering radius for l=0
        a2: scattering length for l=2
        r2: scattering radius for l=2
        q2: momentum squared

    Returns:
        The value of the determinant equation for the given parameters.
    """
    d = np.array([1., 1., 0.])
    gamma = calc_gamma(q2, mpi, L, d)
    omega00 = omega(q2, gamma=gamma, l=0, m=0, d=d)
    omega20 = omega(q2, gamma=gamma, l=2, m=0, d=d)
    omega22 = omega(q2, gamma=gamma, l=2, m=2, d=d)
    omega40 = omega(q2, gamma=gamma, l=4, m=0, d=d)
    omega42 = omega(q2, gamma=gamma, l=4, m=2, d=d)
    omega44 = omega(q2, gamma=gamma, l=4, m=4, d=d)
    q = np.sqrt(q2) * 2. * np.pi / float(L)
    delta0 = a0 / q + 0.5 * r0 * q
    delta2 = a2 / q**5 + 0.5 * r2 / q**3
    # splitted over several lines, maybe there is a way to make structure
    # more clear
    term1 =-(10. * np.sqrt(2.) * omega22 / 7. -\
           3. * np.sqrt(30.) * omega42 / 7.) * \
           ((3. * np.sqrt(30.) * omega42 / 7. -\
           10. * np.sqrt(2.) * omega22 / 7.) * (omega00 - delta0) -\
           5. * np.sqrt(2.) * omega20 * omega22)
    term2 = np.sqrt(5.) * omega20 *\
           (-np.sqrt(5.) * delta2 * omega20 -\
           10. * np.sqrt(5.) * omega20**2 / 7. +\
           np.sqrt(5.) * omega00 * omega20 +\
           3. * np.sqrt(5.0) * omega40 * omega20 / 7. -\
           15. * np.sqrt(2.0) * omega44 * omega20 / np.sqrt(7.) -\
           20. * np.sqrt(5.0) * omega22**2 / 7. +\
           30. * np.sqrt(3.0) * omega22 * omega42 / 7.)
    term3 = (omega00 + 10. * omega20 / 7. + 18. * omega40 / 7. - delta2) *\
           ((omega00 - delta0) *\
           (omega00 - 10. * omega20 / 7. + 3. * omega40 / 7. -\
           3. * np.sqrt(10.) * omega44 / np.sqrt(7.) - delta2) +\
           10. * omega22**2)
    return (term1 - term2 + term3).real

@memoize(50)
def det111(L, mpi, a0, r0, a2, r2, q2):
    """Calculates the determinant equation for MF3 in A1 irrep.

    Args:
        mpi: lattice pion mass
        L: lattice size
        a0: scattering length for l=0
        r0: scattering radius for l=0
        a2: scattering length for l=2
        r2: scattering radius for l=2
        q2: momentum squared

    Returns:
        The value of the determinant equation for the given parameters.
    """
    d = np.array([1., 1., 1.])
    gamma = calc_gamma(q2, mpi, L, d)
    omega00 = omega(q2, gamma=gamma, l=0, m=0, d=d)
    omega22 = omega(q2, gamma=gamma, l=2, m=2, d=d)
    omega40 = omega(q2, gamma=gamma, l=4, m=0, d=d)
    omega42 = omega(q2, gamma=gamma, l=4, m=2, d=d)
    q = np.sqrt(q2) * 2. * np.pi / float(L)
    delta0 = a0 / q + 0.5 * r0 * q
    delta2 = a2 / q**5 + 0.5 * r2 / q**3
    return ( (omega00 - delta0) *\
        (omega00 - 12. * omega40 / 7. - 12.j * np.sqrt(10.) * omega42 / 7. -\
        10.j * np.sqrt(6.) * omega22 / 7. - delta2) +\
        30. * omega22**2 ).real

@memoize(500)
def root(L, mpi, a0, r0, a2, r2, d=np.array([0., 0., 0.]), irrep="A1", n=1):
    """Returns roots of the determinant equation.
    
    Args:
        L: lattice size
        mpi: lattice pion mass
        a0: scattering length for l=0
        r0: scattering radius for l=0
        a2: scattering length for l=2
        r2: scattering radius for l=2
        d: total momentum of the system
        irrep: the chosen irrep
        n: number of roots to look for

    Returns:
        The values of the roots.
    """
    #print("Entering root")
    #print(irrep, L, d, n)
    # These variables were global variables
    r_prec = 1e-10
    # setup of variables
    nroot = 0
    roots = np.zeros(n)
    # CJ: Used lamda functions to make code more compact
    if (irrep == "A1"):
        if (np.array_equal(d, np.array([0., 0., 0.]))):
            calc_det = lambda q: det000(L, mpi, a0, r0, q)
            singular_points = lambda i: float(i)
            n_interval = 5
            n_blocks = 10
        elif (np.array_equal(d, np.array([0., 0., 1.]))):
            calc_det = lambda q: det001(L, mpi, a0, r0, a2, r2, q)
            singular_points = lambda i: SinglePointsP1(mpi, L, i)
            n_interval = 6
            n_blocks = 20
        elif (np.array_equal(d, np.array([1., 1., 0.]))):
            calc_det = lambda q: det110(L, mpi, a0, r0, a2, r2, q)
            singular_points = lambda i: SinglePointsP2(mpi, L, i)
            n_interval = 7
            n_blocks = 20
        elif (np.array_equal(d, np.array([1., 1., 1.]))):
            calc_det = lambda q: det111(L, mpi, a0, r0, a2, r2, q)
            singular_points = lambda i: SinglePointsP3(mpi, L, i)
            n_interval = 7
            n_blocks = 20
        else:
            print("wrong value of dVec")
            os.sys.exit(-5)
    elif (irrep == "E"):
        if (np.array_equal(d, np.array([0., 0., 0.]))):
            calc_det = lambda q: det000_E(L, mpi, a2, r2, q)
            singular_points = lambda i: float(i)
            n_interval = 5
            n_blocks = 10
        else:
            print("wrong value of dVec")
            os.sys.exit(-5)
    elif (irrep == "T2"):
        if (np.array_equal(d, np.array([0., 0., 0.]))):
            calc_det = lambda q: det000_T2(L, mpi, a2, r2, q)
            singular_points = lambda i: float(i)
            n_interval = 5
            n_blocks = 10
        else:
            print("wrong value of dVec")
            os.sys.exit(-5)
    else:
        print("wrong irrep")

    # set up grid
    q2_range_min = np.zeros(n_interval*n_blocks)
    q2_range_max = np.zeros(n_interval*n_blocks)
    for i in range(n_interval):
        q2_min = singular_points(i) + 10e-4
        q2_max = singular_points(i+1) - 10e-4
        q2_delta = (q2_max - q2_min) / float(n_blocks)
        for j in range(n_blocks):
            q2_range_min[i*n_blocks + j] = q2_min + j * q2_delta
            q2_range_max[i*n_blocks + j] = q2_min + (j + 1) * q2_delta

    # main loop
    loop_i = 0
    while(nroot < n):
        det1 = calc_det(q2_range_min[loop_i])
        det2 = calc_det(q2_range_max[loop_i])
        if (det1 * det2 < 0):
            try:
                roots[nroot] = opt.brentq(calc_det, q2_range_min[loop_i], \
                                          q2_range_max[loop_i], disp=True)
                nroot += 1
            except RuntimeError:
                print("next loop")
        loop_i += 1
        if(loop_i == q2_range_min.shape[0]):
            print("root out of range. d = (%lf, %lf, %lf)" % (d[0], d[1], d[2]))
            os.sys.exit(-5)
    return roots

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
    Wroot = ma.zeros(np.sum(len(t[-1]) for t in infolist))
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
    chi = ma.dot((data[N] - Wroot), ma.dot(cov, (data[N] - Wroot)))
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
    print(par)
    calc_root = lambda p, i: root(s[0], mpi[N,s[1]], p[0], p[1], p[2], 0., \
                                  s[3], s[2], i)
    verbose = False
    Wroot = ma.zeros(np.sum(len(t[-1]) for t in infolist))
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
        print("")
    for Edata, Ecalc in zip(data[N], Wroot):
        print("%.7lf, %.7lf, %.4e\n" % (Edata, Ecalc, abs(Edata - Ecalc)))

    # calculate chi^2
    chi = ma.dot((data[N] - Wroot), ma.dot(cov, (data[N] - Wroot)))
    print(chi)
    return chi
