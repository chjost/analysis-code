################################################################################
#
# Author: Christian Jost (jost@hiskp.uni-bonn.de)
# Date:   Februar 2015
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
# Function: A set of functions to analyze correlation functions.
#
# For informations on input parameters see the description of the function.
#
################################################################################

import os
import numpy as np
import zeta

def average_corr_fct(data, nbcfg, T):
    """Average over the set of correlation functions.

    Args:
        data: The data to average over in a numpy array.
        nbcfg: The number of configurations.
        T: The time extent of the data.

    Returns:
        A numpy array averaged over one axis.
    """
    average = np.zeros((T))
    for _t in range(T):
        average[int(_t)] = np.average(data[_t*nbcfg:(_t+1)*nbcfg])
    return average

def simple_ratio(d1, d2, d3):
    """Calculates a simple ratio of three data sets.

    Calculates d1(t)/(d2(t)*d3(t)).
    Assumes that all data sets are numpy arrays with two axis, the first being
    the bootstrap number and the second being the time.
    
    Args:
        d1, d2, d3: The three data sets.

    Returns:
        The ratio, its mean and its standard deviation.
    """
    # create array from dimensions of the data
    ratio = np.zeros((d1.shape[0], d1.shape[1]))
    for _s in range(d1.shape[0]):
        for _t in range(d1.shape[1]):
            # calculate ratio
            ratio[_s,_t] = d1[_s,_t]/(d2[_s,_t]*d3[_s,_t])
    # TODO(CJ): test if the following give the same result
    # ratio = d1/(d2*d3)
    # get mean and standard deviation
    mean, err = return_mean_corr(ratio)
    return ratio, mean, err

def simple_ratio_subtract(d1, d2, d3):
    """Calculates a simple ratio of three data sets, combining two time slices.

    Calculates [d1(t)-d1(t+1)]/[(d2(t)*d3(t))-(d2(t+1)*d3(t+1))].
    Assumes that all data sets are numpy arrays with two axis, the first being
    the bootstrap number and the second being the time. The time extend is
    reduced by one, as the ratio cannot be calculated on the last slice.
    
    Args:
        d1, d2, d3: The three data sets.

    Returns:
        The ratio, its mean and its standard deviation.
    """
    # create array from dimensions of the data
    ratio = np.zeros((d1.shape[0], d1.shape[1]-1))
    for _s in range(d1.shape[0]):
        for _t in range(d1.shape[1]-1):
            # calculate ratio
            ratio[_s,_t] = (d1[_s,_t] - d1[_s,_t+1]) / ((d2[_s,_t]*d3[_s,_t]) -
                            (d2[_s,_t+1]*d3[_s,_t+1]))
    # get mean and standard deviation
    mean, err = return_mean_corr(ratio)
    return ratio, mean, err

def ratio(d1, d2, d3, dE):
    """Calculates a ratio of three data sets, combining two time slices and
       the energy difference between the energy levels.

    Calculates [d1(t)-d1(t+1)]/[(d2(t)*d3(t))-(d2(t+1)*d3(t+1))].
    Assumes that all data sets are numpy arrays with two axis, the first being
    the bootstrap number and the second being the time. The time extend is
    reduced by one, as the ratio cannot be calculated on the last slice.
    
    Args:
        d1, d2, d3: The three data sets.
        dE: Energy difference between the the data sets d2 and d3.

    Returns:
        The ratio, its mean and its standard deviation.
    """
    # create array from dimensions of the data
    ratio = np.zeros((d1.shape[0], d1.shape[1]-1))
    for _s in range(d1.shape[0]):
        for _t in range(d1.shape[1]-1):
            # calculate numerator and denominator first
            num=(d1[_s,_t] - d1[_s,_t+1] * np.exp(dE * (_t+1))) * np.exp(-dE*_t)
            den=(d2[_s,_t] * d3[_s,_t] - d2[_s,_t+1] * d3[_s,_t+1] *
                 np.exp(dE*(_t+1))) * np.exp(-dE*_t)
            # calculate ratio
            ratio[_s,_t] = num/den
    # get mean and standard deviation
    mean, err = return_mean_corr(ratio)
    return ratio, mean, err

def compute_derivative(data):
    """Computes the derivative of a correlation function.

    The data is assumed to a numpy array with datastrap samples as first axis
    and time as second axis.  The time extend of the mass reduced by one
    compared with the original data since the energy cannot be calculated on
    the first slice.

    Args:
        data: The datastrapped correlation function.

    Returns:
        The derivative of the correlation function and its mean and standard
        deviation.
    """
    # creating derivative array from data array
    derv = np.empty([data.shape[0], data.shape[1]-1], dtype=float)
    # computing the derivative
    for b in range(0, data.shape[0]):
        row = data[b,:]
        for t in range(0, len(row)-1):
            derv[b, t] = row[t+1] - row[t]
    mean, err = print_mean_corr(derv, msg="\nderivative:\n-------------------\n")
    return derv, mean, err

def compute_mass(data):
    """Computes the energy of a correlation function.

    The data is assumed to a numpy array with bootstrap samples as first axis
    and time as second axis.  The time extend of the mass reduced by two
    compared with the original data since the energy cannot be calculated on
    the first and last slice.

    Args:
        data: The bootstrapped correlation function.

    Returns:
        The energy of the correlation function and its mean and standard
        deviation.
    """
    # creating mass array from data array
    mass = np.zeros([data.shape[0], data.shape[1]-2], dtype=float)
    # computing the energy via formula
    for b in range(0, data.shape[0]):
        row = data[b,:]
        for t in range(1, len(row)-1):
            mass[b, t-1] = (row[t-1] + row[t+1])/(2.0*row[t])
    mass = np.arccosh(mass)
    # print energy
    mean, err = return_mean_corr(mass)
    return mass, mean, err

def calculate_cm_energy(E, L, d=np.array([0., 0., 1.])):
    """Calculates the center of mass energy and the boost factor.

    Calculates the Lorentz boost factor and the center of mass energy
    for moving frames.

    Args:
        E: The energy of the moving frame.
        d: The total momentum of the moving frame.

    Returns:
        The boost factor and the center of mass energies.
    """
    # if the data is from the cm system, return immediately
    if not np.any(d):
        gamma = np.ones(E.shape[0])
        return gamma, E
    # create the array for results
    Ecm = np.zeros((E.shape[0]))
    gamma = np.zeros((E.shape[0]))
    # norm of the cm momentum
    _p2 = np.dot(d, d) * (4 * np.pi**2 / float(L)**2)
    for _i in range(E.shape[0]):
        # calculate center of mass energy
        Ecm[_i] = np.sqrt(E[_i]**2 - _p2)
        gamma[_i] = Ecm[_i] / E[_i]
    return gamma, Ecm

def calculate_q(E, mpi, L):
    """Calculates q.

    Calculates the difference in momentum between interaction and non-
    interacting systems. The energy must be the center of mass energy.

    Args:
        E: The energy values for the bootstrap samples.
        mpi: The pion mass of the lattice.
        L: The spatial extent of the lattice.

    Returns:
        An array of the q^2 values
    """
    # create array for results
    q2 = np.zeros((E.shape[0]))
    # continuum dispersion relation
    # this does not change, calculate only once
    _p = float(L) / (2. * np.pi)
    for _i in range(E.shape[0]):
        # calculate q
        q2[_i] = (0.25*E[_i]**2 - mpi**2) * _p**2
    # lattice dispersion relation
    #for _i in range(E.shape[0]):
    #    # calculate q
    #    q2[_i] = 4*np.arcsin(np.sqrt((np.cosh(E[_i]/2.) - np.cosh(mpi))/2.))**2
    return q2

def calculate_delta(q2, gamma=None, l=0, m=0, d=np.array([0., 0., 0.]),
                    prec=10e-6, verbose=0):
    """Calculates the phase shift using Luescher's Zeta function.

    Most arguments are for the Zeta function. For each q2 a gamma is needed.

    Args:
        q2: The momentum shift squared.
        gamma: The Lorentz factor for moving frames.
        l: The orbital quantum number.
        m: The magnetic quantum number.
        d: The total three momentum of the system.
        prec: The precision of the Zeta function calculation.
        verbose: The amount of information printed to screen.

    Returns:
        An array of the phase shift, its mean and standard deviation and
        tan(delta).
    """
    # create array for results
    delta=np.zeros(q2.shape[0])
    tandelta=np.zeros(q2.shape[0])
    _gamma = gamma
    if _gamma == None:
        _gamma = np.ones(q2.shape[0])
    # read in momenta
    # file name must not change!
    pathmomenta = "./momenta.npy"
    if not os.path.isfile(pathmomenta):
        import create_momentum_array
    _n = np.load(pathmomenta)
    _pi3 = np.pi**3
    for _i in range(q2.shape[0]):
        _z = zeta.Z(q2[_i], _gamma[_i], l, m, d, 1., prec, verbose, _n)
        tandelta[_i] = np.sqrt( _pi3 * q2[_i]) / _z.real
        delta[_i] = np.arctan2(np.sqrt( _pi3 * q2[_i]), _z.real)
    return delta, tandelta

def return_mean_corr(data, axis=0):
    """Calculates the mean and standard deviation of the correlation function.

    Args:
        data: The data on which the mean is calculated.
        axis: Along which axis the mean is calculated.

    Returns:
        The mean and standard deviation of the data.
    """
    # calculate mean and standard deviation
    mean = np.mean(data, axis, dtype=np.float64)
    err  = np.std(data, axis, dtype=np.float64)
    return mean, err

def print_mean_corr(data, axis=0, msg=""):
    """Prints the mean and standard deviation of the correlation function.

    Args:
        data: The data on which the mean is calculated.
        axis: Along which axis the mean is calculated.
        msg: Message printed before data is written to screen.

    Returns:
        The mean and standard deviation of the data.
    """
    if msg != "":
        print("\nmean correlator:\n----------------\n")
    else:
        print(str(msg))
    mean, err = return_mean_corr(data, axis)
    # print the mean and standard deviation
    for t, m, e in zip(range(0, len(mean)), mean, err):
        print("%2d %.6e %.6e" % (t, m, e))
    return mean, err
