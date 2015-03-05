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

import numpy as np

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

def compute_energy(data):
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
    mean, err = print_mean_corr(mass, msg="\nenergy:\n-------------\n")
    return mass, mean, err

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
