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

def compute_derivative(data):
    """Computes the derivative of a correlation function.

    The data is assumed to a numpy array with datastrap samples as first axis
    and time as second axis.  The time extend of the mass reduced by one
    compared with the original data since the energy cannot be calculated on
    the first slice.

    Args:
        data: The bootstrapped correlation function.

    Returns:
        The derivative of the correlation function and its mean and standard
        deviation.
    """
    # creating derivative array from data array
    derv = np.zeros_like(data[:,:-1], dtype=float)
    # computing the derivative
    for b in range(0, data.shape[0]):
        row = data[b]
        for t in range(0, len(row)-1):
            derv[b, t] = row[t+1] - row[t]
    mean, err = calc_error(derv)
    return derv, mean, err

def compute_mass(data, usecosh=True):
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
    mass = np.zeros_like(data[:,1:-1])
    # computing the energy
    if usecosh:
       for b in range(0, data.shape[0]):
           row = data[b,:]
           for t in range(1, len(row)-1):
               mass[b, t-1] = (row[t-1] + row[t+1])/(2.0*row[t])
       mass = np.arccosh(mass)
    else:
       for b in range(0, data.shape[0]):
           row = data[b,:]
           for t in range(1, len(row)-1):
               mass[b, t-1] = np.log(row[t]/row[t+1])
    # print energy
    mean, err = calc_error(mass)
    return mass, mean, err

def calc_error(data, axis=0):
    """Calculates the mean and standard deviation of the correlation function.

    Args:
        data: The data on which the mean is calculated.
        axis: Along which axis the mean is calculated.

    Returns:
        The mean and standard deviation of the data.
    """
    # calculate mean and standard deviation
    mean = np.mean(data, axis)
    err  = np.std(data, axis)
    return mean, err

def square(sample):
  return map(lambda x: x**2, sample)

def calc_scat_length(dE, E, L):
    """Finds root of the Energyshift function up to order L^{-6} only applicable
    in 0 momentum case, effective range not included!
       Args:
           dE: the energy shift of the system due to interaction
           E: the single particle energy
           L: The spatial lattice extent
       Returns:
           a: roots of the function
    """
    # coefficients according to Luescher
    c=[-2.837297, 6.375183, -8.311951]
    # creating data array from empty array
    a = np.zeros_like(dE[:])
    for b in range(0, dE.shape[0]):
    # Prefactor common to every order
      comm=-4.*np.pi/(E[b]*L*L)
      # build up coefficient array
      p=[comm*c[1]/(L*L*L),comm*c[0]/(L*L),comm/L, -1*dE[b]]
      a[b] = np.roots(p)[2].real
    return a
