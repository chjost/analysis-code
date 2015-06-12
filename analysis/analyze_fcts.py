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
import plot as plt
import _quantiles as qlt
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

def sys_error(data, pvals, d, lattice):
    """Calculates the statistical and systematic error of an np-array of 
    fit results on bootstrap samples of a quantity and the corresponding 
    p-values.

    Args:
        data: A numpy array with three axis. The first axis is the bootstrap 
              sample number, the second axis is the number of correlators, the third axis is 
              the fit range index

        pvals: The p value indicating the quality of the fit.
        lattice: The name of the lattice, used for the output file.
        d:    The total momentum of the reaction.

    Returns:
        res: The weighted median value on the original data
        res_std: The standard deviation derived from the deviation of 
              medians on the bootstrapped data.
        res_syst: 1 sigma systematic uncertainty is the difference 
              res - 16%-quantile or 84%-quantile - res respectively
        weights: numpy array of the calculated weights for every bootstrap
        sample and fit range
    """

    d2 = np.dot(d, d)
    
    
    data_std = np.empty([data.shape[2]], dtype=float)
    data_weight = np.empty([data.shape[2]], dtype=float)

    res = np.empty([data.shape[0], data.shape[1]], dtype=float)
    res_std = np.empty([data.shape[1]], dtype=float)
    res_syst = np.empty([2, data.shape[1]], dtype=float)

    path="./plots/"

    # loop over principal correlators
    for _j in range(0, data.shape[1]):
        # calculate the standard deviation of the bootstrap samples and from
        # that and the p-values the weight of the fit for every chosen interval
        data_std = np.std(data[:,0])
        data_weight = (1. - 2. * np.fabs(pvals[0, _j] - 0.5) *
                      np.amin(data_std)/data_std)**2
        # draw original data as histogram
        plotlabel = 'hist_%d"' % _j
        label = ["", "", "principal correlator"]
        print data[0,_j].shape, data_weight.shape
        plt.plot_histogram(data[0,_j], data_weight, lattice, d2, label, path,
                           plotlabel)

        # using the weights, calculate the median over all fit intervalls for
        # every bootstrap sample.
        for _i in range(0, data.shape[0]):
            res[_i, _j] = qlt.weighted_quantile(data[_i, _j], data_weight, 0.5)
        # the statistical error is the standard deviation of the medians over
        # the bootstrap samples.
        res_std[_j] = np.std(res[:,_j])
        # the systematic error is given by difference between the median on the 
        # original data and the 16%- or 84%-quantile respectively
        res_syst[0, _j] = res[0, _j] - qlt.weighted_quantile(data[0, _j], data_weight, 0.16)
        res_syst[1, _j] = qlt.weighted_quantile(data[0, _j], data_weight, 0.84) - res[0, _j]

#    print('res 1 %lf +- %lf (stat) + %lf - %lf (syst)' % (res[0][0], 
#          res_std, res[0][0] - res_16[0], res_84[0] - res[0][0]))

    # only median on original data is of interest later
    return res[0], res_std, res_syst, data_weight

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
    a = np.zeros_like(dE[])
    # loop over fit ranges
    for _r in range(0,dE.shape[1]):
        # loop over bootstrap samples
        for _b in range(0, dE.shape[0]):
        # Prefactor common to every order
          comm=-4.*np.pi/(E[_b,_r]*L*L)
          # build up coefficient array
          p=[comm*c[1]/(L*L*L),comm*c[0]/(L*L),comm/L, -1*dE[_b,_r]]
          a[_b,_r] = np.roots(p)[2].real
    return a
