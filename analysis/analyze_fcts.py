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

def sys_error_der(data, weights, d, lattice, path="./plots/"):
    """Error calculation for derived results
    
    Args:
        data: the derived data
        weight: a (multidimensional) np-array holding the combination of weights
        for all combination of fit intervals

    Returns:
        res: The weighted median value on the original data
        res_std: The standard deviation derived from the deviation of 
              medians on the bootstrapped data.
        res_syst: 1 sigma systematic uncertainty is the difference 
              res - 16%-quantile or 84%-quantile - res respectively
        
    """
    # check the deepness of the list structure
    depth = lambda L: isinstance(L, list) and max(map(depth, L))+1
    deep = depth(data)
    if deep == 1:
        # initialize empty arrays
        res, res_std, res_sys = [], [], []
        # loop over principal correlators
        for i, p in enumerate(data):
            # append the necessary data arrays
            res.append(np.zeros(p.shape[0]))
            res_std.append(np.zeros((1,)))
            res_sys.append(np.zeros((2,)))

            # draw data in histogram
            plotlabel = 'hist_der_'
            label = ["", "", "principal correlator"]
            plt.plot_histogram(p[0], weights[i], lattice, d, label, path,
                               plotlabel)
            # using the weights, calculate the median over all fit intervals
            # for every bootstrap sample.
            for b in xrange(p.shape[0]):
                res[i][b] = qlt.weighted_quantile(p[b], weights[i], 0.5)
            # the statistical error is the standard deviation of the medians
            # over the bootstrap samples.
            res_std[i] = np.std(res[i])
            # the systematic error is given by difference between the median 
            # on the original data and the 16%- or 84%-quantile respectively
            res_sys[i][0]=res[i][0]-qlt.weighted_quantile(p[0],weights[i],0.16)
            res_sys[i][1]=qlt.weighted_quantile(p[0],weights[i],0.84)-res[i][0]
            # keep only the median of the original data
            res[i] = res[i][0]
    elif deep == 2:
        # initialize empty arrays
        res, res_std, res_sys = [], [], []
        # loop over principal correlators
        for i, p in enumerate(data):
            res.append([])
            res_std.append([])
            res_sys.append([])
            for j, q in enumerate(p):
                # append the necessary data arrays
                res[i].append(np.zeros(q.shape[0]))
                res_std[i].append(np.zeros((1,)))
                res_sys[i].append(np.zeros((2,)))

                # draw data in histogram
                plotlabel = 'hist_der_'
                label = ["", "", "principal correlator"]
                plt.plot_histogram(q[0], weights[i][j], lattice, d, label, path,
                                   plotlabel)
                # using the weights, calculate the median over all fit intervals
                # for every bootstrap sample.
                for b in xrange(q.shape[0]):
                    res[i][j][b] = qlt.weighted_quantile(q[b].ravel(), weights[i][j].ravel(), 0.5)
                # the statistical error is the standard deviation of the medians
                # over the bootstrap samples.
                res_std[i][j] = np.std(res[i][j])
                # the systematic error is given by difference between the median 
                # on the original data and the 16%- or 84%-quantile respectively
                res_sys[i][j][0]=res[i][j][0]-qlt.weighted_quantile(q[0].ravel(),weights[i][j].ravel(),0.16)
                res_sys[i][j][1]=qlt.weighted_quantile(q[0].ravel(),weights[i][j].ravel(),0.84)-res[i][j][0]
                # keep only the median of the original data
                res[i][j] = res[i][j][0]
    return res, res_std, res_sys

def sys_error(data, pvals, d, lattice, par=0, path="./plots/"):
    """Calculates the statistical and systematic error of an np-array of 
    fit results on bootstrap samples of a quantity and the corresponding 
    p-values.

    Args:
        data: A numpy array with three axis. The first axis is the bootstrap 
              sample number, the second axis is the number of correlators, the third axis is the fit range index
        pvals: The p value indicating the quality of the fit.
        lattice: The name of the lattice, used for the output file.
        d:    The total momentum of the reaction.
        par: which parameter to plot (second index of data arrays)
        path: path where the plots are saved

    Returns:
        res: The weighted median value on the original data
        res_std: The standard deviation derived from the deviation of 
              medians on the bootstrapped data.
        res_syst: 1 sigma systematic uncertainty is the difference 
              res - 16%-quantile or 84%-quantile - res respectively
        weights: numpy array of the calculated weights for every bootstrap
        sample and fit range
    """
    # check the deepness of the list structure
    depth = lambda L: isinstance(L, list) and max(map(depth, L))+1
    deep = depth(data)
    if deep == 1:
        # initialize empty arrays
        data_weight = []
        res, res_std, res_sys = [], [], []
        # loop over principal correlators
        for i, p in enumerate(data):
            # append the necessary data arrays
            data_weight.append(np.zeros((p.shape[-1])))
            res.append(np.zeros(p.shape[0]))
            res_std.append(np.zeros((1,)))
            res_sys.append(np.zeros((2,)))

            # calculate the weight for the fit ranges using the standard
            # deviation and the p-values of the fit
            data_std = np.std(p[:,par])
            data_weight[i] = (1. - 2. * np.fabs(pvals[i][0] - 0.5) *
                              np.amin(data_std) / data_std)**2
            # draw data in histogram
            plotlabel = 'hist_%d' % i
            label = ["", "", "principal correlator"]
            plt.plot_histogram(p[0,par], data_weight[i], lattice, d, label, path,
                               plotlabel)
            # using the weights, calculate the median over all fit intervals
            # for every bootstrap sample.
            for b in xrange(p.shape[0]):
                res[i][b] = qlt.weighted_quantile(p[b,par], data_weight[i], 0.5)
            # the statistical error is the standard deviation of the medians
            # over the bootstrap samples.
            res_std[i] = np.std(res[i])
            # the systematic error is given by difference between the median 
            # on the original data and the 16%- or 84%-quantile respectively
            res_sys[i][0]=res[i][0]-qlt.weighted_quantile(p[0,par],data_weight[i],0.16)
            res_sys[i][1]=qlt.weighted_quantile(p[0,par],data_weight[i],0.84)-res[i][0]
            # keep only the median of the original data
            res[i] = res[i][0]
    elif deep == 2:
        # initialize empty arrays
        data_weight = []
        res, res_std, res_sys = [], [], []
        # loop over principal correlators
        for i, p in enumerate(data):
            data_weight.append([])
            res.append([])
            res_std.append([])
            res_sys.append([])
            for j, q in enumerate(p):
                # append the necessary data arrays
                data_weight[i].append(np.zeros(q.shape[-2:]))
                res[i].append(np.zeros(q.shape[0]))
                res_std[i].append(np.zeros((1,)))
                res_sys[i].append(np.zeros((2,)))

                # calculate the weight for the fit ranges using the standard
                # deviation and the p-values of the fit
                data_std = np.std(q[:,par], axis=0)
                data_weight[i][j] = (1. - 2. * np.fabs(pvals[i][j][0] - 0.5) *
                                  np.amin(data_std) / data_std)**2
                # draw data in histogram
                plotlabel = 'hist_%d_%d' % (i, j)
                label = ["", "", "principal correlator"]
                plt.plot_histogram(q[0,par].ravel(), data_weight[i][j].ravel(), lattice, d, label, path, plotlabel)
                # using the weights, calculate the median over all fit intervals
                # for every bootstrap sample.
                for b in xrange(q.shape[0]):
                    res[i][j][b] = qlt.weighted_quantile(q[b,par].ravel(), 
                                       data_weight[i][j].ravel(), 0.5)
                # the statistical error is the standard deviation of the medians
                # over the bootstrap samples.
                res_std[i][j] = np.std(res[i][j])
                # the systematic error is given by difference between the median
                # on the original data and the 16%- or 84%-quantile respectively
                res_sys[i][j][0] = res[i][j][0] - qlt.weighted_quantile(q[0,par].ravel(), data_weight[i][j].ravel(), 0.16)
                res_sys[i][j][1] = qlt.weighted_quantile(q[0,par].ravel(), data_weight[i][j].ravel(), 0.84) - res[i][j][0]
                # keep only the median of the original data
                res[i][j] = res[i][j][0]
    else:
        print("made for lists of depth < 3")
        os.sys.exit(-10)
    return res, res_std, res_sys, data_weight

def square(sample):
  return map(lambda x: x**2, sample)

def calc_scat_length(dE, E, weight_dE, weight_E, L, pars=(1, 0)):
    """Finds root of the Energyshift function up to order L^{-5} only applicable
    in 0 momentum case, effective range not included!
    Only working for lists of lists.

    Args:
        dE: the energy shift of the system due to interaction
        E: the single particle energy
        weights_dE: weights of the energy shift
        weights_E: weights of the single particle energy
        L: The spatial lattice extent
        pars: which fit parameters to use
    Returns:
        a: roots of the function
        weights: the weight of a
    """
    ncorr_single = len(E)
    # check if dE has same length
    if len(dE) is not ncorr_single:
        print("error in calc_scat_length, data shapes incompatible")
        print(len(dE), len(E))
        os.sys.exit(-10)
    ncorr_ratio = [len(d) for d in dE]
    # check number of bootstrap samples and fit intervals for single particle
    for i in xrange(ncorr_single):
        for j in xrange(ncorr_ratio[i]):
            if E[i].shape[0] != dE[i][j].shape[0]:
                print("number of bootstrap samples is different in calc_scat_length")
                print(E[i].shape[0], dE[i][j].shape[0])
                os.sys.exit(-10)
            if E[i].shape[-1] != dE[i][j].shape[-1]:
                print("number of fit intervals is different in calc_scat_length")
                print(E[i].shape[-1], dE[i][j].shape[-1])
                os.sys.exit(-10)
    #print("scat length begin")
    ## number of correlation functions for the single particle energy
    #print(len(dE), len(E))
    ## number of correlation functions for the ratio
    #print(len(dE[0]))
    ## shape of the fit results
    #print(dE[0][0].shape, E[0].shape)
    # coefficients according to Luescher
    c=[-2.837297, 6.375183, -8.311951]
    # creating data array from empty array
    a = []
    weight = []
    for i in xrange(ncorr_single):
        a.append([])
        weight.append([])
        for j in xrange(ncorr_ratio[i]):
            a[i].append(np.zeros((E[i].shape[0], dE[i][j].shape[-2], E[i].shape[-1])))
            weight[i].append(np.zeros((dE[i][j].shape[-2], E[i].shape[-1])))
    # loop over the correlation functions
    for _r in xrange(ncorr_single): # single particle
        for _s in xrange(ncorr_ratio[_r]): # ratio
            # calculate prefactor
            # TODO(CJ): can the shape of E[i] change?
            pre = -4.*np.pi/(E[_r][:,0,:]*float(L*L*L))
            # loop over fitranges
            for _f in xrange(E[_r].shape[-1]): # single particle
                for _g in xrange(dE[_r][_s].shape[-2]): # ratio
                    # loop over bootstrap samples
                    for _b in xrange(E[_r].shape[0]):
                        p = np.asarray((pre[_b,_f]*c[1]/float(L*L), 
                            pre[_b,_f]*c[0]/float(L), pre[_b,_f],
                            -1.*dE[_r][_s][_b,0,_g,_f]))
                        # calculate roots
                        root = np.roots(p)
                        # sort according to absolute value of the imaginary part
                        ind_root = np.argsort(np.fabs(root.imag))
                        # the first entry of the sorted array is the one we want
                        a[_r][_s][_b,_g, _f] = root[ind_root][0].real
                        weight[_r][_s][_g, _f] = weight_dE[_r][_s][_g,_f] * weight_E[_r][_f]
    return a, weight
