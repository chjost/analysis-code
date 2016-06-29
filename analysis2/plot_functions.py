"""
Functions for plotting.
"""

import numpy as np
import matplotlib.pyplot as plt

from statistics import compute_error
from functions import (func_single_corr, func_ratio, func_const, func_two_corr,
    func_single_corr2, compute_eff_mass)

def print_label(keys, vals, xpos=0.7, ypos=0.8):
    """Print a label in the plot.

    Parameters
    ----------
    keys : list of str
        The label of the data to print.
    vals : list of floats
        The data to print.
    xpos, ypos : float, optional
        The position in relativ to maximum of x and y axis,
        respectively. Should be between 0 and 1.
    """
    datalabel = "%s = %.4f" %(keys[0], vals[0])
    for k, v in zip(keys[1:], vals[1:]):
        datalabel = "\n".join((datalabel, "%s = %.4f" %(k, v)))
    x = xlim()[1] * xpos
    y = ylim()[1] * ypos

def plot_function(func, X, args, label, add=None, plotrange=None, ploterror=False,
        fmt="k", col="black"):
    """A function that plots a function.

    Parameters
    ----------
    func : callable
        The function to plot.
    Y : ndarray
        The data for the y axis.
    args : ndarray
        The arguments to the function.
    label : list of str
        A list with labels for data and fit.
    add : ndarray, optional
        Additional arguments to the fit function.
    plotrange : list of ints, optional
        The lower and upper range of the plot.
    ploterror : bool, optional
        Plot the error of the fit function.
    """
    # check for plotting range
    if isinstance(plotrange, (np.ndarray, list, tuple)):
        _plotrange = np.asarray(plotrange).flatten()
        if _plotrange.size < 2:
            raise IndexError("fitrange has not enough indices")
        else:
            lfunc = _plotrange[0]
            ufunc = _plotrange[1]
            x1 = np.linspace(X[lfunc], X[ufunc], 1000)
    else:
        x1 = np.linspace(X[0], X[-1], 1000)
    #print("option summary:")
    #print("function name is %s" % func)
    #print("shape of arguments (nb_samples, nb_parameters):")
    #print(args.shape)
    #print("Plot an errorband: %s" % ploterror)
    # check dimensions of args, if more than one,
    # iterate over first dimension
    _args = np.asarray(args)
    if add is not None:
        _add = np.asarray(add)
    if _args.ndim > 1:
        # the first sample contains original data,
        # also save min and max at each x
        y1, ymin, ymax = [], [], []
        # check for dimensions of add
        if add is not None:
            # need to check size of first axis
            args0 = _args.shape[0]
            add0 = _add.shape[0]
            if args0 == add0:
                # first axis has same size for both
                # iterate over the x range
                for i, x in enumerate(x1):
                    # the actual value is given by the first sample
                    y1.append(func(_args[0], x, _add[0]))
                    #if i % 100 == 0:
                    #    print(x)
                    #    print(_args[0])
                    #    print(_add[0])
                    #    print(y1[-1])
                    if ploterror:
                        tmp = [y1[-1]]
                        # iterate over the rest of the arguments
                        for i in range(1,args0):
                            tmp.append(func(_args[i], x, _add[i]))
                        mean, std = compute_error(np.asarray(tmp))
                        ymin.append(float(mean-std))
                        ymax.append(float(mean+std))
            elif (args0 % add0) == 0:
                # size of add is a divisor of size of args
                # iterate over x
                for x in x1:
                    # the actual value is given by the first sample
                    y1.append(func(_args[0], x, _add[0]))
                    if ploterror:
                        tmp = [y1[-1]]
                        # iterate over the rest of the arguments
                        for i in range(1,args0):
                            tmp.append(func(_args[i], x, _add[i%add0]))
                        mean, std = compute_error(np.asarray(tmp))
                        ymin.append(float(mean-std))
                        ymax.append(float(mean+std))
            elif (add0 % args0) == 0:
                # size of args is a divisor of size of add
                # iterate over x
                for x in x1:
                    # the actual value is given by the first sample
                    y1.append(func(_args[0], x, _add[0]))
                    if ploterror:
                        tmp = [y1[-1]]
                        # iterate over the rest of the arguments
                        for i in range(1,add0):
                            tmp.append(func(_args[i%args0], x, _add[i]))
                        mean, std = compute_error(np.asarray(tmp))
                        ymin.append(float(mean-std))
                        ymax.append(float(mean+std))
        else:
            # no additional arguments, iterate over args
            #iterate over x
            #print("using function")
            for j,x in enumerate(x1):
                y1.append(func(_args[0], x))
                if ploterror:
                    tmp = [y1[-1]]
                    for i in range(1, _args.shape[0]):
                        tmp.append(func(_args[i], x))
                    mean, std = compute_error(np.asarray(tmp))
                    ymin.append(float(mean-std))
                    ymax.append(float(mean+std))
    # only one args
    else:
        # the first sample contains original data
        y1 = []
        # calculate minimal and maximal y1 for
        # error checking
        ymax = []
        ymin = []
        # iterate over x values
        for x in x1:
            # check for additional arguments
            if add is not None:
                tmp = func(_args, x, _add)
                if np.asarray(tmp).size > 1:
                    y1.append(tmp[0])
                    if ploterror:
                        mean, std = compute_error(np.asarray(tmp))
                        ymin.append(float(mean-std))
                        ymax.append(float(mean+std))
                else:
                    y1.append(tmp)
            else:
                # calculate on original data
                y1.append(func(_args, x))
    #print(len(x1),len(y1))
    plt.plot(x1, y1, fmt, label=label)
    if ymax and ymin:
        #print(ymax[0])
        #print(x1[0])
        plt.fill_between(x1, ymin, ymax, facecolor=col,
            edgecolor=col, alpha=0.2)
    plt.legend()

def plot_data(X, Y, dY, label, plotrange=None, fmt="x",col='b'):
    """A function that plots data.

    Parameters
    ----------
    X : ndarray
        The data for the x axis.
    Y : ndarray
        The data for the y axis.
    dY : ndarray
        The error on the y axis data.
    label : list of str
        A list with labels for data and fit.
    plotrange : list of ints, optional
        The lower and upper range of the plot.
    """
    # check boundaries for the plot
    if isinstance(plotrange, (np.ndarray, list, tuple)):
        plotrange = np.asarray(plotrange).flatten()
        if plotrange.size < 2:
            raise IndexError("plotrange is too small")
        else:
            l = int(plotrange[0])
            u = int(plotrange[1])
        # plot the data
        plt.errorbar(X[l:u], Y[l:u], dY[l:u], fmt=fmt, label=label, c=col)
    else:
        # plot the data
        plt.errorbar(X, Y, dY, fmt=fmt, label=label,c=col)
    plt.legend()

def plot_data_with_fit(X, Y, dY, fitfunc, args, label, plotrange=None,
                   fitrange=None, addpars=None, pval=None,col='b'):
    """A function that plots data and the fit to the data.

    Parameters
    ----------
    X : ndarray
        The data for the x axis.
    Y : ndarray
        The data for the y axis.
    dY : ndarray
        The error on the y axis data.
    fitfunc : callable
        The function to fit to the data.
    args : ndarray
        The parameters of the fit function from the fit.
    label : list of str
        A list with labels for data and fit.
    plotrange : list of ints, optional
        The lower and upper range of the plot.
    fitrange : list of ints, optional
        A list with two entries, bounds of the fitted function.
    addpars : bool, optional
        if there are additional parameters for the fitfunction 
             contained in args, set to true
    pval : float, optional
        write the p-value in the plot if given
    """
    # plot the data
    plot_data(X, Y, dY, label[0], plotrange=plotrange,col=col)

    # plot the function
    plot_function(fitfunc, X, args, label[1], addpars, fitrange,col=col)

    # adjusting the plot style
    plt.legend()

    # print label if available
    if pval is not None:
        # x and y position of the label
        x = np.max(X) * 0.7
        y = np.max(Y) * 0.8
        datalabel = "p-val = %.5f" % pval
        try:
            for k, d in enumerate(args[0]):
                datalabel = "".join((datalabel, "\npar %d = %.4e" % (k, d)))
        except TypeError:
            datalabel = "".join((datalabel, "\npar = %.4e" % (args[0])))
        plt.text(x, y, datalabel)

def plot_histogram(data, data_weight, label, nb_bins=20, debug=0):
    """Plots histograms for the given data set.

    The function plots the weighted distribution of the data, the unweighted
    distribution and a plot containing both the weighted and the unweighted
    distribution.

    Parameters
    ----------
    nb_bins : int
        The number of equally distanced bins in the histogram
    data : ndarray
        Data set for the histogram.
    data_weight : ndarray
        The weights for the data, must have same shape as data.
    label : list of strs
        The title of the plots, the x-axis label and the label of the data.
    debug : int
        The amount of info printed.
    """
    # The histogram
    # generate weighted histogram
    hist, bins = np.histogram(data, nb_bins, weights=data_weight, density=True)
    # generate the unweighted histogram
    uhist, ubins = np.histogram(data, nb_bins, weights=np.ones_like(data_weight),
                                density=True)

    # prepare the plot
    width = 0.7 * (bins[1] - bins[0])
    uwidth = 0.7 * (ubins[1] - ubins[0])
    center = (bins[:-1] + bins[1:]) / 2
    ucenter = (ubins[:-1] + ubins[1:]) / 2

    #print(bins)
    #print(width)
    #print(center)

    # plot both histograms in same plot
    plt.title(label[0])
    plt.xlabel(label[1])
    plt.ylabel("".join(("distribution of ", label[2])))
    plt.grid(True)
    # plot
    plt.bar(center, hist, align='center', width=width, color='r', alpha=0.5,
            label='weighted data')
    plt.bar(center, uhist, align='center', width=width, color='b', alpha=0.5,
            label='unweighted data')

def plot_eff_mass(X, corr, dcorr, mass, dmass, fit, label, mass_shift=1, masspar=1, fmt1='xb', fmt2='xr'):
    # create a subplot
    f, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.plot_errorbar(X[mass_shift:], mass, dmass, fmt=fmt1, label="")
    ax2.plot_errorbar(X, corr, dcorr, fmt=fmt1, label="")

if __name__ == "__main__":
    pass
