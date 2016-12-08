"""
Functions for plotting.
"""

import numpy as np
import matplotlib.pyplot as plt
import itertools

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
        fmt="k", col="black", samples=300, debug=0):
    """A function that plots a function.

    Parameters
    ----------
    func : callable
        The function to plot.
    X : tuple or list or ndarray
        The start and end x range of function plot.
    args : ndarray
        The arguments to the function.
    label : list of str
        A label for the function.
    add : ndarray, optional
        Additional arguments to the function.
    plotrange : list of ints, optional -- deprecated
        The lower and upper range of the plot.
    ploterror : bool, optional
        Plot the error of the fit function.
    fmt : string, optional
        The format of the line.
    col : string, optional
        The color of the line and errorband.
    samples : int
        The number of points between the min and max x values.
    debug : int
        The level of verboseness.
    """
    _X = np.asarray(X).flatten()
    if _X.size > 2:
        _X  = _X[:2]
        if debug > 1:
            print("more than 2 _X values, truncating...")
    elif _X.size < 2:
        raise RuntimeError("need min and max x values for plot, only one given.")
    x1 = np.linspace(_X[0], _X[1], samples)
    if debug > 2:
        print("option summary:")
        print("function name is %s" % func)
        print("shape of arguments (nb_samples, nb_parameters):")
        print(args.shape)
        print("Plot an errorband: %s" % ploterror)
        print("_X values: %r" % _X)
    # check dimensions of args, if more than one,
    # iterate over first dimension
    _args = np.asarray(args)
    print(_args.shape)
    if add is not None:
        _add = np.asarray(add)
    if _args.ndim > 1:
        # the first sample contains original data,
        # also save min and max at each x
        y1, ymin, ymax = [], [], []
        # check for dimensions of add
        if add is not None:
            ## need to check size of first axis
            #args0 = _args.shape[0]
            #add0 = _add.shape[0]
            #if args0 % add0 == 0:
            #    # size of add is a divisor of size of args
            #    _add = itertools.repeat(_add, args0/add0)
            #elif add0 %  args0 == 0:
            #    # size of args is a divisor of size of add
            #    _args = itertools.repeat(_args, add0/args0)
            #else:
            #    raise RuntimeError("args and add are both given, but size does not match.")
            # iterate over the x range
            #for x in x1x_plot:
            for x in x1:
                # the actual value is given by the first sample
                y1.append(func(np.atleast_2d(_args[0]), x, _add[0]))
                # if plotting the error bar, iterate over arguments
                if ploterror:
                    tmp = func(_args, x, _add)
                    mean, std = compute_error(np.asarray(tmp))
                    ymin.append(float(mean-std))
                    ymax.append(float(mean+std))
        else:
            # no additional arguments, iterate over args
            #iterate over x
            #print("using function")
            print("Shape for parameters:")
            print(_args.shape)
            print("Shape for x1:")
            print(x1.shape)
            for j,x in enumerate(x1):
                # ATM _args[0] has to be dimension of fit parameters
                y1.append(func(np.atleast_2d(_args[0]), x))
                if ploterror:
                    tmp = func(_args, x)
                    mean, std = compute_error(np.asarray(tmp))
                    ymin.append(float(mean-std))
                    ymax.append(float(mean+std))
    # only one args
    else:
        # the first sample contains original data,
        # also save min and max at each x
        y1, ymin, ymax = [], [], []
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
    plt.plot(x1, y1, fmt, label=label)
    if ymax and ymin:
        plt.fill_between(x1, ymin, ymax, facecolor=col, edgecolor=col,
            alpha=0.2)
#def plot_function(func, X, args, label, add=None, plotrange=None, ploterror=False,
#        fmt="k", col="black",debug=0):
#    """A function that plots a function.
#
#    Parameters
#    ----------
#    func : callable
#        The function to plot.
#    Y : ndarray
#        The data for the y axis.
#    args : ndarray
#        The arguments to the function.
#    label : list of str
#        A list with labels for data and fit.
#    add : ndarray, optional
#        Additional arguments to the fit function.
#    plotrange : list of ints, optional
#       The lower and upper range of the plot.
#
#    ploterror : bool, optional
#        Plot the error of the fit function.
#    """
#    # check for plotting range
#    if isinstance(plotrange, (np.ndarray, list, tuple)):
#        _plotrange = np.asarray(plotrange).flatten()
#        if _plotrange.size < 2:
#            raise IndexError("fitrange has not enough indices")
#        else:
#            lfunc = _plotrange[0]
#            ufunc = _plotrange[1]
#            # handle data with more than two dimensions
#            if len(X.shape) > 1:
#                x1 = np.zeros((1000,X.shape[1]))
#                for i in range(x1.shape[1]):
#                    print("bounds at index %d are:" %i)
#                    print(X[lfunc,i],X[ufunc,i])
#                    x1[:,i] = np.linspace(X[lfunc,i],X[ufunc,i], 1000)
#            else:
#                x1 = np.linspace(X[lfunc], X[ufunc], 1000)
#    else:
#        # handle data with more than two dimensions
#        if len(X.shape) > 1:
#            print(X.shape)
#            x1 = np.zeros((1000,X.shape[1]))
#            #x1 = np.linspace(X[0,0],X[-1,0], 1000)
#            #try:
#            #x1[:,0] = np.linspace(X[0,0],X[-1,0], 1000) 
#            for i in range(x1.shape[1]):
#                print("bounds at index %d are:" %i)
#                print(X[0,i],X[-1,i])
#                x1[:,i] = np.linspace(X[0,i],X[-1,i], 1000)
#                #if X.shape[0] == 1:
#                x1[:,0] = np.linspace(0,50,1000)
#
#            #except:
#            #  print RuntimeWarning("Data not truly two dimensional")
#            #  x1 = np.asarray((_x1,))
#            #  pass
#        else:
#            x1 = np.linspace(X[0], X[-1], 1000)
#    if debug > 2:
#      print("option summary:")
#      print("function name is %s" % func)
#      print("shape of arguments (nb_samples, nb_parameters):")
#      print(args.shape)
#      print("Plot an errorband: %s" % ploterror)
#    # check dimensions of args, if more than one,
#    # iterate over first dimension
#    print(args)
#    _args = np.asarray(args)
#    if add is not None:
#        _add = np.asarray(add)
#    if _args.ndim > 1:
#        # the first sample contains original data,
#        # also save min and max at each x
#        y1, ymin, ymax = [], [], []
#        # check for dimensions of add
#        if add is not None:
#            # need to check size of first axis
#            args0 = _args.shape[0]
#            add0 = _add.shape[0]
#            if args0 == add0:
#                # first axis has same size for both
#                # iterate over the x range
#                for i, x in enumerate(x1):
#                    # the actual value is given by the first sample
#                    y1.append(func(_args[0], x, _add[0]))
#                    #if i % 100 == 0:
#                    #    print(x)
#                    #    print(_args[0])
#                    #    print(_add[0])
#                    #    print(y1[-1])
#                    if ploterror:
#                        tmp = [y1[-1]]
#                        # iterate over the rest of the arguments
#                        for i in range(1,args0):
#                            tmp.append(func(_args[i], x, _add[i]))
#                        mean, stdev = compute_error(np.asarray(tmp))
#                        ymin.append(float(mean-stdev))
#                        ymax.append(float(mean+stdev))
#                        #ymin.append(np.min(tmp))
#                        #ymax.append(np.max(tmp))
#            elif (args0 % add0) == 0:
#                # size of add is a divisor of size of args
#                # iterate over x
#                for x in x1:
#                    # the actual value is given by the first sample
#                    y1.append(func(_args[0], x, _add[0]))
#                    if ploterror:
#                        tmp = [y1[-1]]
#                        # iterate over the rest of the arguments
#                        for i in range(1,args0):
#                            tmp.append(func(_args[i], x, _add[i%add0]))
#                        mean, stdev = compute_error(np.asarray(tmp))
#                        ymin.append(float(mean-stdev))
#                        ymax.append(float(mean+stdev))
#                        #ymin.append(np.min(tmp))
#                        #ymax.append(np.max(tmp))
#            elif (add0 % args0) == 0:
#                # size of args is a divisor of size of add
#                # iterate over x
#                for x in x1:
#                    # the actual value is given by the first sample
#                    y1.append(func(_args[0], x, _add[0]))
#                    if ploterror:
#                        tmp = [y1[-1]]
#                        # iterate over the rest of the arguments
#                        for i in range(1,add0):
#                            tmp.append(func(_args[i%args0], x, _add[i]))
#                        mean, stdev = compute_error(np.asarray(tmp))
#                        ymin.append(float(mean-stdev))
#                        ymax.append(float(mean+stdev))
#                        #ymin.append(np.min(tmp))
#                        #ymax.append(np.max(tmp))
#        else:
#            # no additional arguments, iterate over args
#            #iterate over x
#            print("using function")
#            for j,x in enumerate(x1):
#                y1.append(func(_args[0], x))
#                if ploterror:
#                    tmp = [y1[-1]]
#                    for i in range(1, _args.shape[0]):
#                        tmp.append(func(_args[i], x))
#                    mean, stdev = compute_error(np.asarray(tmp))
#                    ymin.append(float(mean-stdev))
#                    ymax.append(float(mean+stdev))
#                    #ymin.append(np.min(tmp))
#                    #ymax.append(np.max(tmp))
#    else:
#        # the first sample contains original data
#        y1 = []
#        # calculate minimal and maximal y1 for
#        # error checking
#        ymax = []
#        ymin = []
#        # iterate over x values
#        for x in x1:
#            # check for additional arguments
#            if add is not None:
#                tmp = func(_args, x, _add)
#                if np.asarray(tmp).size > 1:
#                    y1.append(tmp[0])
#                    if ploterror:
#                        mean, stdev = compute_error(np.asarray(tmp))
#                        ymin.append(float(mean-stdev))
#                        ymax.append(float(mean+stdev))
#                        #ymax.append(np.max(tmp))
#                        #ymin.append(np.min(tmp))
#                else:
#                    y1.append(tmp)
#            else:
#                # calculate on original data
#                y1.append(func(_args, x))
#    #print(len(x1),len(y1))
#    if len(X.shape) > 1:
#      plt.plot(x1[:,0], y1, fmt, label=label)
#    else:
#      plt.plot(x1, y1, fmt, label=label)
#    if ymax and ymin:
#        #print(ymax[0])
#        #print(x1[0])
#        if len(X.shape) > 1:
#          #print("fill shape:")
#          #print(x1.shape)
#          plt.fill_between(x1[:,0], ymin, ymax, facecolor=col,
#              edgecolor=col, alpha=0.2)
#        else:
#          plt.fill_between(x1, ymin, ymax, facecolor=col,
#              edgecolor=col, alpha=0.2)
#    plt.legend()

def plot_data(X, Y, dY, label, dX=None,plotrange=None,
    fmt="x",col='b',markerfill=None,alpha=1.):
    """A function that plots data.

    Parameters
    ----------
    X : ndarray
        The data for the x axis.
    Y : ndarray
        The data for the y axis.
    dX : ndarray (optional)
        The error on the x axis data 
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
        if dX is not None:
          plt.errorbar(X[l:u], Y[l:u], dY[l:u], dX[l:u], fmt=fmt, label=label,
                c=col,markerfacecolor=markerfill,alpha=alpha)
        else:
            plt.errorbar(X[l:u], Y[l:u], dY[l:u], fmt=fmt, label=label,
                c=col,markerfacecolor=markerfill,alpha=alpha)
    else:
        # plot the data
        if dX is not None:
            plt.errorbar(X, Y, dY, dX,fmt=fmt, label=label,
                c=col, markerfacecolor=markerfill,alpha=alpha)
        else:
            plt.errorbar(X, Y, dY, fmt=fmt, label=label,
                c=col, markerfacecolor=markerfill,alpha=alpha)
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
