"""
The class for fitting.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from fit import LatticeFit, FitResult
from correlator import Correlators
from statistics import compute_error

class LatticePlot(object):
    def __init__(self, filename):
        """Initialize a plot.

        Parameters
        ----------
        filename : str
            The filename of the plot.
        """
        self.plotfile = PdfPages(filename)
        self.p = []

    def __del__(self):
        if self.p:
            self.save()
        self.plotfile.close()

    def set_title(self, title, axis):
        """Set the title and axis labels of the plot.

        Parameters
        ----------
        title : str
            The title of the plot.
        axis : list of strs
            The labels of the axis.
        """
        plt.title(title)
        plt.xlabel(axis[0])
        plt.ylabel(axis[1])

    def plot_data_with_fit(self, X, Y, dY, fitfunc, args, label, plotrange=None,
                       logscale=False, xlim=None, ylim=None, fitrange=None,
                       addpars=None, pval=None):
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
        logscale : bool, optional
            Make the y-scale a logscale.
        xlim, ylim : list of ints, optional
            Limits for the x and y axis, respectively
        fitrange : list of ints, optional
            A list with two entries, bounds of the fitted function.
        addpars : bool, optional
            if there are additional parameters for the fitfunction 
                 contained in args, set to true
        pval : float, optional
            write the p-value in the plot if given
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
            self.p.append(plt.errorbar(X[l:u], Y[l:u], dY[l:u], fmt='x' + 'b',
                label = label[0]))
        else:
            # plot the data
            self.p.append(plt.errorbar(X, Y, dY, fmt='x' + 'b', label=label[0]))

        # plot the function
        self.plot_function(fitfunc, X, args, addpars, fitrange, label[1])

        # adjusting the plot style
        plt.grid(True)
        plt.legend()
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
        if logscale:
            plt.yscale("log")
        # set the axis ranges
        if xlim:
            plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)

    def plot_function(self, func, X, args, add=None, plotrange=None, label=""):
        # plotting the fit function, check for seperate range
        if isinstance(plotrange, (np.ndarray, list, tuple)):
            plotrange = np.asarray(plotrange).flatten()
            if plotrange.size < 2:
                raise IndexError("fitrange has not enough indices")
            else:
                lfunc = int(plotrange[0])
                ufunc = int(plotrange[1])
        else:
            lfunc = X[0]
            ufunc = X[-1]
        x1 = np.linspace(lfunc, ufunc, 1000)
        if add:
            y1 = func(args, x1, add)
        else:
            y1 = []
            for x in x1:
                y1.append(func(args, x))
            y1 = np.asarray(y1)
            #y1 = func(args, x1)
        self.p.append(plt.plot(x1, y1, "r", label=label))
        plt.grid(True)

    def plot_data(self, X, Y, dY, label, plotrange=None, logscale=False,
            xlim=None, ylim=None):
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
        logscale : bool, optional
            Make the y-scale a logscale.
        xlim, ylim : list of ints, optional
            Limits for the x and y axis, respectively
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
            self.p.append(plt.errorbar(X[l:u], Y[l:u], dY[l:u], fmt='x' + 'b',
                label = label[0]))
        else:
            # plot the data
            self.p.append(plt.errorbar(X, Y, dY, fmt='x' + 'b', label=label[0]))

        # adjusting the plot style
        plt.grid(True)
        plt.legend()
        if logscale:
            plt.yscale("log")
        # set the axis ranges
        if xlim:
            plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)

    def plot_histogram(data, data_weight, label, bug=0):
        """Plots histograms for the given data set.
    
        The function plots the weighted distribution of the data, the unweighted
        distribution and a plot containing both the weighted and the unweighted
        distribution.
    
        Parameters
        ----------
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
        hist, bins = np.histogram(data, 20, weights=data_weight, density=True)
        # generate the unweighted histogram
        uhist, ubins = np.histogram(data, 20, weights=np.ones_like(data_weight),
                                    density=True)
    
        # prepare the plot for the weighted histogram
        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
    
        # set labels for axis
        plt.title(label[0])
        plt.xlabel(label[1])
        plt.ylabel('weighted distribution')
        plt.grid(True)
        # plot
        plt.bar(center, hist, align='center', width=width, color='r', alpha=0.5,
                label=label[2])
        plt.legend()
        # save and clear
        self.save()
    
        # prepare plot for unweighted histogram
        # the center and width stays the same for comparison
        plt.title(label[0])
        plt.xlabel(label[1])
        plt.ylabel('unweighted distribution')
        plt.grid(True)
        # plot
        plt.bar(center, uhist, align='center', width=width, color='b', alpha=0.5,
                label=label[2])
        plt.legend()
        # save and clear
        self.save()
    
        # plot both histograms in same plot
        plt.title(label[0])
        plt.xlabel(label[1])
        plt.ylabel(label[2])
        plt.grid(True)
        # plot
        plt.bar(center, hist, align='center', width=width, color='r', alpha=0.5,
                label='weighted data')
        plt.bar(center, uhist, align='center', width=width, color='b', alpha=0.5,
                label='unweighted data')
        plt.legend()
        # save and clear
        self.save()

    def save(self):
        #print(plt.get_fignums())
        for i in plt.get_fignums():
            self.plotfile.savefig(plt.figure(i))
        #if self.p:
        #    for x in self.p:
        #        self.plotfile.savefig(x)
        plt.clf()
        self.p = []

    def genplot(self, corr, fitresult, fitfunc, label, add=None):
        """Plot the data of a Correlators object and a FitResult object
        together.

        Parameters
        ----------
        corr : Correlators
            The correlation function data.
        fitresult : FitResult
            The fit data.
        fitfunc : LatticeFit
            The fit function.
        label : list of strs
            The title of the plot, the x- and y-axis titles, and the data label.
        """
        if len(label) < 4:
            raise RuntimeError("not enough labels")
        if len(label) < 5:
            label.append("")
        if corr.matrix:
            raise RuntimeError("Cannot plot correlation function matrix")
        # get needed data
        ncorr = corr.shape[-1]
        ranges = fitresult.fit_ranges
        shape = fitresult.fit_ranges_shape
        X = np.linspace(0., float(corr.shape[1]), corr.shape[1], endpoint=False)
        label_save = label[0]

        # iterate over correlation functions
        for n in range(ncorr):
            mdata, ddata = compute_error(corr.data[:,:,n])
            
            # iterate over fit intervals
            for r in range(shape[0][n]):
                fi = ranges[n][r]
                mpar, dpar = compute_error(fitresult.data[n][:,:,r])

                # set up labels
                label[0] = "%s, pc %d" % (label_save, n)
                self.set_title(label[0], label[1:3])
                label[4] = "fit [%d, %d]" % (fi[0], fi[1])

                # plot
                self.plot_data_with_fit(X, corr.data[0,:,n], ddata,
                        fitfunc.fitfunc, mpar, label[3:], logscale=False,
                        plotrange=[1,23], addpars=add)
                self.save()

        label[0] = label_save

if __name__ == "__main__":
    pass
