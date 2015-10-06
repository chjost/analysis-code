"""
The class for fitting.
"""

import numpy as np
import itertools
import matplotlib
matplotlib.use('Agg')
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
        # plot environment variables
        self.xlog=False
        self.ylog=False
        self.xlim=None
        self.ylim=None
        self.grid=True
        self.legend=True

    def __del__(self):
        self.plotfile.close()

    def new_file(self, filename):
      """Open a new plot file.
      """
      self.plotfile.close()
      plt.clf()
      self.plotfile = PdfPages(filename)

    def save(self):
        if plt.get_fignums():
            for i in plt.get_fignums():
                self.plotfile.savefig(plt.figure(i))
        plt.clf()

    def _set_env_normal(self):
        plt.grid(self.grid)
        # set the axis to log scale
        if self.xlog:
            plt.xscale("log")
        if self.ylog:
            plt.yscale("log")
        # set the axis ranges
        if self.xlim:
            plt.xlim(self.xlim)
        if self.ylim:
            plt.ylim(self.ylim)
        if self.legend:
            plt.legend()

    def _est_env_hist(self):
        plt.grid(self.grid)
        if self.legend:
            plt.legend()

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
                       fitrange=None, addpars=None, pval=None):
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
        self.plot_data(X, Y, dY, label[0], plotrange=plotrange)

        # plot the function
        self.plot_function(fitfunc, X, args, label[1], addpars, fitrange)

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
    def _print_label(self, keys, vals, xpos=0.7, ypos=0.8):
        """Print a label in the plot.

        """

    def plot_function(self, func, X, args, label, add=None, plotrange=None):
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
        """
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
        if add is not None:
            y1 = []
            for j, x in enumerate(x1):
                y1.append(func(args, x, add[j]))
            y1 = np.asarray(y1)
        else:
            y1 = []
            for x in x1:
                y1.append(func(args, x))
            y1 = np.asarray(y1)
        plt.plot(x1, y1, "r", label=label)

    def plot_data(self, X, Y, dY, label, plotrange=None):
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
            plt.errorbar(X[l:u], Y[l:u], dY[l:u], fmt='x' + 'b', label = label)
        else:
            # plot the data
            plt.errorbar(X, Y, dY, fmt='x' + 'b', label=label)
        plt.legend()

    def plot_histogram(self, data, data_weight, label, debug=0):
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
    
        # prepare the plot
        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
    
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
        plt.legend()
        # save and clear
        self.save()

        # plot the weighted histogram
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
    
        # plot the unweighted histogram
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

    def _genplot_single(self, corr, fitresult, fitfunc, label, add=None,
            debug=0):
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
        add : ndarray, optional
            Additional arguments to the fit function. This is stacked along
            the third dimenstion to the oldfit data.
        debug : int, optional
            The amount of info printed.
        """
        if len(label) < 4:
            raise RuntimeError("not enough labels")
        if len(label) < 5:
            label.append("")
        if corr.matrix:
            raise RuntimeError("Cannot plot correlation function matrix")
        # get needed data
        ncorr = corr.shape[-1]
        T = corr.shape[1]
        ranges = fitresult.fit_ranges
        shape = fitresult.fit_ranges_shape
        X = np.linspace(0., float(corr.shape[1]), corr.shape[1], endpoint=False)
        label_save = label[0]

        # iterate over correlation functions
        for n in range(ncorr):
            if debug > 1:
                print("plotting correlators %d" % (n))
            mdata, ddata = compute_error(corr.data[:,:,n])
            
            # iterate over fit intervals
            for r in range(shape[0][n]):
                if debug > 1:
                    print("plotting fit ranges %s" % str(r))
                fi = ranges[n][r]
                mpar, dpar = compute_error(fitresult.data[n][:,:,r])

                # set up labels
                label[0] = "%s, pc %d" % (label_save, n)
                self.set_title(label[0], label[1:3])
                label[4] = "fit [%d, %d]" % (fi[0], fi[1])

                # plot
                self._set_env_normal()
                self.plot_data_with_fit(X, corr.data[0,:,n], ddata,
                        fitfunc.fitfunc, mpar, label[3:], plotrange=[1,T],
                        addpars=add, fitrange=fi)
                self.save()

        label[0] = label_save

    def _genplot_comb(self, corr, fitresult, fitfunc, label, oldfit, add=None,
            oldfitpar=None, debug=0):
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
        oldfit : None or FitResult, optional
            Reuse the fit results of an old fit for the new fit.
        add : ndarray, optional
            Additional arguments to the fit function. This is stacked along
        oldfitpar : None, int or sequence of int, optional
            Which parameter of the old fit to use, if there is more than one.
                the third dimenstion to the oldfit data.
        debug : int, optional
            The amount of info printed.
        """
        if len(label) < 4:
            raise RuntimeError("not enough labels")
        if len(label) < 5:
            label.append("")
        if corr.matrix:
            raise RuntimeError("Cannot plot correlation function matrix")
        # get needed data
        ncorrs = fitresult.corr_num
        X = np.linspace(0., float(corr.shape[1]), corr.shape[1], endpoint=False)
        label_save = label[0]
        T = corr.shape[1]
        franges = fitresult.fit_ranges
        fshape = fitresult.fit_ranges_shape

        # iterate over correlation functions
        ncorriter = [[x for x in range(n)] for n in ncorrs]
        for item in itertools.product(*ncorriter):
            if debug > 1:
                print("plotting correlators %s" % str(item))
            n = item[-1]
            mdata, ddata = compute_error(corr.data[:,:,n])
            # create the iterator over the fit ranges
            tmp = [fshape[i][x] for i,x in enumerate(item)]
            rangesiter = [[x for x in range(m)] for m in tmp]
            # iterate over the fit ranges
            for ritem in itertools.product(*rangesiter):
                if debug > 1:
                    print("plotting fit ranges %s" % str(ritem))
                r = ritem[-1]
                # get fit interval
                fi = franges[n][r]
                _par = fitresult.get_data(item + ritem)
                mpar, dpar = compute_error(_par)

                # set up labels
                label[0] = "%s, pc %d (%s)" % (label_save, n, str(item[:-1]))
                self.set_title(label[0], label[1:3])
                label[4] = "fit [%d, %d]\nold %s" % (fi[0], fi[1], str(ritem[:-1]))

                # get old data
                add_data = oldfit.get_data(item[:-1] + ritem[:-1]) 
                # get only the wanted parameter if oldfitpar is given
                if oldfitpar is not None:
                    add_data = add_data[:,oldfitpar]
                # if there is additional stuff needed for the fit
                # function add it to the old data
                if add is not None:
                    # get the shape right, atleast_2d adds the dimension
                    # in front, we need it in the end
                    if add.ndim == 1:
                        add.shape = (-1, 1)
                    if add_data.ndim == 1:
                        add_data.shape = (-1, 1)
                    add_data = np.hstack((add_data, add))

                # plot
                self._set_end_normal()
                self.plot_data_with_fit(X, corr.data[0,:,n], ddata,
                        fitfunc.fitfunc, mpar, label[3:], plotrange=[1,T],
                        addpars=add_data, fitrange=fi)
                self.save()

    def plot(self, corr, fitresult, fitfunc, label, oldfit=None, add=None,
            oldfitpar=None, debug=0):
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
        oldfit : None or FitResult, optional
            Reuse the fit results of an old fit for the new fit.
        add : ndarray, optional
            Additional arguments to the fit function. This is stacked along
        oldfitpar : None, int or sequence of int, optional
            Which parameter of the old fit to use, if there is more than one.
                the third dimenstion to the oldfit data.
        debug : int, optional
            The amount of info printed.
        """
        if oldfit is None:
            self._genplot_single(corr, fitresult, fitfunc, label, add=add,
                    debug=debug)
        else:
            self._genplot_comb(corr, fitresult, fitfunc, label, oldfit, add,
                    oldfitpar, debug)

    def histogram(self, fitresult, label, par=None):
        """Plot the histograms.

        Parameters
        ----------
        fitresult : FitResult
            The fit data.
        label : list of strs
            The title of the plots, the x-axis label and the label of the data.
        par : int, optional
            For which parameter the histogram is plotted.
        """
        fitresult.calc_error()
        label_save = label[0]
        if par is None:
            for p, w in enumerate(fitresult.weight):
                for i, d in enumerate(fitresult.data):
                    label[0] = " ".join((label_save, str(fitresult.label[i])))
                    self.plot_histogram(d[0,p], w[i], label)
        else:
            w = fitresult.weight[par]
            for i, d in enumerate(fitresult.data):
                label[0] = " ".join((label_save, str(fitresult.label[i])))
                self.plot_histogram(d[0,par], w[i], label)
        label[0] = label_save

    def set_env(self, xlog=False, ylog=False, xlim=None, ylim=None, grid=True):
        """Set different environment variables for the plot.
        
        Parameters
        ----------
        xlog, ylog : bool, optional
            Make the respective axis log scale.
        xlim, ylim : list of ints, optional
            Limits for the x and y axis, respectively
        grid : bool, optional
            Plot the grid.
        """
        self.xlog=xlog
        self.ylog=ylog
        self.xlim=xlim
        self.ylim=ylim
        self.grid=grid

if __name__ == "__main__":
    pass
