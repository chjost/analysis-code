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
        #if self.legend:
        #    plt.legend()

    def _est_env_hist(self):
        plt.grid(self.grid)
        #if self.legend:
        #    plt.legend()

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
        # check for plotting range
        if isinstance(plotrange, (np.ndarray, list, tuple)):
            _plotrange = np.asarray(plotrange).flatten()
            if _plotrange.size < 2:
                raise IndexError("fitrange has not enough indices")
            else:
                lfunc = X[_plotrange[0]]
                ufunc = X[_plotrange[1]]
        else:
            lfunc = X[0]
            ufunc = X[-1]
        x1 = np.linspace(lfunc, ufunc, 1000)

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
                    for x in x1:
                        # the actual value is given by the first sample
                        y1.append(func(_args[0], x, _add[0]))
                        tmp = [y1[-1]]
                        # iterate over the rest of the arguments
                        for i in range(1,args0):
                            tmp.append(func(_args[i], x, _add[i]))
                        ymin.append(np.min(tmp))
                        ymax.append(np.max(tmp))
                elif (args0 % add0) == 0:
                    # size of add is a divisor of size of args
                    # iterate over x
                    for x in x1:
                        # the actual value is given by the first sample
                        y1.append(func(_args[0], x, _add[0]))
                        tmp = [y1[-1]]
                        # iterate over the rest of the arguments
                        for i in range(1,args0):
                            tmp.append(func(_args[i], x, _add[i%add0]))
                        ymin.append(np.min(tmp))
                        ymax.append(np.max(tmp))
                elif (add0 % args0) == 0:
                    # size of args is a divisor of size of add
                    # iterate over x
                    for x in x1:
                        # the actual value is given by the first sample
                        y1.append(func(_args[0], x, _add[0]))
                        tmp = [y1[-1]]
                        # iterate over the rest of the arguments
                        for i in range(1,add0):
                            tmp.append(func(_args[i%args0], x, _add[i]))
                        ymin.append(np.min(tmp))
                        ymax.append(np.max(tmp))
            else:
                # no additional arguments, iterate over args
                #iterate over x
                for x in x1:
                    y1.append(func(_args[0], x))
                    tmp = [y[-1]]
                    for i in range(1, _args.shape[0]):
                        tmp.append(func(_args[i], x))
                    ymin.append(np.min(tmp))
                    ymax.append(np.max(tmp))
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
                        ymax.append(np.max(tmp))
                        ymin.append(np.min(tmp))
                    else:
                        y1.append(tmp)
                else:
                    # calculate on original data
                    y1.append(func(_args, x))
        plt.plot(x1, y1, "r", label=label)
        if ymax and ymin:
            plt.fill_between(x1, ymin, ymax, facecolor="red",
                edgecolor="red", alpha=0.3)
        plt.legend()

    def plot_data(self, X, Y, dY, label, plotrange=None, fmt="xb"):
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
            plt.errorbar(X[l:u], Y[l:u], dY[l:u], fmt=fmt, label=label)
        else:
            # plot the data
            plt.errorbar(X, Y, dY, fmt=fmt, label=label)
        plt.legend()

    def plot_data_annotate(self, X, Y, dY, fitfunc, args, label, plotrange=None,
              fitrange=None, addpars=None, pval=None, hconst=None, vconst=None):
        """Plot data with an optional annotation on x and y axis
        """
        self.plot_data_with_fit(self, X, Y, dY, fitfunc, args, label, plotrange=None,
               fitrange=fitrange, addpars=addpars, pval=pval)
        self.decorate_plot(hconst=hconst, vconst=vconst)
        
    def decorate_plot(self, hconst=None, vconst=None):
        """Decorate the plot with an optional horizontal and vertical constant
        """
        # Plotting an additional constant
        if isinstance(hconst, (np.ndarray,list,tuple)):
            plt.axhline(hconst[0],color='#b58900')
            plt.text(X[0],hconst[0]+X[0]/100.,label[5])
            plt.axhspan(hconst[0]+hconst[1],hconst[0]-hconst[1],alpha=0.35,color='gray')
        if isinstance(vconst, (np.ndarray,list,tuple)):
            plt.axvline(vconst[0],color='#859900')
            plt.text(vconst[0],Y[0],label[6])
            plt.axvspan(vconst[0]+vconst[1],vconst[0]-vconst[1],alpha=0.35,color='gray')

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
        uwidth = 0.7 * (ubins[1] - ubins[0])
        center = (bins[:-1] + bins[1:]) / 2
        ucenter = (ubins[:-1] + ubins[1:]) / 2
    
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
        plt.bar(center, hist, align='center', width=width, color='r', alpha=0.5)
        # save and clear
        self.save()
    
        # plot the unweighted histogram
        # the center and width stays the same for comparison
        plt.title(label[0])
        plt.xlabel(label[1])
        plt.ylabel('unweighted distribution')
        plt.grid(True)
        # plot
        plt.bar(ucenter, uhist, align='center', width=uwidth, color='b', alpha=0.5)
        # save and clear
        self.save()

    def _genplot_single(self, corr, label, fitresult=None, fitfunc=None,
            add=None, xshift=0., rel=False, debug=0, join=False):
        """Plot the data of a Correlators object and a FitResult object
        together.

        Parameters
        ----------
        corr : Correlators
            The correlation function data.
        label : list of strs
            The title of the plot, the x- and y-axis titles, and the data label.
        fitresult : FitResult, optional
            The fit data.
        fitfunc : LatticeFit, optional
            The fit function.
        add : ndarray, optional
            Additional arguments to the fit function. This is stacked along
            the third dimenstion to the oldfit data.
        xshift : Optional scalar shift in xrange
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
        if fitresult is not None:
            ranges = fitresult.fit_ranges
            shape = fitresult.fit_ranges_shape
        X = np.linspace(0.+xshift, float(corr.shape[1])+xshift, corr.shape[1], endpoint=False)
        label_save = label[0]

        # iterate over correlation functions
        for n in range(ncorr):
            if debug > 1:
                print("plotting correlators %d" % (n))
            mdata, ddata = compute_error(corr.data[:,:,n])
            
            if fitresult is None:
                # set up labels
                label[0] = "%s, pc %d" % (label_save, n)
                self.set_title(label[0], label[1:3])
                # plot
                self._set_env_normal()
                # plot the relative error instead of data and error
                if rel is True:
                  self.plot_data(X, np.divide(ddata,corr.data[0,:,n]),
                      np.zeros_like(ddata), label=label[3], plotrange=[3,T])
                else:
                  self.plot_data(X, corr.data[0,:,n], ddata, label[3],
                          plotrange=[3,T])
                plt.legend()
                if join is False:
                  self.save()
            else:
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
                    self.plot_data(X, corr.data[0,:,n], ddata, label[3],
                            plotrange=[3,T])
                    self.plot_function(fitfunc.fitfunc, X, mpar, label[4],
                            add, fi)
                    plt.legend()
                    if join is False:
                      self.save()
        label[0] = label_save

    def _genplot_comb(self, corr, label, fitresult, fitfunc, oldfit, add=None,
            oldfitpar=None, xshift=0., debug=0):
        """Plot the data of a Correlators object and a FitResult object
        together.

        Parameters
        ----------
        corr : Correlators
            The correlation function data.
        label : list of strs
            The title of the plot, the x- and y-axis titles, and the data label.
        fitresult : FitResult
            The fit data.
        fitfunc : LatticeFit
            The fit function.
        oldfit : None or FitResult, optional
            Reuse the fit results of an old fit for the new fit.
        add : ndarray, optional
            Additional arguments to the fit function. This is stacked along
        oldfitpar : None, int or sequence of int, optional
            Which parameter of the old fit to use, if there is more than one.
                the third dimenstion to the oldfit data.
        xshift : Optional scalar shift for xdata
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
        X = np.linspace(0.+xshift, float(corr.shape[1])+xshift, corr.shape[1], endpoint=False)
        label_save = label[0]
        T = corr.shape[1]
        franges = fitresult.fit_ranges
        fshape = fitresult.fit_ranges_shape
        print(fshape)

        # iterate over correlation functions
        ncorriter = [[x for x in range(n)] for n in ncorrs]
        for item in itertools.product(*ncorriter):
            if debug > 1:
                print("plotting correlators %s" % str(item))
            n = item[-1]
            mdata, ddata = compute_error(corr.data[:,:,n])
            # create the iterator over the fit ranges
            tmp = [fshape[i][x] for i,x in enumerate(item)]
            print(tmp)
            #limit fitranges to be plotted
            rangesiter = [[x for x in range(m)] for m in tmp ]
            # iterate over the fit ranges
            for ritem in itertools.product(*rangesiter):
                if debug > 1:
                    print("plotting fit ranges %s" % str(ritem))
                r = ritem[-1]
                print r
                # get fit interval
                fi = franges[n][r]
                _par = fitresult.get_data(item + ritem)

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
                self._set_env_normal()
                self.plot_data(X, corr.data[0,:,n], ddata, label[3],
                        plotrange=[4,T])
                self.plot_function(fitfunc.fitfunc, X, _par, label[4], 
                        add_data, fi)
                keys = [r'$\delta E$:',r'$\Delta\delta E$:']
                print _par
                #vals = 
                #self._print_label(, vals, xpos=0.7, ypos=0.8)
                plt.legend()
                self.save()

    def plot(self, corr, label, fitresult=None, fitfunc=None, oldfit=None,
            add=None, oldfitpar=None, xshift=0., rel=False, join=False, debug=0):
        """Plot the data of a Correlators object and a FitResult object
        together.

        Parameters
        ----------
        corr : Correlators
            The correlation function data.
        label : list of strs
            The title of the plot, the x- and y-axis titles, and the data label.
        fitresult : FitResult, optional
            The fit data.
        fitfunc : LatticeFit, optional
            The fit function.
        oldfit : FitResult, optional
            Reuse the fit results of an old fit for the new fit.
        add : ndarray, optional
            Additional arguments to the fit function. This is stacked along
            the third dimenstion to the oldfit data.
        oldfitpar : None, int or sequence of int, optional
            Which parameter of the old fit to use, if there is more than one.
        xshift : optional shift of xrange
        debug : int, optional
            The amount of info printed.
        """
        if oldfit is None:
            self._genplot_single(corr, label, fitresult, fitfunc, add=add,\
                    xshift=xshift, rel=rel, join=join, debug=debug)
        else:
            self._genplot_comb(corr, label, fitresult, fitfunc, oldfit, add,\
                    oldfitpar, xshift=xshift, debug=debug)

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
        if fitresult.derived:
            w = fitresult.weight[0]
            for i, d in enumerate(fitresult.data):
                label[0] = " ".join((label_save, str(fitresult.label[i])))
                self.plot_histogram(d[0], w[i], label)
        else:
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

    def history(self, data, label, par=None):
        self._set_env_normal()
        self.set_title(label[0],label[1:3])
        self.plot_data(np.arange(data.shape[0]),data,np.zeros_like(data),label[-1])
        self.save()
        

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

    def cov_plot(self, data, label, cut=False, inverse=False, norm=True):
        """ This function is used as a wrapper to plot_covariance
        """
        print(label)
        self.plot_covariance(data, label, cut=cut, inverse=inverse, norm=norm)

    def corr_plot(self, data, label,inverse=False):
        """ This function is used as a wrapper to plot_covariance
        """
        print(label)
        self.plot_correlation(data, label, inverse=inverse)

    def plot_correlation(self, data, label, inverse=False):
        """Plots the covariance matrix of given data

        Parameters
        ----------
        data : The data to plot the covariance matrix from
        """
        cov = np.corrcoef(data)
        print cov.shape
        if inverse is True:
          cov = np.linalg.inv(cov)
        print cov
        self.set_title(label[0],label[1])
        self.set_env(xlim=[0,cov.shape[0]],ylim = [0,cov.shape[0]])
        plt.pcolor(cov, cmap=matplotlib.cm.bwr, vmin=np.amin(cov),
            vmax=np.amax(cov))
        plt.colorbar()
        self.save()
        plt.clf()
        
    def plot_covariance(self, data, label, cut=False, inverse=False, norm=True):
        """Plots the covariance matrix of given data

        Parameters
        ----------
        data : The data to plot the covariance matrix from
        """
        cov1 = np.cov(data)
        print cov1.shape
        cov = np.empty(cov1.shape)
        # building the heat map of covariance matrix and filter all unwanted parts out
        if norm is True:
          for i in range(0, cov.shape[0]):
              for j in range(0, cov.shape[1]):
                    #cov[i,j] = cov1[i,j]/np.sqrt(cov1[i,i]*cov1[j,j])
                    cov[i,j] = cov1[i,j]
                    if cut is True:
                      if cov[i,j] < 0.2 and cov[i,j] > -0.2:
                          cov1[i,j] = 0.0
                    #cov[i,j] = cov1[i,j]/np.sqrt(cov1[i,i]*cov1[j,j])
                    cov[i,j] = cov1[i,j]/(data[i,0]*data[j,0])
        else:
          for i in range(0, cov.shape[0]):
              for j in range(0, cov.shape[1]):
                    cov[i,j] = cov1[i,j]
                    if cut is True:
                      if cov[i,j] < 0.2 and cov[i,j] > -0.2:
                          cov1[i,j] = 0.0
                    cov[i,j] = cov1[i,j]

        if inverse is True:
          cov = np.linalg.inv(cov)
        print(np.linalg.cond(cov))
        self.set_title(label[0],label[1])
        self.set_env(xlim=[0,cov.shape[0]],ylim = [0,cov.shape[0]])
        plt.pcolor(cov, cmap=matplotlib.cm.bwr, vmin=np.amin(cov),
            vmax=np.amax(cov))
        plt.colorbar()
        self.save()
        plt.clf()
                                  

if __name__ == "__main__":
    pass
