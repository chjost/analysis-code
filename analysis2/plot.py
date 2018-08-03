"""
The class for plotting.
"""

import numpy as np
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
#matplotlib.rcParams['axes.labelsize']='large'
from fit import LatticeFit, FitResult
from correlator import Correlators
from statistics import compute_error, draw_gauss_distributed, acf
from plot_functions import plot_data, plot_function, plot_function_multiarg, plot_histogram
from plot_layout import set_plotstyles, set_layout
from plot_utils import get_hist_bins
from in_out import check_write
from chipt_basic_observables import *
import chiral_utils as chut

class LatticePlot(object):
    def __init__(self, filename, join=False, debug=0):
        """Initialize a plot.

        Parameters
        ----------
        filename : str
            The filename of the plot.
        """
        check_write(filename)
        self.plotfile = PdfPages(filename)
        # plot environment variables
        self.xlog=False
        self.ylog=False
        self.xlim=None
        self.ylim=None
        self.grid=False
        self.legend=True
        self.join=join
        self.title=None
        self.filename=filename
        # set debug level for depending funtions default is no debug (0)
        self.debug=debug
        if join:
            self.cycol = itertools.cycle('bgrcmk').next
            self.cyfmt = itertools.cycle('^vsd>o').next
        else:
            self.cycol = itertools.cycle('b').next
            self.cyfmt = itertools.cycle('^').next

    def __del__(self):
        self.plotfile.close()

    def new_file(self, filename):
        """Open a new plot file.
        """
        self.plotfile.close()
        plt.clf()
        check_write(filename)
        self.plotfile = PdfPages(filename)

    def save(self):
        if self.title is not None:
            plt.title(self.title)
        if plt.get_fignums():
            for i in plt.get_fignums():
                self.plotfile.savefig(plt.figure(i))
        else:
            plt.savefig()
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

    def _est_env_hist(self):
        plt.grid(self.grid)

    def _set_limits(self,marks):
        plt.locator_params(nbins=marks)

    def set_title(self, title, axis):
        """Set the title and axis labels of the plot.

        Parameters
        ----------
        title : str
            The title of the plot.
        axis : list of strs
            The labels of the axis.
        """
        if self.title is True:
            plt.title(title)
        plt.xlabel(axis[0],fontsize=24)
        plt.ylabel(axis[1],fontsize=24)

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

    def plot_data_annotate(self, X, Y, dY, fitfunc, args, label, plotrange=None,
              fitrange=None, addpars=None, pval=None, hconst=None, vconst=None):
        """Plot data with an optional annotation on x and y axis
        """
        plot_data_with_fit(self, X, Y, dY, fitfunc, args, label, plotrange=None,
               fitrange=fitrange, addpars=addpars, pval=pval)
        self.decorate_plot(hconst=hconst, vconst=vconst)
        
    # TODO: repair function
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

    def _save_data_ascii(self,x,y,dy,label):
        _head = label[1]+"\t"+label[2]+"\td"+label[2]
        _dat = np.column_stack((x,y,dy))
        np.savetxt(self.filename+".dat",_dat,delimiter='\t', header=_head)
    
    def _genplot_ratio_single(self, corr, label, fitresult=None, fitfunc=None,
            add=None, xshift=0., ploterror=False, rel=False, debug=0,
            join=False):
        """ Plot ratio of correlator and evaluated fit function
        """
        # iterate over fit intervals, calculate ratio per fitrange
        label_save = label[0]
        X = np.linspace(0., float(corr.shape[1]), corr.shape[1], endpoint=False) + xshift
        T = corr.shape[1]
        ranges = fitresult.fit_ranges
        shape = fitresult.fit_ranges_shape
        label[0] = "%s, pc %d" % (label_save, 0)
        for r in range(shape[1][0]):
            if debug > 1:
                print("plotting fit ranges %s" % str(r))
            fi = ranges[0][r]
            _datlabel = label[3]
            # TODO data used here stem from combined fit
            par = fitresult.data[0][0,:,0,r]
            # set up labels
            self.set_title(label[0], label[1:3])
            rangelabel = "fit [%d, %d]" % (fi[0], fi[1])

            # plot
            self._set_env_normal()
            # This should give a T/2 array
            corr_fit = fitfunc.fitfunc(par,X,add)
            label_save = label[0]
            ratio = corr.data[...,0]/corr_fit
            mdata, ddata = compute_error(ratio)
            plot_data(X+xshift, ratio[0,:], ddata, rangelabel,
                    plotrange=[1,T],col=self.cycol())
            plt.legend(loc='best')
            plt.ylim([0.85,1.25])
            plt.axhline(y=1,color='k')
            if self.join is False:
              self.save()
        label[0] = label_save

    def _genplot_single(self, corr, label, fitresult=None, fitfunc=None,
            add=None, xshift=0., ploterror=False, rel=False, debug=0,
            join=False):
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
        X = np.linspace(0., float(corr.shape[1]), corr.shape[1], endpoint=False) + xshift
        label_save = label[0]

        # iterate over correlation functions
        print("Number of correlators is: %d" %ncorr)
        x_shift=-0.1
        for n in range(ncorr):
            # Check if there are ncorr datalabels
            if len(label[3]) == ncorr:
                _datlabel = label[3][n]
            else:
                _datlabel = label[3]
            if debug > 1:
                print("plotting correlators %d" % (n))
            mdata, ddata = compute_error(corr.data[:,:,n])
            #mdata = np.mean(corr.data[:,:,n],axis=0)

            #data_var = np.sum(np.square(np.diff(corr.data[:,:,n]-mdata,axis=0)),axis=0)/(corr.shape[0]*(corr.shape[0]-1))
            #ddata=np.sqrt(data_var)
            if fitresult is None:
                # set up labels
                label[0] = "%s, pc %d" % (label_save, n)
                self.set_title(label[0], label[1:3])
                # plot
                self._set_env_normal()
                # plot the relative error instead of data and error
                if rel is True:
                    #plot_data(X+xshift, np.divide(ddata,corr.data[0,:,n]),
                    #    np.zeros_like(ddata), label=_datlabel,
                    #    plotrange=[0,T],col=self.cycol(),fmt=self.cyfmt())
                    plot_data(X+xshift, ddata,
                        np.zeros_like(ddata), label=_datlabel,
                        plotrange=[0,T],col=self.cycol(),fmt=self.cyfmt())
                else:
                    # print data to screen
                    if debug > 3:
                      self._save_data_ascii(X, corr.data[0,:,n], ddata, label)
                    plot_data(X+xshift, corr.data[0,:,n], ddata, _datlabel,
                        plotrange=[0,T],col=self.cycol(),fmt=self.cyfmt())
                plt.legend(loc='best')
                if self.title is not None:
                    self.set_title(label[0], label[1:3])
                if self.join is False:
                  self.save()
            else:
                # iterate over fit intervals
                for r in range(shape[0][n]):
                    if debug > 1:
                        print("plotting fit ranges %s" % str(r))
                    fi = ranges[n][r]
                    mpar, dpar = compute_error(fitresult.data[n][:,:,r])
                    #print(mpar,dpar)

                    # set up labels
                    label[0] = "%s, pc %d" % (label_save, n)
                    self.set_title(label[0], label[1:3])
                    label[4] = "fit [%d, %d]" % (fi[0], fi[1])

                    # plot
                    self._set_env_normal()
                    plot_data(X+xshift, corr.data[0,:,n], ddata, _datlabel,
                            plotrange=[1,T],col=self.cycol())
                    # The argument X has changed
                    #_X = [X[0],X[-1]]
                    plot_function(fitfunc.fitfunc, fi, mpar, label[4],
                            add, fi, ploterror,col=self.cycol())
                    plt.legend(loc='best')
                    if self.join is False:
                      self.save()
            xshift+=0.1
        label[0] = label_save

    def _genplot_comb(self, corr, label, fitresult, fitfunc, oldfit, add=None,
            oldfitpar=None, ploterror=False, xshift=0., debug=0):
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
        X = np.linspace(0., float(corr.shape[1]), corr.shape[1], endpoint=False) + xshift
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
            print(corr.data.shape)
            mdata, ddata = compute_error(corr.data[:,:, n])
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
                plot_data(X, corr.data[0,:, n], ddata, label[3],
                        plotrange=[1,T])
                plot_function(fitfunc.fitfunc, fi, _par, label[4], 
                        add_data, fi, ploterror)
                plt.legend()
                self.save()


    def plot(self, corr, label, fitresult=None, fitfunc=None, oldfit=None,
            add=None, oldfitpar=None, ploterror=False, rel=False, xshift=0., debug=0):
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
        ploterror : bool, optional
            Plot the error of the fit function.
        xshift : float
            optional shift of xrange
        debug : int, optional
            The amount of info printed.
        """
        if oldfit is None:
            self._genplot_single(corr, label, fitresult, fitfunc, add=add,
                    ploterror=ploterror, xshift=xshift, rel=rel, debug=debug)
        else:
            self._genplot_comb(corr, label, fitresult, fitfunc, oldfit, add,
                    oldfitpar, ploterror, xshift, debug)

    def plot_ratio(self, corr, label, fitresult=None, fitfunc=None, oldfit=None,
            add=None, oldfitpar=None, ploterror=False, rel=False, xshift=0., debug=0):
        """Plot the ratio of a Correlators object and a FitResult object.

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
        ploterror : bool, optional
            Plot the error of the fit function.
        xshift : float
            optional shift of xrange
        debug : int, optional
            The amount of info printed.
        """
        self._genplot_ratio_single(corr, label, fitresult, fitfunc, add=add,
                 ploterror=ploterror, xshift=xshift, rel=rel, debug=debug)
        #else:
        #    self._genplot_comb(corr, label, fitresult, fitfunc, oldfit, add,
        #            oldfitpar, ploterror, xshift, debug)
    def plot_matrix( self, corr, label, fitresult=None, fitfunc=None, oldfit=None,
            add=None, oldfitpar=None, ploterror=False, rel=False, xshift=0., debug=0):
        """Plot a correlation function matrix
        """
        if corr.matrix is False:
            RuntimeError("Can only plot correlator matrix")
        else:
            rows = corr.data.shape[-2] 
            cols = corr.data.shape[-1] 
            fig, axes = plt.subplots(nrows = rows, ncols = cols,
                                     sharex='all',sharey='all')
            X = np.linspace(0., float(corr.shape[1]), corr.shape[1],
                            endpoint=False) + xshift
            label_save = label[0]
            self._set_env_normal()
            for r in range(rows):
                for c in range(cols):
                    mdata, ddata = compute_error(corr.data[:,:,r,c])
                    print("Plotting: %d\t%d" %(r,c))
                    print(corr.data[0,:,r,c])
                    axes[r,c].errorbar(X,corr.data[0,:,r,c],ddata,fmt = 'ob',
                            label = r'$C_{%d%d}$'%(r,c))
                    if c == 0:
                        axes[r,c].set_ylabel(label[2])
                    if r == cols-1:
                        axes[r,c].set_xlabel(label[1])
                    axes[r,c].legend()
            fig.tight_layout()
            print(self.filename)
            self.save()

    def bs_hist(self,data,label,nb_bins):
        plt.hist(data,nb_bins,normed=True,alpha=0.55,label=label[0])
        plt.title(self.title)
        plt.xlabel(label[1])
        plt.ylabel('Frequency')
        plt.legend()


    def histogram(self, fitresult, label, nb_bins=20, par=None):
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
        if nb_bins is None:
            _bins = get_hist_bins(fitresult.fit_ranges_shape)
        else:
            _bins = nb_bins
        if fitresult.derived:
            w = fitresult.weight[0]
            for i, d in enumerate(fitresult.data):
                label[0] = " ".join((label_save, str(fitresult.label[i])))
                plot_histogram(d[0], w[i], label, nb_bins=_bins)
                plt.legend()
                self.save()
        else:
            if par is None:
                for p, w in enumerate(fitresult.weight):
                    for i, d in enumerate(fitresult.data):
                        label[0] = " ".join((label_save, str(fitresult.label[i])))
                        plot_histogram(d[0,p], w[i], label, nb_bins=_bins)
                        plt.legend()
                        self.save()
            else:
                w = fitresult.weight[par]
                for i, d in enumerate(fitresult.data):
                    label[0] = " ".join((label_save, str(fitresult.label[i])))
                    plot_histogram(d[0,par], w[i], label, nb_bins=_bins)
                    plt.legend()
                    self.save()
        label[0] = label_save
    
    def correlogram(self, correlator, label, start=0, num=0):
        """Plot the autocorrelation versus lagtime for data in a correlator
        object


        Parameters
        ----------
        correlator : a Correlator object
        label : the label of the correlogram
        start : time index where to start the calculation of acf
        num : which entry of the correlator object to take
        """
        correlogram_raw = acf(correlator.data,start=start)
        _mean,_std = compute_error(correlogram_raw,axis=0)
        print("Data shapes in correlogram:")
        print(_mean.shape)
        print(_std.shape)
        plt.stem(_mean)
        plt.xlim(-0.2,correlator.data.shape[1])
        if self.title is True:
            plt.title(label[0])
        plt.xlabel(r'Lag $k$ from $t_{i}=$%s' %start)
        plt.ylabel(r'acf($k$)')
        self.save()
        plt.clf()


    def qq_plot(self, fitresult, label, par=0, corr=0, fitrange=False):
        """A quantile-quantile-plot for Fitresults

        Calculate the theoretical quantiles (gaussian with mean and standard
        deviation from fitresult), and the measured ones, then plot them against
        each other together with a straight line to visualize the deviation

        Parameters
        ----------
        fitresult : FitResult object
        label : str title of plot
        par : int, which parameter of fit result should be taken
        fitrange : bool, if True every fitrange is plotted separately
        """
        if fitrange is False:
            if fitresult.error is None:
              fitresult.calc_error()
            q_meas = fitresult.error[par][corr][0]
            _dummy,_std = compute_error(q_meas)
            print("Gaussian input:")
            print("mean : %.4f" %_dummy)
            print("stdev : %.4f" % _std)
            # draw gaussian distributed data with same size
            q_theo = draw_gauss_distributed(q_meas[0],_std ,(q_meas.shape[0],))
            # Lambda function for bisection
            bisec = lambda p,x : x
            plot_data(np.sort(q_theo),np.sort(q_meas),None,label[1],
                fmt='o')
            _range = (np.amin(np.sort(q_theo)),np.amax(np.sort(q_theo)))
            plot_function(bisec,_range,None,'',fmt='r')
            plt.legend(loc='best')
            plt.locator_params(axis='x',nbins=4)
            plt.locator_params(axis='y',nbins=4)
            if self.title is True:
                plt.title(label[0])
            plt.xlabel(r'theoretical quantiles $N_{\bar{\mu},\Delta\mu}$',fontsize=24)
            plt.ylabel(r'measured quantiles',fontsize=24)
            #plt.axes().set_aspect('equal')
            self.save()
            plt.clf()

        else:
            
            # set up plot
            if self.title is True:
                plt.title(label[0])
            plt.xlabel(r'theoretical quantiles $N_{\bar{\mu},\Delta\mu}$')
            plt.ylabel(r'measured quantiles')
            print("QQ-plot for multiple fitranges")
            # loop over fitranges (if shape is correct)
            #if len(fitresult.fit_ranges_shape) == 1:
            shape = fitresult.fit_ranges_shape
            for r in range(shape[1][0]):
                plt.clf()
                #print("Fit range index: %d" %r)
                q_meas = fitresult.data[corr][:,par,0,r]
                #print("plotting fit ranges %s" % str(r))
                _dummy, _std = compute_error(fitresult.data[corr][:,:,0,r])
                #print ("Result from compute error:")
                #print(_dummy,_std)
                #print("Gaussian input: mean : %.4f; stdev: %.4f"
                #    %(_dummy[par],_std[par]))
                # draw gaussian distributed data with same size
                q_theo = draw_gauss_distributed(_dummy[par],_std[par] ,(fitresult.data[0].shape[0],))
                # Lambda function for bisection
                bisec = lambda p,x : x
                plot_data(np.sort(q_theo),np.sort(q_meas),None,fitresult.fit_ranges[0,r],
                    fmt='o',markerfill ='none')
                plot_function(bisec,np.sort(q_theo),None,'',fmt='r')
                plt.legend(loc='lower right')
                plt.locator_params(axis='x',nbins=4)
                plt.locator_params(axis='y',nbins=4)
                #plt.axes().set_aspect('equal')
                self.save()
            #
            #self.save()


    def plot_func(self, func, args, interval, label, fmt="k", col="black"):
        X = np.linspace(interval[0], interval[1], 1000)
        plot_function(func, X, args, label, ploterror=True, fmt=fmt, col=col)

    def history(self, data, label, ts=0, boot=False, par=None, fr=None, subplot=True):
        """Plots the history of the input data either with bootstrapsamples or
        configuration number

        Parameters
        ----------
        data : a Corr or FitResult object, atm only one Correlator is supported
        label : the label of the plot, x-axis is set automatically
                taken
        boot : bool, is data correlator or FitResult
        ts : int, timeslice to take for configuration history
        par : int, parameter from a fitresult
        fr : int, fit range index
        """
        self._set_env_normal()
        self.set_title(label[0],label[1:3])
        print data.data[0].shape
        if boot is False:
          _data = data.data[:,ts,0]
          #if data.conf is not None:
          #    _x = np.unique(data.conf)
          #else:
          _x = np.arange(_data.shape[0])
          plot_data(_x,_data,np.zeros_like(_data),label[-1])
        else:
          if data.conf is not None:
              _x = np.unique(data.conf)
          else:
              _x = np.arange(_data.shape[0])
          # take all bootstrapsamples
          if len(data.data[0].shape)>3:
            _data = data.data[0][:,par,0]
            print(_data.shape)
          else:
            _data = data.data[0][:,par]
            print(_data.shape)
          if subplot:
            plt.subplot(2,1,1)
            _lbl = 'fitrange %d' % (fr)
            plot_data(_x,_data[:,fr],np.zeros_like(_data[:,fr]),_lbl)
            plt.subplot(2,1,2)
            _lbl = 'fitrange %d' % (fr+1)
            plot_data(_x,_data[:,fr],np.zeros_like(_data[:,fr]),_lbl)
          else:
            _data = data.data[0][:,fr]
            print(_data.shape)
            plot_data(_x,_data,np.zeros_like(_data),label[-1])

        self.save()

    def set_env(self, xlog=False, ylog=False, xlim=None, ylim=None,
        grid=True,title=False):
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
        self.title=title

    def cov_plot(self, data, label, cut=False, inverse=False, norm=True,corr=0):
        """ This function is used as a wrapper to plot_covariance
        """
        print(label)
        if len(data.shape) == 3:
          _data = data.data.squeeze().T
        else:
          _data = data.data.T
        self.plot_covariance(_data, label, cut=cut, inverse=inverse, norm=norm)

    def corr_plot(self, data, label,inverse=False,corr=0,tstart=0):
        """ This function is used as a wrapper to plot_covariance
        """
        print("Data shape handed to corr_plot")
        print(data.data.shape)
        _data = data.data
        if tstart > 0:
          _data = _data[:,tstart:]
          print("Modified data to shape")
          print(_data.shape)
        if len(data.data.shape) == 3:
          print("squeezing array")
          _data = _data.squeeze().T
        else:
          print("transpose array")
          _data = _data.T
        self.plot_correlation(_data, label, inverse=inverse)

    
    def plot_correlation(self, data, label, inverse=False):
        """Plots the covariance matrix of given data

        Parameters
        ----------
        data : The data to plot the covariance matrix from
        """
        print("data shape handed to plot_correlation")
        print(data.shape)
        cov = np.corrcoef(data)
        print cov.shape
        if inverse is True:
          cov = np.linalg.inv(cov)
        print cov
        if self.title is True:
            plt.title(label[0])
        plt.figure(figsize=(14,12))
        plt.xlabel(label[1])
        plt.ylabel(label[1])
        #plt.xticks(np.arange(3),(0.0185,0.0225,0.02464))
        #plt.yticks(np.arange(3),(0.0185,0.0225,0.02464))
        self.set_env(xlim=[0,cov.shape[0]],ylim = [0,cov.shape[0]])
        #plt.pcolor(cov, cmap=matplotlib.cm.bwr, vmin=np.amin(cov),
        #    vmax=np.amax(cov))
        plt.pcolor(cov, cmap=matplotlib.cm.bwr, vmin=-1,
            vmax=1)
        plt.colorbar()
        #Ensure quadratic plot
        self.save()
        plt.clf()

    def plot_heatmap(self,data,label):
        """ plot a general heatmap of symmetric data"""

        if data.shape[0] != data.shape[1]:
          raise ValueError("data not symmetric")
        self.set_title(label[0],label[1])
        self.set_env(xlim=[0,data.shape[0]],ylim = [0,data.shape[0]])
        plt.pcolor(data, cmap=matplotlib.cm.bwr, vmin=np.amin(data),
            vmax=np.amax(data))
        plt.colorbar()
        self.save()
        plt.clf()
        
    def plot_covariance(self, data, label, cut=False, inverse=False, norm=True):
        """Plots the covariance matrix of given data

        Parameters
        ----------
        data : The data to plot the covariance matrix from
        """
        print data.shape
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
        print(cov)
        self.set_title(label[0],label[1])
        self.set_env(xlim=[0,cov.shape[0]],ylim = [0,cov.shape[0]])
        plt.pcolor(cov, cmap=matplotlib.cm.bwr, vmin=np.amin(cov),
            vmax=np.amax(cov))
        plt.colorbar()
        self.save()
        plt.clf()

    def delete(self):
      if self.join is True:
        self.save()
      del self

#TODO: code doubling
    def plot_single_line(self,x,y,label,col):
      """plot horizontal and vertical lines at the specific points, labeled with
      the specific values
    
      Parameters
      ----------
      x,y : in general multidimensional ndarrays 
      """
      #print("line properties")
      #print(x.shape)
      #print(y)
      x_val=np.zeros(2)
      y_val=np.zeros(2)
      try:
         x_val[0],x_val[1] = compute_error(x)
      except:
         x_val[0] = x
         x_val[1] = 0
      try:
         y_val[0],y_val[1] = compute_error(y)
      except:
         y_val[0] = y
         y_val[1] = 0
      #print(x_val)
      #print(y_val)
      #l = plt.axhline(y=y_val[0],ls='solid',color=col)
      #l = plt.axvline(x=x_val[0],ls='solid',color=col)
      plt.errorbar(x_val[0],y_val[0],y_val[1],x_val[1],fmt =
          #'d',color=col,label=r'%2.4e,%2.4e' % (x_val[0],y_val[0]) )
          'd',color=col,label=label)
      plt.legend()
    
    def plot_comparison(self,chirana,beta,label,xlim,ylim=None,dep=None,debug=0):
        # Plot the data for the given lattice spacings
        # Initialize symbols and colors for lattice spacings
        col = ['r','b','g']
        fmt_pts = ['^','v','o']

        for i,a in enumerate(beta):
            _x, _y = conv_data_plot()
            # get data for beta, the data passed should be 3 arrays (X,Y,dy)
            # the quark mass values
            if dep is not None:
                _X = chirana.x_data[i][:,:,dep,0].flatten()
                _dX = chirana.x_data[i][:,:,dep,:].reshape((chirana.x_data[i].shape[0]*chirana.x_data[i].shape[1],chirana.x_data[i].shape[-1]))
            else:
                _X = chirana.x_data[i][:,:,0,0].flatten()
                _dX = chirana.x_data[i][:,:,0,:].reshape((chirana.x_data[i].shape[0]*chirana.x_data[i].shape[1],chirana.x_data[i].shape[-1]))
            _Y = chirana.y_data[i][:,:,0,0].flatten()
            _dY = chirana.y_data[i][:,:,0,:].reshape((chirana.y_data[i].shape[0]*chirana.y_data[i].shape[1],chirana.y_data[i].shape[-1]))
            print("yerror shape is:")
            print(_Y)
            x_mean, _dx = compute_error(_dX,axis=1)
            y_mean, _dy = compute_error(_dY,axis=1)
            if debug > 0:
                print("x_data to be plotted:")
                print(_X)
                print("y_data to be plotted:")
                print(_Y)
            #plot_data(_X,_Y,_dy,label=a,col=col[i],fmt=fmt_pts[i])
            plot_data(_X,_Y,_dy,label=a,dX=_dx,col=col[i],fmt=fmt_pts[i])
            plt.xlabel(label[0])
            plt.ylabel(label[1])
        plt.xlim(xlim[0],xlim[1])
        if ylim is not None:
            plt.ylim(ylim[0],ylim[1])
        plt.xlabel(label[0])
        plt.ylabel(label[1])
        if len(label) > 2:
                plt.title(label[2])
        plt.legend(loc='best',ncol=1,fontsize=14)
        if self.join is False:
          self.save()
          plt.clf()
        

    def plot_chiral_fit(self,chirana,beta,label,xlim,ylim=None,
                        func=None,args=None,x_phys=None):
        """ Function to plot a chiral fit.
        
        This function sets up a plotter object, puts in the data in the right
        shape and plots the data itself as well as the function
        
        """
        # Plot the data for the given lattice spacings
        # Initialize symbols and colors for lattice spacings
        col = ['r','b','g']
        fmt_pts = ['x','x','x']

        for i,a in enumerate(beta):
            # get data for beta, the data passed should be 3 arrays (X,Y,dy)
            # the light quark mass values
            _X = chirana.x_data[i][:,:,0,0].flatten()
            _Y = chirana.y_data[i][:,:,0,0].flatten()
            _dy = chirana.y_data[i][:,:,0,:].reshape((chirana.y_data[i].shape[0]*chirana.y_data[i].shape[1],chirana.y_data[i].shape[-1]))
            print("data for error calculation")
            print(_dy.shape)
            _mean, _yerr = compute_error(_dy,axis=1,mean=_Y)
            plot_data(_X,_Y,_yerr,label=a,col=col[i],fmt=fmt_pts[i],debug=self.debug)
            if self.debug > 3:
                print("Data used for plotting chiral fit:")
                print("x:")
                print(_X)
                print("y:")
                print(_Y)
                print("dy")
                print(_yerr)
            # Check if we want to plot a function in addition to the data
            if func is not None:
                if self.debug > 0:
                    print("Arguments for plotting function")
                    print(args[i])
                for s in chirana.x_data[i][0,:,1,0]: 
                   #Check for numbers of lattice spacings, if > 1 loop over args
                  if len(beta)==1:
                      plotargs = np.hstack((args,s,r))
                  else:
                   #adapt shape for errorbands in plot_function
                      _mus = np.full((args.shape[1],1),s) 
                      plotargs = np.hstack((args[i],_mus))
                  #plotargs=None
                  plot_function(func,xlim,plotargs[i],label=None,ploterror=True)
            plt.xlim(xlim[0],xlim[1])
            plt.locator_params(nbins=4)
            plt.xlabel(label[0],fontsize=24)
            plt.ylabel(label[1],fontsize=24)
            self.save()
        # plot a vertical dashed line at physical x_value
        #if func is not None:
        #    plot_function(func,xlim,args,label=r'LO $\chi$-pt',ploterror=True)
        if x_phys is not None:
            plt.axvline(x=x_phys, color='k', ls='--', label=label[0]+'_phys.')
        plt.xlim(xlim[0],xlim[1])
        #plt.ylim(ylim[0],ylim[1])
        plt.locator_params(nbins=4)
        plt.xlabel(label[0],fontsize=24)
        plt.ylabel(label[1],fontsize=24)
        if len(label) > 2:
            plt.title(label[2])
        plt.legend(loc='best',ncol=1,fontsize=14)
        if self.join is False:
          self.save()
          plt.clf()
# TODO: Move the shape_data functions to an own file
    # Dirty helper function to abbreviate plot_chiral_ext
    def shape_data_kk(self,x,y,args):
        _X = x[:,:,0,0].flatten()*args[0,0]/args[0,1]
        _Y = y[:,:,0,0].flatten()
        _dy = y[:,:,0,:].reshape((y.shape[0]*y.shape[1],y.shape[-1]))
        if self.debug > 0:
            print("argument shape:")
            print(args.shape)
            print("yerror shape is:")
            print(_dy.shape)
            print("x-data:")
            print(_X)
        return _X,_Y,_dy

    # Dirty helper function to abbreviate plot_chiral_ext
    def shape_data_pik(self,x,y,gamma=False):
        # calculate mu over fpi squared, plot_function does not support
        # arbitrary many x-values, TODO: change that
        if gamma is False:
            _X = reduced_mass(x[:,:,0,0].flatten(),
                              x[:,:,1,0].flatten())/x[:,:,2,0].flatten()
            #_X = x[:,:,1,0].flatten()/x[:,:,0,0].flatten()
        
        else:
            _X = x[:,:,0,0].flatten()
       
        _Y = y[:,:,0,0].flatten()
        _dy = y[:,:,0,:].reshape((y.shape[0]*y.shape[1],y.shape[-1]))
        if self.debug > 0:
            print("x-data:")
            print(_X)
            print("yerror shape is:")
            print(_dy.shape)
        return _X,_Y,_dy
# TODO: Split this up into at least four functions
    def plot_chiral_ext(self, chirana, beta, label, xlim, ylim=None, func=None,
                       args=None,calc_x=None, ploterror=True, kk=True,
                       gamma=False, x_phys=None,xcut=None,plotlim=None,
                       argct=None,sublo=False):
        """ Function to plot a chiral extrapolation fit.
        
        This function sets up a plotter object, puts in the data in the right
        shape and plots the data itself as well as the function
        
        """
        # Plot the data for the given lattice spacings
        # Initialize symbols and colors for lattice spacings
        if args is not None and args.shape[0] == 1:
            col = ['r','b','g']
            fmt_pts = ['^','v','o']
            fmt_ls = ['--',':','-.']
            dat_label = [r'NLO $\chi$-PT']

        else:
            col = ['r','b','g']
            fmt_pts = ['^','v','o']
            fmt_ls = ['--',':','-.']
            dat_label = [r'$a=0.0885$fm',r'$a=0.0815$fm',r'$a=0.0619$fm']
        for i,a in enumerate(beta):
            #TODO: DAC is too specialized, leave that to another function
            # get data for beta, the data passed should be 3 arrays (X,Y,dy)
            # the light quark mass values
            if kk is True:
                _X,_Y,_dy = self.shape_data_kk(chirana.x_data[i],
                                               chirana.y_data[i],args[i])
            else:
                if sublo is True:
                    lo = -(reduced_mass(chirana.x_data[i][:,:,0],
                                     chirana.x_data[i][:,:,1])/chirana.x_data[i][:,:,2])**2/(4.*np.pi)
                    _lo = np.zeros_like(chirana.y_data[i])
                    _lo[:,:,0] = lo
                    y_in = (chirana.y_data[i]-_lo)/chirana.y_data[i]
                else:
                    y_in = chirana.y_data[i]
                _X,_Y,_dy = self.shape_data_pik(chirana.x_data[i],
                                                y_in,gamma=gamma)
            _mean, _dy = compute_error(_dy,axis=1)
            plot_data(_X,_Y,_dy,label=a,col=col[i],fmt=fmt_pts[i],alpha=1.,
                      debug=self.debug)
            # Check if we want to plot a function in addition to the data
            # check for lattice spacing dependence
        if func is not None:
            if args.shape[0] > 1:
                for i,a in enumerate(beta):
                    # check for lattice spacing dependence
                    if argct is "multiarg":
                        print("\nIn plot_function args:")
                        print(args[i][0])
                        plot_function_multiarg(func,xlim,args[i],calc_x=calc_x,
                                            label=dat_label[i],
                                            ploterror=ploterror,
                                            fmt=col[i]+fmt_ls[i],col=col[i],
                                            debug=self.debug)
                    else:
                        plot_function(func,xlim,args[i],calc_x=calc_x,
                                  label=dat_label[i], ploterror=ploterror,
                                  fmt=col[i]+fmt_ls[i],col=col[i],
                                  debug=self.debug)

            if args.shape[0]==1:
                col='k'
                fmt_ls ='-'
                if argct is "multiarg":
                    plot_function_multiarg(func,xlim,args[0],calc_x=calc_x,
                                        label=dat_label[0],
                                        ploterror=ploterror,
                                        fmt=col+fmt_ls,col=col,
                                        debug=self.debug)
                else:
                    plot_function(func,xlim,args[0],calc_x=calc_x,
                              label=dat_label[0], ploterror=ploterror,
                              fmt=col+fmt_ls,col=col, debug=self.debug)


        #plt.xlabel(label[0])
        #plt.ylabel(label[1])
        #self.save()
        if x_phys is not None:
            plt.axvline(x=x_phys, color='k', ls='--', label=label[0]+'_phys.')
        # Plot the physical point as well as the continuum function
        if xcut is not None:
            if len(xcut) > 1:
                plot_brace(args,xcut,func,xpos="low")
                plot_brace(args,xcut,func,xpos="up")
                
            else:
                plot_brace(args,xcut,func,xpos="up")
        if plotlim is None:
            plt.xlim(np.amin(_X),np.amax(_X))
        else: 
            plt.xlim(plotlim[0],plotlim[1])
        if ylim is not None:
          plt.ylim(ylim[0],ylim[1])
        plt.xlabel(label[0],fontsize=24)
        plt.ylabel(label[1],fontsize=24)
        if len(label) > 2:
            plt.title(label[2])
        plt.legend(loc='lower left',ncol=2,numpoints=1)
    
    # Function to plot the chi values frim the fit
    def plot_chi_values(self, chirana, lattice_spacings,
                       fit_function, xvalue_function=None,
                       data_label="Fit evaluation",label=None,plotlim=None,
                       ylim=None,prior=None,legend=None,xcut=None):
        """Plot relative deviation of fit function evaluated at lattice input
        values from measurements

        The x-data for the plot are taken from a chiral extrapolation object.
        If given the actual x-axis points are calculated via xvalue_function and
        the fit function is used for the evaluation.
        """
        # In dependence of the lattice_spacings set layout options
        colors, markerstyles, linestyles, data_labels = set_plotstyles(lattice_spacings)
        # Treat every lattice spacing separately
        for i,a in enumerate(lattice_spacings):
            y_input = chirana.y_data[i]
            xvalues_fit = chirana.x_data[i][:,0]
            # Extract fit arguments from chirana instance, include lattice
            # artefact..
            fit_arguments = chirana.fitres.data[0][:,:]
            print("in plot_chi: ")
            print("shape of fit_arguments:")
            print(fit_arguments.shape)
            print("shape of xvalues_fit:")
            print(xvalues_fit.shape)
            y_fit = fit_function(fit_arguments.T,xvalues_fit)
            # determine covariance matrix, including prior
            _y_cov = chut.concat_data_cov(y_input,prior=prior)
            _cov = np.cov(_y_cov)
            if chirana.correlated is False:
                _cov = np.diag(np.diagonal(_cov))
            _cov = (np.linalg.cholesky(np.linalg.inv(_cov))).T
            chi_vector = np.dot(_cov,y_fit-y_input)
            print(chi_vector)
            print("\ny_fit has shape")
            print(y_fit.shape)
            # TODO: compute_error seems not to work for 3d-arrays take 0-th
            # bootstrapsamples for the time being
            xvalues_plot,dummy1,dummy2 = self.shape_data_pik(chirana.x_data[i],
                                                   chirana.y_data[i])
            _ymean, _dy = compute_error(chi_vector,axis=1)
            plot_data(xvalues_plot,_ymean,_dy,label=a,col=colors[i],
                      fmt=markerstyles[i], debug=self.debug)

        if xcut is not None:
            if len(xcut) > 1:
                plot_brace(fit_arguments,xcut,fit_function,xpos="low")
                plot_brace(fit_arguments,xcut,fit_function,xpos="up")
            else:
                plot_brace(fit_arguments,xcut,fit_function,xpos="up")
        plt.axhline(color='k')
        set_layout(physical_x=x_phys,xlimits=plotlim,ylimits=ylim,
                   legend_array=legend,labels=label,labelfontsiz=24)
            

    def plot_fit_proof(self, chirana, lattice_spacings,
                       fit_function, xvalue_function=None,
                       data_label="Fit evaluation",label=None,plotlim=None,
                       ylim=None,x_phys=None,legend=None,xcut=None):
        """Plot relative deviation of fit function evaluated at lattice input
        values from measurements

        The x-data for the plot are taken from a chiral extrapolation object.
        If given the actual x-axis points are calculated via xvalue_function and
        the fit function is used for the evaluation.
        """
        # In dependence of the lattice_spacings set layout options
        colors, markerstyles, linestyles, data_labels = set_plotstyles(lattice_spacings)
        # Treat every lattice spacing separately
        for i,a in enumerate(lattice_spacings):
            y_input = chirana.y_data[i]
            xvalues_fit = chirana.x_data[i][:,0]
            # Extract fit arguments from chirana instance, include lattice
            # artefact..
            fit_arguments = chirana.fitres.data[0][:,:,0]
            y_fit = fit_function(fit_arguments.T,xvalues_fit)
            print("\ny_fit has shape")
            print(y_fit.shape)
            # TODO: compute_error seems not to work for 3d-arrays take 0-th
            # bootstrapsamples for the time being
            xvalues_plot,dummy1,dummy2 = self.shape_data_pik(chirana.x_data[i],
                                                   chirana.y_data[i])
            y_measured=chirana.y_data[i][:,0,0]
            relative_y_deviation = (y_measured-y_fit)/y_measured
            _ymean, _dy = compute_error(relative_y_deviation,axis=1)
            plot_data(xvalues_plot,_ymean,_dy,label=a,col=colors[i],
                      fmt=markerstyles[i], debug=self.debug)

        if xcut is not None:
            if len(xcut) > 1:
                plot_brace(fit_arguments,xcut,fit_function,xpos="low")
                plot_brace(fit_arguments,xcut,fit_function,xpos="up")
            else:
                plot_brace(fit_arguments,xcut,fit_function,xpos="up")
        plt.axhline(color='k')
        set_layout(physical_x=x_phys,xlimits=plotlim,ylimits=ylim,
                   legend_array=legend,labels=label,labelfontsize=24)

    def plot_cont(self,chirana,func,xlim,args,par=None,argct=None,calc_x=None,
                  phys=True,ploterror=True,label=None,xcut=None):
      """ Plot the continuum curve of a chiral analysis and the physical point
      result
      """
      #if par is not None:
      #    #TODO: Generalize to arbitrary long lists
      #    args[0][par] = np.zeros_like(args[0][0])
      if label is not None:
        _label=label
      else:
        _label='cont'
      # Plot the continuum curve
      if argct == 'multiarg':
          plot_function_multiarg(func, xlim, args, _label,
              fmt='k-',calc_x=calc_x, ploterror=ploterror, debug=self.debug)
      elif argct == 'plain':
          print("No continuum function plotted")
      else:
          plot_function(func, xlim, args, _label, fmt='k-',
              ploterror=ploterror,debug=self.debug)
      if xcut is not None:
          if len(xcut) > 1:
              plot_brace(args,xcut,func,xpos="low")
              plot_brace(args,xcut,func,xpos="up")
              
          else:
              plot_brace(args,xcut,func,xpos="up")
      if phys ==True:
          plt.errorbar(chirana.phys_point[0,0],chirana.phys_point[1,0],
                       chirana.phys_point[1,1],xerr=chirana.phys_point[0,1],
                       fmt='d', color='darkorange', label='phys.')
      plt.legend(loc='lower left',ncol=2,numpoints=1,fontsize=16)
    
def plot_brace(args, xcut, func=None, xpos=None):
    """ internal function that plots vertical braces at xcut"""
    if len(xcut) > 1:
        if xpos == "low":
            try:
                _y = func(args[0,...], xcut[0])[0]
            except:
                _y = func(args[0,...], xcut[0])
            plt.hlines(0.9*_y, xcut[0]*1.02, xcut[0], colors="k", label="")
            plt.hlines(1.1*_y, xcut[0]*1.02, xcut[0], colors="k", label="")
            # TODO just a hotfix
            plt.vlines(xcut[0], 0.9*_y, 1.1*_y, colors="k", label="")
                
        
        elif xpos == "up":
            try:
                _y = np.asarray(func(args[0,...], xcut[1]))[0]
            except:
                _y = np.asarray(func(args[0,...], xcut[1]))
            plt.hlines(0.9*_y, xcut[1]*0.98, xcut[1], colors="k", label="")
            plt.hlines(1.1*_y, xcut[1]*0.98, xcut[1], colors="k", label="")
            plt.vlines(xcut[1], 0.9*_y, 1.1*_y, colors="k", label="")
        else:
            print("x position not known, not plotting anything")
    else:
        try:
            _y = func(args[0,...], xcut)[0]
        except:
            _y = func(args[0,...], xcut)
        plt.hlines(0.9*_y, xcut*0.98, xcut, colors="k", label="")
        plt.hlines(1.1*_y, xcut*0.98, xcut, colors="k", label="")
        plt.vlines(xcut, 0.9*_y, 1.1*_y, colors="k", label="")

        
def plot_single_line(x,y,label,col):
  """plot horizontal and vertical lines at the specific points, labeled with
  the specific values

  Parameters
  ----------
  x,y : in general multidimensional ndarrays 
  """
  #print("line properties")
  #print(x.shape)
  #print(y)
  x_val=np.zeros(2)
  y_val=np.zeros(2)
  try:
     x_val[0],x_val[1] = compute_error(x)
  except:
     x_val[0] = x
     x_val[1] = 0
  try:
     y_val[0],y_val[1] = compute_error(y)
  except:
     y_val[0] = y
     y_val[1] = 0
  #print(x_val)
  #print(y_val)
  #l = plt.axhline(y=y_val[0],ls='dashed',color=col)
  #l = plt.axvline(x=x_val[0],ls='dashed',color=col)
  # Public use
  plt.errorbar(x_val[0],y_val[0],y_val[1],x_val[1],fmt =
      'd',color=col,label=label)
  # Plot for internal use
  #plt.errorbar(x_val[0],y_val[0],y_val[1],x_val[1],fmt =
  #    'd',color=col,label=r'%2.4e,%2.4e' % (x_val[0],y_val[0]) )

def plot_lines(x,y,label,proc=None):
  """plot horizontal and vertical lines at the specific points, labeled with
  the specific values

  Parameters
  ----------
  x,y : in general multidimensional ndarrays 
  """
  # determine iterable data
  # if proc is match, x is iterable
  # if proc is eval, y is iterable
  print(x)
  print(y)
  if hasattr(x,"__iter__"):
    if hasattr(y,"__iter__"):
      plot_single_line(x[0],y[0],label,col='k')
      plot_single_line(x[1],y[1],label,col='r')
      plot_single_line(x[2],y[2],label,col='b')
    else:
      plot_single_line(x[0],y,label,col='k')
      plot_single_line(x[1],y,label,col='r')
      plot_single_line(x[2],y,label,col='b')

#def plot_gell_mann_okubo(self,dataframe,x-names,y-names,func):
#    col = ['r','b','g']
#    fmt_pts = ['^','v','o']
#    fmt_ls = ['--',':','-.']
#    dat_label = [r'$a=0.0885$fm',r'$a=0.0815$fm',r'$a=0.0619$fm']
#    for i,a in enumerate(dataframe.beta.unique()):
#        data=dataframe.where(dataframe.beta==a)
#        # Calculate x-data, x-err,y-err based on light quark mass value
#        data.groupby('m_ud').apply([own_mean,own_std])
#        x_data = data.M_pi.own_mean
#        x_err = data.M_pi.own_std
#        y_data = data.M_eta.own_mean
#        y_err = data.M_eta.own_std
#        plot_errorbar()
    
    

if __name__ == "__main__":
  pass
