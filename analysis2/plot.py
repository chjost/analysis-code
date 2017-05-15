"""
The class for fitting.
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
from plot_functions import plot_data, plot_function, plot_histogram
from in_out import check_write

class LatticePlot(object):
    def __init__(self, filename, join=False):
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
        if join:
            self.cycol = itertools.cycle('bgrcmk').next
            self.cyfmt = itertools.cycle('^vsd>o').next
        else:
            self.cycol = itertools.cycle('b').next
            self.cyfmt = itertools.cycle('x').next

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
        for n in range(ncorr):
            # Check if there are ncorr datalabels
            if len(label[3]) == ncorr:
                _datlabel = label[3][n]
            else:
                _datlabel = label[3]
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
                    plot_data(X, np.divide(ddata,corr.data[0,:,n]),
                        np.zeros_like(ddata), label=_datlabel,
                        plotrange=[0,T],col=self.cycol(),fmt=self.cyfmt())
                else:
                    plot_data(X, corr.data[0,:,n], ddata, _datlabel,
                        plotrange=[0,T],col=self.cycol(),fmt=self.cyfmt())
                plt.legend()
                if self.join is False:
                  self.save()
            else:
                # iterate over fit intervals
                for r in range(shape[0][n]):
                    if debug > 1:
                        print("plotting fit ranges %s" % str(r))
                    fi = ranges[n][r]
                    mpar, dpar = compute_error(fitresult.data[n][:,:,r])
                    print(mpar,dpar)

                    # set up labels
                    label[0] = "%s, pc %d" % (label_save, n)
                    self.set_title(label[0], label[1:3])
                    label[4] = "fit [%d, %d]" % (fi[0], fi[1])

                    # plot
                    self._set_env_normal()
                    plot_data(X, corr.data[0,:,n], ddata, _datlabel,
                            plotrange=[1,T],col=self.cycol())
                    # The argument X has changed
                    #_X = [X[0],X[-1]]
                    plot_function(fitfunc.fitfunc, fi, mpar, label[4],
                            add, fi, ploterror,col=self.cycol())
                    plt.legend()
                    if self.join is False:
                      self.save()
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
            _bins = fitresult.fit_ranges_shape[-1][0] / 2
        else:
            _bins = nb_bins
        if fitresult.derived:
            w = fitresult.weight[0]
            for i, d in enumerate(fitresult.data):
                label[0] = " ".join((label_save, str(fitresult.label[i])))
                plot_histogram(d[0], w[i], label, nb_bins=nb_bins)
                plt.legend()
                self.save()
        else:
            if par is None:
                for p, w in enumerate(fitresult.weight):
                    for i, d in enumerate(fitresult.data):
                        label[0] = " ".join((label_save, str(fitresult.label[i])))
                        plot_histogram(d[0,p], w[i], label, nb_bins=nb_bins)
                        plt.legend()
                        self.save()
            else:
                w = fitresult.weight[par]
                for i, d in enumerate(fitresult.data):
                    label[0] = " ".join((label_save, str(fitresult.label[i])))
                    plot_histogram(d[0,par], w[i], label, nb_bins=nb_bins)
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
          if data.conf is not None:
              _x = np.unique(data.conf)
          else:
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
        plt.xlabel(label[1])
        plt.ylabel(label[1])
        plt.xticks(np.arange(3),(0.0185,0.0225,0.02464))
        plt.yticks(np.arange(3),(0.0185,0.0225,0.02464))
        self.set_env(xlim=[0,cov.shape[0]],ylim = [0,cov.shape[0]])
        plt.pcolor(cov, cmap=matplotlib.cm.bwr, vmin=np.amin(cov),
            vmax=np.amax(cov))
        plt.colorbar()
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
        

    def plot_chiral_fit(self,chirana,beta,label,xlim,func=None,args=None):
        """ Function to plot a chiral fit.
        
        This function sets up a plotter object, puts in the data in the right
        shape and plots the data itself as well as the function
        
        """
        # Plot the data for the given lattice spacings
        # Initialize symbols and colors for lattice spacings
        col = ['r','b','g']
        fmt_pts = ['^','v','o']

        for i,a in enumerate(beta):
            # get data for beta, the data passed should be 3 arrays (X,Y,dy)
            # the light quark mass values
            _X = chirana.x_data[i][:,:,0,0].flatten()
            _Y = chirana.y_data[i][:,:,0,0].flatten()
            _dy = chirana.y_data[i][:,:,0,:].reshape((chirana.y_data[i].shape[0]*chirana.y_data[i].shape[1],chirana.y_data[i].shape[-1]))
            print("yerror shape is:")
            print(_dy.shape)
            _mean, _dy = compute_error(_dy,axis=1)
            print(_dy.shape)
            plot_data(_X,_Y,_dy,label=a,col=col[i],fmt=fmt_pts[i])
            # Check if we want to plot a function in addition to the data
            if func is not None:
                print("Arguments for plotting function")
                print(args[i])
                for s in chirana.x_data[i][0,:,1,0]: 
                  # Check for numbers of lattice spacings, if > 1 loop over args
                  #if len(beta)==1:
                  #    plotargs = np.hstack((args,s,r))
                  #else:
                  # adapt shape for errorbands in plot_function
                  _mus = np.full((args.shape[1],1),s) 
                  plotargs = np.hstack((args[i],_mus))
                  print("Arguments to plotting function")
                  print(plotargs)
                  plot_function(func,xlim,plotargs,label=None,ploterror=True)
            plt.xlim(xlim[0],xlim[1])
            plt.locator_params(nbins=4)
            plt.xlabel(label[0],fontsize=24)
            plt.ylabel(label[1],fontsize=24)
            #self.save()
        plt.axvline(x=0.83,color='k',linestyle='--',label=label[-1])
        plt.xlim(xlim[0],xlim[1])
        plt.locator_params(nbins=4)
        plt.xlabel(label[0],fontsize=24)
        plt.ylabel(label[1],fontsize=24)
        if len(label) > 2:
            plt.title(label[2])
        plt.legend(loc='best',ncol=1,fontsize=14)
        if self.join is False:
          self.save()
          plt.clf()

    def plot_chiral_ext(self,chirana,beta,label,xlim,ylim=None,func=None,
                        args=None,ploterror=True):
        """ Function to plot a chiral extrapolation fit.
        
        This function sets up a plotter object, puts in the data in the right
        shape and plots the data itself as well as the function
        
        """
        # Plot the data for the given lattice spacings
        # Initialize symbols and colors for lattice spacings
        col = ['r','b','g']
        fmt_pts = ['^','v','o']
        fmt_ls = ['--',':','-.']
        dat_label = [r'$a=0.0885$fm',r'$a=0.0815$fm',r'$a=0.0619$fm']

        for i,a in enumerate(beta):
            # get data for beta, the data passed should be 3 arrays (X,Y,dy)
            # the light quark mass values
            print("argument shape:")
            print(args[i].shape)
            _X = chirana.x_data[i][:,:,0,0].flatten()*args[i,0,0]/args[i,0,1]
            print("x-data:")
            print(_X)
            _Y = chirana.y_data[i][:,:,0,0].flatten()
            _dy = chirana.y_data[i][:,:,0,:].reshape((chirana.y_data[i].shape[0]*chirana.y_data[i].shape[1],chirana.y_data[i].shape[-1]))
            print("yerror shape is:")
            print(_dy.shape)
            _mean, _dy = compute_error(_dy,axis=1)
            print(_dy.shape)
            plot_data(_X,_Y,_dy,label=a,col=col[i],fmt=fmt_pts[i])
            # Check if we want to plot a function in addition to the data
            if func is not None:
                print("Arguments for plotting function")
                print(args[i])
                plot_function(func,xlim,args[i],label=dat_label[i],
                              ploterror=ploterror,fmt=col[i]+fmt_ls[i],col=col[i])
            plt.xlabel(label[0],fontsize=24)
            plt.ylabel(label[1],fontsize=24)
            #self.save()
        # Plot the physical point as well as the continuum function
        plt.xlim(xlim[0],xlim[1])
        if ylim is not None:
          plt.ylim(ylim[0],ylim[1])
        plt.xlabel(label[0])
        plt.ylabel(label[1])
        if len(label) > 2:
            plt.title(label[2])
        plt.legend(loc='best',ncol=2,numpoints=1,fontsize=16)
        #if self.join is False:
        #  self.save()
        #  plt.clf()

    def plot_cont(self,chirana,func,phys_x,xlim,args):
      """ Plot the continuum curve of a chiral analysis and the physical point
      result
      """
      # Plot the continuum curve
      plot_function(func, xlim, args, 'cont.', fmt='k--', ploterror=True)
      plt.errorbar(chirana.phys_point[0,0],chirana.phys_point[1,0],
                   chirana.phys_point[1,1],xerr=chirana.phys_point[0,1],
                   fmt='d', color='darkorange', label='phys.')
      plt.legend(loc='best',ncol=2,numpoints=1,fontsize=16)
    
        
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


if __name__ == "__main__":
  pass
