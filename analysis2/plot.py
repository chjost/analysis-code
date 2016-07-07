"""
The class for fitting.
"""

import numpy as np
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
matplotlib.rcParams['axes.labelsize']='large'
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
        self.grid=True
        self.legend=True
        self.join=join
        if join:
            self.cycol = itertools.cycle('bgrcmk').next
        else:
            self.cycol = itertools.cycle('b').next

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
            add=None, xshift=0., ploterror=False, rel=False, debug=0, join=False):
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
                    plot_data(X, np.d(ddata,corr.data[0,:,n]),
                        np.zeros_like(ddata), label=label[3],
                        plotrange=[0,T],col=self.cycol())
                else:
                    plot_data(X, corr.data[0,:,n], ddata, label[3],
                        plotrange=[0,T],col=self.cycol())
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
                    plot_data(X, corr.data[0,:,n], ddata, label[3],
                            plotrange=[1,T],col=self.cycol())
                    plot_function(fitfunc.fitfunc, X, mpar, label[4],
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
                plot_function(fitfunc.fitfunc, X, _par, label[4], 
                        add_data, fi, ploterror)
                plt.legend()
                self.save()

    def plot(self, corr, label, fitresult=None, fitfunc=None, oldfit=None,
            add=None, oldfitpar=None, ploterror=False, xshift=0., debug=0):
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
                    ploterror=ploterror, xshift=xshift, debug=debug)
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
    
    def correlogram(self, correlator, label, num=0):
        """Plot the autocorrelation versus lagtime for data in a correlator
        object

        Implemented version is the one with more bias: Have T timesteps and B
        measurements at eacht time t of observable Y
        
        r(k) = \sum_t=1^{T}

        Parameters
        ----------
        correlator : a Correlator object
        label : the label of the correlogram
        num : which entry of the correlator object to take
        """

        correlogram_raw = acf(correlator.data)
        _t = np.linspace(0,correlator.data.shape[1],correlator.data.shape[1])
        _mean,_std = compute_error(correlogram_raw,axis=0)
        print("Data shapes in correlogram:")
        print(_mean.shape)
        print(_std.shape)
        print(_t.shape)
        #plot_data(_t,_mean,_std,None,
        #    fmt='o',markerfill ='none')
        plt.stem(_t,_mean)
        plt.title(label[0])
        plt.xlabel(r'Lag $k$ in $t/a$')
        plt.ylabel(r'acf($k$)')
        self.save()
        plt.clf()



    def qq_plot(self, fitresult, label, par=0, corr=0):
        """A quantile-quantile-plot for Fitresults

        Calculate the theoretical quantiles (gaussian with mean and standard
        deviation from fitresult), and the measured ones, then plot them against
        each other together with a straight line to visualize the deviation

        Parameters
        ----------
        fitresult : FitResult object
        label : str title of plot
        par : int, which parameter of fit result should be taken
        """

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
            fmt='o',markerfill ='none')
        plot_function(bisec,np.sort(q_theo),None,'',fmt='r')
        #plt.legend(loc='best')
        plt.locator_params(axis='x',nbins=4)
        plt.locator_params(axis='y',nbins=4)
        plt.title(label[0])
        plt.xlabel(r'theoretical quantiles $N_{\bar{\mu},\Delta\mu}$')
        plt.ylabel(r'measured quantiles')
        plt.axes().set_aspect('equal')
        self.save()
        plt.clf()


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
  l = plt.axhline(y=y_val[0],ls='solid',color=col)
  l = plt.axvline(x=x_val[0],ls='solid',color=col)
  plt.errorbar(x_val[0],y_val[0],y_val[1],x_val[1],fmt =
      'd',color=col,label=r'%2.4e,%2.4e' % (x_val[0],y_val[0]) )

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
