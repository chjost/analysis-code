
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import analyze_fcts as af

def plot_data_with_fit(X, Y, dY, fitfunc, args, plotrange, label, pdfplot,
                       logscale=False, xlim=None, ylim=None, fitrange=None,
                       addpars=False, pval=None, hconst=None, vconst=None):
    """A function that plots data and the fit to the data.

    The plot is saved to pdfplot. It is assumed that pdfplot is a pdf backend to
    matplotlib so that multiple plots can be saved to the object.

    Args:
        X: The data for the x axis.
        Y: The data for the y axis.
        dY: The error on the y axis data.
        fitfunc: The function to fit to the data.
        args: The parameters of the fit function from the fit.
        plotrange: A list with two entries, the lower and upper range of the
                   plot.
        label: A list with labels for title, x axis, y axis, data and fit.
        pdfplot: A PdfPages object in which to save the plot.
        logscale: Make the y-scale a logscale.
        xlim, ylim: limits for the x and y axis, respectively
        setLimits: Set limits to the y range of the plot.
        fitrange: A list with two entries, bounds of the fitted function.
        addpars: if there are additional parameters for the fitfunction 
                 contained in args, set to true
        pval: write the p-value in the plot if given

    Returns:
        Nothing.
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
        p1 = plt.errorbar(X[l:u], Y[l:u], dY[l:u], fmt='x',color='#dc322f', label = label[3])
    else:
        # plot the data
        p1 = plt.errorbar(X, Y, dY, fmt='x',color='#dc322f', label = label[3])

    # plotting the fit function, check for seperate range
    if isinstance(fitrange, (np.ndarray, list, tuple)):
        fitrange = np.asarray(fitrange).flatten()
        if fitrange.size < 2:
            raise IndexError("fitrange has not enough indices")
        else:
            lfunc = int(fitrange[0])
            ufunc = int(fitrange[1])
    else:
        lfunc = X[0]
        ufunc = X[-1]
    x1 = np.linspace(lfunc, ufunc, 1000)
    y1 = []
    if addpars:
        for i in x1:
            # the star in front of the args is needed
            y1.append(fitfunc(args[0],i,*args[1:]))
    else:    
        for i in x1:
            y1.append(fitfunc(args,i))
    y1 = np.asarray(y1)
    p2, = plt.plot(x1, y1, color='#2aa198',alpha=0.75, label = label[4])
    # Plotting an additional constant
    if isinstance(hconst, (np.ndarray,list,tuple)):
        plt.axhline(hconst[0],color='#b58900')
        plt.text(X[0],hconst[0]+X[0]/100.,label[5])
        plt.axhspan(hconst[0]+hconst[1],hconst[0]-hconst[1],alpha=0.35,color='gray')
    if isinstance(vconst, (np.ndarray,list,tuple)):
        plt.axvline(vconst[0],color='#859900')
        plt.text(vconst[0],Y[0],label[6])
        plt.axvspan(vconst[0]+vconst[1],vconst[0]-vconst[1],alpha=0.35,color='gray')
    # adjusting the plot style
    plt.grid(True)
    plt.title(label[0])
    plt.xlabel(label[1])
    plt.ylabel(label[2])
    plt.legend(loc='best',framealpha=0.75)
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
    # save pdf
    pdfplot.savefig()
    plt.clf()

# this can be used to plot the chisquare distribution of the fits
#  x = np.linspace(scipy.stats.chi2.ppf(1e-6, dof), scipy.stats.chi2.ppf(1.-1e-6, dof), 1000)
#  hist, bins = np.histogram(chisquare, 50, density=True)
#  width = 0.7 * (bins[1] - bins[0])
#  center = (bins[:-1] + bins[1:]) / 2
#  plt.xlabel('x')
#  plt.ylabel('chi^2(x)')
#  plt.grid(True)
#  plt.plot(x, scipy.stats.chi2.pdf(x, dof), 'r-', lw=2, alpha=1, label='chi2 pdf')
#  plt.bar(center, hist, align='center', width=width)
#  plt.show()

def plot_data(X, _Y, dY, pdfplot, label, plotrange=None, logscale=False, xlim=None, ylim=None):
    """A function that plots a correlation function.

    This function plots the given data points and the fit to the data. The plot
    is saved to pdfplot. It is assumed that pdfplot is a pdf backend to
    matplotlib so that multiple plots can be saved to the object.

    Args:
        X: The data for the x axis.
        Y: The data for the y axis.
        dY: The error on the y axis data.
        pdfplot: A PdfPages object in which to save the plot.
        label: label for the plot
        plotrange: A list with two entries, the lower and upper range of the
                   plot.
        logscale: Make the y-scale a logscale.
        xlim: tuple of the limits on the x axis
        ylim: tuple of the limits on the y axis

    Returns:
        Nothing.
    """
    Y=np.atleast_2d(_Y)
    # check boundaries for the plot
    if isinstance(plotrange, (np.ndarray, list, tuple)):
        plotrange = np.asarray(plotrange).flatten()
        if plotrange.size < 2:
            raise IndexError("plotrange is too small")
        else:
            l = int(plotrange[0])
            u = int(plotrange[1])
        # plot the data
        print l,u
        p1 = plt.errorbar(X[l:u], Y[0,l:u], dY[l:u], marker='x', color='teal',linestyle='', label=label[3])
    else:
        # plot the data
        p1 = plt.errorbar(X, Y, dY, marker='x', color='teal', linestyle='',  label=label[3])

    # adjusting the plot style
    plt.grid(True)
    plt.xlabel(label[1])
    plt.ylabel(label[2])
    plt.title(label[0])
    plt.legend(loc='best')
    if logscale:
        plt.yscale('log')
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)

    # save pdf and clear plot
    pdfplot.savefig()
    plt.clf()

    return

def plot_histogram(data, data_weight, lattice, d, label, path="./plots/", 
                   plotlabel="hist", verbose=True):
    """Plots histograms for the given data set.

    The function plots the weighted distribution of the data, the unweighted
    distribution and a plot containing both the weighted and the unweighted
    distribution.

    Args:
        data: Numpy-array of fit values for mulitple fit intervalls. Will be 
              depicted on x-axis.
        data_weight: The weights corresponding to data. Must have same shape
              and order as data. Their sum per bin is the bin height.
        lattice: The name of the lattice, used for the output file.
        d:    The total momentum of the reaction.
        label: Labels for the title and the axis.
        path: Path to the saving place of the plot.
        plotlabel: Label for the plot file.
        verbose: Amount of information printed to screen.

    Returns:
    """
    d2 = np.dot(d,d)
    ninter = data.shape[0]

    histplot = PdfPages("%s/%s_%s_TP%d.pdf" % (path,plotlabel,lattice,d2))

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
    plt.ylabel('weighted distribution of ' + label[2])
    plt.title('fit methods individually')
    plt.grid(True)
    # plot
    plt.bar(center, hist, align='center', width=width, color='r', alpha=0.5,
            label='weighted data')
    # save and clear
    histplot.savefig()
    plt.clf()

    # prepare plot for unweighted histogram
    # the center and width stays the same for comparison
    plt.ylabel('unweighted distribution of ' + label[2])
    plt.title('fit methods individually')
    plt.grid(True)
    # plot
    plt.bar(center, uhist, align='center', width=width, color='b', alpha=0.5,
            label='unweighted data')

    # save and clear
    histplot.savefig()
    plt.clf()

    # plot both histograms in same plot
    plt.ylabel('distribution of ' + label[2])
    plt.title('fit methods individually')
    plt.grid(True)
    # plot
    plt.bar(center, hist, align='center', width=width, color='r', alpha=0.5,
            label='weighted data')
    plt.bar(center, uhist, align='center', width=width, color='b', alpha=0.5,
            label='unweighted data')
    plt.legend()

    # save and clear
    histplot.savefig()
    plt.clf()

    # close plotfile
    histplot.close()
    return

def genplot(_data, par, pvals, fit_intervals, fitfunc, tmin, lattice, d, label,
            path="./plots/", plotlabel="corr", verbose=False):
    """Plots data with fit.
    
    Args:
        _data: The correlation functions.
        par: The fit results of the fits
        pvals: The p-values of the fits
        fit_intervalls: List of intervalls for the fit for the different
              correlation functions.
        fitfunc: The function to fit to the data.
        tmin: Lower bound of the plot.
        lattice: The name of the lattice, used for the output file.
        d: The total momentum of the reaction.
        label: Labels for the title and the axis.
        path: Path to the saving place of the plot.
        plotlabel: Label for the plot file.
        verbose: Amount of information printed to screen.
    """
    data = np.atleast_3d(_data)
    # init variables
    nboot = data.shape[0]
    T2 = data.shape[1]
    ncorr = data.shape[2]
    d2 = np.dot(d,d)
    ninter = [len(fitint) for fitint in fit_intervals]
    # set fit data
    tlist = np.linspace(0., float(T2), float(T2), endpoint=False)
    # outputfile for the plot
    corrplot = PdfPages("%s/fit_check_%s_%s_TP%d.pdf" % (path,plotlabel,lattice,d2))
    # check the labels
    if len(label) < 3:
        print("not enough labels, using standard labels.")
        label = ["fit", "time", "C(t)", "", ""]
    if len(label) < 4:
        label.append("data")
        label.append("")
    if len(label) < 5:
        label.append("")
    label_save = label[0]
    for _l in range(ncorr):
        # setup
        mdata, ddata = af.calc_error(data[:,:,_l])
        for _i in range(ninter[_l]):
            fi = fit_intervals[_l][_i]

            mpar, dpar = af.calc_error(par[_l][:,:,_i])

            # set up the plot labels
            fitlabel = "fit %d:%d" % tuple(fi)
            title="%s, %s, TP %d, pc %d, [%d, %d]" % (label_save, lattice, d2, 
                                                      _l, fi[0], fi[1])
            label[0] = title
            label[4] = fitlabel

            # plot the data and the fit
            plot_data_with_fit(tlist, data[0,:,_l], ddata, fitfunc, mpar,
                               [tmin,T2], label, corrplot, logscale=True,
                               fitrange=fi, pval=pvals[_l][0,_i])
    corrplot.close()

def genplot_comb(_data, pvals, fitint_data, fitint_par, fitfunc, par_data, 
                 par_par, tmin, lattice, label, par_par_index=0, 
                 path="./plots/", plotlabel="corr", verbose=False):
    """Fit and plot a function. With varying parameter, determined in a previous
    fit
    
    Args:
        data: The correlation functions.
        pvals: pvalues of the fit
        fitint_data: List of intervals for the fit of the functions.
        fitint_par: List of intervals for the varying parameter
        fitfunc: The function to fit to the data.
        par_data: The fitted parameters
        par_par: the varying parameters
        tmin: Lower bound of the plot.
        lattice: The name of the lattice, used for the output file.
        label: Labels for the title and the axis.
        par_par_index: the index for the parameter to pass to the fit function
        path: Path to the saving place of the plot.
        plotlabel: Label for the plot file.
        verbose: Amount of information printed to screen.
        """
    # ensure at least 3d 
    data = np.atleast_3d(_data)
    # init variables
    # nboot: number of bootstrap samples
    # npar: number of parameters to fit to
    # ncorr: number of correlators
    # ncorr_par: number of correlators of the varying parameter
    # nint_data: number of fit ranges for data
    # nint_par: number of fit ranges of the varying parameter
    nboot = data.shape[0]
    T2 = data.shape[1]
    ncorr = data.shape[2]
    ncorr_par = len(par_par)
    nint_data = [len(fitint) for fitint in fitint_data]
    nint_par = [len(fitint) for fitint in fitint_par]
    # set fit x data for fit
    tlist = np.linspace(0., float(T2), float(T2), endpoint=False)
    # outputfile for the plot and the overview plot
    corroverview = PdfPages("%s/fit_check_overview_%s_%s.pdf" % (path,plotlabel,lattice))
    corrplot = PdfPages("%s/fit_check_%s_%s.pdf" % (path,plotlabel,lattice))
    # check the labels
    if len(label) < 3:
        print("not enough labels, using standard labels.")
        label = ["fit", "time", "C(t)", "", ""]
    if len(label) < 4:
        label.append("data")
    if len(label) < 5:
        label.append("")
    label_save = label[0]
    # loop over the correlation functions of the data
    for l in range(ncorr):
        # setup
        mdata, ddata = af.calc_error(data[:,:,l])
        # loop over the fit intervals
        for i in range(nint_data[l]):
            fi = fitint_data[l][i]
            # loop over the varying parameter and its fit intervals
            for k in range(ncorr_par):
                # generate overview plot by averaging over "j"
                mpar_data, dpar_data = af.calc_error(par_data[k][l][0,:,i], axis=1)
                mpar_par, dpar_par = af.calc_error(par_par[k][0], axis=1)
                # set up the plot labels
                fitlabel = "fit %d:%d" % (fi[0], fi[1])
                title = "%s, %s, pc %d, [%d, %d]" % (label_save, lattice, l,
                                                     fi[0], fi[1])
                label[0] = title
                label[4] = fitlabel
                # plot an overview
                plot_data_with_fit(tlist, data[0,:,l], ddata, fitfunc,
                    (mpar_data, mpar_par[par_par_index]), [tmin,T2], label,
                    corroverview, logscale=False, fitrange=fi, addpars=True)

                for j in range(nint_par[k]):
                    mpar_data, dpar_data = af.calc_error(par_data[k][l][:,:,i,j])
                    # just for shorter writing
                    tmp_par = par_par[k][0,par_par_index,j]

                    # set up the plot labels
                    fitlabel = "fit %d:%d\nm$_{\pi}$ = %f" % (fi[0], fi[1], tmp_par)
                    label[4] = fitlabel

                    # plot the original data and the fit for every fit range
                    plot_data_with_fit(tlist, data[0,:,l], ddata, fitfunc,
                        (mpar_data, tmp_par), [tmin,T2], label, corrplot,
                        logscale=False, fitrange=fi, addpars=True, 
                        pval=pvals[k][l][0,i,j])
    corrplot.close()
    corroverview.close()
