
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def plot_data_with_fit(X, Y, dY, fitfunc, args, plotrange, label, pdfplot,
                       logscale=False, xlim=None, ylim=None, fitrange=None,
                       pval=None):
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
        pval: write the p-value in the plot if given

    Returns:
        Nothing.
    """
    # check boundaries for the plot
    if isinstance(plotrange, (np.ndarray, list, tuple)):
        plotrange = np.asarray(plotrange).flatten()
        if plotrange.size < 2:
            raise IndexError("plotrange has not enough indices")
        else:
            l = int(plotrange[0])
            u = int(plotrange[1])
    else:
        l = 0
        u = x.shape[0]
    p1 = plt.errorbar(X[l:u], Y[l:u], dY[l:u], fmt='x' + 'b', label = label[3])
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
    for i in x1:
        if len(args) > 1:
            # the star in front of the args is needed
            y1.append(fitfunc(args[0],i,*args[1:]))
        else:    
            y1.append(fitfunc(args,i))
    y1 = np.asarray(y1)
    p2, = plt.plot(x1, y1, "r", label = label[4])
    # adjusting the plot style
    plt.grid(True)
    plt.title(label[0])
    plt.xlabel(label[1])
    plt.ylabel(label[2])
    plt.legend()
    if pval is not None:
        # x and y position of the label
        x = np.max(X) * 0.7
        y = np.max(Y) * 0.8
        print(x,y)
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

def plot_data(X, Y, dY, pdfplot, plotrange=None, logscale=False, xlim=None, ylim=None):
    """A function that plots a correlation function.

    This function plots the given data points and the fit to the data. The plot
    is saved to pdfplot. It is assumed that pdfplot is a pdf backend to
    matplotlib so that multiple plots can be saved to the object.

    Args:
        X: The data for the x axis.
        Y: The data for the y axis.
        dY: The error on the y axis data.
        pdfplot: A PdfPages object in which to save the plot.
        plotrange: A list with two entries, the lower and upper range of the
                   plot.
        logscale: Make the y-scale a logscale.
        xlim: tuple of the limits on the x axis
        ylim: tuple of the limits on the y axis

    Returns:
        Nothing.
    """
    # check boundaries for the plot
    if isinstance(plotrange, (np.ndarray, list, tuple)):
        plotrange = np.asarray(plotrange).flatten()
        if plotrange.size < 2:
            raise indexerror("plotrange is too small")
        else:
            l = int(plotrange[0])
            u = int(plotrange[1])
    else:
        l = 0
        u = x.shape[0]

    # plot the data
    p1 = plt.errorbar(X[l:u], Y[l:u], dY[l:u], fmt='x' + 'b', label="data")

    # adjusting the plot style
    plt.grid(True)
    #plt.xlabel(label[1])
    #plt.ylabel(label[2])
    #plt.title(label[0])
    plt.legend()
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
