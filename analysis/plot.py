
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def corr_fct_with_fit(X, Y, dY, fitfunc, args, plotrange, label, pdfplot,
                      logscale=False, setLimits=False, fitrange=None):
    """A function that plots a correlation function.

    This function plots the given data points and the fit to the data. The plot
    is saved to pdfplot. It is assumed that pdfplot is a pdf backend to
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
        setLimits: Set limits to the y range of the plot.
        fitrange: A list with two entries, bounds of the fitted function.

    Returns:
        Nothing.
    """
    # plotting the data
    l = int(plotrange[0])
    u = int(plotrange[1])
    p1 = plt.errorbar(X[l:u], Y[l:u], dY[l:u], fmt='x' + 'b', label = label[3])
    # plotting the fit function, check for seperate range
    if fitrange:
        lfunc = fitrange[0]
        ufunc = fitrange[1]
    else:
        lfunc = l
        ufunc = u
    x1 = np.linspace(lfunc, ufunc, 1000)
    y1 = []
    for i in x1:
        if len(args) is 3:
            y1.append(fitfunc(args[:-1],i,args[2]))
        else:
            y1.append(fitfunc(args,i))
    y1 = np.asarray(y1)
    p2, = plt.plot(x1, y1, 'r', label = label[4])
    # adjusting the plot style
    plt.grid(True)
    plt.xlabel(label[1])
    plt.ylabel(label[2])
    plt.title(label[0])
    plt.legend()
    if logscale:
        plt.yscale('log')
    # set the yaxis range
    if setLimits:
        plt.ylim(0.25, 1.)
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

def plot_histogram(data, data_weight, lattice, d, label, path=".plots/", 
                   plotlabel="hist", verbose=True):
    """plot a weighted histogramm

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

    histplot = PdfPages("%s/fit_%s_%s_TP%d.pdf" % (path,plotlabel,lattice,d2))
    # The histogram

    hist, bins = np.histogram(data, 20, weights=data_weight, density=True)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2

    plt.ylabel('weighted distribution of ' + label[2])
    plt.title('fit methods individually with a p-value between 0.01 and 0.99')
    plt.grid(True)
    x = np.linspace(center[0], center[-1], 1000)

#    plt.plot(x, scipy.stats.norm.pdf(x, loc=a_pipi_median_derv[0], \
#             scale=a_pipi_std_derv), 'r-', lw=3, alpha=1, \
#             label='median + stat. error')
    plt.bar(center, hist, align='center', width=width, color='r', alpha=0.5,
            label='derivative')

    histplot.savefig()
    histplot.close()
    
    return
