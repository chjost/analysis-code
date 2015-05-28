
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def corr_fct_with_fit(X, Y, dY, fitfunc, args, plotrange, label, pdfplot,
                      logscale=False, setLimits=False):
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

    Returns:
        Nothing.
    """
    # plotting the data
    l = int(plotrange[0])
    u = int(plotrange[1])
    p1 = plt.errorbar(X[l:u], Y[l:u], dY[l:u], fmt='x' + 'b', label = label[3])
    # plotting the fit function
    x1 = np.linspace(l, u, 1000)
    y1 = []
    for i in x1:
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
