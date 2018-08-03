#!/usr/bin/python

# Plot the results of a chipt extrapolation
# 3 x 2 plots are generated:
# per fitrange
#   1) mu_piK_a32 vs. mu_piK/fpi with errors and LO ChPT curve
#   2) relative deviation between the data points for mu_piK_a32 and the fitted
# function per ensemble

import argparse
import itertools as it
import matplotlib
matplotlib.use('pgf') # has to be imported before the next lines
import matplotlib.pyplot as plt
plt.style.use('paper_side_by_side')
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import sys
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import analysis2 as ana
import chiron as chi

def plot_comparison(data,shift=0.,fmt='ok'):
    """Vertical comparison plot for few points"""
    y = data['Collab']
    x = data['mpia0_32']
    xerr = data['dmpia0_32']
    xsysdn = data['sys_dn(mpia0_32)']
    xsysup = data['sys_up(mpia0_32)']
    plt.yticks(np.arange(0,y.shape[0]),y,
               verticalalignment='center')
    x_err = np.asarray([np.sqrt(xerr**2+xsysdn**2),np.sqrt(xerr**2+xsysup**2)])
    plt.errorbar(x,np.arange(len(y)),xerr=x_err,
                 fmt = '.', color='deepskyblue')
    plt.errorbar(x,np.arange(len(y)),xerr=xerr,fmt='o',color='darkblue',
            label=None)
def main():
    #load data
    path = '/hiskp4/helmes/analysis/scattering/pi_k/I_32_publish/results/'
    plotdir= '/hiskp4/helmes/analysis/scattering/pi_k/I_32_publish/plots/'
    filename = path+'/final_comparison.txt'
    outname = plotdir+'/final_comparison'
    datapoints = pd.read_csv(filename, sep='\s+')
    datapoints=datapoints.fillna(0)
    datapoints = datapoints[datapoints['Collab']!='ETMC_I']
    fig=plt.figure()
    plot_comparison(datapoints)
    plt.xlabel(r'$(M_{\pi}a_0)^{\mathrm{phys}}$')
    plt.xlim([-0.075,-0.045])
    plt.legend()
    plt.savefig(outname+'.pgf')
    matplotlib.backends.backend_pgf.FigureCanvasPgf(fig).print_pdf(outname+'.pdf',bbox='standard')
    plt.clf()
     

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt")

