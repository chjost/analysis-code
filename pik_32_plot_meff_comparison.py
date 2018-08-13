#!/usr/bin/python

# Plot the results of a chipt extrapolation
# 3 x 2 plots are generated:
# per fitrange
#   1) mu_piK_a32 vs. mu_piK/fpi with errors and LO ChPT curve
#   2) relative deviation between the data points for mu_piK_a32 and the fitted
# function per ensemble

import argparse
import matplotlib
matplotlib.use('pgf') # has to be imported before the next lines
import matplotlib.pyplot as plt
plt.style.use('paper_side_by_side')
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import sys

import analysis2 as ana
import chiron as chi

def set_axeslabels(x=None,y=None):
    plt.xlabel(x)
    plt.ylabel(y)

def main():
    # make global choices
    matplotlib.rcParams["errorbar.capsize"] = '2'
    matplotlib.rcParams["lines.markersize"] = '4'
    matplotlib.rcParams["lines.linewidth"] = '.75'
    #load data
    path = '/hiskp4/helmes/analysis/scattering/pi_k/I_32_blocked/data/m_eff_comparison/A30.32/'
    plotdir= '/hiskp4/helmes/analysis/scattering/pi_k/I_32_publish/plots/'
    dataname = path + 'm_eff_comparison.h5'
    keyname = 'comparison'
    plotname = plotdir + 'meff_cmp_A3032'
    data = pd.read_hdf(dataname,keyname)
    print(data)
    tmp = data.loc[data['poll']=='E1']
    fig=plt.figure()
    set_axeslabels(x=r'$t/a$',y=r'$am_{\mathrm{eff}\,,\pi K}$')
    plt.errorbar(tmp['t'],tmp['C(t)'],tmp['dC(t)'],label='E1',fmt ='^',
                 color='firebrick')
    tmp = data.loc[data['poll']=='E2']
    plt.errorbar(tmp['t'],tmp['C(t)'],tmp['dC(t)'],label='E2',fmt ='v',
                 color='darkblue')
    #choice for A30.32
    plt.xlim([9.5,23.5])
    # Tweak number of yticks
    ymin = 0.35
    ymax = 0.375
    step = 0.005
    plt.ylim([ymin,ymax])
    plt.yticks(np.arange(ymin,ymax,step))
    plt.xticks(np.arange(10,24,2))
    #choice for A40.24
    # Tweak number of yticks
    #ymin = 0.38
    #ymax = 0.405
    #step = 0.005
    #xmin = 8.5
    #xmax= 18.5
    #xstep=2
    #plt.xlim([xmin,xmax])
    #plt.ylim([ymin,ymax])
    #plt.yticks(np.arange(ymin,ymax,step))
    #plt.xticks(np.arange(xmin+0.5,xmax+0.5,xstep))
    plt.legend(frameon=False)
    plt.savefig(plotname+'.pgf')
    matplotlib.backends.backend_pgf.FigureCanvasPgf(fig).print_pdf(plotname+'.pdf',bbox='standard')
    plt.clf()
    

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
