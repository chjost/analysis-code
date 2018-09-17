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
plt.style.use('paper_standalone')
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import sys
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import analysis2 as ana
import chiron as chi

def plot_comparison(data,shift=0.,fmt='ok'):
    """Horizontal comparison plot for few points"""
    # format array for branch ids
    fmts=['^','^','v','v']
    cols=['darkblue','red','darkblue','red']
    cnt=0
    xpos = np.array([])
    labels = []
    for f,g in enumerate(data['branch_id'].unique()):
        group = data.loc[data['branch_id']==g]
        y = group['mua0_mean']
        yerr = group['mua0_std']
        x = cnt+group['fr_ind'].astype(int).values
        xpos=np.append(xpos,x)
        plt.errorbar(x,y,yerr,label=g,fmt = fmts[f],color=cols[f])
        cnt+=len(x)
        labels+=(list(group['fr_ind'].values))
    plt.xticks(xpos,labels)

def idstring(x):
    if x =='gamma':
        return r'$\Gamma$'
    else :
        return r'NLO ChPT'
def main():
    #load data
    pd.set_option("display.width",300)
    path = '/hiskp4/helmes/analysis/scattering/pi_k/I_32_cov_false/results/'
    plotdir= '/hiskp4/helmes/analysis/scattering/pi_k/I_32_cov_false/plots/'
    filename = path+'/branch_comparison.txt'
    outname = plotdir+'/branch_comparison'
    datapoints = pd.read_csv(filename, sep='\s+')
    # need discriminator variables
    datapoints['branch_id']=datapoints['ChPT'].map(idstring)+', '+datapoints['poll']
    datapoints=datapoints.fillna(0)
    datapoints['fr_ind'] = pd.Series(np.array((1,2,3,1,2,3,3,2,1,3,2,1))).astype(str)
    datapoints['fit_mag'] = (datapoints['fr_end']-datapoints['fr_bgn']).abs()
    datapoints=datapoints.sort_values('fit_mag',ascending=True)
    print(datapoints)
    fig=plt.figure()
    plt.xlabel('fit range index')
    plt.ylabel(r'$\mu_{\pi K}\,a_0^{3/2}$')
    xmin=-1
    xmax=15
    v = -4.632e-2
    sd = 8.16e-04  
    sup = np.sqrt(4.7e-4**2+7.9e-5**2+1.24e-3**2)
    sdn = sup
    plt.fill_between(np.array((xmin,xmax)),v+np.sqrt(sd**2+sup**2),v-np.sqrt(sd**2+sdn**2),alpha=0.2,color='deepskyblue')
    plt.fill_between(np.array((xmin,xmax)),v+sd,v-sd,alpha=0.2,color='darkblue')
    plt.axhline(v,color='k')
    plot_comparison(datapoints)
    plt.xlim([0.5,12.5])
    plt.legend(loc='lower center',bbox_to_anchor=(0.5,1.05),
                ncol=4,borderaxespad=0.)
    plt.savefig(outname+'.pgf')
    matplotlib.backends.backend_pgf.FigureCanvasPgf(fig).print_pdf(outname+'.pdf')
    plt.clf()
     

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt")

