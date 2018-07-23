#!/usr/bin/python

################################################################################
#
# Author: Christopher Helmes (helmes@hiskp.uni-bonn.de)
# Date:   July 2018
#
# Copyright (C) 2018 Christopher Helmes
# 
# This program is free software: you can redistribute it and/or modify it under 
# the terms of the GNU General Public License as published by the Free Software 
# Foundation, either version 3 of the License, or (at your option) any later 
# version.
# 
# This program is distributed in the hope that it will be useful, but WITHOUT 
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS 
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with tmLQCD. If not, see <http://www.gnu.org/licenses/>.
################################################################################
# Plot the results of a chipt extrapolation
# 3 x 2 plots are generated:
# per fitrange
#   1) mu_piK_a32 vs. mu_piK/fpi with errors and LO ChPT curve
#   2) relative deviation between the data points for mu_piK_a32 and the fitted
# function per ensemble
################################################################################

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
import chiron as chi

def main():
    #load data
    path = '/hiskp4/helmes/analysis/scattering/pi_k/I_32_blocked/results/'
    plotdir= '/hiskp4/helmes/analysis/scattering/pi_k/I_32_blocked/plots/'
    filename = path+'/summarycurves.h5'
    key = 'summarycurves'
    storer=pd.HDFStore(filename)
    summary = storer.get(key)
    # check whether read in is correct
    print(summary.head(n=20))
    # filter data by method
    # data for datapoints
    filename= path + '/pi_K_I32_overview.h5'
    zp = ['M1','M2']
    poll = ['E1','E3']
    storer=pd.HDFStore(filename)
    datapoints = storer.get('fse_true/data_collection')
    datapoints.info()
    print(datapoints.sample(n=10))
    cols = ('RC','beta')
    for col in cols:
        datapoints[col] = datapoints[col].apply(str)
    pointsneeded = datapoints.loc[datapoints['ChPT']=='nlo_chpt']
    pointsneeded = pointsneeded[['beta','L','mu_l','RC','poll','sample',
                               'mu_piK/fpi','mu_piK_a32']]
    pointsneeded.info()
    for m in it.product(zp,poll):
        point_data = pointsneeded.loc[(pointsneeded['RC']==m[0][1]) &
                                      (pointsneeded['poll']==m[1])]
        point_data.info()
        groups = ['beta','L','mu_l']
        obs = ['mu_piK/fpi','mu_piK_a32']
        plot_means = chi.bootstrap_means(point_data,groups,obs)
        print(plot_means)

        # load correct data file
        print(m)
        curve_method = summary.loc[(summary['zpfixing']==m[0]) &
                (summary['mupikmethod']==m[1])]
        beta = ['beta1.90','beta1.95','beta2.10','continuum']
        beta_points = ['1.9','1.95','2.1',None]
        beta_labels = [r'$a=0.0885\,\mathrm{fm}$',r'$a=0.0815\,\mathrm{fm}$',
                       r'$a=0.0619\,\mathrm{fm}$','continuum']
        fmt = ['^','v','o','-']
        col = ['r','b','g','darkgoldenrod']
        plotname = plotdir+'/pi_K_I32_chpt_fit_%s_%s'%(m[0],m[1]) 
        fig,ax=plt.subplots()
        plt.xlim((0.75,1.6))
        plt.xlabel(r'$\mu_{\pi K}/f_{\pi}$',fontsize=11)
        plt.ylabel(r'$\mu_{\pi K}a_0$',fontsize=11)
        axins = zoomed_inset_axes(ax,15,loc=3)
        x1, x2, y1, y2 = 0.80, 0.82, -0.049, -0.046 # specify the limits
        axins.set_xlim(x1, x2) # apply the x-limits
        axins.set_ylim(y1, y2) # apply the y-limits
        for a,bfc in enumerate(zip(beta,fmt,col)):
            beta_curve = curve_method.loc[curve_method['latticespacing']==bfc[0]]
            x = beta_curve['mupikoverfpi']
            y_lochpt  = -x**2/(4.*np.pi)
            y = beta_curve['mupiktimesa32']
            yerr = beta_curve['errora32']
            if bfc[0]=='continuum':
                ymin = y - yerr
                ymax = y + yerr
                ax.errorbar(x,y,color='k',label=beta_labels[a])
                ax.fill_between(x,ymin,ymax,color='gray',alpha=0.4)
                axins.errorbar(x,y,color='k')
                axins.fill_between(x,ymin,ymax,color='gray',alpha=0.4)
                ax.errorbar(x,y_lochpt,fmt='-.k',label=r'LO-ChPT')
                axins.errorbar(x,y_lochpt,fmt='-.k')
            else:
                xd = plot_means.xs(beta_points[a]).loc[:,[('mu_piK/fpi',
                                                  'own_mean')]].values[:,0]
                yd = plot_means.xs(beta_points[a]).loc[:,[('mu_piK_a32',
                                                  'own_mean')]].values[:,0]
                yerrd = plot_means.xs(beta_points[a]).loc[:,[('mu_piK_a32',
                                                     'own_std')]].values[:,0]
                if xd is not None:
                    ax.errorbar(xd,yd,yerr=yerrd,fmt=bfc[1]+bfc[2])
                             #label = beta_labels[a])
                ax.errorbar(x,y,color=bfc[2],label=beta_labels[a])
                axins.errorbar(x,y,color=bfc[2])
        axins.axvline(x=0.812,linewidth=1,color='k',linestyle='dashed')
        ax.axvline(x=0.812,linewidth=1,color='k',linestyle='dashed',label='Physical Point')
        axins.xaxis.set_visible(False)
        axins.yaxis.tick_right()
        #axins.tick_params(axis="y",direction="in", pad=-35)
        #axins.tick_params(axis="x",direction="in", pad=-25)
        mark_inset(ax, axins, loc1=2, loc2=1, fc="none", ec="0.5")
        ax.legend(loc='best')
        plt.savefig(plotname+'.pgf')
        matplotlib.backends.backend_pgf.FigureCanvasPgf(fig).print_pdf(plotname+'.pdf',bbox='standard')
        plt.clf()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
