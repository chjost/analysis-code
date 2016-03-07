#!/hadron/knippsch/Enthought/Canopy_64bit/User/bin/python
################################################################################
#
# Author: Christopher Helmes (helmes@hiskp.uni-bonn.de), Christian Jost (jost@hiskp.uni-bonn.de)
# Date:   Januar 2016
#
# Copyright (C) 2015 Christian Jost
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
#
################################################################################
#
#   Function: Test interpolation of reduced observables 
#
################################################################################

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg') # has to be imported before the next lines
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import analysis2 as ana

def plot_samples(path,ens,mus_fix,mus_match,data_low,data_high,data_ext,
                 label,name,debug=0):
    outpath = path+ens
    if debug > 0:
        print("Plotting to " + outpath)
    # stack data, be careful with order of masses
    # we want to look at B ensembles matched to D
    x = np.array((mus_fix[0],mus_fix[1],mus_match))
    # now the y values, again careful with order
    y = np.column_stack((data_low[:,0],data_high[:,0],data_ext))
    ext_plot = PdfPages(outpath+name)
    # Plot 50 samples in each plot
    # plot a subset of 50 samples
    # amin and amax of xvalues
    x_min = np.amin(x)
    x_max = np.amax(x)
    for i in range(1500):
        #_y = y[i:i+15]
        _y = y[i]
        #for i,d in enumerate(_y):
        #    plt.plot(x,d,'o--')
        plt.plot(x,_y,'o--')
    plt.xlim(x_min-x_min*0.01,x_max+x_max*0.01)
    plt.title(label[0])
    plt.xlabel(label[1])
    plt.ylabel(label[2])
    ext_plot.savefig()
    plt.clf()
    ext_plot.close()

def own_corrcoef(data,debug=0):
    m = np.empty(data.shape[0])
    for l in range(data.shape[0]):
        print data[l].shape
        m[l], std= ana.compute_error(data[l])
    x_c = data[0] - m[0]
    y_c = data[1] - m[1]
    sum_off = np.dot(x_c,y_c)
    sum_on1 = np.dot(x_c,x_c)
    sum_on2 = np.dot(y_c,y_c)
    sum_den1 = np.sqrt(sum_on1)
    sum_den2 = np.sqrt(sum_on2)
    rho = np.array([[sum_on1/(sum_den1**2),sum_off/(sum_den1*sum_den2)],
                    [sum_off/(sum_den1*sum_den2),sum_on2/(sum_den2**2)]])
    return rho

def main():

    # parse the input file
    if len(sys.argv) < 2:
        ens = ana.LatticeEnsemble.parse("A40.24.ini")
    else:
        ens = ana.LatticeEnsemble.parse(sys.argv[1])

    # get data from input file
    lat = ens.name()
    latA = ens.get_data("nameb")
    #quark = ens.get_data("quark")
    datadir = ens.get_data("datadir")
    plotdir = ens.get_data("plotdir")
    d2 = ens.get_data("d2")
    strange = ens.get_data("strangeb")
    amu_s = ens.get_data("amu_s_b")
    obs_match = 0.0197
    try:
      overwrite = ens.get_data("overwrite")
    except KeyError:
      overwrite = True

    plot_boot=False
    print(datadir)
    # Place fit results in new empty fitresults objects
    #qm_matched = ana.FitResult.create_empty(shape1,shape1,2)
    #mk_a0_matched = ana.FitResult.create_empty(shape1,shape,1)
    for i,a in enumerate(latA):
        # Read low m
        mk_low = ana.FitResult.read("%s/%s/%s/fit_k_%s.npz" % (datadir,a,strange[0],a))
        mk_low.print_data(par=1)
        mk_low.calc_error()
        #print(mk_low.weight)
        #print mk_low.pval[0].shape
        # To use the median in the interpolations, we have to switch shrinking
        # and multiplication
        mk_low = mk_low.singularize()
        #obs1 = obs1.singularize()
        obs1 = mk_low.mult_obs_single(mk_low, "m_low_sq")
        #print obs1.data[0][0,1,0]
        
        #obs1 = mk_low.res_reduced(samples=200,m_a0=True)
        #obs1 = obs1.res_reduced(samples = 20)

        # Read high m
        mk_high = ana.FitResult.read("%s/%s/%s/fit_k_%s.npz" % (datadir,a,strange[1],a))
        mk_high.print_data(par=1)
        mk_high.calc_error()
        #print(mk_high.weight)
        mk_high = mk_high.singularize()
        obs2 = mk_high.mult_obs_single(mk_high, "m_high_sq")
        #obs2 = obs2.res_reduced(samples = 20)

        print("Correlation coefficient for M_K^2 on Ensemble: %s" % a)
        data=np.hstack((obs2.data[0][:,1],obs1.data[0][:,1])).T 
        #print(np.corrcoef(data))
        #print(own_corrcoef(data))
        
        qmatch = ana.FitResult('match', derived=True)
        #print(obs1.data[0].shape)
        #print(obs2.data[0].shape)
        qmatch.evaluate_quark_mass(amu_s,obs_match, obs1, obs2)
        qmatch.print_data()
        if overwrite:
            qmatch.save("%s/%s/match_k_%s.npz" % (datadir,a,a))
        #print(qmatch.data)
        
        # Read low ma0
        mka0_low = ana.FitResult.read("%s/%s/%s/mk_akk_%s.npz" % (datadir,a,strange[0],a))
        mka0_low.print_data(par=1)
        mka0_low.calc_error()
        #print(mka0_low.weight)
        #obs3 = mka0_low.res_reduced(samples = 20,m_a0=True)
        obs3 = mka0_low.singularize()
        #print(obs3.data[0].shape)

        # Read high ma0
        mka0_high = ana.FitResult.read("%s/%s/%s/mk_akk_%s.npz" % (datadir,a,strange[1],a))
        mka0_high.print_data(par=1)
        mka0_high.calc_error()
        #print(mka0_high.weight)
        #obs4 = mka0_high.res_reduced(samples = 20,m_a0=True)
        obs4 = mka0_high.singularize()
        #print(obs4.data[0].shape)
        print("Correlation coefficient for M_K*a_KK on Ensemble: %s" % a)
        data=np.hstack((obs3.data[0],obs4.data[0])).T[0] 
        print(data.shape)
        #print(np.corrcoef(data))
        #print(own_corrcoef(data))

        mka0_ipol = ana.FitResult('eval',derived=True)
        mka0_ipol.evaluate_quark_mass(amu_s,obs_match,obs3, obs4,parobs=0)
        mka0_ipol.print_data()
        if overwrite:
            mka0_ipol.save("%s/%s/match_mk_akk_%s.npz" % (datadir,a,a))

        if plot_boot:
            label = ['Scattering length samplewise %s' % a,r'$a\mu_s$',r'$M_Ka_0^{I=1}$']
            plot_samples(plotdir,a,amu_s,obs_match,obs3.data[0],obs4.data[0],
                mka0_ipol.data[0],label,name='/mka0_ext_samples_all.pdf')
            label = ['Kaon Mass samplewise %s' % a,r'$a\mu_s$',r'$M_K^2$']
            plot_samples(plotdir,a,amu_s,obs_match,obs1.data[0][:,1],obs2.data[0][:,1],
                qmatch.data[0],label,name='/mk_sq_ext_sample_all.pdf')
        
        # Make nice plots for interpolation
# make this script importable, according to the Google Python Style Guide
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass


