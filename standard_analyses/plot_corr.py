#!/hadron/knippsch/Enthought/Canopy_64bit/User/bin/python

import sys
import numpy as np
import itertools

import analysis2 as ana

def plot_corr(dnames, fnames, ens, nboot, m_eff=True, read=False, save=True):
    """ Wrapper to plot a correlation function 
    
    Read in correlation function from dname[0]/fname[0], symmetrize and
    bootstrap it and plot the result to dname[1]/fname[1].

    Parameters
    ----------

    Returns
    -------
    
    """

    #################### Prepare correlator #################### 
    print("read single particle corrs")
    files = ["%s/%s.dat" % (dnames[0],fnames[0])]
    if read == False:
        corr = ana.Correlators(files, matrix=False)
        # Symmetrization in time and bootstrapping
        corr.sym_and_boot(nboot)
        if debug > 1:
            print(corr.shape)
        corr.save("%s/%s_%s.npy" % (dnames[0], fnames[0], ens))
    else:
        corr = ana.Correlators.read("%s/%s_%s.npz" % (dnames[0], fnames[0], ens))

  
    #################### Plot two point ####################  
    print("Plot correlator")
    label=[fnames[0], "t", "C(t)","data"]
    plotter = ana.LatticePlot("%s/%s_%s.pdf" %(dnames[1],fnames[1],ens))
    plotter.set_env(ylog=True,grid=False)
    plotter.plot(corr, label)
    if m_eff is True:
        corr.mass(usecosh=True)
        plotter.new_file("%s/%s_%s_m_eff.pdf" %(dnames[1],fnames[1],ens))
        plotter.set_env(ylog=False,grid=False)
        # adjust label
        label[2] = "m_eff(t)"
        plotter.plot(corr, label)
    del plotter


