import numpy as np
from chipt_logs import *
from chipt_basic_observables import *

#TODO: Is that placed correctly here?
def chipt_logs_fk_by_fpi(mpi,mk,meta,f,mu):
    """Helper functions for chiral formulae

    Arguments
    ---------
    Returns
    -------
    """
    logs = np.zeros((3,mpi.shape[-1]))
    logs[0] =1.25*mass_log(mpi,f,mu) 
    logs[1] =-0.5*mass_log(mk,f,mu) 
    logs[2] =-0.75*mass_log(meta,f,mu)
    return logs

def fk_by_fpi(mpi,mk,meta,f,mu,l5):
    """ Compute the ratio fk/fpi from meson masses, and chiral constants

    As a prerequisite the chiral logarithms are determined first before summing
    everything up. The formula is taken in the continuum from NPLQCD
    arxiv:hep-lat/0606023

    Arguments
    ---------
    mpi, mk, meta: np-arrays for the masses of pion, kaon and eta meson
    f: np-array for the chiral decay constant
    mu: renormalization constant
    l5: Low energy constant L_5

    Returns
    -------
    fk/fpi: np-array of same dimension as mpi
    """
    logs = chipt_logs_fk_by_fpi(mpi,mk,meta,f,mu)
    ratio = 1+np.sum(logs,axis=0)+8./f**2*(mk**2-mpi**2)*l5
    return ratio

def l5(mpi,mk,meta,fk,fpi,f,mu):
    """Rearrangement from fk_by_fpi (therefore the negative of the sum over logs
    is taken)
    """
    logs = chipt_logs_fk_by_fpi(mpi,mk,meta,f,mu) 
    l5 = (fk/fpi -1 -np.sum(logs,axis=0))*f**2/(8.*(mk**2-mpi**2))
    return l5
