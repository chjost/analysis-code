#import sys
from scipy import stats
from scipy import interpolate as ip
import time
import matplotlib
matplotlib.use('Agg') # has to be imported before the next lines
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.lines as mlines
import numpy as np
from numpy.polynomial import polynomial as P
from fit_routines import fitting
from chipt_nlo import *
# Christian's packages
import analysis2 as ana

# chipt functions for fitting

def chi_I32_nlo(ren,mpi,mk):
    # initialize a 2d-array (5,nboot) for the terms
    # take sum afterwards
    _chi = np.zeros((5,mpi.shape[-1]))
    _chi[0] = kappa_pi(mpi,mk)*np.log(mpi**2/ren**2) 
    _chi[1] = kappa_k(mpi,mk)*np.log(mk**2/ren**2)
    _chi[2] = kappa_eta(mpi,mk)*np.log(m_eta_sq(mpi,mk)/ren**2)
    _chi[3] = 86./9.*mpi*mk
    _chi[4] = kappa_tan(mpi,mk)*nlo_arctan(mpi,mk)
    return np.sum(_chi,axis=0)


def reduced_mass(m_1,m_2):
    """ reduced mass of a system of two different particles"""
    return m_1*m_2/(m_1+m_2)

def pik_I32_chipt_nlo(mpi, mk, fpi, p, lambda_x=None):
    """ Calculate mu_{piK} a_3/2 in continuum chipt at NLO

    Takes values for mpi, mk and fpi and returns the product mu_{piK} a_3/2

    Parameters
    ----------
    mpi : 1d-array, pion mass
    mk : 1d-array, kaon mass
    fpi : 1d-array, pion decay constant
    p : nd-array, the LECs to fit

    Returns
    -------
    _mua32 : 1d-array, the calculated values of _mua32
    """
    if lambda_x is None:
        lambda_x = fpi
    # Term with L_piK, a collection of Lower LECs
    _sum1 = p[0]*32.*mpi*mk/fpi**2
    # Term with L5
    _sum2 = p[1]*16.*mpi**2/fpi**2
    # Term with NLO function (does not take eta mass at the moment)
    _sum3 = chi_I32_nlo(lambda_x, mpi, mk)
    _mua32 = (reduced_mass(mpi,mk)/fpi)**2/(4.*np.pi)*(-1.+_sum1-_sum2+_sum3)
    return _mua32

def pik_I32_chipt_lo(mpi, mk, fpi, r0, p):
    
    _mua32 = -(reduced_mass(mpi,mk)/fpi)**2/(4.*np.pi)+(p/(r0**2))
    return _mua32
