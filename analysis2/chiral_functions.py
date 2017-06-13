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

def calc_x_plot(x):
    """ Function that calculates reduced mass divided by f_pi from mk,mpi and
    fpi"""
    xplot=reduced_mass(x[:,0],x[:,1])/x[:,2]
    return xplot

def pik_I32_chipt_plot(args, x):
    """ Wrapper for plotfunction"""
    _x = x.reshape((len(x),1))
    print(args.shape)
    return pik_I32_chipt_nlo(_x[0],_x[1],_x[2], args[0,3], args[0,0:3])

def pik_I32_chipt_lo_plot(args, x):
    """ Wrapper for plotfunction"""
    _x = x.reshape((len(x),1))
    print(args.shape)
    return pik_I32_chipt_lo(_x[0],_x[1],_x[2], args[:,1], args[:,0])

def pik_I32_chipt_nlo(mpi, mk, fpi, r0, p, lambda_x=None):
    """ Calculate mu_{piK} a_3/2 in continuum chipt at NLO plus a lattice
    artifact

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
    _mua32 = (reduced_mass(mpi,mk)/fpi)**2/(4.*np.pi)*(-1.+_sum1-_sum2+_sum3)+p[2]/(r0**2)
    return _mua32

def pik_I32_chipt_lo(mpi, mk, fpi, r0, p):
    
    _mua32 = -(reduced_mass(mpi,mk)/fpi)**2/(4.*np.pi)+(p/(r0**2))
    return _mua32

# also implement crossing even and crossing odd terms for a_0^+ and a_0^- as in
# arxiv:hep-lat/0607036

def chi_nlo_neg(ren,mpi,mk,fpi):
    # Helper observables
    mpi_sq = mpi**2
    mk_sq = mk**2
    sq_dif = mk_sq-mpi_sq
    _chi = np.zeros((5,mpi.shape[-1]))
    _chi[0] = (8.*mk_sq-5.*mpi_sq)/(2.*sq_dif) * np.log(mpi/ren)
    _chi[1] = -23.*mk_sq/(9.*sq_dif) * np.log(mk/ren)
    _chi[2] = (28.*mk_sq-9.*mpi_sq)/(18.*(sq_dif)) * np.log(np.sqrt(m_eta_sq(mpi,mk))/ren)
    _chi[3] = 4.*mk/(9.*mpi)*np.sqrt(2.*mk**2-mk*mpi-mpi_sq)/(mk+mpi)*nlo_atan_opp(mpi,mk)
    _chi[4] = -4.*mk/(9.*mpi)*np.sqrt(2.*mk**2+mk*mpi-mpi_sq)/(mk-mpi)*nlo_arctan(mpi,mk) 
    return -mpi_sq/(8.*np.pi**2*fpi**2) *np.sum(_chi,axis=0)

def chi_nlo_pos(ren,mpi,mk):

    # Helper observables
    mpi_sq = mpi**2
    mk_sq = mk**2
    sq_dif = mk_sq-mpi_sq
    _chi = np.zeros((5,mpi.shape[-1]))
    _chi[0] = 11.*mpi_sq/(2.*sq_dif)*np.log(mpi/ren) 
    _chi[1] = -(67.*mk_sq-8.*mpi_sq)/(9.*sq_dif)*np.log(mk/ren)
    _chi[2] = (24.*mk_sq-5.*mpi_sq)/(18.*sq_dif)*np.log(np.sqrt(m_eta_sq(mpi,mk)))
    _chi[3] = -4./9.*np.sqrt(2.*mk**2-mk*mpi-mpi_sq)/(mk+mpi)*nlo_atan_opp(mpi,mk) 
    _chi[4] = -4./9.*np.sqrt(2.*mk**2+mk*mpi-mpi_sq)/(mk-mpi)*nlo_arctan(mpi,mk)
    return 1./(16.*np.pi**2) *(np.sum(_chi,axis=0)+43./9.)

def gamma_pik(mpi, mk, mu_a0, fpi, ren=None):
    if ren is None:
        ren = fpi 
    _res = np.zeros((3,mpi.shape[-1]))
    _res[0] = 4.*np.pi*fpi**2/reduced_mass(mpi,mk)**2*mu_a0
    _res[1] = 1.+chi_nlo_neg(ren,mpi,mk,fpi)
    _res[2] = -2.*mk*mpi/fpi**2*chi_nlo_pos(ren,mpi,mk)
    _sum = np.sum(_res,axis=0)
    _gamma = -fpi**2/(16.*mpi**2)*_sum 
    print(_gamma[0])
    return _gamma
