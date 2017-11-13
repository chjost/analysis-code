import numpy as np
from chipt_basic_observables import *
from chipt_nlo import *

# collection of the pik scattering length formulae
def pik_I32_chipt_lo(mpi, mk, fpi, r0, p):
    
    _mua32 = -(reduced_mass(mpi,mk)/fpi)**2/(4.*np.pi)+(p/(r0**2))
    return _mua32
def pik_I32_chipt_nlo(mpi, mk, fpi, p, lambda_x=None, meta=None, lat=None):
    """ Calculate mu_{piK} a_3/2 in continuum chipt at NLO plus a lattice
    artifact

    Takes values for mpi, mk and fpi and returns the product mu_{piK} a_3/2

    Parameters
    ----------
    mpi : 1d-array, pion mass
    mk : 1d-array, kaon mass
    fpi : 1d-array, pion decay constant
    p : nd-array, the LECs to fit
    lambda_x: 1d-array, chiral renormalization scale
    meta: 1d-array, optional eta mass
    lat: 1d-array, optional lattice artefact. if none set to mpi^2


    Returns
    -------
    _mua32 : 1d-array, the calculated values of _mua32
    """
    #check inputs
    if lambda_x is None:
        lambda_x = fpi
    _p=p
    # Term with L_piK, a collection of Lower LECs, dependent on L_5
    _sum1 = _p[0]*32.*mpi*mk/fpi**2
    # Term with L5
    _sum2 = _p[1]*16.*mpi**2/fpi**2
    # Term with NLO function (does not take eta mass at the moment)
    _sum3 = chi_I32_nlo(lambda_x, mpi, mk, meta)/(16.*np.pi**2*fpi**2)
    _mua32 = (reduced_mass(mpi,mk)/fpi)**2/(4.*np.pi)*(-1.+_sum1-_sum2+_sum3)
    # Ensure that lattice value of Mpi is used if no lattice artefact is given
    if lat is None:
        lat = mpi**2
    # Add Lattice artefact to evaluation
    if _p[2].any != 0:
        _mua32 += _p[2]*lat
    return _mua32
# TODO: Ugly code doubling but ok for trying out
def pik_I32_chipt_nlo_cont(mpi, mk, fpi, r0, p, lambda_x=None, meta=None):
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
    #check inputs
    if lambda_x is None:
        lambda_x = fpi
    _p=p
    # Term with L_piK, a collection of Lower LECs, dependent on L_5
    _sum1 = _p[0]*32.*mpi*mk/fpi**2
    # Term with L5
    #_sum2 = _p[1]*16.*mpi**2/fpi**2
    _sum2 = _p[1]*16.*mpi**2/fpi**2
    # Term with NLO function (does not take eta mass at the moment)
    _sum3 = chi_I32_nlo(lambda_x, mpi, mk, meta)/(16.*np.pi**2*fpi**2)
    _mua32 = (reduced_mass(mpi,mk)/fpi)**2/(4.*np.pi)*(-1.+_sum1-_sum2+_sum3)
    return _mua32

# Crossing even and crossing odd scattering lengths a_0^\pm
def a_pik_pos(ren,mpi,mk,fpi,l_pik):
    """ Calculate pi-K crossing even scattering length

    Parameters
    ----------
    ren: 1darray, renormalisation scale 
    mpi: 1darray, pion mass
    mk:  1darray, kaon mass
    fpi: 1darray, pion decay constant
    l_pik: 1darray, Collection of Gasser-Leutwyler LECs: 
                    L_pik = 2L_1 + 2L_2 + L_3 - 2L_4 - L_5/2 + 2L_6 + L_8

    Returns
    -------
    a_pik_pos: 1darray, crossing even scattering length
    """
    _paren = 16.*l_pik + chi_nlo_pos(ren,mpi,mk)
    _pre = reduced_mass(mpi,mk)*mpi*mk/(2.*np.pi*fpi**4)
    return _pre * _paren

def a_pik_neg(ren,mpi,mk,fpi,l_5):
    """ Calculate pi-K crossing odd scattering length

    Parameters
    ----------
    ren: 1darray, renormalisation scale 
    mpi: 1darray, pion mass
    mk:  1darray, kaon mass
    fpi: 1darray, pion decay constant
    l_5: 1darray, Gasser-Leutwyler LEC L_5

    Returns
    -------
    a_pik_pos: 1darray, crossing odd scattering length
    """
    _paren = 1.+16.*mpi**2/fpi**2*l_5+chi_nlo_neg(ren,mpi,mk,fpi)
    _pre = reduced_mass(mpi,mk)/(4.*np.pi*fpi**2)
    return _pre * _paren

def a_I32(ren,mpi,mk,fpi,l_5,l_pik):
    return a_pik_pos(ren,mpi,mk,fpi,l_pik) - a_pik_neg(ren,mpi,mk,fpi,l_5)

def a_I12(ren,mpi,mk,fpi,l_5,l_pik):
    return a_pik_pos(ren,mpi,mk,fpi,l_pik) + 2 * a_pik_neg(ren,mpi,mk,fpi,l_5)

def mu_aI32(ren,mpi,mk,fpi,l_5,l_pik):
    return reduced_mass(mpi,mk) * a_I32(ren,mpi,mk,fpi,l_5,l_pik)

def mu_aI12(ren,mpi,mk,fpi,l_5,l_pik):
    return reduced_mass(mpi,mk) * a_I12(ren,mpi,mk,fpi,l_5,l_pik)
  # Wrapper functions for evaluate_phys in chiral_utils.py
  # Operates bootstrapsample wise

def gamma_pik(mpi, mk, mu_a0, fpi, meta=None, ren=None):
    if ren is None:
        ren = fpi 
    _res = np.zeros((3,mpi.shape[-1]))
    _res[0] = 4.*np.pi*fpi**2/reduced_mass(mpi,mk)**2*mu_a0
    _res[1] = 1.+chi_nlo_neg(ren,mpi,mk,fpi,meta)
    _res[2] = -2.*mk*mpi/fpi**2*chi_nlo_pos(ren,mpi,mk,meta)
    _sum = np.sum(_res,axis=0)
    _gamma = -fpi**2/(16.*mpi**2)*_sum 
    return _gamma
