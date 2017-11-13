import numpy as np
from chipt_basic_observables import *
# kappa realtions relevant to the nlo-formula for pik-scattering depend on
# isospin
def kappa_pi(mpi,mk,i=32):
    _den = 2.*(mk**2-mpi**2) 
    if i == 32:
        _kappa = (11.*mk*mpi**3+8.*mpi**2*mk**2-5.*mpi**4)/_den
    elif i == 12:
        _kappa = (11.*mk*mpi**3-16.*mpi**2*mk**2+10.*mpi**4)/_den
    else:
        raise ValueError("Isospin Value not known!")
    return _kappa

def kappa_k(mpi,mk,i=32):
    _den = 9.*(mk**2-mpi**2)
    if i == 32:
        _kappa = -(67.*mk**3*mpi-8.*mpi**3*mk+23.*mk**2*mpi**2)/_den
    elif i == 12:
        _kappa = -(67.*mk**3*mpi-8.*mpi**3*mk-46.*mk**2*mpi**2)/_den
    else:
        raise ValueError("Isospin Value not known!")
    return _kappa

def kappa_eta(mpi,mk,i=32):
    _den = 18.*(mk**2-mpi**2)
    if i == 32:
        _kappa = (24.*mpi*mk**3-5.*mk*mpi**3+28.*mk**2*mpi**2-9*mpi**4)/_den
    elif i == 12:
        _kappa = (24.*mpi*mk**3-5.*mk*mpi**3-56.*mk**2*mpi**2+18*mpi**4)/_den
    else:
        raise ValueError("Isospin Value not known!")
    return _kappa

def kappa_tan(mpi,mk,i=32):
    if i == 32:
        _kappa = -(16.*mk*mpi*np.sqrt(2.*mk**2+mk*mpi-mpi**2))/(9.*(mk-mpi))
    elif i == 12:
        _kappa = (8.*mk*mpi*np.sqrt(2.*mk**2-mk*mpi-mpi**2))/(9.*(mk+mpi))
    else:
        raise ValueError("Isospin Value not known!")
    return _kappa

def nlo_arctan(mpi,mk):
    _prod1 = 2.*(mk-mpi)/(mk+2*mpi)
    _prod2 = np.sqrt((mk+mpi)/(2.*mk-mpi))
    return np.arctan(_prod1 * _prod2)

# same as above but switched signs
def nlo_atan_opp(mpi,mk):
    _prod1 = 2.*(mk+mpi)/(mk-2*mpi)
    _prod2 = np.sqrt((mk-mpi)/(2.*mk+mpi))
    return np.arctan(_prod1 * _prod2)
# also implement crossing even and crossing odd terms for a_0^+ and a_0^- as in
# arxiv:hep-lat/0607036

def chi_nlo_neg(ren,mpi,mk,fpi,meta=None):
    # Helper observables
    mpi_sq = mpi**2
    mk_sq = mk**2
    sq_dif = mk_sq-mpi_sq
    _chi = np.zeros((5,mpi.shape[-1]))
    _chi[0] = (8.*mk_sq-5.*mpi_sq)/(2.*sq_dif) * np.log(mpi/ren)
    _chi[1] = -23.*mk_sq/(9.*sq_dif) * np.log(mk/ren)
    if meta is None:
        _chi[2] = (28.*mk_sq-9.*mpi_sq)/(18.*(sq_dif)) * np.log(np.sqrt(m_eta_sq(mpi,mk))/ren)
    else:
        _chi[2] = (28.*mk_sq-9.*mpi_sq)/(18.*(sq_dif)) * np.log(meta/ren)
    _chi[3] = 4.*mk/(9.*mpi)*np.sqrt(2.*mk_sq-mk*mpi-mpi_sq)/(mk+mpi)*nlo_atan_opp(mpi,mk)
    _chi[4] = -4.*mk/(9.*mpi)*np.sqrt(2.*mk_sq+mk*mpi-mpi_sq)/(mk-mpi)*nlo_arctan(mpi,mk) 
    return -mpi_sq/(8.*np.pi**2*fpi**2) *np.sum(_chi,axis=0)

def chi_nlo_pos(ren,mpi,mk,meta=None):

    # Helper observables
    mpi_sq = mpi**2
    mk_sq = mk**2
    sq_dif = mk_sq-mpi_sq
    _chi = np.zeros((5,mpi.shape[-1]))
    _chi[0] = 11.*mpi_sq/(2.*sq_dif)*np.log(mpi/ren) 
    _chi[1] = -(67.*mk_sq-8.*mpi_sq)/(9.*sq_dif)*np.log(mk/ren)
    if meta is None:
        _chi[2] = (24.*mk_sq-5.*mpi_sq)/(18.*sq_dif)*np.log(np.sqrt(m_eta_sq(mpi,mk))/ren)
    else:
        _chi[2] = (24.*mk_sq-5.*mpi_sq)/(18.*sq_dif)*np.log(meta/ren)
    _chi[3] = -4./9.*np.sqrt(2.*mk_sq-mk*mpi-mpi_sq)/(mk+mpi)*nlo_atan_opp(mpi,mk) 
    _chi[4] = -4./9.*np.sqrt(2.*mk_sq+mk*mpi-mpi_sq)/(mk-mpi)*nlo_arctan(mpi,mk)
    return 1./(16.*np.pi**2) *(np.sum(_chi,axis=0)+43./9.)

# chipt functions for fitting, according to arXiv:1110.1422
# TODO: Do that via linear combination of chi_pos and chi_neg?
def chi_I32_nlo(ren,mpi,mk,meta=None):
    # initialize a 2d-array (5,nboot) for the terms
    # take sum afterwards
    _chi = np.zeros((5,mpi.shape[-1]))
    _chi[0] = kappa_pi(mpi,mk)*np.log(mpi**2/ren**2) 
    _chi[1] = kappa_k(mpi,mk)*np.log(mk**2/ren**2)
    if meta is None:
        _chi[2] = kappa_eta(mpi,mk)*np.log(m_eta_sq(mpi,mk)/ren**2)
    else:
        _chi[2] = kappa_eta(mpi,mk)*np.log(meta**2/ren**2)
    _chi[3] = 86./9.*mpi*mk
    _chi[4] = kappa_tan(mpi,mk)*nlo_arctan(mpi,mk)
    return np.sum(_chi,axis=0)
