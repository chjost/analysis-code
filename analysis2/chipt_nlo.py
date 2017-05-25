import numpy as np

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

# Gell-Mann-Okubo formula to calculate squared eta mass
def m_eta_sq(mpi,mk):
    return (4.*mk**2-mpi**2)/3.

def nlo_arctan(mpi,mk):
    _prod1 = 2.*(mk-mpi)/(mk+2*mpi)
    _prod2 = np.sqrt((mk+mpi)/(2.*mk-mpi))
    return np.arctan(_prod1*prod_2)
