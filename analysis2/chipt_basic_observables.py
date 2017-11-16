import numpy as np
from statistics import draw_gauss_distributed

def mpi_sq(ml, b0=None):
    if b0 is None:
      #TODO: where to take B_0 from?
      b0 = draw_gauss_distributed(origin=True)
    return 2.*b0*ml

def mk_sq(ml, ms=None, b0=None):
    if b0 is None:
      #TODO: where to take B_0 from?
      b0 = draw_gauss_distributed(origin=True)
    if ms is None:
      ms = draw_gauss_distributed(origin=True)
    return b0*(ml+ms)

# Gell-Mann-Okubo formula to calculate squared eta mass
def m_eta_sq(mpi,mk):
    return (4.*mk**2-mpi**2)/3.

def f_pi(ml,f0=None,l4=None,b0=None):
    if f0 is None:
      #f0 = ana.draw_gauss_distributed(93.3,0,ml.shape[0],origin=True)
      f0 = np.full_like(ml,121.1)
    #if l4 is None:
    #  l4 = ana.draw_gauss_distributed(,,ml.shape[0],origin=True,seed=1337)
    #return f0 - 2*mpi_sq(ml,b0)*l4/((4*np.pi*f0)**2)
    return f0 

def reduced_mass(m_1,m_2):
    """ reduced mass of a system of two different particles"""
    return m_1*m_2/(m_1+m_2)
