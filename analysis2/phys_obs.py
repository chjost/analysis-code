import numpy as np
from .statistics import draw_gauss_distributed
# compute tau_1S for pi K atoms
def pi_k_tau(ma12,ma32,phys_in):
    """
    Calculate pik atom lifetime input parameters taken from arxiv:1707.02184
    """
    # hbar in Mev*s
    hbar = 6.58211899e-22
    hbarc = 197.33
    # bootstraps:
    # \delta_K
    delta_K = draw_gauss_distributed(0.04,0.022,ma12.shape,origin=True)
    # \alpha
    alpha = np.full_like(delta_K,7.29735254e-3)
    # reduced mass of piK atom
    mu_pik = np.full_like(delta_K,109.)
    # p*
    p_star = np.full_like(delta_K,11.8)
    # isosppin odd scattering length
    a_0 = (ma12-ma32)*hbarc/(3.*phys_in.get('mpi_0'))
    _tau = hbar*hbarc**2/(8*alpha**3*a_0**2*mu_pik**2*p_star*(1+delta_K))
    return _tau

def pi_k_tau_pandas(mpia12,mpia32,delta_K,mpi,mk,p_star):
    """
    Calculate pik atom lifetime input parameters taken from arxiv:1707.02184
    """
    # hbar in Mev*s
    hbar = 6.58211899e-22
    hbarc = 197.33
    # bootstraps:
    # \alpha
    alpha = np.full_like(delta_K,7.29735254e-3)
    # reduced mass of piK atom
    mu_pik  = mpi*mk/(mpi+mk) 
    # isospin odd scattering length
    # a_neg only includes L_5 better to take difference involving L_piK?
    #_a_neg = a_neg*hbarc
    _a_neg = (mpia12-mpia32)*hbarc/(3.*mpi)
    _tau = hbar*hbarc**2/(8*alpha**3*_a_neg**2*mu_pik**2*p_star*(1+delta_K))
    return _tau
