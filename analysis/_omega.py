import numpy as np
import zeta_func as zf

from ._memoize import memoize

@memoize(15000)
def omega(q2, gamma=None, l=0, m=0, d=np.array([0., 0., 0.]), m_split=1.,
         prec=10e-6, exFac=False, verbose=False):
    """Calculates the Zeta function including the some prefactor.

    Args:
        q2: The squared momentum transfer.
        gamma: The Lorentz boost factor.
        l, m: The quantum numbers.
        d: The total momentum vector of the system.
        m_split: The mass difference between the particles.
        prec: The calculation precision.
        exFac: excludes 1/(gamma*q*sqrt(PI)^3) from the result
        verbose: The amount of info printed.

    Returns:
        The value of the Zeta function.
    """
    if exFac:
        if l is not 0:
            factor = np.power(np.sqrt(q2), l) * np.sqrt(2*l)
        else:
            factor = 1.
    else:
        factor = gamma*np.power(np.sqrt(q2), l+1)*np.power(np.pi, 1.5)*\
                 np.sqrt(2*l+1)
    var =  zf.Z(q2, gamma, l, m, d, m_split, prec, verbose)
    return var / factor
