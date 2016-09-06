"""
Functions to calculate the phaseshift.
"""

import numpy as np
from zeta_wrapper import omega
from utils import loop_iterator
from timeit import default_timer as timer

def compute_phaseshift(q2, q2_w, gamma, gamma_w, L=24, isdependend=True,
        d2=0, irrep="A1"):
    # setup variables
    if np.any(gamma < 1.):
        print(gamma)
    nsamples = gamma[0].shape[0]
    needed = np.zeros((nsamples,))
    print("calculating phaseshift, irrep %s, d2 = %d"% (irrep, d2))
    for i, q in enumerate(q2):
        print("correlator %d" % i)
        res = np.zeros_like(needed)
        tmpweight = q2_w[i]*gamma_w[i]
        if np.any(gamma[i] < 1.):
            print("skipping gamma < 1 in correlator %d" % i)
            print(gamma[i])
            continue
        if isdependend:
            tmp, tmp1 = get_solution(q, gamma[i], d2, irrep)
            # iterate over fit ranges
            for item in loop_iterator(q.shape[1:]):
                res = tmp[(slice(None),)+tuple(item)]
                res1 = tmp1[(slice(None),)+tuple(item)]
                weight = np.ones_like(needed) * tmpweight[item]
                yield (((0,i)+tuple(item), res, needed, weight),
                       ((0,i)+tuple(item), res1, needed, weight))
        else:
            # iterate over fit ranges
            for item in loop_iterator(q.shape[1:]):
                tmp, tmp1 = get_solution(q[:,item[0]], gamma[i], d2, irrep)
                res = tmp[(slice(None),)+tuple(item[1:])]
                res1 = tmp1[(slice(None),)+tuple(item[1:])]
                weight = np.ones_like(needed) * tmpweight[item]
                yield (((0,i)+tuple(item), res, needed, weight),
                       ((0,i)+tuple(item), res1, needed, weight))
        if i == 0:
            raise StopIteration

def get_solution(q2, gamma, d2, irrep="A1"):
    if np.any(gamma < 1.):
        print("error: gamma < 1.")
        raise ValueError("blub")
    if irrep == "A1":
        cotd, delta = solutions_A1(q2, gamma, d2)
    elif irrep == "E":
        cotd, delta = solutions_E(q2, gamma, d2)
    elif irrep == "T2":
        cotd, delta = solutions_T2(q2, gamma, d2)
    else:
        raise RuntimeWarning("irrep %s not implemented" % irrep)
        cotd, delta = None, None
    if cotd is None or delta is None:
        raise RuntimeWarning("irrep %s with momentum %d not implemented, returning 0!" % (
            irrep, d2))
        cotd = np.zeros_like(q2)
        delta = np.zeros_like(q2)
    return cotd, delta

def solutions_A1(q2, gamma, d2):
    """Calculates cot(delta_0) and delta_0 for the A1 irrep.

    Parameters
    ----------
    q2 : ndarray
        The squared momentum.
    gamma : ndarray
        The Lorentz boost of the system.
    d2 : int
        The total momentum squared.

    Returns
    -------
    cotd : ndarray
        The cotangent of the phaseshift.
    delta : ndarray
        The phaseshift.
    """
    w_00 = omega(q2, gamma).real
    if d2 == 0:
        cotd = w_00
        delta = np.arctan2(1.,cotd)*180./np.pi
        #delta = np.arctan(1./cotd)*180./np.pi
    elif d2 == 1:
        w_20 = omega(q2, gamma, l=2).real
        w_40 = omega(q2, gamma, l=4).real
        tmp = w_00 +10./7.*w_20+18./7.*w_40-1./np.tan(1./(-65.13*np.sqrt(q2)))
        cotd = w_00 - 5. * np.square(w_20)/tmp
        delta = np.arctan2(1.,cotd) * 180. / np.pi
        #delta = np.arctan(1./cotd) * 180. / np.pi
    elif d2 == 2:
        w_20 = np.square(omega(q2, gamma, l=2)).real
        w_22 = np.square(omega(q2, gamma, l=2, m=2)).real
        w_42 = np.square(omega(q2, gamma, l=4, m=2)).real
        tmp1 = 5. * w_20 + 10. * w_22
        tmp2 = 1. - 200./49.*w_22 - 270./49. * w_42
        cotd = - (tmp1) / (tmp2) + w_00
        delta = np.arctan(1./cotd)*180./np.pi
        #res1 = np.arctan2(-tmp2, tmp2 * w_00 + tmp1) * 180. / np.pi
    elif d2 == 3:
        w_22 = np.square(omega(q2, gamma, l=2, m=2)).real
        cotd = 30.*w_22 + w_00
        delta = np.arctan(1./cotd)*180./np.pi
    else:
        codt = None
        delta = None
    return cotd, delta

def solutions_B1(q2, gamma, d2):
    """Calculates cot(delta_2) and delta_2 for the B1 irrep.

    Parameters
    ----------
    q2 : ndarray
        The squared momentum.
    gamma : ndarray
        The Lorentz boost of the system.
    d2 : int
        The total momentum squared.

    Returns
    -------
    cotd : ndarray
        The cotangent of the phaseshift.
    delta : ndarray
        The phaseshift.
    """
    w_00 = omega(q2, gamma).real
    w_20 = omega(q2, gamma, l=2).real
    w_40 = omega(q2, gamma, l=4).real
    if d2 == 1:
        w_44 = omega(q2, gamma, l=4, m=4).real
        cotd = w_00 - 10./7.*w_20 + 3./7.*w_40 + 3./7.*np.sqrt(70)*w_44
        delta = np.arctan(1./cotd) * 180. / np.pi
    elif d2 == 2:
        w_22 = omega(q2, gamma, l=2, m=2).imag
        w_42 = omega(q2, gamma, l=4, m=2).imag
        cotd = w_00 + 5./.7*w_20 - 12./7.*w_40 + 5.*np.sqrt(6.)/7.*w_22 +\
            6.*np.sqrt(10.)/7.*w_42
    else:
        cotd = None
        delta = None
    return cotd, delta

def solutions_B2(q2, gamma, d2):
    """Calculates cot(delta_2) and delta_2 for the B2 irrep.

    Parameters
    ----------
    q2 : ndarray
        The squared momentum.
    gamma : ndarray
        The Lorentz boost of the system.
    d2 : int
        The total momentum squared.

    Returns
    -------
    cotd : ndarray
        The cotangent of the phaseshift.
    delta : ndarray
        The phaseshift.
    """
    w_00 = omega(q2, gamma).real
    w_20 = omega(q2, gamma, l=2).real
    w_40 = omega(q2, gamma, l=4).real
    w_44 = omega(q2, gamma, l=4, m=4).real
    if d2 == 1:
        cotd = w_00 - 10./7.*w_20 + 3./7.*w_40 - 3./7.*np.sqrt(70)*w_44
        delta = np.arctan(1./cotd) * 180. / np.pi
    elif d2 == 2:
        cotd = w_00 - 10./7.*w_20 + 3./7.*w_40 + 3./7.*np.sqrt(70)*w_44
        delta = np.arctan(1./cotd) * 180. / np.pi
    else:
        cotd = None
        delta = None
    return cotd, delta

def solutions_E(q2, gamma, d2):
    """Calculates cot(delta_2) and delta_2 for the E irrep.

    Parameters
    ----------
    q2 : ndarray
        The squared momentum.
    gamma : ndarray
        The Lorentz boost of the system.
    d2 : int
        The total momentum squared.

    Returns
    -------
    cotd : ndarray
        The cotangent of the phaseshift.
    delta : ndarray
        The phaseshift.
    """
    w_00 = omega(q2, gamma).real
    if d2 == 0:
        w_40 = omega(q2, gamma, l=4).real
        cotd = w_00 + 18./7. * w_40
        delta = np.arctan2(1.,cotd) * 180. / np.pi
    elif d2 == 1:
        w_20 = omega(q2, gamma, l=2).real
        w_40 = omega(q2, gamma, l=4).real
        cotd = w_00 - 5./7.*w_20 - 12./7.*w_40
        delta = np.arctan(1./cotd) * 180. / np.pi
    elif d2 == 3:
        w_22 = omega(q2, gamma, l=2, m=2).real
        w_40 = omega(q2, gamma, l=4).real
        w_42 = omega(q2, gamma, l=4, m=2).real
    else:
        cotd = None
        delta = None
    return cotd, delta

def solutions_T2(q2, gamma, d2):
    """Calculates cot(delta_2) and delta_2 for the E irrep.

    Parameters
    ----------
    q2 : ndarray
        The squared momentum.
    gamma : ndarray
        The Lorentz boost of the system.
    d2 : int
        The total momentum squared.

    Returns
    -------
    cotd : ndarray
        The cotangent of the phaseshift.
    delta : ndarray
        The phaseshift.
    """
    w_00 = omega(q2, gamma).real
    if d2 == 0:
        w_40 = omega(q2, gamma, l=4).real
        cotd = w_00 - 12./7. * w_40
        delta = np.arctan2(1.,cotd) * 180. / np.pi
    else:
        cotd = None
        delta = None
    return cotd, delta

def calculate_phaseshift(q2, gamma=None, d2=0, irrep="A1", prec=1e-5, debug=0):
    """Calculates the phase shift using Luescher's Zeta function.

    The names of the Irreps are the same as in M. Goeckeler et al., Phys. Rev.
    D 86, 094513 (2012).

    Parameters
    ----------
    q2 : float or ndarray
        The momentum shift squared.
    gamma : float or ndarray, optional
        The Lorentz factor for moving frames.
    d2 : int, optional
        The squared total three momentum of the system.
    irrep : str, optional
        The irrep for which to calculate, defaults to A1.
    prec : float, optional
        The precision of the Zeta function calculation.
    debug : int, optional
        The amount of information printed to screen.

    Returns:
        An array of the phase shift and tan(delta).
    """
    if irrep == "A1":
        return phaseshift_A1(q2, gamma, d2, prec, debug)
    elif irrep == "T1":
        return phaseshift_T1(q2, gamma, d2, prec, debug)
    else:
        raise RuntimeError("Irrep %s not implemented" % irrep)

def phaseshift_A1(q2, gamma=None, d2=0, prec=1e-5, debug=0):
    """Calculates the phase shift for irrep A1 using Luescher's Zeta function.

    The names of the Irreps are the same as in M. Goeckeler et al., Phys. Rev.
    D 86, 094513 (2012).

    Parameters
    ----------
    q2 : float or ndarray
        The momentum shift squared.
    gamma : float or ndarray, optional
        The Lorentz factor for moving frames.
    d2 : int, optional
        The squared total three momentum of the system.
    prec : float, optional
        The precision of the Zeta function calculation.
    debug : int, optional
        The amount of information printed to screen.

    Returns:
        An array of the phase shift and tan(delta).
    """
    if gamma == None:
        _gamma = np.ones_like(q2)
    else:
        _gamma = gamma
    # init calculation
    _pi3 = np.pi**3
    _num = _gamma * np.sqrt(_pi3 * q2)
    #CMF
    if d2 == 0:
        raise NotImplementedError("")
        #_den = omega(q2, _gamma, 0, 0, d, exFac=True).real
        #tandelta = _num / _den
        #delta = np.arctan2( _num, _den)
    # MF1
    elif d2 == 1 or d2 ==4:
        if d2 == 1:
            d = np.array([0., 0., 1.])
        else:
            d = np.array([0., 0., 2.])
        _den = omega(q2, _gamma, 0, 0, d, exFac=True).real + \
               2 * omega(q2, _gamma, 2, 0, d, exFac=True).real
        tandelta = _num / _den
        delta = np.arctan2( _num, _den)
    # MF2
    elif d2 == 2:
        d = np.array([1., 1., 0.])
        _den = omega(q2, _gamma, 0, 0, d, exFac=True).real -\
               omega(q2, _gamma, 2, 0, d, exFac=True).real +\
               np.sqrt(6) * omega(q2, _gamma, 2, 2, d, exFac=True).imag
        tandelta = _num / _den
        delta = np.arctan2( _num, _den)
    # MF3
    elif d2 == 3:
        d = np.array([1., 1., 1.])
        _den = omega(q2, _gamma, 0, 0, d, exFac=True).real +\
               2 * np.sqrt(6) * omega(q2, _gamma, 2, 2, d, exFac=True).imag
        tandelta = _num / _den
        delta = np.arctan2( _num, _den)
    else:
        raise RuntimeError("moving frame for d2 = %d not implemented" % d2)
    sindelta = np.sin(delta)**2
    return delta, tandelta, sindelta

def phaseshift_T1(q2, gamma=None, d2=0, prec=1e-5, debug=0):
    if gamma == None:
        _gamma = np.ones_like(q2)
    else:
        _gamma = gamma
    # init calculation
    _pi3 = np.pi**3
    _num = _gamma * np.sqrt(_pi3 * q2)
    #CMF
    if d2 == 0:
        d = np.array([0., 0., 0.])
        _den = omega(q2, _gamma, 0, 0, d, exFac=True).real
        tandelta = _num / _den
        delta = np.arctan2( _num, _den)
    else:
        raise RuntimeError("moving frame for d2 = %d not implemented" % d2)
    sindelta = np.sin(delta)**2
    return delta, tandelta, sindelta

if __name__ == "main":
    pass
