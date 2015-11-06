
import numpy as np
from ._omega import omega

def calculate_delta(q2, gamma=None, d=np.array([0., 0., 0.]), prec=10e-6,
                    verbose=0):
    """Calculates the phase shift using Luescher's Zeta function.

    Most arguments are for the Zeta function. For each q2 a gamma is needed.
    WARNING: The momentum vectors d are compared to hardcoded momentum vectors
    because of the zeta function calculation, which was derived for exactly
    these momentum vectors. Make sure to use the right ones.
    The names of the Irreps are the same as in M. Goeckeler et al., Phys. Rev.
    D 86, 094513 (2012).

    Args:
        q2: The momentum shift squared.
        gamma: The Lorentz factor for moving frames.
        d: The total three momentum of the system.
        prec: The precision of the Zeta function calculation.
        verbose: The amount of information printed to screen.

    Returns:
        An array of the phase shift and tan(delta).
    """
    # create array for results
    delta=np.zeros(q2.shape)
    tandelta=np.zeros(q2.shape)
    sindelta=np.zeros(q2.shape)
    _gamma = gamma
    if _gamma == None:
        _gamma = np.ones(q2.shape)
    # init calculation
    _pi3 = np.pi**3
    _num = _gamma * np.sqrt(_pi3 * q2)
    #CMF
    if np.array_equal(d, np.array([0., 0., 0.])):
        # Irrep. T_1
        #_z1 = zeta.Zp(q2, _gamma, 0, 0, d, 1., prec, verbose).real
        #tandelta = np.sqrt( _pi3 * q2) / _z1.real
        #delta = np.arctan2(np.sqrt( _pi3 * q2), _z1.real)
        _den = omega(q2, _gamma, 0, 0, d, exFac=True).real
        tandelta = _num / _den
        delta = np.arctan2( _num, _den)
    # MF1
    elif np.array_equal(d, np.array([0., 0., 1.])) or \
         np.array_equal(d, np.array([0. ,0., 2.])):
        # Irrep. A_1
        #_z1 = zeta.Zp(q2, _gamma, 0, 0, d, 1., prec, verbose).real
        #_z2 = zeta.Zp(q2, _gamma, 2, 0, d, 1., prec, verbose).real
        #_num = _gamma * np.sqrt(_pi3 * q2)
        #_den = (_z1 + (2. / (np.sqrt(5) * q2)) * _z2).real
        _den = omega(q2, _gamma, 0, 0, d, exFac=True).real + \
               2 * omega(q2, _gamma, 2, 0, d, exFac=True).real
        print 'tandelta'
        tandelta = _num / _den
        print 'delta'
        delta = np.arctan2( _num, _den)
    # MF2
    elif np.array_equal(d, np.array([1., 1., 0.])):
        # Irrep. A_1
        #_z1 = zeta.Zp(q2, _gamma, 0, 0, d, 1., prec, verbose).real
        #_z2 = zeta.Zp(q2, _gamma, 2, 0, d, 1., prec, verbose).real
        #_z3 = zeta.Zp(q2, _gamma, 2, 2, d, 1., prec, verbose).imag
        #_num = _gamma * np.sqrt(_pi3 * q2)
        #_den = (_z1 - (1. / (np.sqrt(5) * q2)) * _z2 + ( np.sqrt(6./5.) /
        #        q2) * _z3 ).real
        _den = omega(q2, _gamma, 0, 0, d, exFac=True).real -\
               omega(q2, _gamma, 2, 0, d, exFac=True).real +\
               np.sqrt(6) * omega(q2, _gamma, 2, 2, d, exFac=True).imag
        tandelta = _num / _den
        delta = np.arctan2( _num, _den)
    # MF3
    elif np.array_equal(d, np.array([1., 1., 1.])):
        # Irrep. A_1
        #_z1 = zeta.Zp(q2, _gamma, 0, 0, d, 1., prec, verbose).real
        #_z2 = zeta.Zp(q2, _gamma, 2, 2, d, 1., prec, verbose).imag
        #_num = _gamma * np.sqrt(_pi3 * q2)
        #_den = (_z1 - ( 2. * np.sqrt(6./5.) / q2) * _z2 ).real
#        _den = omega(q2, _gamma, 0, 0, d, exFac=True).real -\
        _den = omega(q2, _gamma, 0, 0, d, exFac=True).real +\
               2 * np.sqrt(6) * omega(q2, _gamma, 2, 2, d, exFac=True).imag
        tandelta = _num / _den
        delta = np.arctan2( _num, _den)
    else:
        print("for the current vector d delta is not implemented")
    sindelta = np.sin(delta)**2
    return delta, tandelta, sindelta
