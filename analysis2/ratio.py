"""
Functions for the ratio calculation.
"""

import numpy as np
from energies import WfromMass_lat

def twopoint_ratio(d1, d2, d3, shift=1, dE=None, useall=False, p2=0, L=24, irep="A1"):
    """Calculates a simple ratio of three data sets.

    Calculates d1(t)/d2(t).
    Assumes that all data sets are numpy arrays with two axis, the first being
    the bootstrap number and the second being the time.
    
    Args:
        d1, d2 The three data sets.

    Returns:
        The ratio, its mean and its standard deviation.
    """
    # create array from dimensions of the data
    rshape = list(d1.shape)
    ratio = np.zeros(rshape)
    for _s in range(rshape[0]):
        for _t in range(rshape[1]):
            # calculate ratio
            #ratio[_s,_t] = d1[_s,_t]/d2[_s,_t]-1.
            ratio[_s,_t] = d1[_s,_t]/d2[_s,_t]
    # TODO(CJ): test if the following give the same result
    # ratio = d1/(d2*d3)
    return ratio

def simple_ratio(d1, d2, d3, shift=1, dE=None, useall=False, p2=0, L=24, irep="A1"):
    """Calculates a simple ratio of three data sets.

    Calculates d1(t)/(d2(t)*d3(t)).
    Assumes that all data sets are numpy arrays with two axis, the first being
    the bootstrap number and the second being the time.
    
    Args:
        d1, d2, d3: The three data sets.

    Returns:
        The ratio, its mean and its standard deviation.
    """
    # create array from dimensions of the data
    rshape = list(d1.shape)
    ratio = np.zeros(rshape)
    for _s in range(rshape[0]):
        for _t in range(rshape[1]):
            # calculate ratio
            ratio[_s,_t] = d1[_s,_t]/(d2[_s,_t]*d3[_s,_t])
    # TODO(CJ): test if the following give the same result
    # ratio = d1/(d2*d3)
    return ratio

def ratio_shift(d1, d2, d3, shift=1, dE=None, useall=False, p2=0, L=24, irrep="A1"):
    """Calculates a simple ratio of three data sets.

    Calculates d1(t)/(d2(t)*d3(t) - d2(t+shift)*d3(t+shift)).
    Assumes that all data sets are numpy arrays with two axis, the first being
    the bootstrap number and the second being the time.
    
    Parameters
    ----------
    d1 : ndarray
        The numerator of the ratio, at least 3D.
    d2, d3 : ndarray
        The denominator of the ratio, at least 3D.
    shift : int, optional
        The number of slices that d2 and d3 are shifted.
    dE : {None, float}, optional
        The exponent of the weight used for d1.
    useall : bool, optional
        Use all correlators of d2 and d3 or just the first.
    usecomb : list of list of ints
        The combinations of d2 and d3 entries to use for d1 entries.

    Returns
    -------
    ndarray:
        The calculated ratio.
    """
    # create array from dimensions of the data
    rshape = list(d1.shape)
    if (d2.shape[1] - shift) > rshape[1]:
        raise RuntimeError("The ratio with shift %d cannot be computed" % shift)

    tmp2, tmp3 = get_states(d2, d3, p2, irrep, useall)
    ## fill temporary arrays for the denominator calculation
    #tmp2 = np.zeros_like(d2)
    #tmp3 = np.zeros_like(d3)
    #tmp2[...,0] = d2[...,0]
    #tmp3[...,0] = d3[...,0]
    ## fill with the expected value for the higher momentum modes
    ## TODO: differentiate different momenta and irreps
    #if useall:
    #    for i in range(1, d1.shape[2]):
    #        tmp2[...,i] = WfromMass_lat(tmp2[...,0], i, L)
    #        tmp3[...,i] = WfromMass_lat(tmp3[...,0], i, L)
    ## only use ground state
    #else:
    #    for i in range(1, d1.shape[2]):
    #        tmp2[...,i] = d2[...,0]
    #        tmp3[...,i] = d3[...,0]
    # if no weighting was used, don't use it here
    if dE is None:
        d = tmp2[:,:-shift]*tmp3[:,:-shift] - tmp2[:,shift:]*tmp3[:,shift:]
    else:
        tmp = np.zeros_like(d1)
        d = np.zeros_like(d1)
        # weight the two point functions
        for t in range(d2.shape[1]):
            weight = np.exp(dE*t)
            # the following is needed since the weight and data
            # axis have the first axis in common
            tmp[:,t] = ((tmp2[:,t]*tmp3[:,t]).T * weight).T
        # calculate the shift and reweight
        for t in range(d1.shape[1]):
            weight = np.exp(-dE*t)
            # the following is needed since the weight and data
            # axis have the first axis in common
            d[:,t] = ((tmp[:,t] - tmp[:,t+shift]).T * weight).T
    # calculate ratio
    ratio = d1 / d
    return ratio

def simple_ratio_subtract(d1, d2, d3, shift=1, dE=None, useall=False, p2=0,
        L=24, irep="A1"):
    """Calculates a simple ratio of three data sets, combining two time slices.

    Calculates [d1(t)-d1(t+1)]/[(d2(t)*d3(t))-(d2(t+1)*d3(t+1))].
    Assumes that all data sets are numpy arrays with two axis, the first being
    the bootstrap number and the second being the time. The time extend is
    reduced by one, as the ratio cannot be calculated on the last slice.
    
    Args:
        d1, d2, d3: The three data sets.

    Returns:
        The ratio, its mean and its standard deviation.
    """
    # create array from dimensions of the data
    rshape = list(d1.shape)
    rshape[1] -= shift
    ratio = np.zeros(rshape)
    for _s in range(rshape[0]):
        for _t in range(rshape[1]):
            # calculate ratio
            ratio[_s,_t] = (d1[_s,_t] - d1[_s,_t+shift]) / ((d2[_s,_t]*d3[_s,_t]) -
                           (d2[_s,_t+shift]*d3[_s,_t+shift]))
    return ratio

def ratio(d1, d2, d3, shift=1, dE=None, useall=False, p2=0, L=24, irep="A1"):
    """Calculates a ratio of three data sets, combining two time slices and
       the energy difference between the energy levels.

    Calculates [d1(t)-d1(t+1)]/[(d2(t)*d3(t))-(d2(t+1)*d3(t+1))].
    Assumes that all data sets are numpy arrays with two axis, the first being
    the bootstrap number and the second being the time. The time extend is
    reduced by one, as the ratio cannot be calculated on the last slice.
    
    Args:
        d1, d2, d3: The three data sets.
        dE: Energy difference between the the data sets d2 and d3.

    Returns:
        The ratio, its mean and its standard deviation.
    """
    # create array from dimensions of the data
    ratio = np.zeros((d1.shape[0], d1.shape[1]-1))
    for _s in range(d1.shape[0]):
        for _t in range(d1.shape[1]-1):
            # calculate numerator and denominator first
            num=(d1[_s,_t] - d1[_s,_t+1] * np.exp(dE * (_t+1))) * np.exp(-dE*_t)
            den=(d2[_s,_t] * d3[_s,_t] - d2[_s,_t+1] * d3[_s,_t+1] *
                 np.exp(dE*(_t+1))) * np.exp(-dE*_t)
            # calculate ratio
            ratio[_s,_t] = num/den
    return ratio

def get_states(mass1, mass2, d2, irrep, useall):
    """Calculate the expected energy for states for a given
    irrep and total momentum d2."""
    res1 = np.zeros_like(mass1)
    res2 = np.zeros_like(mass2)
    # calculate ground states, always needed
    if d2 == 0:
        res1[...,0] = mass1[...,0]
        res2[...,0] = mass2[...,0]
    elif d2 == 1:
        res1[...,0] = (WfromMass_lat(mass1[...,0], 1, L) + mass1[...,0])
        res2[...,0] = (WfromMass_lat(mass2[...,0], 1, L) + mass2[...,0])
    elif d2 == 2:
        res1[...,0] = (WfromMass_lat(mass1[...,0], 2, L) + mass1[...,0])
        res2[...,0] = (WfromMass_lat(mass2[...,0], 2, L) + mass2[...,0])
    else:
        raise ValueError("not implemented yet")

    # get all states
    if useall:
        if d2 == 0:
            for n in range(1, mass1.shape[-1]):
                res1[...,n] = 2.*WfromMass_lat(mass1[...,0], n, L)
            for n in range(1, mass2.shape[-1]):
                res2[...,n] = 2.*WfromMass_lat(mass2[...,0], n, L)
        elif d2 == 1:
            k1 = [0, 1, 2, 1, 2, 4, 3]
            k2 = [1, 2, 3, 4, 5, 5, 6]
            for n in range(1, mass1.shape[-1]):
                res1[...,n] = (WfromMass_lat(mass1[...,0], k1[n], L) + 
                               WfromMass_lat(mass1[...,0], k2[n], L))
            for n in range(1, mass2.shape[-1]):
                res2[...,n] = (WfromMass_lat(mass2[...,0], k1[n], L) + 
                               WfromMass_lat(mass2[...,0], k2[n], L))
        elif d2 == 2:
            k1 = [0, 1, 1, 2, 2, 1, 3, 2]
            k2 = [2, 1, 3, 2, 4, 5, 5, 6]
            for n in range(1, mass1.shape[-1]):
                res1[...,n] = (WfromMass_lat(mass1[...,0], k1[n], L) + 
                               WfromMass_lat(mass1[...,0], k2[n], L))
            for n in range(1, mass2.shape[-1]):
                res2[...,n] = (WfromMass_lat(mass2[...,0], k1[n], L) + 
                               WfromMass_lat(mass2[...,0], k2[n], L))
        
    # use only ground state
    else:
        for n in range(1, mass1.shape[-1]):
            res1[...,n] = res1[...,0]
        for n in range(1, mass2.shape[-1]):
            res2[...,n] = res2[...,0]

if __name__ == "main":
    pass
