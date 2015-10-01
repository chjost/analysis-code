"""
Functions for the ratio calculation.
"""

import numpy as np

def simple_ratio(d1, d2, d3):
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

def ratio_shift(d1, d2, d3, shift=1):
    """Calculates a simple ratio of three data sets.

    Calculates d1(t)/(d2(t)*d3(t) - d2(t+shift)*d3(t+shift)).
    Assumes that all data sets are numpy arrays with two axis, the first being
    the bootstrap number and the second being the time.
    
    Args:
        d1, d2, d3: The three data sets.

    Returns:
        The ratio, its mean and its standard deviation.
    """
    # create array from dimensions of the data
    rshape = list(d1.shape)
    if (d2.shape[1] - shift) > rshape[1]:
        raise RuntimeError("The ratio with shift %d cannot be computed" % shift)
    ratio = np.zeros(rshape)
    for _s in range(rshape[0]):
        for _t in range(rshape[1]):
            # calculate ratio
            ratio[_s,_t] = d1[_s,_t] / (d2[_s,_t]*d3[_s,_t] -
                d2[_s,_t+shift]*d3[_s,_t+shift])
    return ratio

def simple_ratio_subtract(d1, d2, d3):
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
    rshape[1] -= 1
    ratio = np.zeros(rshape)
    for _s in range(rshape[0]):
        for _t in range(rshape[1]):
            # calculate ratio
            ratio[_s,_t] = (d1[_s,_t] - d1[_s,_t+1]) / ((d2[_s,_t]*d3[_s,_t]) -
                            (d2[_s,_t+1]*d3[_s,_t+1]))
    return ratio

def ratio(d1, d2, d3, dE):
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

if __name__ == "main":
    pass
