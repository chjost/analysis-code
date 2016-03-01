"""
Useful functions
"""

import numpy as np
import itertools

def loop_iterator(ranges):
    items = [[x for x in range(n)] for n in ranges]
    for it in itertools.product(*items):
        yield it

def mean_std(data, axis=0, mean=None):
    """Calculate the mean and standard deviation using
    bootstrap sample 0 as mean.
    
    Parameters
    ----------
    data : ndarray
        The data.
    axis : int, optional
        The axis along which the mean and std is taken.
    mean : float, ndarray, tuplt, list, optional
        Used as mean if given.
        
    Returns
    -------
    ndarray
        The mean of the data
    ndarray
        The standard deviation of the data
    """
    if mean is None:
        select = [slice(None),] * data.ndim
        select[axis] = 0
        _mean = data[select]
    else:
        _mean = mean
    var = np.sum(np.square(data - _mean), axis=axis) / data.shape[axis]
    std = np.sqrt(var)
    return _mean, std

if __name__ == "__main__":
    pass

