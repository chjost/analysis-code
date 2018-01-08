"""
Routines for calculating covariance matrices 
"""

import os
import itertools
from scipy.optimize import leastsq
import scipy.stats
import numpy as np
def mute(cov):
    _cov = np.zeros_like(cov)
    for i in range((_cov.shape[0]-6)/3):
      _cov[3*i:3*i+3,3*i:3*i+3]=cov[3*i:3*i+3,3*i:3*i+3]
    for i in range(_cov.shape[0]-6,_cov.shape[0]):
      _cov[i,i]=cov[i,i]
    return _cov
def custom_cov(y, correlated=False, debug=0):
    """ compute custom covariance matrix of y-data depending on correlation
    flag
    
    Parameters
    ----------
    y: y-data object, array or list
    correlated: bool, if True return full covariance matrix else return diagonal

    Returns
    -------
    cov: 2d array,the calculated covariance matrix
    """
    if isinstance(y, list):
        _y = np.asarray(y)
    else:
        _y = y
    if debug > 1:
        print("\nIn custom cov:\ny_data has shape:")
        print(_y.shape)
    if _y.ndim != 2:
        raise ValueError("\nIn custom cov:\n y-data is not 2d")
    if correlated is not True:
        _cov = np.diag(np.diagonal(np.cov(_y)))
    else:
        _cov = np.cov(_y)
    if debug > 2:
        print("\nIn custom cov:\ncovariance matrix is:")
        print(_cov)
    return _cov
