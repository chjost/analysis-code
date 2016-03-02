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
    var = np.nansum(np.square(data - _mean), axis=axis) / data.shape[axis]
    std = np.sqrt(var)
    return _mean, std

def r0_mass(amps,ens):
  """Calculates the physical mass from pseudoscalar masses in lattice units

  Parameters:
  amps : array of masses
  """
  #dictionary of Sommer parameter (arxiv:1403.4504v3)
  r = {'A':5.31, 'B':5.77, 'D':7.60}
  r0_m =np.multiply(r[ens],amps) 
  return r0_m

def physical_mass(amps,ens):
  """Calculates the physical mass from pseudoscalar masses in lattice units

  Parameters:
  amps : array of masses
  """
  hbar_c = 0.197327
  #dictionary of lattice spacings in fm (arxiv:1403.4504v3)
  a = {'A':0.0885, 'B':0.0815, 'D':0.0619}
  pre = hbar_c / a[ens]
  print amps[0]
  phys_mass =np.multiply(pre,amps) 
  return phys_mass 

if __name__ == "__main__":
    pass

