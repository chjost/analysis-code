"""
Useful functions
"""

import numpy as np
import scipy.linalg as la
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
        #_mean = np.mean(data[select])
        _mean = data[select]
    else:
        _mean = np.asarray(mean)
    diff = axis_subtract(data,_mean,axis)
    var = np.nansum(np.square(diff), axis=axis) / data.shape[axis]
    std = np.sqrt(var)
    return _mean, std

def axis_subtract(data,mean,axis):
  diff = np.zeros_like(data)
  # check first dimension
  if axis > 0:
    if data.shape[0] == mean.shape[0]:
     for i,d in enumerate(data):
        diff[i] = d-mean[i]
  else:
    diff=data-mean
  return diff

def r0_mass(amps,ens,square=False):
  """Calculates the physical mass from pseudoscalar masses in lattice units

  Parameters:
  amps : array of masses
  """
  #dictionary of Sommer parameter (arxiv:1403.4504v3)
  r = {'A':5.31, 'B':5.77, 'D':7.60}
  if square is False:
    r0_m =np.multiply(r[ens],amps)
  else:
    r0_m =np.multiply(r[ens]**2,amps)
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
  print(amps[0])
  phys_mass =np.multiply(pre,amps) 
  return phys_mass 

def eig_decomp(mat):
  """ Compute eigendecomposition of a matrix and print it to screen.

  If the matrix is symmetric and positive definite, the eigenvalues and
  eigenvectors are calculated and printed to screen

  Parameters
  ----------
  mat : np.array, the matrix to decompose
  """
  #TODO: Check symmetry

  # decompose matrix into eigenvalues and eigenvectors
  l, v = la.eigh(mat)
  # TODO: sort eigenvalues starting with lowest eigenvalues
  # np.sort
  # TODO: sort eigenvectors as the same structure
  # print eigenvalues and corresponding eigenvalues to screen
  for nev in range(len(l)):
    print("\neigenpair %d:" % nev)
    print("lambda : %f " % l[nev])
    print("vec :")
    print(v[:,nev])

if __name__ == "__main__":
    pass

