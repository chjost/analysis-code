"""
Class for matching procedures. It behaves similar to the FitResult class,
perhaps make it a derived class of some Metaobject
"""
import numpy as np
import interpol as ip

class MatchResult(object):
    """Class to process FitResults for Interpolation

    The data has a similar layout to FitResult
    """
    def __init__(self, obs_id):

      """Allocate objects for initiated instance
      Parameters
      ----------
        obs_id : Observable that is investigated
      """

      self.data = None
      self.weight = None
      self.error = None
      self.label = None
      self.obs_id = obs_id
    
    @classmethod
    def read()
    def save()
    def create_empty()
    def get_data()
    def add_data()


def match_quark_mass(obs0, obs1, obs2=None, meth=0, amu_s, obs_match):
    """Matches the quark mass to a given observable as an iterator

    Parameters
    ----------
    obs0, obs1, obs2: Observables at different strange quark masses 
    weight0, weight1, weight2: The corresponding weights as 1- or 2-dim numpy
        array
    meth: How to match: 0: linear interpolation (only two values)
                        1: linear fit
                        2: quadratic interpolation
    obs_match: Observable to match to (lattice units)
    """
    if meth > 2:
      raise ValueError("Method not implemented yet, meth < 3")
    # weights of different observables are given by their numpy products.
    # shape of observable weights
    wght_lyt = weight0.shape
    # for weights flatten arrays, multiply and reshape them 
    wght0_f = weight0.flatten() 
    wght1_f = weight1.flatten()
    wght_res_f = np.multiply(wght0_f, wght1_f)

    if obs2 != None:
      wght2_f = weight2.flatten()
      wght_res_f = np.multiply(wght_res_f, wght2_f)

    wght_res = wght_res_f.reshape(wght_lyt)

    # Depending on the method "meth" calculate coefficients first and evaluate
    # the root of the function to find obs_match
    # Method 0 only applicable for 2 observables
    if meth == 0:
          print("Using linear interpolation")
          if weight0.ndim < 2:
          for i in range(obs0.shape[-1]):
              boot_obs = np.column_stack(obs0[i],obs1[i])
              # coefficients of all bootstrapsamples
              boot_lin = ipol_lin(boot_obs, amu_s)



    yield (0, 0, i), result, needed, weight
    if meth == 1:
          print("Using linear fit")

    yield (0, 0, i), result, needed, weight
    if meth == 2:
          print("Using quadratic interpolation")
    yield (0, 0, i), result, needed, weight
          
