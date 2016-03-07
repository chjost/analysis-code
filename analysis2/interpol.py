"""
The class for interpolation of results
"""
from scipy.optimize import leastsq
import scipy.stats
import math
import numpy as np
import fit
import itertools as it
#import analyze_fcts as af
#import chiral_fits as chf


def match_lin(y1, y2, x, w1, w2, obs_match, w_match=None):
    """Function to match the quark mass using a linear interpolation (check if it
    is really interpolated)

    This function calls ipol_lin, and afterwards eval_lin to get the interpolated
    quark mass value per fit range. Should be usable for other observables as
    well.

    Parameters
    ----------
    Returns
    """

    # check number of weights
    if w1.shape != w2.shape:
      raise ValueError("Shapes of observables incompatible")
    #print y1.shape
    for i in range(w1.shape[-1]):
      # for each fit range interpolate the two bootstrapsamples
      coeff = ipol_lin(y1[:,i], y2[:,i], x)
      # if obs_match is a FitResult, evaluate it at the coefficients
      # else solve c0*mu_s + c1 - obs_match = 0 for mu_s
      result = solve_lin(coeff, obs_match)
      weight = np.multiply(w1[i],w2[i])
      needed = np.zeros_like(weight)

      yield (0,i), result, needed, weight

def evaluate_lin(y1, y2, x, w1, w2, obs_eval):
    """Function to match the quark mass using a linear interpolation (check if it
    is really interpolated)

    This function calls ipol_lin, and afterwards eval_lin to get the interpolated
    quark mass value per fit range. Should be usable for other observables as
    well.

    Parameters
    ----------

    par : The output parameter to place the result
    Returns
    """

    #check number of weights
    if w1.shape != w2.shape:
      raise ValueError("Shapes of observables incompatible")
    if combine_all:
      fr1 = range(w1.shape[-1])
      fr2 = range(w2.shape[-1])
      fr_tot = [fr1,fr2]
      i_yield = 0
      for cmb in it.product(*fr_tot):
        # for each fit range interpolate the two bootstrapsamples
        coeff = ipol_lin(y1[:,cmb[0]], y2[:,cmb[1]], x)
        # if obs_match is a FitResult, evaluate it at the coefficients
        # else solve c0*mu_s + c1 - obs_match = 0 for mu_s
        result = eval_lin(coeff, obs_eval)
        print result.shape
        weight = np.multiply(w1[cmb[0]],w2[cmb[1]])
        needed = np.zeros_like(weight)
      
        yield (0,i_yield), result, needed, weight
        i_yield += 1

    else:
      for i in range(w1.shape[-1]):
        # for each fit range interpolate the two bootstrapsamples
        coeff = ipol_lin(y1[:,i], y2[:,i], x)
        # if obs_match is a FitResult, evaluate it at the coefficients
        # else solve c0*mu_s + c1 - obs_match = 0 for mu_s
        result = eval_lin(coeff, obs_eval)
        print result.shape
        weight = np.multiply(w1[i],w2[i])
        needed = np.zeros_like(weight)
      
      yield (0,i), result, needed, weight
  
def ipol_lin(y1, y2, x):
    """ Interpolate bootstrapsamples of data linearly

        This function calculates a linear interpolation from 2 x values and
        bootstrapsamples of 2 yvalues y = c0*x+c1
        
        Parameters
        ----------
            y1, y2 : bootstrapsamples of the data points to interpolate.
            x: the x-values to use not bootstrapped with shape[0] = 2

        Returns:
            The interpolation coefficients c for all bootstrapsamples
            
    """
    #print(y1.shape)
    # Use a bootstrapsamplewise linear, newtonian interpolation 
    c0 = np.divide((y2-y1),(x[1]-x[0]))
    c1 = y1-np.multiply(c0,x[0])
    # save slope and y-intercept
    interpol = np.zeros((len(y1),2))
    if len(c0.shape) == 2:
      interpol[:,0], interpol[:,1] = np.ravel(c0), np.ravel(c1)
    else:
      interpol[:,0], interpol[:,1] = c0, c1
    return interpol

def interp_fk(name, mul, mus_match):
    """ This function reads values for mk from a textfile, filters them and
    interpolates them to a given valence strange quark mass.

    Parameters:
    -----------
      name : the filename of the fk data
             Source for fk: 
             /freedisk/urbach/delphyne/4flavour/b<beta>/mu<mu_l>/result<ens>.dat
      mul : light quark mass of the ensemble
      mus_match : the value to evaluate interploation

    Returns:
    --------
      fk_ipol, dfk_ipol : the value and the error of the interpolated decay constant
    """
    #print("Input is:")
    #print("name : %s, mu_l = %f, match = %f" % (name, mul, mus_match))
    # Read in textfile for fk_values (usually placed in the data folders together
    # numpy array holding 3 strange quark masses, 3 kaon masses and 3 values
    # fk with the correlators)
    #Source for fk: /freedisk/urbach/delphyne/4flavour/b<beta>/mu<mu_l>/result<ens>.dat
    OS_fk = np.loadtxt(name, skiprows=1,
        usecols=(1,2,3,4,5,6))
    # delete everything with wrong light quark mass
    OS_fk = OS_fk[np.logical_not(OS_fk[:,0]!= mul)]

    # filter the textfile for the right light quark mass
    # make numpy arrays with x and y values
    mus = OS_fk[:,1]
    fk  = OS_fk[:,4]
    dfk = OS_fk[:,5]
    # use np.interp to interpolate to given value
    fk_ipol = np.interp(mus_match, mus, fk)
    dfk_ipol = np.interp(mus_match, mus,dfk)
    return np.array((fk_ipol, dfk_ipol))
  
def solve_lin(lin_coeff, match):
  """ Solves linear equation for x value 

      Args:
          lin_coeff: (nb_samples,coeff) NumPy array
          match: value to match to
      Returns:
          eval_x: The bootstrapsamples of y values
  """
  eval_x = np.divide((match-lin_coeff[:,1]),lin_coeff[:,0])
  return eval_x

def eval_lin(lin_coeff, x):
  """ Evaluates bootstrapsamples of coefficients at bootstraps of x for y =
      m*x+b

      Args:
          lin_coeff: (nb_samples,coeff) NumPy array
          x: Bootstrapsamples of xvalues
      Returns:
          eval_boot: The bootstrapsamples of y values
  """
  eval_boot = np.multiply(lin_coeff[:,0],x)+lin_coeff[:,1]
  return eval_boot

#def ipol_quad(y_boot, x):
#    """ Interpolate bootstrapsamples of data quadratically
#
#        This function calculates a quadratic interpolation from 3 x values and
#        bootstrapsamples of 3 yvalues like y = c0*x**2 + c1*x + c2
#        
#        Args:
#            y_boot: the bootstrapsamples of the data points to interpolate. Need
#            shape[1] = 3
#            x: the x-values to use not bootstrapped with shape[0] = 3
#
#        Returns:
#            The interpolation coefficients c for all bootstrapsamples
#            
#    """
#    # Use a bootstrapsamplewise quadratic interpolation 
#    # result coefficients
#    interpol = np.zeros_like(y_boot)
#    # loop over bootstrapsamples
#    for _b in range(y_boot.shape[0]):
#        # the known function values
#        y = y_boot[_b,:]
#        m = np.zeros((y.shape[0],y.shape[0])) 
#        mu_sq = np.square(x)
#        # Setting the coefficient matrix m with the x values
#        #TODO: Have to automate setting somehow
#        m[:,0] = np.square(x) 
#        m[:,1] = np.asarray(x)
#        m[:,2] = np.ones_like(x)
#        # Solve the matrix wise problem with linalg
#        coeff = np.linalg.solve(m,y)
#        if np.allclose(np.dot(m, coeff), y) is False:
#            print("solve failed in sample %d" % _b)
#        else:
#            interpol[_b:] = coeff
#
#    return interpol
#
#
#def eval_quad(quad_coeff, x):
#  """ Evaluates bootstrapsamples of coefficients at bootstraps of x for y =
#      c0*x^2 + c1*x + c2
#
#      Args:
#          quad_coeff: (nb_samples,coeff) NumPy array
#          x: Bootstrapsamples of xvalues
#      Returns:
#          eval_boot: The bootstrapsamples of y values
#  """
#  eval_boot = np.multiply(quad_coeff[:,0],np.square(x))+ np.multiply(quad_coeff[:,1],x)+quad_coeff[:,2]
#  return eval_boot
#
#def eval_chi_pt_cont(p,mpi):
#  """ Continuum chiral perturbation formula for KK I = 1 scattering
#
#  This function calculates the product MK*akk for a given set of input
#  parameters. This is the continuum extrapolation formula for chi-pt
#  
#  Args:
#    mpi: The pion mass 
#    mk: The kaon mass
#    fk: the kaon decay constant
#    meta: the eta mass
#    ren: the value of the chosen renormalization scale
#    lkk: the counterterm involving the Gasser-Leutwyler coefficients
#
#  Returns:
#    mk*akk: The product of scattering length and Kaon mass at one set of
#    parameters
#  """
#  lkk, Bms = p
#  # try fit with physical values (MeV)
#  fk = 160
#  ren = fk
#  #ren = 130.7
#  #convert mpi to phys
#  _mpi = chf.lat_to_phys(mpi)
#  # Overall prefactor
#  pre_out = (2.*Bms - _mpi**2)/(16*math.pi*fk**2)
#  # inner prefactor
#  pre_in = (2.*Bms + _mpi**2)/(32*math.pi**2*fk**2)
#  # 3 coefficients to the logarithms
#  coeff = np.array([2, 1./(2.*Bms/_mpi**2-1.), 20./9.*(Bms-_mpi**2)/(2.*Bms-_mpi**2)])
#  # 3 logarithms
#  log = np.log(np.array([(_mpi**2+2.*Bms)/ren**2,_mpi**2/ren**2,(_mpi**2+4.*Bms)/(3.*ren**2)]))
#  # sum_i coeff[i]*log[i]
#  prod = np.multiply(coeff,log)
#  # decorated counterterm
#  count = 14./9. + 32.*(4*math.pi)**2*lkk
#  brac_in = prod[0] - prod[1] + prod[2] - count
#  brac_out = 1. + pre_in*brac_in
#  mk_akk = (pre_out*brac_out)
#  return mk_akk
#
#def err_prop_gauss(_a,_b,oper='div'):
#  """ Evaluates gaussian propagated error without correlation for different
#      operations
#      Args:
#          a,b: numpy arrays of the values of interest
#          da,db: numpy arrays of the corresponding errors
#          oper: flag to determine derived value (default: a/b)
#      Returns:
#          err_der: a numpy array of the derived errors
#  """
#  a,b = _a[:,0],_b[:,0]
#  da,db = _a[:,1], _b[:,1]
#  if oper == 'div':
#    sq_1 = np.square(np.divide(da,b))
#    tmp_prod = np.multiply(a,db)
#    sq_2 = np.square(np.divide(tmp_prod,np.square(b)))
#    err_der = np.sqrt(np.add(sq_1,sq_2))
#  else:
#    print("Not able to determine error")
#    err_der = 0
#  return err_der
#
#def sum_error_sym(meas):
#  """gets a n _mpi 3 numpy array holding a value, a statistical and a systematic
#  uncertainty to be added in quadrature
#  returns a n _mpi 2 array holding the value and the combined uncertainty for each
#  row
#  """
#  print meas.shape[0]
#  val_err = np.zeros((meas.shape[0],2))
#  val_err[:,0] = meas[:,0]
#  val_err[:,1] = np.sqrt(np.add(np.square(meas[:,1]),np.square(meas[:,2])))
#  return val_err
#
#def sum_error_asym(meas):
#  """gets a n _mpi 4 numpy array holding a value, a statistical and two systematic
#  uncertainties to be added in quadrature
#  returns a n _mpi 2 array holding the value and the combined uncertainty for each
#  row
#  """
#  print meas.shape[0]
#  val_err = np.zeros((meas.shape[0],2))
#  val_err[:,0] = meas[:,0]
#  sys_err_sum =np.add( np.square(meas[:,2]), np.square(meas[:,3]) )
#  val_err[:,1] = np.sqrt(np.add(np.square(meas[:,1]),sys_err_sum))
#  return val_err
#
