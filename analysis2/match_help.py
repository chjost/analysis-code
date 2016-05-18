# Linear helper functions
import numpy as np
from .fit_routines import fitting
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

def eval_lin(coeff,x):
  """
  Evaluate a linear function given its coefficients and the x- value to evaluate
  at
  """
  y = coeff[:,0]*x+coeff[:,1]
  return y

# quadratic helper functions
def ipol_quad(y1,y2,y3, x):
    """ Interpolate bootstrapsamples of data quadratically

        This function calculates a quadratic interpolation from 3 x values and
        bootstrapsamples of 3 yvalues like y = c0*x**2 + c1*x + c2
        
        Args:
            y1,y2,y3: the bootstrapsamples of the data points to interpolate.
            x: the x-values to use not bootstrapped with shape[0] = 3

        Returns:
            The interpolation coefficients c for all bootstrapsamples
            
    """
    # Use a bootstrapsamplewise quadratic interpolation 
    # result coefficients
    interpol = np.zeros((y1.shape[0],3))
    # loop over _bootstrapsamples
    for i,_b in enumerate(zip(y1,y2,y3)):
        # the known function values
        m = np.zeros((3,3)) 
        # Setting the coefficient matrix m with the x values
        #TODO: Have to automate setting somehow
        m[:,0] = np.square(x) 
        m[:,1] = np.asarray(x)
        m[:,2] = np.ones_like(x)
        # Solve the matrix wise problem with linalg
        coeff = np.linalg.solve(m,_b)
        if np.allclose(np.dot(m, coeff), _b) is False:
            print("solve failed in sample %d" % i)
        else:
            interpol[i] = coeff

    return interpol


def solve_quad(coeff, obs_match):
  """Function to match the quark mass using a quadratic interpolation (check if it
  is really interpolated)

  ipol_quad is used to determine the coefficients of the quadratic polynomial,
  the coefficeients are evaluated by finding the root mu_s^match to the second
  degree polynomial obs_match = c2*mu_s,match^2 + c1*mu_s,match + c0

  Parameters
  ----------
  y1, y2, y3 : the observed data values used to determine c2, c1 and c0
  x : the x-values to the observed data
  w1, w2, w3 : the weights for all fitranges
  obs_match : the observable to match to
  combine_all : Should all fit ranges for the y-values be combined?

  Returns
  the matched x-value (e.g. mu_s,match)
  """
  result = np.zeros(coeff.shape[0])
  coeff[:,-1] = np.subtract(coeff[:,-1],obs_match)
  print(coeff.shape)
  for b in range(result.shape[0]):
    tmp = np.roots(coeff[b])
    result[b]=tmp[1]
  return result

def eval_quad(coeff, x_match):
  y = coeff[:,0]*x_match**2+coeff[:,1]*x_match+coeff[:,2]
  return y

# fitting functions
# pass a general ndarray since the number of x and y points for a fit may vary
def fit_lin(y,x):
  f = lambda p,x: p[0]*x+p[1]
  start = [1.,1.]
  res, chi2, pval = fitting(f,x,y.T,start,add=None, correlated=True) 
  return res

def in_ival(x,a,b,):
  """ Decide if x is in [a,b]
  This function determines the mean values of x, a and b and checks their
  relation
  """
  try:
    _x = np.mean(x)
  except:
    _x = x
  try:
    _a = np.mean(a)
  except:
    _a = a
  try:
    _b = np.mean(b)
  except:
    _b = b
  if (_a <= _x) & (_x <= _b):
    return True
  else:
    return False

def choose_ival(x,mu):
  try:
    _x = np.mean(x)
  except:
    _x = x
  if _x <= mu[0]:
    i_dn=0
    i_up=1
  if mu[2] <= _x :
    i_dn = 1
    i_up = 2
  return i_dn, i_up



