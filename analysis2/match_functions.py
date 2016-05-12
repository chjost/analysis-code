import match_help as mh
import numpy as np

# used in evaluation
def calc_y_lin(y1,y2,x,x_match):
  """Function to evaluate observables at the quark mass given using a linear interpolation (check if it
  is really interpolated)

  This function calls ipol_lin, and afterwards solve_lin to get the interpolated
  quark mass value per fit range. Should be usable for other observables as
  well.

  Parameters
  ----------
  y1, y2 : the y data used in the matching
  x : the x-data corresponding to the 
  obs_match : the y data to match to
  Returns
  """
  coeff = mh.ipol_lin(y1, y2, x)
  result = mh.eval_lin(coeff,x_match)
  print(coeff.shape)
  print(result.shape)
  return result, coeff

def calc_y_quad(y1,y2,y3,x,x_match):
  coeff = mh.ipol_quad(y1,y2,y3,x)
  result = mh.eval_quad(coeff, x_match)
  return result, coeff

def calc_y_fit(y1,y2,y3,x,x_match):

  y=np.asarray((y1,y2,y3))
  coeff = mh.fit_lin(y,x)
  result = mh.eval_lin(coeff,x_match)
  return result, coeff

# used in matching  
def get_x_lin(y1,y2,x,obs_match):
  """Function to match the quark mass using a linear interpolation (check if it
  is really interpolated)

  This function calls ipol_lin, and afterwards solve_lin to get the interpolated
  quark mass value per fit range. Should be usable for other observables as
  well.

  Parameters
  ----------
  y1, y2 : the y data used in the matching
  x : the x-data corresponding to the 
  obs_match : the y data to match to
  Returns
  """
  coeff = mh.ipol_lin(y1, y2, x)
  # if obs_match is a FitResult, evaluate it at the coefficients
  # else solve c0*mu_s + c1 - obs_match = 0 for mu_s
  result = mh.solve_lin(coeff, obs_match)
  return result, coeff

def get_x_quad(y1,y2,y3,x,obs_match):
  coeff = mh.ipol_quad(y1,y2,y3,x)
  result = mh.solve_quad(coeff,obs_match)
  coeff[:,-1]=np.add(coeff[:,-1],obs_match)
  return result, coeff

# TODO: adapt for more than three masses
def get_x_fit(y1,y2,y3,x,obs_match):
  y=np.asarray((y1,y2,y3))
  coeff = mh.fit_lin(y,x)
  result = mh.solve_lin(coeff,obs_match)
  return result, coeff
