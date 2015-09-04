################################################################################
#
# Author: Christopher Helmes 
# Date:   August 2015
#
# Copyright (C) 2015 Christopher Helmes
# 
# This program is free software: you can redistribute it and/or modify it under 
# the terms of the GNU General Public License as published by the Free Software 
# Foundation, either version 3 of the License, or (at your option) any later 
# version.
# 
# This program is distributed in the hope that it will be useful, but WITHOUT 
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS 
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with tmLQCD. If not, see <http://www.gnu.org/licenses/>.
#
################################################################################
#
# Function: Functions for linear and quadratic interpolation
#
# For informations on input parameters see the description of the function.
#
################################################################################

from scipy.optimize import leastsq
import scipy.stats
import math
import numpy as np
import analyze_fcts as af
import chiral_fits as chf
__all__=["ipol_lin","ipol_quad","eval_lin","eval_quad","err_prop_gauss",
         "eval_chi_pt_cont","sum_error_sym","sum_error_asym"]

def ipol_lin(y_boot,x):
    """ Interpolate bootstrapsamples of data linearly

        This function calculates a linear interpolation from 2 x values and
        bootstrapsamples of 2 yvalues y = c0*x+c1

        Args:
            y_boot: the bootstrapsamples of the data points to interpolate. Need
            shape[1] = 2
            x: the x-values to use not bootstrapped with shape[0] = 2

        Returns:
            The interpolation coefficients c for all bootstrapsamples
            
    """
    # Use a bootstrapsamplewise linear, newtonian interpolation 
    b_m = np.divide((y_boot[:,1]-y_boot[:,0]),(x[1]-x[0]))
    b_b = y_boot[:,0]-np.multiply(b_m,x[0])
    interpol = np.zeros_like(y_boot)
    interpol[:,0], interpol[:,1] = b_m, b_b
    return interpol

def ipol_quad(y_boot, x):
    """ Interpolate bootstrapsamples of data quadratically

        This function calculates a quadratic interpolation from 3 x values and
        bootstrapsamples of 3 yvalues like y = c0*x**2 + c1*x + c2
        
        Args:
            y_boot: the bootstrapsamples of the data points to interpolate. Need
            shape[1] = 3
            x: the x-values to use not bootstrapped with shape[0] = 3

        Returns:
            The interpolation coefficients c for all bootstrapsamples
            
    """
    # Use a bootstrapsamplewise quadratic interpolation 
    # result coefficients
    interpol = np.zeros_like(y_boot)
    # loop over bootstrapsamples
    for _b in range(y_boot.shape[0]):
        # the known function values
        y = y_boot[_b,:]
        m = np.zeros((y.shape[0],y.shape[0])) 
        mu_sq = np.square(x)
        # Setting the coefficient matrix m with the x values
        #TODO: Have to automate setting somehow
        m[:,0] = np.square(x) 
        m[:,1] = np.asarray(x)
        m[:,2] = np.ones_like(x)
        # Solve the matrix wise problem with linalg
        coeff = np.linalg.solve(m,y)
        if np.allclose(np.dot(m, coeff), y) is False:
            print("solve failed in sample %d" % _b)
        else:
            interpol[_b:] = coeff

    return interpol

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

def eval_quad(quad_coeff, x):
  """ Evaluates bootstrapsamples of coefficients at bootstraps of x for y =
      c0*x^2 + c1*x + c2

      Args:
          quad_coeff: (nb_samples,coeff) NumPy array
          x: Bootstrapsamples of xvalues
      Returns:
          eval_boot: The bootstrapsamples of y values
  """
  eval_boot = np.multiply(quad_coeff[:,0],np.square(x))+ np.multiply(quad_coeff[:,1],x)+quad_coeff[:,2]
  return eval_boot

def eval_chi_pt_cont(p,mpi):
  """ Continuum chiral perturbation formula for KK I = 1 scattering

  This function calculates the product MK*akk for a given set of input
  parameters. This is the continuum extrapolation formula for chi-pt
  
  Args:
    mpi: The pion mass 
    mk: The kaon mass
    fk: the kaon decay constant
    meta: the eta mass
    ren: the value of the chosen renormalization scale
    lkk: the counterterm involving the Gasser-Leutwyler coefficients

  Returns:
    mk*akk: The product of scattering length and Kaon mass at one set of
    parameters
  """
  lkk, Bms = p
  # try fit with physical values (MeV)
  fk = 160
  ren = fk
  #ren = 130.7
  #convert mpi to phys
  _mpi = chf.lat_to_phys(mpi)
  # Overall prefactor
  pre_out = (2.*Bms - _mpi**2)/(16*math.pi*fk**2)
  # inner prefactor
  pre_in = (2.*Bms + _mpi**2)/(32*math.pi**2*fk**2)
  # 3 coefficients to the logarithms
  coeff = np.array([2, 1./(2.*Bms/_mpi**2-1.), 20./9.*(Bms-_mpi**2)/(2.*Bms-_mpi**2)])
  # 3 logarithms
  log = np.log(np.array([(_mpi**2+2.*Bms)/ren**2,_mpi**2/ren**2,(_mpi**2+4.*Bms)/(3.*ren**2)]))
  # sum_i coeff[i]*log[i]
  prod = np.multiply(coeff,log)
  # decorated counterterm
  count = 14./9. + 32.*(4*math.pi)**2*lkk
  brac_in = prod[0] - prod[1] + prod[2] - count
  brac_out = 1. + pre_in*brac_in
  mk_akk = (pre_out*brac_out)
  return mk_akk

def err_prop_gauss(_a,_b,oper='div'):
  """ Evaluates gaussian propagated error without correlation for different
      operations
      Args:
          a,b: numpy arrays of the values of interest
          da,db: numpy arrays of the corresponding errors
          oper: flag to determine derived value (default: a/b)
      Returns:
          err_der: a numpy array of the derived errors
  """
  a,b = _a[:,0],_b[:,0]
  da,db = _a[:,1], _b[:,1]
  if oper == 'div':
    sq_1 = np.square(np.divide(da,b))
    tmp_prod = np.multiply(a,db)
    sq_2 = np.square(np.divide(tmp_prod,np.square(b)))
    err_der = np.sqrt(np.add(sq_1,sq_2))
  else:
    print("Not able to determine error")
    err_der = 0
  return err_der

def sum_error_sym(meas):
  """gets a n _mpi 3 numpy array holding a value, a statistical and a systematic
  uncertainty to be added in quadrature
  returns a n _mpi 2 array holding the value and the combined uncertainty for each
  row
  """
  print meas.shape[0]
  val_err = np.zeros((meas.shape[0],2))
  val_err[:,0] = meas[:,0]
  val_err[:,1] = np.sqrt(np.add(np.square(meas[:,1]),np.square(meas[:,2])))
  return val_err

def sum_error_asym(meas):
  """gets a n _mpi 4 numpy array holding a value, a statistical and two systematic
  uncertainties to be added in quadrature
  returns a n _mpi 2 array holding the value and the combined uncertainty for each
  row
  """
  print meas.shape[0]
  val_err = np.zeros((meas.shape[0],2))
  val_err[:,0] = meas[:,0]
  sys_err_sum =np.add( np.square(meas[:,2]), np.square(meas[:,3]) )
  val_err[:,1] = np.sqrt(np.add(np.square(meas[:,1]),sys_err_sum))
  return val_err

