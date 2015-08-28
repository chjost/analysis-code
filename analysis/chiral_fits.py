################################################################################
#
# Author: Christopher Helmes (helmes@hiskp.uni-bonn.de)
# Date:   Februar 2015
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
# Function: ChiPT formulae for extracted scattering data 
#
# For informations on input parameters see the description of the function.
#
################################################################################

#system_imports

from scipy.optimize import leastsq
import scipy.stats
import math
import numpy as np
import analyze_fcts as af
from .plot import *

def chi_pt_cont(p,mpi):
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
  #mk = p[0], fk = p[1], meta = p[2], ren = p[3], lkk = p[4]
  mk, fk, meta, lkk = p
  ren = 
  meta = 
  # From pipi analysis
  rat_k = mk/fk
  pre = -(rat_k**2/(8*math.pi)) 
  args = np.array([mk/ren, meta/ren])
  log = np.log(np.square(args))
  mk_akk = []
  for x in mpi:
    coeff = np.array([2, 2./3. * x**2/(meta**2 - x**2),
                    2./27.*((20.*mk**2 -11.*x**2))])
    log_tmp = np.insert(log,1,np.log(np.square(x/ren)))
    prod = np.multiply(coeff,log_tmp)
    brac_in = prod[0]- prod[1] + prod[2] - 14./9. - 32.*(4*math.pi)**2*lkk
    brac_out = 1. + rat_k**2/(4.*math.pi)**2*brac_in
    mk_akk.append(pre*brac_out)
  return mk_akk

def nlo_kk():
  """ Afunction for calculating the NLO corrected Chiral Perturbation
  extrapolation to KK scattering.
  
  Args:
  Returns:
  """

