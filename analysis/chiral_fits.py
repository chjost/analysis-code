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
def lat_to_phys(q_lat):
  # Lattice spacing for a ensembles:
  a =0.5/5.31
  hc = 197.33
  q_phys = np.multiply(q_lat,(hc/a))
  return q_phys

def phys_to_lat(q_phys):
  # Lattice spacing for a ensembles:
  a =0.5/5.31
  hc = 197.33
  q_lat = np.multiply(q_phys,(a/hc))
  return q_lat

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
  lkk, Bms = p
  # try fit with physical values (MeV)
  fk = 160
  ren = fk
  #ren = 130.7
  mk_akk = []
  #convert mpi to phys
  _mpi = lat_to_phys(mpi)
  for i,x in enumerate(_mpi):
    # Overall prefactor
    pre_out = (2.*Bms - x**2)/(16*math.pi*fk**2)
    # inner prefactor
    pre_in = (2.*Bms + x**2)/(32*math.pi**2*fk**2)
    # 3 coefficients to the logarithms
    coeff = np.array([2, 1./(2.*Bms/x**2-1.), 20./9.*(Bms-x**2)/(2.*Bms-x**2)])
    # 3 logarithms
    log = np.log(np.array([(x**2+2.*Bms)/ren**2,x**2/ren**2,(x**2+4.*Bms)/(3.*ren**2)]))
    # sum_i coeff[i]*log[i]
    prod = np.multiply(coeff,log)
    # decorated counterterm
    count = 14./9. + 32.*(4*math.pi)**2*lkk
    brac_in = prod[0] - prod[1] + prod[2] - count
    brac_out = 1. + pre_in*brac_in
    mk_akk.append(pre_out*brac_out)
  return mk_akk

def nlo_kk():
  """ Afunction for calculating the NLO corrected Chiral Perturbation
  extrapolation to KK scattering.
  
  Args:
  Returns:
  """

