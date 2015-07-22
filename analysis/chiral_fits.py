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
import numpy as np
import analyze_fcts as af
from .plot import *

def nlo_kk():
  """ Afunction for calculating the NLO corrected Chiral Perturbation
  extrapolation to KK scattering.
  
  Args:
  Returns:
  """

