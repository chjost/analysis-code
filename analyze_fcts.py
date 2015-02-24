################################################################################
#
# Author: Christian Jost (jost@hiskp.uni-bonn.de)
# Date:   Februar 2015
#
# Copyright (C) 2015 Christian Jost
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
# Function: A set of functions to analyze correlation functions.
#
# For informations on input parameters see the description of the function.
#
################################################################################

import numpy as np

def average_corr_fct(data, nbcfg, T):
    """Average over the set of correlation functions.

    Args:
        data: The data to average over in a numpy array.
        axis: The numpy axis to average over. Usually the first axis

    Returns:
        A numpy array averaged over one axis.
    """
    average = np.zeros((T))
    for _t in range(T):
        average[int(_t)] = np.average(data[_t*nbcfg:(_t+1)*nbcfg])
    return average

### only copied, not yet functional
# ratio computation
def compute_ratio(C4, C2):
  print '\ncompute ratio:\n--------------\n' 
  ratio,  sigma, val = [], [], []
  for t in range(1, T/2-1):
    for b in range(0, bootstrapsize):
      a = (C4[t*bootstrapsize + b] - C4[(t+1)*bootstrapsize + b]) / \
          ((C2[t*bootstrapsize + b])**2 - (C2[(t+1)*bootstrapsize + b])**2)
      ratio.append(a)
    print t, mean(ratio[(t-1)*bootstrapsize:(t)*bootstrapsize]), \
                  std_error(ratio[(t-1)*bootstrapsize:(t)*bootstrapsize])
    sigma.append(std_error(ratio[(t-1)*bootstrapsize:(t)*bootstrapsize]))
    val.append(mean(ratio[(t-1)*bootstrapsize:(t)*bootstrapsize]))
  return ratio, sigma, val

# derivative
def compute_derivative(boot):
  print '\ncompute derivative:\n-------------------\n' 
  derv = np.empty([boot.shape[0], boot.shape[1]-1], dtype=float)
  # computing the derivative
  for b in range(0, boot.shape[0]):
    row = boot[b,:]
    for t in range(0, len(row)-1):
      derv[b, t] = row[t+1] - row[t]
  mean, err = mean_error_print(derv)
  return derv, mean, err

# mass computation
def compute_mass(boot):
  print '\ncompute mass:\n-------------\n' 
  # creating mass array from boot array
  mass = np.empty([boot.shape[0], boot.shape[1]-2], dtype=float)
  # computing the mass via formula
  for b in range(0, boot.shape[0]):
    row = boot[b,:]
    for t in range(1, len(row)-1):
      mass[b, t-1] = (row[t-1] + row[t+1])/(2.0*row[t])
  mass = np.arccosh(mass)
  mean, err = mean_error_print(mass)
  return mass, mean, err
