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
