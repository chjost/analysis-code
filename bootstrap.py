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
# Function: These functions deal with the bootstrapping of data
#
# For informations on input parameters see the description of the function.
#
################################################################################

import math
import numpy as np

def bootstrap(source, nbsamples):
    """Bootstraping of data.

    Creates nbsamples bootstrap samples of source.

    Args:
        source: Data on which the bootstrap samples are created.
        nbsamples: Number of bootstrap samples created.

    Returns:
        A numpy array with the bootstrap samples.
    """
    # seed the random number generator
    # the seed is hardcoded to be able to recreate the samples
    np.random.seed(125013)
    # initialize the bootstrapsamples to 0.
    boot = np.empty(nbsamples, dtype=float)
    # the first entry is the average over the original data
    boot[0] = np.mean(source)
    # create the rest of the bootstrap samples
    for _i in range(1, nbsamples):
        _rnd = np.random.randint(0, len(source)-1, size=len(source))
        _sum = 0.
        for _r in range(0, len(source)):
            _sum += source[_rnd[_r]]
        boot[_i] = _sum / float(len(source))
    return boot

def sym_and_boot(source, T, nbcfg, nbsamples = 1000):
    """Symmetrizes and boostraps correlation functions.

    Symmetrizes the correlation functions given in source and creates bootstrap
    samples.

    Args:
        source: List with the correlation functions.
        T: time extent of the lattice.
        nbcfg: number of configurations in source.
        nbsamples: Number of bootstrap samples created.

    Returns:
        A two dimensional numpy array containing the bootstrap samples. The
        first axis is the bootstrap number, the second axis is the time index.
        The time extent is reduced to T/2+1 due to the symmetrization.
    """
    # the first timslice is not symmetrized
    boot = bootstrap(source, nbsamples)
    for _t in range(1, int(T/2)):
        # symmetrize the correlation function
        _symm = []
        for _x, _y in zip(source[_t * nbcfg:(_t+1) * nbcfg],
                          source[(T-_t) * nbcfg:(T-_t+1) * nbcfg]):
            _symm.append((_x + _y) / 2.)
        # bootstrap the timeslice
        boot = np.c_[boot, bootstrap(_symm, nbsamples)]
    # the timeslice at t = T/2 is not symmetrized
    boot = np.c_[boot, bootstrap(source[int(T/2) * nbcfg: (int(T/2)+1) * nbcfg],
                                 nbsamples)]
    return boot
