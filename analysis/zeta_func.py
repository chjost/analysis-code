################################################################################
#
# Author: Christian Jost (jost@hiskp.uni-bonn.de)
# Date:   Mai 2015
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
# Function: Some wrapper functions around Luescher's Zeta function for easier
#           usage.
#
# For informations on input parameters see the description of the function.
#
################################################################################

__all__ = ["Z"]

from .zeta import Z as _Z
import numpy as np

################################################################################
#
#                            Luescher's Zeta Function
#
# This is the ONLY function which should and needs to be called from outside.
#
# input: q2       : (IMPORTANT:) SQUARED scattering momentum fraction, ONLY 
#                   MANDATORY INPUT PARAMETER !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#        gamma    : Lorentz factor for moving frames, see e.g. arXiv:1011.5288
#        l        : orbital quantum number
#        m        : magnetic quantum number
#        d        : total three momentum of the system. (TBC: d can be used as 
#                   a twist angle as well. The correspondance is:
#                          d = -theta/pi     !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#        m_split  : coefficient when the masses of the scattering particles are
#                   different. It is given by: m_split = 1+(m_1^2-m_2^2)/E_cm^2
#                   where E_cm^2 is the interacting energy in the cms.
#        precision: precision of the calculation
#        verbose  : 0, no output on screen; 1, detailed output with convergence
#                   informations 
#        n        : Do not use, used for wrapping the function.
#
# return: The value of Luescher's Zeta function as a COMPLEX number.
#
################################################################################
def Z(q2, gamma = None, l = 0, m = 0, d = np.array([0., 0., 0.]), \
      m_split = 1, precision = 10e-6, verbose = 0):
    # check if more than one value for q2 was given by checking the type of q2
    if isinstance(q2, (tuple, list, np.ndarray)):
        _q2 = np.asarray(q2)
        if gamma == None:
            _gamma = np.ones(_q2.shape)
        else:
            _gamma = np.asarray(gamma)
        # check if q2 and gamma have the same number of entries
        assert _q2.size == _gamma.size
        res = np.zeros(_q2.shape, dtype=np.complex)

        for _i in range(_q2.size):
            res.flat[_i] = _Z(_q2.flat[_i], _gamma.flat[_i], l, m, d, m_split,
                              precision, verbose)
    else:
        if gamma == None:
            gamma = 1.
        res = _Z(q2, gamma, l, m, d, m_split, precision, verbose)
    return res
