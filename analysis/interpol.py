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
import numpy as np
import analyze_fcts as af

__all__=["ipol_lin","ipol_quad"]

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
