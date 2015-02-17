################################################################################
#
# Author: Benedikt Sauer and Christian Jost (jost@hiskp.uni-bonn.de)
# Date:   Februar 2015
#
# Copyright (C) 2015 Benedikt Sauer and Christian Jost
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
# Function: Solve the GEVP for a correlation function matrix. The original code
#           was written by B. Sauer and changed to work with the data structure
#           presented in corr_matrix.py.
#
# For informations on input parameters see the description of the function.
#
################################################################################

import numpy as np
import pandas as pd
import scipy.linalg as spla

def permutation_indices(data):
    """Sorts the data according to their value.

    This function is called by solve_gevp_gen to sort the eigenvalues according
    to their absolut values. This works on the assumption that the eigenvalues
    are real.

    Args:
        data: A list of values.

    Returns:
        An index list where the first entry corresponds to the index of largest
        value, the last entry is corresponds to the index of the smallest value.
    """
    return list(reversed(sorted(range(len(data)), key = data.__getitem__)))

def reorder_by_ev(ev1, ev2, B):
    """Creates an index list based on eigenvectors and the matrix B.

    Args:
        ev1, ev2: The eigenvectors to sort.
        B: The matrix to sort by.

    Returns:
        An index list where the first entry corresponds to the index of 'largest'
        value, the last entry is corresponds to the index of the 'smallest' value.
    """
    # TODO: res seems to be broken here, investigate!
    ev1_b = np.dot(np.array(B), ev1)

    dot_products = [
            np.abs(np.dot(e, ev1_b)) for e in ev2.transpose()
            ]

    res = []
    for m in dot_products:
        for candidate in permutation_indices(m):
            if not candidate in res:
                res.append(candidate)
                break

    return res

def solve_gevp_gen(a, t_0, sort_by_vectors=True):
    """Generator that returns the eigenvalues for t_0 -> t where t is in
       (t_0, t_max].
       
       Calculate the eigenvalues of the generalised eigenvalue problem using
       the scipy.linalg.eigh solver.

       At the moment only the eigenvalues are returned.
    
       Args:
           a: The matrix that is used.
           t_0: The timeslice of the inverted matrix.
           sort_by_vectors: Choose to sort the eigensystem by eigenvector or
               eigenvalues. Standard is using the vectors.

       Returns:
           A list of the eigenvalues and a list of the eigenvectors.
    """
    # B is the matrix at t=t_0
    B = a[t_0]
    # define the eigensystem solver function as a lambda function
    try:
        f = lambda A: spla.eigh(b=B, a=A)
    except LinAlgError:
        return

    # initialization
    eigenvectors = None
    count = 0

    # calculate the eigensystem for t in (t_0, T/2+1)
    for j in range(t_0 + 1, a.shape[0]):
        try:
            # calculate the eigensystems
            eigenvalues, new_eigenvectors = f(a[j])
            # initialize the new eigenvector array if not done already
            if eigenvectors is None:
                eigenvectors = np.zeros_like(new_eigenvectors)

            # The eigensystem can be sorted by the value of the eigenvalues or
            # by using the eigenvectors. 
            # Here the reordered index lists are created.
            if not sort_by_vectors:
                perm = permutation_indices(eigenvalues)
            else:
                perm = reorder_by_ev(new_eigenvectors, eigenvectors, B)
            # permutation of the lists
            eigenvectors = new_eigenvectors[:,perm]
            eigenvalues = eigenvalues[perm]
                
            count += 1

            yield eigenvalues, eigenvectors, j

        except (spla.LinAlgError, TypeError) as e:
            return

def calculate_gevp(m, t0=4, sort_by_vectors=True):
    """Solves the generalized eigenvalue problem of a correlation function
    matrix.

    The function takes a bootstrapped correlation function matrix and calculates
    the eigenvectors and eigenvalues of the matrix. The algorithm relies on the
    matrix being symmetric or hermitian. The matrix m should have 4 axis, as
    laid out in corr_matrix.py.
    The eigenvectors are calculated but not stored.

    Args:
        m: The data in a numpy array.
        t0: The timeslice used for the inversion.
        sort_by_vectors: Choose to sort the eigensystem by eigenvector or
               eigenvalues. Standard is using the vectors.

    Returns:
        A numpy array with three axis. The first axis is the bootstrap sample
        number, the second axis is the time, the third axis is the eigenvalue
        numbers. The time extend is the same as in original data, but the times
        up to and including t0 are filled with zeros.
    """
    # Initialize the eigenvalue array
    values_array = np.zeros((m.shape[0], m.shape[1], m.shape[2]))
    # iterate over the bootstrap samples
    for _samples in range(0, m.shape[0]):
        # iterate over the eigensystems
        for eigenvalues, _eigenvectors, _t in \
            solve_gevp_gen(m[_samples], t0, sort_by_vectors=sort_by_vectors):
            # save the eigenvalues to the array
            values_array[_samples, _t] = eigenvalues
    return values_array
