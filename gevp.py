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

def solve_gevp_gen(a, t_0, sort_by_vectors=15, **kwargs):
    """Generator that returns the eigenvalues for t_0 -> t where t is in
       (t_0, t_max].
       
       Calculate the eigenvalues of the generalised eigenvalue problem using
       the scipy.linalg.eigh solver.

       At the moment only the eigenvalues are returned.
    
       Args:
           a: The matrix that is used.
           t_0: The timeslice of the inverted matrix.
           sort_by_vectors:??
           kwargs: For currying.

       Returns:
           A list of the eigenvalues.
    """
    B = a[t_0]
    # TODO(CJ): originally written to support multiple algorithms, not needed
    # any more. Change to one hardcoded algorithm.
    try:
        f = spla.eigh(b=B, **kwargs)
    except TypeError:
        # If the function doesn't do currying, implement that here
        f = lambda A: spla.eigh(b=B, a=A)
    except LinAlgError:
        return

    eigenvectors = None
    count = 0

    for j in range(t_0 + 1, len(a[:,0,0])):
        try:
            eigenvalues, new_eigenvectors = f(a[j])
            
            if eigenvectors is None:
                eigenvectors = np.zeros_like(new_eigenvectors)

            if j < sort_by_vectors:
                # TODO Sortieren nach Eigenwert
                perm = permutation_indices(eigenvalues)
            else:
                perm = reorder_by_ev(new_eigenvectors, eigenvectors, B)

            eigenvectors = new_eigenvectors[:,perm]
            eigenvalues = eigenvalues[perm]
                
            count += 1

            yield eigenvalues, eigenvectors

        except (spla.LinAlgError, TypeError) as e:
            #import traceback
            #traceback.print_exc()
            return

def calculate_gevp(m, sort_by_vectors=99, max_t0=4, **kwargs):
    res_values = {}
    for i in range(max_t0 + 1):
        ev = []
        for eigenvalues, _eigenvectors in \
                solve_gevp_gen(m, i, sort_by_vectors=sort_by_vectors, **kwargs):
            ev.append(eigenvalues)

        if len(ev):
            res_values[i] = pd.DataFrame(ev)

    return pd.concat(res_values)
