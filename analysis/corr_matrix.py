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
# Function: Implements functions for manipulating correlation function matrices.
#
# For informations on input parameters see the description of the function.
#
################################################################################

__all__ = ["create_corr_matrix", "calculate_gevp", "shift_corr_matrix"]

import os
import numpy as np
import scipy.linalg as spla
import input_output as io
import bootstrap

def shift_corr_matrix(cmat, dt, dE=None, axis=1):
    """Weight-shift the correlation function matrix.

    This is based on the paper by Dudek et al, Phys.Rev. D86, 034031 (2012).
    First the matrix is weighted by exp(dE*t) on every timeslice and then shifted.
    If dE is not given, the matrix is only shifted

    Args:
        cmat: correlation function matrix
        dt: number of timeslices to shift
        dE: the energy for the weighting
        axis: axis along which the shift is performed

    Returns:
        shifted correlation function matrix
    """
    # check whether axis is in the number of dimension
    ndim = cmat.ndim
    if axis > ndim or axis < 0:
        print("axis not in range of the dimensions. Not doing anything")
        return cmat

    # calculate shape of the new matrix
    cmshape = np.asarray(cmat.shape)
    cmshape[axis] -= dt

    # weighting of the matrix
    if dE:
        for t in xrange(cmat.shape[axis]):
          # TODO(CJ): Think of a way to do this more general
          cmat[:,t] = np.exp(dE*t) * cmat[:,t]

    # if dt is zero, don't shift
    if dt is 0:
        return cmat

    # create the new array
    cm = np.zeros(cmshape)

    # fill the new array
    for i in xrange(cmshape[axis]):
        cm[:,i] = cmat[:,i] - cmat[:,i+dt]

    # return shifted matrix
    return cm

def create_corr_matrix(nbsamples, filepath, filestring, filesuffix=".dat",
                       column=(1,), verbose=0):
    """Creates a correlation function matrix.

    Reads different correlation functions and inserts them into a matrix. The
    matrix is filled row majored, the correlation functions matrix is stored
    column majored. It is assumed that the matrix is symmetric, the off
    diagonal elements are symmetrized.
    WARNING: Up to now a maximum matrix size of 20x20 operators is implemented.

    Args:
        nbsamples: Number of bootstrap samples created.
        filepath: The path to the data, including the fileprefix.
        filestring: A list of the changing parts of the filenames. The length
                    of the list gives the size of the matrix.
        filesuffix: The suffix of the data files.
        column: The column of the input file to be read. The same column is
                read from every file!
        verbose: Changes the amount of information printed.

    Returns:
        A numpy array with four axis. The first axis is the column of operators,
        the second axis is the row of operators, the third axis is the number
        of the bootstrap sample and the last axis is the time dependence.
        The second returned argument is the time extend of the correlation
        function matrix, that is (T/2)+1, where T is the length of the original
        correlation function in time.
    """
    # calculate the matrix size based on the number of elements of filestring
    # at the moment up to 20x20 matrices are supported
    _nbops = 0
    for _i in range(1, 20):
        if len(filestring) == _i*_i:
            _nbops = _i
            break
    # if the size could not be determined then return
    if _nbops == 0:
        print("ERROR: size of the correlation matrix could not be determined")
        os.sys.exit(-4)
    # Treat first element differently so we can create an array of the correct
    # size. This allows also to check whether the other files have the same
    # number of configurations and the same time extent.
    _name = "".join((filepath, filestring[0], filesuffix))
    if verbose:
        print("filename " + _name)
    _data1 = io.read_data_ascii(_name, column, verbose)
    _nbcfg = _data1.shape[0]
    _T = _data1.shape[1]
    _boot1 = bootstrap.sym_and_boot(_data1, nbsamples)
    # create correlation function matrix
    corr_mat = np.zeros(_boot1.shape[0:2] + (_nbops,) * 2)
    corr_mat[:,:,0,0] = _boot1
    # read in all other correlation functions, bootstrap them and write them to
    # the numpy array
    for _nb, _sub in enumerate(filestring[1:], start=1):
        # generate filename
        _name = "".join((filepath, _sub, filesuffix))
        if verbose:
            print("filename " + _name)
        # read in data
        _data = io.read_data_ascii(_name, column)
        _nbcfg1 = _data.shape[0]
        _T1 = _data.shape[1]
        # check if size is the same as the first operator
        if _nbcfg != _nbcfg1 or _T != _T1:
            print("ERROR while reading file " + _name)
            print("\tnumber of configurations or time extent is wrong")
        else:
            _boot = bootstrap.sym_and_boot(_data, nbsamples)
            corr_mat[:, :, int(_nb/_nbops), int(_nb%_nbops)] = _boot 

    corr_mat_symm = np.zeros_like(corr_mat)
    for _s in range(0, nbsamples):
        for _t in range(0, int(_T1/2)+1):
            corr_mat_symm[_s, _t] = (corr_mat[_s, _t] + corr_mat[_s, _t].T) / 2.
    return corr_mat_symm

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

    Creates an index list where the first entry corresponds to the index of the
    eigenvector ev2 with largest overlap to the first eigenvector of ev1. The
    last index corresponds to the index of the eigenvector ev2 with largest
    overlap to the last eigenvector ev2 that did not have a large largest
    overlap with a previous eigenvector ev1.  
    WARNING: If more than one eigenvector ev2 has the (numerically) same
    largest overlap with some eigenvector ev1, the behaviour is not specified.

    Args:
        ev1: The eigenvectors to sort by, assumes these are already sorted.
        ev2: The eigenvectors to sort.
        B: The matrix to sort by, used for normalization.

    Returns:
        An index list of the sorted eigenvectors ev2.
    """
    # Calculate all scalar products of the eigenvectors ev1 and ev2. The matrix
    # B is used for the normalization, due to the SciPy eigh solver used. The
    # absolute value is needed because the eigenvectors can also be
    # antiparallel.
    # WARNING: It might be possible that more than one eigenvector ev2 has the
    # (numerically) same largest overlap with some eigenvector ev1. In this
    # case the behaviour is not specified.
    ev1_b = np.dot(np.array(B), ev1)
    dot_products = [ np.abs(np.dot(e, ev1_b)) for e in ev2.transpose() ]
    # Sort the eigenvectors ev2 according to overlap with eigenvectors ev1 by
    # using the scalar product. This assumes that ev1 was already sorted.
    res = []
    # this iterates through the eigenvectors ev1 and looks for the greatest
    # overlap
    for m in dot_products:
        # sort the eigenvectors ev2 according to their overlap with ev1
        for candidate in permutation_indices(m):
            # add each eigenvector ev2 only once to the index list and break
            # when a vector has been added so that only one eigenvector ev2
            # is added for each eigenvector ev1
            if not candidate in res:
                res.append(candidate)
                break
    return res

def solve_gevp_gen(a, t_0):
    """Generator that returns the eigenvalues for t_0 -> t where t is in
       (t_0, t_max].
       
       Calculate the eigenvalues of the generalised eigenvalue problem using
       the scipy.linalg.eigh solver.
    
       Args:
           a: The matrix that is used.
           t_0: The timeslice of the inverted matrix.

       Returns:
           A list of the eigenvalues, a list of the eigenvectors and the time
           slice on which the eigensystem was calculated.
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
            # initialize the new eigenvector array if not done already and
            # sort the eigenvectors and eigenvalues according the absolute
            # value of the eigenvalues
            if eigenvectors is None:
                eigenvectors = np.zeros_like(new_eigenvectors)
                perm = permutation_indices(eigenvalues)
            # Sort the eigensystem by the eigenvectors (for details see the
            # function description). The matrix B is used for the normalization
            # due to the normalization of the eigenvectors return by the eigh
            # solver.
            else:
                perm = reorder_by_ev(new_eigenvectors, eigenvectors, B)
            # permutation of the lists
            eigenvectors = new_eigenvectors[:,perm]
            eigenvalues = eigenvalues[perm]
                
            count += 1

            yield eigenvalues, eigenvectors, j

        except (spla.LinAlgError, TypeError) as e:
            return

def calculate_gevp(m, t0=1):
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

    Returns:
        A numpy array with three axis. The first axis is the bootstrap sample
        number, the second axis is the time, the third axis is the eigenvalue
        numbers. The time extend is the same as in original data, but the times
        up to t0 are filled with zeros.
    """
    # Initialize the eigenvalue array
    values_array = np.zeros((m.shape[0], m.shape[1], m.shape[2]))
    # iterate over the bootstrap samples
    for _samples in range(0, m.shape[0]):
        # iterate over the eigensystems
        for eigenvalues, _eigenvectors, _t in solve_gevp_gen(m[_samples], t0):
            # save the eigenvalues to the array
            values_array[_samples, _t] = eigenvalues
    # set the eigenvalues for t=t0 to 1.0
    values_array[:,t0] = 1.0
    return values_array
