"""
Functions for the GEVP and similar functions.
"""

import numpy as np
import scipy.linalg as spla
import itertools

def gevp_shift_1(data, dt, dE=None, debug=0):
    """Weight-shift the correlation function matrix.

    This is based on the paper by Dudek et al, Phys.Rev. D86, 034031 (2012).
    First the matrix is weighted by exp(dE*t) on every timeslice and
    then shifted. If dE is not given, the matrix is only shifted.

    Parameters
    ----------
    data : ndarray
        The data to shift
    dt : int
        The amount of shift.
    dE : {None, float}, optional
        The exponent of the weight.
    axis : int, optional
        The axis along witch the shift is done.
    debug : int, optional
        Amount of info printed.

    Returns
    -------
    ndarray
        The shifted array.
    """
    # if dt is zero, don't shift
    if dt is 0:
        return data

    # calculate shape of the new matrix
    dshape = np.asarray(data.shape)
    dshape[1] -= dt

    # weighting of the matrix
    if dE is not None:
        _data = np.zeros_like(data)
        for t in range(_data.shape[1]):
            weight = np.exp(dE*t)
            # the following is needed since the weight and data
            # axis have the first axis in common
            _data[:,t] = (data[:,t].T * weight).T
            #if isinstance(dE, (int, float)):
            #    _data[:,t] = data[:,t] * weight
            #else:
            #    for b in range(dshape[0]):
            #        _data[b,t] = data[b,t] * weight[b]
    else:
        _data = np.copy(data)

    # create the new array
    sdata = np.zeros(dshape)

    # fill the new array
    for i in range(dshape[1]):
        sdata[:,i] = _data[:,i] - _data[:,i+dt]

    # reweighting of the matrix to cancel
    if dE is not None:
        for t in range(dshape[1]):
            weight = np.exp(-dE*t)
            # the following is needed since the weight and data
            # axis have the first axis in common
            sdata[:,t] = (sdata[:,t].T * weight).T
            #if isinstance(dE, (int, float)):
            #    sdata[:,t] = sdata[:,t] * weight
            #else:
            #    for b in range(dshape[0]):
            #        sdata[b,t] = sdata[b,t] * weight[b]
    # return shifted matrix
    return sdata

def gevp_shift_2(data, dt, dE, debug=0):
    """Weight-shift the correlation function matrix.

    This is based on the paper by Feng et al, Phys.Rev. D91, 054504 (2015).
    The shift is defined in equation (30).

    Parameters
    ----------
    data : ndarray
        The data to shift
    dt : int
        The amount of shift.
    dE : float
        The factor of the weight.
    debug : int, optional
        Amount of info printed.

    Returns
    -------
    ndarray
        The shifted array.
    """
    if dt == 0:
        return data

    # calculate shape of the new matrix
    dshape = np.asarray(data.shape)
    dshape[1] -= dt
    T = data.shape[1]

    # if dt is zero, don't shift
    if dt is 0:
        return data

    # create the new array
    sdata = np.zeros(dshape)

    # fill the new array
    for i in range(dshape[1]):
        sdata[:,i] = data[:,i] - data[:,i+dt] * (np.cosh(dE*(T-i)) /
            np.cosh(dE*(T-i+dt)))

    # return shifted matrix
    return sdata

#####
# Everything below coded by Benedikt Sauer
#####

def permutation_indices(data):
    """Sorts the data according to their value.

    This function is called by solve_gevp_gen to sort the eigenvalues
    according to their absolut values. This works on the assumption
    that the eigenvalues are real.

    Parameters
    ----------
    data : ndarray
        The data.

    Returns
    -------
    list of int
        A list where the first entry corresponds to the index of
        largest value, the last entry is corresponds to the index of
        the smallest value.
    """
    return list(reversed(sorted(range(len(data)), key = data.__getitem__)))

def reorder_by_ev(ev1, ev2, B, verbose=False):
    """Creates an index list based on eigenvectors and the matrix B.

    Creates an index list where the first entry corresponds to the
    index of the eigenvector ev2 with largest overlap to the first
    eigenvector of ev1. The last index corresponds to the index of the
    eigenvector ev2 with largest overlap to the last eigenvector ev2
    that did not have a large largest overlap with a previous
    eigenvector ev1.
    WARNING: If more than one eigenvector ev2 has the (numerically)
    same largest overlap with some eigenvector ev1, the behaviour is
    not specified.

    Parameters
    ----------
    ev1 : ndarray
        The eigenvectors to sort by, assumes they are already sorted.
    ev2 : ndarray
        The eigenvectors to sort.
    B : ndarray
        The matrix used during sorting, needed for normalization.

    Returns
    -------
    list
        The indices of the sorted eigenvectors.
    """
    # Calculate all scalar products of the eigenvectors ev1 and ev2. The matrix
    # B is used for the normalization, due to the SciPy eigh solver used. The
    # absolute value is needed because the eigenvectors can also be
    # antiparallel.
    # WARNING: It might be possible that more than one eigenvector ev2 has the
    # (numerically) same largest overlap with some eigenvector ev1. In this
    # case the behaviour is not specified.
    ev1_b = np.dot(np.array(B), ev1)
    if verbose:
        print("sort by evec, ev1_b:")
        print(ev1_b.shape)
        print(ev1_b)
    dot_products = [ np.abs(np.dot(e, ev1_b)) for e in ev2.transpose() ]
    if verbose:
        print("sort by evec, dot_products:")
        print(dot_products)
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
    if verbose:
        print("sort by evecs, permutation:")
        print(res)
    return res

def solve_gevp_gen(data, t0, verbose=False):
    """Generator that returns the eigenvalues for t_0 -> t where t is in
    (t0, t_max].
       
    Calculate the eigenvalues of the generalised eigenvalue problem
    using the scipy.linalg.eigh solver.
    
    Parameters
    ----------
    a : ndarray
        The time dependent data for the GEVP.
    t0 : int
        The index for the inverted matrix.

    Yields
    ------
    ndarray
        The eigenvalues of the respective time
    ndarray
        The eigenvectors of the respective time
    int
        The time.
    """
    # B is the matrix at t=t0
    B = data[t0]
    # define the eigensystem solver function as a lambda function
    try:
        f = lambda A: spla.eigh(b=B, a=A)
    except LinAlgError:
        return

    # initialization
    eigenvectors = None
    count = 0
    evdone = np.zeros((data.shape[1],))

    # calculate the eigensystem for t in (t0, T/2+1)
    for j in range(t0 + 1, data.shape[0]):
        try:
            # calculate the eigensystems
            eigenvalues, new_eigenvectors = f(data[j])
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
                perm = reorder_by_ev(eigenvectors, new_eigenvectors, B, verbose)
            # permutation of the lists
            eigenvectors = new_eigenvectors[:,perm]
            eigenvalues = eigenvalues[perm]
                
            count += 1

            yield eigenvalues, eigenvectors, j

        except (spla.LinAlgError, TypeError) as e:
            #print(e)
            return

def calculate_gevp(data, t0=1):
    """Solves the generalized eigenvalue problem of a correlation
    function matrix.

    The function takes a bootstrapped correlation function matrix and
    calculates the eigenvectors and eigenvalues of the matrix. The
    algorithm relies on the matrix being symmetric or hermitian.

    The eigenvectors are calculated but not stored.

    Parameters
    ----------
    data : ndarray
        The time dependent data for the GEVP.
    t0 : int
        The index for the inverted matrix.

    Returns
    -------
    ndarray
        The array contains the eigenvalues of the solved GEVP. The
        dimension of the array is reduced by one and the data up to t0
        is filled with zeros.
    """
    # Initialize the eigenvalue array
    values_array = np.zeros(data.shape[:-1])
    vec_array = np.zeros(data.shape)
    if data.ndim == 4:
        # iterate over the bootstrap samples
        for _samples in range(data.shape[0]):
            # iterate over the eigensystems
            for eigenvalues, _eigenvectors, _t in solve_gevp_gen(data[_samples], t0):
                # save the eigenvalues to the array
                values_array[_samples, _t] = eigenvalues
                vec_array[_samples, _t] = _eigenvectors
        # set the eigenvalues for t=t0 to 1.0
        values_array[:,t0] = 1.0
    else:
        # iterate over the additional dimensions
        item = [[n for n in range(x)] for x in data.shape[2:-2]]
        for it in itertools.product(*item):
            # iterate over the bootstrap samples
            for _samples in range(data.shape[0]):
                # select the entries
                s = (_samples, slice(None)) + it + (Ellipsis,)
                # iterate over the eigensystems
                for eigenvalues, _eigenvectors, _t in solve_gevp_gen(data[s], t0):
                    # select the entries
                    s1 = (_samples, _t) + it + (Ellipsis,)
                    # save the eigenvalues to the array
                    values_array[s1] = eigenvalues
                    vec_array[s1] = _eigenvectors
            # set the eigenvalues for t=t0 to 1.0
            values_array[:,t0] = 1.0

    return values_array
