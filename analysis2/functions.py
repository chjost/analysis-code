"""
Different functions with no other place to go.
"""

import os
import numpy as np

def compute_derivative(data):
    """Computes the derivative of a correlation function.

    The data is assumed to a numpy array. The derivative is calculated
    along the second axis.

    Parameter
    ---------
    data : ndarray
        The data.

    Returns
    -------
    ndarray
        The derivative of the data.

    Raises
    ------
    IndexError
        If array has only 1 axis.
    """
    # creating derivative array from data array
    dshape = list(data.shape)
    dshape[1] = data.shape[1] - 1
    derv = np.zeros(dshape, dtype=float)
    # computing the derivative
    for b, row in enumerate(data):
        for t in range(len(row)-1):
            derv[b,t] = row[t+1] - row[t]
    return derv

def compute_eff_mass(data, usecosh=True):
    """Computes the effective mass of a correlation function.

    The effective mass is calculated along the second axis. The extend
    along the axis is reduced, depending on the effective mass formula
    used. The standard formula is based on the cosh function, the
    alternative is based on the log function.

    Parameters
    ----------
    data : ndarray
        The data.
    usecosh : bool
        Toggle between the two implemented methods.

    Returns
    -------
    ndarray
        The effective mass of the data.
    """
    if usecosh:
        # creating mass array from data array
        mass = np.zeros_like(data[:,:-2])
        for b, row in enumerate(data):
            for t in range(1, len(row)-1):
                mass[b,t-1] = (row[t-1] + row[t+1])/(2.*row[t])
        mass = np.arccosh(mass)
    else:
        # creating mass array from data array
        mass = np.zeros_like(data[:,:-1])
        for b, row in enumerate(data):
            for t in range(len(row)-1):
               mass[b, t] = np.log(row[t]/row[t+1])
    return mass

def func_single_corr(p, t, T2):
    """A function that describes two point correlation functions.

    The function is given by 0.5*p0^2*(exp(-p1*t)+exp(-p1*(T2-t))),
    where
    * p0 is the amplitude,
    * p1 is the energy of the correlation function,
    * t is the time, and
    * T2 is the time around which the correlation function is symmetric,
    usually half the lattice time extend.

    Parameters
    ----------
    p : sequence of float
        The parameters of the function.
    t : float
        The variable of the function.
    T2 : float
        The time around which the function is symmetric.

    Returns
    -------
    float
        The result.
    """
    return 0.5*p[0]*p[0]*(np.exp(-p[1]*t)+np.exp(-p[1]*(T2-t)))

def func_ratio(p, t, o):
    """A function which describes the ratio of a four and a two point
    function.

    The function is given by p0*(cosh(p1*(t-o0-1))+sinh(p1*(t-o0/2))/
    (tanh(2*o1*(t-o0/2)))), where
    * p0 is the amplitude
    * p1 is the energy difference
    * t is the time,
    * o1 is the time extent of the lattice, and
    * o0 is the single particle energy.

    Parameters
    ----------
    p : sequence of float
        The parameters of the function.
    t : float
        The variable of the function.
    o : sequence of float
        The constants of the function.

    Returns
    -------
    float
        The result.
    """
    return p[0]*(np.cosh(p[1]*(t-o[0]/2.+0.5))+np.sinh(p[1]*(t-o[0]/2.+0.5))/
            (np.tanh(2.*o[0]*(t-o[1]/2.+0.5))))

def func_const(p, t):
    """A constant function.

    The function is given by p.
    The further arguments are needed to be compatible to the other functions
    func_*.

    Parameters
    ----------
    p : float
        The parameter of the function
    t : float
        Not used, but needed.

    Returns
    -------
    float
        The result.
    """
    return p

def compute_weight(corr, pvals, par=1):
    """compute the weights for the histogram

    Parameters
    ----------
    corr : ndarray
        the correlation functions.
    pvals : ndarray
        The p-values for each correlation function.
    par : int
        The parameter for which to calculate the weight.

    Returns
    -------
    list
        The calculated weights.
    """
    errors = np.std(corr, axis=1)
    max_err = np.amax(errors)
    weights = []
    if len(pvals) != 0:
        for i in range(pvals.shape[0]):
            w = (1.-2*abs(pvals[i,par]-0.5))*max_err/errors[i]
            weights.append(w**2)
    return weigths
