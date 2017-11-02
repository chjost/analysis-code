"""
Different functions with no other place to go.
"""

import os
import numpy as np
from scipy.optimize import fsolve

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
#TODO: Clean up this mess of a mass function
def compute_eff_mass(data, usecosh=True, exp=False, weight=None, shift=None):
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
    if exp is True:
       # Write a numerical solve for the effective mass per timeslice
       # args are (t,T2)
       T2 = data.shape[1]-1
       mass = np.zeros_like(data[:,:-1])
       print(mass.shape)
       for b, row in enumerate(data):
           for t in range(len(row)-1):
                #mass[b, t] = fsolve(corr_exp,0.5,args=(row[t],row[t+1],t,T2))
                mass[b, t] = fsolve(corr_exp_mod,0.5,args=(row[t],weight[b],t,T2))
      
    elif (usecosh == True and weight is None):
        # creating mass array from data array
        mass = np.zeros_like(data[:,:-2])
        for b, row in enumerate(data):
            for t in range(1, len(row)-1):
                mass[b,t-1] = (row[t-1] + row[t+1])/(2.*row[t])
        mass = np.arccosh(mass)
    elif (usecosh == False and weight is None):
       # Write a numerical solve for the effective mass per timeslice
       # args are (t,T2)
       T2=data.shape[1]
       mass = np.zeros_like(data[:,:-1])
       print(mass.shape)
       for b, row in enumerate(data):
           for t in range(len(row)-1):
                #print(fsolve(corr_shift_ratio,0.5,args=(row[t],row[t+1],t,T2)))
                mass[b, t] = fsolve(corr_shift_ratio,0.5,args=(row[t],row[t+1],t,T2))
    
    elif(weight is not None and shift is not None):
       print("Using shifted weighted")
       T2=data.shape[1]
       mass = np.zeros_like(data[:,:-1])
       print(mass.shape)
       for b, row in enumerate(data):
            for t in range(len(row)-1):
                 mass[b, t] = fsolve(corr_shift_weight,0.5,
                                     args=(row[t],row[t+1],t,T2,weight[b],
                                     shift),maxfev=1000)
    else:
        # creating mass array from data array
        mass = np.zeros_like(data[:,:-1])
        for b, row in enumerate(data):
            for t in range(len(row)-2):
               mass[b, t] = np.log(row[t]/row[t+1])
    return mass

def corr_shift_weight(m,r0,r1,t,T2,weight,shift=1.):
    _num = np.exp(-m*t) + np.exp(-m*(2*T2-t))-np.exp(weight*shift) * ( np.exp(-m*(t+shift)) + np.exp(-m*(2*T2-t-shift)) )
    _den = np.exp(-m*(t+1)) + np.exp(-m*(2*T2-t-1))-np.exp(weight*shift) * ( np.exp(-m*(t+1+shift)) + np.exp(-m*(2*T2-t-1-shift)) )
    _diff = r0/r1 - _num/_den
    return _diff

def corr_exp(m,r0,r1,t,T2):
    """
    Parameters
    ----------
    p: tuple
    """
    _den = np.exp(-m*t) + np.exp(-m*(2*T2-t))
    _num = np.exp(-m*(t+1)) + np.exp(-m*(2*T2-(t+1))) 
    _diff = r0/r1 - _den/_num 
    return _diff

def corr_exp_mod(m,r0,p,t,T2):
    """
    Parameters
    ----------
    p: tuple
    """
    _den = p[0]*np.exp(m*(t-T2)) + p[2]*np.exp(-m*(t-T2))
    _diff = r0 - _den 
    return _diff

def corr_shift_ratio(m,r0,r1,t,T2):
    """
    Parameters
    ----------
    p: tuple
    """
    #_den = np.exp(-m*t) - np.exp(-m*(2*T2-t))
    #_num = np.exp(-m*(t+1)) - np.exp(-m*(2*T2-(t+1))) 
    _den = np.sinh(m*(T2-t-0.5))
    _num = np.sinh(m*(T2-t-1.-0.5)) 
    _diff = r0/r1 - _den/_num 
    return _diff

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

def func_single_corr2(p, t, T2=None):
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
    #return 0.5*p[0]*p[0]*(np.exp(-p[1]*t))
    return p[0]*(np.exp(-p[1]*t))

def func_two_corr(p, t, o):
    """A function which describes the four point correlation
    function.

    The function is given by p0*cosh(p1*(t-o1/2.)) +
    p2*exp(-o0*o1), where
    * p0 is the first amplitude
    * p1 is the energy
    * p2 is the second amplitude
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
    return p[0]*np.cosh(p[1]*(t-(o[1]/2.))) + p[2]*np.exp(-o[0]*o[1])

def func_two_corr_therm(p,t,T):
    """ Function describing four point correlation function with temporally
    constant thermal states

    Parameters
    ----------
    p : sequence of float
        The parameters of the function
    t : float
        The variable of the function
    T : int 
        the time extent of the lattice
    Returns
    -------
    """
    #return p[0]*(np.exp(-p[1]*(t)) + np.exp(-p[1]*(T-t))) + p[2]
    return p[0]*np.cosh(p[1]*(0.5*T-t)) + p[2]

def func_two_corr_shifted(p, t, T):
    """A function which describes the shifted four point correlation
    function C(t+1) - C(t)

    The function is given by 2.*p[0]*(np.cosh(p[1]*(T2-(t+1))) -
    np.cosh(p[0]*(T2-t))), where
    * p0 is the amplitude
    * p1 is the energy
    * p2 is a constant in time contribution
    * t is the time,
    * T2 is half the lattice time extent

    Parameters
    ----------
    p : sequence of float
        The parameters of the function.
    t : float
        The variable of the function.
    T2 : 0.5 * L_T of the lattice

    Returns
    -------
    float
        The result.
    """
    #print("Using lattice time extent %f timeslices " % T)
    #return -p[0]*np.exp(-p[1]*(t+T+1.)) * (-1.+np.exp(p[1])) * \
    #                     (np.exp(p[1]+2.*p[1]*t)-np.exp(p[1]*T))
    return 0.5*p[0]*p[0]*(np.exp(-(t+0.5)*p[1]) - np.exp(-(T-t-0.5)*p[1]))
    # Alternative expression
    #return 2*p[0]*np.sinh(p[1]*(0.5*T-t-0.5))
    #return 2.*p[0]*(np.cosh(p[1]*(T2-(t+1))) - np.cosh(p[1]*(T2-t)))

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
    return p[0]*(np.cosh(p[1]*(t-(o[1]/2.)))+np.sinh(p[1]*(t-(o[1]/2.)))/
            (np.tanh(2.*o[0]*(t-(o[1]/2.)))))

def func_const(p, t, e):
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
    e : float
        Not used, but needed.

    Returns
    -------
    float
        The result.
    """
    return p

def func_sinh(p, t, o):
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
    try:
        if len(o) > 1:
            _o = o[1]
        else:
            _o = o
    except TypeError:
        _o = o
    return p[0]*np.sinh(p[1]/2.) * np.sinh(p[1]*(t-_o/2.))

def simple_difference(d1, d2=None):
    """Calculates the difference of two data sets

    Parameters
    ----------
    d1, d2 : three data sets
    
    Returns:
      the difference between the data sets
    """

    # create array from dimensions of the data
    rshape = d1.shape
    print(d1)
    print(rshape[0],rshape[1])
    difference = np.zeros_like(d1)
    if d2 is None:
      for _s in range(rshape[0]):
      
        for _t in range(rshape[1]):
          print(_s,_t)
          # calculate difference
          difference[_s,_t] = d1[_s,_t,0] - d1[_s,_t,1]
    else:
      if rshape != d2.shape:
        raise ValueError("data sets have different shapes")
      for _s in range(rshape[0]):
      
        for _t in range(rshape[1]):
          print(_s,_t)
          # calculate difference
          difference[_s,_t] = d1[_s,_t] - d2[_s,_t]


    return difference
    
def compute_derivative_back(data,a=1):
    """Computes the backward derivative of a correlation function.
    as used for example in the ratios

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
    print dshape
    dshape[1] = data.shape[1] - a
    derv = np.zeros(dshape, dtype=float)
    # computing the derivative
    for b, row in enumerate(data):
        for t in range(len(row)-a):
            derv[b,t] = row[t] - row[t+a]
    return derv

def multiply(data,fac):
    """Multiply correlator by a factor
    """
    return fac*data

def compute_square(data):
    """ Compute the squared correlator
    
    Parameter
    ---------
    data : ndarray
        The data.

    Returns
    -------
    ndarray
        The square of the data.
    """
    return np.square(data)

def func_corr_shift_therm(p, t, add):
    """A function that describes a shifted four point correlation function.
    including time dependent thermal states.

    The function is given by p0^2*(exp(-p1*t)+exp(-p1*(T2-t)) -
    exp(-p1*(t+s))+exp(-p1*(T2-(t+s)))
    + p[2] * np.exp(-add[1]*add[2]) * (1-np.exp(2*s*(add[1]-add[0]))) * np.exp((add[1]-add[0])*t) 
    ,
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
    s=1.
    gs = p[0] * p[0] * (np.exp(-p[1]*t) + np.exp(-p[1]*(add[2]-t)) - np.exp(s*(add[1]-add[0]))*(np.exp(-p[1]*(t+s)) + np.exp(-p[1]*(add[2]-(t+s)))))
    ts = p[2] * np.exp(-add[1]*add[2]) * (1-np.exp(2*s*(add[1]-add[0]))) * np.exp((add[1]-add[0])*t) 
    return gs+ts

def func_two_corr_dws(p,t,add):
    """
    Function for a doubly shifted and weighted correlation function

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
    return p[0]*np.exp(p[1]*(t-add[0]))+p[2]*np.exp(-p[1]*(t-add[0]))


def func_single_corr_bare(p, t, T2):
    """A function that describes two point correlation functions.

    The function is given by p0^2*(exp(-p1*t)+exp(-p1*(T2-t))),
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
    return p[0]*p[0]*(np.exp(-p[1]*t)+np.exp(-p[1]*(T2-t)))
