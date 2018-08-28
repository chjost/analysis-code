#import sys
from scipy import stats
from scipy import interpolate as ip
import time
import matplotlib
matplotlib.use('Agg') # has to be imported before the next lines
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.lines as mlines
import numpy as np
from numpy.polynomial import polynomial as P
from fit_routines import fitting
from chipt_basic_observables import *
from chipt_nlo import *
from pik_scat_len import *
from chipt_decayconstants import *

"""Wrapper functions for fits and plots"""
def calc_x_plot(x):
    """ Function that calculates reduced mass divided by f_pi from mk,mpi and
    fpi"""
    xplot=reduced_mass(x[:,0],x[:,1])/x[:,2]
    #xplot=np.asarray((x[:,1]/x[:,0]))
    return xplot

def calc_x_plot_cont(x):
    """ Function that calculates reduced mass divided by f_pi from ml, ms, b0 and
    l4"""
    mpi=np.sqrt(mpi_sq(x[:,0],b0=x[:,2]))
    mk=np.sqrt(mk_sq(x[:,0],ms=x[:,1],b0=x[:,2]))
    fpi=f_pi(x[:,0],f0=None,l4=x[:,3],b0=x[:,2])
    xplot=reduced_mass(mpi,mk)/fpi
    #xplot=mk/mpi
    return xplot
# TODO Deprecated?
#def err_func(p, x, y, error):
#    # for each lattice spacing and prior determine the dot product of the error
#    chi_a = y.A - pik_I32_chipt_fit(p,x.A)
#    chi_b = y.B - pik_I32_chipt_fit(p,x.B)
#    #chi_d = y.D - pik_I32_chipt_fit(p,x.D)
#    # TODO: find a runtime way to disable prior
#    if y.p is not None:
#        chi_p = y.p - p[1]
#        #return np.dot(error,np.r_[chi_a,chi_b,chi_d])
#        return np.dot(error,np.r_[chi_a,chi_b])
#    else:
#        #return np.dot(error,np.r_[chi_a,chi_b,chi_d,chi_p])
#        return np.dot(error,np.r_[chi_a,chi_b,chi_p])
#
#def gamma_errfunc(p,x,y,error):
#    chi_a = y.A - line(p,x.A)
#    chi_b = y.B - line(p,x.B)
#    #chi_d = y.D - line(p,x.D)
#    # p[0] is L_5
#    chi_p = y.p - p[0]
#    # calculate the chi values weighted with inverse covariance matrix
#    #return np.dot(error,np.r_[chi_a,chi_b,chi_d,chi_p])
#    return np.dot(error,np.r_[chi_a,chi_b,chi_p])
def amk_sq_wopmu(r,z,p,x):
    mk_sq = p[0]/(r*z) * (x[:,0]+x[:,1])* (1+p[1]*(r/z)*x[:,0]+p[2]/(r**2))
    return mk_sq

def global_ms_errfunc_wopmu(p,x,y,error):

    # define the fitfunction for a single beta
    #_func = lambda r, z, p, x,: p[0]/(r*z) * (x[:,0]+x[:,1])* (1+p[1]*(r/z)*x[:,0]+p[2]/(r**2))

    # TODO: Automate the array shapes, otherwise very errorprone
    chi_a = y.A - amk_sq_wopmu(p[0],p[3],p[6:9],x.A)
    chi_b = y.B - amk_sq_wopmu(p[1],p[4],p[6:9],x.B) 
    chi_d = y.D - amk_sq_wopmu(p[2],p[5],p[6:9],x.D)
    chi_d45 = y.D45 - amk_sq_wopmu(p[2],p[5],p[6:9],x.D45)
    # have several priors here as a list
    # y.p is list of 6 priors
    chi_p = np.asarray(y.p) - np.asarray(p[0:6])  
    _residuals = np.concatenate((np.r_[chi_a,chi_b,chi_d,chi_d45], chi_p))
    # calculate the chi values weighted with inverse covariance matrix
    _chi = np.dot(error,_residuals)
    return _chi

def global_ms_errfunc(p,x,y,error):

    # define the fitfunction for a single beta
    #_func = lambda r, z, p, x,: p[0]/(r*z) * (x[:,0]+x[:,1]) * (1+p[1]*(r/z)*x[:,0]+p[2]/(r**2))
    _func = lambda r, z, p, mu, x,: p[0]/(r*z) * (x[:,0]+mu*x[:,1])* (1+p[1]*(r/z)*x[:,0]+p[2]/(r**2))
    #_func = lambda r, z, p, x,: 1./(r*z) * (p[0]*x[:,0]+p[3]*x[:,1]) *
    #(1+p[1]*(r/z)*x[:,0]+p[2]/(r**2))+
    #_func = lambda r, z, mu, p, x,: 1./(r*z) * (p[0]*x[:,0]+p[3]*x[:,1]) * (1+p[1]*(r/z)*x[:,0]+p[2]/(r**2)) + mu/r**2
    #_func = lambda r, z, mu, p, x,: p[0]/(r*z) * (x[:,0]+x[:,1]*(1+mu)) * (1+p[1]*(r/z)*x[:,0]+p[2]/(r**2))

    # TODO: Automate the array shapes, otherwise very errorprone
    #chi_a = y.A - _func(p[0],p[3],p[6:9],x.A)
    #chi_b = y.B - _func(p[1],p[4],p[6:9],x.B) 
    #chi_d = y.D - _func(p[2],p[5],p[6:9],x.D)
    chi_a = y.A - _func(p[0],p[3],p[6:9],p[9],x.A)
    chi_b = y.B - _func(p[1],p[4],p[6:9],p[10],x.B) 
    chi_d = y.D - _func(p[2],p[5],p[6:9],p[11],x.D)
    # modification for musigma dependent fit parameter
    chi_d45 = y.D45 - _func(p[2],p[5],p[6:9],p[12],x.D45)
    #chi_a = y.A - _func(p[0],p[2],p[4:7],x.A)
    #chi_b = y.B - _func(p[1],p[3],p[4:7],x.B) 
    # have several priors here as a list
    # y.p is list of 6 priors
    chi_p = np.asarray(y.p) - np.asarray(p[0:6])  
    #chi_p = np.asarray(y.p) - np.asarray(p[0:4])  
    # collect residuals as one array
    # chi_p[0] are the r_0 residuals, chi_p[1] are the zp residuals
    #_residuals = np.concatenate((np.r_[chi_a,chi_b,chi_d], chi_p))
    _residuals = np.concatenate((np.r_[chi_a,chi_b,chi_d,chi_d45], chi_p))
    #_residuals = np.concatenate((np.r_[chi_a,chi_b], chi_p))

    # calculate the chi values weighted with inverse covariance matrix
    _chi = np.dot(error,_residuals)
    return _chi
    
def line(p,x):
    _res = p[0]-2.*x*p[1]
    return _res

def pik_I32_chipt_fit(p,x,add=None):
    """ Wrapper for fitfunction"""
    # x and args need to have the same number of entries in last dimension
    # (bootstrapsamples)
    # broadcast _x values to same shape as arguments
    #if hasattr(x,'__iter__') is not True:
    #    _x = np.zeros((len(x),args.shape[0]))
    #    for i,d in enumerate(np.asarray(x)):
    #        _x[i] = np.full((1500,),d)
    #else:
    #    _x = x.reshape(len(x),1)
    if p.ndim == 2 and p.shape[0]> p.shape[1]:
        _args = p.T
    else:
        _args=p
        #_res = pik_I32_chipt_nlo(_x[0],_x[1],_x[2], _args[-1], _args[0:3],meta=_x[4])
        #print("In fitfunction: x_shapem, arguments_shape")
        #print(x.shape,_args.shape)
        _res = pik_I32_chipt_nlo(x[:,0],x[:,1],x[:,2],
            _args[0:3],meta=x[:,4])
    if add is not None:
        _ret = np.r_[_res,np.atleast_2d(add)]
    else:
        _ret = _res
    #print("pik_I32_chipt_fit returns:")
    #print(_ret)
    return _ret

def pik_I32_chipt_plot(args, x):
    """ Wrapper for plotfunction"""
    # x and args need to have the same number of entries in last dimension
    # (bootstrapsamples)
    # broadcast _x values to same shape as arguments
    if hasattr(x,'__iter__') is not True:
        _x = np.zeros((len(x),args.shape[0]))
        for i,d in enumerate(np.asarray(x)):
            _x[i] = np.full((1500,),d)
    else:
        _x = x.reshape(len(x),1)
    if args.ndim == 2 and args.shape[0]> args.shape[1]:
        _args = args.T
    else:
        _args=args
    #return pik_I32_chipt_nlo(_x[0],_x[1],_x[2], args[0,3], args[0,0:3])
    #return pik_I32_chipt_nlo(_x[0],_x[1],_x[2], _args[-1], _args[0:3],meta=_x[4])
    return pik_I32_chipt_nlo(_x[0],_x[1],_x[2], _args[0:3],meta=None)

def pik_I32_chipt_plot_cont(args, x):
    # x and args need to have the same number of entries in last dimension
    # (bootstrapsamples)
    # broadcast _x values to same shape as arguments
    if hasattr(x,'__iter__') is not True:
        _x = np.zeros((len(x),args.shape[0]))
        for i,d in enumerate(np.asarray(x)):
            _x[i] = np.full((1500,),d)
    else:
        _x = x.reshape(len(x),1)
    if args.ndim == 2 and args.shape[0]> args.shape[1]:
        _args = args.T
    else:
        _args=args
    # B_0 (args[-1]), F_0 and m_s (x[1]) fixed
    # use GMOR relations for meson masses
    mpi=np.sqrt(mpi_sq(_x[0],_args[-1]))
    print(_x[1])
    mk=np.sqrt(mk_sq(_x[0],_x[1],_args[-1]))
    fpi=f_pi(_x[0],f0=None,l4=_args[-2],b0=_args[-1])
    _lat = _args[3]**2*mk**2/197.37**2
    return pik_I32_chipt_nlo(mpi, mk, fpi, _args[0:3], lat=_lat)

def pik_I32_chipt_nlo_plot(args, x):
    """ Wrapper for plotfunction subtract LO before plotting"""
    # x and args need to have the same number of entries in last dimension
    # (bootstrapsamples)
    # broadcast _x values to same shape as arguments
    if hasattr(x,'__iter__') is not True:
        _x = np.zeros((len(x),args.shape[0]))
        for i,d in enumerate(np.asarray(x)):
            _x[i] = np.full((1500,),d)
    else:
        _x = x.reshape(len(x),1)
    if args.ndim == 2 and args.shape[0]> args.shape[1]:
        _args = args.T
    else:
        _args=args
    # LO contribution
    lo = -(reduced_mass(_x[0],_x[1])/_x[2])**2/(4.*np.pi)
    _pik = pik_I32_chipt_nlo(_x[0],_x[1],_x[2], _args[-1], _args[0:3])
    return (_pik-lo)/_pik

def pik_I32_chipt_cont(args, x):
    """ Wrapper for plotfunction
    ATM: last entry in arguments is lattice artefact.
    """
    # x and args need to have the same number of entries in last dimension
    # (bootstrapsamples)
    # broadcast _x values to same shape as arguments
    if hasattr(x,'__iter__') is not True:
        _x = np.zeros((len(x),args.shape[0]))
        for i,d in enumerate(np.asarray(x)):
            _x[i] = np.full((1500,),d)
    else:
        _x = x.reshape(len(x),1)
    if args.ndim == 2 and args.shape[0]> args.shape[1]:
        _args = args.T
    else:
        _args=args
    #_args[2]=np.zeros_like(args[2])
    #return pik_I32_chipt_nlo(_x[0],_x[1],_x[2], args[0,3], args[0,0:3])
    return pik_I32_chipt_nlo(_x[0],_x[1],_x[2], _args[0:3], meta=_x[4])
    #return pik_I32_chipt_nlo(_x[0],_x[1],_x[2], _args[0:3],
    #        meta=_x[4], lat=args[-1])

def pik_I32_chipt_lo_plot(args, x):
    """ Wrapper for plotfunction"""
    _x = x.reshape((len(x),1))
    return pik_I32_chipt_lo(_x[0],_x[1],_x[2], args[:,1], args[:,0])

def pik_I32_chipt_nlo_plot(args, x):
    """ Wrapper for plotfunction subtract LO before plotting"""
    # x and args need to have the same number of entries in last dimension
    # (bootstrapsamples)
    # broadcast _x values to same shape as arguments
    if hasattr(x,'__iter__') is not True:
        _x = np.zeros((len(x),args.shape[0]))
        for i,d in enumerate(np.asarray(x)):
            _x[i] = np.full((1500,),d)
    else:
        _x = x.reshape(len(x),1)
    if args.ndim == 2 and args.shape[0]> args.shape[1]:
        _args = args.T
    else:
        _args=args
    # LO contribution
    lo = -(reduced_mass(_x[0],_x[1])/_x[2])**2/(4.*np.pi)
    _pik = pik_I32_chipt_nlo(_x[0],_x[1],_x[2], _args[-1], _args[0:3])
    return (_pik-lo)/_pik

def pik_I32_chipt_lo_plot(args, x):
    """ Wrapper for plotfunction"""
    _x = x.reshape((len(x),1))
    return pik_I32_chipt_lo(_x[0],_x[1],_x[2], args[:,1], args[:,0])

def mua0_I32_from_fit(pars,x):
    # Ensure that x has at least 2 dimensions
    _x = np.atleast_2d(x)
    _mua0 = mu_aI32(_x[:,0],_x[:,1],_x[:,2],_x[:,3],pars[0],pars[1])
    return _mua0

def mua0_I32_hkchpt_from_fit(pars,x):
    # Ensure that x has at least 2 dimensions
    _x = np.atleast_2d(x)
    _mua0 = pik_I32_hkchpt(_x[:,0],_x[:,1],_x[:,2],_x[:,3],(pars[0],pars[1]))
    return _mua0

def mua0_I12_from_fit(pars,x):
    # Ensure that x has at least 2 dimensions
    _x = np.atleast_2d(x)
    _mua0 = mu_aI12(_x[:,0],_x[:,1],_x[:,2],_x[:,3],pars[0],pars[1])
    return _mua0

def mua0_I32_nlo_from_fit(pars,x):
    _x = np.atleast_2d(x)
    _mua0 = pik_I32_chipt_nlo(_x[:,1], _x[:,2], _x[:,3], _x[:,4], pars, lambda_x=None) 
    return _mua0

def fk_by_fpi_fit(p, x, add=None):
    ratio_fit = fk_by_fpi(x[:,2],x[:,3],x[:,4],x[:,0],x[:,0],p)
    print("Ratio fit return has shape:")
    print(ratio_fit.shape)
    return ratio_fit

def fk_fpi_ratio_errfunc(p, x, y, error):
    # for each lattice spacing and prior determine the dot product of the error
    chi_a = y.A - fk_by_fpi_fit(p,x.A)
    chi_b = y.B - fk_by_fpi_fit(p,x.B)
    chi_d = y.D - fk_by_fpi_fit(p,x.D)
    # TODO: find a runtime way to disable prior
    print(error.shape)
    print(y.A.shape)
    print(y.B.shape)
    print(y.D.shape)
    print(chi_a.shape)
    print(chi_b.shape)
    print(chi_d.shape)
    print(np.r_[chi_a,chi_b,chi_d].shape)
    if y.p is not None:
        chi_p = y.p - p[1]
        return np.dot(error,np.r_[chi_a,chi_b,chi_d,chi_p])
    else:
        return np.dot(error,np.r_[chi_a,chi_b,chi_d])
