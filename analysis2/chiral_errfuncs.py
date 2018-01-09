import numpy as np
from scipy.optimize import leastsq
import scipy.stats
import matplotlib
matplotlib.use('Agg') # has to be imported before the next lines
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['axes.labelsize'] = 'large'

import chiral_utils as chut
#from chiral_functions import *
import extern_bootstrap as extboot
import plot as plot
from fit import FitResult
from globalfit import ChiralFit
from statistics import compute_error, draw_gaussian_correlated
from plot_functions import plot_function

"""Errorfunctions associated with chiral extrapolations"""
################################################################################
#           Errorfunctions  for M_K a_0                                        #
################################################################################
def mka0_func(r,z,p,x):

    # define the fitfunction for a single beta
    #_func = lambda r, z, p, x,: p[0]*r*x[:,0]/z + p[1]/r**2 + p[2]
    return p[0]*r*x[:,0]/z + p[3]*(r*x[:,0]/z)**2 + p[1]/r**2 + p[2] 

def mka0_errfunc(p,x,y,cov):

    # define the fitfunction for a single beta
    _func = lambda r, z, p, x,: (p[0]*r*x[:,0]/z) + p[1]/r**2 + p[2]
    #_func = lambda r, z, p, x,: p[0]*r*x[:,0]/z*(1 + p[3]*(r*x[:,0]/z)) + p[1]/r**2 + p[2]
    # residuals for
    # With A40.24
    ## Try out an additional term mu_ell^2
    ## TODO: Automate the array shapes, otherwise very errorprone
    #_res_a = _func(p[0],p[3],p[6:10],x[0:6])-y[0:6]
    #_res_b = _func(p[1],p[4],p[6:10],x[6:9])-y[6:9]
    #_res_d = _func(p[2],p[5],p[6:10],x[9:11])-y[9:11]
    # TODO: Automate the array shapes, otherwise very errorprone
    #_res_a = _func(p[0],p[3],p[6:9],x[0:4])-y[0:4]
    #_res_b = _func(p[1],p[4],p[6:9],x[4:7])-y[4:7]
    #_res_d = _func(p[2],p[5],p[6:9],x[7:8])-y[7:8]
    _res_a = _func(p[0],p[2],p[4:7],x[0:4])-y[0:4]
    _res_b = _func(p[1],p[3],p[4:7],x[4:7])-y[4:7]
    # residuals of r0 and zp are stored separately at the moment
    #_res_r0 = np.r_[(y[8]-p[0]),(y[9]-p[1]),(y[10]-p[2])]
    #_res_zp = np.r_[(y[11]-p[3]),(y[12]-p[4]),(y[13]-p[5])]
    _res_r0 = np.r_[(y[7]-p[0]),(y[8]-p[1])]
    _res_zp = np.r_[(y[9]-p[2]),(y[10]-p[3])]
    ## Without A40.24
    ## TODO: Automate the array shapes, otherwise very errorprone
    #_res_a = _func(p[0],p[3],p[6:9],x[0:5])-y[0:5]
    #_res_b = _func(p[1],p[4],p[6:9],x[5:8])-y[5:8]
    #_res_d = _func(p[2],p[5],p[6:9],x[8:10])-y[8:10]
    ## residuals of r0 and zp are stored separately at the moment
    #_res_r0 = np.r_[(y[10]-p[0]),(y[11]-p[1]),(y[12]-p[2])]
    #_res_zp = np.r_[(y[13]-p[3]),(y[14]-p[4]),(y[15]-p[5])]
    # collect residuals as one array
    #_residuals = np.r_[_res_a,_res_b,_res_d,_res_r0,_res_zp ]
    _residuals = np.r_[_res_a,_res_b,_res_r0,_res_zp ]

    # calculate the chi values weighted with inverse covariance matrix
    _chi = np.dot(cov,_residuals)
    return _chi

##############################################################################
#          Errorfunctions  for mu_piK a_3/2                                  #
##############################################################################
# TODO: implement gamma better
def mu_a32_errfunc(p,x,y,cov):
    #print("In NLO-Errfunc shape of x-values is:")
    #print(x.shape)
    # expect two priors
    if y.shape[0] > x.shape[0]:
        #_res = pik_I32_chipt_nlo(x[:,0],x[:,1],x[:,2],x[:,3],p)-y[:-2]
        _res = pik_I32_chipt_nlo(x[:,0],x[:,1],x[:,2],x[:,3],p,meta=x[:,4])-y[:-1]
        #_res = np.r_[_res,p[1]-y[-2],p[2]-y[-1]]
        _res = np.r_[_res,p[1]-y[-1]]
    else: 
      _res = pik_I32_chipt_nlo(x[:,0],x[:,1],x[:,2],x[:,3],p,meta=x[:,4])-y
    # calculate the chi values weighted with inverse covariance matrix
    _chi = np.dot(cov,_res)
    return _chi

def mu_a32_gamma_errfunc(p,x,y,cov):
    print(y.shape[0],x.shape[0]) 
    if y.shape[0] > x.shape[0]:
        _res = p[0]-2.*x.ravel()*p[1]-y[:-1]
        _res = np.r_[_res,p[0]-y[-1]]
    else: 
        _res = p[0]-2.*x.ravel()*p[1]-y
    # calculate the chi values weighted with inverse covariance matrix
    _chi = np.dot(cov,_res)
    return _chi

def mu_a32_lo_errfunc(p,x,y,cov):
    # pik_I32_chipt_lo includes an a^2 term
    print("In LO-Errfunc shape of x-values is:")
    print(x.shape)
    _res = pik_I32_chipt_lo(x[:,0],x[:,1],x[:,2],x[:,3],p)-y
    _chi = np.dot(cov,_res)
    return _chi
