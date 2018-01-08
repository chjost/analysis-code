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
from chiral_functions import *
# Christian's packages
import analysis2 as ana

# Wrapper of gamma_pik because NPLQCD did not release fpi
# f denotes the ratio with fpi
def gamma_pik_nplqcd(mpif,mkf,muf,mu_a0):
    ren = np.ones_like(mpif)
    fpi = np.ones_like(mpif)
    _res = np.zeros((3,mpif.shape[-1]))
    _res[0] = 4.*np.pi/muf**2*mu_a0
    _res[1] = 1.+chi_nlo_neg(ren,mpif,mkf,fpi)
    _res[2] = -2.*mkf*mpif/fpi**2*chi_nlo_pos(ren,mpif,mkf)
    _sum = np.sum(_res,axis=0)
    _gamma = -fpi**2/(16.*mpif**2)*_sum 
    return _gamma
