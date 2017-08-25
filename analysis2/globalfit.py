"""
The class for chiralfitting.
"""

import time
import itertools
import numpy as np

import fit
from fit_routines import (fit_comb, fit_single, calculate_ranges, compute_dE,
     get_start_values, get_start_values_comb, globalfitting)
from in_out import read_fitresults, write_fitresults
from interpol import match_lin, match_quad, evaluate_lin
from functions import (func_single_corr, func_ratio, func_const, func_two_corr,
    func_two_corr_shifted, func_single_corr2, func_sinh, compute_eff_mass,
    func_two_corr_therm)
from statistics import (compute_error, sys_error, sys_error_der, draw_weighted,
    freq_count, draw_gauss_distributed)
from energies import calc_q2, calc_Ecm
from zeta_wrapper import Z
from scattering_length import calculate_scat_len
from chiral_utils import evaluate_phys

class ChiralFit(fit.LatticeFit):
    
    def __init__(self,fit_id, errfunc):
        self.fit_id = fit_id
        self.fitfunc = None
        self.errfunc = errfunc
        
    def cut_data(self,x,y,interval):
        """ cut data of chiral fit object according to given interval
        Parameters
        ----------
        interval: tuple or float, interval where to cut
        
        Returns
        -------
        _data: cutted data with same shape as input except for 0th axis
        """
        # determine shapes
        _x_shape = x.shape
        _y_shape = y.shape
 
        #cut the xdata if necessary
        # implement a cut on the data if given, negative means everything above
        # that x-value
        print("interval is: %r" %interval)
        # only interested in first range
        if hasattr(interval,"__iter__"):
            sub = (0,)*len(_x_shape[1:])
            select = (slice(None),)+sub
            print(x[select])
            lo = x[select] > interval[0]
            hi = x[select] < interval[1]
            tmp = np.logical_and(lo,hi)
            print("Shape for cutting:")
            # should be a 1d array
            print(tmp)
        elif interval >= 0.:
            tmp = x[:,0] < interval
        elif interval < 0.:
            tmp = x[:,0] > -interval
        print("y-shape before cut:")
        print(y.shape)
        _x = x[tmp]
        _y = y[tmp]
        print("y-data after cut:")
        print(_y[:,0])
        print("y-shape after cut:")
        print(_y.shape)
        
        return _x, _y

    def chiral_fit(self, X, Y, start=None, xcut=None, ncorr=1,
        parlim=None,correlated=False,cov=None,add=None, debug=3):
        """Fit function to data.
        
        Parameters
        ----------
        X, Y : ndarrays
            The data arrays for X and Y. Assumes ensemble as first axis
            and bootstrap samples as second axis.
        corrid : str
            Identifier for the fit result.
        start : list or tuple or ndarray
            Start value for the fit.
        xcut : float
            A maximal value for the X values. Everything above (below) will not
            be used in the fit. 
        parlim : tuple, errors on the fitparameters, if given they are
            considered as weights to the parameter constraints
        debug : int
            The amount of information printed to screen.
        """
        # if no start value given, take an arbitrary value
        if start is None:
            _start = [3.0]
        else:
          _start = start

        if xcut is not None:
            _X, _Y = self.cut_data(X,Y,xcut)
        else:
            _X = X
            _Y = Y
        # create FitResults
        fitres = fit.FitResult("chiral_fit")
        #shape1 = (_X.shape[0], 1, _X.shape[0])
        #shape1 = (_X.shape[0], len(start), _Y.shape[0])
        #shape2 = (_X.shape[0], _Y.shape[0])
        #TODO: Get array shapes for fitresult, perhaps a member variable
        #shape1 = (1500, len(start), 1)
        #shape2 = (1500, 1)
        shape1 = (_Y.shape[-1], len(start), 1)
        shape2 = (_Y.shape[-1], 1)
        if ncorr is None:
          fitres.create_empty(shape1, shape2, 1)
        elif isinstance(ncorr, int):
          fitres.create_empty(shape1, shape2,ncorr)
        else:
          raise ValueError("ncorr needs to be integer")
        # fit the data
        dof = _X.shape[0] - len(_start)
        #dof = 11 - len(_start)
        print("In global fit, dof are: %d" %dof)
         #fit every bootstrap sample
        tmpres, tmpchi2, tmppval = globalfitting(self.errfunc, _X, _Y, _start,
            parlim=parlim, debug=debug,correlated=correlated,cov=cov, add=add)
        fitres.add_data((0,0), tmpres, tmpchi2, tmppval)
        #timing = []
        #for i in range(ncorr): 
        #    timing.append(time.clock())
        #    #if i % 100:
        #    #    print("%d of %d finished" % (i+1, _X.shape[0]))
        #t1 = np.asarray(timing)
        #print("total fit time %fs" % (t1[-1] - t1[0]))
        #t2 = t1[1:] - t1[:-1]
        #print("time per fit %f +- %fs" % (np.mean(t2), np.std(t2)))
        return fitres

#class ChiralRes(FitResult):

