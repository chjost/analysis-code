"""
The class for chiralfitting.
"""

import time
import itertools
import numpy as np

import fit
from fit_routines import (fit_comb, fit_single, calculate_ranges, compute_dE,
    compute_phaseshift, get_start_values, get_start_values_comb, globalfitting)
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

    def chiral_fit(self, X, Y, start=None, xcut=None, ncorr=1,
        parlim=None,correlated=False, debug=3):
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
            A maximal value for the X values. Everything above will not
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
        # create FitResults
        fitres = fit.FitResult("chiral_fit")
        #shape1 = (_X.shape[0], 1, _X.shape[0])
        #shape1 = (_X.shape[0], len(start), _Y.shape[0])
        #shape2 = (_X.shape[0], _Y.shape[0])
        shape1 = (Y.shape[-1], len(start), 1)
        shape2 = (Y.shape[-1], 1)
        if ncorr is None:
          fitres.create_empty(shape1, shape2, 1)
        elif isinstance(ncorr, int):
          fitres.create_empty(shape1, shape2,ncorr)
        else:
          raise ValueError("ncorr needs to be integer")

        # fit the data
        dof = X.shape[0] - len(_start)
         #fit every bootstrap sample
        tmpres, tmpchi2, tmppval = globalfitting(self.errfunc, X, Y, _start,
            parlim=parlim, debug=debug,correlated=correlated)
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

