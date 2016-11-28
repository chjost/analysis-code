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

class ChiralFit(LatticeFit):

    self.fitfunc=None
    self.errfunc=None

    def chiral_fit(self, args, corrid="", start=None, xcut=None, ncorr=None,debug=0):
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
        debug : int
            The amount of information printed to screen.
        """
        # if no start value given, take an arbitrary value
        if start is None:
            _start = [3.0]
        # make sure start is a tuple, list, or ndarray for leastsq to work
        #elif not isinstance(start, (np.ndarray, tuple, list)):
        #    _start = list(start)
        #else:
        #    _start = start
        ## implement a cut on the data if given
        #if xcut:
        #    tmp = X[:,0] < xcut
        #    _X = X[tmp].T
        #    _Y = Y[tmp].T
        #else:
        #    _X = X.T
        #    _Y = Y
        # create FitResults
        fitres = FitResult("chiral_fit")
        #shape1 = (_X.shape[0], 1, _X.shape[0])
        #shape1 = (_X.shape[0], len(start), _Y.shape[0])
        #shape2 = (_X.shape[0], _Y.shape[0])
        shape1 = (_Y.shape[0], len(start), _X.shape[0])
        shape2 = (_Y.shape[0], _X.shape[0])
        if ncorr is None:
          fitres.create_empty(shape1, shape2, 1)
        elif isinstance(ncorr, int):
          fitres.create_empty(shape1, shape2,ncorr)
        else:
          raise ValueError("ncorr needs to be integer")

        # fit the data
        dof = _X.shape[-1] - len(_start)
        # fit every bootstrap sample
        timing = []
        for i, x in enumerate(_X): 
            timing.append(time.clock())
            tmpres, tmpchi2, tmppval = globalfitting(self.errfunc, args, _start, debug=debug)
            fitres.add_data((0,i), tmpres, tmpchi2, tmppval)
            #if i % 100:
            #    print("%d of %d finished" % (i+1, _X.shape[0]))
        t1 = np.asarray(timing)
        print("total fit time %fs" % (t1[-1] - t1[0]))
        t2 = t1[1:] - t1[:-1]
        print("time per fit %f +- %fs" % (np.mean(t2), np.std(t2)))
        return fitres

class ChiralRes(FitResult):

