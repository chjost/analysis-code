"""
The class for fitting.
"""

import itertools
import numpy as np

from fit_routines import fit, fit_comb
from functions import func_single_corr, func_ratio, func_const

class LatticeFit(object):
    def __init__(self, verbose=False):
        self.verbose = verbose

    def fit(self, fitfunc, start, corr, ranges, oldfit=None, oldfitpar=None):
        """Fits fitfunc to a Correlators object.

        The predefined functions describe a single particle correlation
        function, a ratio of single and two-particle correlation
        functions and a constant function.

        Parameters
        ----------
        fitfunc : {0, 1, 2, callable}
            Choose between three predefined functions or an own
            fit function.
        start : float or sequence of floats
            The start parameters for the fit.
        corr : Correlators
            A correlators object with the data.
        ranges : sequence of ints or sequence of sequences of int
            The ranges in which to fit, either one range for all or one
            range for each data set in corr. Each range consists of a
            lower and an upper bound.
        oldfit : None or FitResult, optional
            Reuse the fit results of an old fit for the new fit.
        oldfitpar : None, int or sequence of int, optional
            Which parameter of the old fit to use, if there is more
            than one.

        Returns
        -------
        FitResult
            A class that holds all results.
        """
        # chose the correct function if using predefined function
        if isinstance(fitfunc, int):
            if fitfunc > 2:
                raise ValueError("No fit function choosen")
            functions = {0: func_single_corr, 1: func_ratio, 2: func_const}
            fitfunc = functions.get(fitfunc)

        # check if old data is reused
        if oldfit is None:
            fit(fitfunc, start, corr, ranges)
        else:
            fit_comb(fitfunc, start, corr, ranges, oldfit, oldfitpar)

class FitResult(object):
    """Class to hold the results of a fit.

    The data is assumed to have the following layout:
    (nbsample, npar, range [, range, ...])
    where nsample the number of samples is, npar the number of
    parameters and range is a fit range number. Arbitrary many fit
    ranges can be used.

    To keep track labels generated for the data to keep track of the
    fit a fit range comes from and the number of the correlator.

    Next to the data the chi^2 data and the p-values of the fit are
    saved.
    """
    def __init__(self):
        self.data = []
        self.pvals = []
        self.chi2 = []
        self.label = []
        self.fit_ranges = None

    def add_data(self, data, chi2, corr_id):
        """Add data to FitResult.

        Parameters
        ----------
        data : ndarray or sequence of ndarrays
            The data to add.
        chi2 : ndarray or sequence of ndarraysi, optional
            The chi2 to the fits
        """
        if isinstance(data, (tuple, list)):
            for num, d in enumerate(data):
                self.data.append(data)
                self.chi2.append(chi2[num])
                self.label.append(self._get_label(corr_id, num))
        else:
            self.data.append(data)
            self.label.append(self._get_label(corr_id, 0))

    def get_empty(self, shape, corr_id, corr_num):
        if len(corr_id) != len(corr_num):
            raise ValueError("corr_id and corr_num are not of the same length")
        data = []
        label = []
        comb = [[x for x in range(n)] for n in corr_num]
        for item in itertools.product(*comb):
            data.append(np.zeros(shape))
            label.append(self._get_label(corr_id, item))
        return None

    def _get_label(self, c_id, c_num):
        tmp = np.ndarray((2,), dtype=object)
        tmp[0] = c_id
        tmp[1] = c_num
        return tmp

    def get_ranges(self):
        """Returns the fit ranges."""
        return self.fit_ranges

class FitArray(np.ndarray):
    """Subclass of numpy array."""
    def __new__(subtype, shape, dtype=float, buffer=None, offset=0,
          strides=None, order=None, corr_id=[], corr_num=[]):
        """Creats a numpy array with two additional attributes.

        For details on parameters not presented here, see numpy
        documentation.

        Parameters
        ----------
        shape : int of sequence of ints
            The shape of the new array.
        corr_id : sequence of str
            Identifiers of the correlators already used.
        corr_num : sequence of int
            The number of the correlators used.

        Returns
        -------
        obj : FitArray
            The new array.
        """
        obj = np.ndarray.__new__(subtype, shape, dtype, buffer, offset, strides,
                         order)

        obj.corr_id = corr_id
        obj.corr_num = corr_num

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.corr_id = getattr(obj, 'corr_id', [])
        self.corr_num = getattr(obj, 'corr_num', [])

    #def __mul__(self, x):
    #    return np.multiply(self, x)

    #def __add__(self, x):
    #    return np.add(self, x)

    #def __div__(self, x):
    #    return np.div(self, x)

    #def __sub__(self, x):
    #    return np.subtract(self, x)
