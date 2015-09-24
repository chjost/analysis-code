"""
The class for fitting.
"""


import itertools
import numpy as np

from fit_routines import fit_comb, fit_single
from in_out import read_fitresults, write_fitresults
from functions import func_single_corr, func_ratio, func_const, compute_error

class LatticeFit(object):
    def __init__(self, fitfunc, verbose=False):
        """Create a class for fitting fitfunc.

        Parameters
        ----------
        fitfunc : {0, 1, 2, callable}
            Choose between three predefined functions or an own
            fit function.
        """
        self.verbose = verbose
        # chose the correct function if using predefined function
        if isinstance(fitfunc, int):
            if fitfunc > 2:
                raise ValueError("No fit function choosen")
            functions = {0: func_single_corr, 1: func_ratio, 2: func_const}
            self.fitfunc = functions.get(fitfunc)
        else:
            self.fitfunc = fitfunc

    def fit(self, start, corr, ranges, add=None, oldfit=None, oldfitpar=None):
        """Fits fitfunc to a Correlators object.

        The predefined functions describe a single particle correlation
        function, a ratio of single and two-particle correlation
        functions and a constant function.

        Parameters
        ----------
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
        # check if old data is reused
        if oldfit is None:
            res = fit_single(self.fitfunc, start, corr, ranges, add=add)
        else:
            res = fit_comb(self.fitfunc, start, corr, ranges, add, oldfit, oldfitpar)
        return res

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
    def __init__(self, corr_id):
        """Create FitResults with given identifier.

        Parameters
        ----------
        corr_id : str
            The identifier of the fit results.
        """
        self.data = None
        self.pval = None
        self.chi2 = None
        self.label = None
        self.corr_id = corr_id
        self.corr_num = None
        self.fit_ranges = None
        self.fit_ranges_shape = None

    @classmethod
    def read(cls, filename, corr_id):
        """Read data from file.
        """
        obj = cls()
        tmp = read_fitresults(filename)
        obj.fit_ranges = tmp[0]
        obj.data = tmp[1]
        obj.chi2 = tmp[2]
        obj.pval = tmp[3]
        obj.label = tmp[4]

    def save(self, filename):
        """Save data to disk.

        Parameters
        ----------
        filename : str
            The name of the file.
        """
        write_fitresults(filename, self.fit_ranges, self.data, self.chi2,
            self.pval, self.label, False)

    def add_data(self, index, data, chi2, pval):
        """Add data to FitResult.

        The index contains first the indices of the correlators
        and then the indices of the fit ranges.

        Parameters
        ----------
        index : tuple of int
            The index where to save the data
        data : ndarray
            The fit data to add.
        chi2 : ndarray
            The chi^2 of the data.
        pval : ndarray
            The p-values of the data.

        Raises
        ------
        ValueError
            If Index cannot be calculated.
        RuntimeError
            If FitResult object is not initialized.
        """
        if self.data is None:
            raise RuntimeError("No place to store data, call create_empty first")
        if isinstance(self.corr_num, int):
            if len(index) != 2:
                raise ValueError("Index has wrong length")
            lindex = self._get_index(index[0])
            self.data[lindex][:,:,index[1]] = data
            self.chi2[lindex][:,index[1]] = chi2
            self.pval[lindex][:,index[1]] = pval
        else:
            if len(index) != 2*len(self.corr_num):
                raise ValueError("Index has wrong length")
            lindex = self._get_index(index[:len(self.corr_num)])
            rindex = [slice(None), slice(None)] + [slice(x, x+1) for x in index[len(self.corr_num):]]
            self.data[lindex][rindex] = data
            rindex = [slice(None)] + [slice(x, x+1) for x in index[len(self.corr_num):]]
            self.chi2[lindex][rindex] = chi2
            self.pval[lindex][rindex] = pval


    def _get_index(self, index):
        """Linearize index.

        Parameters
        ----------
        index : int or tuple of ints
            The index to linearize.

        Returns
        -------
        int
            The linearized index.

        Raises
        ------
        ValueError
            If Index cannot be calculated.
        RuntimeError
            If FitResult object is not initialized.
        """
        if self.corr_num is None:
            raise RuntimeError("No place to store data, call create_empty first")

        for n, la in enumerate(self.label):
            if np.array_equal(np.asarray(index), la):
                return n
        else:
            raise ValueError("Index cannot be calculated")

    def create_empty(self, shape1, shape2, corr_num):
        """Create empty data structures.

        If corr_num is a sequence of ints then shape can be a tuple,
        assuming the same shape for all correlators or a sequence,
        assuming different shapes for every correlator.

        Parameters
        ----------
        shape1, shape2 : tuple of ints or sequence of tuples of ints
            Shape of the data structures, where shape1 has an axis for
            the parameters and shape2 not.
        corr_num : int of sequence of ints.
            Number of correlators.

        Raises
        ------
        ValueError
            If shape and corr_num are incompatible.
        """
        self.data = []
        self.pval = []
        self.chi2 = []
        self.label = []
        self.corr_num = corr_num
        if isinstance(corr_num, (tuple, list)):
            # prepare a combination of all possible correlators using
            # list comprehension and itertools
            comb = [[x for x in range(n)] for n in corr_num]
            if isinstance(shape1[0], int):
                # one shape for all correlators
                if len(shape1) != (len(shape2) + 1):
                    raise ValueError("shape1 and shape2 incompatible")
                # iterate over all correlator combinations
                for item in itertools.product(*comb):
                    self.data.append(np.zeros(shape1))
                    self.chi2.append(np.zeros(shape2))
                    self.pval.append(np.zeros(shape2))
                    self.label.append(np.asarray(item))
            else:
                # one shape for every correlator combination
                if len(shape1) != len(shape2):
                    raise ValueError("shape1 and shape2 incompatible")
                if len(shape1) != np.prod(np.asarray(corr_num)):
                    raise ValueError("number of shapes and correlators"\
                            + "incompatible")
                # initialize arrays
                for s1, item in zip(shape1, shape2, itertools.product(*comb)):
                    self.data.append(np.zeros(s1))
                    self.chi2.append(np.zeros(s2))
                    self.pval.append(np.zeros(s2))
                    self.label.append(np.asarray(item))
        # corr_num is an int
        else:
            if isinstance(shape1[0], int):
                if len(shape1) != (len(shape2) + 1):
                    raise ValueError("shape1 and shape2 incompatible")
                # one shape for all correlators
                for i in range(corr_num):
                    self.data.append(np.zeros(shape1))
                    self.chi2.append(np.zeros(shape2))
                    self.pval.append(np.zeros(shape2))
                    self.label.append(np.asarray(i))
            else:
                # one shape for every correlator combination
                if len(shape1) != corr_num:
                    raise ValueError("number of shapes and correlators"\
                            + "incompatible")
                # initialize arrays
                for s1, s2, i in zip(shape1, shape2, range(corr_num)):
                    self.data.append(np.zeros(s1))
                    self.chi2.append(np.zeros(s2))
                    self.pval.append(np.zeros(s2))
                    self.label.append(np.asarray(i))

    def _get_label(self, c_id, c_num):
        tmp = np.ndarray((2,), dtype=object)
        tmp[0] = c_id
        tmp[1] = c_num
        return tmp
    
    def set_ranges(self, ranges, shape):
        self.fit_ranges = ranges
        self.fit_ranges_shape = shape

    def get_ranges(self):
        """Returns the fit ranges."""
        return self.fit_ranges, self.fit_ranges_shape

    def print_data(self):
        for d, c, p in zip(self.data, self.chi2, self.pval):
            mean, std = compute_error(d)
            print(mean[1])
            print(std[1])
            print(c[0])
            print(p[0])

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
