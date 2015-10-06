"""
The class for fitting.
"""


import itertools
import numpy as np

from fit_routines import fit_comb, fit_single, calculate_ranges
from in_out import read_fitresults, write_fitresults
from functions import func_single_corr, func_ratio, func_const
from statistics import compute_error, sys_error

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

    def fit(self, start, corr, ranges, corrid="", add=None, oldfit=None,
            oldfitpar=None, useall=False, debug=0):
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
        corrid : str, optional
            Identifier of the fit result.
        add : None or ndarray, optional
            Additional parameters for the fit function.
        oldfitpar : None, int or sequence of int, optional
            Which parameter of the old fit to use, if there is more
            than one.
        useall : bool
            Using all correlators in the single particle correlator or
            use just the lowest.
        debug : int, optional
            The amount of info printed.

        Returns
        -------
        FitResult
            A class that holds all results.
        """
        # check if it is a combined fit or not
        if oldfit is None:
            # no combined fit
            # get the fitranges
            dshape = corr.shape
            ncorr = dshape[-1]
            franges, fshape = calculate_ranges(ranges, dshape)

            # prepare storage
            fitres = FitResult(corrid)
            fitres.set_ranges(franges, fshape)
            shapes_data = [(dshape[0], len(start), fshape[0][i]) for i in range(ncorr)]
            shapes_other = [(dshape[0], fshape[0][i]) for i in range(ncorr)]
            fitres.create_empty(shapes_data, shapes_other, ncorr)
            del shapes_data, shapes_other

            # do the fitting
            if add is None:
                for res in fit_single(self.fitfunc, start, corr, franges,
                        debug=debug):
                    fitres.add_data(*res)
            else:
                for res in fit_single(self.fitfunc, start, corr, franges, add,
                        debug):
                    fitres.add_data(*res)
        else:
            # handle the fitranges
            dshape = corr.shape
            oldranges, oldshape = oldfit.get_ranges()
            franges, fshape = calculate_ranges(ranges, dshape, oldshape)

            # generate the shapes for the data
            shapes_data = []
            shapes_other = []
            # iterate over the correlation functions
            ncorr = [len(s) for s in fshape]
            if not useall:
                ncorr[-2] = 1

            ncorriter = [[x for x in range(n)] for n in ncorr]
            for item in itertools.product(*ncorriter):
                # create the iterator over the fit ranges
                tmp = [fshape[i][x] for i,x in enumerate(item)]
                shapes_data.append(tuple([dshape[0], len(start)] + tmp))
                shapes_other.append(tuple([dshape[0]] + tmp))

            # prepare storage
            fitres = FitResult(corrid)
            fitres.set_ranges(franges, fshape)
            fitres.create_empty(shapes_data, shapes_other, ncorr)
            del shapes_data, shapes_other

            # do the fitting
            if add is None:
                for res in fit_comb(self.fitfunc, start, corr, franges, fshape,
                        oldfit, oldfitpar, debug=debug):
                    fitres.add_data(*res)
            else:
                for res in fit_comb(self.fitfunc, start, corr, franges, fshape,
                        oldfit, add, oldfitpar, debug=debug):
                    fitres.add_data(*res)

        return fitres

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
        self.derived = False
        self.error = None
        self.weight = None

    @classmethod
    def read(cls, filename):
        """Read data from file.
        """
        tmp = read_fitresults(filename)
        obj = cls(tmp[0][0])
        obj.fit_ranges = tmp[1]
        obj.data = tmp[2]
        obj.chi2 = tmp[3]
        obj.pval = tmp[4]
        obj.label = tmp[5]
        obj.corr_num = tmp[0][1]
        obj.fit_ranges_shape = tmp[0][2]
        obj.derived = tmp[0][3]
        return obj

    def save(self, filename):
        """Save data to disk.

        Parameters
        ----------
        filename : str
            The name of the file.
        """
        tmp = np.empty((4,), dtype=object)
        tmp[0] = self.corr_id
        tmp[1] = self.corr_num
        tmp[2] = self.fit_ranges_shape
        tmp[3] = self.derived
        write_fitresults(filename, tmp, self.fit_ranges, self.data, self.chi2,
            self.pval, self.label, False)

    def get_data(self, index):
        """Returns the data at the index.

        Parameters
        ----------
        index : tuple of int
            The index of the data.

        Returns
        -------
        ndarray
            The data.
        """
        if self.data is None:
            raise RuntimeError("No data stored, add data first")
        if isinstance(self.corr_num, int):
            if len(index) != 2:
                raise ValueError("Index has wrong length")
            lindex = self._get_index(index[0])
            return self.data[lindex][:,:,index[1]]
        else:
            if len(index) != 2*len(self.corr_num):
                raise ValueError("Index has wrong length")
            lindex = self._get_index(index[:len(self.corr_num)])
            rindex = [slice(None), slice(None)] + [x for x in index[len(self.corr_num):]]
            return self.data[lindex][rindex]

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
            rindex = [slice(None), slice(None)] + [x for x in index[len(self.corr_num):]]
            self.data[lindex][rindex] = data
            rindex = [slice(None)] + [x for x in index[len(self.corr_num):]]
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
        if self.data is not None:
            raise RuntimeError("already initialized!")
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
                for s1, s2, item in zip(shape1, shape2, itertools.product(*comb)):
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

    def calc_error(self):
        """Calculates the error and weight of data."""
        if self.error is None:
            npar = self.data[0].shape[1]
            self.error = []
            self.weight = []
            nfits = [d[0,0].size for d in self.data]
            for i in range(npar):
                if self.derived:
                    r, r_std, r_syst, w = sys_error_der(self.data, self.weight, i)
                else:
                    r, r_std, r_syst, w = sys_error(self.data, self.pval, i)
                self.error.append((r, r_std, r_syst, nfits))
                self.weight.append(w)

    def print_data(self, par=0):
        """Prints the errors etc of the data."""
        self.calc_error()

        print("summary for %s" % self.corr_id)
        r, rstd, rsys, nfits = self.error[par]
        for i, lab in enumerate(self.label):
            print("correlator %s, %d fits" %(str(lab), nfits[i]))
            print("%.5f +- %.5f -%.5f +%.5f" % (r[i], rstd[i], rsys[i][0], rsys[i][1]))

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
