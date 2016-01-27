"""
The class for fitting.
"""

import itertools
import numpy as np

from fit_routines import fit_comb, fit_single, calculate_ranges
from in_out import read_fitresults, write_fitresults
from interpol import match_lin, evaluate_lin
from functions import func_single_corr, func_ratio, func_const, func_two_corr
from statistics import compute_error, sys_error, sys_error_rel, sys_error_der, draw_weighted, freq_count
from energies import calc_q2
from zeta_wrapper import Z
from scattering_length import calculate_scat_len

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
            if fitfunc > 3:
                raise ValueError("No fit function chosen")
            functions = {0: func_single_corr, 1: func_ratio, 2: func_const,
                3: func_two_corr}
            self.fitfunc = functions.get(fitfunc)
        else:
            self.fitfunc = fitfunc

    def fit(self, start, corr, ranges, corrid="", add=None, oldfit=None,
            oldfitpar=None, useall=False, step=1, min_size=4, xshift=0.,
            debug=0, fixend=False):
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
        step : int, optional
            The steps in the loops.
        min_size : int, optional
            The minimal size of the interval.
        xshift : A scalar to shift the xrange of the fit. 
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
            franges, fshape = calculate_ranges(ranges, dshape, step=step,
                    min_size=min_size, debug=debug, fixend=fixend)

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
            franges, fshape = calculate_ranges(ranges, dshape, oldshape,
                    step=step, min_size=min_size, debug=debug, fixend=fixend)

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
                        oldfit, None, oldfitpar, xshift=xshift, debug=debug):
                    fitres.add_data(*res)
            else:
                for res in fit_comb(self.fitfunc, start, corr, franges, fshape,
                        oldfit, add, oldfitpar, xshift=xshift, debug=debug):
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
    def __init__(self, corr_id, derived=False):
        """Create FitResults with given identifier.

        Parameters
        ----------
        corr_id : str
            The identifier of the fit results.
        derived : bool
            If the data is derived or not.
        """
        self.data = None
        self.pval = None
        self.chi2 = None
        self.label = None
        self.corr_id = corr_id
        self.corr_num = None
        self.fit_ranges = None
        self.fit_ranges_shape = None
        self.derived = derived
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
            if self.derived:
                self.data[lindex][:, index[1]] = data
            else:
                self.data[lindex][:,:,index[1]] = data
            self.chi2[lindex][:,index[1]] = chi2
            self.pval[lindex][:,index[1]] = pval
        else:
            if len(index) != 2*len(self.corr_num):
                raise ValueError("Index has wrong length")
            lindex = self._get_index(index[:len(self.corr_num)])
            if self.derived:
                rindex = [slice(None)] + [x for x in index[len(self.corr_num):]]
            else:
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

        try:
            if len(index) == 1:
                index = index[0]
        except TypeError:
            pass
        for n, la in enumerate(self.label):
            if np.array_equal(np.asarray(index), la):
                return n
        else:
            raise ValueError("Index cannot be calculated")

    def singularize(self, val):
        singular = FitResult("singluar", True)
        shape1 = self.data[0].shape
        shape2 = shape1
        singular.create_empty(shape1,shape2,1)
        nboot = self.data[0].shape[0]
        data = np.linspace(val,val,nboot)
        chi2 = np.zeros_like(data)
        pval = np.ones_like(data)
        singular.add_data((0,0),np.linspace(val,val,nboot),chi2,pval)
        return singular

    def create_empty(self, shape1, shape2, corr_num):
        """Create empty data structures.

        If corr_num is a sequence of ints then shape can be a tuple,
        assuming the same shape for all correlators or a sequence,
        assuming different shapes for every correlator.

        Parameters
        ----------
        shape1, shape2 : tuple of ints or sequence of tuples of ints
        #TODO: The described layout did not work for me (Christopher), do I use
        it wrongly?
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
                if (self.derived == False and len(shape1) != (len(shape2)+1)):
                    raise ValueError("shape1 and shape2 incompatible")
                elif (self.derived == True and len(shape1) != len(shape2)): 
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
                if (self.derived == False and len(shape1) != (len(shape2)+1)):
                    raise ValueError("shape1 and shape2 incompatible")
                elif (self.derived == True and len(shape1) != len(shape2)):
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

    def set_ranges(self, ranges, shape):
        self.fit_ranges = ranges
        self.fit_ranges_shape = shape

    def get_ranges(self):
        """Returns the fit ranges."""
        return self.fit_ranges, self.fit_ranges_shape

    def calc_error(self, rel=False):
        """Calculates the error and weight of data.
        Parameters:
        -----------
          rel : Boolean to control whether the relative error is used
        """
        if self.error is None:
            self.error = []
            self.weight = []
            if self.derived:
                nfits = [d[0].size for d in self.data]
            else:
                nfits = [d[0,0].size for d in self.data]
            if self.derived:
                if rel is False:
                  r, r_std, r_syst, w = sys_error_der(self.data, self.pval)
                else:
                  r, r_std, r_syst, w = sys_error_der_rel(self.data, self.pval)
                self.error.append((r, r_std, r_syst, nfits))
                self.weight.append(w)
            else:
                npar = self.data[0].shape[1]
                for i in range(npar):
                    if rel is False:
                      r, r_std, r_syst, w = sys_error(self.data, self.pval,
                          i)
                    else:
                      r, r_std, r_syst, w = sys_error_rel(self.data, self.pval,
                          i)
                    self.error.append((r, r_std, r_syst, nfits))
                    self.weight.append(w)

    def print_data(self, par=0):
        """Prints the errors etc of the data."""
        self.calc_error()

        print("------------------------------")
        print("summary for %s" % self.corr_id)
        if self.derived:
            print("derived values")
            r, rstd, rsys, nfits = self.error[0]
        else:
            print("parameter %d" % par)
            r, rstd, rsys, nfits = self.error[par]
        for i, lab in enumerate(self.label):
            print("correlator %s, %d fits" %(str(lab), nfits[i]))
            print("%.8f +- %.8f -%.5f +%.5f" % (r[i][0], rstd[i], rsys[i][0],
                rsys[i][1]))
        print("------------------------------\n\n")

    def data_for_plot(self, par=0):
        """Prints the errors etc of the data."""
        self.calc_error()

        #print("------------------------------")
        #print("summary for %s" % self.corr_id)
        if self.derived:
            #print("derived values")
            r, rstd, rsys, nfits = self.error[0]
        else:
            #print("parameter %d" % par)
            r, rstd, rsys, nfits = self.error[par]
        for i, lab in enumerate(self.label):
            #print("correlator %s, %d fits" %(str(lab), nfits[i]))
            res = np.array((r[i][0], rstd[i], rsys[i][0],
                rsys[i][1]))
        return res
        #print("------------------------------\n\n")

    def calc_cot_delta(self, mass, parself=0, parmass=0, L=24, isratio=False):
        """Calculate the cotangent of the scattering phase.

        Warning
        -------
        This overwrites the data, so be careful to save the data before.

        Parameters
        ----------
        mass : FitResult
            The masses of the single particles.
        parself, parmass : int, optional
            The parameters for which to do this.
        L : int
            The spatial extend of the lattice.
        isratio : bool
            If self is already the ratio.
        """
        # we need the weight of both mass and self
        self.calc_error()
        mass.calc_error()
        # get the mass of the single particles, assuming the
        # first entry of the mass FitResults contains them.
        _ma = mass.data[0][:,parmass]
        _ma_w = mass.weight[parmass][0]
        _data = self.data[0][:,parself]
        _data_w = self.weight[parself][0]
        nsam = _ma.shape[0]
        cotd = [np.zeros((nsam, _ma.shape[-1], _data.shape[-1]))]
        cotd_w = [np.zeros((_ma.shape[-1], _data.shape[-1]))]
        # loop over fitranges of self
        for i in range(_data.shape[-1]):
            # loop over fitranges of mass
            for j in range(_ma.shape[-1]):
                if isratio:
                    q2 = ((_data[:,j,i]*_data[:,j,i]/4.+_data[:,j,i]*_ma[:,j]) *
                          (2. * np.pi) / float(L))
                else:
                    q2 = calc_q2(_data[:,i], _ma[:,j], L)
                cotd[0][:,j,i] = Z(q2).real / (np.pow(np.pi, 3./2.) * np.sqrt(q2))
                if isratio:
                    cotd_w[0][j,i] = _ma_w[j] * _data_w[j,i]
                else:
                    cotd_w[0][j,i] = _ma_w[j] * _data_w[i]
        np.save("cotd_test.npy", cotd[0])
        np.save("cotd_w_test.npy", cotd_w[0])
        res, std, syst = sys_error_der(cotd, cotd_w)
        print(res[0][0])
        print(std[0])
        print(syst[0])

    def calc_dE(self, mass, parself=0, parmass=0):
        """Calculate dE from own data and the mass of the particles.

        Parameters
        ----------
        mass : FitResult
            The masses of the single particles.
        parself, parmass : int, optional
            The parameters for which to do this.
        """
        # we need the weight of both mass and self
        self.calc_error()
        mass.calc_error()
        # get the mass of the single particles, assuming the
        # first entry of the mass FitResults contains them.
        _ma = mass.data[0][:,parmass]
        _ma_w = mass.weight[parmass][0]
        _dE = []
        _dE_w = []
        nsamples = self.data[0].shape[0]
        for i, d in enumerate(self.data):
            # create the empty arrays
            _dE.append(np.zeros((nsamples,)+_ma.shape[1:]+d.shape[2:]))
            _dE_w.append(np.zeros(_ma.shape[1:]+d.shape[2:]))
            len1 = len(_ma.shape[1:])
            # iterate over the new array
            niter = [[x for x in range(n)] for n in _dE[-1].shape[1:]]
            for item in itertools.product(*niter):
                s = d[(slice(None), parself)+item[len1:]]
                a = _ma[(slice(None),)+item[:len1]]
                _dE[-1][(slice(None),)+item] = s - 2. * a
                _dE_w[-1][item] = (_ma_w[item[:len1]] *
                        self.weight[parself][i][item[len1:]])
        self.dE = _dE
        self.dE_w = _dE_w
        res, std, syst = sys_error_der(self.dE, self.dE_w)
        print(res)
        print(std)
        print(syst)

    #def sort_res(self, )
    #    """Function resorting a FitResult object by a specific axis

    #    Parameters
    #    ----------
    #    """

    def calc_scattering_length(self, mass, parself=0, parmass=0, L=24,
            isratio=False, isdependend=True):
        """Calculate the scattering length.
        This only makes sense for correlation functions with no momentum.

        Warning
        -------
        This overwrites the data, so be careful to save the data before.

        Parameters
        ----------
        mass : FitResult
            The masses of the single particles.
        parself, parmass : int, optional
            The parameters for which to do this.
        L : int
            The spatial extend of the lattice.
        isratio : bool
            If self is already the ratio.
        """
        # we need the weight of both mass and self
        self.calc_error()
        mass.calc_error()
        # get the data
        _mass = mass.data[0][:,parmass]
        _massweight = mass.weight[parmass][0]
        _energy = self.data[0][:,parself]
        _energyweight = self.weight[parself][0]
        nsam = _mass.shape[0]
        # create the new shapes
        scatshape = (nsam, _mass.shape[-1], _energy.shape[-1])
        scatshape_w = scatshape
        # prepare storage
        scat = FitResult("scat_len", True)
        scat.create_empty(scatshape, scatshape_w, [1,1])
        # calculate scattering length
        for res in calculate_scat_len(_mass, _massweight, _energy, _energyweight,
                L, isdependend, isratio):
            scat.add_data(*res)
        #print(scat.pval.shape)
        return scat

    def evaluate_quark_mass(self, amu_s, obs_eval, obs1, obs2=None, obs3=None,
        meth=0):
      """ evaluate the strange quark mass at obs_match

      Parameters
      ----------
      obs1, obs2, obs3: Up to 3 different Observables are supported at the
          moment (should be easy to extend). Every Observable is a FitResult
          object

      meth: How to match: 0: linear interpolation (only two values)
                          1: linear fit
                          2: quadratic interpolation

      """
      if obs2==None and obs3==None:
        raise ValueError("Matching not possible, check input of 2nd (and 3rd) observable!")
      #if obs3==None:
      # Get the we
      # Result has the same layout as one of the observables!
      # TODO: If observables have different layouts break
      layout = obs1.data[0].shape
      print(layout)
      _obs1 = obs1.data[0]
      _obsweight1 = obs1.pval[0][0]
      if obs2 is not None:
        _obs2 = obs2.data[0]
        _obsweight2 = obs2.pval[0][0]
      if obs3 is not None:
        _obs3 = obs3.data[0]
        _obsweight3 = obs3.pval[0][0]
      _obs_eval = obs_eval
      print("observable to evaluate at")
      print(_obs_eval)

      boots = layout[0] 
      ranges1 = layout[1]
      if obs1.data[0].ndim == 3:
        ranges2 = layout[2]
      else:
        ranges2 = 0
      self.create_empty(layout, layout, 1)
      # Decide method beforehand, cheaper in the end

      if meth == 0:
        for res in evaluate_lin(_obs1, _obs2, amu_s, _obsweight1,
            _obsweight2, _obs_eval):
            self.add_data(*res)

      if meth == 1:
        for res in evaluate_quad(_obs1, _obs2, _obs3, _obsweight1,
            _obsweight2, _obsweight3, amu_s, obs_match):
            self.add_data(*res)

      if meth == 2:
        for res in evaluate_fit(_obs1, _obs2, _obs3, _obsweight1,
            _obsweight2, _obsweight3, amu_s, obs_match):
            self.add_data(*res)

    def match_quark_mass(self, amu_s, obs_match, obs1, obs2=None, obs3=None,
        meth=0, evaluate=False):
      """ Match the strange quark mass to an observable in lattice units.

      Parameters
      ----------
      obs1, obs2, obs3: Up to 3 different Observables are supported at the
          moment (should be easy to extend). Every Observable is a FitResult
          object

      meth: How to match: 0: linear interpolation (only two values)
                          1: linear fit
                          2: quadratic interpolation

      """
      if obs2==None and obs3==None:
        raise ValueError("Matching not possible, check input of 2nd (and 3rd) observable!")
      #if obs3==None:
      # Get the we
      # Result has the same layout as one of the observables!
      # TODO: If observables have different layouts break
      layout = obs1.data[0].shape
      print(layout)
      _obs1 = obs1.data[0]
      _obsweight1 = obs1.pval[0][0]
      if obs2 is not None:
        _obs2 = obs2.data[0]
        _obsweight2 = obs2.pval[0][0]
      if obs3 is not None:
        _obs3 = obs3.data[0]
        _obsweight3 = obs3.pval[0][0]
      if evaluate is True:
        _obs_match = obs_match.data[0]
        _obs_match_weight = obs_match.pval[0][0]
      else:
        _obs_match = obs_match
        _obs_match_weight = None
      print("observable to match")
      print(_obs_match)

      boots = layout[0] 
      ranges1 = layout[1]
      if obs1.data[0].ndim == 3:
        ranges2 = layout[2]
      else:
        ranges2 = 0
      self.create_empty(layout, layout, 1)
      # Decide method beforehand, cheaper in the end

      if meth == 0:
        for res in match_lin(_obs1, _obs2, amu_s, _obsweight1,
            _obsweight2, _obs_match, _obs_match_weight, evaluate):
            self.add_data(*res)

      if meth == 1:
        for res in match_quad(_obs1, _obs2, _obs3, _obsweight1,
            _obsweight2, _obsweight3, amu_s, obs_match):
            self.add_data(*res)

      if meth == 2:
        for res in match_fit(_obs1, _obs2, _obs3, _obsweight1,
            _obsweight2, _obsweight3, amu_s, obs_match):
            self.add_data(*res)


    def mult_obs(self, other, corr_id="Product"):
      """Multiply two observables in order to treat them as a new observable.

        Warning
        -------
        This overwrites the data, so be careful to save the data before.
      
      Parameters
      ----------
      other: FitResult object that gets multiplied with self in a
          weightpreserving way.
      corr_id: Id of derived observable

      """
      # Self determines the resulting layout
      layout = self.data[0].shape
      boots = layout[0] 
      ranges1 = layout[1]
      if self.data[0].ndim == 3:
        ranges2 = layout[2]
      else:
        ranges2 = 0
      
      # Check ranges and samples for compliance
      if layout[0] != other.data[0].shape[0]:
        raise ValueError("Number of Bootstrapsamples not compatible!")
      if layout[1] != other.data[0][0].shape[1]:
        raise ValueError("Number of same parameter fit ranges not compatible!\n"
            + "%d vs. %d" % (layout[1], other.data[0][0].shape[1]))

      # Deal with observables
      product = np.zeros_like(self.data[0])
      for b, arr0 in enumerate(other.data[0]):
        for r_self, arr1 in enumerate(arr0[1]):
          product[b][r_self] = np.multiply(arr1, self.data[0][b][r_self])

      # Deal with observable weights
      # Get weights for all fit ranges (1 sample is sufficient)
      weights_0 = self.pval[0][0]
      # prepare new weight array
      weights_prod = np.zeros_like(weights_0)
      # multiply weights of first observable with observable of second
      for idx, weights_1 in enumerate(other.weight[1][0]):
        weights_prod[idx] = np.multiply(weights_1, weights_0[idx])

      # Get array into right shape for calculation
      weights = np.tile( weights_prod.flatten(), boots ).reshape(boots, ranges1,
          ranges2)

      # prepare storage
      mult_obs = FitResult(corr_id, True)
      mult_obs.create_empty(layout, layout, [1,1])
      mult_obs.data[0] = product
      mult_obs.pval[0] = weights
      return mult_obs

    def mult_obs_single(self, other, corr_id="Product"):
      """Multiply two observables in order to treat them as a new observable.

        Warning
        -------
        This overwrites the data, so be careful to save the data before.
      
      Parameters
      ----------
      other: FitResult object that gets multiplied with self in a
          weightpreserving way.
      corr_id: Id of derived observable

      """
      # Self determines the resulting layout
      layout = self.data[0].shape
      boots = layout[0] 
      ranges1 = layout[2]
      if self.data[0].ndim == 3:
        ranges2 = layout[2]
      else:
        ranges2 = 0
      
      # Check ranges and samples for compliance
      if layout[0] != other.data[0].shape[0]:
        raise ValueError("Number of Bootstrapsamples not compatible!")
      if layout[1] != other.data[0].shape[1]:
        raise ValueError("Number of same parameter fit ranges not compatible!\n"
            + "%d vs. %d" % (layout[1], other.data[0].shape[1]))

      # Deal with observables
      product = np.zeros_like(self.data[0])
      for b, arr0 in enumerate(other.data[0]):
        for r_self, arr1 in enumerate(arr0):
          product[b][r_self] = np.multiply(arr1, self.data[0][b][r_self])

      # Deal with observable weights
      # Get weights for all fit ranges (1 sample is sufficient)
      weights_0 = self.pval[0][0]
      # prepare new weight array
      weights_prod = np.zeros_like(weights_0)
      # multiply weights of first observable with observable of second
      for idx, weights_1 in enumerate(other.weight[1][0]):
        weights_prod[idx] = np.multiply(weights_1, weights_0[idx])

      # Get array into right shape for calculation
      weights = np.tile( weights_prod.flatten(), boots ).reshape(boots, ranges1)

      # prepare storage
      mult_obs = FitResult(corr_id, True)
      mult_obs.create_empty(layout, layout, [1,1])
      mult_obs.data[0] = product
      mult_obs.pval[0] = weights
      return mult_obs
    def res_reduced(self, samples=20, corr_id='reduced', m_a0 = False):
      """Take boolean 1d intersection of two arrays to choose certain fitranges and
      corresponding data.
  
      This function flattens the data wrt the fit ranges. Thus the structure of
      different observables will get lost.
      
      Parameters
      ----------
          vals : The chosen weights as result of draw_weighted
      
      Returns:
          res_sorted : The intersected data and weights as a new FitResult object.
      
      """
      # Self determines the resulting layout
      layout = self.data[0].shape
      boots = layout[0] 

      # Reshape data in dependence of correlator type
      if self.derived == True:
        if m_a0 is True:
          ndim = self.data[0].shape[1]*self.data[0].shape[2]
          flat_data = self.data[0].reshape((boots,ndim))
          flat_weights = self.pval[0][0].reshape(ndim)
        else:
          ndim = self.data[0].shape[2]
          print ndim
          print self.data[0][:,1].shape
          flat_data = self.data[0][:,1].reshape((boots,ndim))
          flat_weights = self.pval[0][0].reshape(ndim)
      else:
        ndim = self.data[0].shape[2]
        flat_data = self.data[0][:,1].reshape((boots,ndim))
        self.calc_error()
        flat_weights = self.weight[1]

      vals = draw_weighted(flat_weights, samples=samples)
      ranges = vals.shape[0]
      # Get frequency count of sorted vals 
      freq_vals = freq_count(vals, verb=False)
      # Create empty fitresult to add data
      res_sorted = FitResult(corr_id, derived=True)
      store1 = (boots, ranges)
      store2 = (boots,ranges)
      res_sorted.create_empty(store1, store2 ,1)
      # get frequencies and indices in original data
      intersect = np.zeros_like(freq_vals)
      # replace first column
      wght_draw_unq = freq_vals[:,0]
      intersect[:,0] = np.asarray(np.nonzero(np.in1d(flat_weights, wght_draw_unq)))
      intersect[:,1] = freq_vals[:,1]
      # TODO: solve this by an iterator
      ind=0
      for i,v in enumerate(intersect):
        for cnt in range(int(v[1])):
          targ_ind = (0,ind)
          weight = np.tile(freq_vals[i,0],boots)
          data = flat_data[:,v[0]]
          chi2_dummy = np.zeros_like(weight)
          res_sorted.add_data(targ_ind,data,chi2_dummy,weight)
          ind += 1

      return res_sorted
     
