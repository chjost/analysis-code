"""
The class for fitting.
"""

import itertools
import numpy as np

from fit_routines import (fit_comb, fit_single, calculate_ranges, compute_dE,
    compute_phaseshift, get_start_values, get_start_values_comb)
from in_out import read_fitresults, write_fitresults
from functions import (func_single_corr, func_ratio, func_const, func_two_corr,
    func_single_corr2, compute_eff_mass)
from statistics import compute_error, sys_error, sys_error_der
from energies import calc_q2, calc_Ecm
from zeta_wrapper import Z
from scattering_length import calculate_scat_len

class LatticeFit(object):
    def __init__(self, fitfunc, dt_i=2, dt_f=2, dt=4, xshift=0.,
            correlated=True, debug=0):
        """Create a class for fitting fitfunc.

        Parameters
        ----------
        fitfunc : {0, 1, 2, 3, callable}
            Choose between three predefined functions or an own
            fit function.
        dt_i, dt_f : ints, optional
            The step size for the first and last time slice for the fitting.
        dt : int, optional
            The minimal size of the interval.
        xshift : float, optional
            A shift for the x values of the fit
        correlated : bool
            Use the full covariance matrix or just the errors.
        debug : int, optional
            The level of debugging output
        """
        self.debug = debug
        # chose the correct function if using predefined function
        if isinstance(fitfunc, int):
            if fitfunc > 4:
                raise ValueError("No fit function choosen")
            if fitfunc == 2:
                self.npar = 1
            elif fitfunc == 3:
                self.npar = 3
            else:
                self.npar = 2
            functions = {0: func_single_corr, 1: func_ratio, 2: func_const,
                3: func_two_corr, 4: func_single_corr2}
            self.fitfunc = functions.get(fitfunc)
        else:
            self.fitfunc = fitfunc
        self.xshift = xshift
        self.dt = dt
        self.dt_i = dt_i
        self.dt_f = dt_f
        self.correlated = correlated

    def fit(self, start, corr, ranges, corrid="", add=None, oldfit=None,
            oldfitpar=None, useall=False):
        """Fits fitfunc to a Correlators object.

        The predefined functions describe a single particle correlation
        function, a ratio of single and two-particle correlation
        functions and a constant function.

        Parameters
        ----------
        start : float or sequence of floats or None
            The start parameters for the fit. If None is given the start
            parameters are calculated.
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
            franges, fshape = calculate_ranges(ranges, dshape, dt_i=self.dt_i,
                    dt_f=self.dt_f, dt=self.dt, debug=self.debug)

            # prepare storage
            fitres = FitResult(corrid)
            fitres.set_ranges(franges, fshape)
            shapes_data = [(dshape[0], self.npar, fshape[0][i]) for i in range(ncorr)]
            shapes_other = [(dshape[0], fshape[0][i]) for i in range(ncorr)]
            fitres.create_empty(shapes_data, shapes_other, ncorr)
            del shapes_data, shapes_other

            if start is None:
                # set starting values
                start = get_start_values(ncorr, franges, corr.data, self.npar)

            # do the fitting
            for res in fit_single(self.fitfunc, start, corr, franges,
                    add=add, debug=self.debug, correlated=self.correlated):
                fitres.add_data(*res)
        else:
            # handle the fitranges
            dshape = corr.shape
            oldranges, oldshape = oldfit.get_ranges()
            franges, fshape = calculate_ranges(ranges, dshape, oldshape,
                    dt_i=self.dt_i, dt_f=self.dt_f, dt=self.dt,
                    debug=self.debug)

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
                shapes_data.append((dshape[0], self.npar) + tuple(tmp))
                shapes_other.append((dshape[0],) + tuple(tmp))
            # prepare storage
            fitres = FitResult(corrid)
            fitres.set_ranges(franges, fshape)
            fitres.create_empty(shapes_data, shapes_other, ncorr)
            del shapes_data, shapes_other

            if start is None:
                start = get_start_values_comb(ncorr, franges, corr.data, self.npar)
            # do the fitting
            for res in fit_comb(self.fitfunc, start, corr, franges, fshape,
                    oldfit, add, oldfitpar, useall, self.debug,
                    self.correlated):
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
        obj = cls(tmp[0][0], tmp[0][3])
        obj.fit_ranges = tmp[1]
        obj.data = tmp[2]
        obj.chi2 = tmp[3]
        obj.pval = tmp[4]
        obj.label = tmp[5]
        obj.corr_num = tmp[0][1]
        obj.fit_ranges_shape = tmp[0][2]
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

    def calc_error(self):
        """Calculates the error and weight of data."""
        if self.error is None:
            self.error = []
            self.weight = []
            if self.derived:
                nfits = [d[0].size for d in self.data]
                r, r_std, r_syst, w = sys_error_der(self.data, self.pval)
                self.error.append((r, r_std, r_syst, nfits))
                self.weight.append(w)
            else:
                nfits = [d[0,0].size for d in self.data]
                npar = self.data[0].shape[1]
                for i in range(npar):
                    r, r_std, r_syst, w = sys_error(self.data, self.pval, i)
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
            print("%.5f +- %.5f -%.5f +%.5f" % (r[i][0], rstd[i], rsys[i][0],
                rsys[i][1]))
        print("------------------------------\n\n")

    def print_details(self):
        """Prints details for every fit."""
        print("------------------------------")
        print("details for %s" % self.corr_id)
        if self.derived:
            # iterate over the correlators
            for i, lab in enumerate(self.label):
                print("correlator %s" % (str(lab)))
                if self.data[i].ndim < 3:
                    for j in range(self.data[i].shape[-1]):
                        # create a string containing the fit parameters
                        tmpstring = " ".join(("%2d:" % (j),
                                              "weight %e" % (self.pval[i][0,j]),
                                              "par: %e" % (self.data[i][0,j])))
                        print(tmpstring)
                else:
                    for j in range(self.data[i].shape[-1]):
                        # iterate over additional fit intervals
                        nintervals = self.data[i].shape[1:-1]
                        ninteriter = [[x for x in range(n)] for n in nintervals]
                        for item in itertools.product(*ninteriter):
                            # create a string containing the fit parameters
                            select = (slice(None),) + item + (j,)
                            tmpstring = " ".join(("%2d:" % (j),
                                                  "add ranges %s" % str(item),
                                                  "weight %e" % (self.pval[i][select][0]),
                                                  "par: %e +- %e" % (self.data[i][select][0], np.std(self.data[i][select]))))
                            print(tmpstring)
        else:
            # iterate over the correlators
            for i, lab in enumerate(self.label):
                print("correlator %s" % (str(lab)))
                if self.data[i].ndim < 4:
                    for j, r in enumerate(self.fit_ranges[i]):
                        # create a string containing the fit parameters
                        tmppar = ["par:"]
                        for p in range(self.data[i].shape[1]):
                          tmppar.append("%e +- %e" % (self.data[i][0,p,j], np.std(self.data[i][:,p,j])))
                        tmppar = " ".join(tmppar)
                        tmpstring = " ".join(("%d: range %2d:%2d" % (j, r[0],r[1]),
                                              "chi^2 %e" % (self.chi2[i][0,j]),
                                              "pval %5f" % (self.pval[i][0,j]),
                                              tmppar))
                        print(tmpstring)
                else:
                    for j, r in enumerate(self.fit_ranges[i]):
                        # iterate over additional fit intervals
                        nintervals = self.data[i].shape[2:-1]
                        ninteriter = [[x for x in range(n)] for n in nintervals]
                        for item in itertools.product(*ninteriter):
                            # create a string containing the fit parameters
                            tmppar = ["par:"]
                            for p in range(self.data[i].shape[1]):
                                select = (slice(None), p) + item + (j,)
                                tmppar.append("%e" % (self.data[i][select])[0])
                            tmppar = " ".join(tmppar)
                            tmpstring = " ".join(("%d: range %2d:%2d" % (j, r[0],r[1]),
                                                  "add ranges %s" % str(item),
                                                  "chi^2 %e" % (self.chi2[i][select][0]),
                                                  "pval %5f" % (self.pval[i][select][0]),
                                                  tmppar))
                            print(tmpstring)

    def calc_cot_delta(self, mass, parmass=0, L=24):
        """Calculate the cotangent of the scattering phase.

        Warning
        -------
        This overwrites the data, so be careful to save the data before.

        Parameters
        ----------
        mass : FitResult
            The mass of the particle.
        parmass : 
            The parameter of the mass fit to tuse.
        L : int, optional
            The spatial extend of the lattice.
        """
        if not self.derived or self.corr_id != "Ecm":
            raise RuntimeError("change to center of mass frame first")
        # we need the weight
        self.calc_error()
        mass.calc_error()
        _ma = mass.data[0][:,parmass]
        _ma_w = mass.weight[parmass][0]
        nsam = self.data[0].shape[0]
        newshape = [(nsam,) + _ma.shape[1:] + d.shape[2:] for d in self.data]
        delta = FitResult("delta", True)
        delta.create_empty(newshape, newshape, [1, len(self.data)])
        for res in compute_phaseshift(self.data, self.weight, _ma, _ma_w, L):
            delta.add_data(*res)
        return delta
        ## loop over fitranges of self
        #for i in range(_data.shape[-1]):
        #    # loop over fitranges of mass
        #    for j in range(_ma.shape[-1]):
        #        if isratio:
        #            q2 = ((_data[:,j,i]*_data[:,j,i]/4.+_data[:,j,i]*_ma[:,j]) *
        #                  (2. * np.pi) / float(L))
        #        else:
        #            q2 = calc_q2(_data[:,i], _ma[:,j], L)
        #        cotd[0][:,j,i] = Z(q2).real / (np.pow(np.pi, 3./2.) * np.sqrt(q2))
        #        if isratio:
        #            cotd_w[0][j,i] = _ma_w[j] * _data_w[j,i]
        #        else:
        #            cotd_w[0][j,i] = _ma_w[j] * _data_w[i]
        #np.save("cotd_test.npy", cotd[0])
        #np.save("cotd_w_test.npy", cotd_w[0])
        #res, std, syst = sys_error_der(cotd, cotd_w)
        #print(res[0][0])
        #print(std[0])
        #print(syst[0])

    def calc_dE(self, mass, parself=0, parmass=0, isdependend=True):
        """Calculate dE from own data and the mass of the particles.

        Parameters
        ----------
        mass : FitResult
            The masses of the single particles.
        parself, parmass : int, optional
            The parameters for which to do this.
        isdependend : bool
            If mass and self are dependend on each other.
        """
        # we need the weight of both mass and self
        self.calc_error()
        mass.calc_error()
        # get the mass of the single particles, assuming the
        # first entry of the mass FitResults contains them.
        _ma = mass.data[0][:,parmass]
        _ma_w = mass.weight[parmass][0]
        _energy = self.data[0][:,parself]
        _energy_w = self.weight[parself][0]
        nsam = self.data[0].shape[0]
        newshape = (nsam, _ma.shape[-1], _energy.shape[-1])
        dE = FitResult("dE", True)
        dE.create_empty(newshape, newshape, [1,1])
        for res in compute_dE(_ma, _ma_w, _energy, _energy_w, isdependend):
            dE.add_data(*res)
        return dE

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
        isdependend : bool
            If mass and self are dependend on each other.
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
        return scat

    def to_CM(self, par, L=24, d=np.array([0., 0., 1.]), uselattice=True):
        """Transform data to center of mass frame.

        Parameters
        ----------
        par : int
            Which of the fit parameters to transform.
        L : int, optional
            The lattice size.
        d : ndarray, optional
            The total momentum vector of the system.
        uselattice : bool, optional
            Use the lattice formulas or the continuum formulas.
        """
        if self.derived:
            return
        self.calc_error()
        newshapes = [p.shape for p in self.pval]
        Ecm = FitResult("Ecm", True)
        Ecm.create_empty(newshapes, newshapes, self.corr_num)
        nsamples = self.data[0].shape[0]
        for i, data in enumerate(self.data):
            ranges = [[x for x in range(n)] for n in data.shape[2:]]
            for item in itertools.product(*ranges):
                select = (slice(None), par) + item
                gamma, res = calc_Ecm(data[select], d=d, L=L, lattice=uselattice)
                weight = np.ones(nsamples) * self.weight[par][i][item]
                if not isinstance(self.label[i], tuple):
                    tmp = (self.label[i],)
                else:
                    tmp = self.label[i]
                #if np.any(res > 4*0.14463):
                #    print("%s: Ecm over 4*mpi" % str(tmp+item))
                Ecm.add_data(tmp + item, res, gamma, weight)
        return Ecm

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
