"""
The class for fitting.
"""

import time
import itertools
import numpy as np
import pandas as pd

from fit_routines import (fit_comb, fit_single, calculate_ranges, compute_dE,
    get_start_values, get_start_values_comb, fitting)
from in_out import read_fitresults, write_fitresults, fitres_to_pandas
from interpol import match_lin, match_quad, evaluate_lin
from functions import (func_single_corr,func_single_corr_bare, func_ratio, func_const, func_two_corr,
    func_two_corr_shifted, func_single_corr2, func_sinh, compute_eff_mass,
    func_two_corr_therm, func_corr_shift_therm, func_two_corr_dws,
    func_corr_shift_therm_subtract,func_corr_shift_poll_removal)
from statistics import (compute_error, sys_error, sys_error_der,sys_error_cons, sys_error_der_cons, draw_weighted, freq_count, draw_gauss_distributed)
from energies import calc_q2, calc_Ecm
from zeta_wrapper import Z
from scattering_length import calculate_scat_len, calculate_parametrised_scat_len
from chiral_utils import evaluate_phys
from phaseshift_functions import compute_phaseshift

class LatticeFit(object):
    def __init__(self, fitfunc, dt_i=2, dt_f=2, dt=4, xshift=0.,
            correlated=True, npar=2, debug=0):
        """Create a class for fitting fitfunc.

        Parameters
        ----------
        fitfunc : {0, 1, 2, 3, 4, 5, 6, 7, 8, 9 callable}
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
        conf : list of int, the configurations under investigation
        """
        self.debug = debug
        # chose the correct function if using predefined function
        if isinstance(fitfunc, int):
            if fitfunc > 12:
                raise ValueError("No fit function choosen")
            if fitfunc == 2:
                self.npar = 1
            elif fitfunc == 3:
                self.npar = 3
            elif fitfunc == 6:
                self.npar = 2
            elif fitfunc == 7:
                self.npar = 3
            elif fitfunc == 8:
                self.npar = 3
            elif fitfunc == 9:
                self.npar = 2
            elif fitfunc == 10:
                self.npar = 3
            elif fitfunc == 10:
                self.npar = 2
            elif fitfunc == 11:
                self.npar = 2
            elif fitfunc == 12:
                self.npar = 2
            else:
                self.npar = 2
            functions = {0: func_single_corr, 1: func_ratio, 2: func_const,
                3: func_two_corr, 4: func_single_corr2, 5: func_sinh,
                6: func_two_corr_shifted, 7: func_two_corr_therm,
                8: func_corr_shift_therm, 9: func_single_corr_bare,
                10: func_two_corr_dws, 11: func_corr_shift_therm_subtract,
                12: func_corr_shift_poll_removal}
            self.fitfunc = functions.get(fitfunc)
        else:
            self.npar = npar
            self.fitfunc = fitfunc
        self.xshift = xshift
        self.dt = dt
        self.dt_i = dt_i
        self.dt_f = dt_f
        self.correlated = correlated
        self.conf = None

    def fit(self, start, corr, ranges, corrid="", add=None, oldfit=None,
            oldfitpar=None, useall=False, lint=False):
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
        median : bool
            Adjusts fit ranges of fitresult if median is used

        Returns
        -------
        FitResult
            A class that holds all results.
        """
        # sanity check
        #if isinstance(ranges[0], (tuple, list, np.ndarray)):
        #    for r in ranges:
        #        if r[0] > r[1]:
        #            raise ValueError("final t is smaller than initial t")
        #else:
        #    if ranges[0] > ranges[1]:
        #        raise ValueError("final t is smaller than initial t")

        # check if it is a combined fit or not
        if oldfit is None:
            # no combined fit
            # get the fitranges
            dshape = corr.shape
            ncorr = dshape[-1]
            franges, fshape = calculate_ranges(ranges, dshape, dt_i=self.dt_i,
                    dt_f=self.dt_f, dt=self.dt, debug=self.debug, lintervals=lint)

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
            print self.correlated

            # do the fitting
            for res in fit_single(self.fitfunc, start, corr, franges,
                    add=add, debug=self.debug, correlated=self.correlated,
                    xshift=self.xshift, npar=self.npar):
                fitres.add_data(*res)
        else:
            # handle the fitranges
            dshape = corr.shape
            oldranges, oldshape = oldfit.get_ranges()
            franges, fshape = calculate_ranges(ranges, dshape, oldshape,
                    dt_i=self.dt_i, dt_f=self.dt_f, dt=self.dt,
                    debug=self.debug, lintervals=lint)

            # generate the shapes for the data
            shapes_data = []
            shapes_other = []
            # iterate over the correlation functions
            #print(fshape)
            ncorr = [len(s) for s in fshape]
            if not useall:
                print(ncorr)
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
                #print("ncorr")
                #print(ncorr)
                start = get_start_values_comb(ncorr, franges, corr.data, self.npar)
            # do the fitting
            for res in fit_comb(self.fitfunc, start, corr, franges, fshape,
                    oldfit, add, oldfitpar, useall, self.debug, self.xshift,
                    self.correlated):
                fitres.add_data(*res)
        # get the configuration numbers from the correlator object
        fitres.conf = corr.conf
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
        conf : list of integers, the configuration numbers of the original data
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
        self.conf = None

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
        obj.conf = tmp[6]
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
            self.pval, self.label, self.conf,False)

    def save_h5(self, filename, keyname):
        df = fitres_to_pandas(self,keyname)
        pd.DataFrame.to_hdf(df,filename,keyname)

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

    def get_fr_int(self,range_r,t_min,t_max):
        # get indices for fitranges of interval
        ranges=[]
        # exactly one fitrange
        for s,i in enumerate(range_r[0]):
            if i[0] == t_min and i[1] == t_max:
                ranges.append(s)
            else:
                continue
        #for s,i in enumerate(range_r[0]):
        #  if i[0] >= t_min and i[1] <= t_max:
        #    if i[1]-i[0] >= min_dat:
        #      ranges.append(s)
        #    else:
        #      continue
        return ranges

    def cut_data(self, t_min, t_max, min_dat=7, par=1):
        """ Function to cut data in FitResult object to certain fit ranges
    
        Parameters:
        -----------
        fitres : A FitResult object containing the parameter of interest for all fit
                 ranges
        t_min, t_max : minumum and maximum time of fit ranges
        min_dat : minimum amount of data points in cut
        par : the parameter of the fit result
    
        Returns:
        --------
        fitres_cut : A truncated FitResult object containing only, data, chi2 and pvals for the fit
        ranges of interst
        """
        # Create an empty correlator object for easier plotting
        fitres_cut = FitResult('delta E', derived = False)
        # get fitranges from ratiofit
        range_r, r_r_shape = self.get_ranges()
        ranges=get_fr_int(range_r[0],t_i,t_min)
        # shape for 1 Correlator, data and pvalues
        shape_dE = (self.data[0].shape[0], self.data[0].shape[1], len(ranges))
        shape_pval = (self.pval[0].shape[0], len(ranges))
        shape1 = [shape_dE for dE in self.data]
        shape2 = [shape_pval for p in self.pval]
        fitres_cut.create_empty(shape1, shape2, 1)
    
        # Get data for error calculation
        # 0 in data is for median mass
        dat = self.data[0][:,:,:,ranges]
        pval = self.pval[0][:,:,ranges]
        chi2 = self.chi2[0][:,:,ranges]
        # add data to FitResult
        fitres_cut.data[0] = dat
        fitres_cut.pval[0] = pval
        fitres_cut.chi2[0] = chi2
        #fitres_cut.fit_ranges = range_r[ranges]
        #fitres_cut.add_data((0,),dat,chi2,pval)
    
        return fitres_cut

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
        #print(data.shape)
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

    def append_data(self, index, data, chi2, pval):
        """Append data to FitResult.

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
        #print(data.shape)
        if isinstance(self.corr_num, int):
            if len(index) != 2:
                raise ValueError("Index has wrong length")
            lindex = self._get_index(index[0])
            tmp_index = data.shape[0]*index[1]
            self.data[lindex][tmp_index:tmp_index+data.shape[0],:,0] = data
            self.chi2[lindex][tmp_index:tmp_index+data.shape[0],0] = chi2
            self.pval[lindex][tmp_index:tmp_index+data.shape[0],0] = pval
        #else:
        #    if len(index) != 2*len(self.corr_num):
        #        raise ValueError("Index has wrong length")
        #    lindex = self._get_index(index[:len(self.corr_num)])
        #    if self.derived:
        #        rindex = [slice(None)] + [x for x in index[len(self.corr_num):]]
        #    else:
        #        rindex = [slice(None), slice(None)] + [x for x in index[len(self.corr_num):]]
        #    self.data[lindex][rindex] = data
        #    rindex = [slice(None)] + [x for x in index[len(self.corr_num):]]
        #    self.chi2[lindex][rindex] = chi2
        #    self.pval[lindex][rindex] = pval

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

    def fr_select(self, frange, debug=0):
        """ Select one fit range of self, keep all shapes as is
        1. This makes combined fits faster

        WARNING,
        data gets overwritten

        Parameters:
        -----------
        par : Parameter to use to singularize (usually it is not the Amplitude
        but the first one)
        """
        self.calc_error()
        select = FitResult("one fit range", False)
        nboot = self.data[0].shape[0]
        npars = self.data[0].shape[1]
        if isinstance(frange, int):
            fr_idx = frange
        else:
            # search index corresponding to fit range
            fr_self, fr_self_shape = self.get_ranges()
            print("fit ranges for %s:" % self.corr_id)
            print(fr_self, fr_self.shape)
            fr_idx = get_fr_idx(fr_self[0],frange)
            print(frange)
            print(fr_self[0][fr_idx])
        shape1 = (nboot,npars,1)
        if debug > 0:
          print("is data derived?")
          print(self.derived)
        shape2 = (nboot,1)
        select.create_empty(shape1,shape2,1)
        select.set_ranges(np.array([[[10,15]]]),[[1,]])
        # usually only one correlator is taken into account
        # copy data to singular
        if self.derived is False:
            res, res_std, res_sys, n_fits = self.error[0]
            select.data[0][:,0,0] = self.data[0][:,0,fr_idx]
            res, res_std, res_sys, n_fits = self.error[1]
            select.data[0][:,1,0] = self.data[0][:,1,fr_idx] 
        else:
            res, res_std, res_sys, n_fits = self.error[0]
            select.data[0][:,0,0] = self.data[0][:,0,fr_idx] 
            
        # set weights accordingly
        select.weight = [[np.array([1.])] for d in range(2)]
        select.error = []
        #select.weight = np.ones(shape2)
        #print select.weight[0]
        #select.pval[0]=np.full(nboot,singular.weight)
        return select

    def singularize(self, par=0,debug=0 ):
        """ Set data of self to weighted medians over fit ranges and weights to
        1. This makes combined fits faster

        WARNING,
        data gets overwritten

        Parameters:
        -----------
        par : Parameter to use to singularize (usually it is not the Amplitude
        but the first one)
        """
        self.calc_error()
        singular = FitResult("singular", False)
        ncorr = len(self.data)
        nboot = self.data[0].shape[0]
        if debug > 0:
          print("fitresult is derived: %s" % self.derived)
        if self.derived:
          npars = 1
          # not needed?
          #shape1 = self.data[0].shape
          shape1 = (nboot,1,1)
        else:
          npars = self.data[0].shape[1]
          shape1 = (nboot,npars,1)
        if debug > 0:
          print("is self derived?")
          print(self.derived)
        #print("parameters of fitresult: %d" %npars)
        shape2 = (nboot,1)
        # Get number of correlators self.data is a list of fitresults
        ncorr = len(self.data)
        # One correlator
        if ncorr == 1:
            singular.create_empty(shape1,shape2,1)
            singular.set_ranges(np.array([[[10,15]]]),[[1,]])
            # usually only one correlator is taken into account
            # copy data to singular
            if self.derived is False:
                
                for n in range(npars):
                    if debug > 0:
                      print("FitRes parameter is %d." % n)
                    res, res_std, res_sys, n_fits = self.error[n]
                    singular.data[0][:,n,0] = res[0]
                #res, res_std, res_sys, n_fits = self.error[0]
                #singular.data[0][:,0,0] = res[0]
                #res, res_std, res_sys, n_fits = self.error[1]
                #singular.data[0][:,1,0] = res[0]
                #if npars > 2:
                #      res, res_std, res_sys, n_fits = self.error[2]
                #      singular.data[0][:,2,0] = res[0]
            else:
                for n in range(npars):
                    res, res_std, res_sys, n_fits = self.error[n]
                    singular.data[0][:,n,0] = res[n]
                #res, res_std, res_sys, n_fits = self.error[par]
                #singular.data[0][:,0,0] = res[0]
            if debug > 2:
                print("Original values:")
                print(self.error[0][0])
                print("Singular values:")
                print(singular.data[0][0,0,0])
                
            # set weights accordingly
            singular.weight = [[np.array([1.])] for d in range(2)]
            singular.error = []
        # several correlators
        else:
            singular.create_empty(shape1,shape2,ncorr)
            singular.set_ranges(np.array([[[10,15]]]),[[1,]])
            for n in range(ncorr):
                # usually only one correlator is taken into account
                # copy data to singular
                if self.derived is False:
                    res, res_std, res_sys, n_fits = self.error[0]
                    singular.data[n][:,0,0] = res[n]
                    res, res_std, res_sys, n_fits = self.error[1]
                    singular.data[n][:,1,0] = res[n]
                    if npars > 2:
                        res, res_std, res_sys, n_fits = self.error[2]
                        singular.data[n][:,2,0] = res[n]
                else:
                    res, res_std, res_sys, n_fits = self.error[par]
                    singular.data[n][:,0,0] = res[n]
                if debug > 2:
                    print("Original values:")
                    print(self.error)
                    print("Singular values:")
                    print(singular.data[n][0,0,0])
                    
                # set weights accordingly
                singular.weight = [[np.array([1.])] for d in range(2)]
                singular.error = None
        singular.derived = self.derived
        return singular

    
    def pick_data(self, chi_sq_dof=None, pval=None):
        """ Pick data from multifitrange fitresult

        The data gets selected on either a reduced chi^2 value or the p-value of
        the fit. Either one of chi_sq_dof or pval needs to be set.
        
        Parameters
        ----------
        chi_sq_dof: float, value of chi^2/dof for choice of fitresult
        pval: float, p-value for choice of fitresult
        """
        # debug output
        print("Shapes of data,chi2 and pval:")
        print(self.data[0].shape)
        print(self.chi2[0].shape)
        print(self.pval[0].shape)

        # define default behaviour
        if chi_sq_dof is None and pval is None:
            print("no arguments given, proceeding with pval=0.4")
            pval=0.4
            fitrange_index = get_fitrange_index(self.pval[0][0],pval)
        # get fit range index
        elif chi_sq_dof is not None and pval is None:
            fitrange_index = get_array_index(self.chi2[0][0],chi_sq_dof)
        elif chi_sq_dof is None and pval is not None:
            fitrange_index = get_array_index(self.pval[0][0],pval)
        # based on fitrange_index build new fitresult instance
        fitres_cut = FitResult(self.corr_id,derived=self.derived)
        # determine shape
        shape_data = (self.data[0].shape[0], self.data[0].shape[1], 1) 
        shape_pval = (self.pval[0].shape[0], 1)
        shape1 = [shape_data for d in self.data]
        shape2 = [shape_pval for p in self.pval]
        fitres_cut.create_empty(shape1, shape2, 1)

        # Get data for error calculation
        # 0 in data is for median mass
        try:
            dat = self.data[0][:,:,fitrange_index]
            pval = self.pval[0][:,fitrange_index]
            chi2 = self.chi2[0][:,fitrange_index]
        except:
            dat = self.data[0][:,:,0,fitrange_index]
            pval = self.pval[0][:,0,fitrange_index]
            chi2 = self.chi2[0][:,0,fitrange_index]

        # add data to FitResult
        fitres_cut.data[0] = dat.reshape(dat.shape+(1,))
        fitres_cut.pval[0] = pval.reshape(pval.shape+(1,))
        fitres_cut.chi2[0] = chi2.reshape(chi2.shape+(1,))
        # build fitrange array
        fitranges,fitrange_shape = self.get_ranges()
        _fr = np.zeros((1,1,2))
        _fr[0,0] = fitranges[0,fitrange_index]
        fitres_cut.fit_ranges = _fr
        return fitres_cut
    
        

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

    def calc_error(self, rel=False, conservative=False,fr_disc=None):
        """Calculates the error and weight of data.
        Parameters:
        -----------
          rel : Boolean to control whether the relative error is used
          weight: Boolean, if false a conservative estimate over the fit
                            intervals is made
        """
        if self.error is None:
            self.error = []
            self.weight = []
            if self.derived:
                nfits = [d[0].size for d in self.data]
                #TODO: get rid of this if clause, bad style
                if conservative is False:
                    r, r_std, r_syst, w = sys_error_der(self.data, self.pval)
                    self.error.append((r, r_std, r_syst, nfits))
                    #print(w[0].shape)
                    self.weight.append(w)
                else:
                    r, r_std, r_syst = sys_error_der_cons(self.data,fr_disc)
                    self.error.append((r, r_std, r_syst, nfits))
            else:
                #print(self.data)
                nfits = [d[0,0].size for d in self.data]
                npar = self.data[0].shape[1]
                for i in range(npar):
                    if conservative is False:
                        r, r_std, r_syst, w = sys_error(self.data, self.pval,
                            i,rel=rel)
                        self.error.append((r, r_std, r_syst, nfits))
                        self.weight.append(w)
                    else:
                        r, r_std, r_syst = sys_error_cons(self.data,
                            i,fr_disc=fr_disc)
                        self.error.append((r, r_std, r_syst, nfits))

    def print_data(self, par=0,tex=False):
        #TODO: Latex output not working properly
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
            if np.fabs(r[i][0]) < 1e-4:
                if tex:
                    prec=1e4
                    print("%.4f(%1.0f)(_{-%1.0f}^{+%1.0f})" % (r[i][0],
                      rstd[i], rsys[i][0], rsys[i][1]))
                else:
                    print("%.4e +- %.4e -%.4e +%.4e" % (r[i][0], rstd[i], rsys[i][0], rsys[i][1]))
            else:
                if tex:
                    prec=1e4
                    print("%.4f(%1.0f)(_{-%1.0f}^{+%1.0f})" % (r[i][0],
                      rstd[i], rsys[i][0], rsys[i][1]))
                else:
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
                                                  "par: %e +- %e" %
                                                  (self.data[i][select][0],
                                                    np.std(self.data[i][select]))))
                            print(tmpstring)
        else:
            #print("Print details: data")
            #print(self.data[0])
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
                        # Relative error for tmpstring
                        #rel_err = np.std(self.data[i][:,p,j])/self.data[i][select][0] 
                        tmpstring = " ".join(("%d: range %2d:%2d" % (j, r[0],r[1]),
                                              "chi^2/dof %e" %
                                              (self.chi2[i][0,j]/(r[1]-r[0]+1-self.data[i].shape[1])),
                                              "pval %5f" % (self.pval[i][0,j]),
                                              #"rel. err: %e" % rel_err,
                                              #"p-val*rel.err: %e" 
                                              #%(rel_err*self.pval[i][0,j]),
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
                            # Relative error for tmpstring
                            rel_err = np.std(self.data[i][select])/self.data[i][select][0] 
                            select = (slice(None),) + item + (j,)
                            tmppar = " ".join(tmppar)
                            select = (0,) + item + (j,)
                            frcoord = item+(j,)
                            tmpstring = " ".join(("%d: range %2d:%2d" % (j, r[0],r[1]),
                                                  "add ranges %s" % str(item),
                                                  "chi^2/dof %e" % 
                                                  (self.chi2[i][select]/(r[1]-r[0]+1-self.data[i].shape[1])),
                                                  "pval %5f" % (self.pval[i][select]),
                                                  #"weight %5e" % (self.weight[1][0][frcoord]),
                                                  tmppar))
                            print(tmpstring)
    def reduced_chi2(self):
        r = self.fit_ranges[0,0]
        print(r)
        print(self.chi2[0].shape)
        #red_chi2 =self.chi2[0][0,0]/(r[1]-r[0]-self.data[0].shape[1]) 
        red_chi2 =self.chi2[0][0,0]/(r[1]-r[0]-self.data[0].shape[1]) 
        return red_chi2

    def data_for_plot(self, par=0, new=False):
        """Prints the errors etc of the data."""
        if new is True:
          self.error=None
        self.calc_error()

        if self.derived:
            r, rstd, rsys, nfits = self.error[0]
        else:
            r, rstd, rsys, nfits = self.error[par]
        for i, lab in enumerate(self.label):
            res = np.array((r[i][0], rstd[i], rsys[i][0],
                rsys[i][1]))
        return res

    def calc_cot_delta(self, Ecm, L=24, isdependend=True,
            d2=0, irrep="A1"):
        """Calculate the cotangent of the scattering phase.

        Parameters
        ----------
        mass : FitResult
            The mass of the particle.
        parmass : 
            The parameter of the mass fit to tuse.
        L : int, optional
            The spatial extend of the lattice.
        """
        # we need the weight
        self.calc_error()
        Ecm.calc_error()

        # the shape is the same as self
        newshape = [d.shape for d in self.data]
        delta = FitResult("delta", True)
        delta.create_empty(newshape, newshape, self.corr_num)
        cotdelta = FitResult("cotdelta", True)
        cotdelta.create_empty(newshape, newshape, self.corr_num)
        # the Lorentz boost is saved in Ecm.chi2
        for res, res1 in compute_phaseshift(self.data, self.weight[0], Ecm.chi2,
                Ecm.weight[0], L, isdependend, d2, irrep):
            cotdelta.add_data(*res)
            delta.add_data(*res1)
        return delta, cotdelta

    def calc_cot_delta_twopart(self, mass, parmass=0, L=24, isdependend=False,
            d2=0, irrep="A1"):
        """Calculate the cotangent of the scattering phase if two different
        particles are involved.

        Parameters
        ----------
        mass : tuple of FitResult
            The masses of the particle.
        parmass : 
            The parameter of the mass fit to tuse.
        L : int, optional
            The spatial extend of the lattice.
        """
        if not self.derived or self.corr_id != "Ecm":
            raise RuntimeError("change to center of mass frame first")
        # we need the weight
        self.calc_error()
        mass[0].calc_error()
        mass[1].calc_error()
        _ma0 = mass[0].data[0][:,parmass]
        _ma_w0 = mass[0].weight[parmass]
        _ma1 = mass[0].data[0][:,parmass]
        _ma_w1 = mass[1].weight[parmass]
        nsam = _ma.shape[0]
        if isdependend:
            newshape = [d.shape for d in self.data]
        else:
            newshape = [(nsam,_ma.shape[-1],d.shape[-1]) for d in self.data]
        delta = FitResult("delta", True)
        delta.create_empty(newshape, newshape, [1, len(self.data)])
        for res in compute_phaseshift(self.data, self.weight, _ma, _ma_w, L,
                isdependend, d2, irrep):
            delta.add_data(*res)
        return delta

    def calc_mk_a0_phys(self, val_phys, func, parself=0, parmass=0, isdependend=True):
        """Calculate the physical point result from fitresult parameters 

        Parameters
        ----------
        val_phys : The physical value at which to evaluate, could change to a
            fitresult
        func : callable, chipt function used for evaluation
        parself, parmass : int, optional
            The parameters for which to do this.
        isdependend : bool
            If mass and self are dependend on each other.
        """
        # we need the weight of both mass and self are the fit parameters of a
        # ChiPT fit
        self.calc_error()
        # mass.calc_error()
        _pars = self.data[0]
        _pars_w = self.weight[:][0]
        nsam = self.data[0].shape[0]
        newshape = (nsam, _pars.shape[-1])
        mka0_phys = FitResult("mka0_phys", True)
        mka0_phys.create_empty(newshape, newshape, 1)
        #for res in evaluate_phys(_ma, _ma_w, _energy, _energy_w, isdependend):
        for res in evaluate_phys(val_phys, _pars, _pars_w, func, isdependend):
            mka0_phys.add_data(*res)
        return mka0_phys

    #TODO:  Think about placing this somewhere else
    def calc_mua0_pik_phys(self, x, func,debug =2):
        """Calculate m0ua0 for pi-K from fitted LECs and continuum input
        """ 
        self.calc_error()
        _pars = np.copy(self.data[0])
        # Last parameter is the lattice artefact, want to suppress that for
        # continuum
        #_pars[-1] = np.zeros_like(_pars[0])
        _pars_w = self.weight[:][0]
        _nsam = self.data[0].shape[0]
        _mua0_phys = FitResult("mua0_phys",True)
        _shape = (_nsam,1)
        _mua0_phys.create_empty(_shape, _shape, len(self.data))
        #print("physical x-values:")
        #print(np.asarray((x[0,0],x[0,1],x[0,2],x[0,3],x[0,4])))
        if debug > 0:
            print("parameters")
            print(_pars.shape)
        #print(func(_pars[0],np.asarray((x[0,0],x[0,1],x[0,2],x[0,3],x[0,4]))))
        # Third _pars entry is the lattice artefact
        for res in evaluate_phys(x, _pars[...,0:2,0], _pars_w, func, False):
            _mua0_phys.add_data(*res)
        return _mua0_phys

    def calc_dE(self, mass, parself=0, parmass=0, flv_diff=False, isdependend=True):
        """Calculate dE from own data and the mass of the particles.

        Parameters
        ----------
        mass : FitResult
            The masses of the single particles.
        parself, parmass : int, optional
            The parameters for which to do this. Only relevant if fitresults are
            not derived
        isdependend : bool
            If mass and self are dependend on each other.
        flv_diff : bool
            If particles are different, then mass is the sum of the particle
            masses
        """
        # we need the weight of both mass and self
        self.calc_error()
        mass.calc_error()
        # get the mass of the single particles, assuming the
        # first entry of the mass FitResults contains them.
        if mass.derived is False:
            _ma = mass.data[0][:,parmass]
            _ma_w = mass.weight[parmass][0]
        else:
            _ma = mass.data[0]
            _ma_w = mass.weight[0][0]
        if self.derived is False:
            _energy = self.data[0][:,parself]
            _energy_w = self.weight[parself][0]
        else:
            _energy = self.data[0]
            _energy_w = self.weight[0][0]

        nsam = self.data[0].shape[0]
        newshape = (nsam, _ma.shape[-1], _energy.shape[-1])
        print(newshape)
        dE = FitResult("dE", True)
        dE.create_empty(newshape, newshape, [1,1])
        print(_energy[0,0],_ma[0,0])
        for res in compute_dE(_ma, _ma_w, _energy, _energy_w, isdependend,
                              flv_diff=flv_diff):
            dE.add_data(*res)
        return dE

    def calc_scattering_length(self, mass, parself=0, parmass=0, L=24,
            isratio=False, isdependend=True,rf_est=None):
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
        truncated : bool
            If energy has only one fit range dimension
        isdependend : bool
            If mass and self are dependend on each other.
        """
        # we need the weight of both mass and self
        self.calc_error()
        mass.calc_error()
        # get the data
        if mass.derived:
            _mass = mass.data[0]
            _massweight = mass.weight[0][0]
        else:
            _mass = mass.data[0][:,parmass]
            _massweight = mass.weight[parmass][0]
        if self.derived:
            _energy = self.data[0]
            _energyweight = self.weight[0][0]
        else:
            _energy = self.data[0][:,parself]
            _energyweight = self.weight[parself][0]
        print(_energy[0,0])
        nsam = _mass.shape[0]
        # create the new shapes
        scatshape = (nsam, _mass.shape[-1], _energy.shape[-1])
        scatshape_w = scatshape
        # prepare storage
        scat = FitResult("scat_len", True)
        scat.create_empty(scatshape, scatshape_w, [1,1])
        # calculate scattering length
        print("_energy has shape:")
        print _energy.shape
        print("_mass has shape:")
        print(_mass.shape)
        for res in calculate_scat_len(_mass, _massweight, _energy, _energyweight,
                L, isdependend, isratio, rf_est):
            scat.add_data(*res)
        return scat
#TODO: Code doubling, safer against errors 
    def calc_parametrised_scattering_length(self, mass, parself=0, parmass=0, L=24,
            isdependend=True):
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
        truncated : bool
            If energy has only one fit range dimension
        isdependend : bool
            If mass and self are dependend on each other.
        """
        # we need the weight of both mass and self
        self.calc_error()
        mass.calc_error()
        # get the data
        if mass.derived:
            _mass = mass.data[0]
            _massweight = mass.weight[0][0]
        else:
            _mass = mass.data[0][:,parmass]
            _massweight = mass.weight[parmass][0]
        if self.derived:
            _energy = self.data[0]
            _energyweight = self.weight[0][0]
        else:
            _energy = self.data[0][:,parself]
            _energyweight = self.weight[parself][0]
        print(_energy[0,0])
        nsam = _mass.shape[0]
        # create the new shapes
        scatshape = (nsam, _mass.shape[-1], _energy.shape[-1])
        scatshape_w = scatshape
        # prepare storage
        scat = FitResult("scat_len", True)
        scat.create_empty(scatshape, scatshape_w, [1,1])
        # calculate scattering length
        print("_energy has shape:")
        print _energy.shape
        print("_mass has shape:")
        print(_mass.shape)
        for res in calculate_parametrised_scat_len(_mass, _massweight, _energy, _energyweight,
                L, isdependend):
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
                if isinstance(self.label[i], np.ndarray):
                    if self.label[i].ndim == 0:
                        tmp = (self.label[i].item(),)
                    else:
                        tmp = tuple(self.label[i])
                elif not isinstance(self.label[i]):
                    tmp = tuple(self.label[i])
                else:
                    tmp = self.label[i]
                #if np.any(res > 4*0.14463):
                #    print("%s: Ecm over 4*mpi" % str(tmp+item))
                Ecm.add_data(tmp + item, res, gamma, weight)
        return Ecm

    def calc_momentum(self, mass, parmass, L=24, uselattice=True, isdependend=False):
        """Calculate the lattice momentum of the system.

        Parameters
        ----------
        mass : FitResult, the particle mass, possibly several masses stacked
               along the last axis of the data array
        parmass: int, parameter of fitresult for mass data
        L : int, optional
            The lattice size.
        uselattice : bool, optional
            Use the lattice formulas or the continuum formulas (two particle
            case only implemented for non-lattice versions).
        """
        self.calc_error()

        _ma = mass.data[0][:,parmass]
        _ma_w = mass.weight[parmass][0]
        q2 = FitResult("q2", True)
        if isdependend:
            newshapes = [p.shape for p in self.pval]
            q2.create_empty(newshapes, newshapes, self.corr_num)
        else:
            newshapes = [(d.shape[0],_ma.shape[-1])+tuple(d.shape[1:]) for d in self.data]
            q2.create_empty(newshapes, newshapes, [1, self.corr_num])
        nsamples = self.data[0].shape[0]
        needed = np.zeros((nsamples,))
        if isdependend:
            for i, data in enumerate(self.data):
                weight = (_ma_w * self.weight[0][i].T).T
                for n in range(data.shape[-1]):
                    res = calc_q2(data[...,n], _ma, L=L, lattice=uselattice)
                    #print(res.shape)
                    for m in range(data.shape[-2]):
                        tmp = weight[m,n] * np.ones((nsamples,))
                        q2.add_data((0, i, m, n), res[:,m], needed, tmp)
        else:
            for i, data in enumerate(self.data):
                for n in range(data.shape[-1]):
                    for m in range(_ma.shape[-1]):
                        res = calc_q2(data[...,n], _ma[...,m], L=L, lattice=uselattice)
                        weight = np.ones((nsamples,))*_ma_w[...,m]*self.weight[0][i][n]
                        q2.add_data((0, i, m, n), res, needed, weight)
        return q2

    def evaluate_quark_mass(self, amu_s, obs_eval, obs1, obs2=None, obs3=None,
        meth=0, parobs=1, combine_all=True, debug=0):
      """ evaluate the strange quark mass at obs_match

      Parameters
      ----------
      obs1, obs2, obs3: Up to 3 different Observables are supported at the
          moment (should be easy to extend). Every Observable is a FitResult
          object

      meth: How to match: 0: linear interpolation (only two values)
                          1: linear fit
                          2: quadratic interpolation
      parobs : int
               Which parameter of the results should be taken
      combine_all : Boolean
                    Whether or not all combinations of fit ranges are taken
      """
      if obs2==None and obs3==None:
        raise ValueError("Matching not possible, check input of 2nd (and 3rd) observable!")
      #if obs3==None:
      # Get the we
      # Result has the same layout as one of the observables!
      # TODO: If observables have different layouts break
      shape1 = obs1.data[0].shape
      shape2 = obs1.pval[0].shape
      # prepare shapes
      if combine_all:
        dim_fr1 = shape1[-1]
        dim_fr2 = obs2.data[0].shape[-1]
        layout1 = (shape1[0],dim_fr1*dim_fr2)
      else:
        layout1 = (shape1[0],shape1[-1])

      _obs1 = obs1.data[0][:,parobs]
      _obsweight1 = obs1.pval[0][0]
      if obs2 is not None:
        _obs2 = obs2.data[0][:,parobs]
        _obsweight2 = obs2.pval[0][0]
      if obs3 is not None:
        _obs3 = obs3.data[0][:,parobs]
        _obsweight3 = obs3.pval[0][0]
      _obs_eval = obs_eval
      if debug > 0:
        print("observable to evaluate at")
        print(_obs_eval)

      self.create_empty(layout1, layout1, 1)
      # Decide method beforehand, cheaper in the end

      if meth == 0:
        for res in evaluate_lin(_obs1, _obs2, amu_s, _obsweight1,
            _obsweight2, _obs_eval,combine_all=combine_all):
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
        meth=0, parobs=1, combine_all = True, debug=0):
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
      if obs2==None and obs3==None:
        raise ValueError("Matching not possible, check input of 2nd (and 3rd) observable!")
      #if obs3==None:
      # Get the we
      # Result has the same layout as one of the observables!
      # TODO: If observables have different layouts break
      shape1 = obs1.data[0].shape
      shape2 = obs1.pval[0].shape
      # prepare shapes
      if combine_all:
        dim_fr1 = shape1[-1]
        dim_fr2 = obs2.data[0].shape[-1]
        dim_fr3 = obs3.data[0].shape[-1]
        # if we need 3 observables more fit ranges needed
        if meth > 0:
          layout1 = (shape1[0],dim_fr1*dim_fr2*dim_fr3)
        else:
          layout1 = (shape1[0],dim_fr1*dim_fr2)
      else:
        layout1 = (shape1[0],shape1[-1])

      _obs1 = obs1.data[0][:,parobs]
      _obsweight1 = obs1.pval[0][0]
      if obs2 is not None:
        _obs2 = obs2.data[0][:,parobs]
        _obsweight2 = obs2.pval[0][0]
      if obs3 is not None:
        _obs3 = obs3.data[0][:,parobs]
        _obsweight3 = obs3.pval[0][0]
      _obs_match = obs_match
      if debug > 0:
        print("observable to evaluate at")
        print(_obs_match)

      self.create_empty(layout1, layout1, 1)
      # Decide method beforehand, cheaper in the end

      if meth == 0:
        for res in match_lin(_obs1, _obs2, amu_s, _obsweight1,
            _obsweight2, _obs_match, combine_all):
            self.add_data(*res)

      if meth == 1:
        for res in match_quad(_obs1, _obs2, _obs3, _obsweight1,
            _obsweight2, _obsweight3, amu_s, _obs_match, combine_all):
            self.add_data(*res)

      if meth == 2:
        for res in match_fit(_obs1, _obs2, _obs3, _obsweight1,
            _obsweight2, _obsweight3, amu_s, _obs_match):
            self.add_data(*res)

    def mult_obs(self, other, corr_id="Product", isdependend=False):
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
      if isdependend:
          print(other.data[0][0].shape)
          if other.derived:
              if layout[1] != other.data[0].shape[1]:
                raise ValueError("Number of same parameter fit ranges not compatible!\n"
                    + "%d vs. %d" % (layout[1], other.data[0].shape[1]))
          else:
              if layout[1] != other.data[0][0].shape[1]:
                raise ValueError("Number of same parameter fit ranges not compatible!\n"
                    + "%d vs. %d" % (layout[1], other.data[0][0].shape[1]))
      # Deal with observable
      product = np.zeros_like(self.data[0])
      for b, arr0 in enumerate(other.data[0]):
          if self.derived is False:
              for r_self, arr1 in enumerate(arr0[1]):
                  print(r_self,arr1,arr0)
                  product[b][r_self] = np.multiply(arr1, self.data[0][b][r_self])
          else:
              product[b] = np.multiply(arr0.reshape(arr0.size,1), self.data[0][b])
              

      # Deal with observable weights
      # Get weights for all fit ranges (1 sample is sufficient)
      weights_0 = self.pval[0][0]
      # prepare new weight array
      weights_prod = np.zeros_like(weights_0)
      # multiply weights of first observable with observable of second
      if isdependend:
        for idx, weights_1 in enumerate(other.weight[0][0]):
           weights_prod[idx] = np.multiply(weights_1, weights_0[idx])
      else:
        for idx, weights_1 in enumerate(other.weight):
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
        vals : 
            The chosen weights as result of draw_weighted

        Returns
        -------
        res_sorted :
            The intersected data and weights as a new FitResult object.
        
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
          print ndim
          print self.data[0][:,1].shape
          flat_data = self.data[0][:,1].reshape((boots,ndim))
          flat_weights = self.pval[0][0].reshape(ndim)

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
        print intersect
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

    def fse_multiply(self, mean, std):
        """Do finite size corrections to the data."""
        # loop over principal correlators
        for d in self.data:
            # get the needed shape of samples
            shape = (d.shape[0],)
            # get bootstrap samples of corrections
            fse = draw_gauss_distributed(mean, std, shape)
            for x in np.nditer(d, op_flags=["readwrite"], 
                    flags=["external_loop"], order="F"):
                x[...]=x*fse
        if self.error is not None:
            self.error = None
            self.calc_error()

    def fse_divide(self, mean, std):
        """Do finite size corrections to the data."""
        # loop over principal correlators
        for d in self.data:
            # get the needed shape of samples
            shape = (d.shape[0],)
            # get bootstrap samples of corrections
            fse = draw_gauss_distributed(mean, std, shape)
            for x in np.nditer(d, op_flags=["readwrite"], 
                    flags=["external_loop"], order="F"):
                x[...]=x/fse
        if self.error is not None:
            self.error = None
            self.calc_error()

    def fse_add(self, mean, std):
        """Do finite size corrections to the data."""
        # loop over principal correlators
        for d in self.data:
            # get the needed shape of samples
            shape = (d.shape[0],)
            # get bootstrap samples of corrections
            fse = draw_gauss_distributed(mean, std, shape)
            for x in np.nditer(d, op_flags=["readwrite"], 
                    flags=["external_loop"], order="F"):
                x[...]=x + fse
        if self.error is not None:
            self.error = None
            self.calc_error()

    def fse_subtract(self, mean, std):
        """Do finite size corrections to the data."""
        # loop over principal correlators
        for d in self.data:
            # get the needed shape of samples
            shape = (d.shape[0],)
            # get bootstrap samples of corrections
            fse = draw_gauss_distributed(mean, std, shape)
            for x in np.nditer(d, op_flags=["readwrite"], 
                    flags=["external_loop"], order="F"):
                x[...]=x - fse
        if self.error is not None:
            self.error = None
            self.calc_error()

    # new function interface with class        
    def comb_fitres(self,res1,par):
        """Combine parameters of a fitresult in a new fitresult
    
        The data of several fitresults are placed in a new FitResult object
        combining all fitranges with each other. The p-value of every combination is
        determined as the product of the individual p-values. The fit parameter axis
        of the combined FitResult hosts the different parameters of the old
        FitResulut objects.
    
        Parameter
        ---------
        fitres : list of FitResult, several fitresults with the same parameter
                  number and (possibly) different fit ranges. For energies sort
                  after lowest first.
        par : parameter to use
    
        Returns
        -------
        _fitres : a combined FitResult, with the parameters on the parameter axis
                  and combined fit ranges
        """
        # Gather necessary data from list of fitresults
        # data have to have the same number of bootstrapsamples
        fitres = [self,res1]
        nboot = fitres[0].data[0].shape[0]
        npars = len(fitres)
        # calculate the number of fitranges
        nbranges = 1
        for f in fitres:
          nbranges *=f.data[0].shape[2]
        print("number of fitranges is %d" % nbranges)
        shape1 = (nboot,npars,nbranges)
        shape2 = (nboot,nbranges)
        ncorr=1
        # Initialize an empty fitresult
        _comb = FitResult("combined",False)
        _comb.create_empty(shape1,shape2,ncorr)
        _comb.set_ranges([[[10,15] for r in range(nbranges)]],[[nbranges,]])
        # Fill the fitresult
        # Loop o'er fitresult list
        # With itertools get product of all fitrange index combinations
        # a list of 
        friter = [[r for r in range(f.data[0].shape[-1])] for f in fitres]
        for i,item in enumerate(itertools.product(*friter)):
            pval = np.ones_like(_comb.pval[0][:,0])
            # loop over fitrange combination j is fitres entry, r is
            # fitrange index
            for j,r in enumerate(item):
                _comb.data[0][:,j,i] = fitres[j].data[0][:,par,r]
                pval *= fitres[j].pval[0][:,r]
            _comb.pval[0][:,i] = pval
        _comb.calc_error()
    
        return _comb

    def reduced_mass(self,mass, par=1):
        """Calculate reduced mass mu = m1*m2/(m1+m2) for different particles

        The function takes one additional fitresult argument and calculates mu. It
        takes care of several fitrange combinations and correlators

        Parameters
        ----------
        Returns
        ----------
        _mu : FitRes, the reduced mass returned as a fitresult
        """
        # Gather necessary data from list of fitresults
        # data have to have the same number of bootstrapsamples
        fitres = [self,mass]
        nboot = fitres[0].data[0].shape[0]
        npars = 1
        # calculate the number of fitranges
        nbranges = 1
        for f in fitres:
          nbranges *=f.data[0].shape[2]
        print("number of fitranges is %d" % nbranges)
        shape1 = (nboot,nbranges)
        ncorr=1
        # Initialize an empty fitresult
        _mu = FitResult("reduced_mass",True)
        _mu.create_empty(shape1,shape1,ncorr)
        _mu.set_ranges([[[10,15] for r in range(nbranges)]],[[nbranges,]])
        # Fill the fitresult
        # Loop o'er fitresult list
        # With itertools get product of all fitrange index combinations
        # a list of 
        friter = [[r for r in range(f.data[0].shape[-1])] for f in fitres]
        # loop over combinations of fitranges, i is index of fitrange, item is
        # array of fitrange indices from single masses
        for i,item in enumerate(itertools.product(*friter)):
            # loop over fitrange combination j is fitres entry, r is
            # fitrange index
            _mu.data[0][:,i] = fitres[0].data[0][:,par,item[0]]*fitres[1].data[0][:,par,item[1]]/(fitres[0].data[0][:,par,item[0]]+fitres[1].data[0][:,par,item[1]])
            _mu.pval[0][:,i] = np.ones((nboot,))*fitres[0].weight[par][0][item[0]]*fitres[1].weight[par][0][item[1]]
        return _mu

    def summ_int(self, pars, fac, fac_par):
        """add different parameters of the same fitresult

        The function takes one additional fitresult argument and calculates mu. It
        takes care of several fitrange combinations and correlators

        Parameters
        ----------
        Returns
        ----------
        _mu : FitRes, the reduced mass returned as a fitresult
        """

        # Calculate errors and weights on all observables
        self.calc_error()
        #print mass.weight[0][0]
        # Gather necessary data from list of fitresults
        # data have to have the same number of bootstrapsamples
        nboot = self.data[0].shape[0] 
        npars = 1
        # calculate the number of fitranges
        nbranges = 1
        nbranges = self.data[0].shape[2]
        print("number of fitranges is %d" % nbranges)
        shape1 = (nboot,npars,nbranges)
        ncorr=1
        # Initialize an empty fitresult
        _sum = FitResult("mass_sum",True)
        _sum.create_empty(shape1,shape1,ncorr)
        # FitResult weights are a double list of (nboot,fitrange) arrays. first
        # list index is for parameters, secon is for correlator
        #_sum.pval = [[np.zeros((nbranges)) for c in range(ncorr)]for n
        #    in range(npars)]
        _sum.set_ranges([[[10,15] for r in range(nbranges)]],[[nbranges,]])
        # loop over combinations of fitranges, i is index of fitrange, item is
        # array of fitrange indices from single masses
        # loop over fitrange combination j is fitres entry, r is
        # fitrange index
        # calculate sum of masses for each fitrange combination
        # determine array for factor
        _fac = np.ones_like(self.data[0],dtype='float')
        _fac[:,fac_par]=fac
        print(_fac[0,1,0])
        _tmp = self.data[0]*_fac
        for fr in range(nbranges):
            for p in pars:
                print("summ_int: par")
                print(_tmp[0])
                _sum.data[0][:,0,fr] += _tmp[:,p,fr]
                _sum.pval[0][:,fr] = np.ones((nboot,1))*np.square(self.weight[p][0][fr])
        return _sum

    def add_mass(self, mass, par=1,neg=False):
        """add one mass to the mass of self m += m1 for different particles

        The function takes one additional fitresult argument and calculates mu. It
        takes care of several fitrange combinations and correlators

        Parameters
        ----------
        neg: bool, if True difference is calculated
        Returns
        ----------
        _mu : FitRes, the reduced mass returned as a fitresult
        """

        # Calculate errors and weights on all observables
        self.calc_error()
        mass.calc_error()
        #print mass.weight[0][0]
        # Gather necessary data from list of fitresults
        # data have to have the same number of bootstrapsamples
        nboot = self.data[0].shape[0] 
        npars = 1
        # calculate the number of fitranges
        nbranges = 1
        fitres = [self,mass]
        for f in fitres:
          nbranges *=f.data[0].shape[2]
        print("number of fitranges is %d" % nbranges)
        shape1 = (nboot,nbranges)
        ncorr=1
        # Initialize an empty fitresult
        _sum = FitResult("mass_sum",True)
        _sum.create_empty(shape1,shape1,ncorr)
        # FitResult weights are a double list of (nboot,fitrange) arrays. first
        # list index is for parameters, secon is for correlator
        #_sum.pval = [[np.zeros((nbranges)) for c in range(ncorr)]for n
        #    in range(npars)]
        _sum.set_ranges([[[10,15] for r in range(nbranges)]],[[nbranges,]])
        # Fill the fitresult
        # Loop o'er fitresult list
        # With itertools get product of all fitrange index combinations
        # a list of 
        friter = [[r for r in range(f.data[0].shape[-1])] for f in fitres]
        # loop over combinations of fitranges, i is index of fitrange, item is
        # array of fitrange indices from single masses
        for i,item in enumerate(itertools.product(*friter)):
            # loop over fitrange combination j is fitres entry, r is
            # fitrange index
            # calculate sum of masses for each fitrange combination
            if neg is False: 
                _sum.data[0][:,i] = fitres[0].data[0][:,par,item[0]]+fitres[1].data[0][:,par,item[1]]
            else:
                _sum.data[0][:,i] = fitres[0].data[0][:,par,item[0]]-fitres[1].data[0][:,par,item[1]]
            _sum.pval[0][:,i] = np.ones((nboot,))*fitres[0].weight[par][0][item[0]]*fitres[1].weight[par][0][item[1]]
        return _sum

def init_fitreslst(fnames):
  """Read fitresults from a list of filenames and return the list
  """
  reslst = []
  for f in fnames:
    tmp = FitResult.read(f)
    reslst.append(tmp)
  return reslst


def get_fr_idx(search,find):
    
    # loop over outer dimension of search
    idx=[]
    for i,r in enumerate(search):
      if r[0] == find[0]:
        if r[1] == find[1]:
          idx.append(i)
    if len(idx) == 0:
      idx.append(0)
      print("fit range not found")
    return idx[0]

def get_array_index(array,value):
    return np.argmin(np.abs(array-value))

if __name__ == "__main__":
    pass
