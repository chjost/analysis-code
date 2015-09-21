# a class to contain the fit results

import numpy as np

from analyze_fcts import calc_error
from ensemble import LatticeEnsemble
from fit import genfit, genfit_comb, set_fit_interval
from fit import fit as fit1
from plot import genplot, genplot_comb
from input_output import write_fitresults, read_fitresults
from module_global import multiprocess

# this function circumvents the problem that only top-level functions
# of a module can be pickled, which is needed for the multiprocessing
# to work
def fitting_func(args, kwargs):
    return fit1(args, kwargs)

class FitResults(object):
    """class to hold fit results.

    Nothing is immutable, so be careful!
    """
    def _depth(self, var):
        return isinstance(var, list) and max(map(self._depth, var)) + 1

    def __init__(self, ensemble, label):
        if not isinstance(ensemble, LatticeEnsemble):
            raise TypeError("FitResults expected LatticeEnsemble, got %s" % 
                (type(ensemble)))
        # always needed
        self.ensemble = ensemble
        self.name = ensemble.get_data("name")
        self.label = label
        self.depth = 0
        self.fitranges = []
        self.verbose = False
        self.old_data = None
        self.data = None
        self.fitfunc = None
        # needed for combined fit
        self.combfit = False
        self.par = None
        self.par_index = None
        # results
        self.res = None
        self.pvals = None
        self.chi2 = None

    @classmethod
    def combined_fit(cls, prev_fr, label, fitrange, par_index=0):
        """Initialize a new combined fit using the results of
        prev_fr.
        """
        tmp = cls(prev_fr.ensemble, label)
        tmp.prepare_combined_fit(fitrange, prev_fr.fitranges[0],
            prev_fr.data, par_index)
        return tmp

    @classmethod
    def from_file(cls, ensemble, label, filename):
        """Initialize a fit from a file."""
        raise NotImplementedError()
        tmp = cls(ensemble, label)
        res = read_fitresults(filename)
        if len(res) == 4:
            cls.set_fit(res)
        elif len(res) == 5:
            cls.set_fit_comb(res)
        else:
            raise RuntimeError("Cannot make sense initializing fit from file")
        return tmp

    def toggle_verbose(self):
        if self.verbose:
            self.verbose = False
        else:
            self.verbose = True

    def add_fitrange(self, fitrange):
        """Add a fitrange for fitting a single fit"""
        if len(self.fitranges) == 0:
            self.fitranges.append(np.asarray(fitrange))
        else:
            raise RuntimeError("%s already has a fitrange, cannot add another")

    def add_fitranges(self, fitrange_data, fitrange_par):
        """Add two fitrange for fitting a combined fit.
        Args:
            fitint_data: List of intervals for the fit of the functions.
            fitint_par: List of intervals for the varying parameter
        """
        if len(self.fitranges) == 0:
            self.fitranges.append(np.asarray(fitrange_data))
            self.fitranges.append(np.asarray(fitrange_par))
            self.combfit = True
        else:
            raise RuntimeError("%s already has a fitrange, cannot add another"%\
                               self.__repr__)

    def set_fitrange(self, _data, lo, upi, step=2):
        """Set fit interval"""
        self.data = np.atleast_3d(_data)
        self.add_fitrange(set_fit_interval(_data, lo, up, skip))

    def add_par(self, par, par_index=0):
        """Add parameters for a combined fit and the index needed."""
        self.par = par
        self.par_index = par_index
        self.combfit = True

    def use_old_data(self, old_data):
        """Reuse the data located at 'old_data' if possible"""
        self.old_data = old_data

    def prepare_fit(self, fitrange, old_data=None):
        """Set everything needed for a fit."""
        self.comb_fit = False
        self.add_fitrange(fitrange)
        self.use_old_data(old_data)

    def prepare_combined_fit(self, fitrange_data, fitrange_par, par,
          par_index=0, old_data=None):
        """Set everything needed for a combined fit."""
        self.comb_fit = True
        self.add_fitranges(fitrange_data, fitrange_par)
        self.add_par(par, par_index)
        self.use_old_data(old_data)

    def do_fit(self, _data, fitfunc, start_params):
        if self.data is not None:
            if not (self.data==_data).all():
                raise RuntimeError("Fitresult has already data which is" +
                    "compatible with new data")
        else:
            self.data = np.atleast_3d(_data)
        # init variables
        nboot = self.data.shape[0]
        T2 = self.data.shape[1]
        ncorr = self.data.shape[2]
        npar = len(start_params)
        ninter = [len(fitint) for fitint in self.fitranges[0]]
        # set fit data
        tlist = np.linspace(0., float(T2), float(T2), endpoint=False)
        # initialize empty arrays
        self.res = []
        self.chi2 = []
        self.pval = []
        func_args = []
        func_kwargs = []
        # initialize array for every principal correlator
        for _l in range(ncorr):
            self.res.append(np.zeros((nboot, npar, ninter[_l])))
            self.chi2.append(np.zeros((nboot, ninter[_l])))
            self.pval.append(np.zeros((nboot, ninter[_l])))
        def ffunc(args, kwargs):
            return fit1(fitfunc, args, kwargs)
        for _l in range(ncorr):
            # setup
            #mdata, ddata = calc_error(data[:,:,_l])
            for _i in range(ninter[_l]):
                lo, up = self.fitranges[0][_l][_i]
                if self.verbose:
                    print("Interval [%d, %d]" % (lo, up))
                    print("correlator %d" % _l)

                # fit the energy and print information
                if self.verbose:
                    print("fitting correlation function")
                    print(tlist[lo:up+1])
                func_args.append((tlist[lo:up+1], self.data[:,lo:up+1,_l], start_params))
                y=len(func_kwargs)
                func_kwargs.append({"num":y, "verbose":False})
                #res[_l][:,:,_i], chi2[_l][:,_i], pval[_l][:,_i] = fitting(fitfunc, 
                #        tlist[lo:up+1], data[:,lo:up+1,_l], start_params, verbose=False)
                #if verbose:
                #    print("p-value %.7lf\nChi^2/dof %.7lf\nresults:"
                #          % (pval[_l][ 0, _i], chi2[_l][0,_i]/( (up - lo + 1) -
                #                                               len(start_params))))
                #    for p in enumerate(res[_l][0,:,_i]):
                #        print("\tpar %d = %lf" % p)
                #    print(" ")
        #for a, b in zip(func_args, func_kwargs):
        #    print(a, b)
        #fit1(*(func_args[0]), **(func_kwargs[0]))
        multiprocess(ffunc, func_args, func_kwargs)
        return

    def fit(self, _data, fitfunc, start_params):
        """Fit the data using the fitfunction.

        Args:
            _data: The correlation functions.
            fitfunc: The function to fit to the data.
            start_params: The starting parameters for the fit function.
        """
        if self.verbose:
            print("fitting %s '%s'"% (self.name, self.label))
        self.fitfunc = fitfunc
        self.data = _data
        if self.combfit:
            # sanity checks
            if len(self.fitranges) != 2:
                raise RuntimeError("%s needs 2 fitranges for combined fit" %\
                    self.__repr__)
            if not self.par:
                raise RuntimeError("%s needs parameter data for combined fit"%\
                    self.__repr__)
            # fit
            myargs = [_data, self.fitranges[0], self.fitranges[1], fitfunc,
                start_params, par]
            mykwargs = {"par_index": self.par_index, "olddata": self.old_data,
                "verbose": self.verbose}
            self.res, self.chi2, self.pvals = genfit_comb(*myargs, **mykwargs)
        else:
            myargs = [_data, self.fitranges[0], fitfunc, start_params]
            mykwargs = {"olddata": self.old_data, "verbose": self.verbose}
            self.res, self.chi2, self.pvals = genfit(*myargs, **mykwargs)
        self.depth = self._depth(self.res)

    def set_results(self, res):
        """Set results when reading from file."""
        self.res, self.chi2, self.pvals = res[:3]
        self.add_fitrange(res[3])

    def set_results_comb(self, res):
        """Set results when reading from file."""
        self.res, self.chi2, self.pvals = res[:3]
        self.add_fitranges(res[3], res[4])
        self.combfit = True

    def get_results(self):
        """Returns the fit results, the $\chi^2$ values and the p-values."""
        return self.res, self.chi2, self.pvals, self.fitranges[0]

    def save(self, filename):
        """save data to disk."""
        if self.verbose:
            print("saving %s '%s'"% (self.name, self.label))
        if self.combfit:
            write_fitresults(filename, self.fitranges[0], self.res,
                self.chi2, self.pvals, self.fitranges[1],
                self.verbose)
        else:
            write_fitresults(filename, self.fitranges[0], self.res,
                self.chi2, self.pvals, self.verbose)

    def save2(self, filename):
        """Save class to disk."""
        if self.combfit:
            raise NotImplementedError()
            dic = {'fi0' : self.fitranges[0]}
            dic = {'fi1' : self.fitranges[1]}
            dic.update({'pi%02d' % i: p for (i, p) in enumerate(self.res)})
            dic.update({'ch%02d' % i: p for (i, p) in enumerate(self.chi2)})
            dic.update({'pv%02d' % i: p for (i, p) in enumerate(self.pvals)})
            dic.update({'data': self.data})
            dic.update({'par': self.par})
            np.savez(filename, **dic)
        else:
            arr = numpy.array(2, dtype=object)
            dic = {'fi0' : self.fitranges[0]}
            dic.update({'pi%02d' % i: p for (i, p) in enumerate(self.res)})
            dic.update({'ch%02d' % i: p for (i, p) in enumerate(self.chi2)})
            dic.update({'pv%02d' % i: p for (i, p) in enumerate(self.pvals)})
            dic.update({'data': self.data})
            np.savez(filename, **dic)

    def plot(self, label, path="./plots/", plotlabel="corr"):
        """Plot data.
        
            label: Labels for the title and the axis.
            path: Path to the saving place of the plot.
            plotlabel: Label for the plot file.
        """
        if self.verbose:
            print("plotting %s '%s'"% (self.name, self.label))
        if self.combfit:
            myargs = [self.data, self.pvals, self.fitranges[0],
                self.fitranges[1], self.fitfunc, self.res, self.par,
                self.ensemble.get_data("tmin"), self.name,
                self.ensemble.get_data("d"), label]
            mykwargs = {"path": path, "plotlabel": plotlabel,
                "verbose":self.verbose, "par_par_index": self.par_index}
            genplot_comb(*myargs, **mykwargs)
        else:
            myargs = [self.data, self.res, self.pvals, self.fitranges[0],
                self.fitfunc, self.ensemble.get_data("tmin"), self.name,
                self.ensemble.get_data("d"), label]
            mykwargs = {"path": path, "plotlabel": plotlabel,
                "verbose":self.verbose}
            genplot(*myargs, **mykwargs)

    def __str__(self):
        restring = "FitResult %s '%s' with depth %d" % (self.name, self.label,
            self.depth)
        if self.data:
            restring = "\n".join((restring,"Data:\n"))
            for key in self.data:
                restring = "".join((restring, "\t%s: " % (str(key)),
                               str(self.data[key]), "\n"))
        else:
            retstring = "".join((retstring, "\n"))
        return restring

    def __repr__(self):
        return "[ FitResult %s '%s' with depth %d]" % (self.name, self.label, 
            self.depth)

