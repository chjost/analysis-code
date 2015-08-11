# a class to contain the fit results

import numpy as np

from ensemble import LatticeEnsemble
from fit import genfit, genfit_comb
from plot import genplot, genplot_comb
from input_output import write_fitresults, read_fitresults

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
        self.name = ensemble.name
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
    def from_file(self, ensemble, label, filename):
        """Initialize a fit from a file."""
        tmp = cls(ensemble, label)


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

    def add_par(self, par, par_index=0):
        """Add parameters for a combined fit and the index needed."""
        self.par = par
        self.par_index = par_index
        self.combfit = True

    def use_old_data(self, old_data):
        """Reuse the data located at 'old_data' if possible"""
        self.old_data = old_data

    def prepare_combined_fit(self, fitrange_data, fitrange_par, par,
          par_index=0, old_data=None):
        """Set everything needed for a combined fit."""
        self.comb_fit = True
        self.add_fitranges(fitrange_data, fitrange_par)
        self.add_par(par, par_index)
        self.use_old_data(old_data)

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

