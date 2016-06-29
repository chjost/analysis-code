"""
A class for correlation functions.
"""

import os

import numpy as np
import itertools

import in_out
import bootstrap as boot
import gevp
import functions as func
from ratio import simple_ratio, ratio_shift, simple_ratio_subtract, twopoint_ratio
from energies import WfromMass_lat, WfromMass

class Correlators(object):
    """Correlation function class.
    """

    def __init__(self, filename=None, column=(1,), matrix=True, skip=1,
        conf_col=None, debug=0):
        """Reads in data from an ascii file.

        The file is assumed to have in the first line the number of
        data sets, the length of each dataset and three further numbers
        not used here. This info is used to shape the data accordingly.

        If a sequence of strings is used for filenames, a correlation
        function matrix is created, otherwise a single correlation
        function is created. This has implications for some of the
        class functions.

        Parameters
        ----------
        filename : str or sequence of str, optional
            The filename of the file.
        column : sequence, optional
            The columns that are read.
        matrix : bool, optional
            Read data as matrix or not
        skip : int, optional
            The number of header lines that are skipped.
        debug : int, optional
            The amount of debug information printed.

        Raises
        ------
        IOError
            If the directory of the file or the file is not found.
        ValueError
            If skip < 1 or a non-existing column is read
        """
        if skip < 1:
            raise ValueError("File is assumed to have info in first line")
        else:
            self.skip = skip
        self.debug = debug
        self.data = None
        self.matrix = None
        self.conf = None
        if filename is not None:
            if isinstance(filename, (list, tuple)):
                if matrix:
                    self.data = in_out.read_matrix(filename, column, skip, debug)
                    self.matrix = True
                    if isinstance(conf_col,int):
                        self.conf = in_out.read_matrix(filename, (conf_col,), skip, debug)
                else:
                    self.data = in_out.read_vector(filename, column, skip, debug)
                    self.matrix = False
                    if isinstance(conf_col,int):
                        self.conf = in_out.read_vector(filename, (conf_col,), skip, debug)
            else:
                tmp = in_out.read_single(filename, column, skip, debug)
                self.data = np.atleast_3d(tmp)
                self.matrix = False
                if isinstance(conf_col,int):
                    print("Column for configuration numbers is %d\n" % conf_col)
                    self.conf = in_out.read_single(filename, (conf_col,), skip, debug)
        if self.data is not None:
            self.shape = self.data.shape
        else:
            self.shape = None

    @classmethod
    def read(cls, filename, debug=0):
        """Reads data in numpy format.

        If the last two axis have the same extent, it is assumed a
        correlation function matrix is read, otherwise a single
        correlation function is assumed. This has implications on some
        class functions.

        Parameters
        ----------
        filename : str
            The name of the data file.
        debug : int, optional
            The amount of debug information printed.

        Raises
        ------
        IOError
            If file or folder not found.
        """
        data = in_out.read_data(filename)
        tmp = cls(debug=debug)
        if isinstance(data.files,list):
            tmp.data = data['arr_0']
            tmp.conf = data['arr_1']
        else:
            # set the data directly
            tmp.data = data
            if data.shape[-2] != data.shape[-1]:
                tmp.matrix = False
        tmp.shape = tmp.data.shape
        if tmp.data.shape[-2] != tmp.data.shape[-1]:
            tmp.matrix = False
        else:
            tmp.matrix = True
        return tmp

    @classmethod
    def create(cls, data, conf=None, debug=0):
        """Create correlator class from preexisting data.

        Parameters
        ----------
        data : ndarray
            The correlation function data.
        debug : int, optional
            The amount of debug information printed.
        """
        tmp = cls(debug=debug)
        tmp.data = data
        tmp.shape = data.shape
        if conf is not None:
            tmp.conf=conf
        if data.shape[-2] != data.shape[-1]:
            tmp.matrix = False
        return tmp

    def save(self, filename, asascii=False):
        """Saves the data to disk.
        
        The data can be saved in numpy format or plain ascii.

        Parameters
       ----------
        filename : str
            The name of the file to write to.
        asascii : bool, optional
            Toggle numpy format or ascii.
        """
        verbose = (self.debug > 0) and True or False
        if asascii:
            if self.conf is not None:
                print("Saving configuration numbers")
                in_out.write_data_ascii(self.data, filename, verbose, self.conf)
            else:
                in_out.rite_data_ascii(self.data, filename, verbose)
        else:
            if self.conf is not None:
                print("Saving configuration numbers")
                in_out.write_data(self.data, filename, self.conf, verbose)
            else:
                in_out.write_data(self.data, filename, verbose)

    def symmetrize(self):
        """Symmetrizes the data around the second axis.
        """
        self.data = boot.sym(self.data)
        self.shape = self.data.shape

    def bootstrap(self, nsamples):
        """Creates bootstrap samples of the data.

        Parameters
        ----------
        nsamples : int
            The number of bootstrap samples to be calculated.
        """
        self.data = boot.bootstrap(self.data, nsamples)
        self.shape = self.data.shape

    def sym_and_boot(self, nsamples):
        """Symmetrizes the data around the second axis and then
        create bootstrap samples of the data

        Parameters
        ----------
        nsamples : int
            The number of bootstrap samples to be calculated.
        """
        self.data = boot.sym_and_boot(self.data, nsamples)
        self.shape = self.data.shape

    def shift(self, dt, mass=None, shift=1, d2=0, L=24, irrep="A1",
            uselattice=True):
        """Shift and weight the data.

        This function only works with matrices.

        Two shifts are implemented and can be selected using the flag
        shift.
        The first is due to Dudek et al, Phys.Rev. D86, 034031 (2012).
        The second is due to Feng et al, Phys.Rev. D91, 054504 (2015).
        For the second dE must be given.

        Parameters
        ----------
        dt : int
            The amount to shift.
        mass : {None, float, ndarray}, optional
            The weight factor.
        shift : {1, 2}, optional
            Which shift to use, see above.
        d2 : int, optional
            The total momentum squared, used for determining dE.
        L : int, optional
            The spatial extent of the lattice.
        irrep : string, optional
            The lattice irrep to use, influences dE.
        uselattice : bool, optional
            Use the lattice version of the dispersion relation.
        """
        ## if the data is not a matrix, do nothing
        #if not self.matrix:
        #    return

        # calculate the dE for weighting if needed
        if mass is None:
            dE=None
        else:
            dE = get_dE(mass, d2, L, irrep)
            # TODO: differentiate the different d2 and irreps
            dE = np.asarray(0.5*WfromMass_lat(mass, d2, L) - mass)

        # calculate the shift
        if shift == 1:
            # if dE has more than just 1 axis, add the axis to the correlation
            # function
            if dE is not None and dE.ndim > 1:
                newshape = list(self.shape[:-2] + dE.shape[1:] + self.shape[-2:])
                newshape[1] -= dt
                tmp = np.zeros(newshape)
                # iterate over the axis > 1 of dE
                item = [[n for n in range(x)] for x in dE.shape[1:]]
                for it in itertools.product(*item):
                    # select the correct entries for tmp and dE
                    s = (Ellipsis,) + it +(slice(None), slice(None))
                    s1 = (slice(None),) + it
                    tmp[s] = gevp.gevp_shift_1(self.data, dt, dE[s1], self.debug)
                self.data = tmp
            else:
                self.data = gevp.gevp_shift_1(self.data, dt, dE, self.debug)
        elif dE is None:
            raise ValueError("dE is mandatory for the second implemented shift")
        else:
            self.data = gevp.gevp_shift_2(self.data, dt, dE, self.debug)
        self.shape = self.data.shape

    def gevp(self, t0):
        """Calculate the GEVP of the matrix.

        This function only works with matrices.

        Parameters
        ----------
        t0 : int
            The index of the inverted matrix.
        """
        if not self.matrix:
            return

        self.data = gevp.calculate_gevp(self.data, t0)
        self.shape = self.data.shape
        self.matrix = False

    def mass(self, usecosh=True):
        """Computes the effective mass.

        Two formulas are implemented. The standard formula is based on the
        cosh function, the alternative is based on the log function.

        Parameters
        ----------
        usecosh : bool
            Toggle between the two implemented methods.
        """
        self.data = func.compute_eff_mass(self.data, usecosh)
        self.shape = self.data.shape

    def get_data(self):
        """Returns a copy of the data.
        
        Returns
        -------
        ndarray
            Returns the saved data.
        """
        return np.copy(self.data)

    def ratio(self, single_corr, ratio=0, shift=1, single_corr1=None,
            useall=False, mass=None, d2=0, L=24, irrep="A1"):
        """Calculates the ratio and returns a new Correlator object.

        The implemented ratios are:
        * corr/(single_corr^2)
        * corr(t)/(single_corr(t)^2 - single_corr(t+shift)^2)
        * (corr(t)-corr(t+1))/(single_corr(t)^2 - single_corr(t+1)^2)
        where shift is an additional parameter.

        If single_corr1 is given, single_corr^2 becomes
        single_corr*single_corr1.

        Parameters
        ----------
        single_corr : Correlators
            The single particle correlators used for the ratio.
        ratio : int
            Chose the ratio to use.
        shift : int
            Additional parameter for the second ratio.
        single_corr1 : Correlators
            The single particle correlators used for the ratio.
        useall : bool
            Using all correlators in the single particle correlator or
            use just the lowest.
        mass : {None, float, ndarray}, optional
            The weight factor.
        d2 : int, optional
            The total momentum squared, used for determining dE.
        L : int, optional
            The spatial extent of the lattice.
        irrep : string, optional
            The lattice irrep to use, influences dE.
        
        Returns
        -------
        Correlators
            The ratio.
        """
        # if any correlator is a matrix, raise an error
        if self.matrix or single_corr.matrix:
            raise RuntimeError("cannot do ratio with a matrix")
        if single_corr1 is not None and single_corr1.matrix:
            raise RuntimeError("cannot do ratio with a matrix")

        # calculate the dE for weighting if needed
        if mass is None:
            dE=None
        else:
            # TODO: differentiate the different d2 and irreps
            #dE = np.asarray(WfromMass_lat(mass, d2, L) - mass)
            dE = get_dE(mass, d2, L, irrep)

        # get the actual ratio being calculated
        # TODO check ratio 4 implementation
        functions = {0: simple_ratio, 1: ratio_shift, 2: simple_ratio_subtract, 3: twopoint_ratio}
        #functions = {0: ratio.simple_ratio, 1: ratio.ratio_shift,
        #        2: ratio.simple_ratio_subtract, 3: ratio.ratio}
        ratiofunc = functions.get(ratio)

        if single_corr1 is None:
            if dE is not None and dE.ndim > 1:
                tmp = np.zeros_like(self.data)
                # iterate over the axis > 1 of dE
                item = [[n for n in range(x)] for x in dE.shape[1:]]
                for it in itertools.product(*item):
                    # select the correct entries for tmp and dE
                    s = (Ellipsis,) + it +(slice(None), slice(None))
                    s1 = (slice(None),) + it
                    tmp[s] = ratiofunc(self.data, single_corr.data, single_corr.data,
                        shift, dE[s1], useall, d2, L, irrep)
                obj = Correlators(debug=self.debug)
                obj.data = tmp

            else:
                obj = Correlators(debug=self.debug)
                obj.data = ratiofunc(self.data, single_corr.data, single_corr.data,
                    shift, dE, useall, d2, L, irrep)
        else:
            obj = Correlators(debug=self.debug)
            obj.data = ratiofunc(self.data, single_corr.data, single_corr1.data,
                shift, dE, useall, p2=d2, L=L, irrep=irrep)
        obj.shape = obj.data.shape
        return obj

    def back_derivative(self):
        derive = Correlators(debug=self.debug)
        derive.data = func.compute_derivative_back(self.data[0])
        derive.shape = derive.data.shape
        
        return derive

    def square_corr(self):
        derive = Correlators(debug=self.debug)
        derive.data = func.compute_square(self.data[0])
        derive.shape = derive.data.shape
        
        return derive

    def diff(self, single_corr=None):
        """Calculates the difference between two correlators and returns a new Correlator object.

        The implemented ratios are:
        * corr/(single_corr^2)
        * corr(t)/(single_corr(t)^2 - single_corr(t+shift)^2)
        * (corr(t)-corr(t+1))/(single_corr(t)^2 - single_corr(t+1)^2)
        where shift is an additional parameter.

        If single_corr1 is given, single_corr^2 becomes
        single_corr*single_corr1.

        Parameters
        ----------
        single_corr : Correlators
            The single particle correlators used for the ratio.
        ratio : int
            Chose the ratio to use.
        shift : int
            Additional parameter for the second ratio.
        single_corr1 : Correlators
            The single particle correlators used for the ratio.
        useall : bool
            Using all correlators in the single particle correlator or
            use just the lowest.
        usecomb : list of list of ints
            The combinations of entries of single_corr (and single_corr1)
            to use.
        
        Returns
        -------
        Correlators
            The ratio.
        """
        obj = Correlators(debug=self.debug)
        if self.data.shape[-1] == 2:
          obj.data = func.simple_difference(self.data)
        else:
          obj.data = func.simple_difference(self.data, single_corr.data)
        obj.shape = obj.data.shape
        print(obj.shape)
        return obj

    def hist(self, time):
        """Returns the history over Configurations at a given time
        WARNING : Works only before bootstrapping
        Parameters:
        -----------
        time : the timeslice to view
        """
        nb_cfg = self.data.shape[0]
        history = np.asarray([self.data[c][time][0] for c in range(nb_cfg)])
        return history

    def omit_iqr(self, in_iqr=None, debug=1):
      """Based on the iqr delete configurations with a certain value
      
      This function uses the difference between the 25 and 75 percentile
      percentile to detect outliers.
      IQR = |p75-p25|
      An outlier is defined as such if not lying in the interval
      (p25-1.5*IQR, p75+1.5*IQR). The list of the configuration numbers is
      truncated as well as the configurations of all timeslices.

      Parameters
      ----------
      conf : optional list of configurations to omit

      Returns
      -------
      a list of the indices of the deleted configurations to ensure that
      correlation is not lost
      """
      if in_iqr is None:
          # get the 25% and 75% percentiles from the real part of the first timeslice of all
          # configurations
          q25, q75 = np.percentile(self.data[:,0],(25,75))
          iqr_dn = q25-1.5*np.abs(q75-q25)
          iqr_up = q75+1.5*np.abs(q75-q25)
          # boolean arrays of outlier positions
          idx_up = np.greater(self.data[:,0], iqr_up)
          idx_dn = np.less(self.data[:,0], iqr_dn)
          print("index shapes are:")
          print(idx_up.shape,idx_dn.shape)
          omit_idx = np.logical_or(idx_up,idx_dn)
          # reshape necessary for correct slicing
          print("omit_idx has shape")
          print(omit_idx.shape)
          in_iqr = np.invert(omit_idx).reshape(omit_idx.shape[0])
          #in_iqr = np.invert(omit_idx)
          # store ommited configurations
          omitted = self.conf[omit_idx]
          print self.data.shape
          self.conf = self.conf[in_iqr]
          self.data = self.data[in_iqr]
          print(self.data.shape)
          self.shape = self.data.shape
          if debug > 0:
            print("omitted configurations:")
            print(omitted)
          return omitted,in_iqr

      else:
          # turn the configuration list into a boolean array...
          #
          # store ommited configurations
          omitted = np.unique(self.conf[np.invert(in_iqr)])
          self.conf = self.conf[in_iqr]
          self.data = self.data[in_iqr]
          if debug > 0:
            print("omitted configurations:")
            print(omitted)
          return omitted,in_iqr
          

    def omit(self, par, corr):
      """Based on the first timeslice delete configurations with a certain value
      TODO:This function should be improved. I do not know how to choose the
      value to cut, perhaps by the mean over the first timeslice?

      Parameters
      ----------
      par : Either the value to cut or a list of configurations to omit

      Returns
      -------
      a list of the indices of the deleted configurations to ensure that
      correlation is not lost
      """
      # Check whether to cut configurations or a value
      # Check whether par is iterable
      if hasattr(par,"__iter__"):
        omitted = []
        # loop over configurations 
        for c, v in enumerate(self.data[...,corr]):
          if v[0] > par[0]:
            omitted.append(c)
      # Check for interval
      if isinstance(par[0],float) and len(par) == 2:

        par = sorted(par)

        omitted = []
        # loop over configurations 
        for c, v in enumerate(self.data[...,corr]):
          if  v[0] < par[0] or par[1] < v[0]:
            omitted.append(c)

      else: 
        omitted = par
      tmp = np.delete(self.data[...,corr],omitted,0)
      cut = Correlators()
      cut.skip = self.skip
      cut.debug = self.debug
      cut.shape = (tmp.shape[0],self.shape[1],self.shape[2])
      cut.data = np.zeros(cut.shape)
      cut.data[...,0] = tmp
      print("nconf original:")
      print(self.data.shape[0])
      print("nconf after cut:")
      print(cut.data.shape[0])
      cut.matrix = self.matrix
      return cut, omitted

def get_dE(mass, d2, L, irrep="A1"):
    """Calculate the energy gap for the leading finite-T effect.
    """
    if irrep == "A1":
        dE = np.asarray(0.5*WfromMass_lat(mass, d2, L) - mass)
    else:
        dE = np.asarray(0.5*WfromMass_lat(mass, d2, L) - mass)
    return dE

if __name__ == "main":
    pass
