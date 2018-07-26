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
            self.T = self.data.shape[1]
            print("Total time extent is: %d" %self.T)
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
        print(data)
        print(type(data))
        tmp = cls()
        try:
            tmp.data = data['arr_0']
            tmp.conf = data['arr_1']
            tmp.shape = tmp.data.shape
        except:
            # set the data directly
            tmp.data = data
            #print("data shape read in:")
            #print(tmp.data[:][1])
            tmp.shape = data.shape
            # WTF does this stand here?!
            #if data.shape[1] > 2:
            #    tmp.shape = data.shape[:-1]
            if data.shape[-2] != data.shape[-1]:
                tmp.matrix = False
            else:
                tmp.matrix = True
        return tmp

    @classmethod
    def create(cls, data, conf=None, T=None, debug=0):
        """Create correlator class from preexisting data.

        Parameters
        ----------
        data : ndarray data has shape [BS,T,r] with r labeling the correlator
            The correlation function data.
        debug : int, optional
            The amount of debug information printed.
        """
        tmp = cls(debug=debug)
        # Make a copy instead of a view
        tmp.data = np.copy(np.atleast_3d(data))
        print("data has dimension:")
        print(tmp.data.shape)
        tmp.shape = tmp.data.shape
        if conf is not None:
            tmp.conf=conf
        print("original data shape:")
        print(data.shape)
        print(data.shape[-2],data.shape[-1])
        if data.shape[-2] != data.shape[-1]:
            tmp.matrix = False
        tmp.ncorr = data.shape[-1]
        if tmp.data.ndim < 3:
            tmp.ncorr = 1
        else:
            tmp.ncorr = tmp.data.shape[0]
        if T is not None:
            tmp.T=T
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
                in_out.write_data_ascii(self.data, filename, verbose)
        else:
            if self.conf is not None:
                print("Saving configuration numbers")
                in_out.write_data(self.data, filename, self.conf, verbose)
            else:
                in_out.write_data(self.data, filename, verbose)
    
    def block(self,bl=1):
        """Block data into blocks of length bl

        The data are blocked and the mean over each block is taken
        """
        self.data = boot.block(self.data, l=bl)
        self.shape = self.data.shape

    def symmetrize(self,blocking=False,bl=1):
        """Symmetrizes the data around the second axis.
        """
        self.data = boot.sym(self.data,blocking=blocking,bl=bl)
        self.shape = self.data.shape

    def bootstrap(self, nsamples, blocking= False, bl = None, method='naive'):
        """Creates bootstrap samples of the data.

        Parameters
        ----------
        nsamples : int
            The number of bootstrap samples to be calculated.
        blocking : bool
            Should naive blocking be used?
        bl : int
           blocklength for blocking methods
        method : string 
           'naive' or 'stationary' for bootstrap method
        """
        self.data = boot.bootstrap(self.data, nsamples, blocking, bl,method)
        self.shape = self.data.shape 
        
    def reflect(self, kind="axis"):
        """reflect coorelation function point symmetrically or by axis reflection
    
        Parameters
        ----------
        source : ndarray
            The data to antisymmetrize.
        kind : String
            available reflections are "point" and "axis: reflections, defaults to
            axis
    
        Returns
        -------
        the symmetrized source
        """
    
        if kind == "axis":
            self.data = boot.sym(self.data)
        elif kind == "point":
            self.data = boot.asym(self.data)
        else:
            print("Reflection type not known")
        self.shape = self.data.shape

    def sym_and_boot(self, nsamples,blocking=False, bl=None ,method='naive'):
        """Symmetrizes the data around the second axis and then
        create bootstrap samples of the data

        Parameters
        ----------
        nsamples : int
            The number of bootstrap samples to be calculated.
        blocking : bool
            Should data be divided into blocks
        bl : int
            If blocking enabled states length of each block
        """
        self.data = boot.sym_and_boot(self.data,
                nsamples,blocking=blocking,bl=bl,method=method)
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
        # if the data is not a matrix, do nothing
        if not self.matrix:
            return

        # calculate the dE for weighting if needed
        if mass is None:
            dE=None
        else:
            # TODO: differentiate the different d2 and irreps
            # for piK without momenta this is enough for a try
            # TODO: Automate that with get_dE function
            dE = mass
            #dE = np.asarray(0.5*WfromMass_lat(mass, d2, L) - mass)

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
               print("not a matrix")
               return

           tmp = gevp.calculate_gevp(self.data, t0)
           self.data = tmp
           if not self.data is tmp:
               raise RuntimeError("data not assinged correctly")
           self.shape = self.data.shape
           self.matrix = False

    # new interface
    def mass(self, function=0, add=None):
        """Computes the effective mass.

        Three formulae are implemented. The standard formula is based on the
        cosh function, the alternative is based on the sinh function. The third
        option (chosen via usecosh=None) is the log

        Parameters
        ----------
        func : integer
            Toggle between the implemented methods.
            implemented_functions = {0: corr_arcosh,
                                     1: corr_exp,
                                     2: corr_exp_asym,
                                     3: corr_log,
                                     4: corr_shift_weight,
                                     5: corr_shift_weight_div}
        add : list of additional arguments to the different implementations
        """
        # more versatile interface:
        # self.data = func.compute_eff_mass(self.data,method,weight,shift)
        self.data = func.compute_eff_mass(self.data, self.T,
                                          function=function, add=add)
        self.shape = self.data.shape

    #def mass(self, usecosh=True, exp=False, weight=None, shift=None, T=None):
    #    """Computes the effective mass.

    #    Three formulae are implemented. The standard formula is based on the
    #    cosh function, the alternative is based on the sinh function. The third
    #    option (chosen via usecosh=None) is the log

    #    Parameters
    #    ----------
    #    usecosh : bool
    #        Toggle between the two implemented methods.
    #    """
    #    # more versatile interface:
    #    # self.data = func.compute_eff_mass(self.data,method,weight,shift)
    #    self.data = func.compute_eff_mass(self.data, usecosh, exp, weight,
    #            shift,T=T)
    #    self.shape = self.data.shape

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
            dE = np.asarray(WfromMass_lat(mass, d2, L) - mass)

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
                shift, dE, useall, d2, L, irrep)
        obj.shape = obj.data.shape
        return obj

    def back_derivative(self,t=1):
        derive = Correlators(debug=self.debug)
        derive.data = func.compute_derivative_back(self.data,a=t)
        derive.shape = derive.data.shape
        
        return derive

    def square_corr(self,corr=0):
        derive = Correlators(debug=self.debug)
        derive.data = func.compute_square(self.data[0])
        derive.shape = derive.data[0].shape
        
        return derive

    def mult_corr(self,fac):
        derive = Correlators(debug=self.debug)
        derive.data = func.multiply(self.data[0],fac)
        derive.shape = derive.data[0].shape
        
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

    def omit_iqr(self, in_iqr=None, ts=0, debug=1):
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
          q25, q75 = np.percentile(self.data[:,ts],(25,75))
          iqr_dn = q25-1.5*np.abs(q75-q25)
          iqr_up = q75+1.5*np.abs(q75-q25)
          # boolean arrays of outlier positions
          idx_up = np.greater(self.data[:,ts], iqr_up)
          idx_dn = np.less(self.data[:,ts], iqr_dn)
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

    def sub_subleading(self, fit1, fit2, T):
        """subtract subleading thermal pollution numerically from correlator
          
        Function takes data from Correlator at self and two sets of fit parameters
        It returns the difference between data and:
        WARNING: corr gets modified!
    
        Parameter
        ---------
        fit1, fit2: FitResult, Amplitude and mass parameters of the thermal pollution
        amp_guess: float, optional factor to tune amplitude
    
        Returns
        -------
        Correlator
        """
        _corr = Correlators.create(self.data,T=self.T) 
        # Correlators have shape [nboot,T,real]
        T2 = _corr.shape[1]
        print("Half lattice time extent is %d" %T2)
        # singularize fitresults
        _fit1 = fit1.singularize()
        _fit2 = fit2.singularize()
        _fit1.calc_error()
        _fit2.calc_error()
        # Fitted amplitude from corrws fit
        # CAUTION: Uses half lattice time extent! 2*T2 needed, T is 1500,1500
        # array
        _amp = _fit1.data[0][:,2,-1] * np.exp(-T[:,0]*_fit2.data[0][:,1,-1])*(1-np.exp(2*(_fit2.data[0][:,1,-1]-_fit2.data[0][:,0,-1])))
        sub = np.zeros_like(self.data)
        for t in range(0,T2):
            sub[:,t,0] = _amp*np.exp((_fit2.data[0][:,1,-1]-_fit2.data[0][:,0,-1])*t)
        _corr.data -= sub
        return _corr
# TODO: This can be made more efficient by introducing a few more functions
    def subtract_pollution(self, fit1, fit2):
        #_corr = Correlators.create(self.data) 
        # Correlators have shape [nboot,T/2,real] after symmetrization
        T = self.T
        # singularize fitresults
        _fit1 = fit1.singularize()
        _fit2 = fit2.singularize()
        _fit1.calc_error()
        _fit2.calc_error()
        # pollution is:
        # A_1**2 * p(t) = A_1**2 * {exp[(E_K-E_pi)*t]*exp[-E_K*T] 
        #                           + exp[(-(E_K-E_pi)*t]*exp[-E_pi*T]}
        # A_1**2 = A_pi*A_K
        amplitude_squared = _fit1.data[0][:,0,-1]**2 *_fit2.data[0][:,0,-1]**2
        print(amplitude_squared)
        # Energy values
        ek = _fit2.data[0][:,1,-1] 
        epi = _fit1.data[0][:,1,-1] 
        diff_ek_epi =  ek - epi
        pollution = np.zeros_like(self.data)
        for t in range(0,self.data.shape[1]):
            pollution[:,t,0] = amplitude_squared*(np.exp(diff_ek_epi*t) * np.exp(-ek*T) +
                               np.exp(-diff_ek_epi*t) * np.exp(-epi*T))
            #pollution[:,t,0] = amplitude_squared*(np.exp(-epi*t) * np.exp(-ek*(T-t)) +
            #                   np.exp(-ek*t) * np.exp(-epi*(T-t)))
        self.data -= pollution
    
    def divide_out_pollution(self, fit1, fit2):
        # only true for symmetrized correlators
        #T2 = self.shape[1]
        T = self.T
        # singularize fitresults
        _fit1 = fit1.singularize()
        _fit2 = fit2.singularize()
        _fit1.calc_error()
        _fit2.calc_error()
        # Energy values
        ek = _fit2.data[0][:,1,0] 
        epi = _fit1.data[0][:,1,0]
        pollution_exp = np.zeros_like(self.data)
        for t in range(0,self.data.shape[1]):
            pollution_exp[:,t,0] = np.exp(-epi*t) * np.exp(-ek*(T-t)) + np.exp(-ek*t) * np.exp(-epi*(T-t))
            #pollution_exp[:,t,0] = (np.exp(diff_ek_epi*t) * np.exp(-ek*T) +
            #                   np.exp(-diff_ek_epi*t) * np.exp(-epi*T))
            self.data[:,t] /= pollution_exp[:,t]

    def multiply_pollution(self,fit1,fit2):
        # only true for symmetrized correlators
        #T2 = self.shape[1]
        T= self.T
        # singularize fitresults
        _fit1 = fit1.singularize()
        _fit2 = fit2.singularize()
        _fit1.calc_error()
        _fit2.calc_error()
        # Energy values
        ek = _fit2.data[0][:,1,0] 
        epi = _fit1.data[0][:,1,0] 
        diff_ek_epi =  ek - epi
        pollution_exp = np.zeros_like(self.data)
        for t in range(0,self.data.shape[1]):
            pollution_exp[:,t,0] = np.exp(-epi*t) * np.exp(-ek*(T-t)) + np.exp(-ek*t) * np.exp(-epi*(T-t))
            #pollution_exp[:,t,0] = (np.exp(diff_ek_epi*t) * np.exp(-ek*T) +
            #                   np.exp(-diff_ek_epi*t) * np.exp(-epi*T))
            self.data[:,t] *= pollution_exp[:,t]

if __name__ == "main":
    pass
