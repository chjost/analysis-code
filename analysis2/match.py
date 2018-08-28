"""
Class for matching procedures. It behaves similar to the FitResult class,
perhaps make it a derived class of some Metaobject
"""
import matplotlib
matplotlib.use('Agg') # has to be imported before the next lines
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
#matplotlib.rcParams['font.size'] = 14
import numpy as np

import chiral_utils as chut
import extern_bootstrap as extboot
import fit
from match_functions import *
from match_help import in_ival
from .plot import plot_lines, plot_single_line
from .plot_functions import plot_data, plot_function
from .statistics import compute_error

class MatchResult(object):
    """Class to process FitResults for Interpolation

    The data has a similar layout to FitResult
    """
    def __init__(self, obs_id=None, save=None, debug=0):

      """Allocate objects for initiated instance
      Parameters
      ----------
        obs_id : Observable that is investigated

      Members
      -------
      x, y : the raw data coming in 2d arrays of number of observables and
      number of bootstrap samples
      amu_match : nd array,the resulting matched values of the quark mass,
                  ordered by matching method
      eval_obs : ndarray holding data evaluated at amu_match
      coeffs : coefficients of the matching as list
      obs : nd-array, Observables used in the matching
      amu : nd-array, the quark masses belonging to the observables
      """
      # the matched amu_s_values (at least 3 values for 3 different methods)
      self.amu_match = None
      self.eval_obs=None
      # the observables to match to
      self.obs = None
      # x-values to use in the matching should have same length as slf.obs
      self.amu = None
      # Coeffs for the plots
      self.coeffs = None
      self.weight = None
      self.error = None
      self.label = None
      # observable for with matching is done
      self.obs_id = obs_id
      # Save results to
      self.savedir = save
      # Debug flag
      self.debug=debug
    
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
        if isinstance(data.files,list):
            tmp = cls()
            tmp.data = data['arr_0']
            tmp.conf = data['arr_1']
            tmp.shape = tmp.data.shape
        else:
            # set the data directly
            tmp = cls()
            tmp.data = data
            print("data shape read in:")
            print(tmp.data[:][1])
            tmp.shape = data.shape
            if data.shape[1] > 2:
                tmp.shape = data.shape[:-1]
            if data.shape[-2] != data.shape[-1]:
                tmp.matrix = False
            else:
                tmp.matrix = True
        return tmp
    #@classmethod
    #def read()
    def save(self, filename):
        """Save a match result
        
        For every ensemble, and possibly different stragne
        quark masses, the match result is saved in a binary numpy format.
        In principle the data layout is oriented at the correlator layout.
        
        Parameters
        ----------
        filename : string indicating where to save the data
        """
        _match_save = fit.FitResult(corr_id=self.obs_id,derived=True)
        _match_save.create_empty((4,3,1,1500),(4,3,1,1500),1)
        _match_save.corr_num=1
        # place data in numpy array
        # 3 evaluation methods or strange quarkmasses, 1 "fitrange", samples
        tmp = np.zeros((4,3,1,1500))
        if self.debug > 0:
            print(self.amu.shape)
            print(self.obs.shape)
            print(self.amu_match.shape)
            print(self.eval_obs.shape)
        # shape of one data element
        el_shape = tmp[0].shape
        ## save x_data used in matching
        ## the amu values used in the matching procedure
        _amu_save = np.repeat(self.amu,el_shape[1]*el_shape[-1]).reshape(el_shape)
        tmp[0] = _amu_save
        ## the observables belonging to the amu values
        tmp[1] = self.obs.reshape(el_shape)
        ## the matched amu value
        _amu_match_save = np.zeros(el_shape)
        if len(self.amu_match.shape) < 2:
          _amu_match_save=self.amu_match.reshape(el_shape[1],el_shape[2]).repeat(el_shape[0],axis=0)
          print(_amu_match_save)
        tmp[2] = _amu_match_save.reshape(el_shape)
        ## the evaluated observable at amu_match
        tmp[3] = self.eval_obs.reshape(el_shape)
        _match_save.data=tmp
        ## save coefficients as fit parameters, distinguish between matching and
        ## evaluation`
        _match_save.save(filename)

    def save_samples_ascii(self,data,name,head,fmt = '%.4f'):
      """ Save the samples of a data array to a text file """
      # Append an index number as first column, taken from shape[-1] of data to
      # be written
      nboot = data.shape[-1]
      smp = np.linspace(0,(nboot-1),nboot)
      idx = np.zeros((1,nboot))
      idx[0] = smp
      _data = np.atleast_2d(data)
      _to_save = np.concatenate((idx,_data)).T
      if self.debug > 0:
          print("Shape of idx:")
          print(idx.shape)
          print("Shape of data:")
          print(_data.shape)
          print("Data will be saved with shape:")
          print(_to_save.shape)
      np.savetxt(name,_to_save,header=head,fmt = fmt)


    def create_empty(self, nboot, nb_obs, obs_id=None):
      """Create an empty matching class, setting up the appropriate shapes

      Parameters
      ----------
      nboot : int, the number of bootstrapsamples for the data
      nb_obs : int, the number of observables used in the matching
      obs_id : str, an identifier for the matching
      """
      # number of methods and matched quark masses fixed
      self.amu_match = np.zeros((3,nboot))
      self.obs = np.zeros((nb_obs,nboot))
      self.eval_obs = np.zeros((3,nboot))
      self.amu = np.zeros((nb_obs,nboot))
      if self.obs_id is None:
        self.obs_id = obs_id

    # get data from collection of fitresults
    def load_data(self,fitreslst,par,amu,square=False,mult=None,ti=None,debug=0):
      """load a list of fitresults and a list of amu values into an existing instance of
      MatchResult

      The Median of the FitResults is used to set the data in the MatchResult
      instance

      Parameters
      ----------
      fitreslst: list, folder and datanames of fitresult
      par: which parameter to take from fitresult
      amu: float,set strange quarkmass accordingly
      square: bool, should data be squared?
      mult, ndarray, if shape is right factor to multiply fitres with
      """
      # Check lengths
      if len(fitreslst) != amu.shape[0]:
        raise ValueError("Fitresults and amu have different length")
      # from a list of fitresults, check if singular, singularize, if needed,
      for i,r in enumerate(fitreslst):
          # Either take the weighted median
        if ti is None:
            r = r.singularize()
        else:
            r = r.fr_select(ti[i])
        self.obs[i] = r.data[0][:,par,0]
        #print(r.data[0][:,0,0])
        if debug > 0:
          print("Read in %s: %.4e" %(r,self.obs[i][0]))
        # square first
        if square:
          self.obs[i] = np.square(self.obs[i])
        # multiply with observable of right shape
        if mult is not None:
          if hasattr(mult,'__iter__'):
            if len(mult) is len(fitreslst):
              self.obs[i]*=mult[i]
            else:
              try:
                self.obs[i]*=mult
              except:
                raise ValueError("mult has wrong length!")
          else:
            self.obs[i]*=mult
        self.amu[i] = amu[i]
      
    # need to set all data at once and add data one at a time

    def set_data(self,obs,amu):
      """Set observable data and amu for all observables
      """
      self.obs = np.copy(obs)
      self.amu = np.copy(amu)

    def add_data(self,data,idx,op=False,amu=None,obs=True,fac=None):
      """Add x- or y-data at aspecific index

      Parameters
      ----------
      data : ndarray, the data to add
      idx : tuple, the index where to place the data
      op : string, should existent data at index and dimension be operated
             with data to add?
      obs : Bool whether to add to observable or evaluated observables
      fac : float, optional factor to multiply data with beforehand
      """
      #print("xdata to add has shape")
      #print(data.shape)
      if data is not None:
          if fac is not None:
            data *= fac
          if obs == True:
            #print("xdata to add to has shape")
            #print(self.obs[idx].shape)
            if op == 'mult':
              self.obs[idx] *= data
            # divide existent obs by data
            elif op == 'div':
              self.obs[idx] /= data
            elif op == 'min':
              self.obs[idx] -= data
            else:
              self.obs[idx] = data
            if amu is not None:
              self.amu[idx]=amu
            #print(self.obs[idx])
          else:
            #print("xdata to add to has shape")
            #print(self.eval_obs[idx].shape)
            if op == 'mult':
              self.eval_obs[idx] *= data
            # divide existent eval_obs by data
            elif op == 'div':
              self.eval_obs[idx] /= data
            elif op == 'min':
              self.eval_obs[idx] -= data
            else:
              self.eval_obs[idx] = data
      if amu is not None:
          self.amu[idx]=amu
        #print(self.eval_obs[idx])


    def add_extern_data(self,filename,ens,idx=None,amu=None,square=True,
                        read=None,physical=False,op=False,fse=False,r0=True,fac=None):
      """ This function adds data from an extern textfile to the analysis object
      
      The data is classified by names, the filename is given and the read in and
      prepared data is placed at the index 

      Parameters
      ---------
      filename : string, the filename of the extern data file
      idx : tuple, index where to place prepared data in analysis
      ens : string, identifier for the ensemble
      dim : string, dimension to place the extern data at
      square : bool, should data be squared?
      read : string, identifier for the data read
      op : string, if operation is given the data is convoluted with existing data
          at that index. if no operation is given data gets overwritten
      """
      #if read is None:
      #  plot, data = np.zeros((self.x_data[0].shape[-1]))
      #if read is 'r0_inv':
      #  _plot,_data = extboot.prepare_r0(ens,self.x_data[0].shape[-1])
      #  plot=1./np.square(_plot)
      #  data=1./np.square(_data)
      if read is 'r0':
        _plot,_data = extboot.prepare_r0(ens,self.obs.shape[-1])
        if square:
          plot=np.square(_plot)
          data=np.square(_data)
        else:
          plot=_plot
          data=_data

      if read is 'mss':
        _plot,_data = extboot.prepare_mss(ens,self.obs.shape[-1],meth=2)
        if square:
          plot=np.square(_plot)
          data=np.square(_data)
        else:
          plot=_plot
          data=_data 

      if read is 'fit_mpi':
        if len(filename) == 2: 
        # Instead of pseudosamples provide real fit results
          _mpi = fit.FitResult.read(filename[0])
          _mpi_data = _mpi.singularize().data[0][:,1,0] 
          ext_help = extboot.read_extern(filename[1],(1,2))
          # Data gets squared afterwards
          _fse_plot,_fse_data = extboot.prepare_fse(ext_help,ens,self.obs.shape[-1],
                                        square=False,had='pi')
          #print("mpi and error:")
          #print(compute_error(_mpi_data))
          data = _mpi_data/_fse_data
          
        else:
          _mpi = fit.FitResult.read(filename)
          data = _mpi.singularize().data[0][:,1,0]
          #print("mpi data has shape:")
          #print(data.shape)
          #print("mpi and error:")
          #print(compute_error(data))
        if square is True:
          data = data**2
        if isinstance(fac,float):
          data *= fac
        plot = compute_error(data)
        #print("%f M_pi**2" %fac)
        #print(plot)

      if read is 'mpi':
        #print("indexto insert extern data:")
        #print(dim,idx)
        #ext_help = extboot.read_extern(filename,(1,2))
        # take any y data dimension, since bootstrap samplesize shold not differ
        # build r0M_pi
        #plot,data = extboot.prepare_mpi(ext_help,ens,self.obs.shape[-1],
        #                                square=square,r0=r0)
        # Instead of pseudosamples provide real fit results
        _mpi = ana.FitResult.read(filename)
        data = _mpi.singularize().data[0][1]
        if isinstance(fac,float):
          data *= fac
        plot = compute_error(data)


      if read is 'fse_mk':
        ext_help = extboot.read_extern(filename,(1,2))
        plot,data = extboot.prepare_fse(ext_help,ens,self.obs.shape[-1],
                                        square=square,had='k')

      if read is 'fse_mpi':
        ext_help = extboot.read_extern(filename,(1,2))
        plot,data = extboot.prepare_fse(ext_help,ens,self.obs.shape[-1],
                                        square=square,had='pi')
      if read is 'halfmpi':
        #print("indexto insert extern data:")
        #print(dim,idx)
        if len(filename) > 1:
          ext_help = extboot.read_extern(filename[0],(1,2))
        else:
          ext_help = extboot.read_extern(filename,(1,2))
        # take any y data dimension, since bootstrap samplesize shold not differ
        # build r0M_pi
        if fse is False:
            _plot,_data = extboot.prepare_mpi(ext_help,ens,self.obs.shape[-1],
                                              square=square,r0=False)
        else:
            if len(filename) != 2:
              print("Not enough files to read from")
            # read Mpi lattice data
            ext_help1 = extboot.read_extern(filename[0],(1,2))
            # read K_Mpi
            ext_help2 = extboot.read_extern(filename[1],(1,2))
            _plot, _data = extboot.prepare_mpi_fse(ext_help1,ext_help2,ens,
                          self.obs.shape[-1], square=square,physical=physical)
        data = 0.5*_data
        plot = compute_error(data)
        #print("0.5*M_pi^FSE**2")
        #print(plot)

      if read is 'fse_pi':
        if len(filename) != 2:
          print("Not enough files to read from")
        # read Mpi lattice data
        ext_help1 = extboot.read_extern(filename[0],(1,2))
        # read K_Mpi
        ext_help2 = extboot.read_extern(filename[1],(1,2))
        plot,data = extboot.prepare_mpi_fse(ext_help1,ext_help2,ens,
                      self.obs.shape[-1], square=square,physical=physical)
      # Place data in MatchResult instance
      if idx is None:
        for i in range(self.obs.shape[0]):
          self.add_data(data,i,op=op,amu=amu)
      else:
        self.add_data(data,idx,op=op,amu=amu)

    def match_to(self, obs_to_match, meth=None, plot=True,plotdir=None,ens=None,
        label=None,debug=0):
      """Matches the quark mass to a given observable

      Parameters
      ----------
      obs : Observables at different strange quark masses 
      meth: How to match: 0: linear interpolation (only two values)
                          1: linear fit
                          2: quadratic interpolation      
      observable_match: Observable to match to (lattice units)

      Returns
      -------
      amu_s_match : nd-array holding the matched strange quark mass
                    parameters for 3 different methods
      """
      # init result array:
      self.coeffs = [0,0,0]
      obs=self.obs
      amu = self.amu
      
      # do only one method
      if isinstance(meth,int):
        if meth > 2:
          raise ValueError("Method not implementedi, yet. Need meth < 3")
        
        # Depending on the method "meth" calculate coefficients first and evaluate
        # the root of the function to find obs_match
        # Method 0 only applicable for 2 observables
        if meth == 0:
          print("Using linear interpolation")
          # TODO: decide which observables to take depending on matching
          if (np.mean(obs[0]) <= np.mean(obs_to_match)) & (np.mean(obs_to_match) <= np.mean(obs[1])):
            self.amu_match[0], self.coeffs[0] = get_x_lin(obs[0],obs[1],
                                                  amu[0:2], obs_to_match)
          else:
            self.amu_match[0], self.coeffs[0] = get_x_lin(obs[1],obs[2],
                                                  amu[1:3], obs_to_match)
        if meth == 1:
          if obs.shape[0] == 3:
            print("Using quadratic interpolation")
            self.amu_match[1], self.coeffs[1] = get_x_quad(obs[0],obs[1],obs[2],
                                                           amu,obs_to_match)
        if meth == 2:
          print("Using linear fit")
          self.amu_match[2], self.coeffs[2] = get_x_fit(obs[0],obs[1],obs[2],
                                                        amu,obs_to_match,debug=debug)
          # Summary
        # print summary to screen
        orig, std = compute_error(self.amu_match[meth])
        print("%s = %f +/- %f:" %(self.obs_id, orig, std))

      # Do all methods  
      else:
          if (np.mean(obs[0]) <= np.mean(obs_to_match)) & (np.mean(obs_to_match) <= np.mean(obs[1])):
            self.amu_match[0], self.coeffs[0] = get_x_lin(obs[0],obs[1],amu[0:2],
                                                        obs_to_match)
          else:
            self.amu_match[0], self.coeffs[0] = get_x_lin(obs[1],obs[2],amu[1:3],
                                                        obs_to_match)
          if obs.shape[0] == 3:
            self.amu_match[1], self.coeffs[1] = get_x_quad(obs[0],obs[1],obs[2],
                                                           amu, obs_to_match)
            self.amu_match[2], self.coeffs[2] = get_x_fit(obs[0],obs[1],obs[2],
                                                          amu, obs_to_match)
          else:
            self.amu_match[2], self.coeffs[2] = get_x__fit(obs[0],obs[1],obs[2],
                                                           amu, obs_to_match) 
          # print summary to screen
          meths = ['lin. ipol','quad. ipol.','lin fit']
          for i,m in enumerate(meths):
            orig, std = compute_error(self.amu_match[i])
            print("%s: %s = %f +/- %f:" %(m,self.obs_id, orig, std))
      
      # Write out data as ascii
      # write x-values
      head = "Sample  r_0m_s_0  r_0m_s_1  r_0m_s_2"
      name = plotdir +ens + "/match_x_vals.txt"
      fmt = '%d %.4f %.4f %.4f'
      self.save_samples_ascii(amu,name,head,fmt=fmt)
      # write y-values
      head = "Sample  M_diff_0  M_diff_1  M_diff_2"
      fmt = '%d %.4f %.4f %.4f'
      name = plotdir + ens + "/match_y_vals.txt"
      self.save_samples_ascii(obs,name,head,fmt=fmt)
      # write y-values to match
      head = "Sample  M_diff_Match"
      fmt = '%d %.4f'
      name = plotdir + ens + "/phys_y_vals.txt"
      self.save_samples_ascii(obs_to_match,name,head,fmt=fmt)
      # write matched x-value
      head = "Sample  r_0m_s_fix"
      fmt = '%d %.4f'
      name = plotdir + ens + "/phys_x_vals.txt"
      self.save_samples_ascii(self.amu_match[2],name,head,fmt=fmt)

      if plot:
        self.plot_match(obs_to_match,plotdir,ens,meth=meth,proc='match',label=label)

    def eval_at(self, amu_match, amu_x=None, meth = None,correlated=True, ens=None,plot = True,
                plotdir=None,label=None,y_lim=None):
      """Evaluates a given observable at a matched strange quark mass

      Parameters
      -----------
      observable : the observables to evaluate
      meth: How to match: 0: linear interpolation (only two values)
                          1: quadratic interpolation       
                          2: linear fit 
      amu_s_match: amu_s to evaluate at (lattice units)
      amu_x: ndarray, The x-values used in matching, if none use the predefined
           amu values

      Returns
      -------
      self.eval_obs : nd-array holding the matched observable values
                    for 3 different methods
      """
      
      # init result array:
      self.coeffs = [0,0,0]
      obs=self.obs
      # Get the shape of the amu_s values to match to
      dshape = np.asarray(amu_match).shape
      # if the array is 1d blow it up to 2d, all 3 matching methods will get the
      # same amu_s values
      if len(dshape) < 2:
        amu_match=np.full((3,dshape[0]),amu_match)
      self.amu_match=amu_match
      if self.debug > 0:
          print("matched value shape %r" % dshape)
      # Set the x-values in the evaluation procedure
      if amu_x is None:
        amu = self.amu
      else:
        if len(amu_x) < 2:
          raise ValueError("Inserted x values do not have right shape")
        else:
          amu = amu_x
      #print("In evaluation amu_x values are:")
      #print(amu_x[:,0])
      # do only one method
      if isinstance(meth,int):
        if meth > 2:
          raise ValueError("Method not implemented yet, meth < 3")
        
        if self.debug > 0:
            print("\nEvaluate %s" %self.obs_id)
        # Depending on the method "meth" calculate coefficients first and evaluate
        # the root of the function to find obs_match
        # Method 0 only applicable for 2 observables
        if meth == 0:
          print("Using linear interpolation")
          if in_ival(amu_match[0],amu[0],amu[1]):
            print("using smaller intervall")
            self.eval_obs[0], self.coeffs[0] = calc_y_lin(obs[0],obs[1],amu[0:2],amu_match[0])
          elif in_ival(amu_match[0],amu[1],amu[2]):
            self.eval_obs[0], self.coeffs[0] = calc_y_lin(obs[1],obs[2],amu[1:3],amu_match[0])
          else:
            print("Only extrapolation possible:")
            a,b = mh.choose_ival(amu_match[0],amu)
            self.eval_obs[0], self.coeffs[0] = calc_y_lin(obs[a],obs[b],amu[a:b+1],amu_match[0])
            
        if meth == 1:
          if obs.shape[0] == 3:
            print("Using quadratic interpolation")
            self.eval_obs[1], self.coeffs[1] = calc_y_quad(obs[0],obs[1],obs[2],amu,amu_match[1])

        if meth == 2:
          print("Using linear fit")
          self.eval_obs[2], self.coeffs[2] = calc_y_fit(obs[0],obs[1],obs[2],amu,amu_match[2],correlated)
        # print summary to screen
        orig, std = compute_error(self.eval_obs[meth])
        if self.debug > 0:
            print("%s = %f +/- %f:" %(self.obs_id, orig, std))
      # Do all methods  
      else:
        print("Using linear interpolation")
        if in_ival(amu_match,amu[0],amu[1]):
            self.eval_obs[0], self.coeffs[0] = calc_y_lin(obs[0],obs[1],amu[0:2],amu_match[0])
        else:
            self.eval_obs[0], self.coeffs[0] = calc_y_lin(obs[1],obs[2],amu[1:3],amu_match[0])
        if obs.shape[0] == 3:
          self.eval_obs[1], self.coeffs[1] = calc_y_quad(obs[0],obs[1],obs[2],amu,amu_match[1])
          self.eval_obs[2], self.coeffs[2] = calc_y_fit(obs[0],obs[1],obs[2],
                                                        amu,amu_match[1],
                                                        correlated)
        else:
          self.eval_obs[2], self.coeffs[2] = calc_y_fit(obs[0],obs[1],obs[2],amu,amu_match[2],correlated)

        # print summary to screen
        meths = ['lin. ipol','quad. ipol.','lin fit']
        for i,m in enumerate(meths):
          orig, std = compute_error(self.eval_obs[i])
          if self.debug > 0:
              print("%s: %s = %f +/- %f:" %(m,self.obs_id, orig, std))
      if plot:
        self.plot_match(self.eval_obs,plotdir,ens,meth=meth,proc='eval',
                label=label,y_lim = y_lim)

    def get_plot_error(self,dim='y'):
      y = np.zeros((self.obs.shape[0],2))
      if dim is 'y':
        for i,v in enumerate(self.obs):
          y[i,0],y[i,1] = compute_error(v)
      if dim is 'x':
        for i,v in enumerate(self.amu):
          y[i,0],y[i,1] = compute_error(v)

      return y 

    def plot_match(self,obs,plotdir,ens, proc=None, meth=None, label=None,
            y_lim=None):
      if self.debug > 0:
          print("match/eval amu_s value for plot:")
          print(compute_error(self.amu_match[meth]))
      if meth == 0:
        _meth = '_lipol'
      elif meth == 1:
        _meth = '_qipol'
      elif meth == 2:
        _meth = '_lfit'
      else:
        _meth='cmp'
      pmatch = PdfPages(plotdir+ens+'/'+proc+_meth+'_%s.pdf' % self.obs_id)

      # functions for interpolation methods
      line = lambda p,x: p[:,0]*x+p[:,1]
      para = lambda p,x: p[:,0]*x**2+p[:,1]*x+p[:,2]
      # calculate errors for plot
      y_plot = self.get_plot_error()
      x_plot = self.get_plot_error(dim='x')
      # Data points to plot
      if self.debug > 0:
          print("x-data for match:")
          print(x_plot)
          print("y-data for match:")
          print(y_plot)
          print("parameters to function:")
          print(self.coeffs[meth])
      if meth is None:
        # plot data
        plot_data(x_plot[:,0],y_plot[:,0],y_plot[:,1],label='data',dX=x_plot[:,1],col='k')
        # plot the functions
        plot_function(line,x_plot[:,0],self.coeffs[0],'lin.  ipol',fmt='--')
        plot_function(para,x_plot[:,0],self.coeffs[1],'quad. ipol',
                      fmt='r--',col='red')
        plot_function(line,x_plot[:,0],self.coeffs[2],'lin. fit',
                      fmt='b--',col='blue')
      elif meth == 0:
        plot_data(x_plot[:,0],y_plot[:,0],y_plot[:,1],label='data',dX=x_plot[:,1],col='k')
        #plot_function(line,x_plot[:,0],self.coeffs[meth],'lin. ipol',fmt='k--',
        #              ploterror=True)
        plot_function(line,(x_plot[0,0],x_plot[-1,0]),self.coeffs[meth],'lin.  ipol',
            ploterror = True, fmt='k--',debug =self.debug )
        if proc == 'match':
          plot_single_line(self.amu_match[meth],obs,label[2],col='g')
        else:
          plot_single_line(self.amu_match[meth],self.eval_obs[meth],label[2],col='g')
      elif meth == 1:
        plot_data(x_plot[:,0],y_plot[:,0],y_plot[:,1],label='data',dX=x_plot[:,1],col='k')
        #plot_function(para,x_plot[:,0],self.coeffs[meth],'quad.  ipol',
        #              ploterror=True,fmt='r--',col='red')
        plot_function(line,(x_plot[0,0],x_plot[-1,0]),self.coeffs[meth],'quad.  ipol',
            ploterror = True, fmt='r--', col='red',debug =self.debug )
        if proc == 'match':
          plot_single_line(self.amu_match[meth],obs,label[2],col='g')
        else:
          plot_single_line(self.amu_match[meth],self.eval_obs[meth],label[2],col='g')
      elif meth == 2:
        plot_data(x_plot[:,0],y_plot[:,0],y_plot[:,1],label='data',dX=x_plot[:,1],col='b')
        #plot_function(line,x_plot[:,0],self.coeffs[meth],'lin. fit',
        #    ploterror = True, fmt='b--', col='blue')
        plot_function(line,(x_plot[0,0],x_plot[-1,0]),self.coeffs[meth],'lin. fit',
            ploterror = True, fmt='b--', col='blue',debug =self.debug )
        if proc == 'match':
          plot_single_line(self.amu_match[meth],obs,label[2],col='r')
        else:
          plot_single_line(self.amu_match[meth],self.eval_obs[meth],label[2],col='r')
      else:
        raise ValueError("Method not known")
      #plt.autoscale_view(enable=True,axis='both',tight=False)
      #TODO: Automate that, since autoscale is not working
      #plt.xlim(0.015,0.022)
      xlo = np.amin(self.amu)-0.05*np.amin(self.amu)
      xhi = np.amax(self.amu)+0.05*np.amax(self.amu)
      if y_lim is not None:
        print("Plot match: Using ylim:")
        print(y_lim)
        plt.ylim(y_lim[0],y_lim[1])
        plt.yticks(np.linspace(y_lim[0],y_lim[1],4))
      else:
        plt.locator_params(axis='y',nbins=4, min_n_ticks=4)
      plt.xlim(xlo,xhi)
      plt.locator_params(axis='x',nbins=5, min_n_ticks=2)
      plt.tick_params(axis='both',labelsize=20)
      plt.xlabel(label[0],fontsize=24)
      plt.ylabel(label[1],fontsize=24)
      plt.legend(loc='upper right',numpoints=1,ncol=1,fontsize=18)
      pmatch.savefig()
      pmatch.close()
      plt.clf()
    
    def scale_obs_std(self,ix,fac):   
        """Scale the array such that the standard variation increases by fac

        The deviations of the array get scaled by the factor fac. Afterwards the
        mean gets added again and the 0th bootstrapsample gets set to its correct
        value

        Parameters
        ----------
        array: ndarray holding the bootstrap samples of interest
        fac: float, the factor by which the standard deviation is scaled

        """
        array=self.eval_obs[ix]
        mean = array[0]
        n = array.shape[0]
        tmp=fac*(array-mean)
        tmp+=mean
        tmp[0] = mean
        self.eval_obs[ix]=tmp
    
