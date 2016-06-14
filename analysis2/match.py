"""
Class for matching procedures. It behaves similar to the FitResult class,
perhaps make it a derived class of some Metaobject
"""
import matplotlib
matplotlib.use('Agg') # has to be imported before the next lines
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
matplotlib.rcParams['font.size'] = 14
import numpy as np

from match_functions import *
from match_help import in_ival
from .plot import plot_lines,plot_single_line
from .plot_functions import plot_data, plot_function
from .statistics import compute_error

class MatchResult(object):
    """Class to process FitResults for Interpolation

    The data has a similar layout to FitResult
    """
    def __init__(self, obs_id=None):

      """Allocate objects for initiated instance
      Parameters
      ----------
        obs_id : Observable that is investigated

      Members
      -------
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
    
    #@classmethod
    #def read()
    #def save()
    def create_empty(self, nboot, nb_obs, obs_id):
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
      self.amu = np.zeros(nb_obs)
      if self.obs_id is None:
        self.obs_id = obs_id

    # get data from collection of fitresults
    def load_data(self,fitreslst,par,amu,square=False,mult=None,debug=0):
      """load a list of fitresults and a list of amu values into an existing instance of
      MatchResult

      The Median of the FitResults is used to set the data in the MatchResult
      instance

      Parameters
      ----------
      """
      # Check lengths
      if len(fitreslst) != amu.shape[0]:
        raise ValueError("Fitresults and amu have different length")
      # from a list of fitresults, check if singular, singularize, if needed,
      for i,r in enumerate(fitreslst):
        r = r.singularize()
        self.obs[i] = r.data[0][:,par,0]
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
      self.obs = obs
      self.amu = amu

    def add_data(self,idx,obs,amu):
      """Set observable data and amu-values at one index
      """
      if self.obs is None:
        raise RuntimeError("MatchResult not initialized call create empty first")
      else:
        self.obs[idx] = obs
        self.amu[idx] = amu

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
      
      if plot:
        self.plot_match(obs_to_match,plotdir,ens,meth=meth,proc='match',label=label)

    def eval_at(self, amu_match, meth = None, ens=None,plot = True,plotdir=None,label=None):
      """Evaluates a given observable at a matched strange quark mass

      Parameters
      -----------
      observable : the observables to evaluate
      meth: How to match: 0: linear interpolation (only two values)
                          1: linear fit
                          2: quadratic interpolation      
      amu_s_match: amu_s to evaluate at (lattice units)

      Returns
      -------
      self.eval_obs : nd-array holding the matched observable values
                    for 3 different methods
      """
      
      # init result array:
      self.coeffs = [0,0,0]
      obs=self.obs
      amu = self.amu
      if  hasattr(amu_match,"__iter__") is False:
        amu_match=np.full((3),amu_match)
      self.amu_match=amu_match

      # do only one method
      if isinstance(meth,int):
        if meth > 2:
          raise ValueError("Method not implemented yet, meth < 3")
        
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
          self.eval_obs[2], self.coeffs[2] = calc_y_fit(obs[0],obs[1],obs[2],amu,amu_match[2])
        # print summary to screen
        orig, std = compute_error(self.eval_obs[meth])
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
          self.eval_obs[2], self.coeffs[2] = calc_y_fit(obs[0],obs[1],obs[2],amu,amu_match[1])
        else:
          self.eval_obs[2], self.coeffs[2] = calc_y_fit(obs[0],obs[1],obs[2],amu,amu_match[2])

        # print summary to screen
        meths = ['lin. ipol','quad. ipol.','lin fit']
        for i,m in enumerate(meths):
          orig, std = compute_error(self.eval_obs[i])
          print("%s: %s = %f +/- %f:" %(m,self.obs_id, orig, std))
      if plot:
        self.plot_match(self.eval_obs,plotdir,ens,meth=meth,proc='eval',label=label)

    def get_plot_error(self):
      y = np.zeros((self.obs.shape[0],2))
      for i,v in enumerate(self.obs):
        y[i,0],y[i,1] = compute_error(v)
      return y 

    def plot_match(self,obs,plotdir,ens, proc=None, meth=None, label=None):
      if meth == 0:
        _meth = '_lipol'
      elif meth == 1:
        _meth = '_qipol'
      elif meth == 2:
        _meth = '_lfit'
      else:
        _meth=None
      pmatch = PdfPages(plotdir+ens+'/'+proc+_meth+'_%s.pdf' % self.obs_id)
      line = lambda p,x: p[0]*x+p[1]
      para = lambda p,x: p[0]*x**2+p[1]*x+p[2]
      # calculate errors for plot
      y_plot = self.get_plot_error()
      if meth is None:
        # plot data
        plot_data(self.amu,y_plot[:,0],y_plot[:,1],label='data')
        # plot the functions
        plot_function(line,self.amu,self.coeffs[0],'lin.  ipol',)
        plot_function(para,self.amu,self.coeffs[1],'quad. ipol',
                      fmt='r',col='red')
        plot_function(line,self.amu,self.coeffs[2],'lin. fit',
                      fmt='b',col='blue')
      elif meth == 0:
        plot_data(self.amu,y_plot[:,0],y_plot[:,1],label='data')
        plot_function(line,self.amu,self.coeffs[meth],'lin. ipol',
                      ploterror=True)
        if proc == 'match':
          plot_single_line(self.amu_match[meth],obs,label[0:2],col='k')
        else:
          plot_single_line(self.amu_match[meth],self.eval_obs[meth],label[0:2],col='k')
      elif meth == 1:
        plot_data(self.amu,y_plot[:,0],y_plot[:,1],label='data')
        plot_function(para,self.amu,self.coeffs[meth],'quad.  ipol',
                      ploterror=True,fmt='r',col='red')
        if proc == 'match':
          plot_single_line(self.amu_match[meth],obs,label[0:2],col='r')
        else:
          plot_single_line(self.amu_match[meth],self.eval_obs[meth],label[0:2],col='r')
      elif meth == 2:
        plot_data(self.amu,y_plot[:,0],y_plot[:,1],label='data')
        plot_function(line,self.amu,self.coeffs[meth],'lin. fit',
            ploterror = True, fmt='b', col='blue')
        if proc == 'match':
          plot_single_line(self.amu_match[meth],obs,label[0:2],col='b')
        else:
          plot_single_line(self.amu_match[meth],self.eval_obs[meth],label[0:2],col='b')
      else:
        raise ValueError("Method not known")
      plt.title(ens)
      plt.xlabel(label[0])
      plt.ylabel(label[1])
      plt.legend(loc='best',numpoints=1,ncol=2)
      pmatch.savefig()
      pmatch.close()
      plt.clf()

