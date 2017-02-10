import numpy as np
from scipy.optimize import leastsq
import scipy.stats
import matplotlib
matplotlib.use('Agg') # has to be imported before the next lines
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['axes.labelsize'] = 'large'

import chiral_utils as chut
import extern_bootstrap as extboot
import plot as plot
from fit import FitResult
from globalfit import ChiralFit
from statistics import compute_error, draw_gaussian_correlated
from plot_functions import plot_function

"""A class used for chiral analyses of matched results, does fitting and
plotting as well
"""

class ChirAna(object):
  """Class to conduct a chiral analysis in mu_l and mu_s for different
  observables

  The observables are, in the most general case, lists of ndarrays, reflecting the
  degrees of freedom in lattice spacing, mu_l, mu_s, fitting dimensions and bootstrapsamples.
  Thus the shape of classobject is a tuple with 5 entries, the 0th entry is the
  length of the 1st entry, which is a tuple itself. 
  an example of a shape tuple is (3,(5,4,3),3,2,1500)
  The dependent observables usually have
  the same layout
  In addition it is possible to fit the data to a given function
  """
  def __init__(self,proc_id=None,match=False,correlated=True,combined=False,
               fit_ms=False):
    """ Initialize Chiral Analysis object

    Parameters
    ----------
    proc_id : String identifying the analysis
    match : Boolean to determine if the data is matched in amu_s
    correlated : Boolean if fit is Correlated
    combined : Boolean to decide if combined fit is done
    fit_ms : is the strange quark mass fitted
    
    Datatypes
    ---------
    fitres : A FitResult or a list of FitResults
    """
    self.y_shape = None
    self.x_shape = None
    self.y_data = None
    self.x_data = None
    # Parameters of extrapolation
    self.fitres = None
    # the physical point in the analysis, x and y-value with errors
    self.phys_point_fitres = None
    self.phys_point = None
    self.proc_id = proc_id
    self.match = match
    self.combined = combined
    self.correlated = correlated
    # Result of a matching procedure
    self.amu_matched_to = None
    # Dictionary of values to evaluate at sorted by lattice spacing and
    # ensemblename
    self.eval_at = None
    # Use a flag to distinguish between fit for m_s and m_l
    self.fit_ms=fit_ms
    # store external data in dictionaries sorted by lattice spacing.
    # All variables that cannot be calculated go here
    self.ext_dat_lat=None
    # Furthermore we want to use the continuum extraploated data
    self.cont_dat = None
    # The functions for the continuum extrapolation and the function in the
    # continuum, defined for fitting and plotting separately
    
    # Some ensemble names
    self.lat_dict = None
    # This is the function for fitting
    self.cont_ext = None
    # This is the function for continuum (no a-dependence)
    self.cont_func = None
    # This is the function for plotting
    self.plot_cont_ext = None
    # This is the function for continuum plotting
    self.plot_cont_func = None

  def wipe_data(self, dim):

    if dim is 'y':
      lyt_y=self.y_shape
      self.y_data = []
      for a in range(lyt_y[0]):
        if len(lyt_y) == 5:
          tp_y = (lyt_y[1][a],lyt_y[2],lyt_y[3],lyt_y[4])
        elif len(lyt_y) == 4:
          tp_y = (lyt_y[1][a],lyt_y[2],lyt_y[3])
        else:
          tp_y = (lyt_y[1][a],lyt_y[2])
        self.y_data.append(np.zeros(tp_y))

    if dim is 'x':
      lyt_x=self.x_shape 
      self.x_data=[]
      for a in range(lyt_x[0]):
        if len(lyt_x) == 5:
          tp_x = (lyt_x[1][a],lyt_x[2],lyt_x[3],lyt_x[4])
        elif len(lyt_x) == 4:
          tp_x =g (lyt_x[1][a],lyt_x[2],lyt_x[3])
        else:
          tp_x = (lyt_x[1][a],lyt_x[2])
        self.x_data.append(np.zeros(tp_x))

  def create_empty(self, lyt_x, lyt_y, match=None, lat_dict=None, cont_ext=None,
      plot_cont_ext= None, cont_func=None, plot_cont_func=None):
    """Initialize a chiral analysis with some start parameters
    
    At the moment the first index is used to label the lattice spacing, the
    second index is a tuple encoding the number of lattices per spacing. the
    third index counts the number of matched results and the last index is the
    number of bootstrapsamples.
    Parameters
    ----------
    lyt_y : tuple, shape of the y data
    lyt_x : tuple, shape of the x data
    match : boolean to state if matched data is to be analysed, default is False
    """
    #print(len(lyt_x),len(lyt_y))
    # save layout for later use
    self.y_shape = lyt_y
    self.x_shape = lyt_x
    
      
    if self.match is True:
      self.amu_matched_to = []
      for a in range(lyt_y[0]):
        if len(lyt_y) == 5:
          tp_mu = (lyt_y[1][a],lyt_y[2],lyt_y[3],lyt_y[4])
        elif len(lyt_y) == 4:
          tp_mu = (lyt_y[1][a],lyt_y[2],lyt_y[3])
        else:
          tp_mu = (lyt_y[1][a],lyt_y[2])
        self.amu_matched_to.append(np.zeros(tp_mu))

    self.y_data = []
    for a in range(lyt_y[0]):
      if len(lyt_y) == 5:
        tp_y = (lyt_y[1][a],lyt_y[2],lyt_y[3],lyt_y[4])
      elif len(lyt_y) == 4:
        tp_y = (lyt_y[1][a],lyt_y[2],lyt_y[3])
      else:
        tp_y = (lyt_y[1][a],lyt_y[2])
      self.y_data.append(np.zeros(tp_y))

    self.x_data = []
    for a in range(lyt_x[0]):
      if len(lyt_x) == 5:
        tp_x = (lyt_x[1][a],lyt_x[2],lyt_x[3],lyt_x[4])
      elif len(lyt_x) == 4:
        tp_x = (lyt_x[1][a],lyt_x[2],lyt_x[3])
      else:
        tp_x = g(lyt_x[1][a],lyt_x[2])
      self.x_data.append(np.zeros(tp_x))
      print("x_data tuple for allocation")
      print(tp_x)
    self.phys_point = np.zeros((2,2))
    debug=True
    if debug is True:
      print("Created empty Chiral Analysis object with shapes:")
      print("x-data: length %d" % len(self.x_data))
      for i in self.x_data:
        print(i.shape)
    self.cont_ext = cont_ext
    self.plot_cont_ext = plot_cont_ext
    # Take chiral fit function for plotting unless stated otherwise
    if plot_cont_ext is None:
      self.plot_cont_ext = cont_ext
    else:
      self.plot_cont_ext = plot_cont_ext
    self.cont_func = cont_func
    self.plot_cont_func = plot_cont_func
    # Ensemble names as dictionary sorted by lattice spacing
    self.lat_dict = lat_dict
  #def save()
  #
  #@classmethod
  #def read(cls,):

  def extend_data(self,obs=['None',],dim='x'):
    """Extend data in the givene dimension for different values

    The data values in the given dimension are extended with ensemble specific
    observables, like a/r_0. Thus the dimensionality of the data is raised in
    order to enable global fitting.

    Parameters
    ----------
    obs : list of strings, observabls to add to data
    dim : string, x or ydata
    """
    # using data layouts create data for extra dimension
    if dim is 'x':
      _lyt = self.x_shape
      self.x_shape = _lyt+((len(obs)+1),)
    elif dim is 'y':
      _lyt = self.y_shape
      self.y_shape = _lyt+((len(obs)+1),)
    else:
      raise ValueError
      print("Type not known, choose between 'x' or 'y'")
    # extension is the larger array
    extension = []
    for a in range(_lyt[0]):
      if len(_lyt) == 4:
        _tp = (_lyt[1][a],_lyt[2],_lyt[3],len(obs)+1)
      else:
        _tp = (_lyt[1][a],_lyt[2],len(obs)+1)
      extension.append(np.zeros(_tp))
    # copy data over from self
    full_lat_space = ['A','B','D']
    for i,a in enumerate(extension):
      if dim is 'x':
        a[...,0] = self.x_data[i]
        if obs[0] is 'a':
          # initialise bootstrapsampled r0-values
          r0 = chut.prepare_r0(full_lat_space[i],_lyt[2])
          r0_inv_sq = np.square(1./r0)

          # get array into right shape
          if self.match is False:
            r0_inv_sq = np.tile(r0_inv_sq,_lyt[3]).reshape((_lyt[3],_lyt[2]))
            #print("intermeidate shape of a/r0")
            #print(r0_inv_sq.shape)
          r0_ins = np.zeros_like(a[...,0])
          for i in r0_ins:
            i = r0_inv_sq
          #print("shape for insertion")
          #print(r0_ins.shape)
          a[...,1] = r0_ins
      # update original data
      else:
        raise RuntimeError
        print("global fit only for multiple x dimenisons")
    self.x_data = a
    print("control squared a/r0 values:")
    for i in extension:
      print(i[:,:,0,0])
    return extension

  def add_data(self,data,idx,dim=None,op=False):
    """Add x- or y-data at aspecific index

    Parameters
    ----------
    data : ndarray, the data to add, usually bootstrapsamples
    idx : tuple, index as in creation: lattice spacing, ensemble index, strange
          quark mass, fit dimension
    dim : string, is it x- or y-data?
    op : string, should existent data at index and dimension be operated
           with data to add?
    """
    if dim =='x':
      #print("xdata to add has shape")
      #print(data.shape)
      #print("xdata to add to has shape")
      #print(self.x_data[idx[0]].shape)
      if op == 'mult':
        self.x_data[idx[0]][idx[1],idx[2],idx[3]] *= data
      # divide existent x_data by data
      elif op == 'div':
        self.x_data[idx[0]][idx[1],idx[2],idx[3]] /= data
      elif op == 'min':
        self.x_data[idx[0]][idx[1],idx[2],idx[3]] -= data
      else:
        self.x_data[idx[0]][idx[1],idx[2],idx[3]] = data
        #self.x_data[idx[0]][idx[1],:,idx[2]] = data
      #print(self.x_data[idx[0]][idx[1],idx[2],idx[3]][0])
    if dim =='y':
      if op == 'mult':
        self.y_data[idx[0]][idx[1],idx[2],idx[3]] *= data
      elif op == 'div':
        self.y_data[idx[0]][idx[1],idx[2],idx[3]] /= data
      elif op == 'min':
        self.y_data[idx[0]][idx[1],idx[2],idx[3]] -= data
      else:
        self.y_data[idx[0]][idx[1],idx[2],idx[3]] = data
      #print(self.y_data[idx[0]][idx[1],idx[2],idx[3]][0])
    if dim =='mu':
      if op == 'mult':
        self.amu_matched_to[idx[0]][idx[1],idx[2],idx[3]] *= data
      elif op == 'div':
        self.amu_matched_to[idx[0]][idx[1],idx[2],idx[3]] /= data
      elif op == 'min':
        self.amu_matched_to[idx[0]][idx[1],idx[2],idx[3]] -= data
      else:
        self.amu_matched_to[idx[0]][idx[1],idx[2],idx[3]] = data

  def add_extern_data(self,filename,idx,ens,dim=None,square=True,
                      physical=False,read=None,op=False,meth=1):
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
    physical : bool, determines if read in data should be converted to physical
                one
    op : string, if operation is given the data is convoluted with existing data
        at that index. if no operation is given data gets overwritten
    meth: int, 1 or 2 the method with which zp was determined in
    arxiv:1403.4504v3
    """
    if read is None:
      plot, data = np.zeros((self.x_data[0].shape[-1]))

    if read is 'r0_inv':
      _plot,_data = extboot.prepare_r0(ens,self.x_data[0].shape[-1])
      if square is True:
        plot=1./np.square(_plot)
        data=1./np.square(_data)
      else:
        plot=1./_plot
        data=1./_data

    if read is 'r0':
      _plot,_data = extboot.prepare_r0(ens,self.x_data[0].shape[-1])
      if square:
        plot=np.square(_plot)
        data=np.square(_data)
      else:
        plot=_plot
        data=_data

    if read is 'mss':
      _plot,_data = extboot.prepare_mss(ens,self.x_data[0].shape[-1],meth=2)
      if square:
        plot=np.square(_plot)
        data=np.square(_data)
      else:
        plot=_plot
        data=_data

    if read is 'a':
      _plot,_data = extboot.prepare_a(ens,self.x_data[0].shape[-1])
      if square:
        plot=np.square(_plot)
        data=np.square(_data)
      else:
        plot=_plot
        data=_data

    if read is 'mpi':
      #print("indexto insert extern data:")
      #print(dim,idx)
      ext_help = extboot.read_extern(filename,(1,2))
      # take any y data dimension, since bootstrap samplesize shold not differ
      # build r0M_pi
      if physical:
        plot,data = extboot.physical_mpi(ext_help,ens,
                                      self.x_data[0].shape[-1],
                                      square=square)
      else:
        plot,data = extboot.prepare_mpi(ext_help,ens,
                                     self.x_data[0].shape[-1],
                                     square=square)
    
    if read is 'k_fse':
      ext_help = extboot.read_extern(filename,(1,2))
      plot,data = extboot.prepare_fse(ext_help,ens,self.x_data[0].shape[-1],square=square)

    # Do finite size correction of the pion mass
    # CAVEAT: Only useable on squared pion data
    if read is 'fse_pi':
      if len(filename) != 2:
        print("Not enough files to read from")
      # read Mpi lattice data
      ext_help1 = extboot.read_extern(filename[0],(1,2))
      # read K_Mpi
      ext_help2 = extboot.read_extern(filename[1],(1,2))
      plot,data = extboot.prepare_mpi_fse(ext_help1,ext_help2,ens,
                    self.x_data[0].shape[-1], square=square,physical=physical)
      

    if read is 'mpiL':
      ext_help = extboot.read_extern(filename,(1,2))
      plot,data = extboot.prepare_mpi(ext_help,ens,self.x_data[0].shape[-1],square=square,r0=False)

    if read is 'fk_unit':
      ext_help = extboot.read_extern(filename,(1,2))
      plot,data = extboot.prepare_fk(ext_help,ens,self.x_data[0].shape[-1])
    
    # Read in interpolated data
    if read is 'fk_int':
      ext_help = extboot.read_extern(filename,(3,4))
      plot,data = extboot.prepare_fk(ext_help,ens,self.x_data[0].shape[-1],
                                  square=square) 

    if read is 'mk_unit':
      ext_help = extboot.read_extern(filename,(1,2))
      # prepare_fk also does the right for m_k
      # TODO: unify this at some point
      plot,data = extboot.prepare_fk(ext_help,ens,self.x_data[0].shape[-1])

    if read is 'zp':
      _plot,_data = extboot.prepare_zp(ens,self.x_data[0].shape[-1],meth=meth)
      if square:
        plot=np.square(_plot)
        data=np.square(_data)
      else:
        plot=_plot
        data=_data

    print("extern data added:")
    print("%s: %f " %(read, data[0]))
    self.add_data(data,idx,dim=dim,op=op)

  def get_data(self,idx,dim):
    """return data at agiven index tuple before any reduction"""
    if dim == 'y':
      _data = self.y_data[idx[0]][idx[1],idx[2],idx[3]]
    elif dim == 'x':
      _data = self.x_data[idx[0]][idx[1],idx[2],idx[3]]
    else:
      print("datadimension not known")
    return _data

  def get_data_fit(self,dof,index,v='y'):
    """ Collect the data for the fit from class instance

    Get x- or y-data and choose the data at a given index depending on the fixed
    degree of freedom. Data is reshaped in the end to have only 2 dimensions
    from which the last one is the number of bootstrapsamples
    Parameters
    ----------
    dof : char, the dimension of the data to keep fixed
    index : int, the index of the fixed dimension
    v : char, if it is x or y data to be taken into account
    """ 
    if v == 'y':
      _data = self.y_data
    if v == 'x':
      _data = self.x_data
    # get all data for a distinct lattice spacing
    if dof == 'a':
      tp = index
      data = _data[tp]
    # get all data for a fixed mu_s
    elif dof == 'mu_s':
      data = np.concatenate((_data[0][:,index],_data[1][:,index],_data[2][:,index]))
      _tmp_lyt = data.shape
      data=data.reshape((_tmp_lyt[0]*_tmp_lyt[1],_tmp_lyt[2]))
      #print("chose data")
      #print(data.shape)
    elif dof == 'nboot':
      tp = (slice(None),slice(None),slice(None),index)
      data = _data[tp]
    else:
      raise ValueError("unknown data dimension.")
    return data

  def get_mu_plot(self,dim=0,debug=0):
    """ Returns the data suited for plotting and printing a summary

    The data dimenisons are reduced such that for each datapoint there is an
    array of 4 values.

    Parameters
    ----------
    """
    # Get data for fit (several fit dimensions for x, only one for y)
    dummy,dummy,_mu_data = self.reduction(x_shape_new = (self.x_shape[3],self.x_shape[4]),
                                    y_shape_new = (self.y_shape[3],self.y_shape[4],))
    # initialize plot data
    _mu_plot = np.zeros((_mu_data.shape[0],4))
    # calculate errors and fill
    # Loop over ensembles (eventually including quark mass)
    for e in range(_mu_plot.shape[0]):
      if _mu_data.shape[1] == 2:
        _mu_plot[e][0:2] = compute_error(_mu_data[e,dim])
      else:
        _mu_plot[e][0:2] = compute_error(_mu_data[e])
    if debug > 0:
      print("_mu_data is:")
      print(_mu_data)
      print("get_mu_plot returns:")
      print(_mu_plot)
    return _mu_plot

  def get_data_plot(self,dim=0,mev=False,debug=0):
    """ Returns the data suited for plotting and printing a summary

    The data dimenisons are reduced such that for each datapoint there is an
    array of 4 values.

    Parameters
    ----------
    """
    # Get data for fit (several fit dimensions for x, only one for y)
    _x_data, _y_data, _mu_data = self.reduction(x_shape_new = (self.x_shape[3],self.x_shape[4]),
                                    y_shape_new = (self.y_shape[3],self.y_shape[4],))
    # initialize plot data
    _x_plot = np.zeros((_x_data.shape[0],_x_data.shape[1],4))
    _y_plot = np.zeros((_y_data.shape[0],4))
    # calculate errors and fill
    # check shapes of x and y
    if _x_plot.shape[0] != _y_plot.shape[0]:
      print("x and y have wrong shapes")
    # Loop over ensembles (eventually including quark mass)
    print("x_plot has shape:")
    print(_x_plot.shape)
    for e in range(_x_plot.shape[0]):
      # loop over dimensions of x
      for d in range(_x_plot.shape[1]):
        if mev:
          _x_data[e,0] = np.sqrt(_x_data[e,0])*1.e3
        _x_plot[e,d][0:2] = compute_error(_x_data[e,d])
      
      if _y_data.shape[1] == 2:
        _y_plot[e][0:2] = compute_error(_y_data[e,dim])
      else:
        _y_plot[e][0:2] = compute_error(_y_data[e])
    if debug > 0:
      print("get_data_plot returns:")
      print(_x_plot,_y_plot)
    return _x_plot, _y_plot

  def global_ms_func(self,r,z,p,x):
      return p[0]/(r*z) * (x[:,0]+x[:,1]) * (1+p[1]*(r/z)*x[:,0]+p[2]/(r**2))

  def global_ms_errfunc(self,p,x,y,cov):

      # define the fitfunction for a single beta
      _func = lambda r, z, p, x,: p[0]/(r*z) * (x[:,0]+x[:,1]) * (1+p[1]*(r/z)*x[:,0]+p[2]/(r**2))

      # TODO: Automate the array shapes, otherwise very errorprone
      # With A40.24
      # Get the residuals of all beta values
      #_res_a = _func(p[0],p[3],p[6:9],x[0:18])-y[0:18]
      #_res_b = _func(p[1],p[4],p[6:9],x[18:27])-y[18:27]
      #_res_d = _func(p[2],p[5],p[6:9],x[27:33])-y[27:33]
      ## residuals of r0 and zp are stored separately at the moment
      #_res_r0 = np.r_[(y[33]-p[0]),(y[34]-p[1]), (y[35]-p[2])]
      #_res_zp = np.r_[(y[36]-p[3]),(y[37]-p[4]),(y[38]-p[5])]
      # Get the residuals of all beta values
      # Without A40.24
      # TODO: Automate the array shapes, otherwise very errorprone
      _res_a = _func(p[0],p[3],p[6:9],x[0:15])-y[0:15]
      _res_b = _func(p[1],p[4],p[6:9],x[15:24])-y[15:24]
      _res_d = _func(p[2],p[5],p[6:9],x[24:30])-y[24:30]
      # residuals of r0 and zp are stored separately at the moment
      _res_r0 = np.r_[(y[30]-p[0]),(y[31]-p[1]), (y[32]-p[2])]
      _res_zp = np.r_[(y[33]-p[3]),(y[34]-p[4]),(y[35]-p[5])]
      # collect residuals as one array
      _residuals = np.r_[_res_a,_res_b,_res_d,_res_r0,_res_zp ]

      # calculate the chi values weighted with inverse covariance matrix
      _chi = np.dot(cov,_residuals)
      return _chi

  def fit_strange_mass(self, datadir='./', debug=4):
      r = np.r_[1.,1.,1.]
      z = np.r_[1.,1.,1.]
      p = np.r_[6.,0.1,1.]
      start=np.r_[r,z,p]
      # Get the prior samples
      pr_r0 = np.vstack((self.ext_dat_lat.get('A','r0'),
                    self.ext_dat_lat.get('B','r0'),
                    self.ext_dat_lat.get('D','r0')))

      pr_zp = np.vstack((self.ext_dat_lat.get('A','zp'), 
                    self.ext_dat_lat.get('B','zp'),
                    self.ext_dat_lat.get('D','zp')))
      print("0th zp-sample %r:" %(pr_zp[0]))
      print("0th r0-sample %r:" %(pr_r0[0]))
      print("shape zp-sample:")
      print(pr_zp.shape)
      #start = np.r_[p]
      # handle the data
      # x,y values for beta = 1.9
      # each array should be (n_ensembles(beta),n_ind,samples)
      x_shape_new = (self.x_shape[1][0]*self.x_shape[2],self.x_shape[3],self.x_shape[4])
      y_shape_new = (self.y_shape[1][0]*self.y_shape[2],self.y_shape[4])
      x0 = self.x_data[0].reshape(x_shape_new)
      y0 = self.y_data[0].reshape(y_shape_new)
      # x,y values for beta = 1.95
      x_shape_new = (self.x_shape[1][1]*self.x_shape[2],self.x_shape[3],self.x_shape[4])
      y_shape_new = (self.y_shape[1][1]*self.y_shape[2],self.y_shape[4])
      x1 = self.x_data[1].reshape(x_shape_new)
      y1 = self.y_data[1].reshape(y_shape_new)
      # x,y values for beta = 2.1
      x_shape_new = (self.x_shape[1][2]*self.x_shape[2],self.x_shape[3],self.x_shape[4])
      y_shape_new = (self.y_shape[1][2]*self.y_shape[2],self.y_shape[4])
      x2 = self.x_data[2].reshape(x_shape_new)
      y2 = self.y_data[2].reshape(y_shape_new)
      print("shape of start")
      print(start)
      print("shapes of measurements")
      x = np.r_[x0,x1,x2]
      y = np.r_[y0,y1,y2,pr_r0,pr_zp]
      print(x)
      print(y.shape)
      cov = np.zeros((y.shape[0],y.shape[0]))
      corr = np.zeros_like(cov)
      for i in range(11):
        _tmp = np.cov(y[3*i:3*i+3])
        #print("\nCovariance submatrix %d" %i)
        #print(_tmp)
        cov[3*i:3*i+3,3*i:3*i+3]=_tmp
      for k in range(30,36):
        cov[k,k] = np.cov(y[k])
      for l in range(cov.shape[0]):
        for m in range(cov.shape[0]):
          corr[l,m] = cov[l,m]/np.sqrt(cov[l,l]*cov[m,m])
      corr_heat = plot.LatticePlot('/hiskp2/helmes/analysis/scattering/analysis_vault/k_charged_wo_outliers/plots/global_fit_corr_matrix.pdf',join=True)
      corr_heat.plot_heatmap(corr,label=['correlation from modified covariance','yy'])
      corr_heat.plot_correlation(y,label=['correlation from data','yy'])
      del corr_heat
      # invoke a chiral fit, yielding a fitresult
      mk_phys = ChiralFit("ms_phys",self.global_ms_errfunc)
      self.fitres = mk_phys.chiral_fit(x,y,start,parlim=None,correlated=True,cov=cov,debug=debug)
      # Save the fitresult data
      self.fitres.save(datadir+self.proc_id+'.npz')
      #check the solution by computing relative deviation from measurement
      # get arguments
      args = self.fitres.data[0]
      print(args[0])
      #checkfunc = np.r_[self.global_ms_func(args[0,0,0],args[0,3,0],args[0,6:10,0],x[0:19]),
      #                  self.global_ms_func(args[0,1,0],args[0,4,0],args[0,6:10,0],x[19:28]),
      #                  self.global_ms_func(args[0,2,0],args[0,5,0],args[0,6:10,0],x[28:34])]
      #err = np.divide(1.,np.std(y,axis=1))
      #chisq = np.sum(self.global_ms_errfunc(args[0,:,0],x[...,0],y[...,0],err)**2)
      #print("calculated values:")
      #print(checkfunc[:,0])
      #print("measured values:")
      #print(y[:,0])
      #print("chisquared is: %.2f" %chisq)

  def mka0_func(self,r,z,p,x):

      # define the fitfunction for a single beta
      #_func = lambda r, z, p, x,: p[0]*r*x[:,0]/z + p[1]/r**2 + p[2]
      return p[0]*r*x[:,0]/z + p[3]*(r*x[:,0]/z)**2 + p[1]/r**2 + p[2] 

  def mka0_errfunc(self,p,x,y,cov):

      # define the fitfunction for a single beta
      _func = lambda r, z, p, x,: p[0]*r*x[:,0]/z + p[1]/r**2 + p[2]
      #_func = lambda r, z, p, x,: p[0]*r*x[:,0]/z*(1 + p[3]*(r*x[:,0]/z)) + p[1]/r**2 + p[2]
      # residuals for
      # With A40.24
      # Try out an additional term mu_ell^2
      # TODO: Automate the array shapes, otherwise very errorprone
      #_res_a = _func(p[0],p[3],p[6:10],x[0:6])-y[0:6]
      #_res_b = _func(p[1],p[4],p[6:10],x[6:9])-y[6:9]
      #_res_d = _func(p[2],p[5],p[6:10],x[9:11])-y[9:11]
      ## TODO: Automate the array shapes, otherwise very errorprone
      #_res_a = _func(p[0],p[3],p[6:9],x[0:6])-y[0:6]
      #_res_b = _func(p[1],p[4],p[6:9],x[6:9])-y[6:9]
      #_res_d = _func(p[2],p[5],p[6:9],x[9:11])-y[9:11]
      # residuals of r0 and zp are stored separately at the moment
      #_res_r0 = np.r_[(y[11]-p[0]),(y[12]-p[1]),(y[13]-p[2])]
      #_res_zp = np.r_[(y[14]-p[3]),(y[15]-p[4]),(y[16]-p[5])]
      # Without A40.24
      # TODO: Automate the array shapes, otherwise very errorprone
      _res_a = _func(p[0],p[3],p[6:9],x[0:5])-y[0:5]
      _res_b = _func(p[1],p[4],p[6:9],x[5:8])-y[5:8]
      _res_d = _func(p[2],p[5],p[6:9],x[8:10])-y[8:10]
      # residuals of r0 and zp are stored separately at the moment
      _res_r0 = np.r_[(y[10]-p[0]),(y[11]-p[1]),(y[12]-p[2])]
      _res_zp = np.r_[(y[13]-p[3]),(y[14]-p[4]),(y[15]-p[5])]
      # collect residuals as one array
      _residuals = np.r_[_res_a,_res_b,_res_d,_res_r0,_res_zp ]

      # calculate the chi values weighted with inverse covariance matrix
      _chi = np.dot(cov,_residuals)
      return _chi

  def fit_mka0(self,pr_1,pr_2,dat=None,debug=4):
    """ Global fit of evaluated data of M_K a_0 adapting Parameters for r_0/a
    and Z_P for each lattice spacing

    Parameters
    ----------
    pr1,2: arrays of prior values getting added to the y-values 
    """
    # The data for the fit should be just two arrays containing the
    # bootstrapsamples
    r = np.r_[1.,1.,1.]
    z = np.r_[1.,1.,1.]
    # Fit an additional quadratic dependency
    #p = np.r_[1.,1.,1.,1.]
    p = np.r_[1.,1.,1.]
    start=np.r_[r,z,p]
    # handle the data
    # x,y values for beta = 1.9
    # each array should be (n_ensembles(beta),n_ind,samples)
    x_shape_new = (self.x_shape[1][0]*self.x_shape[2],self.x_shape[3],self.x_shape[4])
    y_shape_new = (self.y_shape[1][0]*self.y_shape[2],self.y_shape[4])
    x0 = self.x_data[0].reshape(x_shape_new)
    y0 = self.y_data[0].reshape(y_shape_new)
    # x,y values for beta = 1.95
    x_shape_new = (self.x_shape[1][1]*self.x_shape[2],self.x_shape[3],self.x_shape[4])
    y_shape_new = (self.y_shape[1][1]*self.y_shape[2],self.y_shape[4])
    x1 = self.x_data[1].reshape(x_shape_new)
    y1 = self.y_data[1].reshape(y_shape_new)
    # x,y values for beta = 2.1
    x_shape_new = (self.x_shape[1][2]*self.x_shape[2],self.x_shape[3],self.x_shape[4])
    y_shape_new = (self.y_shape[1][2]*self.y_shape[2],self.y_shape[4])
    x2 = self.x_data[2].reshape(x_shape_new)
    y2 = self.y_data[2].reshape(y_shape_new)
    x = np.r_[x0,x1,x2]
    # Add the priors to the y-values
    y = np.r_[y0,y1,y2,pr_1,pr_2]
    if debug > 3:
      print("shape of x-data (n_ens,n_ind,n_boot):")
      print(x0.shape,x1.shape,x2.shape)
      print(x0[...,0],x1[...,0],x2[...,0])
      print("shape of y-data (n_ens,n_ind,n_boot):")
      print(y0.shape,y1.shape,y2.shape)
      print(y0[...,0],y1[...,0],y2[...,0])
    # Fit the data
    # invoke a chiral fit, yielding a fitresult
    mk_phys = ChiralFit("ms_phys",self.mka0_errfunc)
    cov=np.cov(y)
    self.fitres = mk_phys.chiral_fit(x,y,start,parlim=None,correlated=False,cov=None,debug=debug)
    # Save the fitresult data
    if dat is not None:
      self.fitres.save(dat+self.proc_id+'.npz')
    # Calculate check of the data for original data
    args = self.fitres.data[0]

  def mka0_phys(self,func,x_phys,path=None):
      # save samples of physical point result as fitresult along with p-values
      self.phys_point_fitres = self.fitres.calc_mk_a0_phys(x_phys,func)
      print ("physical point from fitresult")
      #print(self.phys_point_fitres.data[0],self.phys_point_fitres.pval[0])
      self.phys_point_fitres.calc_error()
      self.phys_point_fitres.print_data()
      self.phys_point = np.zeros((2,2))
      self.phys_point[0] = np.asarray(compute_error(x_phys))
      r,rstd,rsys,nfits = self.phys_point_fitres.error[0] 
      self.phys_point[1] = np.asarray((r[0][0],rstd[0])) 
      print("Calculated physical point to be:")
      print(self.phys_point)
      if path is not None:
        self.phys_point_fitres.save(path+self.proc_id+'phys_pt.npz')


#  def fit(self,index=0, start=[1.,],dim=None, x_phys=None,xcut=False,
#          plot=True,ploterr=True,label=None,datadir=None,read=False,
#          ens=None,debug=0,loc=None,xlim=None,ylim=None):
#    """fit a chiral analysis instance to a given fitfunction
#
#    This function uses the data of the ChirAna instance to fit the data to the
#    fitfunction specified in ChirAna. Different degrees of freedom are appliccable, an optional plot
#    can be made 
#
#    Parameters
#    ----------
#
#    dim : string, which dof to fix (a,mu_l,mu_s,nsamp)
#    index : int, index to fix dim
#    x_phys : x-value for physical point
#    xcut: float, optional cut on x-axis
#    plot: bool, should the fitresults be plotted?
#    label: tuple, x-,y-label for plot
#    datadir : string, directory for saving data
#    read : bool, read in previous fits
#    """
#
#    # Choose the fit data the dimensions of the data are: (a,mu_l,mu_s,nboot)
#    # with lattice spacing a, light and strange quark masses mu_l and mu_s and
#    # bootstrapsaFalses
#    if dim is None:
#      # Get data for fit (several fit dimensions for x, only one for y)
#      x_data, y_data, mu_data = self.reduction(x_shape_new = (self.x_shape[3],self.x_shape[4]),
#                                    y_shape_new = (self.y_shape[4],))
#
#      if read:
#        if xcut:
#          self.fitres = FitResult.read(datadir+self.proc_id+'_xcut_%d.npz'%xcut)
#        else:
#          self.fitres = FitResult.read(datadir+self.proc_id+'.npz')
#      else:
#        if self.combined is True:
#          self.fitres = chut.chiral_fit(x_data,y_data,self.cont_ext,corrid=self.proc_id,
#                                    start=start,xcut=xcut,correlated=self.correlated,
#                                    mute=chut.mutilate_cov,debug=debug)
#        else:
#          self.fitres = chut.chiral_fit(x_data,y_data,self.cont_ext,corrid=self.proc_id,
#                                    start=start,xcut=xcut,correlated=self.correlated,
#                                    debug=debug)
#
#      if xcut:
#        self.fitres.save(datadir+self.proc_id+'_xcut_%d.npz'%xcut)
#      else:
#        self.fitres.save(datadir+self.proc_id+'.npz')
#      args = self.fitres.data[0]
#      # save samples of physical point result as fitresult along with p-values
#      self.phys_point_fitres = self.fitres.calc_mk_a0_phys(x_phys,self.cont_func)
#      print ("physical point from fitresult")
#      #print(self.phys_point_fitres.data[0],self.phys_point_fitres.pval[0])
#      self.phys_point_fitres.calc_error()
#      self.phys_point_fitres.print_data()
#      self.phys_point_fitres.save(datadir+self.proc_id+'phys_pt.npz')
#
#      self.phys_point = np.zeros((2,2))
#      self.phys_point[0] = np.asarray(compute_error(x_phys))
#      r,rstd,rsys,nfits = self.phys_point_fitres.error[0] 
#      self.phys_point[1] = np.asarray((r[0][0],rstd[0])) 
#      print("Calculated physical point to be:")
#      print(self.phys_point)
#    # if a dimension is given set up a list of fitresults
#    else:
#      self.fitres = []
#      if dim == 'a':
#        #for a in range(self.x_shape[0]):
#        # At the moment D has only one Ensemble
#        for a in range(2):
#          _x_lat = self.get_data_fit('a',a,'x')
#          _y_lat = self.get_data_fit('a',a,'y')
#          x_data = _x_lat.reshape(_x_lat.shape[0]*_x_lat.shape[1],_x_lat.shape[2],_x_lat.shape[3])
#          # Usually y data is 1d in 3rd dimension
#          y_data = _y_lat.reshape(_y_lat.shape[0]*_y_lat.shape[1]*_y_lat.shape[2],_y_lat.shape[3])
#          self.fitres.append(chut.chiral_fit(x_data,y_data,self.cont_ext,correlated
#            = self.correlated,corrid=self.proc_id+'_lat_spc_%d'%a,
#                                    start=start,xcut=xcut,debug=debug))
#        self.phys_point = np.zeros((len(self.fitres),2,2))
#        for a in range(2):
#          args = self.fitres[a].data[0]
#          self.phys_point[a,0] = x_phys[0:2]
#          self.phys_point[a,1] = chut.err_phys_pt(args,x_phys,self.cont_ext)
#          print("Calculated physical point to be:")
#          print(self.phys_point)
#    if plot is True:
#      label=label
#      print(x_data.shape)
#      print(y_data.shape)
#      #if self.glob is True:
#      #  self.plot_plain(x_data,y_data,self.cont_ext,xcut=xcut,ens=ens)
#      #  self.plot_glob_func(x_data,y_data,self.cont_ext,xcut=xcut,ens=ens)
#      #else:
#      if self.combined:
#        #self.plot(x_data,y_data,label,xcut=xcut,ens=ens,plotfunc=self.plot_cont_ext,
#        self.plot(label,xcut=xcut,ens=ens,plotfunc=self.plot_cont_ext,
#                    savedir=datadir,loc=loc,xlim=xlim,ylim=ylim,ploterr=ploterr)
#        #label[2]=(r'Continuum extrapolation B')
#        #self.plot(x_data,y_data,label,xcut=xcut,ens=ens,plotfunc=self.plot_cont_ext,
#        self.plot(label,xcut=xcut,ens=ens,plotfunc=self.plot_cont_ext,
#                  savedir=datadir,loc=loc,dim=1,suffix='2',xlim=xlim,ylim=ylim,ploterr=ploterr)
#      else:
#          #self.plot(x_data,y_data,label,xcut=xcut,ens=ens,plotfunc=self.plot_cont_ext,
#          self.plot(label,xcut=xcut,ens=ens,plotfunc=self.plot_cont_ext,
#                    savedir=datadir,loc=loc,xlim=xlim,ylim=ylim,ploterr=ploterr)

  def print_summary(self,dim,index,lat_space,ens_dict,
                    mu_s_dict=None,xcut=2,head=None):
    """This function should print a summary of the whole chiral analysis,
    preferably in latex format
    """
    # Load data
    _x_summ, _y_summ = self.get_data_plot(mev=False)
    print("summary data:")
    print(_x_summ.shape)
    if self.match is True:
      _mu_summ = self.get_mu_plot(dim=0,debug=2)
    if head is None:
      header=['Ens','$a\mu_s$','$(r_0M_{\pi})^2$','$(a/r_0)^2$', '$M_Ka_0$']
    else:
      header=head
    chut.print_table_header(header)

    print('\midrule')
    l = 0
    for i,a in enumerate(lat_space):
      #if i > 0:
      #  l = len(ens_dict[lat_space[i-1]])
      #if i > 1:
      #  l = len(ens_dict[lat_space[i-1]])+ len(ens_dict[lat_space[i-2]])
      #else:
      #  l = 0
      # number of ensembles for each lattice spacing
      if i == 0:
        l = 0
      if i == 1:
        l = len(ens_dict[lat_space[i-1]])
      if i == 2:
        l += len(ens_dict[lat_space[i-1]])
      for j,e in enumerate(ens_dict[a]):
        # format for
        if self.match:
            # this is printing two dimensions
            if hasattr(_mu_summ,"__iter__"):
              chut.print_line_latex(e,_x_summ[l+j],_y_summ[l+j],_mu_summ[l+j])
            else:
              chut.print_line_latex(e,_x_summ[l+j],_y_summ[l+j])
        else:
          for k,s in enumerate(mu_s_dict[a]):
              #chut.print_line_latex(e,_x_summ[i*l*3+j*3+k],_y_summ[i*l*3+j*3+k])
              chut.print_line_latex(e,_x_summ[l*3+j*3+k],_y_summ[l*3+j*3+k])
    # TODO: Automate that
    if hasattr(self.fitres,"__iter__"):
      dof =  _x_summ.shape[0] - self.fitres[0].data[0].shape[1]
      if self.combined:
        dof = 2*_x_summ.shape[0] - self.fitres[0].data[0].shape[1]
      print("%10s & $%.1f$ & $%.4f(%1.0f)$ & $%.2f/%d$ & $%.2e $" %
          (self.proc_id, xcut, self.phys_point[0,1,0], self.phys_point[0,1,1]*1e4,
           self.fitres[0].chi2[0][0], dof, self.fitres[0].pval[0][0]))

      #dof =  _x_summ.shape[0] - self.fitres[1].data[0].shape[1]
      #print("%10s & $%.1f$ & $%.4f(%1.0f)$ & $%.2f/%d$ & $%.2e $" %
      #    (self.proc_id, xcut, self.phys_point[1,1,0], self.phys_point[1,1,1]*1e4,
      #     self.fitres[1].chi2[0][0], dof, self.fitres[1].pval[0][0]))
    else:
      dof = _x_summ.shape[0] - self.fitres.data[0].shape[1]
      if self.combined:
        print(_x_summ.shape[0])
        print(self.fitres[0].data[0].shape[1])
        dof = 2*_x_summ.shape[0] - self.fitres[0].data[0].shape[1]
      print("Phsyical point result:")
      if xcut is None:
        xcut = 2.
      print("%10s & $%.1f$ & %.2e %.2e & $%.4f(%1.0f)$ & $%.2f/%d$ & $%.2e $" %
          (self.proc_id, xcut, self.phys_point[0,0],self.phys_point[0,1],
           self.phys_point[1,0], self.phys_point[1,1]*1e4,
           self.fitres.chi2[0][0], dof, self.fitres.pval[0][0]))

  def reduction(self,x_shape_new=None,y_shape_new=None):
    """Function to reduce the dimensionality of the data to a two dimensional
    ndarray that can be handled by the plot function
    The final layout should concatenate the light and strange quark dependence. 

    The x and y data are cast to an 2d ndarray with same shape (apart from bootstrap
    samples)
    """
    # the outgoing data is 2 dimensional
    # new x-data has shape (ens1*ms+ens2*ms+ens3*ms,fitdim,samples)
    # looping over list place data in array
    print("x and y data get adapted:")
    print x_shape_new
    print y_shape_new
    if x_shape_new is None:
      tmp_x_shape = (self.x_shape[0],self.x_shape[1]*np.asarray(self.x_shape[2]),self.xshape[3],self.x_shape[-1])
    else:
      tmp_x_shape = (self.x_shape[0],self.x_shape[1]*np.asarray(self.x_shape[2]),x_shape_new[0],x_shape_new[1])
    if y_shape_new is None: 
      #tmp_y_shape = (self.y_shape[0],self.y_shape[1]*np.asarray(self.y_shape[2]),self.y_shape[-1])
      tmp_y_shape = (self.y_shape[0],self.y_shape[1]*np.asarray(self.y_shape[2]),self.y_shape[-1])
    else:
      tmp_y_shape = (self.y_shape[0],self.y_shape[1]*np.asarray(self.y_shape[2]),y_shape_new[0])
    #print("New Shapes:")
    #print(tmp_x_shape)
    #print(tmp_y_shape)
    tmp = ChirAna("reduction",match=self.match)
    tmp.create_empty(tmp_x_shape,tmp_y_shape)
    # Loop over lattice spacing
    for i,d in enumerate(self.x_data):
      if x_shape_new is None:
        try:
          new_shape = (d.shape[0]*d.shape[1],d.shape[2],d.shape[3])
        except:
          new_shape = (d.shape[0]*d.shape[1],d.shape[2])
          pass
      else:
        new_shape = (d.shape[0]*d.shape[1],x_shape_new[0],x_shape_new[1])
      #print("reshape to:")
      #print(new_shape)
      tmp.x_data[i] = d.reshape(new_shape)
      #print(tmp.x_data)
    for i,d in enumerate(self.y_data):
      print(d.shape)
      if d.shape[2] == 1:
        new_shape = (d.shape[0]*d.shape[1],d.shape[-1]) 
      else:
        new_shape = (d.shape[0]*d.shape[1],d.shape[2],d.shape[-1])
      tmp.y_data[i] = d.reshape(new_shape)
    if self.match is True:
      for i,d in enumerate(self.amu_matched_to):
        if d.shape[2] == 1:
          new_shape = (d.shape[0]*d.shape[1],d.shape[-1]) 
        else:
          new_shape = (d.shape[0]*d.shape[1],d.shape[2],d.shape[-1])
        tmp.amu_matched_to[i] = d.reshape(new_shape)
    if self.x_shape[0] == 3: 
      x_data = np.concatenate((tmp.x_data[0],tmp.x_data[1],tmp.x_data[2]))
      #print(x_data)
      y_data = np.concatenate((tmp.y_data[0],tmp.y_data[1],tmp.y_data[2]))
      if self.match is True:
        mu_data = np.concatenate((tmp.amu_matched_to[0],tmp.amu_matched_to[1],tmp.amu_matched_to[2]))
      #print(y_data.shape)
      #print(x_data.shape)
        return x_data, y_data, mu_data
      else:
        return x_data, y_data, None
    else:
      return None,None,None

  def calc_plot_ranges(self):
    """ Return the plot ranges for the different lattice spacings
    """
    if self.match is True:
      _a_range = (0,self.x_shape[1][0]*self.x_shape[2])
      _b_range = (_a_range[1],_a_range[1]+self.x_shape[1][1]*self.x_shape[2])
      _d_range = (_b_range[1],_b_range[1]+self.x_shape[1][2]*self.x_shape[2])

    else:
      _a_range = (0,self.x_shape[1][0]*self.x_shape[2])
      _b_range = (_a_range[1],_a_range[1]+self.x_shape[1][1]*self.x_shape[2])
      _d_range = (_b_range[1],_b_range[1]+self.x_shape[1][2]*self.x_shape[2])
    #print(_a_range)
    #print(_b_range)
    #print(_d_range)

    return _a_range, _b_range, _d_range

# TODO: Make the chiral plots interface more with the actual LatticePlot class
  def plot_plain(self,label,ens,savedir=None,xcut=None,xdim=0,debug=0):
    x_plot,y_plot = self.get_data_plot(dim=0,debug=0)
    if debug > 3:
        print("Do we use matched data? %s" %self.match)

        print("\nData used for plot: ")
        print("x-data:")
        print(x_plot)
        print(x_plot.shape)

        print("y-data")
        print(y_plot)

    if savedir is not None:
      path = savedir
    else:
      path = "./plots2/pdf"
    pfit = PdfPages(path+"/%s.pdf" % self.proc_id)
    if xcut:
        pfit = PdfPages(path+"/%s_xcut_%d.pdf" % (self.proc_id,xcut))
    else:
        pfit = PdfPages(path+"/%s.pdf" % self.proc_id)
    print('saving plot in: %s' %path)
    #calc xid's
    a_range, b_range, d_range = self.calc_plot_ranges()
    print(d_range)
    chut.plot_ensemble(x_plot[:,xdim],y_plot,'^vspho','red',ens['A'],
                       xid = a_range,match=self.match)
    chut.plot_ensemble(x_plot[:,xdim],y_plot,'vso','blue',ens['B'],
                       xid = b_range,match=self.match)
    chut.plot_ensemble(x_plot[:,xdim],y_plot,'^d','green',ens['D'],
                       xid = d_range,match=self.match)

    plt.grid(False)
    plt.xlim(0.002,0.011)
    #plt.ylim(-0.45,-0.28)
    plt.legend(loc='best',numpoints=1, ncol=2,fontsize=12)
    plt.ylabel(label[1])
    plt.xlabel(label[0])
    pfit.savefig()
    pfit.close()
    plt.clf()

  def plot_comp(self,cmp_name,savedir,savename,
                cont_func=None,dim=0,label=None,func=None,xlim=None,ylim=None):
    """Plot a continuum extrapolated curve and comparison data

    The arguments to the continuum function are taken from the chiral analysis
    object

    Parameters
    ----------
    cont_func : callable, the continuum function
    cmp_name : data name of the file for comparison
    savename : string of filename
    """
    
    # comparison data is assumed to be a textfile, this will be pretty much hard
    # coded
    # get the data and with plot function plot the continuum curve
    mka0_comp = np.loadtxt(cmp_name[0])
    mka0_comp2 = np.loadtxt(cmp_name[1])

    #plt.axvline(self.phys_point[0,0],color='gray',ls='dashed',label='physical point')
    # Set up a pdfPages object
    pfit = PdfPages(savedir+"/%s.pdf" % savename)
    # convert nplqcd pion masses to GeV^2 
    x_comp =np.square(mka0_comp[:,1]*197.37/(mka0_comp[:,3]*1.e3)) 
    print("NPLQCD to plot:")
    print(x_comp)
    # Plot the NPLQCD Data
    plt.errorbar(x_comp[:-1],mka0_comp[:-1,4],mka0_comp[:-1,5],
           color='black',fmt='s',label='NPLQCD, coarse')
    plt.errorbar(x_comp[-1],mka0_comp[-1,4],mka0_comp[-1,5],
           color='blue',fmt='s',label='NPLQCD, fine')

    # Plot the PACS-CS Data
    plt.errorbar(np.square(mka0_comp2[:,0]),mka0_comp2[:,2],mka0_comp2[:,3],
          color='tomato',fmt='o',label='PACS-CS')
    x_cont = np.linspace(0,50,1000)
    x_plot,y_plot = self.get_data_plot(dim=dim,mev=True)
    a_range, b_range, d_range = self.calc_plot_ranges()
    # Plot the continuum curve
    # Convention: if dim is 1 take argument 0,3,2 instead of 0,1,2,
    if dim == 0:
      args = self.fitres.data[0]
    if dim == 1:
      arg_shape = self.fitres.data[0].shape
      args = np.column_stack((self.fitres.data[0][:,0],self.fitres.data[0][:,3],
                       self.fitres.data[0][:,2])).reshape(arg_shape[0],3,arg_shape[-1])
    #Multiply second parameter with second x-variable
    _args_a = np.copy(args)
    _args_a[:,1,:] *= x_plot[a_range[0],1,0] 
    plot_function(func,xlim,_args_a[:,:,0],
            label=r'$a=0.0885$fm',ploterror=ploterr,fmt='r--', col='red')
    _args_b = np.copy(args)
    _args_b[:,1,:] *= x_plot[b_range[0],1,0] 
    plot_function(func,xlim,_args_b[:,:,0],
            label=r'$a=0.0815$fm',ploterror=ploterr,fmt='b:', col='blue')
    _args_d = np.copy(args)
    _args_d[:,1,:] *= x_plot[d_range[0],1,0] 
    plot_function(func,xlim,args[:,:,0],
            label=r'$a=0.0619$fm',ploterror=ploterr,fmt='g-.', col='green')

    #cont_curve = lambda p,x : p[0]*x+p[2]
    cont_curve = self.cont_func
    plot_function(cont_curve, x_cont,args[:,:,0],fmt='k--',
            label='Cont. Ext. B',ploterror=ploterr)
    phys_pt = []
    l1,a,b = plt.errorbar(self.phys_point[0,0],mka0_comp[0,6],mka0_comp[0,7],
                 color='darkorange',fmt='s')
    l2,a,b = plt.errorbar(self.phys_point[0,0],mka0_comp2[0,4],mka0_comp2[0,5],
                 color='darkorange',fmt='o')
    try:
      for a in self.phys_point:
        print("%f +/- %f" %(a[1,0],a[1,1]))
      l3,a,b = plt.errorbar(self.phys_point[:,0,0],self.phys_point[:,1,0],self.phys_point[:,1,1], fmt='d', color='darkorange', label='Physical Point')
    except:
      print("%f +/- %f" %(self.phys_point[1,0],self.phys_point[1,1]))
      l3,a,b = plt.errorbar(self.phys_point[0,0],self.phys_point[1,0],self.phys_point[1,1], fmt='d', color='darkorange')
      pass
    # Collect legends
    phys_pt.append([l1,l2,l3])
    #limits mka0
    plt.xlim(0,0.52)
    plt.ylim(-0.65,-0.28)
    #plt.vlines(self.phys_point[0,0],y_lim[0],y_lim[1],color="k",label=label[3])
    loc=None
    if loc==None:
      legend1=plt.legend(phys_pt[0],[r'NPLQCD $(M_K/f_K)^2$',
              r'PACS-CS $(M_K/f_K)^2$','ETMC'], title='Extrapolated',
              numpoints=1, ncol=1, fontsize=12,loc='lower left')
      plt.legend(loc='best',numpoints=1,ncol=2,fontsize=12)
      plt.gca().add_artist(legend1)
    else:
      legend1=plt.legend(phys_pt[0],ncol=1,fontsize=12,loc='lower left')
      plt.legend(loc=loc,numpoints=1,ncol=2,fontsize=12)
      plt.gca().add_artist(legend1)
    plt.title('Comparison')
    plt.ylabel(label[1])
    plt.xlabel(label[0])
    pfit.savefig()
    pfit.close()
    plt.clf()
    
  def plot(self,label,xcut=False,ens=None,plotfunc=None,ploterr=True,
           savedir=None,loc=None,suffix=None,dim=0,pfit=None,xlim=None,ylim=None):
    """Plot the chiral analysis data and the fitted function
       for the fitfunction the arguments are retrieved from the analysis object 
    Parameters
    ---------
    x_data : ndarray, The x_data considered in this plot, atm 2d with
             statistical errors is possible
    y_data : nd_array, The y data for the plot and the fit.
    label : a label for the plot
    xcut : should the data be cut to a smaller x-range?
    ens : label for the ensemble
    plotfunc : The function used for plotting 
    """
    #limits for plotting
    try:
        plt.xlim(xlim[0],xlim[1])
        plt.ylim(ylim[0],ylim[1])
    except:
        print("No plotting limits set, using default values.")
    x_plot,y_plot = self.get_data_plot(dim=dim,mev=False)
    print("\nData used for plot: ")
    print("x-data:")
    print(x_plot)
    print("y-data")
    print(y_plot)
    if self.combined is False:
        if savedir is not None:
          path = savedir
        else:
          path = "./plots2/pdf"
        if xcut:
          if suffix is not None:
            pfit = PdfPages(path+"/%s_%s_xcut_%d.pdf" % (self.proc_id,suffix,xcut))
          else:
            pfit = PdfPages(path+"/%s_xcut_%d.pdf" % (self.proc_id,xcut))
        else:
          if suffix is not None:
            pfit = PdfPages(path+"/%s_%s_%d.pdf" % (self.proc_id,suffix))
          else:
            pfit = PdfPages(path+"/%s.pdf" % (self.proc_id))

    else:
        if savedir is not None:
          path = savedir
        else:
          path = "./plots2/pdf"
        if suffix is not None:
          pfit = PdfPages(path+"/%s_%s.pdf" % (self.proc_id,suffix))
        else:
          pfit = PdfPages(path+"/%s.pdf" % self.proc_id)

    x = np.linspace(0,np.amax(x_plot),1000)
    #print("arguments for plotting are:")
    #print(args)
    a_range, b_range, d_range = self.calc_plot_ranges()
    #print("x-data for function plot:")
    #print(x_plot[a_range[0]:a_range[1],:,0])
    #print(x_plot[b_range[0]:b_range[1]])
    #print(x_plot[d_range[0]:d_range[1]])

    # check if several fits are made:
    if hasattr(self.fitres,"__iter__"):
      # D-Ensembles not taken into account!
      args = []
      for i in range(2):
        args.append(self.fitres[i].data[0])

      plot_function(plotfunc,x_plot[a_range[0]:a_range[1],:,0],
                    args[0][:,:,0],label=r'NLO-fit A',ploterror=True,col='red')
      chut.plot_ensemble(x_plot[:,0],y_plot,'^vspho','red',ens['A'],
                         xid = a_range,match=self.match)

      plot_function(plotfunc,x_plot[b_range[0]:b_range[1],:,0],
                    args[1][:,:,0],label=r'NLO-fit B',ploterror=True,col='blue')
      chut.plot_ensemble(x_plot[:,0],y_plot,'^vso','blue',ens['B'],
                         xid = b_range,match=self.match)

    else:
      # Convention: if dim is 1 take argument 0,3,2 instead of 0,1,2,
      if dim == 0:
        args = self.fitres.data[0]
      if dim == 1:
        arg_shape = self.fitres.data[0].shape
        args = np.column_stack((self.fitres.data[0][:,0],self.fitres.data[0][:,3],
                         self.fitres.data[0][:,2])).reshape(arg_shape[0],3,arg_shape[-1])
      print("Using Arguments")
      print(args.shape)
      print(args)
      # Modify plotting parameter for strange quark mass variable r0ms
      # modify arguments such that plot_function can handle them
      # For that to work, the second slice of arguments must be the inverse
      # lattice parameter spacing
      _args_a = np.copy(args)
      _args_b = np.copy(args)
      _args_d = np.copy(args)
      _args_a[:,1,:] *= x_plot[a_range[0],1,0] 
      _args_b[:,1,:] *= x_plot[b_range[0],1,0]
      _args_d[:,1,:] *= x_plot[d_range[0],1,0]
      if self.fit_ms:
          _add = np.copy(args[:,0])
          print("physical x-value for r0ms is:")
          print(self.amu_matched_to[0])
          _add *= 0.223479
          print(_add[0])
          _add = _add.flatten()
          print("Added argument has shape:")
          print(_add)
      else:
          _add = None
      # Plot function can only plot in one dimension.
      plot_function(plotfunc, xlim, _args_a[:,:,0],add=_add,
                    label=r'$a=0.0885\,$fm', ploterror=ploterr, fmt='r--', col='red')
      chut.plot_ensemble(x_plot[:,0], y_plot, '^', 'red', [r'$a=0.0885\,$fm',],
                        xid = a_range, match=self.match)

      plot_function(plotfunc, xlim, _args_b[:,:,0],add=_add,
                    label=r'$a=0.0815\,$fm', ploterror=ploterr, fmt='b:', col='blue')
      chut.plot_ensemble(x_plot[:,0], y_plot,'v', 'blue', [r'$a=0.0815\,$fm',],
                         xid = b_range, match=self.match)
      plot_function(plotfunc, xlim, _args_d[:,:,0],add=_add,
                    label=r'$a=0.0619\,$fm', ploterror=ploterr, fmt='g-.', col='green')
      chut.plot_ensemble(x_plot[:,0], y_plot,'o','green',[r'$a=0.0619\,$fm',],
                           xid = d_range,match=self.match)
      # Plot the continuum curve
      x_cont = np.linspace(0,50,1000)
      cont_curve = self.plot_cont_func 
      plot_function(cont_curve, [x_cont[0],x_cont[-1]],args[:,:,0],fmt='k--',add=_add,
                    label='continuum',ploterror=True)
    if xcut:
      y = plotfunc(args[0,:,0], xcut)
      plt.vlines(xcut, 0.95*y, 1.05*y, colors="k", label="")
      plt.hlines(0.95*y, xcut*0.98, xcut, colors="k", label="")
      plt.hlines(1.05*y, xcut*0.98, xcut, colors="k", label="")
    print("Physical point is:")
    try:
      for a in self.phys_point:
        print("%f +/- %f" %(a[1,0],a[1,1]))
      plt.errorbar(self.phys_point[:,0,0],self.phys_point[:,1,0],
                   self.phys_point[:,1,1],xerr=self.phys_point[:,0,1],
                   fmt='d', color='darkorange', label='Physical Point')
    except:
      print("%f +/- %f" %(self.phys_point[1,0],self.phys_point[1,1]))
      plt.errorbar(self.phys_point[0,0],self.phys_point[1,0],
                   self.phys_point[1,1],xerr=self.phys_point[0,1],
                   fmt='d', color='darkorange', label='Physical Point')
      pass
    plt.grid(False)
    #plt.vlines(self.phys_point[0,0],y_lim[0],y_lim[1],color="k",label=label[3])
    if loc==None:
      plt.legend(loc='best',numpoints=1,ncol=2,fontsize=12)
    else:
      plt.legend(loc=loc,numpoints=1,ncol=2,fontsize=12)
    #switch off title for publishing
    #plt.title(label[2])
    plt.ylabel(label[1])
    plt.xlabel(label[0])
    pfit.savefig()
    pfit.close()
    plt.clf()

  def calc_ms(self, mk_phys, r0_phys, ml_phys):
    """Calculate the strange quark mass from the fit parameters

    Parameters
    ----------
    mk_phys: float, physical kaon mass
    r0_phys: float, continuum extrapolation of Sommer parameter
    ml_phys: float, chirally extrapolated light quark mass
    """
    _hbarc = 197.37
    _p = self.fitres.data[0]
    _num = (r0_phys*mk_phys/_hbarc)**2
    _den = (_p[:,0,0]*(1 + _p[:,2,0]*r0_phys*ml_phys/_hbarc))
    _sub = r0_phys*ml_phys/_hbarc
    #print("result components (num, den, sub): %f, %f, %f" %(_num,_den[0],_sub))
    _r0ms = _num/_den - _sub 
    _m_s = _r0ms*_hbarc/r0_phys
    print("samples of _m_s: %r" % _m_s)
    return _m_s

  def calc_r0ms(self, r0_phys, mk_phys, ml_phys):
    """Calculate the strange quark mass from the fit parameters

    Parameters
    ----------
    mk_phys: float, physical kaon mass
    ml_phys: float, chirally extrapolated light quark mass
    """
    _hbarc = 197.37
    _p = self.fitres.data[0]
    _num = (r0_phys*mk_phys/_hbarc)**2
    _den = (_p[:,0,0]*(1 + _p[:,1,0]*r0_phys*ml_phys/_hbarc))
    _sub = r0_phys*ml_phys/_hbarc
    #print("result components (num, den, sub): %f, %f, %f" %(_num,_den[0],_sub))
    _r0ms = _num/_den - _sub 
    print("samples of _r0ms: %r" % _r0ms)
    return _r0ms
    #return compute_error(_r0ms)

  def bare_mu_s(self, space, ens, cont_data, mul, disc_eff=False, debug = 0):
    """ Fill a dictionary at self.eval_at with amus values computed from
    a fit

    Parameters
    ----------
    """

    if self.eval_at is None:
      self.eval_at = {}
      print(self.eval_at)
      # Get physical variables
      _r0 = cont_data.get('r0')
      _ml = cont_data.get('m_l')
      _mk = cont_data.get('mk')

      #### Look at modeled correlation between data ###
      ##print("Calculating amu_s with full anticorrelation between r0 and (mk and ml)")
      ##_corr = np.array((1,-1,-1,-1,1,1,-1,1,1)).reshape(3,3)
      #print("Calculating amu_s with full correlation between r0,mk and ml")
      #_corr = np.array((1,1,1,1,1,1,1,1,1)).reshape(3,3)
      #_data = np.array((0.474,494.2,3.7))
      #_err = np.array((0.014,0.4,0.17))
      #_samp = draw_gaussian_correlated(_data,_err,_corr)
      #_r0 = _samp[:,0]
      #_mk = _samp[:,1]
      #_ml = _samp[:,2]
      if debug > 0:
          _corrcoef = np.corrcoef(_r0,_mk)
          print("Correlation between r0 and mk: %.3f" % _corrcoef[0,1])
          _corrcoef = np.corrcoef(_r0,_ml)
          print("Correlation between r0 and ml: %.3f" % _corrcoef[0,1])
          _corrcoef = np.corrcoef(_mk,_ml)
          print("Correlation between mk and ml: %.3f" % _corrcoef[0,1])
          # Print relative error of (r0mk)^2 and (r0ml)
          _r0mk_sq = np.square(_r0*_mk/197.37)
          _r0ml = _r0*_ml/197.37
          _m_r0mk,_d_r0mk = compute_error(_r0mk_sq)
          _m_r0ml,_d_r0ml = compute_error(_r0ml)
          print("(r0mk)^2: %.3f +- %.3f, rel.err. = %.2f %%"
              %(_m_r0mk, _d_r0mk, _d_r0mk/_m_r0mk*100))
          print("r0ml: %.2e +- %.2e, rel.err. = %.2f %%"
              %(_m_r0ml, _d_r0ml, _d_r0ml/_m_r0ml*100))

      # Check input of physical observables
      # Select the correct Arguments from fitres for each lattice spacing it is
      # pr,pz,p0,p1,p2
      # TODO: What happens for two lattice spacings
      _tmp = self.fitres.data[0]
      _args = np.asarray([np.hstack((_tmp[:,0+i],_tmp[:,3+i],_tmp[:,6:,0])) 
              for i in range(3)])
      # For loop over lattice spacings
      for i,a in enumerate(space):
        # For loop over each ensemble
        for j,e in enumerate(ens[a]):
          _mul = mul[a][j]
          _mus = chut.compute_bare_mu_s(_r0,_ml,_mk,_mul,_args[i],disc_eff=disc_eff)
          if debug > 0:
            _mus_m, _mus_err = compute_error(_mus)
            # If discretisation effects are taken into account, amu_s formula
            # not valid for continuum value, calculate anew
            if disc_eff is True:
                _tmp_mus = chut.compute_bare_mu_s(_r0,_ml,_mk,_mul,_args[i])
                _ms = _args[i][:,0]/_r0*_tmp_mus/_args[i][:,1]*197.37 
            else:
                _ms = _args[i][:,0]/_r0*_mus/_args[i][:,1]*197.37
            m_s, m_s_err = compute_error(_ms)
            print("mus on Ensemble %s is : %.4f +- %.4f" %(e,_mus_m,_mus_err))
            print(" --> ms_phys = %.3f +- %.3f MeV" %(m_s,m_s_err))
          self.eval_at[e] = _mus
      #compute_bare_mu_s
    else:
      print("Prevented data override, nothing happened.")

################################################################################
################# Scratch region for trying out functions ######################
################################################################################


