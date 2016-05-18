import numpy as np
import matplotlib
matplotlib.use('Agg') # has to be imported before the next lines
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages

import chiral_utils as chut
from fit import FitResult
from statistics import compute_error
from plot_functions import plot_function

"""A class used for chiral analyses of matched results, does fitting and
plotting as well
"""

class ChirAna(object):
  """Class to conduct a chiral analysis in mu_l and mu_s for different
  observables

  The observables are, in the most general case, 2d ndarrays, reflecting the
  degrees of freedom in mu_l and mu_s. The dependent observables usually have
  the same layout
  In addition it is possible to fit the data to a given function
  """
  def __init__(self,proc_id=None):
    """ Initialize Chiral Analysis object

    Parameters
    ----------
    proc_id : String identifying the analysis
    """
    self.y_shape = None
    self.x_shape = None
    self.y_data = None
    self.x_data = None
    self.fitres = None
    self.phys_point = None
    self.proc_id = proc_id

  def create_empty(self, lyt_x, lyt_y):
    """Initialize a chiral analysis with some start parameters
    
    At the moment the first index is used to label the lattice spacing, the
    second index is a tuple encoding the number of lattices per spacing. the
    third index counts the number of matched results and the last index is the
    number of bootstrapsamples.
    Parameters
    ----------
    lyt_y : tuple, shape of the y data
    lyt_x : tuple, shape of the x data
    """
    print(lyt_x,len(lyt_y))
    # save layout for later use
    self.y_shape = lyt_y
    self.x_shape = lyt_x

    self.y_data = []
    for a in range(lyt_y[0]):
      if len(lyt_y) == 4:
        tp_y = (lyt_y[1][a],lyt_y[2],lyt_y[3])
      else:
        tp_y = (lyt_y[1][a],lyt_y[2])
      self.y_data.append(np.zeros(tp_y))
    self.x_data = []
    for a in range(lyt_x[0]):
      if len(lyt_x) == 4:
        tp_x = (lyt_x[1][a],lyt_x[2],lyt_x[3])
      else:
        tp_x = (lyt_x[1][a],lyt_x[2])
      self.x_data.append(np.zeros(tp_x))
    self.phys_point = np.zeros(2)

  def add_data(self,data,idx,dim=None):
    """Add x- or y-data at aspecific index

    Parameters
    ----------
    data : ndarray, the data to add
    idx : tuple, the index where to place the data
    dim : string, is it x- or y-data?
    """
    if dim =='x':
      self.x_data[idx[0]][idx[1:]] = data
    if dim =='y':
      self.y_data[idx[0]][idx[1:]] = data

  def add_extern_data(self,filename,idx,ens,dim=None,square=True,read=None):
    if read is 'mpi':
      print("indexto insert extern data:")
      print(dim,idx)
      ext_help = chut.read_extern(filename,(1,2))
      # take any y data dimension, since bootstrap samplesize shold not differ
      plot,data = chut.prepare_mpi(ext_help,ens,self.x_data[0].shape[-1],square=square)
    self.add_data(data,idx,dim=dim)

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
      print("chose data")
      print(data.shape)
    elif dof == 'nboot':
      tp = (slice(None),slice(None),slice(None),index)
      data = _data[tp]
    else:
      raise ValueError("unknown data dimension.")
    return data

  def fit(self,fitfunc,dim,index,xcut=False,plot=True,label=None,datadir=None,read=False):
    """fit a chiral analysis instance to a given fitfunction

    This function uses the data of the ChirAna instance to fit the data to the
    fitfunction. Different degrees of freedom are appliccable, an optional plot
    can be made 

    Parameters
    ----------

    fitfunc : callable, the fitfunction to use for fitting and plotting
    dim : string, which dof to fix (a,mu_l,mu_s,nsamp)
    index : int, index to fix dim
    xcut: float, optional cut on x-axis
    plot: bool, should the fitresults be plotted?
    label: tuple, x-,y-label for plot
    datadir : string, directory for saving data
    read : bool, read in previous fits
    """
    
    # Choose the fit data the dimensions of the data are: (a,mu_l,mu_s,nboot)
    # with lattice spacing a, light and strange quark masses mu_l and mu_s and
    # bootstrapsaFalses
    y_data = self.get_data_fit(dim,index,'y')
    x_data = self.get_data_fit(dim,index,'x')
    print("data used for fit")
    print(x_data.shape,y_data.shape)
    if read:
      if xcut:
        self.fitres = FitResult.read(datadir+self.proc_id+'_xcut_%d.npz'%xcut)
      else:
        self.fitres = FitResult.read(datadir+self.proc_id+'.npz')
    else:
      self.fitres = chut.chiral_fit(x_data,y_data,fitfunc,corrid=self.proc_id,
                                  start=[1.,1.],xcut=xcut)
    if xcut:
      self.fitres.save(datadir+self.proc_id+'_xcut_%d.npz'%xcut)
    else:
      self.fitres.save(datadir+self.proc_id+'.npz')
    args = self.fitres.data[0]
    phys_pt_x = 0.1124
    self.phys_point = np.zeros(2)
    self.phys_point = chut.err_phys_pt(args,phys_pt_x,fitfunc)
    if plot is True:
      label=label
      print(x_data.shape)
      print(y_data.shape)
      self.plot(x_data,y_data,fitfunc,label,xcut=xcut)

  def print_summary(self,dim,index,lat_space,ens_dict,xcut=2):
    """This function should print a summary of the whole chiral analysis,
    preferably in latex format
    """
    # Load data

    y_data = self.get_data_fit(dim,index,'y')
    x_data = self.get_data_fit(dim,index,'x')
    x_plot=np.zeros((x_data.shape[0],4))
    y_plot=np.zeros_like(x_plot)
    x_plot[:,0],x_plot[:,1] = np.asarray(compute_error(x_data,axis=1))
    y_plot[:,0],y_plot[:,1] = np.asarray(compute_error(y_data,axis=1))
    print(x_plot.shape)
    print(lat_space)
    print(ens_dict)
    # print ens_table
    #TODO: Indexing not correct, think about it
    print('\midrule')
    for i,a in enumerate(lat_space):
      if i > 0:
        l = len(ens_dict[lat_space[i-1]])
      if i > 1:
        l = len(ens_dict[lat_space[i-1]])+ len(ens_dict[lat_space[i-2]])
      else:
        l = 0
      for j,e in enumerate(ens_dict[a]):
        # format for 
        chut.print_line_latex(e,x_plot[j*l+i][0:2],y_plot[j*l+i][0:2])
    dof = x_data.shape[0] - self.fitres.data[0].shape[1]
    print("Phsyical point result:")
    if xcut is None:
      xcut = 2.
    print("%10s & $%.1f$ & $%.4f(%1.0f)$ & $%.2f/%d$ & $%.3f $" %
        (self.proc_id, xcut, self.phys_point[0], self.phys_point[1]*1e4,
          self.fitres.chi2[0][0], dof, self.fitres.pval[0][0]))

  def reduction(self):
    """Function to reduce the dimensionality of the data to a two dimensional
    ndarray that can be handled by the plot function

    The x and y data are cast to an 2d ndarray with same shape (apart from bootstrap
    samples)
    """
    # the outgoing data is 2 dimensional
    #x_data_shape = (self.x_shape[1]*np.asarray(self.x_shape[2]),self.x_shape[-1])
    #y_data_shape = (self.y_shape[1]*np.asarray(self.y_shape[2]),self.y_shape[-1])
    # looping over list place data in array
    tmp_x_shape = (self.x_shape[0],self.x_shape[1]*np.asarray(self.x_shape[2]),self.x_shape[-1])
    tmp_y_shape = (self.y_shape[0],self.y_shape[1]*np.asarray(self.y_shape[2]),self.y_shape[-1])
    print("New Shapes:")
    print(tmp_x_shape)
    print(tmp_y_shape)
    tmp = ChirAna("reduction")
    tmp.create_empty(tmp_x_shape,tmp_y_shape)
    for i,d in enumerate(self.x_data):
      new_shape = (d.shape[0]*d.shape[1],d.shape[2])
      tmp.x_data[i] = d.reshape(new_shape)
    for i,d in enumerate(self.y_data):
      new_shape = (d.shape[0]*d.shape[1],d.shape[2])
      tmp.y_data[i] = d.reshape(new_shape)
    if self.x_shape[0] == 3: 
      x_data = np.concatenate((tmp.x_data[0],tmp.x_data[1],tmp.x_data[2]))
      y_data = np.concatenate((tmp.y_data[0],tmp.y_data[1],tmp.y_data[2]))
      return x_data, y_data
    else:
      return None,None

  def plot_plain(self,x_data,y_data,label):
    x_plot=np.zeros((x_data.shape[0],4))
    y_plot=np.zeros_like(x_plot)
    x_plot[:,0],x_plot[:,1] = np.asarray(compute_error(x_data,axis=1))
    y_plot[:,0],y_plot[:,1] = np.asarray(compute_error(y_data,axis=1))
    print(x_plot)
    print(y_plot)
    pfit = PdfPages("./plots2/pdf/%s.pdf" % self.proc_id)

    chut.plot_ensemble(x_plot,y_plot,'s','red','A Ensembles',xid=(0,19))
    #chut.plot_ensemble(x_plot,y_plot,'o','green','D Ensembles',xid=(6,9))
    #chut.plot_ensemble(x_plot,y_plot,'^','blue','B Ensembles',xid=(9,10))
    
    chut.plot_ensemble(x_plot,y_plot,'^','blue','B Ensembles',xid=(19,22))
    chut.plot_ensemble(x_plot,y_plot,'o','green','D Ensembles',xid=(22,23))

    plt.grid(False)
    #plt.xlim(0.,0.5)
    plt.legend(loc='best',numpoints=1)
    plt.ylabel(r'$M_K a_0$')
    plt.xlabel(r'$\mu_l/\mu_s$')
    pfit.savefig()
    pfit.close()
  
  def plot(self,x_data,y_data,fitfunc,label,xcut=False):
    x_plot=np.zeros((x_data.shape[0],4))
    y_plot=np.zeros_like(x_plot)
    x_plot[:,0],x_plot[:,1] = np.asarray(compute_error(x_data,axis=1))
    y_plot[:,0],y_plot[:,1] = np.asarray(compute_error(y_data,axis=1))
    print(x_plot)
    print(y_plot)
    if xcut:
        pfit = PdfPages("./plots2/pdf/%s_xcut_%d.pdf" % (self.proc_id,xcut))
    else:
        pfit = PdfPages("./plots2/pdf/%s.pdf" % self.proc_id)

    x = np.linspace(0,np.amax(x_plot),1000)
    args = self.fitres.data[0]
    plot_function(fitfunc,x,args[:,:,0],label=r'lin. fit',ploterror=True)
    chut.plot_ensemble(x_plot,y_plot,'s','red','A Ensembles',xid=(0,6))
    #chut.plot_ensemble(x_plot,y_plot,'o','green','D Ensembles',xid=(6,9))
    #chut.plot_ensemble(x_plot,y_plot,'^','blue','B Ensembles',xid=(9,10))
    
    chut.plot_ensemble(x_plot,y_plot,'^','blue','B Ensembles',xid=(6,9))
    chut.plot_ensemble(x_plot,y_plot,'o','green','D Ensembles',xid=(9,10))

    if xcut:
      y = fitfunc(args[0,:,0], xcut)
      plt.vlines(xcut, 0.95*y, 1.05*y, colors="k", label="")
      plt.hlines(0.95*y, xcut*0.98, xcut, colors="k", label="")
      plt.hlines(1.05*y, xcut*0.98, xcut, colors="k", label="")
    print("Physical point is:")
    print("%f +/- %f" %(self.phys_point[0],self.phys_point[1]))
    phys_pt_x = 0.1124
    plt.errorbar(phys_pt_x,self.phys_point[0],self.phys_point[1], fmt='d', color='darkorange', label='Physical Point')
    plt.grid(False)
    plt.xlim(0.,1.6)
    plt.legend(loc='best',numpoints=1)
    plt.ylabel(r'$M_K a_0$')
    plt.xlabel(r'$(r_0M_{\pi})^2$')
    pfit.savefig()
    pfit.close()
