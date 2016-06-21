import sys
from scipy import stats
from scipy import interpolate as ip
import time
import matplotlib
matplotlib.use('Agg') # has to be imported before the next lines
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.lines as mlines
import numpy as np
from numpy.polynomial import polynomial as P
from fit_routines import fitting
# Christian's packages
import analysis2 as ana

def lo_chipt(p,x):
  """
  Chiral perturbation formula for mk_akk in dependence of M_pi at leading order
  
  Parameters:
  ----------
  p : array
      Fit parameters (p[0]: B_0*m_s/(4*C), p[1]:(1-m_s*C1/f^2)/(4*C)
  x : scalar
      The value of the pion mass in GeV
  renorm : Renormalization scale for the chiral log in (GeV)^2,
  """
  return p[0]*x+p[1]

def err_phys_pt(pardata,x,func,axis=0):
  """Compute mean and error of an observable at the physical point specified by
  x
  Parameters
  ----------
  pardata : ndarray, the bootstrapsamples of the parameters after fitting
  x : ndarray, the xvalue to evaluate the function
  func : the function, usually a lambda object
  """
  _y = []
  # evaluate all parameters at one point
  #print("data shape of parameters:")
  #print(pardata.shape)
  if pardata.shape > 2:
    for i in range(pardata.shape[0]):
      for j in range(pardata.shape[-1]):
        tmp_par = pardata[i,:,j]
        #print("info on physical point:")
        #print(tmp_par)
        #print(x)
        _y.append(func(tmp_par,x))
  else:
    raise ValueError("Parameters do not have the right shape")
  y=np.asarray(_y)
  print("Calculated y values for error:")
  #for i,j in enumerate(y):
  #  print(j,pardata[i])
  return ana.compute_error(y,axis=axis)


def read_extern(filename,cols):
  """ Read external data with identifiers into a dicitonary
  ATM the tags are written in the first column and the data in the second and
  third.
  TODO: Rewrite that to be more dynamic
  """
  tags = np.loadtxt(filename,dtype='str', usecols=(0,))
  values = np.loadtxt(filename, usecols=(cols))
  # build a dictionary from the arrays
  data_dict = {}
  for i,a in enumerate(tags):
      data_dict[a] = values[i]
  return data_dict

def prepare_data(name, datadir, ens,strange=None,r0=False, square=True, par=0):
  """Function to prepare matched or interpolated data for plotting.

  The data found under the filename in the datadir is read in as a fit result, singularized and
  the error is calculated. Return values are 4 Values for the plot and the
  bootstrapsamples of the data

  Parameters
  ----------
  name : filename prefix
  datadir : name of the data directory,
  ens : ensemblename
  strange : name of the strange quark mass folder
  r0 : boolean for multiplying data 

  Returns
  data_plot : (1,4) nd-array with value, dstat, -dsys, +dsys
  data_sing : nd-array bootstrapsamples
  """
  # usually data is a fitresult
  data_plot=np.zeros(4)
  if strange is not None:
    name = "%s%s/%s/%s_%s.npz" % (datadir,ens,strange, name, ens) 
  else:
    name = "%s%s/%s_%s.npz" % (datadir,ens, name, ens) 
  data_raw = ana.FitResult.read(name)
  data_raw.calc_error()
  data_sing = data_raw.singularize()
  data_plot = data_raw.data_for_plot()
  nboot=data_sing.data[0].shape[0]
  if strange is None:
    if square is False:
      data_fit = np.sqrt(data_sing.data[0][:,par].reshape(nboot))
    else:
      data_fit = data_sing.data[0][:,par].reshape(nboot)
    if r0:
      data_fit= ana.r0_mass(data_fit,ens[0],square)
    data_plot[0:2] = ana.compute_error(data_fit)
    data_plot[2:] = 0
  else:
    if square is False:
      data_fit = data_sing.data[0][:,par].reshape(nboot)
    else:
      data_fit = np.square(data_sing.data[0][:,par]).reshape(nboot)
    if r0:
      data_fit= ana.r0_mass(data_fit,ens[0],square)
    data_plot[0:2] = ana.compute_error(data_fit)
    data_plot[2:] = 0
  print('returned data has shape:')
  print(data_fit[0])

  return data_plot, data_fit

def prepare_r0(ens,nsamp):
    """Return a list of bootstrapped r0 values"""
    data_plot = np.zeros(4)
    #dictionary of Sommer parameter (arxiv:1403.4504v3)
    r = {'A':[5.31,0.08], 'B':[5.77,0.06], 'D':[7.60,0.08]}
    ens_count = ['A','A','A','A','A','A','B','B','B']
    r0_tmp = ana.draw_gauss_distributed(r[ens][0],r[ens][1],(nsamp,))
    r0_tmp[0] = r[ens][0]
    data_plot[0:2] = ana.compute_error(r0_tmp)
    return data_plot, r0_tmp

def prepare_mpi(x_help,ens,nboot,square=True,r0=True):
  """Build (r0*M_Pi)^2 from M_pi data

  The data for M_Pi is pseudobootstrapped with its statistical error,
  multiplied with the corresponding r0 and squared. After that the statistical
  error is computed
  
  Parameters
  ----------
  x_help : the read in xdata from an external file
  """
  # returns
  data_plot=np.zeros((4))
  # pseudobootstrap x-var from gaussian distribution
  data_sing = ana.draw_gauss_distributed(x_help[ens][0],
                                               x_help[ens][1],(nboot,))
  # first entry needs to be original data
  data_sing[0] = x_help[ens][0]
  if r0 is True:
    # final data is (r0*M_Pi)^2
    if square:
      data_sing = ana.r0_mass(data_sing,ens[0])**2
    else:
      data_sing = ana.r0_mass(data_sing,ens[0])
  # Have no handle on systematic error here
  else:
    if square:
      data_sing *= data_sing
  data_plot[0:2] = ana.compute_error(data_sing)
  return data_plot, data_sing

def prepare_fk(x_help,ens,nboot):
  """Build fK from fK data

  The data for fK is pseudobootstrapped with its statistical error.
  After that the statistical error is computed
  
  Parameters
  ----------
  x_help : the read in xdata from an external file
  """
  # returns
  data_plot=np.zeros((4))
  # pseudobootstrap x-var from gaussian distribution
  data_sing = ana.draw_gauss_distributed(x_help[ens][0],
                                               x_help[ens][1],(nboot,))
  # first entry needs to be original data
  data_sing[0] = x_help[ens][0]
  # final data is fk
  # Have no handle on systematic error here
  data_plot[0:2] = ana.compute_error(data_sing)
  return data_plot, data_sing

def prepare_mk(name,datadir,ens,x_help,nboot,amu_s=None,strange=None):
  """Build (M_K/f_K) from M_K^2 data

  The data for f_K is pseudobootstrapped with its statistical error.
  the squareroot of M_K^2 is taken, and it is then divided by the
  bootstrapsamples of f_K. After that the statistical
  error is computed
  
  Parameters
  ----------
  x_help : the read in xdata from an external file
  """
  # returns
  data_plot=np.zeros((4))
  # usually data is a fitresult
  if strange is None:
    name = "%s%s/%s_%s.npz" % (datadir, ens, name, ens) 
  else:
    name = "%s%s/%s/%s_%s.npz" % (datadir, ens,strange, name, ens) 
  data_raw = ana.FitResult.read(name)
  data_raw.calc_error()
  data_sing = data_raw.singularize()
  # pseudobootstrap x-var from gaussian distribution
  if amu_s is not None:
    ens = ens+"_"+str(amu_s)
  data_help = ana.draw_gauss_distributed(x_help[ens][0],
                                               x_help[ens][1],(nboot,))
  # first entry needs to be original data
  data_help[0] = x_help[ens][0]
  if strange is None:
    data_fit = np.divide(np.sqrt(data_sing.data[0][:,0,0]),data_help)
  else:
    data_fit = np.divide(data_sing.data[0][:,1,0],data_help)
  data_plot[0:2] = ana.compute_error(data_fit)
  return data_plot, data_fit

def plot_ensemble(x,y,form,col,label,xid=None,match=False,fitfunc=None):
    if xid is not None:
      d=xid[0]
      u=xid[1]
      #print(u,d)
      if len(form) is 1:
        plt.errorbar(x[d:u,0], y[d:u,0],
                   xerr=[x[d:u,1]+x[d:u,2],x[d:u,1]+x[d:u,3]],
                   yerr=[y[d:u,1]+y[d:u,2],y[d:u,1]+y[d:u,3]],
                   fmt=form, color=col)
        plt.errorbar(x[d:u,0],y[d:u,0],y[d:u,1],x[d:u,1],
                  fmt=form, color=col, label=label[0])
      else:
        chk = 1
        pts = None
        for i,f in enumerate(form):
          # determine new interval (assumes 3 strange quark masses)
          if match:
            _d = d+i
            _u = _d+1
            plt.errorbar(x[_d:_u,0], y[_d:_u,0], 
                       xerr=[x[_d:_u,1]+x[_d:_u,2],x[_d:_u,1]+x[_d:_u,3]],
                       yerr=[y[_d:_u,1]+y[_d:_u,2],y[_d:_u,1]+y[_d:_u,3]],
                       fmt=f, color=col)
            plt.errorbar(x[_d:_u,0],y[_d:_u,0],y[_d:_u,1],x[_d:_u,1],
                      fmt=f, color=col, label=label[i])
            chk = _u
          else:
            _d = d+i*3
            _u = _d+3
            plt.errorbar(x[_d:_u,0], y[_d:_u,0], 
                       xerr=[x[_d:_u,1]+x[_d:_u,2],x[_d:_u,1]+x[_d:_u,3]],
                       yerr=[y[_d:_u,1]+y[_d:_u,2],y[_d:_u,1]+y[_d:_u,3]],
                       fmt=f, color=col)
            plt.errorbar(x[_d:_u,0],y[_d:_u,0],y[_d:_u,1],x[_d:_u,1],
                      fmt=f, color=col, label=label[i])
            chk = _u

        if chk is not u:
          print("symbol coding wrong")

    else:
      pts, = plt.errorbar(x[:,0], y[:,0], 
          xerr=[x[:,1]+x[:,2],x[:,1]+x[:,3]],
          yerr=[y[:,1]+y[:,2],y[:,1]+y[:,3]],
                 fmt=form, color=col, label=label)
      plt.errorbar(x[:,0],y[:,0],y[:,1],y[:,1],
                fmt=form, color=col)

def chiral_fit(X, Y,fitfunc,corrid="",start=None, xcut=None, ncorr=None,debug=0):
    """Fit function to data.
    
    Parameters
    ----------
    X, Y : ndarrays
        The data arrays for X and Y. Assumes ensemble as first axis
        and bootstrap samples as second axis.
    corrid : str
        Identifier for the fit result.
    start : list or tuple or ndarray
        Start value for the fit.
    xcut : float
        A maximal value for the X values. Everything above will not
        be used in the fit.
    debug : int
        The amount of information printed to screen.
    """
    # if no start value given, take an arbitrary value
    if start is None:
        _start = [3.0]
    # make sure start is a tuple, list, or ndarray for leastsq to work
    elif not isinstance(start, (np.ndarray, tuple, list)):
        _start = list(start)
    else:
        _start = start
    print("\nStart parameters are:")
    print(_start)
    # implement a cut on the data if given
    if xcut:
        tmp = X[:,...,0] < xcut
      # Select first bootstrapsample for fit
        _X = X[tmp,...,0:1].T
        _Y = Y[tmp].T
    else:
      # Select first bootstrapsample for fit
        _X = X[:,...,0:1].T
        _Y = Y.T
    if debug > 0:
      print("original fit data used:")
      print(_X[0])
      print(_Y[0])
    # create FitResults
    fitres = ana.FitResult("chiral_fit")
    #shape1 = (_Y.shape[0], len(start),_X.shape[0])
    #shape2 = (_Y.shape[0], _X.shape[0])
    # shape is (nboot)^2 with to parameters and one fit range
    shape1 = (_Y.shape[0]*_X.shape[0], len(_start),1)
    shape2 = (_Y.shape[0]*_X.shape[0],1)
    if ncorr is None:
      fitres.create_empty(shape1, shape2, 1)
    elif isinstance(ncorr, int):
      fitres.create_empty(shape1, shape2,ncorr)
    else:
      raise ValueError("ncorr needs to be integer")

    # fit the data
    dof = _X.shape[-1] - len(_start)
    # fit every bootstrap sample
    timing = []
    for i, x in enumerate(_X):
        timing.append(time.clock())
        tmpres, tmpchi2, tmppval = fitting(fitfunc, x, _Y, _start, debug=debug)
        fitres.append_data((0,i), tmpres, tmpchi2, tmppval)
        #if i % 100:
        #    print("%d of %d finished" % (i+1, _X.shape[0]))
    t1 = np.asarray(timing)
    #print("total fit time %fs" % (t1[-1] - t1[0]))
    #t2 = t1[1:] - t1[:-1]
    #print("time per fit %f +- %fs" % (np.mean(t2), np.std(t2)))
    return fitres

def print_table_header(col_names):
  """ Print the header for a latex table"""
  print(' & '.join(col_names))

def print_line_latex(lat, dx, dy, prec=1e4):
  """Print summary line.

  Parameter
  ---------
  lat : str
      The lattice name
  d : tuple, list
      The data
  """
  if dx.shape[0] == 2:
    print("%9s & NA & $%.4f(%1.0f)(^{+%1.0f}_{-%1.0f})$ &$%.4f(%1.0f)(^{+%1.0f}_{-%1.0f})$ & $%.4f(%1.0f)(^{+%1.0f}_{-%1.0f})$ \\\\" % 
                (lat, dx[0][0], dx[0][1]*prec, dx[0][2]*prec, dx[0][3]*prec,
                  dx[1][0], dx[1][1]*prec, dx[1][2]*prec, dx[1][3]*prec,
                  dy[0], dy[1]*prec, dy[2]*prec, dy[3]*prec))
  else:
    print("%9s & NA & $%.4f(%1.0f)(^{+%1.0f}_{-%1.0f})$  & $%.4f(%1.0f)(^{+%1.0f}_{-%1.0f})$ \\\\" % 
                (lat, dx[0][0], dx[0][1]*prec, dx[0][2]*prec, dx[0][3]*prec,
                  dy[0], dy[1]*prec, dy[2]*prec, dy[3]*prec))
    
