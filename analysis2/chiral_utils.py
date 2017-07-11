#import sys
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

def samp_phys_pt(pardata,x,func,axis=0):
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

def evaluate_phys(x, args, args_w ,func, isdependend):
  """Evaluate a given function at the specified physical point
  
  The parameters of the given function are fitresult data the physical point x
  is evaluated on every sample. The weights are multiplied if the x-value has a
  weight

  Parameters
  ----------
  x : the x value at which to evaluate, could changeto samples in the future
  func : callable, the function to evaluate
  args : the function arguments from a FitResult
  args_w : the corresponding weights
  
  """
  nsam = args.shape[0]
  # loop over the fitrange of parameters
  weight = np.full(nsam, args_w)
  needed = np.zeros((nsam,))
  result = np.full(nsam,np.nan)
  # loop over samples
  print args[0], x[0]
  for b in range(nsam):
    y = func(args[b],x[b])
    if y.shape[0] == 2:
      y = y[0]
    result[b] = y
  yield (0, 0), result, needed, weight

    
def err_phys_pt(pardata,x,func,axis=0,out=None):
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
        print("pardata indices: %d,%d" %(i,j))
        tmp_par = pardata[i,:,j]
        print("info on physical point:")
        print(tmp_par)
        print(x)
        _y.append(func(tmp_par,x))
  else:
    raise ValueError("Parameters do not have the right shape")
  y=np.asarray(_y)
  print("Calculated y values for error:")
  #for i,j in enumerate(y):
  #  print(j,pardata[i])
  if out is None:
    return ana.compute_error(y,axis=axis)
  # return bootstrapsamples
  elif out is 'boot':
    return y


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
        #plt.errorbar(x[d:u,0],y[d:u,0],y[d:u,1],x[d:u,1],
        #          fmt=form, color=col, label=label[0])
        plt.errorbar(x[d:u,0],y[d:u,0],y[d:u,1],
                  fmt=form, color=col, label=label[0])
      else:
        chk = 1
        pts = None
        for i,f in enumerate(form):
          # determine new interval (assumes 3 strange quark masses)
          if match:
            _d = d+i
            _u = _d+1
            # print the data for debug
            #print(x[_d:_u])
            #print(y[_d:_u])
            plt.errorbar(x[_d:_u,0], y[_d:_u,0], 
                       xerr=[x[_d:_u,1]+x[_d:_u,2],x[_d:_u,1]+x[_d:_u,3]],
                       yerr=[y[_d:_u,1]+y[_d:_u,2],y[_d:_u,1]+y[_d:_u,3]],
                       fmt=f, color=col)
            #plt.errorbar(x[_d:_u,0],y[_d:_u,0],y[_d:_u,1],x[_d:_u,1],
            #          fmt=f, color=col, label=label[i])
            plt.errorbar(x[_d:_u,0],y[_d:_u,0],y[_d:_u,1],
                      fmt=f, color=col, label=label[i])
            chk = _u
          else:
            _d = d+i*3
            _u = _d+3
            print("plotting in interval:")
            print(_d,_u)
            # print the data for debug
            #print(x[_d:_u])
            #print(y[_d:_u])
            plt.errorbar(x[_d:_u,0], y[_d:_u,0], 
                       xerr=[x[_d:_u,1]+x[_d:_u,2],x[_d:_u,1]+x[_d:_u,3]],
                       yerr=[y[_d:_u,1]+y[_d:_u,2],y[_d:_u,1]+y[_d:_u,3]],
                       fmt=f, color=col)
            #plt.errorbar(x[_d:_u,0],y[_d:_u,0],y[_d:_u,1],x[_d:_u,1],
            #          fmt=f, color=col, label=label[i])
            plt.errorbar(x[_d:_u,0],y[_d:_u,0],y[_d:_u,1],
                      fmt=f, color=col, label=label[i])
            chk = _u

        if chk is not u:
          print("symbol coding wrong")

    else:
      pts, = plt.errorbar(x[:,0], y[:,0], 
          xerr=[x[:,1]+x[:,2],x[:,1]+x[:,3]],
          yerr=[y[:,1]+y[:,2],y[:,1]+y[:,3]],
                 fmt=form, color=col, label=label)
      #plt.errorbar(x[:,0],y[:,0],y[:,1],x[:,1],
      #          fmt=form, color=col)
      plt.errorbar(x[:,0],y[:,0],y[:,1],
                fmt=form, color=col)

def mutilate_cov(cov):
    """Set elements of covariance matrix to zero explicitly

    Parameters:
    ----------
    cov : the covariance matrix from the original estimate
    
    Returns : 
    --------
    _cov_out : The mutilated covariance matrix
    """
    d = cov.shape[0]
    d2 = d/2
    _cov_out = np.diag((np.asarray(cov.diagonal(0)).reshape(d)))
    # Get upper subdiagonal from cov and place them in _cov_out
    subdiag_up = cov.diagonal(d2,0,1)
    subdiag_up = np.asarray(subdiag_up).reshape(d2)
    _cov_out[d2:d,0:d2] = np.diag(subdiag_up)
    # Get lower subdiagonal and place it likewise
    subdiag_down = cov.diagonal(d2,1,0)
    subdiag_down = np.asarray(subdiag_down).reshape(d2)
    _cov_out[0:d2,d2:d] = np.diag(subdiag_down)
    return _cov_out
        

def chiral_fit(X, Y,fitfunc,corrid="",start=None, xcut=None,
    ncorr=None,correlated=True,mute=None,debug=0):
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
    if len(Y.shape) > 2:
      Y = np.concatenate((Y[:,0],Y[:,1]),axis=0)
      print("Concatenated Y to shape:")
      print(Y.shape)
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
    print("_X used in chiral fit has shape:")
    print(_X.shape)
    print("_Y used in chiral fit has shape:")
    print(_Y.shape)
    for i, x in enumerate(_X):
        timing.append(time.clock())
        tmpres, tmpchi2, tmppval = fitting(fitfunc, x, _Y,
            _start,correlated=correlated,mute=mute, debug=debug)
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

def print_line_latex(lat, dx, dy, dm=None, prec=1e4):
  """Print summary line.

  Parameter
  ---------
  lat : str
      The lattice name
  d : tuple, list
      The data
  """
  if dx.shape[0] == 2:
    if dm is None:
      print("%9s & NA & $%.4f(%1.0f)(^{+%1.0f}_{-%1.0f})$ &$%.4f(%1.0f)(^{+%1.0f}_{-%1.0f})$ & $%.4f(%1.0f)(^{+%1.0f}_{-%1.0f})$ \\\\" % 
                  (lat, dx[0][0], dx[0][1]*prec, dx[0][2]*prec, dx[0][3]*prec,
                    dx[1][0], dx[1][1]*prec, dx[1][2]*prec, dx[1][3]*prec,
                    dy[0], dy[1]*prec, dy[2]*prec, dy[3]*prec))
    else:
      print("%9s & $%.4f(%1.0f)$ & $%.4f(%1.0f)(^{+%1.0f}_{-%1.0f})$ &$%.4f(%1.0f)(^{+%1.0f}_{-%1.0f})$ & $%.4f(%1.0f)(^{+%1.0f}_{-%1.0f})$ \\\\" % 
                  (lat, dm[0],dm[1]*prec,dx[0][0], dx[0][1]*prec, dx[0][2]*prec, dx[0][3]*prec,
                    dx[1][0], dx[1][1]*prec, dx[1][2]*prec, dx[1][3]*prec,
                    dy[0], dy[1]*prec, dy[2]*prec, dy[3]*prec))

  else:
    print("%9s & NA & $%.4f(%1.0f)(^{+%1.0f}_{-%1.0f})$  & $%.4f(%1.0f)(^{+%1.0f}_{-%1.0f})$ \\\\" % 
                (lat, dx[0][0], dx[0][1]*prec, dx[0][2]*prec, dx[0][3]*prec,
                  dy[0], dy[1]*prec, dy[2]*prec, dy[3]*prec))
    
def amu_q_from_mq_ren(a,zp,nboot,mq_in=(99.6,4.3),mq_guess=None):
  """Calculate the bare strange quark mass per lattice spacing
  
  The lattice data for r0 and the renormalization constant are converted to
  bootstrapsamples of amu_q

  Parameters:
  -----------
  a : tuple,Lattice calculation of Sommer parameter
  zp : tuple, the renormalization constant
  mq_guess : 1darray, bootstrapsamples of guessed quark mass
  """
  #_r0 = ana.draw_gauss_distributed(0.474,0.014,(nboot,),origin=True)
  if mq_guess is None:
    _ms = ana.draw_gauss_distributed(mq_in[0],mq_in[1],(nboot,),origin=True)
  else:
    if mq_guess.shape[0] != nboot:
      raise ValueError("samplesize of guessed quark mass and nboot do not match")
    else:
      _ms = mq_guess
  # Latticer input
  #_r0_lat = ana.draw_gauss_distributed(r0_in[0],r0_in[1],(nboot,),origin=True)
  _dummy, _a = ana.prepare_a(a,nboot) 
  _zp_ms_bar = ana.draw_gauss_distributed(zp[0],zp[1],(nboot,),origin=True)

  #_bare_mass = _r0*_ms*_zp_ms_bar/(197.37*_r0_lat)
  _bare_mass = _a*_ms*_zp_ms_bar/197.37
  return _bare_mass

def r0mq_from_amuq(amuq,a,nboot):
  """Calculate r0mq from amuq samples
  
  The lattice data for r0 and the renormalization constant are converted to
  bootstrapsamples of amu_q

  Parameters:
  -----------
  r0 : tuple,Lattice calculation of Sommer parameter
  zp : tuple, the renormalization constant
  mq_guess : 1darray, bootstrapsamples of guessed quark mass
  """
  # Latticer input
  #_r0_lat = ana.draw_gauss_distributed(r0_in[0],r0_in[1],(nboot,),origin=True)
  _dummy, _r0_lat = ana.prepare_r0(a,nboot) 
  _dummy2, _zp_ms_bar = ana.prepare_zp(a,nboot)

  _r0mq = _r0_lat*amuq/_zp_ms_bar
  return _r0mq

def amuq_from_r0mq(r0mq,a,nboot):
  """Calculate amuq from r0mq samples
  
  The lattice data for r0 and the renormalization constant are converted to
  bootstrapsamples of amu_q

  Parameters:
  -----------
  r0 : tuple,Lattice calculation of Sommer parameter
  zp : tuple, the renormalization constant
  mq_guess : 1darray, bootstrapsamples of guessed quark mass
  """
  # Latticer input
  #_r0_lat = ana.draw_gauss_distributed(r0_in[0],r0_in[1],(nboot,),origin=True)
  _dummy, _r0_lat = ana.prepare_r0(a,nboot) 
  _dummy2, _zp_ms_bar = ana.prepare_zp(a,nboot)
  
  # if r0mq has nboot samples error already has been taken into account
  if r0mq.shape[0] == nboot:
    _amuq = r0mq*_zp_ms_bar[0]/_r0_lat[0]

  else:
    _amuq = r0mq*_zp_ms_bar/_r0_lat
 
  return _amuq

def amu_q_from_mq_pdg(r0_in,zp,nboot):
  """Calculate the bare strange quark mass per lattice spacing
  
  The lattice data for r0 and the renormalization constant are converted to
  bootstrapsamples of amu_q

  Parameters:
  -----------
  r0 : tuple,Lattice calculation of Sommer parameter
  zp : tuple, the renormalization constant
  """
  _r0 = ana.draw_gauss_distributed(0.474,0.014,(nboot,),origin=True)
  _ms = ana.draw_gauss_distributed(95.,5,(nboot,),origin=True)
  # Latticer input
  _r0_lat = ana.draw_gauss_distributed(r0_in[0],r0_in[1],(nboot,),origin=True)
  _zp_ms_bar = ana.draw_gauss_distributed(zp[0],zp[1],(nboot,),origin=True)

  _bare_mass = _r0*_ms*_zp_ms_bar/(197.37*_r0_lat)
  return _bare_mass

def mk_phys_paper(nboot):
  """Returns bootstrapsamples of M_K^2 in MeV"""
  # Value for M_K from arxiv:1403.4504v3
  _mk = ana.draw_gauss_distributed(494.2,0.4,(nboot,),origin=True)
  _res = _mk
  #print(compute_error(res))
  return _res

def mk_phys_from_r0_mk_sq_lat(r0mk_in,nboot):
  """Returns bootstrpasamples of M_K in MeV calculated from lattice values of
  (r_0 M_K)^2 in continuum"""
  print("This is the input data:")
  print(r0mk_in)
  _r0 = ana.draw_gauss_distributed(0.474,0.014,(nboot,),origin=True)
  _res = (np.sqrt(r0mk_in)*197.37/(_r0))
  print(ana.compute_error(_res))
  return _res

def mk_to_lat(ens,nboot):
  # lattice identifier from ensemblename
  a = ens[0]
  # dicitionary of lattice spacings
  """Convert physical kaon mass to lattice units using a"""
  a_dict = {'A':(0.0885,0.0036),'B':(0.0815,0.003),'D':(0.0619,0.0018)}
  _a = ana.draw_gauss_distributed(a_dict[a][0],a_dict[a][1],(nboot,),origin=True)
  # Value for M_K from arxiv:1403.4504v3
  _mk = ana.draw_gauss_distributed(494.2,0.4,(nboot,),origin=True)
  mk_lat = _a*_mk/197.37
  return mk_lat

def r0_mk_sq_phys(nboot):
  """Returns bootstrapped values of (r_0 M_K)^2 in lattice units"""
  # Value for M_K from arxiv:1403.4504v3
  _mk = ana.draw_gauss_distributed(494.2,0.4,(nboot,),origin=True)
  _r0 = ana.draw_gauss_distributed(0.474,0.014,(nboot,),origin=True)
  # Value for M_K from PDG
  #_mk = ana.draw_gauss_distributed(493.677,0.016,(nboot,),origin=True)
  _res = (_r0*_mk/197.37)**2
  #print(compute_error(res))
  return _res

# Idea: Move that to cont_data class
def mk_mpi_diff_phys(a, nboot, cont, ext):
  """ Function to calculate a^2(M_K^2-0.5M_pi^2) from continuum data"""
  
  _r0 = cont.get('r0')
  _r0a = ext.get(a,'r0')
  _mk = cont.get('mk')
  _mpi = cont.get('mpi_0')
  
  return _r0**2*(_mk**2-0.5*_mpi**2)/(197.37**2*_r0a**2)

# Idea: Move that to cont_data class
def r0mk_mpi_diff_phys(cont):
  """ Function to calculate r_0^2(M_K^2-0.5M_pi^2) from continuum data"""
  
  _r0 = cont.get('r0')
  _mk = cont.get('mk')
  _mpi = cont.get('mpi_0')
  
  return _r0**2*(_mk**2-0.5*_mpi**2)/(197.37**2)

def beta_mk_mpi_diff(nboot,beta):
  
  r0 = {'A':(5.31,0.08),'B':(5.77,0.06),'D':(7.6,0.08)}
  space = {'A':(0.0885,0.0036),'B':(0.0815,0.003),'D':(0.0619,0.0018)}
  _r0_a = ana.draw_gauss_distributed(r0[beta][0],r0[beta][1],
                                    (nboot,),origin=True)
  _a = ana.draw_gauss_distributed(space[beta][0],space[beta][1],
                                  (nboot,),origin=True)
  _r0 = _r0_a * _a
  print("bootstrapped error of r0_a *a:")
  print(ana.compute_error(_r0))
  _mk = ana.draw_gauss_distributed(493.677,0.016,(nboot,),origin=True)
  _mpi = ana.draw_gauss_distributed(134.9766,0.0006,(nboot,),origin=True)
  _diff = _r0**2*(_mk**2-0.5*_mpi**2)/(197.37**2)
  return _diff

def r0_ms(data,r0_in,zp):
  """Calculate r0ms from lattice data"""
  nboot = data.shape[0]
  # Latticer input
  _r0 = ana.draw_gauss_distributed(0.474,0.014,(nboot,),origin=True)
  _r0_lat = ana.draw_gauss_distributed(r0_in[0],r0_in[1],(nboot,),origin=True)
  _zp_ms_bar = ana.draw_gauss_distributed(zp[0],zp[1],(nboot,),origin=True)
  hc = 197.37
  _ms_phys = data*_r0_lat[0]*hc/(_zp_ms_bar[0]*_r0[0])

  return _ms_phys

def r0ms_phys(ms,nboot):
  _r0 = ana.draw_gauss_distributed(0.474,0.014,(nboot,),origin=True)
  hcbar = 197.37
  return _r0*ms/hcbar

def ml_ren_from_amu_q(dict_a,a_in,zp_in,nboot,dict_mul):
  if(len(dict_mul)==len(dict_a)):
    _zp_ms_bar = ana.draw_gauss_distributed(zp_in[0],zp_in[1],(nboot,),origin=True)
    _a_lat = ana.draw_gauss_distributed(a_in[0],a_in[1],(nboot,),origin=True)
    _dict = {}
    for i,e in enumerate(dict_a):
      _dict[e] = dict_mul[i]*197.37/(_a_lat*_zp_ms_bar)
    return _dict
  else:
    raise ValueError("Number of ensembles and mu_l values differ")

def r0ml_ren_from_amu_q(dict_r0,r0_in,zp_in,nboot,dict_mul):
  if(len(dict_mul)==len(dict_r0)):
    _zp_ms_bar = ana.draw_gauss_distributed(zp_in[0],zp_in[1],(nboot,),origin=True)
    _r0_lat = ana.draw_gauss_distributed(r0_in[0],r0_in[1],(nboot,),origin=True)
    _dict = {}
    for i,e in enumerate(dict_r0):
      _dict[e] = dict_mul[i]*_r0_lat/(_zp_ms_bar)
    return _dict
  else:
    raise ValueError("Number of ensembles and mu_l values differ")

def calc_r0ms(para, r0_phys, mk_phys, ml_phys):
  """Calculate the strange quark mass from the fit parameters

  Parameters
  ----------
  para: fitresult, parameters from fit 
  mk_phys: fitresult, physical kaon mass
  ml_phys: float, chirally extrapolated light quark mass
  """
  _hbarc = 197.37
  _p = para.data[0]
  _num = (r0_phys*mk_phys/_hbarc)**2
  _den = (_p[:,0,0]*(1 + _p[:,1,0]*r0_phys*ml_phys/_hbarc))
  _sub = r0_phys*ml_phys/_hbarc
  #print("result components (num, den, sub): %f, %f, %f" %(_num,_den[0],_sub))
  _r0ms = _num/_den - _sub 
  print("samples of _r0ms: %r" % _r0ms)
  return _r0ms

def compute_bare_mu_s(r0,ml,mk,mul,args,disc_eff=False):
  """ Compute the bare strange quark mass from the fitparameters, the continuum
  values of ml,r0 and mk

  The function to calculate reads:
  amu_s = (pz (r_0*M_K)^2 )/(pr * p0 * (1 + p1*(r_0*m_l) + p2/pr**2)) - amu_l
  
  pr and pz depend on the lattice spacing, p0-p2 are the global fit parameters.
  
  Parameters
  ----------
  r0,ml,mk: 1d array, pseudosamples for continuum values of the Sommer
            parameter, the light quark mass and the Kaon mass
  mul: float, value of the bare light quark mass of ensemble under consideration
  args: ndarray, fitparameters of the global fit to the squared kaon mass
  
  Returns
  -------
  _mus: 1d array of samples of the bare strange quark mass
  """
  #print("arguments to mu_s function:")
  #print("Physical values of r0, ml and mk: %.3f, %.3f, %.3f"
  #    %(r0[0],ml[0],mk[0]))
  #print("Arguments from fit:")
  #print(args[0])
  #print("Bare light quark mass: %4f" %mul)
  _hbarc = 197.37
  _mul = args[:,1]/args[:,0]*(r0*ml)/_hbarc
  _nom = args[:,1] * (r0*mk/_hbarc)**2
  _par = 1 + args[:,3] * (r0 * ml)/_hbarc

  # Take discretisation effects into account 
  if disc_eff is True:
      _par += args[:,4]/args[:,0]**2

  _denom = (args[:,0] * args[:,2]) * _par
  #print("Parts to construct bare mus:")
  #print(_nom[0],_par[0],_denom[0])
  _frac = np.divide(_nom,_denom)
  #_mus = np.subtract(_frac,mul)
  _mus = np.subtract(_frac,_mul)
  return _mus

################################################################################
################# Scratch region for trying out functions ######################
################################################################################

def concatenate_data(lst,par=0):
    _b=[]                                               
    for i, d in enumerate(lst):
        if par is not 0:
            _n = np.zeros((d.shape[0]*d.shape[1],d.shape[2],d.shape[3]))
        else:
            _n = np.zeros((d.shape[0]*d.shape[1],d.shape[3]))
        for i in range(d.shape[0]):
            for j in range(d.shape[1]):
                # Only one parameter at the moment
                _n[i*d.shape[1]+j] = d[i,j,par]
        _b.append(_n)
    # Take all three lattice spacings
    _c = np.concatenate((_b[0],_b[1],_b[2]))
    return _c
