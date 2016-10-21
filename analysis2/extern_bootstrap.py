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

# Bootstrapped values of extern data, only given by value and standard deviation
# This functions are to be shared between the classes ChirAna and MatchResult
# For the wrappers to the class see the respective libraries

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

def prepare_a(ens,nsamp):
    """Return a list of bootstrapped r0 values"""
    data_plot = np.zeros(4)
    #dictionary of lattice spacing (arxiv:1403.4504v3)
    r = {'A':[0.0885,0.0036], 'B':[0.0815,0.003], 'D':[0.0619,0.0018]}
    r0_tmp = ana.draw_gauss_distributed(r[ens][0],r[ens][1],(nsamp,),origin=True)
    data_plot[0:2] = ana.compute_error(r0_tmp)
    return data_plot, r0_tmp

def prepare_zp(ens,nsamp,meth=1):
    """Return a list of bootstrapped Z_P values"""
    data_plot = np.zeros(4)
    #dictionary of lattice spacing (arxiv:1403.4504v3)
    if meth == 1:
      raw = {'A':[0.529,0.007], 'B':[0.509,0.004], 'D':[0.516,0.002]}
    if meth == 2:
      raw = {'A':[0.574,0.004], 'B':[0.546,0.002], 'D':[0.545,0.002]}
    raw_tmp = ana.draw_gauss_distributed(raw[ens][0],raw[ens][1],(nsamp,),origin=True)
    data_plot[0:2] = ana.compute_error(raw_tmp)
    return data_plot, raw_tmp

#def prepare_fse_pi(ens,nsamp):
#    """ Return bootstrapped values of correction K^{FSE}_{M^2,L} depending on
#    L
#
#    Parameters
#    ----------
#    ens : string, ensemble name, get lattice extent from it.
#    nsamp : number of samples to return
#
#    Returns
#    -------
#    numpy array of bootstrapped correction factors
#    """
#    # Get lattice extent
#    l = int(ens[4:6])
#    if l == 24:
#      _ksq = 1.014
#    if l == 32:
#      _ksq = 1.002
#    # For future use include error
#    #_ksq_boot = ana.draw_gauss_distributed(_ksq,0,nsamp,origin=True)
#    _ksq_boot = np.full((nsamp),_ksq)
#    return _ksq_boot

def prepare_r0(ens,nsamp,ext=True):
    """Return a list of bootstrapped r0 values
    
    The values are build using either the extraploated value of r_0 and the
    lattice spacings or the values of r_0/a 
    ens: lattice spacing key
    nsamp: int, number of samples to draw
    ext: bool, True (default): use r_0_ext and a
               False: Use r_0/a values
    """
    data_plot = np.zeros(4)
    #Sommer parameter continuum extrapolated (arxiv:1403.4504v3)
    if ext is True:
      _r_ext = ana.draw_gauss_distributed(0.474,0.014,(nsamp,),origin=True)
      _a_plt,_a_tmp = prepare_a(ens,nsamp)
      r0_tmp = np.divide(_r_ext[0],_a_tmp)
    else:
      #dictionary of Sommer parameter (arxiv:1403.4504v3)
      r = {'A':[5.31,0.08], 'B':[5.77,0.06], 'D':[7.60,0.08]}
      r0_tmp = ana.draw_gauss_distributed(r[ens][0],r[ens][1],(nsamp,),origin=True)
    data_plot[0:2] = ana.compute_error(r0_tmp)
    return data_plot, r0_tmp

def prepare_zp(a,nsamp,meth=1):
    """Return a list of bootstrapped zp values
    
    a: lattice spacing key
    nsamp: int, number of samples to draw
    """

    data_plot = np.zeros(4)
    #dictionary of Z_P values (arxiv:1403.4504v3)
    if meth == 1:
      zp = {'A':[0.529,0.007], 'B':[0.509,0.004], 'D':[0.516,0.002]}
    elif meth == 2:
      zp = {'A':[0.574,0.004], 'B':[0.546,0.002], 'D':[0.545,0.002]}
    zp_tmp = ana.draw_gauss_distributed(zp[a][0],zp[a][1],(nsamp,),origin=True)
    data_plot[0:2] = ana.compute_error(zp_tmp)
    return data_plot, zp_tmp

def prepare_mss(ens,nsamp,meth=2):
    """Return a list of bootstrapped M_ss values

    Parameters
    ----------
    ens : char, description of beta value
    nsamp : number of samples of observable
    meth : int, which Z_P to take into account

    Returns
    -------
    data_plot : nparray, value, stdev and asymm. systematics
    mss_tmp : nparray, the bootstraps
    """

    data_plot = np.zeros(4)
    #dictionary of PS mass M_ss (arxiv:1403.4504v3)
    if meth is 1:
      mss = {'A':[0.3258,0.0002], 'B':[0.2896,0.0002], 'D':[0.2162,0.0003]}
    elif meth is 2:
      mss = {'A':[0.3391,0.0002], 'B':[0.2986,0.2220], 'D':[0.2220,0.0003]}
    else:
      print("Method not known, default to 1!")

    mss_tmp = ana.draw_gauss_distributed(mss[ens][0],mss[ens][1],(nsamp,),origin=True)
    data_plot[0:2] = ana.compute_error(mss_tmp)
    return data_plot, mss_tmp

def physical_mpi(x_help,ens,nboot,square=True):
  """calculate physical M_Pi from M_pi data

  The data for M_Pi is pseudobootstrapped with its statistical error,
  divided by the corresponding lattice spacing times \hbar c and squared.
  After that the statistical error is computed
  
  Parameters
  ----------
  x_help : the read in xdata from an external file
  ens : the ensemble name
  nboot : int, number of samples for pseudo bootstrap 
  square : bool, should data be squared?
  """
  # returns
  _data_plot=np.zeros((4))
  # pseudobootstrap lattice data from gaussian distribution
  _data_sing = ana.draw_gauss_distributed(x_help[ens][0], x_help[ens][1],
                                          (nboot,),origin = True)
  _a_plot,_a = prepare_a(ens[0],nboot)
  _hbarc_mpi= 197.37 * _data_sing
  # that is the final data in MeV
  _data = np.divide(_hbarc_mpi,_a)
  # convert to GeV before squaring):
  _data = _data*1.e-3
  if square:
    _data = np.square(_data)
  _data_plot[0:2] = ana.compute_error(_data)
  return _data_plot, _data

def prepare_mpi_fse(x_help1,x_help2,ens,nboot,square=False,physical=False):
    _mpi_fse_plot=np.zeros((4))
    _mpi = ana.draw_gauss_distributed(x_help1[ens][0], x_help1[ens][1], 
                                     (nboot,),origin = True)

    _kfse = ana.draw_gauss_distributed(x_help2[ens][0],x_help2[ens][1],
                                     (nboot,), origin = True)

    _mpi_fse = _mpi/_kfse
    print("_Mpi/_Kfse = %.04f/%0.4f" % (_mpi[0],_kfse[0]))
    if physical:
      _a_plot,_a = prepare_a(ens[0],nboot)
      _mpi_fse = 197.37 * _mpi_fse
      # that is the final data in MeV
      _mpi_fse = np.divide(_mpi_fse,_a)
      # convert to GeV before squaring):
      _mpi_fse = _mpi_fse*1.e-3
    if square:
      _mpi_fse = np.square(_mpi_fse)
    _mpi_fse_plot[0:2] = ana.compute_error(_mpi_fse)
    return _mpi_fse_plot, _mpi_fse

def prepare_fse(x_help,ens,nboot,square=False,had='pi'):
  """Bootstrapsamples for Finite size effects

  Parameters & Returns
  ----------
  cf. other functions
  """
  _fse_plot = np.zeros(4)
  _fse = ana.draw_gauss_distributed(x_help[ens][0], x_help[ens][1],
                                   (nboot,),origin = True)
  if had is 'pi':
      if square:
        _fse = np.square(_fse)
      _fse_plot[0:2] = ana.compute_error(_fse)
  if had is 'k':
      if square is False:
        _fse = np.sqrt(_fse)
      _fse_plot[0:2] = ana.compute_error(_fse)
  return _fse_plot, _fse


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

def prepare_fk(x_help,ens,nboot,square=False):
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
  data_sing = ana.draw_gauss_distributed(x_help[ens][0], x_help[ens][1],
                                        (nboot,),origin=True)
  # final data is fk
  # Have no handle on systematic error here
  data_plot[0:2] = ana.compute_error(data_sing)
  if square is True:
      return np.square(data_plot), np.square(data_sing)
  else:
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

def r0mq_a(amq,nboot,num_mass=3,zp_meth=1):
  """ Build a dictionary of products r0mq for each lattice spacing.

  The dictionary entries get used in the evaluation and matching procedures for
  quark masses. One entry per lattice spacing, 3 masses per lattice spacing

  amq: dictionary of the quark mass parameters,
  nboot: int, number of bootstrapsamples to draw
  num_mass: int, the number of masses per lattice spacing
  zp_meth: int, method 1 or 2 for renormalization zp from quark mass paper
  """
  # The dictionary keys are A, B and D
  keys = ["A","B","D"]
  _dict = {key: None for key in keys}
  for a in keys:
    _r0_plot, _r0_data = prepare_r0(a,nboot)
    _zp_plot, _zp_data = prepare_zp(a,nboot, meth=zp_meth)
    # one entry in dictionary is a (num_mass,nboot) array
    _r0_div_zp = np.divide(_r0_data, _zp_data)
    _dict[a] = np.outer(amq[a], _r0_div_zp)
  return _dict

