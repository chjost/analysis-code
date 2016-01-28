#!/hadron/knippsch/Enthought/Canopy_64bit/User/bin/python
################################################################################
#
# Author: Christopher Helmes (helmes@hiskp.uni-bonn.de)
# Date:   Februar 2015
#
# Copyright (C) 2015 Christopher Helmes
# 
# This program is free software: you can redistribute it and/or modify it under 
# the terms of the GNU General Public License as published by the Free Software 
# Foundation, either version 3 of the License, or (at your option) any later 
# version.
# 
# This program is distributed in the hope that it will be useful, but WITHOUT 
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS 
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with tmLQCD. If not, see <http://www.gnu.org/licenses/>.
#
################################################################################
#
# Function: Plots and fits ChiPT formulae to extracted scattering data 
#
# For informations on input parameters see the description of the function.
#
################################################################################

# system imports
import os.path as osp
from scipy import stats
import numpy as np
from numpy.polynomial import polynomial as P
import math
import matplotlib
matplotlib.use('Agg') # has to be imported before the next lines
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages

def sum_error_sym(meas):
  """gets a n _mpi 3 numpy array holding a value, a statistical and a systematic
  uncertainty to be added in quadrature
  returns a n _mpi 2 array holding the value and the combined uncertainty for each
  row
  """
  print meas.shape[0]
  val_err = np.zeros((meas.shape[0],2))
  val_err[:,0] = meas[:,0]
  val_err[:,1] = np.sqrt(np.add(np.square(meas[:,1]),np.square(meas[:,2])))
  return val_err

def err_prop_gauss(_a,_b,oper='div'):
  """ Evaluates gaussian propagated error without correlation for different
      operations
      Args:
          a,b: numpy arrays of the values of interest
          da,db: numpy arrays of the corresponding errors
          oper: flag to determine derived value (default: a/b)
      Returns:
          err_der: a numpy array of the derived errors
  """
  a,b = _a[:,0],_b[:,0]
  da,db = _a[:,1], _b[:,1]
  # observables connected by division
  if oper == 'div':
    der = np.absolute(np.divide(a,b))
    sq_1 = np.square(np.divide(da,a))
    sq_2 = np.square(np.divide(db,b))
    rad = np.add(sq_1,sq_2)
    err_der = np.multiply(der,np.sqrt(rad))
  # observables connected by multiplication
  if oper == 'mult':
    sq_1 = np.multiply(np.square(da),np.square(a))
    sq_2 = np.multiply(np.square(db),np.square(b))
    rad = np.add(sq_1,sq_2)
    err_der = np.sqrt(rad)
  else:
    print("Not able to determine error")
    err_der = 0
  print("Calculated uncorrelated propagated error")
  return err_der

def val_stack(a,b,oper='div'):
  if oper == 'div':
    val_der = np.divide(a[:,0],b[:,0])
  if oper == 'mult':
    val_der = np.multiply(a[:,0],b[:,0])
  else:
    print("Derivation not defined")
  return val_der

def stack_obs(a, b, oper = 'div'):
  val = val_stack(a,b,oper) 
  err = err_prop_gauss(a,b,oper)
  return np.column_stack((val,err))

def err_plot(_X,_Y,_lbl,_cl):
  """ Plots X and Y values with Y error in an open plot and assigns the right
  labels
  """
  plt.errorbar(_X[:,0], _Y[:,0],_X[:,1], _Y[:,1],fmt='o'+_cl,label=_lbl[3])
  plt.grid(True)
  plt.xlabel(_lbl[1])
  plt.ylabel(_lbl[2])
  plt.title(_lbl[0])
  plt.legend(loc='best')


def main():

  #----------- Define global parameters, like number of ensembles

  # General output layout
  np.set_printoptions(precision=5,suppress=True)
  #----------- Read in the data we want and resort it  ------------------------
  rootpath = '/hiskp2/helmes/k-k-scattering/plots/overview/light_qmd/'
  
  plotpath = rootpath+'plots/'
  datapath = rootpath+'data/'

  # Load NPLQCD-data
  file_npl = "ma_mf_npl.dat"
  npl_k_obs = np.loadtxt(datapath+file_npl)
  #mpi_fk = npl_k_obs[:,9:11]
  #mk_fk = npl_k_obs[:,12:14]
  mk_npl = npl_k_obs[:,6:9]
  mpi_npl = npl_k_obs[:,1:4]
  r1_npl = npl_k_obs[:,4:6]
  r0_r1 = np.resize(np.array((1.474,0.007,0.018)),(5,3))

  r0_npl = stack_obs(r0_r1[:,0:2],r1_npl[:,0:2],'mult')
  print("r0_npl values:")
  print r0_npl
  print("M_k from npl:")
  print mk_npl
  x_npl = np.square( stack_obs(r0_npl,mpi_npl[:,0:2],'mult') )
  y_npl = np.square( stack_obs(r0_npl,mk_npl[:,0:2],'mult') )
  
  print x_npl,y_npl

  # Load ETMC-data
  file_etm = "ma_mk_match.dat"
  etm_k_obs = np.loadtxt(datapath+file_etm)
  mpi_etm = etm_k_obs[:,11:14]
  mk_etm = etm_k_obs[:,3:5]
  r0_etm = np.resize(np.array((5.231,0.038)),(4,2))
  print("r0_etm values:")
  print r0_etm
  print("mk etm values:")
  print mk_etm
  x_etm = np.square( stack_obs(r0_etm,mpi_etm[:,0:2],'mult') )
  y_etm = np.square( stack_obs(r0_etm,mk_etm[:,0:2],'mult') )

  #fk_etm = etm_k_obs[:,6:8]
  #mpi_fk_etm = np.column_stack((np.divide(mpi_etm[:,0],fk_etm[:,0]),
  #                              err_prop_gauss(mpi_etm,fk_etm)))
  #mk_fk_etm = np.column_stack((np.divide(mk_etm[:,0],fk_etm[:,0]),
  #                              err_prop_gauss(mk_etm,fk_etm)))

  # plot stuff
  plot = PdfPages(plotpath+"r0_mk_mpi.pdf")
  lbl = [r'light quark dependence of $M_K$',r'$(r_0M_{\pi})^2$',r'$(r_0M_K)^2$',r' NPLQCD']
  err_plot(x_npl, y_npl, lbl,'r')
  lbl[3] = r'ETMC'
  err_plot(x_etm, y_etm,lbl,'b')
  plot.savefig()
  plot.close()

# make this script importable, according to the Google Python Style Guide
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
