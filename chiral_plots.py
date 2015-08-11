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

# Christian's packages
import analysis as ana

def sum_error_sym(meas):
  """gets a n x 3 numpy array holding a value, a statistical and a systematic
  uncertainty to be added in quadrature
  returns a n x 2 array holding the value and the combined uncertainty for each
  row
  """
  print meas.shape[0]
  val_err = np.zeros((meas.shape[0],2))
  val_err[:,0] = meas[:,0]
  val_err[:,1] = np.sqrt(np.add(np.square(meas[:,1]),np.square(meas[:,2])))
  return val_err

def sum_error_asym(meas):
  """gets a n x 4 numpy array holding a value, a statistical and two systematic
  uncertainties to be added in quadrature
  returns a n x 2 array holding the value and the combined uncertainty for each
  row
  """
  print meas.shape[0]
  val_err = np.zeros((meas.shape[0],2))
  val_err[:,0] = meas[:,0]
  sys_err_sum =np.add( np.square(meas[:,2]), np.square(meas[:,3]) )
  val_err[:,1] = np.sqrt(np.add(np.square(meas[:,1]),sys_err_sum))
  return val_err

def main():

  #----------- Define global parameters, like number of ensembles


  #----------- Read in the data we want and resort it  ------------------------
  datapath = '/hiskp2/helmes/k-k-scattering/plots/overview/light_qmd/'
  # The file should contain MK*aKK, Mpi and r0 as columns
  filename_own = datapath + 'ma_mpi.dat'
  filename_nplqcd = datapath + 'ma_mpi_npl.dat'
  filename_pacs = datapath + 'ma_mpi_pacs.dat'
  # TODO: Replace txtfile by binary format in future
  scat_dat = np.loadtxt(filename_own,usecols=(1,2,3,4,5,10,11,12,13))
  # split arrays
  scat_dat_lst = np.split(scat_dat,[4,8,9])
  #scat_dat_lst.pop()
  scat_dat_nplqcd = np.loadtxt(filename_nplqcd,usecols=(1,2,3,7,8,9,13,14,15))
  scat_dat_pacs = np.loadtxt(filename_pacs,usecols=(0,1,4,5,6,7))
  # need mk/fk for plot, how to include statistical and systematical uncertainties?
  #mk_fk_npl = scat_dat_nplqcd[:,3:6] 
  #mk_fk_pacs = np.divide(scat_dat_pacs[:,0], scat_dat_pacs[:,2])
  #mk_akk_npl = scat_dat_nplqcd[:,6:9] 
  # need (mpi*r0)^2 for plot, how to include statistical and systematical uncertainties?
  mpi_r0_npl = np.multiply(scat_dat_nplqcd[:,0:3],(0.5/0.125))
  mpi_r0_pacs = np.multiply(scat_dat_pacs[:,0:2],0.5/(0.19733))
  mk_akk_npl = sum_error_sym(scat_dat_nplqcd[:,6:9])
  mk_akk_pacs = scat_dat_pacs[:,4:6]

  # check data
  print("etmc:\nbmpi\tr_0\tmk_akk")
  for i in range(len(scat_dat_lst)):
      for m,r,a in zip(scat_dat_lst[i][:,0:3],scat_dat_lst[i][:,3:5],scat_dat_lst[i][:,5:9]):
          print m,r,a
      print("\n")

  print("\nnpl:\nbmpi\tmk_akk")
  for m,a in zip(mpi_r0_npl,mk_akk_npl):
      print m, a

  print("\npacs:\nbmpi\tmk_akk")
  for m,a in zip(mpi_r0_pacs,mk_akk_pacs):
      print m, a

  # Concatenate everything
  #TODO: Automate concatenation solve with lists for case of different data
  # lengths
  #mpi_r0_all = np. 
  #mk_by_fk_all = np.concatenate((mk_by_fk, scat_dat_nplqcd[:,0].reshape(1,5), mk_fk_pacs.reshape(1,5)))
  #mk_akk_all = np.concatenate((mk_akk[:,:,0:2],scat_dat_nplqcd[:,3:5].reshape((1,5,2)), scat_dat_pacs[:,4:].reshape((1,5,2))))
  #mk_by_fk_all = np.concatenate((mk_by_fk))
  #mk_akk_all = np.concatenate((mk_akk[:,:,0:2],scat_dat_nplqcd[:,3:5].reshape((1,5,2))))
  #print mk_by_fk_all
  #print mk_akk_all

  #----------- Fit NLO-ChiPT to resorted data ---------------------------------
  #----------- Make a nice plot including CHiPT tree level --------------------
  lbl3 = [r'A, $a\mu_s=0.0225$',
          r'A, $a\mu_s=0.02464$',
          r'B, $a\mu_s=0.01861$',
          r'B, $a\mu_s=0.021$','NPLQCD (2007)','PACS (2013)']
  label = [r'$I=1$ $KK$ scattering length',r'$(r_0M_\pi)^2$',r'$M_K a_{KK}$',lbl3, r'LO $\chi$-PT']
  ## Open pdf
  pfit = PdfPages(datapath+'akk_mpi_sq.pdf')
  #tree = lambda p, x : (-1)*x*x/p[0]
  # define colormap
  colors = cm.Set1(np.linspace(0, 1, 6))
  # A ensembles strange mass 0.0225
  # define x and y data
  mpi_r0_sq_etmc = np.square(np.multiply(scat_dat_lst[0][:,0],scat_dat_lst[0][:,3]))
  mk_akk_tmp = sum_error_asym(scat_dat_lst[0][:,5:9])
  #print mpi_r0_sq_etmc
  #print mk_akk_tmp
  p1 = plt.errorbar(mpi_r0_sq_etmc, mk_akk_tmp[:,0], mk_akk_tmp[:,1], fmt='o' + 'b',
                    label = lbl3[0],color=colors[0])
  # A ensembles strange mass 0.02464
  # define x and y data
  mpi_r0_sq_etmc = np.square(np.multiply(scat_dat_lst[1][:,0],scat_dat_lst[1][:,3]))
  mk_akk_tmp = sum_error_asym(scat_dat_lst[1][:,5:9])
  #print mpi_r0_sq_etmc
  #print mk_akk_tmp
  p1 = plt.errorbar(mpi_r0_sq_etmc, mk_akk_tmp[:,0], mk_akk_tmp[:,1], fmt='o' + 'b',
                    label = lbl3[1],color=colors[1])
  # B ensembles strange mass 0.01861
  # define x and y data
  mpi_r0_sq_etmc = np.square(np.multiply(scat_dat_lst[2][:,0],scat_dat_lst[2][:,3]))
  mk_akk_tmp = sum_error_asym(scat_dat_lst[2][:,5:9])
  #print mpi_r0_sq_etmc
  #print mk_akk_tmp
  p1 = plt.errorbar(mpi_r0_sq_etmc, mk_akk_tmp[:,0], mk_akk_tmp[:,1], fmt='o' + 'b',
                    label = lbl3[2],color=colors[2])
  # B ensembles strange mass 0.021
  # define x and y data
  mpi_r0_sq_etmc = np.square(np.multiply(scat_dat_lst[3][:,0],scat_dat_lst[3][:,3]))
  mk_akk_tmp = sum_error_asym(scat_dat_lst[3][:,5:9])
  #print mpi_r0_sq_etmc
  #print mk_akk_tmp
  p1 = plt.errorbar(mpi_r0_sq_etmc, mk_akk_tmp[:,0], mk_akk_tmp[:,1], fmt='o' + 'b',
                    label = lbl3[3],color=colors[3])
  ### NPLQCD data
  mpi_sq_npl = np.square(mpi_r0_npl[:,0])
  ##mk_sq_npl = np.multiply(scat_dat_nplqcd[:,0], scat_dat_nplqcd[:,0])
  p1 = plt.errorbar(mpi_sq_npl, mk_akk_npl[:,0], mk_akk_npl[:,1], fmt='o' + 'b',
                    label = lbl3[4],color=colors[4])
  ### PACS data
  mpi_sq_pacs = np.square(mpi_r0_pacs[:,0])
  ##mk_sq_npl = np.multiply(scat_dat_nplqcd[:,0], scat_dat_nplqcd[:,0])
  p1 = plt.errorbar(mpi_sq_pacs, mk_akk_pacs[:,0], mk_akk_pacs[:,1], fmt='o' + 'b',
                    label = lbl3[5],color=colors[5])
  ##p1 = plt.errorbar(X, Y, dY, fmt='o' + 'b',
  ##                  label = label[3][_dl],color=next(colors))
  ## plottin the fit function, set fit range
  #lfunc = 3
  #ufunc = 4
  #x1 = np.linspace(lfunc, ufunc, 1000)
  #y1 = []
  #for i in x1:
  #    y1.append(tree([8*math.pi],i))
  #y1 = np.asarray(y1)
  ##p2, = plt.plot(x1, y1, color=colors[3], label = label[4])
  ## adjusting the plot style
  plt.grid(True)
  plt.xlabel(label[1])
  plt.ylabel(label[2])
  plt.title(label[0])
  plt.legend(ncol=2, numpoints=1, loc='best')
  #ana.corr_fct_with_fit(mk_by_fk, mk_akk[:,:,0], mk_akk[:,:,1], tree,
  #    [8*math.pi],
  #    [0,5], label, pfit, xlim=[3,4], ylim=[-0.65,-0.25])
  # set the axis ranges
  #plt.xlim([0,0.1])
  #plt.ylim([-0.65,-0.25])
  # save pdf
  pfit.savefig()
  pfit.close() 

# make this script importable, according to the Google Python Style Guide
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
