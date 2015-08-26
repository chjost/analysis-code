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


def main():

  #----------- Define global parameters, like number of ensembles


  #----------- Read in the data we want and resort it  ------------------------
  rootpath = '/hiskp2/helmes/k-k-scattering/plots/overview/light_qmd/'
  
  plotpath = rootpath+'plots/'
  datapath = rootpath+'data/'
  # The file should contain MK*aKK, Mpi and r0 as columns
  filename_own = datapath + 'ma_mkfk.dat'
  filename_own2 = datapath + 'ma_mk_match.dat'
  filename_nplqcd = datapath + 'ma_mf_npl.dat'
  filename_pacs = datapath + 'ma_mf_pacs.dat'


  # for plotting mk/fk
  scat_dat_nplqcd = np.loadtxt(filename_nplqcd,usecols=(9,10,11,15,16,17))
  scat_dat_pacs = np.loadtxt(filename_pacs,usecols=(2,3,4,5,6,7))
  scat_dat = np.loadtxt(filename_own,usecols=(2,3,4,5,6,7))
  scat_dat2 = np.loadtxt(filename_own2,usecols=(2,3,4,5,6,7,8,9))
  print scat_dat2
  # need mk/fk for plot, how to include statistical and systematical uncertainties?
  mk_fk_npl = scat_dat_nplqcd[:,0:3]
  # Overwrite on purpose
  mk_fk_pacs = np.divide(scat_dat_pacs[:,0], scat_dat_pacs[:,2])
  mk_fk_pacs = np.divide(mk_fk_pacs, math.sqrt(2.))
  mk_akk_npl = ana.sum_error_sym(scat_dat_nplqcd[:,3:])
  mk_akk_pacs = scat_dat_pacs[:,4:6]
  mk_fk_etmc = []
  mk_akk_etmc = []
  mk_fk_etmc = np.divide(scat_dat[:,0], scat_dat[:,2])
  mk_fk_etmc2 = scat_dat2[:,4] 
  dmk_fk_etmc = scat_dat2[:,5]
  mk_akk_etmc = scat_dat2[:,6:]
  # check data
  label_ens = [r'A40',r'A60',r'A80',r'A100']
  print("etmc:\nLat\tbmk\tf_k\tmk/fk\tmk_akk")
  for ens,m,f,mf,a in zip(label_ens,scat_dat[:,0], scat_dat[:,2], mk_fk_etmc, mk_akk_etmc):
    print ens, m, f, mf, a
      # print("\n")

  for ens,m,f,mf,a in zip(label_ens,scat_dat2[:,0], scat_dat2[:,2], mk_fk_etmc2, mk_akk_etmc):
    print ens, m, f, mf, a
      # print("\n")
  print("\nnpl:\nbmk/fk\tmk_akk")
  for m,a in zip(mk_fk_npl,mk_akk_npl):
      print m, a

  print("\npacs:\nbmpi\tmk_akk")
  for m,a in zip(mk_fk_pacs,mk_akk_pacs):
      print m, a


  #----------- Fit NLO-ChiPT to resorted data ---------------------------------
  #----------- Make a nice plot including CHiPT tree level --------------------
  lbl3 = [r'A, $a\mu_s=0.0225$',
          r'A, $a\mu_s=0.02464$',
          r'B, $a\mu_s=0.01861$',
          r'B, $a\mu_s=0.021$','NPLQCD (2007)','PACS-CS (2013)']
  label = [r'$I=1$ $KK$ scattering length, $L = 24^3$',r'$M_K/f_K$',r'$M_K a_{KK}$',lbl3, r'LO $\chi$-PT']
  label_ens = [r'A40',r'A60',r'A80',r'A100']
  ## Open pdf
  pfit = PdfPages(plotpath+'akk_mkfk_unit.pdf')
  tree = lambda p, x : (-1)*x*x/p[0]
  # define colormap
  colors = cm.Dark2(np.linspace(0, 1, 7))
  # A ensembles unitary matching
  #p1 = plt.errorbar(mk_fk_etmc, mk_akk_etmc[:,0], mk_akk_etmc[:,1], fmt='o' + 'b',
  #                  label = r'A, $a\mu_s^\mathrm{unit}$, fk_ipol',color=colors[0])
  #for X, Y,l in zip(mk_fk_etmc,mk_akk_etmc[:,0],label_ens):
  #                      # Annotate the points 5 _points_ above and to the left
  #                      # of the vertex
  #    plt.annotate(l,(X-0.001,Y+0.01))
  mkfk = plt.errorbar(mk_fk_etmc2, mk_akk_etmc[:,0], mk_akk_etmc[:,1], fmt='o' + 'b',
                    label = r'A, $a\mu_s^\mathrm{unit}$',color='blue')
  for X, Y,l in zip(mk_fk_etmc2,mk_akk_etmc[:,0],label_ens):
                        # Annotate the points 5 _points_ above and to the left
                        # of the vertex
      plt.annotate(l,(X-0.001,Y+0.01))
  ## NPLQCD data
  mkfk = plt.errorbar(mk_fk_npl[:,0], mk_akk_npl[:,0], mk_akk_npl[:,1], fmt='x' + 'b',
                    label = lbl3[4],color='green')
  ## PACS data
  mkfk = plt.errorbar(mk_fk_pacs, mk_akk_pacs[:,0], mk_akk_pacs[:,1], fmt='^' + 'b',
                    label = lbl3[5],color='red')
  ## plottin the fit function, set fit range
  lfunc = 2
  ufunc = 5
  x1 = np.linspace(lfunc, ufunc, 1000)
  y1 = []
  for i in x1:
      y1.append(tree([8*math.pi],i))
  y1 = np.asarray(y1)
  p2, = plt.plot(x1, y1, color=colors[6], label = label[4])
  p1 = plt.axvline(x=3.0875, color=colors[6],ls='--',label='phys. point')
  ## adjusting the plot style
  plt.grid(True)
  plt.xlabel(label[1])
  plt.ylabel(label[2])
  plt.title(label[0])
  plt.legend(ncol=1, numpoints=1, loc='best')
  #ana.corr_fct_with_fit(mk_by_fk, mk_akk[:,:,0], mk_akk[:,:,1], tree,
  #    [8*math.pi],
  #    [0,5], label, pfit, xlim=[3,4], ylim=[-0.65,-0.25])
  # set the axis ranges
  plt.xlim([3.0,4.2])
  plt.ylim([-0.65,-0.25])
  # save pdf
  pfit.savefig()
  pfit.close() 

# make this script importable, according to the Google Python Style Guide
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
