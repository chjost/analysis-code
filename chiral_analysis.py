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
  datapath = '/hiskp2/helmes/k-k-scattering/plots/overview/light_qmd/'
  # The file should contain MK*aKK, MK and fK as columns
  filename_own = datapath + 'ma_mf.dat'
  filename_nplqcd = datapath + 'ma_mf_npl.dat'
  filename_pacs = datapath + 'ma_mf_pacs.dat'
  # TODO: Replace txtfile by binary format in future
  scat_dat = np.loadtxt(filename_own,usecols=(1,2,3,4,5,6,7,8,9,14,15,16,17))
  # split arrays
  scat_dat_lst = np.split(scat_dat,[4,8,9,10,12,16])
  scat_dat_lst.pop()
  scat_dat_nplqcd = np.loadtxt(filename_nplqcd,usecols=(4,5,6,7,8,9,13,14,15))
  scat_dat_pacs = np.loadtxt(filename_pacs,usecols=(2,3,4,5,6,7))
  # print scat_dat[0,:,0:3]
  # need mk/fk for plot, how to include statistical and systematical uncertainties?
  mk_fk_npl = scat_dat_nplqcd[:,3:6] 
  mk_fk_pacs = np.divide(scat_dat_pacs[:,0], scat_dat_pacs[:,2])
  mk_akk_npl = scat_dat_nplqcd[:,6:9] 
  # Concatenate everything
  #TODO: Automate concatenation solve with lists for case of different data
  # lengths
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
          r'B, $a\mu_s=0.021$','NPLQCD (2007)']
  label = [r'$I=1$ $KK$ scattering length',r'$M_K^2$',r'$M_K a_{KK}$',lbl3, r'LO $\chi$-PT']
  # Open pdf
  pfit = PdfPages(datapath+'akk_mk_sq.pdf')
  tree = lambda p, x : (-1)*x*x/p[0]
  # define colormap
  colors = cm.Set1(np.linspace(0, 1, 5))
  # A ensembles strange mass 0.0225
  mk_by_fk = np.divide(scat_dat_lst[0][:,3], scat_dat_lst[0][:,7])
  mk_sq = np.multiply(scat_dat_lst[0][:,3], scat_dat_lst[0][:,3])
  mk_akk = scat_dat_lst[0][:,9:]
  p1 = plt.errorbar(mk_sq, mk_akk[:,0], mk_akk[:,1], fmt='v' + 'b',
                    label = lbl3[0],color=colors[1])
  ## A ensembles OLD data strange mass 0.0225
  #mk_by_fk = np.divide(scat_dat_lst[4][:,3], scat_dat_lst[4][:,7])
  #mk_sq = np.multiply(scat_dat_lst[4][:,3], scat_dat_lst[4][:,3])
  #mk_akk = scat_dat_lst[4][:,9:]
  #p1 = plt.errorbar(mk_sq, mk_akk[:,0], mk_akk[:,1], fmt='x' + 'b',
  #                  label = lbl3[0],color=colors[1])
  # A ensembles strange mass 0.2464
  mk_by_fk = np.divide(scat_dat_lst[1][:,3], scat_dat_lst[1][:,7])
  mk_sq = np.multiply(scat_dat_lst[1][:,3], scat_dat_lst[1][:,3])
  mk_akk = scat_dat_lst[1][:,9:]
  p1 = plt.errorbar(mk_sq, mk_akk[:,0], mk_akk[:,1], fmt='v' + 'b',
                    label = lbl3[1],color=colors[2])
  ## A ensembles OLD data strange mass 0.2464
  #mk_by_fk = np.divide(scat_dat_lst[5][:,3], scat_dat_lst[5][:,7])
  #mk_sq = np.multiply(scat_dat_lst[5][:,3], scat_dat_lst[5][:,3])
  #mk_akk = scat_dat_lst[5][:,9:]
  #p1 = plt.errorbar(mk_sq, mk_akk[:,0], mk_akk[:,1], fmt='x' + 'b',
  #                  label = lbl3[1],color=colors[2])
  # B ensemble strange mass 0.1861
  mk_by_fk = np.divide(scat_dat_lst[2][:,3], scat_dat_lst[2][:,7])
  mk_sq = np.multiply(scat_dat_lst[2][:,3], scat_dat_lst[2][:,3])
  mk_akk = scat_dat_lst[2][:,9:]
  p1 = plt.errorbar(mk_sq, mk_akk[:,0], mk_akk[:,1], fmt='^' + 'b',
                    label = lbl3[2],color=colors[1])

  # B ensemble strange mass 0.1861
  mk_by_fk = np.divide(scat_dat_lst[3][:,3], scat_dat_lst[3][:,7])
  mk_sq = np.multiply(scat_dat_lst[3][:,3], scat_dat_lst[3][:,3])
  mk_akk = scat_dat_lst[3][:,9:]
  p1 = plt.errorbar(mk_sq, mk_akk[:,0], mk_akk[:,1], fmt='^' + 'b',
                    label = lbl3[3],color=colors[2])
  ## NPLQCD data
  #mk_sq_npl = np.multiply(scat_dat_nplqcd[:,0], scat_dat_nplqcd[:,0])
  #p1 = plt.errorbar(mk_fk_npl[:,0], mk_akk_npl[:,0], mk_akk_npl[:,1], fmt='o' + 'b',
  #                  label = lbl3[4],color=colors[4])
  #p1 = plt.errorbar(X, Y, dY, fmt='o' + 'b',
  #                  label = label[3][_dl],color=next(colors))
  # plottin the fit function, set fit range
  lfunc = 3
  ufunc = 4
  x1 = np.linspace(lfunc, ufunc, 1000)
  y1 = []
  for i in x1:
      y1.append(tree([8*math.pi],i))
  y1 = np.asarray(y1)
  #p2, = plt.plot(x1, y1, color=colors[3], label = label[4])
  # adjusting the plot style
  plt.grid(True)
  plt.xlabel(label[1])
  plt.ylabel(label[2])
  plt.title(label[0])
  plt.legend(ncol=2, numpoints=1)
  #ana.corr_fct_with_fit(mk_by_fk, mk_akk[:,:,0], mk_akk[:,:,1], tree,
  #    [8*math.pi],
  #    [0,5], label, pfit, xlim=[3,4], ylim=[-0.65,-0.25])
  # set the axis ranges
  plt.xlim([0.0,0.1])
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
