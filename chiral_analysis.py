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
  scat_dat = np.loadtxt(filename_own,usecols=(1,2,3,4,5,6,7,8,9,14,15,16,17)).reshape((3,5,13))
  scat_dat_nplqcd = np.loadtxt(filename_nplqcd,usecols=(7,8,9,13,14,15))
  scat_dat_pacs = np.loadtxt(filename_pacs,usecols=(2,3,4,5,6,7))
  # print scat_dat[0,:,0:3]
  # need mk/fk for plot, how to include statistical and systematical uncertainties?
  mk_by_fk = np.divide(scat_dat[:,:,3], scat_dat[:,:,7])
  mk_fk_pacs = np.divide(scat_dat_pacs[:,0], scat_dat_pacs[:,2])
  mk_akk = scat_dat[:,:,9:]
  print mk_by_fk.shape
  print mk_akk.shape
  # Concatenate everything
  #TODO: Automate concatenation
  mk_by_fk_all = np.concatenate((mk_by_fk, scat_dat_nplqcd[:,0].reshape(1,5), mk_fk_pacs.reshape(1,5)))
  mk_akk_all = np.concatenate((mk_akk[:,:,0:2],scat_dat_nplqcd[:,3:5].reshape((1,5,2)), scat_dat_pacs[:,4:].reshape((1,5,2))))
  print mk_by_fk_all
  print mk_akk_all

  #----------- Fit NLO-ChiPT to resorted data ---------------------------------
  #----------- Make a nice plot including CHiPT tree level --------------------
  lbl3 = [r'$a\mu_s=0.0185$','$a\mu_s=0.0225$',r'$a\mu_s=0.02464$', "NPLQCD", "PACS-CS"]
  label = [r'$I=1$ $KK$ scattering length',r'$M_K/f_K$',r'$M_K\cdot a_{KK}$',lbl3,'LO ChiPT']
  # Open pdf
  pfit = PdfPages(datapath+'akk_chiral.pdf')
  tree = lambda p, x : (-1)*x*x/p[0]
  ana.corr_fct_with_fit(mk_by_fk_all, mk_akk_all[:,:,0], mk_akk_all[:,:,1], tree,
      [8*math.pi],
      [0,5], label, pfit, xlim=[3,6], ylim=[-0.65,-0.25])
  pfit.close() 

# make this script importable, according to the Google Python Style Guide
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
