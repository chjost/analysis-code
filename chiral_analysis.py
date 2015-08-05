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
# Function: Fits and Interpolations of a_KK * m_K for different strange quark
# masses amu_s 
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

  #----------- Define global parameters, like number of ensembles -------------
  
  # A-ensemble strange quark masses
  amu_s = [0.0185,0.0225,0.02464]
  nb_mu_s = len(amu_s)
  # Source path for data
  src_path = "/hiskp2/helmes/k-k-scattering/data/A40.24/"
  # cache path for fit results
  cache_path = "/hiskp2/helmes/k-k-scattering/data/cache/"
  # Path for plots
  plt_path = "/hiskp2/helmes/k-k-scattering/plots/A40.24/"

  # Numpy array for mass and scattering length (dim: nb_samples, nb_mu_s)
  mk_sq_sum = np.zeros((1500,3))
  ma_kk_sum = np.zeros_like(mk_sq_sum)
  print ma_kk_sum.shape


  #----------- Read in samples: m_k, a0, mk_a0 --------------------------------

  for s in range(0, nb_mu_s):
    # mk_a0 input
    infile_ma_kk = src_path + "mk_a0_" + str(amu_s[s])[3:] + ".npy"
    ma_kk = ana.read_data(infile_ma_kk)
    # Kaon mass input
    infile_mk = src_path + "m_k_" + str(amu_s[s])[3:] +".npy"
    mk = ana.read_data(infile_mk)
    print mk

    print ma_kk.shape
  # Append read in results to arrays.
    ma_kk_sum[:,s] = ma_kk
    mk_sq_sum[:,s] = np.square(mk)


  #----------- Fits to resorted data (Bootstrapsamplewise) --------------------

  print ma_kk_sum[0,:], mk_sq_sum[0,:]
  # Linear interpolation
  # Quadratic interpolation
  # linear fit
  
  # calculate mean and standard deviation from samples
  ma_kk_mean, ma_kk_std = ana.calc_error(ma_kk_sum, 0)
  mk_sq_mean, mk_sq_std = ana.calc_error(mk_sq_sum, 0)
  print ma_kk_mean, ma_kk_std, mk_sq_mean, mk_sq_std

  #------------------ Plot mk_a0 and mk^2 vs. amu_s ---------------------------
  
  # Plot original data together with statistical error
  # Savepaths
  pltout_mk_sq = plt_path+"mk_sq_chiral.pdf"
  pltout_ma_kk = plt_path+"ma_kk_chiral.pdf"
  # PDFplots
  pdf_mk = PdfPages(pltout_mk_sq) 
  pdf_ma_kk = PdfPages(pltout_ma_kk) 
  # Labels
  label_mk_sq = [r'Chiral behaviour of $M_K$',r'$a\mu_s$',r'$M_K^2$',r'data']
  label_ma_kk = [r'Chiral behaviour of $a_0M_K$',r'$a\mu_s$',r'$a_0M_K$',r'data']
  ana.plot_data(amu_s, mk_sq_sum[0,:], mk_sq_std, pdf_mk, label_mk_sq)
  ana.plot_data(amu_s, ma_kk_sum[0,:], ma_kk_std, pdf_ma_kk, label_ma_kk)


# make this script importable, according to the Google Python Style Guide
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
