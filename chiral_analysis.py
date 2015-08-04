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

  #----------- Define global parameters, like number of ensembles
  # A-ensemble strange quark masses
  amu_s = [0.0185,0.0225,0.02464]
  nb_mu_s = len(amu_s)
  # Source path for data
  src_path = "/hiskp2/helmes/k-k-scattering/data/A40.24/"
  # cache path for fit results
  cache_path = "/hiskp2/helmes/k-k-scattering/data/cache/"

  # Numpy array for scattering length (dim: nb_samples, nb_mu_s)
  ma_kk_sum = np.zeros((1500,3))
  print ma_kk_sum.shape

  #----------- Read in the data we want and resort it  ------------------------
  for s in range(0,nb_mu_s):
    # Scattering length input
    infile_ma_kk = src_path + "scat_length_" + str(amu_s[s])[3:] + ".npy"
    ma_kk = ana.read_data(infile_ma_kk)
    # Kaon mass input
    infile_mk = cache_path + "A40.24_" + str(amu_s[s])[3:] +"_k_corr_p0.datgenfit_mass.npz"
    ranges, mk, chi2_mk, pvals_mk = ana.read_fitresults(infile_mk)
    print mk

    print ma_kk[0,0].shape
  # Append read in results to array.
    ma_kk_sum[:,s] = ma_kk[0,0]
  #----------- Fits to resorted data (Bootstrapsamplewise) --------------------
  print ma_kk_sum[0,:]
  # Linear interpolation
  # Quadratic interpolation
  # linear fit

  #----------- Make a nice plot including CHiPT tree level --------------------

# make this script importable, according to the Google Python Style Guide
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
