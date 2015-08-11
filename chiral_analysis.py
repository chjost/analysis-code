#!/hadron/knippsch/Enthought/Canopy_64bit/User/bin/python
##!/usr/bin/python
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
from scipy import interpolate as ip
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
    
    # Ensemble name
    ensemble = ['A40.24/','A60.24/','A80.24/','A100.24/']
    # A-ensemble strange quark masses
    amu_s = np.asarray([0.0185,0.0225,0.02464])
    nb_mu_s = len(amu_s)
    # unitary kaon masses from eta eta` paper
    # A40.24  0.25884(43)
    # A60.24  0.26695(52) 
    # A80.24  0.27706(61)
    #A100.24  0.28807(34)
    mk_unit = [[0.25884,0.00043],[0.26695,0.00052],[0.27706,0.00061],[0.28807,0.00034]]
    # loop over Ensembles
    for (ens, mk_unit_sq) in zip(ensemble,mk_unit):

        # Source path for data
        src_path = "/hiskp2/helmes/k-k-scattering/data/"+ens
        # cache path for fit results
        cache_path = "/hiskp2/helmes/k-k-scattering/data/cache/"
        # Path for plots
        plt_path = "/hiskp2/helmes/k-k-scattering/plots/"+ens
  
        # Numpy array for mass and scattering length (dim: nb_samples, nb_mu_s)
        mk_sq_sum = np.zeros((1500,3))
        ma_kk_sum = np.zeros_like(mk_sq_sum)
  
  
        #----------- Read in samples: m_k, a0, mk_a0 --------------------------------
        # loop over strange quark masses
        for s in range(0, nb_mu_s):
            # mk_a0 input
            infile_ma_kk = src_path + "mk_a0_" + str(amu_s[s])[3:] + ".npy"
            ma_kk = ana.read_data(infile_ma_kk)
            # Kaon mass input
            infile_mk = src_path + "m_k_" + str(amu_s[s])[3:] +".npy"
            mk = ana.read_data(infile_mk)
  
            # Append read in results to arrays.
            ma_kk_sum[:,s] = ma_kk
            mk_sq_sum[:,s] = np.square(mk)
  
  
        #------ Fit and interpolations to resorted data (Bootstrapsamplewise) -------

        # Necessary functions 
        # linear fit
        linfit = lambda p, t: p[0]*t+p[1]
        # quadratic interpolation
        sqfit = lambda p, t: p[0]*t**2+p[1]*t+p[2]
  
        # matching to M_K^unit with linear fit
        p_mk_sq, chi2_mk_sq, pvals_mk_sq = ana.fitting(linfit, amu_s, mk_sq_sum, [2.,1.])
        print("amu_s from unitary M_K matching\n")
        b_roots_fit = ana.match_qm(p_mk_sq, np.square(mk_unit_sq))
        mean_amu_s, std_amu_s = ana.calc_error(b_roots_fit)
        print("lin. fit:\tamu_s = %f +/- %f" % (b_roots_fit[0], std_amu_s))
  
        # matching to M_K^unit with linear interpolation
        interp1 = ana.ipol_lin(mk_sq_sum[:,1:], amu_s[1:])
        b_roots_p1 = ana.match_qm(interp1, np.square(mk_unit_sq))
        mean_amu_s_p1, std_amu_s_p1 = ana.calc_error(b_roots_p1)
        print("lin. i-pol.:\tamu_s = %f +/- %f" % (b_roots_p1[0], std_amu_s_p1))
        
        # matching to M_K^unit with quadratic interpolation
        interp2 = ana.ipol_quad(mk_sq_sum,amu_s)
        b_roots_p2 = ana.match_qm(interp2, np.square(mk_unit_sq))
        mean_amu_s_p2, std_amu_s_p2 = ana.calc_error(b_roots_p2)
        print("quadr. i-pol.:\tamu_s = %f +/- %f" % (b_roots_p2[0], std_amu_s_p2))
        
        #------------------ linear interpolation of M_K*a_kk ----------------------

        print("M_K * a_kk from unitary M_K matching\n")
        # Linear interpolation
        a_k_ipol = ana.ipol_lin(ma_kk_sum[:,1:], amu_s[1:])
        a_kk_unit = ana.eval_lin(a_k_ipol,b_roots_p1) 
        mean_a_kk_unit, std_a_kk_unit = ana.calc_error(a_kk_unit)
        print("lin. i-pol.:\tM_K * a_kk = %f +/- %f" % (a_kk_unit[0], std_a_kk_unit))
        
        # quadratic interpolation
        a_k_ipol2 = ana.ipol_quad(ma_kk_sum, amu_s)
        a_kk_unit_q = ana.eval_quad(a_k_ipol2,b_roots_p2)
        a_kk_unit_stat = ana.calc_error(a_kk_unit_q)
        print("quad. i-pol.:\tM_K * a_kk = %f +/- %f" % (a_kk_unit_q[0], a_kk_unit_stat[1]))

        #------------------ Plot mk_a0 and mk^2 vs. amu_s -------------------------
        
        # Get standard deviation for plots
        ma_kk_mean, ma_kk_std = ana.calc_error(ma_kk_sum, 0)
        mk_sq_mean, mk_sq_std = ana.calc_error(mk_sq_sum, 0)
  
        # Plot original data together with statistical error and the constant unitary
        # mass
        # Savepaths
        pltout_mk_sq = plt_path+"mk_sq_chiral.pdf"
        pltout_ma_kk = plt_path+"ma_kk_chiral.pdf"
        # PDFplots
        pdf_mk = PdfPages(pltout_mk_sq) 
        pdf_ma_kk = PdfPages(pltout_ma_kk) 
        # Labels
        label_mk_sq = [r'Chiral behaviour of $M_K$',r'$a\mu_s$',r'$(aM_K)^2$',
            ens[:-1],r'linear fit',r'$(aM_K^{\mathrm{unit}})^2$',
                       r'$a\mu_s^{\mathrm{K}}$']
        label_ma_kk = [r'Chiral behaviour of $a_0M_K$',r'$a\mu_s$',r'$a_0M_K$',
            ens[:-1],r'lin. ipol',r'$a_0M_K$',
                       r'$a\mu_s^{\mathrm{K}}$']

        # Plot the linear fit with its matched amu_s
        ana.plot_data_with_fit(amu_s, mk_sq_sum[0,:], mk_sq_std, linfit, p_mk_sq[0],
            None, label_mk_sq, pdf_mk, hconst = np.square(mk_unit_sq),
            vconst=(b_roots_fit[0],std_amu_s))

        # Plot the linear interpolation with its matched amu_s
        label_mk_sq[4] = r'linear ipol'
        ana.plot_data_with_fit(amu_s, mk_sq_sum[0,:], mk_sq_std, linfit, interp1[0],
            None, label_mk_sq, pdf_mk, hconst = np.square(mk_unit_sq),
            vconst=(b_roots_p1[0],std_amu_s_p1))

        # Plot the quadratic interpolation with its matched amu_s
        label_mk_sq[4] = r'quadr. ipol'
        ana.plot_data_with_fit(amu_s, mk_sq_sum[0,:], mk_sq_std, sqfit, interp2[0],
            None, label_mk_sq, pdf_mk, hconst = np.square(mk_unit_sq),
            vconst=(b_roots_p2[0],std_amu_s_p2))

        # Plot M_K * a_kk vs. amu_s
        # linear interpolation
        ana.plot_data_with_fit(amu_s, ma_kk_sum[0,:], ma_kk_std, linfit,
            a_k_ipol[0], None, label_ma_kk, pdf_ma_kk,
            hconst = (a_kk_unit[0],std_a_kk_unit),vconst=(b_roots_p1[0],std_amu_s_p1))
        # quadratic interpolation
        label_ma_kk[4] = r'qudr. ipol'
        ana.plot_data_with_fit(amu_s, ma_kk_sum[0,:], ma_kk_std, sqfit,
            a_k_ipol2[0], None, label_ma_kk, pdf_ma_kk,
            hconst = (a_kk_unit_q[0],a_kk_unit_stat[1]),vconst=(b_roots_p2[0],std_amu_s_p2))
        # Close pdf files
        pdf_mk.close() 
        pdf_ma_kk.close()

# make this script importable, according to the Google Python Style Guide
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
