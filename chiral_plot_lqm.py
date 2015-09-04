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
  bootpath = '/hiskp2/helmes/k-k-scattering/data/'
  plotpath = rootpath+'plots/'
  datapath = rootpath+'data/'
  # Ensemble name
  ensemble = ['A40.24/','A60.24/','A80.24/','A100.24/']
  # strange quark masses
  mu_s = ['225','2464']
  # physical values
  r0_mpi_ph = 0.3525029
  boot_input = [bootpath+ens for ens in ensemble]
  files = ["mk_a0_"+m+".npy" for m in mu_s]
  a0_mu_hi = [path+"mk_a0_"+mu_s[1]+".npy" for path in boot_input]
  a0_mu_lo = [path+"mk_a0_"+mu_s[0]+".npy" for path in boot_input]
  nb_bs = 1500
  a0_hi = np.zeros((nb_bs,len(a0_mu_hi)))
  a0_lo = np.zeros_like(a0_hi)
  for i,(hel,lel) in enumerate(zip(a0_mu_hi, a0_mu_lo)):
    a0_hi[:,i] = ana.read_data(hel)
    a0_lo[:,i] = ana.read_data(lel)
  # The file should contain MK*aKK, Mpi and r0 as columns
  filename_own = datapath + 'ma_mpi.dat'
  filename_own2 = datapath + 'ma_mk_match.dat'
  filename_nplqcd = datapath + 'ma_mf_npl.dat'
  filename_pacs = datapath + 'ma_mf_pacs.dat'
  # Get mk_akk bootstrap samples for extrapolation
  # TODO: Replace txtfile by binary format in future
  # for plotting r0*Mpi
  scat_dat_nplqcd = np.loadtxt(filename_nplqcd,usecols=(1,2,3,4,5,9,15,16,17))
  scat_dat_pacs = np.loadtxt(filename_pacs,usecols=(0,1,4,5,6,7))
  scat_dat = np.loadtxt(filename_own,usecols=(1,2,3,10,11,12,13))
  
  # load and concatenate bootstrap samples as array of shape ens x samples
  # extrapolate and fit linearly


  #split arrays
  #scat_dat_lst = np.split(scat_dat,[4,8,9])
  #scat_dat_lst.pop()
  # need (mpi*r0)^2 for plot, how to include statistical and systematical uncertainties?
  mpi_r0_npl = np.multiply(scat_dat_nplqcd[:,0],np.multiply(1.474,scat_dat_nplqcd[:,3]))
  mpi_r0_pacs = np.multiply(scat_dat_pacs[:,0:2],0.5/(0.19733))
  mpi_etmc = scat_dat[0:4,0]
  mpi_r0_etmc_low = np.multiply(scat_dat[0:4,0],5.31)
  mpi_r0_etmc_high = np.multiply(scat_dat[4:8,0],5.31)
  print mpi_r0_etmc_low, mpi_r0_pacs, mpi_r0_npl
  mk_akk_npl = ana.sum_error_sym(scat_dat_nplqcd[:,6:])
  mk_akk_pacs = scat_dat_pacs[:,4:6]
  mk_akk_etmc = []
  mk_akk_etmc_low = scat_dat[0:4,3:]
  print mk_akk_etmc_low
  mk_akk_etmc_high = scat_dat[4:8,3:]
  # check data
  label_ens = [r'A40',r'A60',r'A80',r'A100']
  print("etmc:\nLat\tbmk\tf_k\tmk/fk\tmk_akk")
  #for i in range(len(scat_dat_lst)):
      #for m,a in zip(mk_fk_etmc[i],mk_akk_etmc[i]):
  #for ens,m,f,mf,a in zip(label_ens,scat_dat[:,0], scat_dat[:,2], mk_fk_etmc, mk_akk_etmc_low):
  #  print ens, m, f, mf, a
  #    # print("\n")

  #for ens,m,f,mf,a in zip(label_ens,scat_dat2[:,0], scat_dat2[:,2], mk_fk_etmc, mk_akk_etmc_high):
  #  print ens, m, f, mf, a
  #    # print("\n")
  #print("\nnpl:\nbmk/fk\tmk_akk")
  #for m,a in zip(mk_fk_npl,mk_akk_npl):
  #    print m, a

  #print("\npacs:\nbmpi\tmk_akk")
  #for m,a in zip(mk_fk_pacs,mk_akk_pacs):
  #    print m, a


  # -------------- linear extrapolations to (r0*M_pi_phys)^2 ------------------
  # Interpolate the 2 low lying values for a0_mk  and evaluate at physical point
  # for both strange quark masses
  # define a linear function
  linfit = lambda p, t: p[0]*t+p[1]
  # mu_s = 0.2464
  print mu_s[1]
  coeff_ipol_a0hi = ana.ipol_lin(a0_hi[:,0:2],np.square(mpi_r0_etmc_high)[0:2])
  a0_mk_hi_ext = ana.eval_lin(coeff_ipol_a0hi,np.square(r0_mpi_ph))
  mean_a0_hi_ext, std_a0_hi_ext = ana.calc_error(a0_mk_hi_ext)
  print("lin. ext.-pol.:\ta0 * MK = %f +/- %f" % (a0_mk_hi_ext[0], std_a0_hi_ext))

  # linear fit
  p_a0_mk_hi, clo2_a0_mk_hi, pvals_a0_mk_hi = ana.fitting(linfit,
                                np.square(mpi_r0_etmc_low), a0_hi,
                                [2.,1.], verbose=True)
  a0_mk_hi_fit_ext = ana.eval_lin(p_a0_mk_hi,np.square(r0_mpi_ph))
  mean_a0_hi_lin_ext, std_a0_hi_lin_fit =ana.calc_error(a0_mk_hi_fit_ext) 
  print("ext. lin. fit:\ta0 * MK = %f +/- %f" % (a0_mk_hi_fit_ext[0], std_a0_hi_lin_fit))
  #print a0_hi.shape,  mpi_r0_etmc_high.shape
  #p_a0_mk_hi, chi2_a0_mk_hi, pvals_a0_mk_hi = ana.fitting(ana.chi_pt_cont,
  #                              mpi_etmc, a0_hi,
  #                              np.array([1.,1.]), verbose=True)
  #a0_mk_hi_fit_ext = np.zeros(1500)
  #print p_a0_mk_hi[0]
  #for i in range(0,1500):
  #  a0_mk_hi_fit_ext[i] = ana.eval_chi_pt_cont(p_a0_mk_hi[i],ana.phys_to_lat(139.7))
  #  #print a
  #mean_a0_hi_lin_ext, std_a0_hi_lin_fit = ana.calc_error(a0_mk_hi_fit_ext) 
  #print("ext. lin. fit:\ta0 * MK = %f +/- %f" % (a0_mk_hi_fit_ext[0], std_a0_hi_lin_fit))

  # mu_s = 0.225
  print mu_s[0]
  coeff_ipol_a0lo = ana.ipol_lin(a0_lo[:,0:2],np.square(mpi_r0_etmc_low)[0:2])
  a0_mk_lo_ext = ana.eval_lin(coeff_ipol_a0lo,np.square(r0_mpi_ph))
  mean_a0_lo_ext, std_a0_lo_ext = ana.calc_error(a0_mk_lo_ext)
  print("lin. ext.-pol.:\ta0 * MK = %f +/- %f" % (a0_mk_lo_ext[0], std_a0_lo_ext))
  
  p_a0_mk_lo, clo2_a0_mk_lo, pvals_a0_mk_lo = ana.fitting(linfit,
                                np.square(mpi_r0_etmc_low), a0_lo,
                                [2.,1.], verbose=True)
  a0_mk_lo_fit_ext = ana.eval_lin(p_a0_mk_lo,np.square(r0_mpi_ph))
  mean_a0_lo_lin_ext, std_a0_lo_lin_fit =ana.calc_error(a0_mk_lo_fit_ext) 
  print("ext. lin. fit:\ta0 * MK = %f +/- %f" % (a0_mk_lo_fit_ext[0], std_a0_lo_lin_fit))
  # unitary matched case
  coeff_ipol_a0m = ana.ipol_lin(a0_lo[:,0:2],np.square(mpi_r0_etmc_low)[0:2])
  a0_mk_lo_ext = ana.eval_lin(coeff_ipol_a0lo,np.square(r0_mpi_ph))
  mean_a0_lo_ext, std_a0_lo_ext = ana.calc_error(a0_mk_lo_ext)
  print("lin. ext.-pol.:\ta0 * MK = %f +/- %f" % (a0_mk_lo_ext[0], std_a0_lo_ext))
  
  p_a0_mk_lo, clo2_a0_mk_lo, pvals_a0_mk_lo = ana.fitting(linfit,
                                np.square(mpi_r0_etmc_low), a0_lo,
                                [2.,1.], verbose=True)
  a0_mk_lo_fit_ext = ana.eval_lin(p_a0_mk_lo,np.square(r0_mpi_ph))
  mean_a0_lo_lin_ext, std_a0_lo_lin_fit =ana.calc_error(a0_mk_lo_fit_ext) 
  print("ext. lin. fit:\ta0 * MK = %f +/- %f" % (a0_mk_lo_fit_ext[0], std_a0_lo_lin_fit))


  #----------- Fit NLO-ChiPT to resorted data ---------------------------------
  #----------- Make a nice plot including CHiPT tree level --------------------
  lbl3 = [r'A, $a\mu_s=0.0225$',
          r'A, $a\mu_s=0.02464$',
          r'B, $a\mu_s=0.01861$',
          r'B, $a\mu_s=0.021$','NPLQCD (2007)','PACS-CS (2013)']
  label = [r'$I=1$ $KK$ scattering length, $L = 24^3$',r'$(r_0M_{\pi})^2$',r'$M_K a_{KK}$',lbl3, r'LO $\chi$-PT']
  label_ens = [r'A40',r'A60',r'A80',r'A100']
  ## Open pdf
  pfit = PdfPages(plotpath+'akk_r0mpi.pdf')
  tree = lambda p, x : (-1)*x*x/p[0]
  # define colormap
  colors = cm.Dark2(np.linspace(0, 1, 7))
  # A ensembles unitary matching
  #print mpi_r0_sq_etmc
  #p1 = plt.errorbar(mk_fk_etmc, mk_akk_etmc[:,0], mk_akk_etmc[:,1], fmt='o' + 'b',
  #                  label = r'A, $a\mu_s^\mathrm{unit}$, fk_ipol',color=colors[0])
  #for X, Y,l in zip(mk_fk_etmc,mk_akk_etmc[:,0],label_ens):
  #                      # Annotate the points 5 _points_ above and to the left
  #                      # of the vertex
  #    plt.annotate(l,(X-0.001,Y+0.01))
  mpi_etmc_phys=ana.lat_to_phys(mpi_etmc)
  print mpi_etmc_phys
  mpisq = plt.errorbar(np.square(mpi_r0_etmc_low), mk_akk_etmc_low[:,0], mk_akk_etmc_low[:,1], fmt='o' + 'b',
                    label = r'A, $a\mu_s = 0.0225$',color='blue')
  mpisq = plt.errorbar(np.square(mpi_r0_etmc_low), mk_akk_etmc_high[:,0], mk_akk_etmc_high[:,1], fmt='o' + 'b',
                    label = r'A, $a\mu_s = 0.02464$',color='orange')
  for X, Y,l in zip(np.square(mpi_etmc_phys),mk_akk_etmc_high[:,0],label_ens):
                        # Annotate the points 5 _points_ above and to the left
                        # of the vertex
      plt.annotate(l,(X-0.001,Y+0.05))
# plot extrapolations
  ## plottin the fit function, set fit range
  lfunc = 0
  ufunc = 3.5
  x1 = np.linspace(lfunc, ufunc, 1000)
  y1 = []
  y2 = []
  #y3 = []
  for i in x1:
    y1.append(linfit(p_a0_mk_hi[0,:],i))
    y2.append(linfit(p_a0_mk_lo[0,:],i))
    #y3.append(ana.eval_chi_pt_cont(p_a0_mk_hi[0],i))
  y1 = np.asarray(y1)
  y2 = np.asarray(y2)
  #y3 = np.asarray(y3)
  p2, = plt.plot(x1, y1, color='orange', label = "linear ext.")
  p2, = plt.plot(x1, y2, color='blue', label = "linear ext.")
  #p2, = plt.plot(x1, y3, color='darkgreen', label = "cont chipt")
#  # A ensembles strange mass 0.0225
#  #print mpi_r0_sq_etmc
#  p1 = plt.errorbar(mk_fk_etmc[0], mk_akk_etmc[0][:,0], mk_akk_etmc[0][:,1], fmt='o' + 'b',
#                    label = lbl3[0],color=colors[0])
#  # A ensembles strange mass 0.02464
#  #print mpi_r0_sq_etmc
#  p1 = plt.errorbar(mk_fk_etmc[1], mk_akk_etmc[1][:,0], mk_akk_etmc[1][:,1], fmt='o' + 'b',
#                    label = lbl3[1],color=colors[1])
#  # B ensembles strange mass 0.01861
#  # define x and y data
#  #print mpi_r0_sq_etmc
#  p1 = plt.errorbar(mk_fk_etmc[2], mk_akk_etmc[2][:,0], mk_akk_etmc[2][:,1], fmt='o' + 'b',
#                    label = lbl3[2],color=colors[2])
#  # B ensembles strange mass 0.021
#  # define x and y data
#  #print mpi_r0_sq_etmc
#  p1 = plt.errorbar(mk_fk_etmc[3], mk_akk_etmc[3][:,0], mk_akk_etmc[3][:,1], fmt='o' + 'b',
#                    label = lbl3[3],color=colors[3])
  ## NPLQCD data
  mk_sq_npl = np.multiply(scat_dat_nplqcd[:,0], scat_dat_nplqcd[:,0])
  mpisq = plt.errorbar(np.square(mpi_r0_npl), mk_akk_npl[:,0], mk_akk_npl[:,1], fmt='x' + 'b',
                    label = lbl3[4],color='green')
  ## PACS data
  mpisq = plt.errorbar(np.square(mpi_r0_pacs)[:,0], mk_akk_pacs[:,0], mk_akk_pacs[:,1], fmt='^' + 'b',
                    label = lbl3[5],color='red')
  ## plottin the fit function, set fit range
  p1 = plt.axvline(x=0.3525029**2, color='gray',ls='--',label='phys. point')
  ## adjusting the plot style
  plt.grid(True)
  plt.xlabel(label[1])
  plt.ylabel(label[2])
  plt.title(label[0])
  plt.legend(ncol=1, numpoints=1, loc=1)
  plt.xlim([0.0,3.5])
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

