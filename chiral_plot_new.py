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
  rootpath = '/hiskp2/helmes/analysis/scattering/k_charged/'
  bootpath = '/hiskp2/helmes/analysis/scattering/k_charged/data/'
  plotpath = rootpath+'plots/'
  datapath = rootpath+'results/'

  # physical values
  r0_mpi_ph = 0.3525029
  #ETMC r0 values sorted by spacing
  r0 = np.array((5.31,5.77,7.60))

  # The file should contain MK*aKK, Mpi and r0 as columns
  filename_own = './kk_beta_matched.dat'
  filename_nplqcd = datapath + 'ma_mf_npl.dat'
  filename_pacs = datapath + 'ma_mf_pacs.dat'

  # Get the data to plot
  mpi_etmc=np.loadtxt(filename_own, usecols=(1,2,3,8,9,10,11))
  print mpi_etmc.shape
  A_match_B = mpi_etmc[0:6]
  A_match_D = mpi_etmc[6:12]
  xA_match_B = np.square(A_match_B[:,0]*r0[0])
  xA_match_D = np.square(A_match_D[:,0]*r0[0])
  yA_match_B = A_match_B[:,3:]
  yA_match_D = A_match_D[:,3:]
  print(yA_match_D[:,0])
  B = mpi_etmc[12:16]
  B_match_D = mpi_etmc[16:20]
  xB = np.square(B[:,0]*r0[1])
  xB_match_D = np.square(B_match_D[:,0]*r0[1])
  yB = B[:,3:]
  yB_match_D = B_match_D[:,3:]
  D = mpi_etmc[20]
  print D
  xD = np.square(D[0]*r0[2])
  yD = D[3:]

  scat_dat_nplqcd = np.loadtxt(filename_nplqcd,usecols=(1,2,3,4,5,9,15,18,19,20))
  scat_dat_pacs = np.loadtxt(filename_pacs,usecols=(0,1,4,5,6,7))
  #split arrays
  # need (mpi*r0)^2 for plot, how to include statistical and systematical uncertainties?
  mpi_r0_npl = np.multiply(scat_dat_nplqcd[:,0],np.multiply(1.474,scat_dat_nplqcd[:,3]))
  mpi_r0_pacs = np.multiply(scat_dat_pacs[:,0:2],0.5/(0.19733))
  mk_akk_npl = ana.sum_error_sym(scat_dat_nplqcd[:,7:])
  mk_akk_pacs = scat_dat_pacs[:,4:6]
  #----------- Make a nice plot including CHiPT tree level --------------------
  lbl3 = [r'A, $a\mu_s=0.0225$',
          r'A, $a\mu_s=0.02464$',
          r'B, $a\mu_s=0.01861$',
          r'B, $a\mu_s=0.021$','NPLQCD (2007)','PACS-CS (2013)']
  label = [r'',r'$(r_0M_{\pi})^2$',r'$M_K a_{KK}$',lbl3, r'LO $\chi$-PT']
  ## Open pdf
  pfit = PdfPages(plotpath+'akk_r0mpi_matchD.pdf')
  plt.figure(figsize=(8.5,8.5))
  # A matched to D
  mpisq = plt.errorbar(xA_match_D,yA_match_D[:,0],[yA_match_D[:,1]+yA_match_D[:,3],yA_match_D[:,1]+yA_match_D[:,2]],
            fmt ='s',color='red')
  mpisq = plt.errorbar(xA_match_D,yA_match_D[:,0],yA_match_D[:,1],
            fmt ='s',color='red',label = 'A Ensembles')

  # B matched to D
  mpisq = plt.errorbar(xB_match_D,yB_match_D[:,0],[yB_match_D[:,1]+yB_match_D[:,3],yB_match_D[:,1]+yB_match_D[:,2]],
            fmt ='^',color='blue')
  mpisq = plt.errorbar(xB_match_D,yB_match_D[:,0],yB_match_D[:,1],
            fmt ='^',color='blue',label = 'B Ensembles')

  # D
  print yD[3],yD[2]
  mpisq = plt.errorbar(xD,yD[0],yerr=np.array(yD[1]+yD[3],yD[1]+yD[2]),
            fmt ='o',color='green')
  mpisq = plt.errorbar(xD,yD[0],yD[1],
            fmt ='o',color='green',label = 'D Ensembles')
  ## NPLQCD data
  mk_sq_npl = np.multiply(scat_dat_nplqcd[:,0], scat_dat_nplqcd[:,0])
  print('NPL Data:')
  print mk_akk_npl
  mpisq = plt.errorbar(np.square(mpi_r0_npl), mk_akk_npl[:,0], mk_akk_npl[:,1], fmt='x' + 'b',
                    label = lbl3[4],color='black')
  ## PACS data
  #mpisq = plt.errorbar(np.square(mpi_r0_pacs)[:,0], mk_akk_pacs[:,0], mk_akk_pacs[:,1], fmt='^' + 'b',
  #                  label = lbl3[5],color='black')
  ## plottin the fit function, set fit range
  p1 = plt.axvline(x=0.3525029**2, color='gray',ls='--',label='phys. point')
  ## adjusting the plot style
  plt.grid(False)
  plt.xlabel(label[1],fontsize=20)
  plt.ylabel(label[2],fontsize=20)
  plt.title(label[0])
  plt.legend(ncol=1, numpoints=1, loc=1)
  plt.xlim([0.0,2.5])
  plt.ylim([-0.65,-0.22])
  # save pdf
  pfit.savefig()
  pfit.close() 

  #----------- Make a nice plot including CHiPT tree level --------------------
  lbl3 = [r'A, $a\mu_s=0.0225$',
          r'A, $a\mu_s=0.02464$',
          r'B, $a\mu_s=0.01861$',
          r'B, $a\mu_s=0.021$','NPLQCD (2007)','PACS-CS (2013)']
  label = [r'',r'$(r_0M_{\pi})^2$',r'$M_K a_{KK}$',lbl3, r'LO $\chi$-PT']
  ## Open pdf
  pfit = PdfPages(plotpath+'akk_r0mpi_matchB.pdf')
  plt.figure(figsize=(8.5,8.5))
  # A matched to B
  mpisq = plt.errorbar(xA_match_B,yA_match_B[:,0],[yA_match_B[:,1]+yA_match_B[:,3],yA_match_B[:,1]+yA_match_B[:,2]],
            fmt ='s',color='red')
  mpisq = plt.errorbar(xA_match_B,yA_match_B[:,0],yA_match_B[:,1],
            fmt ='s',color='red',label = 'A-Ensembles')

  # B
  mpisq = plt.errorbar(xB,yB[:,0],[yB[:,1]+yB[:,3],yB[:,1]+yB[:,2]],
            fmt ='^',color='blue')
  mpisq = plt.errorbar(xB,yB[:,0],yB[:,1],
            fmt ='^',color='blue',label = 'B-Ensembles')

  ## NPLQCD data
  mk_sq_npl = np.multiply(scat_dat_nplqcd[:,0], scat_dat_nplqcd[:,0])
  print('NPL Data:')
  print mk_akk_npl
  mpisq = plt.errorbar(np.square(mpi_r0_npl), mk_akk_npl[:,0], mk_akk_npl[:,1], fmt='x' + 'b',
                    label = lbl3[4],color='black')
  ## PACS data
  #mpisq = plt.errorbar(np.square(mpi_r0_pacs)[:,0], mk_akk_pacs[:,0], mk_akk_pacs[:,1], fmt='^' + 'b',
  #                  label = lbl3[5],color='black')
  ## plottin the fit function, set fit range
  p1 = plt.axvline(x=0.3525029**2, color='gray',ls='--',label='phys. point')
  ## adjusting the plot style
  plt.grid(False)
  plt.xlabel(label[1],fontsize=20)
  plt.ylabel(label[2],fontsize=20)
  plt.title(label[0])
  plt.legend(ncol=1, numpoints=1, loc=1)
  plt.xlim([0.0,2.5])
  plt.ylim([-0.65,-0.22])
  # save pdf
  pfit.savefig()
  pfit.close() 

# make this script importable, according to the Google Python Style Guide
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard Interrupt")

