#!/hadron/knippsch/Enthought/Canopy_64bit/User/bin/python
##!/usr/bin/python
################################################################################
#
# Author: Christopher Helmes (helmes@hiskp.uni-bonn.de)
# Date:   Februar 2016
#
# Copyright (C) 2016 Christopher Helmes
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
# Function: Fits of interpolated a_KK * m_K for different strange quark
# masses amu_s 
#
# For informations on input parameters see the description of the function.
#
################################################################################

# system imports
import sys
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
import analysis2 as ana


def main():
# Parse the input
  if len(sys.argv) < 2:
      ens = ana.LatticeEnsemble.parse("A40.24.ini")
  else:
      ens = ana.LatticeEnsemble.parse(sys.argv[1])

  # get data from input file
  lat = ens.name()
  latA = ens.get_data("namea")
  latB = ens.get_data("nameb")
  latD = ens.get_data("named")
  strangeA = ens.get_data("strangea")
  strangeB = ens.get_data("strangeb")
  strangeD = ens.get_data("stranged")
  print strangeD
  amusA = ens.get_data("amu_s_a")
  amusB = ens.get_data("amu_s_b")
  amusD = ens.get_data("amu_s_d")
  AB = latA+latB
  print(AB)
  #quark = ens.get_data("quark")
  datadir = ens.get_data("datadir") 
  plotdir = ens.get_data("plotdir") 
  resdir = ens.get_data("resultdir") 
  nboot = ens.get_data("nboot")
  
  readchipt = True 
  #readchipt = False 
  xcut = False

  # Firstly, read in all interpolated data into one array for which the
  # fits are conducted and append them to an array
  
  # read helper values to the x-observable
  x_help = ana.read_extern('./plots2/data/mpi.dat',(1,2))
  y_help = ana.read_extern('./plots2/data/fk_matchD_0.015.dat',(2,3))
  # Initialize a result array for the Ydata
  #nb_ens=len(latA)+len(latB)
  nb_ens=len(latA)+len(latB)+len(latD)

  # arrays for x and y data
  # y_data have layout: (number ensembles,value dstat -dsys  +dsys)
  y_plot = np.zeros((nb_ens,4))
  y_fit = np.zeros((nb_ens,nboot))
  x_plot = np.zeros_like(y_plot)
  x_fit = np.zeros_like(y_fit)

  for j,a in enumerate(AB):
      y_plot[j], y_fit[j] = ana.prepare_mk("match_k",datadir,a,y_help,nboot)  
      x_plot[j], x_fit[j] = ana.prepare_mpi(x_help,a,nboot)    

  ens_AB=len(latA)+len(latB)
  for j,a in enumerate(latD):
      # index offset
      j+=ens_AB
      y_plot[j],y_fit[j]  = ana.prepare_mk("fit_k",datadir,a,y_help,nboot,strangeD[0],strangeD[0])
      #y_plot[j],y_fit[j]  = ana.prepare_mpi(y_help,a,nboot,square=False)
      x_plot[j],x_fit[j]  = ana.prepare_mpi(x_help,a,nboot)

  print(y_plot)
  # Secondly, fit a chiral function to the data
  #define a lambda fit function
  #if readchipt:
  #    if xcut:
  #        chiral_extra=ana.FitResult.read(resdir+"chiptfit_orig_mpi_xcut_%d.npz" % xcut)
  #    else:
  #        chiral_extra=ana.FitResult.read(resdir+'chiptfit_orig_mpi.npz')
  #else:
  #    if xcut:
  #        chipt = ana.LatticeFit(lo_chipt,dt_i=1, dt_f=1)
  #        p=[1.,1.]
  #        chiral_extra = chipt.chiral_fit(x_fit,y_fit,corrid="MA-ChiPT",start=p,xcut=xcut)
  #        chiral_extra.save(resdir+"chiptfit_orig_mpi_xcut_%d.npz" % xcut)
  #    else:
  #        chipt = ana.LatticeFit(lo_chipt,dt_i=1, dt_f=1)
  #        p=[1.,1.]
  #        chiral_extra = chipt.chiral_fit(x_fit,y_fit,corrid="MA-ChiPT",start=p)
  #        chiral_extra.save(resdir+'chiptfit_orig_mpi.npz')
  #chiral_extra.print_data()
  #chiral_extra.print_data(par=1)
  #print(chiral_extra.data[0].shape)
  print("m a0 has shape:")
  print(y_plot.shape)
  # generate fitfunction points and physical point
  #error for physical point
  phys_pt_x = 0.1124
  #mk_a0_fin = err_phys_pt(chiral_extra.data[0],phys_pt_x,lo_chipt)
  #if xcut:
  #    b=x_plot[:,0] < xcut
  #    ndof = x_plot[b].shape[0]
  #else:
  #    ndof = nb_ens-chiral_extra.data[0].shape[1]
  #print("chi_sq/ndof = %f" % (chiral_extra.chi2[0][0,0]/ndof))
  #print("p-val = %f" % chiral_extra.pval[0][0,0])
  #print("M_Ka_0 at physical point is: %e +/- %e" % (mk_a0_fin[0], mk_a0_fin[1]))
  #x = np.linspace(0,np.amax(x_plot),1000)
  #y = np.asarray([lo_chipt(chiral_extra.data[0][0,:,0],_x) for _x in x])
  # Lastly, plot the fitted function and the data
  if xcut:
      pfit = PdfPages("./plots2/pdf/LOChiPT_matchD_xcut_%d.pdf" % xcut)
  else:
      pfit = PdfPages("./plots2/pdf/MK_fK_vs_mpisq.pdf")
  #args = chiral_extra.data[0][...,0]
  #dummy=np.asarray((-1./8*np.pi),0.)
  #ana.plot_function(lo_chipt,x,args,label=r'LO-$\chi$-PT',ploterror=True)
  #ana.plot_function(lo_chipt,x,dummy,label=r'LO-$\chi$-PT',ploterror=True)
  # plot systematic errors on top

  ana.plot_ensemble(x_plot,y_plot,(0,6),'s','red','A Ensembles')
  ana.plot_ensemble(x_plot,y_plot,(6,9),'^','blue','B Ensembles')
  ana.plot_ensemble(x_plot,y_plot,(9,10),'o','green','D Ensembles')

  if xcut:
    y = ana.lo_chipt(chiral_extra.data[0][0,:,0], xcut)
    plt.vlines(xcut, 0.95*y, 1.05*y, colors="k", label="")
    plt.hlines(0.95*y, xcut*0.98, xcut, colors="k", label="")
    plt.hlines(1.05*y, xcut*0.98, xcut, colors="k", label="")
  #plt.errorbar(phys_pt_x,mk_a0_fin[0],mk_a0_fin[1], fmt='d', color='darkorange', label='Physical Point')
  #plt.errorbar(phys_pt_x,mk_a0_fin[0], fmt='d', color='darkorange')
  plt.grid(False)
  plt.xlim(0.,1.6)
  plt.legend(loc='best',numpoints=1)
  plt.ylabel(r'$M_K/f_K$')
  plt.xlabel(r'$(r_0M_{\pi})^2$')
  pfit.savefig()
  pfit.close()
  


# make this script importable, according to the Google Python Style Guide
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard Interrupt")

