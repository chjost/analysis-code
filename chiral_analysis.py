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

def read_extern(filename):
  """ Read external data with identifiers into a dicitonary
  ATM the tags are written in the first column and the data in the second and
  third.
  TODO: Rewrite that to be more dynamic
  """
  tags = np.loadtxt(filename,dtype='str', usecols=(0,))
  values = np.loadtxt(filename, usecols=(1,2))
  # build a dicitonary from the arrays
  data_dict = {}
  for i,a in enumerate(tags):
      data_dict[a] = values[i]
  return data_dict

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
  AB = latA+latB
  print(AB)
  #quark = ens.get_data("quark")
  datadir = ens.get_data("datadir") 
  plotdir = ens.get_data("plotdir") 
  nboot = ens.get_data("nboot")
  
  # Firstly, read in all interpolated data into one array for which the
  # fits are conducted and append them to an array
  fk_unit = read_extern('./plots2/data/fk_unitary.dat')
  # Initialize a result array for the Ydata
  nb_ens=len(latA)+len(latB)+len(latD)
  # arrays for x and y data
  mk_akk = np.zeros((nb_ens,nboot))
  mk_fk = np.zeros_like((mk_akk))
  for j,a in enumerate(AB):
      y_obs = ana.FitResult.read("%s%s/match_mk_akk_%s.npz" % (datadir, a, a))
      y_obs.calc_error()
      data_shape=y_obs.data[0].shape
      y_obs.data[0]=y_obs.data[0].reshape((data_shape[0],1,data_shape[-1]))
  # Each array is the weighted median over the fit ranges
      res, res_std, res_sys, data_weight = ana.sys_error(y_obs.data,y_obs.pval)
      mk_akk[j] = res[0]

      x_obs = ana.FitResult.read("%s%s/match_k_%s.npz" % (datadir, a, a))
      x_obs.calc_error()
      data_shape=x_obs.data[0].shape
      x_obs.data[0]=x_obs.data[0].reshape((data_shape[0],1,data_shape[-1]))
  # Each array is the weighted median over the fit ranges
      res, res_std, res_sys, data_weight = ana.sys_error(x_obs.data,x_obs.pval)
      fk = ana.draw_gauss_distributed(fk_unit[a][0],fk_unit[a][1],(1,nboot))
      # For tree level chipt the square is needed
      mk_fk[j] = np.divide(res[0],fk)**2

  ens_AB=len(latA)+len(latB)
  for j,a in enumerate(latD):
      y_obs = ana.FitResult.read("%s%s/%s/mk_akk_%s.npz" % (datadir, a, strangeD[j], a))
      y_obs.calc_error()
      data_shape=y_obs.data[0].shape
      #y_obs.data[0]=y_obs.data[0].reshape((data_shape[0],1,data_shape[-1]))
      y_obs.pval[0] =y_obs.pval[0].reshape((nboot,y_obs.pval[0].shape[-1]))
  # Each array is the weighted median over the fit ranges
      res, res_std, res_sys, data_weight = ana.sys_error(y_obs.data,y_obs.pval)
      mk_akk[ens_AB+j] = res[0]
  
      x_obs = ana.FitResult.read("%s%s/%s/fit_k_%s.npz" % (datadir, a, strangeD[j], a))
      x_obs.calc_error(1)
      data_shape=x_obs.data[0].shape
  # Each array is the weighted median over the fit ranges
      x_obs.pval[0] =x_obs.pval[0].reshape((nboot,x_obs.pval[0].shape[-1]))
      res, res_std, res_sys, data_weight = ana.sys_error(x_obs.data,x_obs.pval,par=1)
      fk = ana.draw_gauss_distributed(fk_unit[a][0],fk_unit[a][1],(1,nboot))
      # For tree level chipt the square is needed
      mk_fk[ens_AB+j] = np.divide(res[0],fk)**2
  # bootstrapsamples are available for the x-data draw from gaussian
  # distribution

  # Secondly, fit a chiral function to the data
  #define a lambda fit function
  ptfit = lambda p,x: -x/(8.*np.pi)-p*x**2/(128.*np.pi**2)
  chipt = ana.LatticeFit(ptfit,dt_i=1, dt_f=1)
  chiral_extra = chipt.chiral_fit(mk_fk,mk_akk,corrid="MA-ChiPT")
  chiral_extra.print_data()

  # Lastly, plot the fitted function and the data
  plotter = ana.LatticePlot('./plots2/pdf/chiral_ext.pdf')
  label = ["Chiral Extrapolation", r'$M_K/f_K$',
            r'$M_K a_{0}$',r'amu_s^D = 0.015']
  plotter.plot(chiral_extra,label,chiral_extra)
  


# make this script importable, according to the Google Python Style Guide
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
