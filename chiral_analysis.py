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
    #quark = ens.get_data("quark")
    datadir = ens.get_data("datadir") 
    plotdir = ens.get_data("plotdir") 
    d2 = ens.get_data("d2")
  # Firstly, read in all interpolated data into one array for which the
  # fits are conducted.
    for j,a in enumerate(latA):
      mult_obs = ana.FitResult.read("%s%s/match_mk_akk_%s.npz" % (datadir, a, a))
      mult_obs.calc_error()
      print mult_obs.data[0].shape
  # Each array is the weighted median over the fit ranges, if no
  # bootstrapsamples are available for the x-data draw from gaussian
  # distribution

  # Secondly, fit a chiral function to the data

  # Lastly, plot the fitted function and the data
  


# make this script importable, according to the Google Python Style Guide
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
