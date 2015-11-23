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

class ScatDat:
  """Class handling scattering data in specific order
  In current version it expects the same file layout for all files passed to it
  """
  #TODO: Implement Order checking for input ordering
  def __init__(self,_fname,_instname):
    # the data to initialize the class with
    self._tmp_in = np.loadtxt(_fname)
    # some informational stuff about the classes instance
    self.nb_meas = self._tmp_in.shape[0]

    # nb_obs counts observables
    self.nb_obs = 0
    self.mk_akk  = self._tmp_in[:,9:]
    self.nb_obs += 1
    self.mk_fk   = self._tmp_in[:,7:9]  
    self.nb_obs += 1
    self.mk      = self._tmp_in[:,3:5]  
    self.nb_obs += 1
    #self.mpi     = self._tmp_in[:,:]  
    self.fk      = self._tmp_in[:,5:7]
    self.nb_obs += 1
    
      
  def show(self):
    print("mk_fk:")
    print self.mk_fk

  def info(self):
    print self._tmp_in

def main():
  dataname = '/hiskp2/helmes/analysis/scattering/k_charged/results/ma_mk_match.dat'
  etmc = ScatDat(dataname,'etmc')
  etmc.info()
# make this script importable, according to the Google Python Style Guide
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard Interrupt")

