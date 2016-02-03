#!/hadron/knippsch/Enthought/Canopy_64bit/User/bin/python
################################################################################
#
# Author: Christian Jost (jost@hiskp.uni-bonn.de)
# Date:   Februar 2015
#
# Copyright (C) 2015 Christian Jost
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
# Function: A programm to interpolate the kaon decay constant to a certain value 
#
# For informations on input parameters see the description of the function.
#
################################################################################

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg') # has to be imported before the next lines
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import analysis2 as ana


def interp_fk(name, mul, mus_match):
    """ This function reads values for mk from a textfile, filters them and
    interpolates them to a given valence strange quark mass.

    Parameters:
    -----------
      name : the filename of the fk data
             Source for fk: 
             /freedisk/urbach/delphyne/4flavour/b<beta>/mu<mu_l>/result<ens>.dat
      mul : light quark mass of the ensemble
      mus_match : the value to evaluate interploation

    Returns:
    --------
      fk_ipol, dfk_ipol : the value and the error of the interpolated decay constant
    """
    #print("Input is:")
    #print("name : %s, mu_l = %f, match = %f" % (name, mul, mus_match))
    # Read in textfile for fk_values (usually placed in the data folders together
    # numpy array holding 3 strange quark masses, 3 kaon masses and 3 values
    # fk with the correlators)
    #Source for fk: /freedisk/urbach/delphyne/4flavour/b<beta>/mu<mu_l>/result<ens>.dat
    OS_fk = np.loadtxt(name, skiprows=1,
        usecols=(1,2,3,4,5,6))
    # delete everything with wrong light quark mass
    OS_fk = OS_fk[np.logical_not(OS_fk[:,0]!= mul)]

    # filter the textfile for the right light quark mass
    # make numpy arrays with x and y values
    mus = OS_fk[:,1]
    fk  = OS_fk[:,4]
    dfk = OS_fk[:,5]
    # use np.interp to interpolate to given value
    fk_ipol = np.interp(mus_match, mus, fk)
    dfk_ipol = np.interp(mus_match, mus,dfk)
    return np.array((fk_ipol, dfk_ipol))

def main():

    # parse the input file
    if len(sys.argv) < 2:
        ens = ana.LatticeEnsemble.parse("A40.24.ini")
    else:
        ens = ana.LatticeEnsemble.parse(sys.argv[1])

    # get data from input file
    lat = ens.name()
    latA = ens.get_data("namea")
    latB = ens.get_data("nameb")
    lqmA = ens.get_data("amu_l_a")
    lqmB = ens.get_data("amu_l_b")
    print lqmB
    sqmA = ens.get_data("amu_s_a")
    sqmB = ens.get_data("amu_s_b")
    print sqmB
    strangeA = ens.get_data("strangea")
    strangeB = ens.get_data("strangeb")
    #quark = ens.get_data("quark")
    datadir = ens.get_data("datadir") 
    plotdir = ens.get_data("plotdir") 
    d2 = ens.get_data("d2")
    
    for i,s in enumerate(strangeB):
      print(s)
      for j,a in enumerate(latB):
        fk_data = "%s/%s/OSfk_%s.dat" % (datadir,a,a)
        fk_use = interp_fk(fk_data, lqmB[j], sqmB[i])
        print("%f %f %f" %(lqmB[j],fk_use[0],fk_use[1]))
# make this script importable, according to the Google Python Style Guide
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
