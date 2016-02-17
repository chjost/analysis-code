#!/hadron/knippsch/Enthought/Canopy_64bit/User/bin/python
################################################################################
#
# Author: Christopher Helmes (helmes@hiskp.uni-bonn.de), Christian Jost (jost@hiskp.uni-bonn.de)
# Date:   Januar 2016
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
#   Function: Test interpolation of reduced observables 
#
################################################################################

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg') # has to be imported before the next lines
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import analysis2 as ana

def main():

    # parse the input file
    if len(sys.argv) < 2:
        ens = ana.LatticeEnsemble.parse("A40.24.ini")
    else:
        ens = ana.LatticeEnsemble.parse(sys.argv[1])

    # get data from input file
    lat = ens.name()
    latA = ens.get_data("nameb")
    #quark = ens.get_data("quark")
    datadir = ens.get_data("datadir") 
    d2 = ens.get_data("d2")
    strange = ens.get_data("strangeb")
    amu_s = ens.get_data("amu_s_b")
    obs_match = 0.0198

    print(datadir)
    # Place fit results in new empty fitresults objects
    #qm_matched = ana.FitResult.create_empty(shape1,shape1,2)
    #mk_a0_matched = ana.FitResult.create_empty(shape1,shape,1)
    for i,a in enumerate(latA):
        # Read low m
        mk_low = ana.FitResult.read("%s/%s/%s/fit_k_%s.npz" % (datadir,a,strange[0],a))
        mk_low.print_data(par=1)
        mk_low.calc_error()
        print(mk_low.weight)
        #print mk_low.pval[0].shape
        obs1 = mk_low.mult_obs_single(mk_low, "m_low_sq")
        #print mk_low.pval[0].shape
        
        #obs1 = mk_low.res_reduced(samples=200,m_a0=True)
        obs1 = obs1.res_reduced(samples = 20)

        # Read high m
        mk_high = ana.FitResult.read("%s/%s/%s/fit_k_%s.npz" % (datadir,a,strange[1],a))
        mk_high.print_data(par=1)
        mk_high.calc_error()
        print(mk_high.weight)
        obs2 = mk_high.mult_obs_single(mk_high, "m_high_sq")
        obs2 = obs2.res_reduced(samples = 20)
        
        qmatch = ana.FitResult('match',derived=True)
        qmatch.evaluate_quark_mass(amu_s,obs_match, obs1, obs2)
        qmatch.print_data()
        qmatch.save("%s/%s/match_k_%s.npz" % (datadir,a,a))
        print(qmatch.data)
        
        # Read low ma0
        mka0_low = ana.FitResult.read("%s/%s/%s/mk_akk_%s.npz" % (datadir,a,strange[0],a))
        mka0_low.print_data(par=1)
        mka0_low.calc_error()
        print(mka0_low.weight)
        obs3 = mka0_low.res_reduced(samples = 20,m_a0=True)
        print(obs3.data[0].shape)

        # Read high ma0
        mka0_high = ana.FitResult.read("%s/%s/%s/mk_akk_%s.npz" % (datadir,a,strange[1],a))
        mka0_high.print_data(par=1)
        mka0_high.calc_error()
        print(mka0_high.weight)
        obs4 = mka0_high.res_reduced(samples = 20,m_a0=True)
        print(obs4.data[0].shape)

        mka0_ipol = ana.FitResult('eval',derived=True)
        mka0_ipol.evaluate_quark_mass(amu_s,obs_match,obs3, obs4)
        mka0_ipol.print_data()
        mka0_ipol.save("%s/%s/match_mk_akk_%s.npz" % (datadir,a,a))
        
        # Make nice plots for interpolation

# make this script importable, according to the Google Python Style Guide
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass


