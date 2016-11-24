import numpy as np
import matplotlib
matplotlib.use('Agg') # has to be imported before the next lines
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['axes.labelsize'] = 'large'

import chiral_utils as chut
import extern_bootstrap as extboot
from fit import FitResult
from statistics import compute_error, draw_gauss_distributed
from plot_functions import plot_function

"""A class handling external data. It is meant for handling data from median
and standard deviation with a pseudobootstrap. """

class ExtDat(object):
    """Class to hold external data that only differs by the lattice spacing"""
    def __init__(self,seeds,space,nboot=1500):
        self.nsamp=nboot
        # These are the calculated values from arxiv.org/1403.4504v3
        obs_a = {'r0_a':(5.31,0.08),'zp_a':(0.529,0.007)}
        obs_b = {'r0_b':(5.77,0.06),'zp_b':(0.509,0.004)}
        obs_d = {'r0_d':(7.60,0.08),'zp_d':(0.516,0.002)}
        self.table={'a':obs_a,'b':obs_b,'d':obs_d}

        # Start with an empty dictionary
        self.data={}
        # test length of tuples
        if len(seeds) != len(space):
            raise ValueError("Seeds and Space tuples have incompatible lengths")
        # initialize the dictionary with the seeds and the values for r0/a and
        # zp
        for a in zip(space,seeds):
            self.data[a[0]]={'seed':a[1]}
            self.set_obs(a[0],'r0_'+a[0])
            self.set_obs(a[0],'zp_'+a[0])
        
    def set_obs(self, a, obs):
        # From the initialized table take the desired observable of the correct
        # lattice spacing with its error
        lit = self.table[a][obs]
        seed = self.data[a]['seed']
        self.data[a][obs] = draw_gauss_distributed(lit[0],lit[1],
                                (self.nsamp,),origin=True,seed=seed)
  
    #def save
    #def save_txt
    #def get
    #def show
          
