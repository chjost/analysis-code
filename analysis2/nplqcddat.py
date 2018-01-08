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
from statistics import compute_error, draw_gauss_distributed

class NPLDat(object):
    """Class to hold external data from NPLQCD
       Each sampled value needs its own seed. 
    """
    def __init__(self, seeds, nboot=1500):
        self.nsamp = nboot
        # These are the calculated values from arxiv.org/0607036v1
        # Naming is after sea bare light mass
        m007 = {'mpi/fpi':(2.000, 0.017),'mk/fpi':(3.980, 0.025),
            'mu/fpi':(1.332, 0.010), 'mua0':(-0.1263, 0.0075)}
        m010 = {'mpi/fpi':(2.337, 0.011),'mk/fpi':(3.958, 0.016),
            'mu/fpi':(1.469, 0.007), 'mua0':(-0.155, 0.040)}
        m020 = {'mpi/fpi':(3.059, 0.012),'mk/fpi':(3.988, 0.015),
            'mu/fpi':(1.731, 0.007), 'mua0':(-0.213, 0.012)}
        m030 = {'mpi/fpi':(3.484, 0.010),'mk/fpi':(4.004, 0.012),
            'mu/fpi':(1.869, 0.005), 'mua0':(-0.267, 0.012)}
        self.table={'m007': m007, 'm010': m010, 'm020': m020, 'm030': m030}

        # Start with an empty dictionary
        self.data={}
        # test length of tuples
        if len(seeds) != len(self.table)*len(m007):
            raise ValueError("Seeds and Space tuples have incompatible lengths")
        # each observable should get its own seed so the desired layout would be
        # something like 
        # {'A':{'r0':{'seed':int,'boot':samples},'zp':{'seed':int,'boot':samples}}
        # for each observable and lattice spacing
        for i,beta in enumerate(self.table):
            self.data[beta]={}
            for j,obs in enumerate(self.table[beta]):
              self.data[beta][obs]={}
              self.data[beta][obs]['seed']=seeds[2*i+j]
              self.set_obs(beta,obs)
        
    def set_obs(self, a, obs):
        # From the initialized table take the desired observable of the correct
        # lattice spacing with its error
        lit = self.table[a][obs]
        seed = self.data[a][obs]['seed']
        self.data[a][obs]['boot'] = draw_gauss_distributed(lit[0],lit[1],
                                (self.nsamp,),origin=True,seed=seed)
  
    #def save
    #def save_txt
    def get(self,a,obs):
        return self.data[a][obs]['boot']

    #def show

