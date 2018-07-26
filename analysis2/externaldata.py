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

"""Two classes handling external data. They are meant for handling data from median
and standard deviation with a pseudobootstrap. """

class ExtDat(object):
    """Class to hold external data that only differs by the lattice spacing
       Each sampled value needs its own seed. 
    """
    def __init__(self, seeds, space, zp_meth, nboot=1500):
        self.nsamp = nboot
        # These are the calculated values from arxiv.org/1403.4504v3
        if zp_meth == 1:
            obs_a = {'r0':(5.31,0.08),'zp':(0.529,0.007)}
            obs_b = {'r0':(5.77,0.06),'zp':(0.509,0.004)}
            obs_d = {'r0':(7.60,0.08),'zp':(0.516,0.002)}
            #obs_a = {'r0':(5.31,50*0.08),'zp':(0.529,50*0.007)}
            #obs_b = {'r0':(5.77,50*0.06),'zp':(0.509,50*0.004)}
            #obs_d = {'r0':(7.60,50*0.08),'zp':(0.516,50*0.002)}

        elif zp_meth == 2:
            obs_a = {'r0':(5.31,0.08),'zp':(0.574,0.004)}
            obs_b = {'r0':(5.77,0.06),'zp':(0.546,0.002)}
            obs_d = {'r0':(7.60,0.08),'zp':(0.545,0.002)}

        else:
          raise ValueError("Method for Z_P needs to be 1 or 2! (is: %r)" % zp_meth)
        if len(space)==4:
            self.table={'A':obs_a,'B':obs_b,'D':obs_d,'D45':obs_d}
        else:
            self.table={'A':obs_a,'B':obs_b,'D':obs_d}
        # Start with an empty dictionary
        self.data={}
        # test length of tuples
        if len(seeds) < len(space)*len(obs_a):
            raise ValueError("Seeds and Space tuples have incompatible lengths")
        # each observable should get its own seed so the desired layout would be
        # something like 
        # {'A':{'r0':{'seed':int,'boot':samples},'zp':{'seed':int,'boot':samples}}
        # for each observable and lattice spacing
        for i,beta in enumerate(space):
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
class ContDat(object):
    """Class to hold external data from continuum values
    
    The class maintains a dictionary with the independent seeds and
    bootstrapsamples for every continuum observable relevant for the analysis 
    """

    def __init__(self,seeds,zp_meth=None,nboot=1500):
        """ Initialize the class with the seeds, the method for zp and the
        number of samples for each observable.

        An empty dictionary is filled with the observable id, its random seed
        and the samples drawn from the according normal distribution. The
        original value is taken as the 0th sample

        Parameters
        ----------
        seeds: iterable of int, random seeds to generate the samples for the
               observables, should be independent 
        zp_meth: int, the method that was used in arxiv.org/1403.4504v3 to
                 determine zp, important for continuum sommer parameter. If other than 1
                 or 2 defaults to median over the 2 methods.
        nboot: int, number of samples to draw for each observable

        Returns
        -------
        Nothing
        """
        self.nsamp=nboot
        # These are the calculated values from arxiv.org/1403.4504v3
        # Only changes are in r0
        # Update physical inputs to flag review hep-lat/1607.00299
        # TODO: Check results with isospin corrected values from FLAG
        if zp_meth == 1:
            #obs = {'r0':(0.470,0.012),'mk':(494.2,0.4),'mpi_0':(134.9766,0.0006),
            #      'm_l':(3.7,0.17), 'fpi':(130.41,0.03), 'meta':(574.862,0.017),
            #      'b0':(2515,90),'l4':(4.69,0.1)}
            obs = {'r0':(0.470,0.012),'mk':(494.2,0.3),'mpi_0':(134.8,0.3),
                  'm_l':(3.72,0.13), 'fpi':(130.50,0.13), 'meta':(574.862,0.017),
                  'b0':(2515,90),'l4':(4.69,0.1)}

        elif zp_meth == 2:
            #obs = {'r0':(0.471,0.011),'mk':(494.2,0.4),'mpi_0':(134.9766,0.0006),
            #      'm_l':(3.7,0.17), 'fpi':(130.41,0.03), 'meta':(574.862,0.017),
            #      'b0':(2584,88),'l4':(4.73,0.1)}
            obs = {'r0':(0.471,0.011),'mk':(494.2,0.3),'mpi_0':(134.8,0.3),
                  'm_l':(3.63,0.12), 'fpi':(130.50,0.13), 'meta':(574.862,0.017),
                  'b0':(2584,88),'l4':(4.73,0.1)}
        
        elif zp_meth == "phys":
            #obs = {'r0':(0.471,0.011),'mk':(493.677,0.016),'mpi_0':(134.9766,0.0006),
            #    'm_l':(3.7,0.17), 'fpi':(130.41,0.03), 'meta':(574.862,0.017)}
            obs = {'r0':(0.471,0.011),'mk':(494.2,0.3),'mpi_0':(134.8,0.3),
                'm_l':(3.7,0.17), 'fpi':(130.5,0.13), 'meta':(574.862,0.017)}
        else:
            #obs = {'r0':(0.474,0.014),'mk':(494.2,0.4),'mpi_0':(134.9766,0.0006),
            #      'm_l':(3.7,0.17)}
            obs = {'r0':(0.474,0.014),'mk':(494.2,0.3),'mpi_0':(134.8,0.3),
                  'm_l':(3.7,0.17)}

        self.table = obs

        # Start with an empty dictionary
        self.data={}
        # test length of tuples
        if len(seeds) < len(obs):
            raise ValueError("Seeds and Observable tuples have incompatible lengths")
        # initialize the dictionary with one seed for every observable
        for a in zip(obs, seeds):
            self.data[a[0]]={'seed':a[1]}
            self.set_obs(a[0])
        
    def set_obs(self, obs):
        """ Set the samples of the specified observables, given the seeds are
        set

        Parameters
        ----------
        obs: string, the observable to sample.

        Returns
        -------
        Nothing
        
        """
        # From the initialized table take the desired observable of the correct
        # lattice spacing with its error
        lit = self.table[obs]
        seed = self.data[obs]['seed']
        self.data[obs]['boot'] = draw_gauss_distributed(lit[0],lit[1],
                                (self.nsamp,),origin=True,seed=seed)
  
    #def save
    #def save_txt
    def get(self,obs):
        """ Return the samples of obs"""

        return self.data[obs]['boot']

    #def show
class LatPhysDat(object):
    def __init__(self, seeds, space, zp_meth, nboot=1500):
        self.nsamp = nboot
        # These are the calculated values from arxiv.org/1403.4504v3
        # a is the lattice spacing in fm
        obs_a = {'a':(0.0885,0.0036)}
        obs_b = {'a':(0.0815,0.0030)}
        obs_d = {'a':(0.0619,0.0018)}

        self.table={'A':obs_a,'B':obs_b,'D':obs_d}

        # Start with an empty dictionary
        self.data={}
        # test length of tuples
        if len(seeds) < len(space)*len(obs_a):
            raise ValueError("Seeds and Space tuples have incompatible lengths")
        # each observable should get its own seed so the desired layout would be
        # something like 
        # {'A':{'r0':{'seed':int,'boot':samples},'zp':{'seed':int,'boot':samples}}
        # for each observable and lattice spacing
        for i,beta in enumerate(space):
            self.data[beta]={}
            for j,obs in enumerate(self.table[beta]):
              self.data[beta][obs]={}
              print(i,j)
              self.data[beta][obs]['seed']=seeds[i]
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
