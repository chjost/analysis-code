import pandas as pd
from boot_backstage import *

class Boot():
    def __init__(self): 
        return 

    def strap(self,method='parametric',data=np.array((2.,0.3)),
          params={'size':(20,),'seed':1227,'blength':1}):

        """ Bootstrap inputdata to a pandas dataframe

        The data is bootstrapped according to the method chosen. Returns a
        pandas Series object with the 0-th value posing the mean value for the
        analysis

        Parameters
        ----------
        method : string, decide how bootstrap should be taken,
                 implemented are:
                 'parametric','blocked','stationary','conventional'
        data : array, shape determines whether dataframe or (mean,std)
        params : dict,size, seed and blocklength of the data
        Returns
        -------
        pd.Series of bootstrapped data
        """
        methdict = {'parametric':para_strap}
                    #'blocked':block_strap,
                    #'stationary':station_strap,
                    #'conventional':bstrap}
        bdata = methdict[method](data,params) 
        return pd.Series(bdata)

