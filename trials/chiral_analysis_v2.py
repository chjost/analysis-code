import numpy as np
import pandas as pd
import chirana_data_handling as data_handler
"""Class for interfacing chiral analyses. Its usecases are quark mass
interpolations, chiral and continuum extrapolations
"""

class ChirAna(object):
    """ChirAna can be used to conduct a chiral analysis in quark masses, lattice
    spacing and chiral behaviour.

    Data are stored internally as a pandas dataframe. For the output hdf5 is
    used. A hdf5 file holds the original data, fitresults and covariance
    matrices, each in one dataset.
    """

    def __init__(self, proc_id, observable_names):
        """Initialize an empt ChirAna object with a process identifier and the
        names of the initial observables"""

        self.process_id = proc_id
        # dataframe for the bootstrapsamples to store
        self.data = pd.DataFrame(columns = observable_names)
        self.covariance_matrix = None
        self.fitresult = None

    def add_data(self, measured_data, meta_data):
        """Add measurement to an instance of ChirAna

        The measured data is added to a temporary dataframe alongside the
        metadata. Afterwards it gets appended to self.data

        Parameters
        ----------
        measured_data: dictionary of the data, key and bootstrapsamples
        meta_data: dictionary of describing data, key and data (not necessarily
        bootstrapped)
        """
        _dataframe = pd.DataFrame(data=measured_data)
        _meta_data = data_handler.extend_data(meta_data,_dataframe.index)
        _metaframe = pd.DataFrame(data=_meta_data)
        _dataframe = _dataframe.join(_metaframe,how='left')
        self.data = self.data.append(_dataframe)

