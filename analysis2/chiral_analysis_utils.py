"""
Data Handling for ChirAna class.
"""

import numpy as np

def init_array(lyt,nboot_fr):
    """ Initialize data lists for ChirAna object

    Function creates a list of numpy arrays which get filled with lattice data

    Parameters
    ----------
    lyt : Tuple, storage layout
    nboot_fr : int, optional degree of freedom, number of fitranges

    Returns
    -------
    _data : list of ndarrays for data storage

    """
    # data will be a list
    _data = []
    # Check if fitranges are necessary
    if nboot_fr is not None:
        # loop over lattice spacing
        for a in range(lyt[0]):
            if len(lyt) == 5:
                tp = (lyt[1][a],lyt[2],lyt[3],lyt[4],nboot_fr)
            elif len(lyt) == 4:
                tp = (lyt[1][a],lyt[2],lyt[3],nboot_fr)
            else:
                tp = (lyt[1][a],lyt[2],nboot_fr)
            # append to list
            _data.append(np.zeros(tp))
    # no fitranges required
    else:
        # loop over lattice spacing
        for a in range(lyt[0]):
            if len(lyt) == 5:
                tp = (lyt[1][a],lyt[2],lyt[3],lyt[4])
            elif len(lyt) == 4:
                tp = (lyt[1][a],lyt[2],lyt[3])
            else:
                tp = (lyt[1][a],lyt[2])
            # append to list
            _data.append(np.zeros(tp))
    return _data
