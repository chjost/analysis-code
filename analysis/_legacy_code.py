"""This file contains functions that should not be used but might be used by old code.
"""

def average_corr_fct(data, nbcfg, T):
    """Average over the set of correlation functions.

    Args:
        data: The data to average over in a numpy array.
        nbcfg: The number of configurations.
        T: The time extent of the data.

    Returns:
        A numpy array averaged over one axis.
    """
    average = np.zeros((T))
    for _t in range(T):
        average[int(_t)] = np.average(data[_t*nbcfg:(_t+1)*nbcfg])
    return average

