import numpy as np

def weighted_quantile(data, weights, quantile):
    """Compute the weighted quantile, where a fixed percentage of the sum of
    all weights lie below.

    Args:
        data: A numpy-array of the data points the quantile is taken from.
        weights: A numpy-array containing the weights for each point in data. 
              Must be of same shape and have same order as data.
        quantile: The percentage of weights to be below the quantile. 
              0.5 is the weighted median
    """

    ind_sorted = np.argsort(data)
    sorted_data = data[ind_sorted]
    sorted_weights = weights[ind_sorted]
    # Compute the auxiliary arrays
    Sn = np.cumsum(sorted_weights)
    Pn = (Sn-0.5*sorted_weights)/np.sum(sorted_weights)
    # Get the value of the weighted median
    interpolated_quant = np.interp(quantile, Pn, sorted_data)

    return interpolated_quant

