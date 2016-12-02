"""
Statistical functions.
"""

import numpy as np
import itertools

from utils import *

def compute_error(data, axis=0, mean=None):
    """Calculates the mean and standard deviation of the data.

    Parameters
    ----------
    data : ndarray
        The data.
    axis : int
        The axis along which both is calculated.

    Returns
    -------
    ndarray
        The mean of the data.
    ndarray
        The standard deviation of the data.
    """
    #if axis != 0:
    #    raise ValueError("compute_error not implemented for axis = %d" % axis)
    return mean_std(data,axis=axis,mean=mean)

def acf(y,start=0,debug=0):
    """Autocorrelation function of bootstrapped data

    Implemented version is the one with more bias: Have T timesteps and B
    measurements at eacht time t of observable Y
    
    r(k) = \sum_t=1^{T} (y(t)-\bar{y})(y(t+k)-\bar{y})/(var(y))

    \bar{y} is the mean over time

    The bootstrapped data is correlated in time and normalised.

    Parameters
    ----------
    y : 3darray with shape (nsample,t,y(b,t))
    start : starting timeslice for computation of acf
    Returns
    -------
    2darray, the bootstrapsamples of the normalised autocorrelated data
    """
    y_sub = y-np.mean(y, axis=1,keepdims=True)
    if debug > 0:
        print("Subtracted data has shape")
        print y_sub[0]
    r = np.zeros((y.shape[0],y.shape[1]-start))
    # Loop over Bootstrapsamples
    for b in range(y.shape[0]):
        # Fully correlate each bootstrapsample in time with itself
        _tmp =np.correlate(y_sub[b].ravel()[start:],y_sub[b].ravel()[start:],mode='full') 
        # Take half of the full correlation of the signal with itself
        r[b] = _tmp[_tmp.size/2:]
    return r/np.sum(y_sub[:,start:]**2,axis=1)

def weighted_quantile(data, weights, quantile=0.5):
    """Compute the weighted quantile, where a fixed percentage of the sum of
    all weights lie below.

    Parameters
    ----------
    data : ndarray
        The data points the quantile is taken from.
    weights : ndarray
        The weights for each point in data. Must be of same shape 
        and have same order as data.
    quantile : float, optional
        The percentage of weights to be below the quantile. 
    Returns
    -------
    float
        The value of the weighted quantile.
    """
    ind_sorted = np.argsort(data)
    sorted_data = data[ind_sorted]
    sorted_weights = weights[ind_sorted]
    # Compute the auxiliary arrays
    Sn = np.cumsum(sorted_weights)
    Pn = (Sn-0.5*sorted_weights)/np.sum(sorted_weights)
    # Get the value of the weighted median
    return np.interp(quantile, Pn, sorted_data)

def compute_weight(data, pvals, rel=True):
    """Calculate the weight of each fit. The weight is only
    calculated on the original data.

    Parameters
    ----------
    data : ndarray
        The data for which the weight is calculated.
    pvals : ndarray
        The p-value used in the weight function.

    Returns
    -------
    weight : ndarray
    """
    # compute std/mean over bootstrap samples for every fit interval
    if rel:
        errors = np.nanstd(data, axis=0)/np.nanmean(data, axis=0)
    else:
        errors = np.nanstd(data, axis=0)
    # get the minimum of the errors
    min_err = np.amin(errors)
    # prepare storage
    weights = np.zeros((data.shape[1:]))
    # Warning playing with the exponent of the weight
    exp=2
    if weights.ndim == 1:
        for i in range(weights.shape[0]):
            weights[i] = ((1. - 2.*np.abs(pvals[0,i]-0.5)) *
                min_err/errors[i])**exp
    else:
        ranges = [[n for n in range(x)] for x in weights.shape]
        for riter in itertools.product(*ranges):
            weights[riter] = ((1. - 2.*np.abs(pvals[(0,)+riter]-0.5)) *
                min_err/errors[riter])**exp
    return weights

def sys_error(data, pvals, par=0, rel=True):
    """Calculates the statistical and systematic error of an np-array of 
    fit results on bootstrap samples of a quantity and the corresponding 
    p-values.

    Parameters
    ----------
    data : list of ndarrays
        The data for which to compute the errors. Assumes at least
        three dimensions.
    pvals : ndarray
        The p values for the data.
    par : int
        The parameter for which to calculate the errors, is applied to
        second dimension.
    rel : bool
        Decides if the relative error is used in weight calculation

    Returns
    -------
    res : ndarray
        The weighted median value on the original data
    res_std : ndarray
        The standard deviation derived from the deviation of medians on
        the bootstrapped data.
    res_syst : ndarray
        1 sigma systematic uncertainty is the difference (res-16%-quantile)
        and (84%-quantile-res) respectively
    weights : ndarray
        The calculated weights for the data.
    """
    # initialize empty arrays
    data_weight = []
    res, res_std, res_sys = [], [], []
    # loop over principal correlators
    for i, d in enumerate(data):
        # append the necessary data arrays
        #data_weight.append(np.zeros((d.shape[2:])))
        res.append(np.zeros(d.shape[0]))
        res_std.append(np.zeros((1,)))
        res_sys.append(np.zeros((2,)))

        # calculate the weight for the fit ranges
        data_weight.append(compute_weight(d[:,par], pvals[i], rel=rel))
        # using the weights, calculate the median over all fit intervals
        # for every bootstrap sample.
        for b in range(d.shape[0]):
            res[i][b] = weighted_quantile(d[b,par].ravel(),
                    data_weight[i].ravel())
        # the statistical error is the standard deviation of the medians
        # over the bootstrap samples.
        _, s = mean_std(res[i])
        res_std[i] = s
        # the systematic error is given by difference between the median 
        # on the original data and the 16%- or 84%-quantile respectively
        res_sys[i][0]=res[i][0] - weighted_quantile(d[0,par].ravel(),
                data_weight[i].ravel(), 0.16)
        res_sys[i][1]=weighted_quantile(d[0,par].ravel(), data_weight[i].ravel(),
                0.84) - res[i][0]
        # keep only the median of the original data
        #res[i] = res[i][0]
    return res, res_std, res_sys, data_weight

def sys_error_der(data, weights):
    """Calculates the statistical and systematic error of a data set
    that already has the weights calculated.

    Parameters
    ----------
    data : ndarray
        The data for which to compute the errors. Assumes three dimensions.
    weights : ndarray
        The weights of the data.

    Returns:
    res : list
        The weighted median value on the original data
    res_std : list
        The standard deviation derived from the deviation of medians on
        the bootstrapped data.
    res_syst : list
        1 sigma systematic uncertainty is the difference (res-16%-quantile)
        and (84%-quantile-res) respectively
    """
    # initialize empty arrays
    data_weight = []
    res, res_std, res_sys = [], [], []
    # loop over principal correlators
    for i, d in enumerate(data):
        # append the necessary data arrays
        data_weight.append(weights[i][0])
        res.append(np.zeros(d.shape[0]))
        res_std.append(np.zeros((1,)))
        res_sys.append(np.zeros((2,)))

        # using the weights, calculate the median over all fit intervals
        # for every bootstrap sample.
        for b in range(d.shape[0]):
            res[i][b] = weighted_quantile(d[b].ravel(), weights[i][b].ravel())
        # the statistical error is the standard deviation of the medians
        # over the bootstrap samples.
        _, s = mean_std(res[i])
        res_std[i] = s
        # the systematic error is given by difference between the median 
        # on the original data and the 16%- or 84%-quantile respectively
        res_sys[i][0] = res[i][0]-weighted_quantile(d[0].ravel(),
                weights[i][0].ravel(), 0.16)
        res_sys[i][1] = weighted_quantile(d[0].ravel(), weights[i][0].ravel(),
                0.84)-res[i][0]
    return res, res_std, res_sys, data_weight

def estimated_autocorrelation(x):
      """
      Based on these links this function estimates the autocorrelation
      http://stackoverflow.com/q/14297012/190597
      http://en.wikipedia.org/wiki/Autocorrelation#Estimation
      """
      n = len(x)
      variance = x.var()
      x = x-x.mean()
      r = np.correlate(x, x, mode = 'full')[-n:]
      assert np.allclose(r, np.array([(x[:n-k]*x[-(n-k):]).sum() for k in range(n)]))
      result = r/(variance*(np.arange(n, 0, -1)))
      return  result

def draw_weighted(vals, samples=200, seed=1227):
    """Function to draw weighted random numbers after distribution of weights
    for fit ranges

    Parameters
    ----------
    """
    # get cumulated weights from sorted weights
    vals_sort = np.sort(vals)
    vals_cum = np.cumsum(vals_sort)
    # Initialize with Bastians Seed
    np.random.seed(seed)
    # draw nbsample numbers from the interval [weigths_cum[0],weights_cum[-1]]
    a,b = vals_cum[0], vals_cum[-1]
    rnd_cum = np.sort(np.asarray([np.random.uniform(a, b) for i in range(0,
      samples)]))
    # get indices of values in rnd_cum which are nearest to the values in
    # vals_cum, i from v_{i-1} <= r < v_i
    indices = np.searchsorted(vals_cum, rnd_cum, side='right')
    vals_take = np.take(vals_sort, indices)
    return vals_take

def freq_count(arr, verb=False):
    """Get a frequency count for values in an array

    Parameters
    ----------
    vals : The values to count

    Returns
    frequencies : a 2d numpy array with the unique sorted values on the first
                  axis and the number of counts on the second axis
    """
    # intermediate uniqe array
    arr_unq = np.unique(arr)
    frequencies = np.zeros((arr_unq.shape[0],2))
    for i,v in enumerate(arr_unq):
      frequencies[i] = np.asarray([arr_unq[i],(arr == v).sum()])
    if verb == True:
      print(frequencies)
    return frequencies

def draw_gauss_distributed(mean, std, shape, origin=False, seed=None):   
    """Draw normal distributed random numbers.

    Parameters
    ----------
    mean: float, the mean of the gaussian distributed variable
    std: float, the standard deviation of the sample
    shape: 1d-tuple, the samplesize
    origin: bool, save mean as original data in 0th position
    seed: int, if given, set as seed to RandomState
    """

    if seed is None:
      np.random.seed(1227)
    else:
      np.random.seed(seed)
    # for random samples from N(\mu, \sigma^2) use
    # sigma * random(shape) + mu
    res = std * np.random.randn(*shape) + mean 
    # Optionally replace mean
    if origin is True:
      res[0] = mean
    return np.asarray(res)
if __name__ == "main":
    pass
