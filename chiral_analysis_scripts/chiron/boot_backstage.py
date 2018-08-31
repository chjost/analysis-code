import numpy as np
def para_strap(data,params):
    """ Implementation of parametric bootstrap

    Parameters
    ----------
    data : 1d np-array, mean and standard deviation of the normal distribution
           to sample
    params : dictionary of parameters to use for the sample: 
             - size, tuple giving number of samples
             - seed, optional random generator seed. Defaults to 1227
    """
    # TODO move that into a debug class
    # Try to resolve options
    if len(data.shape) > 2:
        print("Parametric Bootstrap: Too many values")
    if len(params) < 1:
        print("Parametric Bootstrap: too few parameters")
    if params['seed'] is not None:
      np.random.seed(params['seed'])
    else:
      np.random.seed(1227)
    # for random samples from N(\mu, \sigma^2) use
    # sigma * random(shape) + mu
    res = data[1] * np.random.randn(*params['size']) + data[0]
    res[0] = data[0]
    return np.asarray(res)

