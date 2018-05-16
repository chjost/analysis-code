"""
Bootstraping routines and similar routines.
"""

import os
import math
import numpy as np
    
def truncate_sb_blocks(bindices,nbmeas):
    """Convert block indices to numpy array and cyclically replace indices
    larger than nbmeas

    Parameters
    ----------
    bindices: list of lists, the untruncated block indices
    nbmeas: int, number of measurements in original sample

    Returns
    -------
    _bi_array: 1d array, the indices of the bootstrapsample
    """
    # list of lists to np array
    _bi_array = np.concatenate([np.array(i) for i in bindices])
    # delete superflouus indices
    _bi_array = np.delete(_bi_array,np.s_[nbmeas:])
    for i,el in enumerate(_bi_array):
        # replace too large indices by rest 
        if el >= nbmeas:
            _bi_array[i] = np.mod(el,nbmeas)
    return _bi_array

def build_sb_blocks(starts,lengths,nbmeas):
    """Build array of indices for stationary bootstrap by combining start
    integers and blocklenghts

    After building a list of lists of integers the block indices get truncated
    to be of length nbmeas and hold only values <= nbmeas (j(mod nbmeas))

    Parameters
    ----------
    starts: 1d array, uniformly distributed integers from the range [0,nbmeas]
    lengths: 1d array, geometrically distributed block lengths
    nbmeas: int, number of measurements in original sample

    Returns
    -------
    _bindex_array: 1d array, the measurement indices for the sample
    """
    _block_indices = []
    for s in zip(starts,lengths):
        _block_indices.append([s[0]+i for i in range(s[1])])
    _bi_array_cut = truncate_sb_blocks(_block_indices,nbmeas)
    return _bi_array_cut

def get_sb_indices(nbmeas,bl):
    """Calculate an index array for the stationary bootstrap according to
    Politis and Romano J. Am. Stat. Ass. Vol. 89, No.428 (1994) 1303-1313
    
    Parameters
    ----------
    nbmeas: int, the number of measurements in the original data

    Returns
    -------
    _indices: 1d array, holding nbmeas indices
    """
    # TODO: How to determine length for tuples?
    _tup_length = nbmeas
    # randint is half open [low,high) -> highest possible value is high-1
    _i = np.random.randint(0, nbmeas, size = _tup_length)
    #_l = 1+np.random.geometric(1./bl,size=_tup_length)
    _l = np.random.geometric(1./bl,size=_tup_length)
    _indices = build_sb_blocks(_i,_l,nbmeas)
    return _indices

def get_naive_indices(nbmeas,bl=None):
    return np.random.randint(0, nbmeas, size=nbmeas) 
    
def get_bootstrap_indices(nbmeas,method="naive",bl=None):
    function_dict = {"naive":get_naive_indices,
                     "stationary":get_sb_indices}
    return function_dict[method](nbmeas,bl)

#TODO: Stationary bootstrap is a blocking method, think of that in interface 
def bootstrap(source, nbsamples,blocking = False, bl=None, method="naive"):
    """Bootstraping of data.

    Creates nbsamples bootstrap samples of source.

    Parameters
    ----------
    source : sequence
        Data on which the bootstrap samples are created.
    nbsamples : int
        Number of bootstrap samples created.
    block: boolean
        if true data will be blocked
    bl: int 
        length of one block
    method: string, which method to use for index generation, naive and
            stationary bootstrap are implemented

    Returns
    -------
    boot : ndarray
        The (blocked) bootstrap samples.
    """
    # check for blocking
    if blocking:
        source = block(source,bl)
    # seed the random number generator
    # the seed is hardcoded to be able to recreate the samples
    # original seed
    #np.random.seed(125013)
    # Bastians seed
    np.random.seed(1227)
    # initialize the bootstrapsamples to 0.
    _rshape = list(source.shape)
    _rshape[0] = nbsamples
    boot = np.zeros(_rshape, dtype=float)
    # the first entry is the average over the original data
    boot[0] = np.mean(source, dtype=np.float64, axis=0)
    # create the rest of the bootstrap samples
    number = len(source)
    for _i in range(1, nbsamples):
        _rnd = get_bootstrap_indices(number,method,bl=bl)
        #print(_rnd)
        _sum = 0.
        for _r in range(0, number):
            _sum += source[_rnd[_r]]
        boot[_i] = _sum / float(number)
    return boot

def block(source, l = 2):
    """ Divide data into blocks and take their mean.
    
    Can be done
    before bootstrapping to check for autocorrelation
    
    Parameters
    ----------
    source: the data array with correlation functions
    l: int, length of the block, defaults to 2

    Returns
    -------
    _blocked: a numpy array that consists of means over each block
  
    """
    # Determine number of blocks
    _rshape = list(source.shape)
    #print("Blocklength is: %d " % l)
    _nb = int(np.floor(_rshape[0]/l))
    #print("Number of blocks is: %d" % _nb)
    _rshape[0] = _nb
    _blocked = np.zeros(_rshape, dtype = source.dtype)
    for i in range(_nb):
        _blocked[i] = np.mean(source[i*l:(i+1)*l],dtype = source.dtype, axis = 0)
    return _blocked

def sym_and_boot(source, nbsamples = 1000, blocking = False, bl = 1,
                 method='naive'):
    """Symmetrizes and boostraps correlation functions.

    Symmetrizes the correlation functions given in source and creates
    bootstrap samples. The data is assumed to be a numpy array with
    two dimensions. The first axis is the sample number and the second
    axis is time.

    Parameters
    ----------
    source : ndarray
        A numpy array with correlation functions
    nbsamples : int
        Number of bootstrap samples created.

    Returns:
    boot : ndarray
        The bootstrapsamples, the sample number is the first axis,
        the symmetrization is around the second axis.
    """
    print("Using method: %s" %method)
    print("Using blocklength: %d" %bl)
    _rshape = list(source.shape)
    _nbcorr = _rshape[0]
    _T = _rshape[1]
    _rshape[0] = nbsamples
    _rshape[1] = int(_T/2)+1

    # initialize the bootstrap samples to 0.
    boot = np.zeros(_rshape, dtype=float)
    # the first timeslice is not symmetrized
    boot[:,0] = bootstrap(source[:,0], nbsamples, blocking = blocking, bl = bl,
                          method=method)
    for _t in range(1, int(_T/2)):
        # symmetrize the correlation function
        _symm = (source[:,_t] + source[:,(_T - _t)]) / 2.
        # bootstrap the timeslice
        boot[:,_t] = bootstrap(_symm, nbsamples,blocking = blocking, bl=bl,method=method)
    # the timeslice at t = T/2 is not symmetrized
    boot[:,-1] = bootstrap(source[:,int(_T/2)], nbsamples,
                           blocking = blocking, bl = bl, method=method)
    return boot


#TODO: Merge with sym at some point
def asym(source):
    """antisymmetrizes correlation functions.

    Antisymmetrizes the correlation functions given in source. The data is
    assumed to be a numpy array with at least two dimensions. The
    antisymmetrization is done about the second axis.
    Parameters
    ----------
    source : ndarray
        The data to antisymmetrize.

    Returns
    -------
    symm : ndarray
        The antisymmetrized data
    """
    _rshape = list(source.shape)
    _T = _rshape[1]
    _rshape[1] = int(_T/2)+1

    # initialize symmetrized data to 0.
    asymm = np.zeros(_rshape, dtype=float)
    # the first timeslice is not symmetrized
    asymm[:,0] = source[:,0]
    for _t in range(1, int(_T/2)):
        # symmetrize the correlation function
        asymm[:,_t] = (source[:,_t] - source[:,(_T - _t)]) / 2.
    # the timeslice at t = T/2 is not symmetrized
    asymm[:,-1] = source[:, int(_T/2)]
    return asymm

def sym(source, blocking=None, bl=1):
    """Symmetrizes correlation functions.

    Symmetrizes the correlation functions given in source. The data is
    assumed to be a numpy array with at least two dimensions. The
    symmetrization is done about the second axis.
    Parameters
    ----------
    source : ndarray
        The data to symmetrize.

    Returns
    -------
    symm : ndarray
        The symmetrized data
    """
    if blocking is not None:
        source=block(source,bl)
    _rshape = list(source.shape)
    _T = _rshape[1]
    _rshape[1] = int(_T/2)+1
    
    # initialize symmetrized data to 0.
    symm = np.zeros(_rshape, dtype=float)
    # the first timeslice is not symmetrized
    symm[:,0] = source[:,0]
    for _t in range(1, int(_T/2)):
        # symmetrize the correlation function
        symm[:,_t] = (source[:,_t] + source[:,(_T - _t)]) / 2.
    # the timeslice at t = T/2 is not symmetrized
    symm[:,-1] = source[:, int(_T/2)]
    return symm

