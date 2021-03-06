"""
Functions for in and output
"""

from __future__ import with_statement

import os
import numpy as np
import ConfigParser as cp

def read_single(fname, column, skip, debug):
    """Read a single correlation function from file.

    Parameters
    ----------
    filename : str
        The filename of the file.
    column : sequence, optional
        The columns that are read.
    skip : int, optional
        The number of header lines that are skipped.
    debug : int, optional
        The amount of debug information printed.

    Returns
    -------
    data : ndarray
        The correlation function
    """
    verbose = (debug > 0) and True or False
    return read_data_ascii(fname, column, False, skip, verbose)

def read_vector(fname, column, skip, debug):
    """Read a correlation function matrix from files.

    Parameters
    ----------
    filename : sequence of str
        The filenames of the files.
    column : sequence, optional
        The columns that are read.
    skip : int, optional
        The number of header lines that are skipped.
    debug : int, optional
        The amount of debug information printed.

    Returns
    -------
    data : ndarray
        The correlation function
    """
    verbose = (debug > 0) and True or False
    # check length of the sequence gives a nxn matrix
    _n = len(fname)

    # read in all data
    data = []
    for f in fname:
        data.append(read_data_ascii(f, column, False, skip, verbose))
    
    # check if shape of all arrays is the same
    _rshape = data[0].shape
    for d in data:
        if d.shape != _rshape:
            raise ValueError("Some correlation functions are not compatible")
    # allocate numpy array for vector
    _rshape = data[0].shape + (_n,)
    vector = np.zeros(_rshape, dtype=float)
    # sort the data into a matrix
    for _i, d in enumerate(data):
        vector[..., _i] = d

    return vector

def inputnames(conf_file, corr_string):
    """ Function to build Correlator input names conformal with B. Knippschilds
    naming scheme
        
    The function invokes a config parser which reads the config file (config.ini) regarding the
    specified correlation functions and returns a list of input names

    Args:
        conf_file: string with path to the configuration file 
        corr_string: array of identifiers for correlation functions to be
                    read in
    Returns:
        A list of inputnames

    """
    # invoke config parser and read config file
    config = cp.SafeConfigParser()
    config.read(conf_file)
    # set section names for get operations, have to be the same in configfile
    #quarks = config.get('quarks')
    #operators = config.get('operator_lists')
    #corr_list = config.get('correlator_lists')
    # result list
    inputnames = []
    # Loop over correlation function names
    for key in config.options('correlator_lists'):
        # get value of each key splited by ':'
        tmp = config.get('correlator_lists',key)
        c0 = tmp.split(':')
        # read only functions in corr_string, sort them into q and op arrays
        if c0[0] in corr_string:
            q_list = []
            op_list = []
            for val in c0[1:]:
                if val[0] == 'Q':
                    q_list.append(config.get('quarks',val))
                elif val[0] == 'O':
                    op_list.append(config.get('operator_lists',val))
                else:
                    print("Identifier not found")
            # TODO: expand entrys in operator lists
            # join single lists together for filename
            join_q = ''.join(q_list)
            join_op = '_'.join(op_list)
            # build the filename
            corrname = c0[0]+"/"+c0[0]+"_"+join_q+"_"+join_op+".dat"
            print corrname
            inputnames.append(corrname)
    return inputnames
  

def read_matrix(fname, column, skip, debug):
    """Read a correlation function matrix from files.

    Parameters
    ----------
    filename : sequence of str
        The filenames of the files.
    column : sequence, optional
        The columns that are read.
    skip : int, optional
        The number of header lines that are skipped.
    debug : int, optional
        The amount of debug information printed.

    Returns
    -------
    data : ndarray
        The correlation function
    """
    verbose = (debug > 0) and True or False
    # check length of the sequence gives a nxn matrix
    _n = int(np.floor(np.sqrt(len(fname))))
    if _n*_n != len(fname):
        raise RuntimeError("Wrong number of files for matrix")

    # read in all data
    data = []
    for f in fname:
        data.append(read_data_ascii(f, column, False, skip, verbose))
    
    # check if shape of all arrays is the same
    _rshape = data[0].shape
    for d in data:
        if d.shape != _rshape:
            raise ValueError("Some correlation functions are not compatible")
    # allocate numpy array for matrix
    _rshape = data[0].shape + (_n,) * 2
    matrix = np.zeros(_rshape, dtype=float)
    # sort the data into a matrix
    for _i, d in enumerate(data):
        matrix[..., _i//_n, _i%_n] = d
    
    # symmetrize matrix
    # create list of all indices needed to iterate over
    sym_matrix = np.zeros_like(matrix)
    for _a in range(matrix.shape[-2]):
        for _b in range(matrix.shape[-1]):
            sym_matrix[...,_a,_b] = (matrix[...,_a,_b]+matrix[...,_b,_a])/2.

    return sym_matrix

def write_data(data, filename, verbose=False):
    """Write numpy array to binary numpy format.

    Parameters
    ----------
    filename : str
        The name of the file to write to.
    data : ndarray
        The data to write.
    verbose : bool
        Toggle info output
    """
    check_write(filename)
    if verbose:
        print("saving to file" + str(filename))
    np.save(filename, data)

def read_data(filename, verbose=False):
    """Reads numpy data from binary numpy format.

    Parameters
    ----------
    filename : str 
        The name of the file to read from.
    verbose : bool
        Toggle info output

    Returns
    -------
    data : ndarray
        The read data
    """
    try:
        check_read(filename)
    except IOError as e:
        raise e

    if verbose:
        print("reading from file " + str(filename))
    data = np.load(filename)
    return data

def write_data_ascii(data, filename, verbose=False):
    """Writes the data into a file.

    The file is written to have L. Liu's data format so that the first
    line has information about the number of samples and the length of
    each sample.

    Parameters
    ----------
    filename : str
        The name of the file to write to.
    data : ndarray
        The data to write.
    verbose : bool
        Toggle info output
    """
    # check file
    check_write(filename)
    if verbose:
        print("saving to file " + str(filename))

    # in case the dimension is 1, treat the data as one sample
    # to make the rest easier we add an extra axis
    if len(data.shape) == 1:
        data = data.reshape(1, -1)
    # init variables
    nsamples = data.shape[0]
    T = data.shape[1]
    L = int(T/2)
    # write header
    head = "%i %i %i %i %i" % (nsamples, T, 0, L, 0)
    # prepare data and counter
    #_data = data.flatten()
    _data = data.reshape((T*nsamples), -1)
    _counter = np.fromfunction(lambda i, *j: i%T,
                               (_data.shape[0],) + (1,)*(len(_data.shape)-1),
                               dtype=int)
    _fdata = np.concatenate((_counter,_data), axis=1)
    # generate format string
    fmt = ('%.0f',) + ('%.14f',) * _data[0].size
    # write data to file
    np.savetxt(filename, _fdata, header=head, comments='', fmt=fmt)

def read_data_ascii(filename, column=(1,), noheader=False, skip=0,
        verbose=False):
    """Reads in data from an ascii file.

    The file is assumed to have L. Liu's data format so that the first
    line has information about the number of samples and the length of
    each sample. This info is used to shape the data into the correct
    array format.

    Parameters
    ----------
    filename : str 
        The name of the file to read from.
    column : sequence
        The columns to read.
    noheader : bool
        Read the header.
    skip : int, optional
        The number of header lines that are skipped.
    verbose : bool
        Toggle info output

    Returns
    -------
    data : ndarray
        The read data, the dimensions depend on the number of columns
        read.
    """
    # check file
    try:
        check_read(filename)
    except IOError as e:
        raise e

    if verbose:
        print("reading from file " + str(filename))

    # check if column is sensible
    if isinstance(column, (list, tuple)):
        nbcol = len(column)
    else:
        print("column must be list or tuple and not %s" % type(column))
        os.sys.exit(-1)

    # open the file to read first line
    if not noheader:
        var = read_header(filename)
        if skip == 0:
            skip = 1
    # read in data from file, skipping the header if needed
    data = np.genfromtxt(filename, skip_header=skip, usecols=column)
    # casting the array into the right shape, sample number as first index,
    # time index as second index
    # if more than one column is read, the third axis reflects this
    if noheader:
        if nbcol > 1:
            data.shape = (-1, nbcol)
    else:
        if nbcol is 1:
            data.shape = (var[0],var[1])
        else:
            data.shape = (var[0],var[1], nbcol)
    return data

def write_data_w_err_ascii(data, error, filename, verbose=False):
    """Writes data with error to a file.

    The file is written to have L. Liu's data format so that the first
    line has information about the number of samples and the length of
    each sample.

    Parameters
    ----------
    filename : str
        The name of the file to write to.
    data : ndarray
        The data to write.
    error : ndarray
        The error of the data.
    verbose : bool
        Toggle info output
    """
    # check if both arrays have the same shape
    if data.shape != error.shape:
        print("data and error must have the dimensions")
        os.sys.exit(-2)
    # if the array is 1D treat it as one sample
    if len(data.shape) == 1:
        data = data.reshape((1,) + data.shape)
        error = error.reshape((1,) + error.shape)
    # concatenate data and array together
    data = data.reshape(data.shape + (1,))
    error = error.reshape(error.shape + (1,))
    _data = np.concatenate((data, error), axis=-1)
    # write to file
    write_data_ascii(_data, filename, verbose=verbose)

def read_data_w_err_ascii(filename, datacol=(1,), errcol=(2,), verbose=False):
    """Reads in data with error from a source file.

    The file is assumed to have L. Liu's data format so that the first
    line has information about the number of samples and the length of
    each sample. This info is used to shape the data into the correct
    array format.

    Parameters
    ----------
    filename : str 
        The name of the file to read from.
    column : tuple
        The columns to read.
    noheader : bool
        Read the header.
    verbose : bool
        Toggle info output

    Returns
    -------
    data : ndarray
        The read data, the dimensions depend on the number of columns
        read.
    error : ndarray
        The error of the data, same dimension as the data.
    """
    # check if columns are sensible
    cols = datacol + errcol
    if not isinstance(cols, (list, tuple)):
        print("column must be list or tuple and not %s" % type(column))
        os.sys.exit(-1)
    # read in data
    _data = read_data_ascii(filename, column=cols, verbose=verbose)
    return _data[:,:,:len(datacol)], _data[:,:,len(datacol):]

def write_fitresults(filename, data, fitint, par, chi2, pvals, label,
    verbose=False):
    """Writes the fitresults to a numpy file.

    The function takes lists of numpy arrays and writes them in the
    npz format.

    Parameters
    ----------
    filename : str
        The name of the file.
    data : ndarray
        Additional info about the fit.
    fitint : ndarray
        The fit intervals used.
    par : ndarray
        The results of the fit.
    chi2 : ndarray
        The chi^2 values of the fit.
    pvals : ndarray
        The p-values of the fit
    label : ndarray
        The labels of the fit.
    verbose : bool
        Toggle info output
    """
    # check file
    check_write(filename)
    if verbose:
        print("saving to file " + str(filename))
    # NOTE: The dictionary building is necessary to recover the data with the
    # corresponding read function. Furthermore the read function needs to know
    # the level of recursion to build. Thus it is necessary that all labels
    # except for the fit intervals are 2 characters plus their two digit index 
    # for each level of recursion. The name of the fit intervals is chosen to
    # be always shorter than the other names.
    dic = {"data": data}
    dic.update({'fi': fitint})
    dic.update({'pi%02d' % i: p for (i, p) in enumerate(par)})
    dic.update({'ch%02d' % i: p for (i, p) in enumerate(chi2)})
    dic.update({'pv%02d' % i: p for (i, p) in enumerate(pvals)})
    dic.update({'la%02d' % i: p for (i, p) in enumerate(label)})
    np.savez(filename, **dic)

def read_fitresults(filename, verbose=False):
    """Reads the fit results from file.

    Parameters
    ----------
    filename : str
        The name of the file.
    verbose : bool
        Toggle info output
    
    Returns
    -------
    data : ndarray
        Additional info about the fit.
    fitint : ndarray
        The fit intervals used.
    par : ndarray
        The results of the fit.
    chi2 : ndarray
        The chi^2 values of the fit.
    pvals : ndarray
        The p-values of the fit
    fitint2 : ndarray, optional
        Additional fit ranges, if saved.
    """
    # check filename
    try:
        check_read(filename)
    except IOError as e:
        raise e

    if verbose:
        print("reading from file " + str(filename))
    f = np.load(filename)
    #with np.load(filename) as f:
    # check the number of levels to build
    # The array names are  2 characters plus the two digit index of the
    # for each level. The name of the fit intervals is only 2 characters
    # to be able to treat it different to the rest
    if verbose:
        print("reading %d items" % len(f.files))
    L = f.files
    n = (len(L) - 2) // 4
    par, chi2, pvals = [], [], []
    fitint, label = [], []
    label = []
    data = f['data']
    fitint = f['fi']
    for i in range(n):
        par.append(f['pi%02d' % i])
        chi2.append(f['ch%02d' % i])
        pvals.append(f['pv%02d' % i])
        label.append(f['la%02d' % i])
    f.close()
    return data, fitint, par, chi2, pvals, label

def read_header(filename, verbose=False):
    """Parses the first line of the data format of L. Liu.

    It is not verified that the file exists or is readable. This has to be
    done before calling this function.

    Parameters
    ----------
    filename : str
        The name of the file

    Returns
    -------
    tuple of int
        The read data containing 5 ints

    Returns:
        The 5 numbers of the first line. These are number of samples, time
        extent, ?, spatial extent and ?.
    """
    with open(filename, "r") as _f:
        _data = _f.readline().split()
    ret = (int(_data[0]), int(_data[1]), int(_data[2]), int(_data[3]),
           int(_data[4]))
    if verbose:
        print("number of samples: %i" % ret[0])
        print("time extent:       %i" % ret[1])
        print("spatial extent:    %i" % ret[3])
        print("number 3:          %i" % ret[2])
        print("number 5:          %i" % ret[4])
    return ret

def write_header(outfile, data):
    """Writes the first line of the data format of L. Liu.

    Parameters
    ----------
    outfile : file object
        The file to write to.
    data : tuple of int
        tuple containing 5 ints
    """
    outstring = ", ".join(data)
    outfile.write(outstring)

def check_read(filename):
    """Do some checks before opening a file.

    Parameters
    ----------
    filename : str
        The name of the file.

    Raises
    ------
    IOError
        If folder or file not found.
    """
    # get path
    _dir = os.path.dirname(filename)
    # check if path exists, if not raise an error
    if _dir and not os.path.exists(_dir):
        raise IOError("directory %s not found" % _dir)
    # check whether file exists
    if not os.path.isfile(filename):
        raise IOError("file %s not found" % os.path.basename(filename))

def check_write(filename, verbose=False):
    """Do some checks before writing a file.

    Checks if the folder containing the file exists and create it if
    if does not.

    Parameters
    ----------
    filename : str
        The name of the file
    verbose : bool
        Toggle info output
    """
    # check if path exists, if not then create it
    _dir = os.path.dirname(filename)
    if not os.path.exists(_dir):
        os.mkdir(_dir)
    # check whether file exists
    if os.path.isfile(filename):
        if verbose:
            print(filename + " already exists, overwritting...")

def inputnames(conf_file, corr_string):
    """ Function to build Correlator input names conformal with B. Knippschilds
    naming scheme
        
    The function invokes a config parser which reads the config file (config.ini) regarding the
    specified correlation functions and returns a list of input names

    Args:
        conf_file: string with path to the configuration file 
        corr_string: array of identifiers for correlation functions to be
                    read in
    Returns:
        A list of inputnames

    """
    # invoke config parser and read config file
    config = cp.SafeConfigParser()
    config.read(conf_file)
    # set section names for get operations, have to be the same in configfile
    #quarks = config.get('quarks')
    #operators = config.get('operator_lists')
    #corr_list = config.get('correlator_lists')
    # result list
    inputnames = []
    # Loop over correlation function names
    for key in config.options('correlator_lists'):
        # get value of each key splited by ':'
        tmp = config.get('correlator_lists',key)
        c0 = tmp.split(':')
        # read only functions in corr_string, sort them into q and op arrays
        if c0[0] in corr_string:
            q_list = []
            op_list = []
            for val in c0[1:]:
                if val[0] == 'Q':
                    q_list.append(config.get('quarks',val))
                elif val[0] == 'O':
                    op_list.append(config.get('operator_lists',val))
                else:
                    print("Identifier not found")
            # TODO: expand entrys in operator lists
            # join single lists together for filename
            join_q = ''.join(q_list)
            join_op = '_'.join(op_list)
            # build the filename
            corrname = c0[0]+"/"+c0[0]+"_"+join_q+"_"+join_op+".dat"
            print corrname
            inputnames.append(corrname)
    return inputnames

def _read_corr(_name, _T=48):
  """ Uses numpy's openfile function to read in binary data and reshape it to
  pairs of complex numbers

  Args: 
      _name: The file's name (best constructed in advance)
      _T: The lattice's time extent defaults to 48

  Returns: 
      corr: The read in reshaped correlation function as T,2 Array
  """
  # C like array order is implied
  tmp = np.fromfile(_name,dtype=float)
  corr = tmp.reshape((_T,2))
  return corr

def read_confs(path,corrname,confs,_T=48,verb=False):
  """ Wrapper to read in correlationfunctions of several configurations

      This file assumes B. Knippschilds binary file layout with all the
      Correlator information in the filename
      Args: 
           path: The path to the data to read
           corrname: The name of the correlationfunction, should be built
           separately
           confs: alist of configuration folder names
           _T: temporal time extent of the functions to read in

      Returns: A numpy array holding the correlation functions. Shape is
      (nb_cfg,T,2) for real and imaginary part
  """
  C = np.zeros((len(confs),_T,2))
  for i,d in enumerate(confs):
    #Generate filename from inputlist
    if verb is True:
      print path, d, corrname
    _fname = path+d+corrname
    _C_tmp = _read_corr(_fname,_T)
    C[i] = _C_tmp
  return C


def confs_subtr(Corr1, Corr2):
  """ function to subtract two diagrams columnwise 
  
  Subtracts two correlation functions like Corr1 - Corr2

  Args:
      Corr1, Corr2: Numpy arrays of one or more correlation functions

  Returns:
      A list of three tuples containing the difference C_1(t) - C_2(t)
  """
  Cdiff = np.zeros_like(Corr1)
  for i in range(Corr1.shape[0]):
    _re = np.subtract(Corr1[i,:,0],Corr2[i,:,0])
    _im = np.subtract(Corr1[i,:,1],Corr2[i,:,1])
    Cdiff[i] = np.column_stack((_re,_im))
  return Cdiff


def conf_abs(Corr):
  """ Compute absolute value of Correlation function
  
  Gets rid of overall signs in Correlation functions

  Args:
      Corr: Numpy array of one or more correlation functions

  Returns:
      A list of three tuples containing abs(C1(t))
  """
  Cabs = np.zeros_like(Corr)
  for i in range(Corr.shape[0]):
    _re = np.absolute(Corr[i,:,0])
    _im = np.absolute(Corr[i,:,1])
    Cabs[i] = np.column_stack((_re,_im))
  return Cabs 


def confs_mult(Corr,scl):
  """ Multiply Correlation function with scalar
  

  Args:
      Corr: Numpy array of one or more correlation functions
      scl: a python scalar (float or int)

  Returns:
      A list of three tuples containing scl*C1(t)
  """
  Cabs = np.zeros_like(Corr)
  for i in range(Corr.shape[0]):
    _re = np.multiply(scl,Corr[i,:,0])
    _im = np.multiply(scl,Corr[i,:,1])
    Cabs[i] = np.column_stack((_re,_im))
  return Cabs 
