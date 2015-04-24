################################################################################
#
# Author: Christian Jost (jost@hiskp.uni-bonn.de)
# Date:   Februar 2015
#
# Copyright (C) 2015 Christian Jost
# 
# This program is free software: you can redistribute it and/or modify it under 
# the terms of the GNU General Public License as published by the Free Software 
# Foundation, either version 3 of the License, or (at your option) any later 
# version.
# 
# This program is distributed in the hope that it will be useful, but WITHOUT 
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS 
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with tmLQCD. If not, see <http://www.gnu.org/licenses/>.
#
################################################################################
#
# Function: Read and write correlation functions from files. Different formats
# are implemented.
#
# For informations on input parameters see the description of the function.
#
################################################################################

import os
import numpy as np

### MAIN FUNCTIONS ###
# available:
# write_data
# read_data
# write_data_ascii
# read_data_ascii
# write_data_w_err_ascii
# read_data_w_err_ascii
# extract_bin_corr_fct <- NOT TESTED

def write_data(data, filename, verbose=False):
    """Write numpy array to binary numpy format.

    Args:
        filename: The name of the file to write to.
        data: The data to write.
        verbose: Changes the amount of info.
    """
    check_write(filename)
    if verbose:
        print("saving to file" + str(filename))
    np.save(filename, data)

def read_data(filename, verbose=False):
    """Reads numpy data from binary numpy format.

    Args:
        filename: The name of the file from which the data is read
        verbose: Changes the amount of information written.
    """
    check_read(filename)
    if verbose:
        print("reading from file " + str(filename))
    data = np.load(filename)
    return data

def write_data_ascii(data, filename, verbose=False):
    """Writes the data into a file.

    The file is written to have L. Liu's data format so that the first line
    has information about the number of samples and the length of each sample.

    Args:
        filename: The filename of the file.
        data: The numpy array with data.
        verbose: The amount of info shown.
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
    savetxt(filename, _fdata, header=head, comments='', fmt=fmt)

def read_data_ascii(filename, column=(1,), verbose=False):
    """Reads in data from an ascii file.

    The file is assumed to have L. Liu's data format so that the first line
    has information about the number of samples and the length of each sample.
    This info is used to shape the data into the correct array format.

    Args:
        filename: The filename of the file.
        column: Which column is read.
        verbose: The amount of info shown.

    Returns:
        A numpy array. In case one column is read, the array is 2D, otherwise
        the array is three dimensional.
    """
    # check file
    check_read(filename)
    if verbose:
        print("reading from file " + str(filename))

    # check if column is sensible
    if isinstance(column, (list, tuple)):
        nbcol = len(column)
    else:
        print("column must be list or tuple and not %s" % type(column))
        os.sys.exit(-1)

    # open the file to read first line
    var = read_header(filename)
    # read in data from file, skipping first row (header) and only using the
    # second column as the first contains the time
    data = np.genfromtxt(filename, skip_header=1, usecols=column)
    # casting the array into the right shape, sample number as first index,
    # time index as second index
    # if more than one column is read, the third axis reflects this
    if nbcol is 1:
        data.shape = (var[0],var[1])
    else:
        data.shape = (var[0],var[1], nbcol)
    return data

def write_data_w_err_ascii(data, error, filename, verbose=False):
    """Writes data with error to a file.

    The file is written to have L. Liu's data format so that the first line
    has information about the number of samples and the length of each sample.

    Args:
        filename: The filename of the file.
        data: The numpy array with data.
        verbose: The amount of info shown.
    """
    # check if both arrays have the same shape
    # TODO(CJ): Think about broadcasting
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

    The file is assumed to have L. Liu's data format so that the first line
    has information about the number of samples and the length of each sample.
    This info is used to shape the data into the correct array format.

    Args:
        filename: The filename of the file.
        column: Which column is read.
        verbose: The amount of info shown.

    Returns:
        Two numpy arrays.
        Each array is 2D if only one column is read, otherwise the array is
        three dimensional.
    """
    # check if columns are sensible
    cols = datacol + errcol
    if not isinstance(cols, (list, tuple)):
        print("column must be list or tuple and not %s" % type(column))
        os.sys.exit(-1)
    # read in data
    _data = read_data_ascii(filename, column=cols, verbose=verbose)
    return _data[:,:,:len(datacol)], _data[:,:,len(datacol):]


#TODO(CJ): still needed?
def extract_bin_corr_fct(name='', start_cfg=0, delta_cfg=0, nb_cfg=0, T=0,
                     verbose=0): 
    """Extracts binary correlation functions and resorts them.

    Reads binary correlation functions from files with prefix name. The file
    names include the configuration numbers genereated from the *_cfg variables.
    The extent of a correlation function is given by T.

    Args:
        name: The file name prefix.
        start_cfg: First configuration number.
        delta_cfg: Increment for the configuration numbers.
        nb_cfg: Number of configurations to process.
        T: Time extend of the configurations.
        verbose: Changes the amount of information printed.

    Returns:
        A list with complex entries containing the correlation functions.
        The list is sorted with the time index as fast index and the
        configuration number as slower index.
    """
    _re = []
    _im = []
    # TODO(CJ): treat gamma
    gamma = 0
    for x in range(start_cfg, start_cfg+delta_cfg*nb_cfg, delta_cfg):
        _filename = name + "%04d" % x + '.dat'
        _f = open(_filename, "rb") # Open a file
        if verbose:
            print("reading from file: " + _f.name)
        _f.seek(2*8*T*gamma)
        for _t in range(0, T):
            _re.insert(_t, struct.unpack('d', _f.read(8))) # returns a tuple -> convers.
            _im.insert(_t, struct.unpack('d', _f.read(8))) # returns a tuple -> convers.
        _f.close(); # close the file  
    # conversion of the tuple to list and reorganise
    corr = [complex(0.0, 0.0)]*nb_cfg*T
    _t = 0
    for _x, _y in zip(_re, _im):
        corr[(_t%T)*nb_cfg + _t/T] = complex(_x[0], _y[0])
        _t += 1
    return corr

### HELPER FUNCTIONS ###

def read_header(filename, verbose=False):
    """Parses the first line of the data format of L. Liu.

    It is not verified that the file exists or is readable. This has to be
    done before calling this function.

    Args:
        filename: The name of the file.

    Returns:
        The 5 numbers of the first line. These are number of samples, time
        extent, ?, spatial extent and ?.
    """
    with open(filename, "r") as _f:
        _data = _f.readline().split()
    if(verbose):
        print("number of samples: %i" % int(_data[0]))
        print("time extent:       %i" % int(_data[1]))
        print("spatial extent:    %i" % int(_data[3]))
        print("number 3:          %i" % int(_data[2]))
        print("number 5:          %i" % int(_data[4]))
    return int(_data[0]), int(_data[1]), int(_data[2]), int(_data[3]), int(_data[4])

def write_header(outfile, nsam, T, L, x=0, y=0):
    """Writes the first line of the data format of L. Liu.

    Args:
        line: First line of the file.
        outfile: The open file to write into.
        nsam: The number of samples.
        T, L: The time and spatial extent of the lattice.
        x, y: The 3rd and 5th number.
    """
    outfile.write("%i %i %i %i %i\n" % (nsam, T, x, L, y))

def check_read(filename):
    """Do some checks before opening a file.
    """
    # get path
    _dir = os.path.dirname(filename)
    # check if path exists, if not raise an error
    if not os.path.exists(_dir):
        print("ERROR: could not read data from file, path not existent")
        os.sys.exit(-1)
    # check whether file exists
    if not os.path.isfile(filename):
        print(filename + " does not exist. Aborting...")
        os.sys.exit(-1)

def check_write(filename):
    """Do some checks before writing a file.
    """
    # check if path exists, if not then create it
    _dir = os.path.dirname(filename)
    if not os.path.exists(_dir):
        os.mkdir(_dir)
    # check whether file exists
    if os.path.isfile(filename):
        print(filename + " already exists, overwritting...")

def savetxt(fname, X, fmt='%.18e', delimiter=' ', newline='\n', header='',
                footer='', comments='# '):
    """This code is from NumPy 1.9.1. For help see there.

    It was included because features are used that were added in version 1.7
    but on some machines only NumPy version 1.6.2 is available.
    """
    ## needed for the rest
    from numpy.compat import asstr, asbytes
    def _is_string_like(obj):
        try:
            obj + ''
        except (TypeError, ValueError):
            return False
        return True

    # Py3 conversions first
    if isinstance(fmt, bytes):
        fmt = asstr(fmt)
        delimiter = asstr(delimiter)

    own_fh = False
    if _is_string_like(fname):
        own_fh = True
        if fname.endswith('.gz'):
            import gzip
            fh = gzip.open(fname, 'wb')
        else:
            if os.sys.version_info[0] >= 3:
                fh = open(fname, 'wb')
            else:
                fh = open(fname, 'w')
    elif hasattr(fname, 'write'):
        fh = fname
    else:
        raise ValueError('fname must be a string or file handle')

    try:
        X = np.asarray(X)

        # Handle 1-dimensional arrays
        if X.ndim == 1:
            # Common case -- 1d array of numbers
            if X.dtype.names is None:
                X = np.atleast_2d(X).T
                ncol = 1

            # Complex dtype -- each field indicates a separate column
            else:
                ncol = len(X.dtype.descr)
        else:
            ncol = X.shape[1]

        iscomplex_X = np.iscomplexobj(X)
        # `fmt` can be a string with multiple insertion points or a
        # list of formats.  E.g. '%10.5f\t%10d' or ('%10.5f', '%10d')
        if type(fmt) in (list, tuple):
            if len(fmt) != ncol:
                raise AttributeError('fmt has wrong shape.  %s' % str(fmt))
            format = asstr(delimiter).join(map(asstr, fmt))
        elif isinstance(fmt, str):
            n_fmt_chars = fmt.count('%')
            error = ValueError('fmt has wrong number of %% formats:  %s' % fmt)
            if n_fmt_chars == 1:
                if iscomplex_X:
                    fmt = [' (%s+%sj)' % (fmt, fmt), ] * ncol
                else:
                    fmt = [fmt, ] * ncol
                format = delimiter.join(fmt)
            elif iscomplex_X and n_fmt_chars != (2 * ncol):
                raise error
            elif ((not iscomplex_X) and n_fmt_chars != ncol):
                raise error
            else:
                format = fmt
        else:
            raise ValueError('invalid fmt: %r' % (fmt,))

        if len(header) > 0:
            header = header.replace('\n', '\n' + comments)
            fh.write(asbytes(comments + header + newline))
        if iscomplex_X:
            for row in X:
                row2 = []
                for number in row:
                     row2.append(number.real)
                     row2.append(number.imag)
                fh.write(asbytes(format % tuple(row2) + newline))
        else:
            for row in X:
                fh.write(asbytes(format % tuple(row) + newline))
        if len(footer) > 0:
            footer = footer.replace('\n', '\n' + comments)
            fh.write(asbytes(comments + footer + newline))
    finally:
        if own_fh:
            fh.close()
