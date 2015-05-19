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

def extract_corr_fct(filename='', column=1, verbose=0, skipSorting=False):
    """Extracts correlation functions and resorts them.

    Reads correlation functions from the file filename. The first line of the
    file contains the number of configuration in that file, the time extent of
    the correlation function, and three further numbers.
    This function implements the format used by L. Liu's code.

    Args:
        filename: The file name.
        column: Which column to read.
        verbose: Changes the amount of information printed.
        skipSorting: Skip the sorting step at the end.

    Returns:
        A list with entries containing the correlation functions. The list is
        sorted with the time index as slow index and the configuration number
        as fast index. In case skipSorting is true, the fastest index is time
        and the slowest is the configuration number.
        The second returned variable is the number of configurations.
        The third returned variable is the time extent of the lattice.
    """
    # check if file we want to read exists
    if not os.path.isfile(filename):
        print("ERROR: " + filename + " is not a file")
        return
    else:
        # open file and parse first line with information
        if verbose:
            print("reading from " + filename)
        _f = open(filename, "r")
        _firstline = _f.readline()
        _latticedata = _firstline.split()
        if verbose:
            print("number of configs " + str(_latticedata[0]))
            print("time extent " + str(_latticedata[1]))
        nbcfg = int(_latticedata[0])
        T = int(_latticedata[1])
        corr = np.zeros((int(nbcfg*T)), dtype=float)
        # read in all correlation functions
        _re = []
        try:
            for _cfg in range(0,nbcfg):
                for _t in range(0, T):
                    _re.append(float(_f.readline().split()[column]))
        except:
            print("ERROR: while reading " + filename + ". Returning...")
        # sort the correlation functions according so that the time index
        # is the fastest index
        if not skipSorting:
            if verbose:
                print("sort correlation function")
            _t = 0
            for x in _re:
                corr[int(_t%T) * nbcfg + int(_t / T)] = x
                _t += 1
        else:
            corr = _re
        # close file
        _f.close()

    # return the correlation functions
    return corr, nbcfg, T

def write_corr_fct(data, filename, T, nbcfg, timesorted=True):
    """Write the correlation function to file.

    Expects numpy array with one axis. A sanity check is done with T and
    nbcfg.
    Can write correlation functions sorted with time as fastest index
    (timesorted=False) and functions with configuration number as fastest index
    (timesorted=True), see also extract_corr_fct.
    This function implements the format used by L. Liu's code.

    Args:
        data: The numpy array to write to the file.
        filename: The filename of the file, including the path.
        T: The time extend of the data.
        nbcfg: The number of configurations in the data.
        timesorted: If time is the fastest index, this should be False.

    Returns:
        Nothing.
    """
    # check whether enough data is in the array
    if data.shape[0] != int(T * nbcfg):
        print("ERROR: the length of data is not T*nbcfg")
        return
    # check whether file exists
    if os.path.isfile(filename):
        print(filename + " already exists, overwritting...")
    # open file for writting
    outfile = open(filename, "w")
    # write the data shape in L. Liu's format
    outfile.write(str(nbcfg) + " " + str(T) + " 0 " +
                  str(int(T/2)) + " 0\n")
    tmp=None
    # write the data points
    if timesorted:
        for _i in range(data.shape[0]):
            tmp='%d %.14f\n' % ((_i%T), (data[(_i%T)*nbcfg + (_i/T)]))
            outfile.write(tmp)
            #outfile.write(str(_i%T) + " " + str(data[(_i%T)*nbcfg + (_i/T)]) + "\n")
    else:
        for _i in range(data.shape[0]):
            tmp='%d %.14f\n' % ((_i%T), (data[_i]))
            outfile.write(tmp)
            #outfile.write(str(_i%T) + " " + str(data[_i]) + "\n")

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

def write_npy_data_ascii(data, filename):
    """Write the data to an ascii file.

    Expects numpy array with three axis.
    This function implements the format used by L. Liu's code.

    Args:
        data: The numpy array to write to the file.
        filename: The filename of the file, including the path.

    Returns:
        Nothing.
    """
    # check whether file exists
    if os.path.isfile(filename):
        print(filename + " already exists, overwritting...")
    # open file for writting
    outfile = open(filename, "w")
    # write the data shape in L. Liu's format
    outfile.write(str(data.shape[0]) + " " + str(data.shape[1]) + " 0 " +
                  str(int(data.shape[1]/2)) + " 0\n")
    # write the data points
    for _i in range(data.shape[0]):
        for _j in range(data.shape[1]):
            outfile.write(str(_j) + " ")
            for _k in range(data.shape[2]):
                outfile.write(str(data[_i, _j, _k]) + " ")
            outfile.write("\n")

def write_npy_data(data, filename, verbose=False):
    """Writes numpy data to disk.

    Args:
        data: The data dumped to disk.
        filename: The name of the file into which the data is dumped.
        verbose: Changes the amount of information written.
    """
    # get path
    _dir = os.path.dirname(filename)
    # check if path exists, if not then create it
    if not os.path.exists(_dir):
        os.mkdirs(_dir)
    if verbose:
        print("saving to file" + str(filename))
    np.save(filename, data)


def read_npy_data(filename, verbose=False):
    """Reads numpy data from disk.

    Args:
        filename: The name of the file from which the data is read
        verbose: Changes the amount of information written.
    """
    # get path
    _dir = os.path.dirname(filename)
    # check if path exists, if not raise an error
    if not os.path.exists(_dir):
        print("ERROR: could not read data from file, path not existant")
        return None
    if verbose:
        print("reading from file " + str(filename))
    data = np.load(filename)
    return data
