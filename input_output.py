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
# Function: read in correlation functions from files. Two formats are
#           implemented at the moment.
#
# For informations on input parameters see the description of the function.
#
################################################################################

import os
import numpy as np

def extract_corr_fct(filename='', Im=False, verbose=0):
    """Extracts correlation functions and resorts them.

    Reads correlation functions from the file filename. The first line of the
    file contains the number of configuration in that file, the time extent of
    the correlation function, and three further numbers.
    This function implements the format used by L. Liu's code.

    Args:
        filename: The file name.
        Im: Flag whether to read the real or imaginary part of the correlation
            function.
        verbose: Changes the amount of information printed.

    Returns:
        A list with entries containing the correlation functions.
        The list is sorted with the time index as fast index and the
        configuration number as slower index.
        The number of configurations.
        The time extent of the lattice.
    """
    # corr contains the correlation functions
    #corr = []

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
        # Check whether to read the real or imaginary part of the correlation
        # function. If the part that is read does not exist, the function
        # returns.
        _arg_corr=1
        if Im:
            _arg_corr=2
        # read in all correlation functions
        _re = []
        try:
            for _cfg in range(0,nbcfg):
                for _t in range(0, T):
                    _re.append(float(_f.readline().split()[_arg_corr]))
        except:
            print("ERROR: while reading " + filename + ". Returning...")
        # sort the correlation functions according so that the time index
        # is the fastest index
        if verbose:
            print("sort correlation function")
        _t = 0
        for x in _re:
            corr[int(_t%T) * nbcfg + int(_t / T)] = x
            _t += 1
        # close file
        _f.close()

    # return the correlation functions
    return corr, nbcfg, T

def write_corr_fct(data, filename, T, nbcfg):
    """Write the correlation function to file.

    Expects numpy array with one axis. A sanity check is done with T and
    nbcfg.

    Args:
        data: The numpy array to write to the file.
        filename: The filename of the file, including the path.
        T: The time extend of the data.
        nbcfg: The number of configurations in the data.

    Returns:
        Nothing.
    """
    # check whether file exists
    if os.path.isfile(filename):
        print(filename + " already exists, overwritting...")
    # open file for writting
    outfile = open(filename, "w")
    # write the data shape in L. Liu's format
    outfile.write(str(nbcfg) + " " + str(T) + " 0 " +
                  str(int(T/2)) + " 0\n")
    # check whether enogh data is in the array
    if data.shape[0] != int(T * nbcfg):
        print("ERROR: the length of data is not T*nbcfg")
        return
    # write the data points
    for _i in range(data.shape[0]):
        outfile.write(str(_i%T) + " " + str(data[_i]) + "\n")

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
    for x in range(start_cfg, start_cfg+delta_cfg*nb_cfg, delta_cfg):
        _filename = name + "%04d" % x + '.dat'
        _f = open(filename, "rb") # Open a file
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

def write_data(data, filename):
    """Write the data to file.

    Expects numpy array with three axis.

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

