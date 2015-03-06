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
# Function: Implements functions for manipulating correlation function matrices.
#
# For informations on input parameters see the description of the function.
#
################################################################################

import os
import numpy as np
import input_output as io
import bootstrap

def create_corr_matrix(nbsamples, filepath, filestring, filesuffix=".dat",
                       column=1, verbose=0):
    """Creates a correlation function matrix.

    Reads different correlation functions and inserts them into a matrix. The
    matrix is filled row majored, the correlation functions matrix is stored
    column majored. It is assumed that the matrix is symmetric, the off
    diagonal elements are symmetrized.
    WARNING: Up to now a maximum matrix size of 20x20 operators is implemented.

    Args:
        nbsamples: Number of bootstrap samples created.
        filepath: The path to the data, including the fileprefix.
        filestring: A list of the changing parts of the filenames. The length
                    of the list gives the size of the matrix.
        filesuffix: The suffix of the data files.
        column: The column of the input file to be read. The same column is
                read from every file!
        verbose: Changes the amount of information printed.

    Returns:
        A numpy array with four axis. The first axis is the column of operators,
        the second axis is the row of operators, the third axis is the number
        of the bootstrap sample and the last axis is the time dependence.
        The second returned argument is the time extend of the correlation
        function matrix, that is (T/2)+1, where T is the length of the original
        correlation function in time.
    """
    # calculate the matrix size based on the number of elements of filestring
    # at the moment up to 20x20 matrices are supported
    _nbops = 0
    for _i in range(1, 20):
        if len(filestring) == _i*_i:
            _nbops = _i
            break
    # if the size could not be determined then return
    if _nbops == 0:
        print("ERROR: size of the correlation matrix could not be determined")
        return
    # Treat first element differently so we can create an array of the correct
    # size. This allows also to check whether the other files have the same
    # number of configurations and the same time extent.
    _name = filepath + filestring[0] + filesuffix
    if verbose:
        print("filename " + _name)
    _data1, _nbcfg1, _T1 = io.extract_corr_fct(_name, column, verbose)
    _boot1 = bootstrap.sym_and_boot(_data1, _T1, _nbcfg1, nbsamples)
    # create correlation function matrix
    corr_mat = np.zeros((nbsamples, int(_T1/2)+1, _nbops, _nbops))
    corr_mat[:,:,0,0] = _boot1
    # read in all other correlation functions, bootstrap them and write them to
    # the numpy array
    for _nb, _sub in enumerate(filestring[1:], start=1):
        # generate filename
        _name = filepath + _sub + filesuffix
        if verbose:
            print("filename " + _name)
        # read in data
        _data, _nbcfg, _T = io.extract_corr_fct(_name, column)
        # check if size is the same as the first operator
        if _nbcfg != _nbcfg1 or _T != _T1:
            print("ERROR while reading file " + _name)
            print("\tnumber of configurations or time extent is wrong")
        else:
            _boot = bootstrap.sym_and_boot(_data, _T, _nbcfg, nbsamples)
            corr_mat[:, :, int(_nb/_nbops), int(_nb%_nbops)] = _boot 

    corr_mat_symm = np.zeros_like(corr_mat)
    for _s in range(0, nbsamples):
        for _t in range(0, int(_T1/2)+1):
            corr_mat_symm[_s, _t] = (corr_mat[_s, _t] + corr_mat[_s, _t].T) / 2.
    return corr_mat_symm, int(_T1/2)+1

def write_corr_matrix(data, filename, verbose=0):
    """ Writes the correlation matrix in numpy format.

    The correlation matrix is dumped to disk using numpy's data format.

    Args:
        data: The data dumped to disk.
        filename: The name of the file into which the data is dumped.
        verbose: Changes the amount of information written.

    Returns:
        Nothing.
    """
    # get path
    _dir = os.path.dirname(filename)
    # check if path exists, if not then create it
    if not os.path.exists(_dir):
        os.mkdirs(_dir)
    if verbose:
        print("saving to file" + str(filename))
    np.save(filename, data)


def read_corr_matrix(filename, verbose=0):
    """ Reades a correlation matrix in numpy format.

    The correlation matrix stored in numpy's data format is read from disk.

    Args:
        filename: The name of the file that is read.
        verbose: Changes the amount of information written.

    Returns:
        The correlation matrix read from file.
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
