"""
A class for correlation functions.
"""

import os

import numpy as np

class Correlators(object):
    """Correlation function class.
    """

    def __init__(self, filename, column=(1,), skip=1, debug=0):
        """Reads in data from an ascii file.

        The file is assumed to have in the first line the number of
        data sets, the length of each dataset and three furthe numbers
        not used here. This info is used to shape the data accordingly.

        Parameters
        ----------
        filename : str
            The filename of the file.
        column : sequence
            The columns that are read.
        skip : int, optional
            The number of header lines that are skipped.
        debug : int, optional
            The amount of debug information printed.

        Raises
        ------
        IOError
            If the directory of the file or the file is not found.
        """
        self.column = column
        if skip < 1:
            raise ValueError("File is assumed to have info in first line")
        else:
            self.skip = skip
        self.debug = debug
        try:
            self._check_file_exists(filename)
        except IOError as e:
            raise e
        else:
            self.read_data_from_file(filename)

    def _check_file_exists(self, filename):
        """Check whether a file exists.

        The function first checks for the directory to exist and
        afterwards if the file exists. Raises and IOError on failure.
        """
        # get path
        _dir = os.path.dirname(filename)
        # check if path exists, if not raise an error
        if _dir and not os.path.exists(_dir):
            raise IOError("directory %s not found" % _dir)
        # check whether file exists
        if not os.path.isfile(filename):
            raise IOError("file %s not found" % os.path.basename(filename))

    def _read_header(self, filename):
        """Parses the first line of the file.
    
        Reads 5 numbers in the first line of the file. The first
        corresponds to the number of data sets, the second to the
        length of the data sets.
    
        Parameters
        ----------
        filename : str
            The name of the file.
    
        Returns
        -------
        tuple of int
            tuple of the 5 numbers of the first line.
        """
        with open(filename, "r") as _f:
            _data = _f.readline().split()
        ret = (int(_data[0]), int(_data[1]), int(_data[2]), int(_data[3]),
               int(_data[4]))
        if self.debug > 0:
            print("number of samples: %i" % ret[0])
            print("time extent:       %i" % ret[1])
            print("spatial extent:    %i" % ret[3])
            print("number 3:          %i" % ret[2])
            print("number 5:          %i" % ret[4])
        return ret

    def _read_data_from_file(self, filename):
        """Reads in data from an ascii file.

        The file is assumed to have in the first line the number of
        data sets, the length of each dataset and three furthe numbers
        not used here. This info is used to shape the data accordingly.

        Args:
            filename: The filename of the file.
            column: Which column is read.
            noheader: Skips reading of the header.
            verbose: The amount of info shown.

        Returns:
            A numpy array. In case one column is read, the array is 2D, otherwise
            the array is three dimensional.
        """

        if self.debug > 0:
            print("reading from file " + str(filename))

        # open the file to read first line
        var = read_header(filename)
        # read in data from file, skipping the header if needed
        self.data = np.genfromtxt(filename, skip_header=self.skip,
                                  usecols=self.column)
        # casting the array into the right shape, sample number as first index,
        # time index as second index
        # if more than one column is read, the third axis reflects this
        if nbcol is 1:
            self.data.shape = (var[0],var[1])
        else:
            self.data.shape = (var[0],var[1], -1)

        if self.debug < 2:
            print("data shape:")
            print(self.data.shape)


if __name__ == "main":
    import unittest
    pass
