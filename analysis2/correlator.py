"""
A class for correlation functions.
"""

import os

import numpy as np

import in_out
import bootstrap as boot
import gevp
import functions as func

class Correlators(object):
    """Correlation function class.
    """

    def __init__(self, filename=None, column=(1,), skip=1, debug=0):
        """Reads in data from an ascii file.

        The file is assumed to have in the first line the number of
        data sets, the length of each dataset and three further numbers
        not used here. This info is used to shape the data accordingly.

        If a sequence of strings is used for filenames, a correlation
        function matrix is created, otherwise a single correlation
        function is created. This has implications for some of the
        class functions.

        Parameters
        ----------
        filename : str or sequence of str, optional
            The filename of the file.
        column : sequence, optional
            The columns that are read.
        skip : int, optional
            The number of header lines that are skipped.
        debug : int, optional
            The amount of debug information printed.

        Raises
        ------
        IOError
            If the directory of the file or the file is not found.
        ValueError
            If skip < 1 or a non-existing column is read
        """
        if skip < 1:
            raise ValueError("File is assumed to have info in first line")
        else:
            self.skip = skip
        self.debug = debug
        self.data = None
        self.matrix = None

        if filename is not None:
            if isinstance(filename, (list, tuple)):
                self.data = in_out.read_matrix(filename, column, skip, debug)
                self.matrix = True
            else:
                self.data = in_out.read_single(filename, column, skip, debug)
                self.matrix = False

        if self.data is not None:
            self.shape = self.data.shape
        else:
            self.shape = None

    @classmethod
    def read(cls, filename, debug=0):
        """Reads data in numpy format.

        If the last two axis have the same extent, it is assumed a
        correlation function matrix is read, otherwise a single
        correlation function is assumed. This has implications on some
        class functions.

        Parameters
        ----------
        filename : str
            The name of the data file.
        debug : int, optional
            The amount of debug information printed.

        Raises
        ------
        IOError
            If file or folder not found.
        """
        data = in_out.read_data(filename)
        # set the data directly
        tmp = cls()
        tmp.data = data
        tmp.shape = data.shape
        if data.shape[-2] != data.shape[-1]:
            tmp.matrix = False
        else:
            tmp.matrix = True
        return tmp

    def save(self, filename, asascii=False):
        """Saves the data to disk.
        
        The data can be saved in numpy format or plain ascii.

        Parameters
        ----------
        filename : str
            The name of the file to write to.
        asascii : bool, optional
            Toggle numpy format or ascii.
        """
        verbose = (self.debug > 0) and True or False
        if asascii:
            in_out.write_data_ascii(self.data, filename, verbose)
        else:
            in_out.write_data(self.data, filename, verbose)

    def symmetrize(self):
        """Symmetrizes the data around the second axis.
        """
        self.data = boot.sym(self.data)
        self.shape = self.data.shape

    def bootstrap(self, nsamples):
        """Creates bootstrap samples of the data.

        Parameters
        ----------
        nsamples : int
            The number of bootstrap samples to be calculated.
        """
        self.data = boot.bootstrap(self.data, nsamples)
        self.shape = self.data.shape

    def sym_and_boot(self, nsamples):
        """Symmetrizes the data around the second axis and then
        create bootstrap samples of the data

        Parameters
        ----------
        nsamples : int
            The number of bootstrap samples to be calculated.
        """
        self.data = boot.sym_and_boot(self.data, nsamples)
        self.shape = self.data.shape

    def shift(self, dt, dE=None, shift=1):
        """Shift and weight the data.

        This function only works with matrices.

        Two shifts are implemented and can be selected using the flag
        shift.
        The first is due to Dudek et al, Phys.Rev. D86, 034031 (2012).
        The second is due to Feng et al, Phys.Rev. D91, 054504 (2015).
        For the second dE must be given.

        Parameters
        ----------
        dt : int
            The amount to shift.
        dE : {None, float}, optional
            The weight factor.
        shift : {1, 2}
            Which shift to use, see above.
        """
        # if the data is not a matrix, do nothing
        if not self.matrix:
            return

        if shift == 1:
            self.data = gevp.gevp_shift_1(self.data, dt, dE, 1, self.debug)
        elif dE is None:
            raise ValueError("dE is mandatory for the second implemented shift")
        else:
            self.data = gevp.gevp_shift_1(self.data, dt, dE, 1, self.debug)
        self.shape = self.data.shape

    def gevp(self, t0):
        """Calculate the GEVP of the matrix.

        This function only works with matrices.

        Parameters
        ----------
        t0 : int
            The index of the inverted matrix.
        """
        if not self.matrix:
            return

        self.data = gevp.calculate_gevp(self.data, t0)
        self.shape = self.data.shape
        self.matrix = False

    def mass(self, usecosh=True):
        """Computes the effective mass.

        Two formulas are implemented. The standard formula is based on the
        cosh function, the alternative is based on the log function.

        Parameters
        ----------
        usecosh : bool
            Toggle between the two implemented methods.
        """
        self.data = func.compute_eff_mass(self.data, usecosh)
        self.shape = self.data.shape

    def get_data(self):
        """Returns a copy of the data.
        
        Returns
        -------
        ndarray
            Returns the saved data.
        """
        return np.copy(self.data)

if __name__ == "main":
    pass
