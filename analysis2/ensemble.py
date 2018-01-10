"""
Lattice Ensemble Class.
"""

import numpy as np
import pickle
# ConfigParser has been renamed in python 3
import ConfigParser
#import configparser

from in_out import check_read, check_write

class LatticeEnsemble(object):
    """A class for the ensemble data.

    Create with LatticeEnsemble(name, L, T). Contains a dictionary data
    which can hold arbitrary data.

    Nothing is immutable, so be careful!
    """
    def __init__(self, name, L, T):
        """Creates a LatticeEnsemble.

        Parameters
        ----------
        name : str
            Identifier of the LatticeEnsemble.
        L : int
            The spatial extent of the lattice.
        T : int
            The temporal extent of the lattice.
        """
        self.data = {}
        self.data["name"] = name
        self.data["L"] = int(L)
        self.data["T"] = int(T)
        self.data["T2"] = int(self.data["T"]/2)+1

    @classmethod
    def parse(cls, filename, verbose=False):
        """Parse an input file.
        Parameters
        ----------

        filename : str
            The name of the file.
        """
        if verbose:
            print("reading %s" % filename)
        config = ConfigParser.SafeConfigParser()
        config.read(filename)
        name = config.get("main", "name")
        T = config.getint("main", "T")
        L = config.getint("main", "L")
        if verbose:
            print("reading ensemble %s" % name)
            print("L = %d, T = %d" % (L, T))
        if config.getboolean("main", "readold"):
            oldname = config.get("main", "olddata")
            if verbose:
                print("reading old data from %s" % oldname)
            obj = cls.read(oldname)
            obj.data["name"] = name
            obj.data["T"] = T
            obj.data["T2"] = int(T//2+1)
            obj.data["L"] = L
        else:
            obj = cls(name, L, T)
        # get all other options
        if config.has_section("strings"):
            for opt in config.options("strings"):
                obj.data.update({opt: config.get("strings", opt, 1)}) 
        if config.has_section("ints"):
            for opt in config.options("ints"):
                obj.data.update({opt: config.getint("ints", opt)}) 
        if config.has_section("floats"):
            for opt in config.options("floats"):
                obj.data.update({opt: config.getfloat("floats", opt)}) 
        if config.has_section("bools"):
            for opt in config.options("bools"):
                obj.data.update({opt: config.getboolean("bools", opt)}) 
        if config.has_section("lists"):
            for opt in config.options("lists"):
                li = config.get("lists", opt).split(",")
                obj.data.update({opt: li}) 
        if config.has_section("int_lists"):
            for opt in config.options("int_lists"):
                li = config.get("int_lists", opt).split(",")
                _li=[int(d) for d in li]
                obj.data.update({opt: _li}) 
        if config.has_section("ndarrays"):
            for opt in config.options("ndarrays"):
                li = config.get("ndarrays", opt).split(",")
                obj.data.update({opt: np.asarray(li, dtype=float)}) 
        return obj

    @classmethod
    def read(cls, _filename):
        """Read LatticeEnsemble from file.

        Parameters
        ----------
        _filename : str
            The name of the file.

        Raises
        ------
        IOError
            If file or directory not found.
        """
        # check suffix of the filename
        if not _filename.endswith(".pkl"):
            filename = "".join((_filename, ".pkl"))
        else:
            filename = _filename

        # check if folder and/or file exists
        try:
            check_read(filename)
        except IOError as e:
            raise e
        # pickle the dictionary
        else:
            with open(filename, "rb") as f: 
                data = pickle.load(f)

        # create class
        tmp = cls(data["name"], data["L"], data["T"])
        tmp.data = data
        return tmp

    def save(self, _filename):
        """Save the LatticeEnsemble to disk.

        Parameters
        ----------
        _filename : str
            The name of the file.
        """
        # check suffix of the filename
        if not _filename.endswith(".pkl"):
            filename = "".join((_filename, ".pkl"))
        else:
            filename = _filename

        # check if folder and/or file exists
        check_write(filename)
        # pickle the dictionary

        with open(filename, "wb") as f: 
            pickle.dump(self.data, f)

    def name(self):
        """Returns name of the LatticeEnsemble.

        Returns
        -------
        str
            The identifier of the LatticeEnsemble.
        """
        return self.data["name"]

    def L(self):
        """Returns spatial extend of the LatticeEnsemble.

        Returns
        -------
        int
            The spatial extent of the LatticeEnsemble.
        """
        return int(self.data["L"])

    def T(self):
        """Returns temporal extend of the LatticeEnsemble.

        Returns
        -------
        int
            The temporal extent of the LatticeEnsemble.
        """
        return int(self.data["T"])

    def T2(self):
        """Returns temporal extend of the symmetrized LatticeEnsemble.

        Returns
        -------
        int
            The temporal extent of the symmetrized LatticeEnsemble.
        """
        return int(self.data["T2"])

    def add_data(self, key, data):
        """Add data to the dictionary.

        Parameters
        ----------
        key : anything
            The key for the dictionary.
        data : anything
            The data to be saved in the dictionary.
        """
        # the check is only to print a warning if key already exists
        #if key in self.data:
        #    print("Key already in data, overwritting")
        self.data[key]=data

    def get_data(self, key):
        """Get data from the dictionary.

        Parameters
        ----------
        key : anything
            The key for the dictionary.

        Returns
        -------
        data : anything
            The data to be saved in the dictionary.
        
        Raises
        ------
        KeyError
            If key not in dictionary.
        """
        if key not in self.data:
            raise KeyError("Ensemble %s has no key '%s'" % (self.name(), key))
        return self.data[key]

    def __str__(self):
        restring = "Ensemble %s with L=%d, T=%d" % (self.name(), self.L(),
                self.T())
        if self.data:
            restring = "\n".join((restring,"Data:\n"))
            for key in self.data:
                if key in ["name", "L", "T", "T2"]:
                    continue
                restring = "".join((restring, "\t%s: " % (str(key)),
                               str(self.data[key]), "\n"))
        return restring

    def __repr__(self):
        return "[ Ensemble %s with L=%d, T=%d and %d data ]" % ( self.name(),
                self.L(), self.T(), len(self.data)-4)
