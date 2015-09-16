# class for the ensemble data, to handle them properly in the whole code
# inherits from object in compatibility to the Google styleguide

import pickle

from input_output import check_read, check_write

class LatticeEnsemble(object):
    """A class for the ensemble data.

    Create with LatticeEnsemble(name, L, T). Contains a dictionary data
    which can hold arbitrary data.

    Nothing is immutable, so be careful!
    """
    def __init__(self, name, L, T):
        self.data = {}
        self.data["name"] = name
        self.data["L"] = int(L)
        self.data["T"] = int(T)
        self.data["T2"] = int(self.data["T"]/2)+1

    @classmethod
    def from_file(cls, _filename):
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
            with open(filename, "r") as f: 
                data = pickle.load(f)

        # create class
        tmp = cls(data["name"], data["L"], data["T"])
        tmp.data = data
        return tmp

    def __str__(self):
        restring = "Ensemble %s with L=%d, T=%d" % (self.data["name"],
            self.data["L"], self.data["T"])
        if self.data:
            restring = "\n".join((restring,"Data:\n"))
            for key in self.data:
                if key in ["name", "L", "T", "T2"]:
                    continue
                restring = "".join((restring, "\t%s: " % (str(key)),
                               str(self.data[key]), "\n"))
        return restring

    def __repr__(self):
        return "[ Ensemble %s with L=%d, T=%d and %d data ]" % (
            self.data["name"], self.data["L"], self.data["T"], 
            len(self.data)-4)

    def add_data(self, key, data):
        # the check is only to print a warning if key already exists
        if key in self.data:
            #print("Key already in data, overwritting")
            self.data[key]=data
        else:
            self.data[key]=data

    def get_data(self, key):
        if key not in self.data:
            raise KeyError("Ensemble %s has no key '%s'" % (self.name, key))
        return self.data[key]

    def save(self, _filename):
        # check suffix of the filename
        if not _filename.endswith(".pkl"):
            filename = "".join((_filename, ".pkl"))
        else:
            filename = _filename

        # check if folder and/or file exists
        try:
            check_write(filename)
        except IOError as e:
            raise e
        # pickle the dictionary
        else:
            with open(filename, "w") as f: 
                pickle.dump(self.data, f)

