# class for the ensemble data, to handle them properly in the whole code
# inherits from object in compatibility to the Google styleguide

class LatticeEnsemble(object):
    """A class for the ensemble data.

    Create with LatticeEnsemble(name, L, T). Contains a dictionary data
    which can hold arbitrary data.

    Nothing is immutable, so be careful!
    """
    def __init__(self, name, L, T):
        self.name = name
        self.L = int(L)
        self.T = int(T)
        self.T2 = int(self.T/2)+1
        self.data = {}

    def __str__(self):
        restring = "Ensemble %s with L=%d, T=%d" % (self.name, self.L, self.T)
        if self.data:
            restring = "\n".join((restring,"Data:\n"))
            for key in self.data:
                restring = "".join((restring, "\t%s: " % (str(key)),
                               str(self.data[key]), "\n"))
        return restring

    def __repr__(self):
        return "[ Ensemble %s with L=%d, T=%d and %d data ]" % (self.name, self.L,
            self.T, len(self.data))

    def add_data(self, key, data):
        # the check is only to print a warning if key already exists
        if key in self.data:
            print("Key already in data, overwritting")
            self.data[key]=data
        else:
            self.data[key]=data

    def get_data(self, key):
        if key not in self.data:
            raise KeyError("Ensemble %s has no key '%s'" % (self.name, key))
        return self.data[key]
