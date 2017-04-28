#!/hadron/knippsch/Enthought/Canopy_64bit/User/bin/python

import os
import glob
import numpy as np

def main():
    topdir = "/hiskp2/ottnad/data/[ABD]*/"
    ensemble = glob.glob(topdir)
    filename = "analyse/eta_trick/conn_only/LF_4x4.blocking.tshift0.Z_M1.masses.cosh.state.0.fit.original_data"

    for e in ensemble:
        fname = os.path.join(e, filename)
        try:
            tmp = np.loadtxt(fname)
            print(os.path.basename(e[:-1]))
            print("%.5f(%.0f)" % (tmp[0], tmp[1]*1e5))
        except IOError:
            pass

# make this script importable, according to the Google Python Style Guide
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
