#!/hadron/knippsch/Enthought/Canopy_64bit/User/bin/python
##!/usr/bin/python

import numpy as np
import zeta

# Test case for the zeta function

def main():
    try:
        # init variables
        q2 = 1e-3
        gamma = 1.
        d = np.array([0., 0., 0.])
        lambd = 1.
        for l in range(4,5):
            print("\nl = %d" % l)
            #for m in range(-l, l+1):
            for m in range(0, 1):
                val = zeta.Z(q2, gamma, l, m, d, lambd, verbose=1)
                print("%.16lf %.16lf" % (val.real, val.imag))

    except KeyboardInterrupt:
        print("\nKeyboard Interrupt, exiting...")

# make this script importable, according to the Google Python Style Guide
if __name__ == '__main__':
    main()
