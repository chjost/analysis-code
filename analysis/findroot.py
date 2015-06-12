__all__ = ["root"]

import os
import numpy as np
import scipy.optimize as opt

from ._memoize import memoize
from ._calc_energies import *
from ._determinants import *
from ._singular_points import *

@memoize(500)
def root(L, mpi, a0, r0, a2, r2, d=np.array([0., 0., 0.]), irrep="A1", n=1):
    """Returns roots of the determinant equation.
    
    Args:
        L: lattice size
        mpi: lattice pion mass
        a0: scattering length for l=0
        r0: scattering radius for l=0
        a2: scattering length for l=2
        r2: scattering radius for l=2
        d: total momentum of the system
        irrep: the chosen irrep
        n: number of roots to look for

    Returns:
        The values of the roots.
    """
    #print("Entering root")
    #print(irrep, L, d, n)
    # These variables were global variables
    r_prec = 1e-10
    # setup of variables
    nroot = 0
    roots = np.zeros(n)
    # CJ: Used lamda functions to make code more compact
    if (irrep == "A1"):
        if (np.array_equal(d, np.array([0., 0., 0.]))):
            calc_det = lambda q: det000(L, mpi, a0, r0, q)
            singular_points = lambda i: float(i)
            n_interval = 5
            n_blocks = 10
        elif (np.array_equal(d, np.array([0., 0., 1.]))):
            calc_det = lambda q: det001(L, mpi, a0, r0, a2, r2, q)
            singular_points = lambda i: SinglePointsP1(mpi, L, i)
            n_interval = 6
            n_blocks = 20
        elif (np.array_equal(d, np.array([1., 1., 0.]))):
            calc_det = lambda q: det110(L, mpi, a0, r0, a2, r2, q)
            singular_points = lambda i: SinglePointsP2(mpi, L, i)
            n_interval = 7
            n_blocks = 20
        elif (np.array_equal(d, np.array([1., 1., 1.]))):
            calc_det = lambda q: det111(L, mpi, a0, r0, a2, r2, q)
            singular_points = lambda i: SinglePointsP3(mpi, L, i)
            n_interval = 7
            n_blocks = 20
        else:
            print("wrong value of dVec")
            os.sys.exit(-5)
    elif (irrep == "E"):
        if (np.array_equal(d, np.array([0., 0., 0.]))):
            calc_det = lambda q: det000_E(L, mpi, a2, r2, q)
            singular_points = lambda i: float(i)
            n_interval = 5
            n_blocks = 10
        else:
            print("wrong value of dVec")
            os.sys.exit(-5)
    elif (irrep == "T2"):
        if (np.array_equal(d, np.array([0., 0., 0.]))):
            calc_det = lambda q: det000_T2(L, mpi, a2, r2, q)
            singular_points = lambda i: float(i)
            n_interval = 5
            n_blocks = 10
        else:
            print("wrong value of dVec")
            os.sys.exit(-5)
    else:
        print("wrong irrep")

    # set up grid
    q2_range_min = np.zeros(n_interval*n_blocks)
    q2_range_max = np.zeros(n_interval*n_blocks)
    for i in range(n_interval):
        q2_min = singular_points(i) + 10e-4
        q2_max = singular_points(i+1) - 10e-4
        q2_delta = (q2_max - q2_min) / float(n_blocks)
        for j in range(n_blocks):
            q2_range_min[i*n_blocks + j] = q2_min + j * q2_delta
            q2_range_max[i*n_blocks + j] = q2_min + (j + 1) * q2_delta

    # main loop
    loop_i = 0
    while(nroot < n):
        det1 = calc_det(q2_range_min[loop_i])
        det2 = calc_det(q2_range_max[loop_i])
        if (det1 * det2 < 0):
            try:
                roots[nroot] = opt.brentq(calc_det, q2_range_min[loop_i], \
                                          q2_range_max[loop_i], disp=True)
                nroot += 1
            except RuntimeError:
                print("next loop")
        loop_i += 1
        if(loop_i == q2_range_min.shape[0]):
            print("root out of range. d = (%lf, %lf, %lf)" % (d[0], d[1], d[2]))
            os.sys.exit(-5)
    return roots

