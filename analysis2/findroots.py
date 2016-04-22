"""
Algorithms for root finding.
"""

import scipy.optimize as opt
import numpy as np
import determinants as dets

def root(L, mpi, par, p2=0, irrep="A1", n=1):
    """Find root for given parameters.

    Parameters
    ----------
    L : int
        The spatial extent of the lattice.
    mpi : ndarray
        The pion mass.
    par : ndarray
        The values for a_0, r_0, a_2, r_2.
    p2 : int
        The total momentum squared.
    irrep : str
        The irrep to look at.
    n : int
        The n lowest energy levels will be returned.

    Returns
    -------
    roots : ndarray
        The n lowest energy levels.
    """
    # These variables were global variables
    r_prec = 1e-10
    # setup of variables
    roots = np.zeros(n)
    # CJ: Used lamda functions to make code more compact
    if irrep == "A1":
        if p2 == 0:
            calc_det = lambda q: dets.det000_A1(L, mpi, par, q)
            singular_points = lambda i: float(i)
            n_interval = 5
            n_blocks = 10
        elif p2 == 1:
            calc_det = lambda q: dets.det001_A1(L, mpi, par, q)
            singular_points = lambda i: SinglePoints(mpi, L, i, 1)
            n_interval = 6
            n_blocks = 20
        elif p2 == 2:
            calc_det = lambda q: dets.det110_A1(L, mpi, par, q)
            singular_points = lambda i: SinglePoints(mpi, L, i, 2)
            n_interval = 7
            n_blocks = 20
        elif p2 == 3:
            calc_det = lambda q: dets.det111_A1(L, mpi, par, q)
            singular_points = lambda i: SinglePoints(mpi, L, i, 3)
            n_interval = 7
            n_blocks = 20
        else:
            raise NotImplementedError("For A1 irrep only p2 < 4 implemented.")
    elif irrep == "E":
        if p2 == 0:
            calc_det = lambda q: dets.det000_E(L, mpi, par, q)
            singular_points = lambda i: float(i)
            n_interval = 5
            n_blocks = 10
        else:
            raise NotImplementedError("For E irrep only p2 < 1 implemented.")
    elif (irrep == "T2"):
        if p2 == 0:
            calc_det = lambda q: dets.det000_T2(L, mpi, par, q)
            singular_points = lambda i: float(i)
            n_interval = 5
            n_blocks = 10
        else:
            raise NotImplementedError("For T2 irrep only p2 < 1 implemented.")
    else:
        raise NotImplementedError("Only A1, T2 and E irreps implemented")

    # set up grid
    q2_range_min = np.zeros(n_interval*n_blocks)
    q2_range_max = np.zeros_like(q2_range_min)
    for i in range(n_interval):
        q2_min = singular_points(i) + 10e-4
        q2_max = singular_points(i+1) - 10e-4
        q2_delta = (q2_max - q2_min) / float(n_blocks)
        for j in range(n_blocks):
            q2_range_min[i*n_blocks + j] = q2_min + j * q2_delta
            q2_range_max[i*n_blocks + j] = q2_min + (j + 1) * q2_delta

    # main loop
    nroot = 0
    for q2_min, q2_max in zip(q2_range_min, q2_range_max):
        det1 = calc_det(q2_min)
        det2 = calc_det(q2_max)
        if det1*det2 < 0.:
            try:
                roots[nroot] = opt.brentq(calc_det, q2_min, q2_max)
                nroot += 1
            except RuntimeError:
                pass
            if nroot >= n:
                break
                
    #loop_i = 0
    #while(nroot < n):
    #    det1 = calc_det(q2_range_min[loop_i])
    #    det2 = calc_det(q2_range_max[loop_i])
    #    if (det1 * det2 < 0):
    #        try:
    #            roots[nroot] = opt.brentq(calc_det, q2_range_min[loop_i], \
    #                                      q2_range_max[loop_i], disp=True)
    #            nroot += 1
    #        except RuntimeError:
    #            #print("next loop")
    #            pass
    #    loop_i += 1
    #    if(loop_i == q2_range_min.shape[0]):
    #        print("root out of range. d = (%lf, %lf, %lf)" % (d[0], d[1], d[2]))
    #        os.sys.exit(-5)
    return roots

def SinglePoints(mpi, L, N, P=1):
    """Calculates the singular points for moving frames.
    Only calculates up to N=6 for MF1 and to N=7 for MF2 and MF3.
    Higher moving frames not available yet.

    Args:
        mpi: lattice pion mass
        L: lattice size
        N: the singular point to be calculated 
        P: The squared total momentum.

    Returns:
        Nth singular point for the Pth moving frame.
    """
    if P == 1 and N > 6:
        raise RuntimeError("Only lowest 7 points for MF1 implemented")
    elif (P == 2 or P == 3) and (N > 7):
        raise RuntimeError("Only lowest 8 points for MF%d implemented" % P)
    # lowest momenta squared for two free particles
    if P == 1:
        k1 = np.array([0. ,1., 1., 2., 2., 3., 4.])
        k2 = np.array([1. ,2., 4., 3., 5., 6., 5.])
    elif P == 2:
        k1 = np.array([0. ,1., 1., 2., 1., 2., 2., 3.])
        k2 = np.array([2. ,1., 3., 2., 5., 4., 6., 5.])
    elif P == 3:
        k1 = np.array([0. ,1., 1., 2., 3., 2., 3., 5.])
        k2 = np.array([3. ,2., 6., 5., 4., 9., 8., 6.])
    piL = np.pi*np.pi / (float(L) * float(L))
    mpi2 = mpi*mpi
    W = np.sqrt(k1[N] * 4. * piL + mpi2) + np.sqrt(k2[N] * 4. * piL + mpi2)
    E = np.sqrt(W*W - P * 4. * piL)
    q2 = ((E*E - 2. * mpi2)*(E*E - 2. * mpi2) - 4. * mpi2*mpi2) / (4. * E*E)
    return q2 * float(L)*float(L) / (4. * np.pi*np.pi)

if __name__ == "__main__":
    pass
