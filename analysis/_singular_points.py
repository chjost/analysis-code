import numpy as np

from ._memoize import memoize

@memoize(200)
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
        print("SinglePoints only calculates the lowest 7 points for MF1")
        os.sys.exit(-5)
    elif (P == 2 or P == 3) and (N > 7):
        print("SinglePoints only calculates the lowest 8 points for MF%1d" % P)
        os.sys.exit(-5)
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
    E = np.sqrt(W*W - 4. * piL)
    q2 = ((E*E - 2. * mpi2)*(E*E - 2. * mpi2) - 4. * mpi2*mpi2) / (4. * E*E)
    return q2 * float(L)*float(L) / (4. * np.pi*np.pi)

@memoize(100)
def SinglePointsP1(mpi, L, N):
    """Calculates the singular points for first moving frame.
    Only calculates up to N=6.

    Args:
        mpi: lattice pion mass
        L: lattice size
        N: the singular point to be calculated 

    Returns:
        nth singular point
    """
    if (N > 6):
        print("SinglePointsP1 only calculates the lowest 7 points")
        os.sys.exit(-5)
    # lowest momenta squared for two free particles for total momentum 1
    k1 = np.array([0. ,1., 1., 2., 2., 3., 4.])
    k2 = np.array([1. ,2., 4., 3., 5., 6., 5.])
    piL = np.pi*np.pi / (float(L) * float(L))
    mpi2 = mpi*mpi
    W = np.sqrt(k1[N] * 4. * piL + mpi2) + np.sqrt(k2[N] * 4. * piL + mpi2)
    E = np.sqrt(W*W - 4. * piL)
    q2 = ((E*E - 2. * mpi2)*(E*E - 2. * mpi2) - 4. * mpi2*mpi2) / (4. * E*E)
    return q2 * float(L)*float(L) / (4. * np.pi*np.pi)

@memoize(100)
def SinglePointsP2(mpi, L, N):
    """Calculates the singular points for second moving frame
    Only calculates up to N=7.

    Args:
        mpi: lattice pion mass
        L: lattice size
        N: the singular point to be calculated 

    Returns:
        nth singular point
    """
    if (N > 7):
        print("SinglePointsP2 only calculates the lowest 8 points")
        os.sys.exit(-5)
    # lowest momenta squared for two free particles for total momentum 2
    k1 = np.array([0. ,1., 1., 2., 1., 2., 2., 3.])
    k2 = np.array([2. ,1., 3., 2., 5., 4., 6., 5.])
    piL = np.pi*np.pi / (float(L) * float(L))
    mpi2 = mpi*mpi
    W = np.sqrt(k1[N] * 4. * piL + mpi2) + np.sqrt(k2[N] * 4. * piL + mpi2)
    E = np.sqrt(W*W - 2. * 4. * piL)
    q2 = ((E*E - 2. * mpi2)*(E*E - 2. * mpi2) - 4. * mpi2*mpi2) / (4. * E*E)
    return q2 * float(L)*float(L) / (4. * np.pi*np.pi)

@memoize(100)
def SinglePointsP3(mpi, L, N):
    """Calculates the singular points for third moving frame
    Only calculates up to N=7.

    Args:
        mpi: lattice pion mass
        L: lattice size
        N: the singular point to be calculated 

    Returns:
        nth singular point
    """
    if (N > 7):
        print("SinglePointsP3 only calculates the lowest 8 points")
        os.sys.exit(-5)
    # lowest momenta squared for two free particles for total momentum 3
    k1 = np.array([0. ,1., 1., 2., 3., 2., 3., 5.])
    k2 = np.array([3. ,2., 6., 5., 4., 9., 8., 6.])
    piL = np.pi*np.pi / (float(L) * float(L))
    mpi2 = mpi*mpi
    W = np.sqrt(k1[N] * 4. * piL + mpi2) + \
        np.sqrt(k2[N] * 4. * piL + mpi2)
    E = np.sqrt(W*W - 3. * 4. * piL)
    q2 = ((E*E - 2. * mpi*mpi)*(E*E - 2. * mpi*mpi) - 4. * mpi2*mpi2) / (4.*E*E)
    return q2 * float(L)*float(L) / (4. * np.pi*np.pi)

