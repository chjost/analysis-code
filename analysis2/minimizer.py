"""
minimizer for a global fit to data
"""

import os
import numpy as np
import scipy.optimize as opt
import multiprocessing

from memoize import memoize
import energies
import findroots


def get_Wroot(arglist):
    # local definitions
    par, mpi, s = arglist
    root = findroots.root
    EfromMpi = energies.EfromMpi
    WfromE = energies.WfromE
    index = s[-1]
    L = s[0]
    m = mpi[s[1]]
    p2 = s[3]
    irr = s[2]
    count = 2*len(index)-1
    # sanity check
    if count < 0:
        raise RuntimeWarning("no levels to check")
        #return (0, np.zeros(1))
    calc_root = lambda L, m, p2, irr, i: root(L, m, par, p2, irr, i)
    # calculation
    croot = calc_root(L, m, p2, irr, count)
    Eroot = 2.*EfromMpi(m, np.sqrt(croot), L)
    Wtmp = WfromE(Eroot, p2, L)
    return (count, Wtmp)

def chi2(par, data, mpi, cov, levels, pool=None):
    """Calculates the total chi^2 of the problem.

    Most things are hardcoded for the test data. This function might change
    a lot.

    Parameters
    ----------
    par : ndarray
        The fit parameters
    data : ndarray
        the energy data
    mpi : ndarray
        the pion masses
    cov : ndarray
        the inverse covariance matrix of the different lattice sizes
    levels : list of tuples
        The info about energy levels (lattice size, irrep, total momentum, etc.)

    Returns
    -------
    ndarray
        The total chi^2
    """
    debug = 0
    if debug > 1:
        print(par)
    _par = np.zeros(4)
    if par.size < 4:
        _par[:par.size] = par
    else:
        _par = par[:4]

    # memory for results
    Wroot = np.zeros(np.sum([len(t[-1]) for t in levels]))
    # calc all different roots parallel
    res = np.empty((len(levels),), dtype=object)
    args = [[_par, mpi, x] for x in levels]
    if debug > 2:
        print("args for the pool")
        print(args)
    if pool is None:
        raise RuntimeError("No worker pool")
    else:
        #pool = multiprocessing.Pool(4)
        res = pool.map(get_Wroot, args)
        # close pool
        #pool.close()
        #pool.terminate()
        #pool.join()
    # iterate over results
    for r,s in zip(res, levels):
        count = r[0]
        Wtmp = r[1]
        index = s[-1]
        if count == 1:
            Wroot[index[0]] = Wtmp
        elif count == 3:
            Wroot[index[0]] = Wtmp[0]
            if (np.fabs(Wtmp[0]-data[index[0]])> \
                np.fabs(Wtmp[1]-data[index[0]])):
                print("unusual first level match")
                print("iE: %i, W: %.8lf, Wroot1: %.8lf, Wroot2: %.8lf"\
                      % (index[0], data[index[0]], Wtmp[0], Wtmp[1]))
            if (np.fabs(Wtmp[1]-data[index[1]])< \
                np.fabs(Wtmp[2]-data[index[1]])):
                Wroot[index[1]] = Wtmp[1]
            else:
                Wroot[index[1]] = Wtmp[2]
            #if (np.fabs(Wtmp[1]-data[count+1])> \
            #    np.fabs(Wtmp[2]-data[count+1])):
                print("unusual second level match")
                print("iE: %i, W: %.8lf, Wroot2: %.8lf, Wroot3: %.8lf"\
                      % (index[1], data[index[1]], Wtmp[1], Wtmp[2]))
        if debug > 1:
            print("Wroot")
            print(Wtmp)
            print("data")
            if count == 1:
                print(data[index[0]])
            elif count == 3:
                print(data[index[0]])
                print(data[index[1]])
    if debug > -1:
        for i, (Edata, Ecalc) in enumerate(zip(data, Wroot)):
            print("DATA %02d: %.10lf, %.10lf, %.10e\n" % (i, Edata, Ecalc, np.abs(Edata - Ecalc)))

    # calculate chi^2
    dx = np.dot(cov, (data - Wroot))
    #if debug > 0:
    print("************")
    print(np.sum(dx**2))
    print("************")
    return dx

def minimizer(data, mpi, cov, start, h, levels, nsamples=None):
    """Minimizer for the chi^2 problem.

    Parameters
    ----------
    data : ndarray
        The data to operate on
    mpi : ndarray
        the pion masses
    cov : ndarray
        The covariance matrix of the data
    start : ndarray
        The starting values for a0, r0, a2, r2
    h : ndarray
        numpy array of length 3

    Returns
    -------
    ndarray
        The final values for a0, r0, a2, r2.
    float
        The minimal chi^2 value calculated.
    """
    # some variables
    min_tol = 1e-4
    verbose = False
    dof = float(data.shape[0] - start.size)
    if nsamples is None:
        samples = 2
        sstart = 0
    else:
        try:
            samples = nsamples[1]
            sstart = nsamples[0]
        except IndexError:
            sstart = int(nsamples)
            samples = 2
    _start = np.asarray(start).ravel()
    pool = multiprocessing.Pool(3)
        # close pool
        #pool.close()
        #pool.terminate()
        #pool.join()
    #if _start.size > 4:
    #    _start = _start[:4]
    #    _start[3] = 0.
    #elif _start.size < 4:
    #    tmp = np.zeros((4,))
    #    tmp[:_start.size] = _start
    #    _start = tmp.copy()
    #else:
    #    _start[3] = 0.
    if _start.size > 3:
        _start = _start[:3]
    #samples = data.shape[1]
    res = np.zeros((samples, _start.size))
    chisquare = np.zeros(samples)
    if sstart+samples > data.shape[1]:
        samples = data.shape[1]-sstart
    print("start = %d, samples = %d" % (sstart, samples))

    # invoke minimizer
    for b in range(samples):
        p,cov1,infodict,mesg,ier = opt.leastsq(chi2, x0=_start, args=(data[:,b+sstart],
            mpi[:,b+sstart], cov, levels, pool), ftol=min_tol, xtol=min_tol, full_output=True)
            #mpi[:,b], cov, levels), ftol=min_tol, xtol=min_tol, diag=h, full_output=True)
        chisquare[b] = float(np.sum(infodict['fvec']**2.))
        res[b] = np.asarray(p)
        print("---------------------------")
        print("sample %d, %d function calls" % (b, infodict["nfev"]))
        print("chi^2 = %e" % chisquare[b])
        print("results:")
        print("a_0 = %e, r_0 = %e" % (1./p[0], p[1]))
        print("a_2 = %e" % (1./p[2]))
        #print("a_2 = %e, r_2 = %e" % (1./p[2], p[3]))
        print("---------------------------")
    # close pool
    pool.close()
    pool.terminate()
    pool.join()
    if verbose:
        print(res)
        print(chisquare)
    return res, chisquare

if __name__ == "__main__":
    pass
