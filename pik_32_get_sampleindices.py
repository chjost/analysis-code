#!/usr/bin/python

# Script for getting the sample indices for a single ensemble
import sys
import numpy as np
import itertools
# Christian's packages
sys.path.append('/hiskp4/helmes/projects/analysis-code/')

import analysis2 as ana
def bootstrap_indices(source, nbsamples):
    """Bootstraping of data.

    Creates nbsamples bootstrap samples of source.

    Parameters
    ----------
    source : sequence
        Data on which the bootstrap samples are created.
    nbsamples : int
        Number of bootstrap samples created.

    Returns
    -------
    index : ndarray
        The bootstrap sample numbers
    """
    # the seed is hardcoded to be able to recreate the samples
    # Bastians seed
    np.random.seed(1227)
    # initialize the bootstrapsamples to 0.
    number=source.data.shape[0]
    index = np.zeros((nbsamples,number))
    index[0] = np.arange(number)
    # create the rest of the bootstrap samples
    for _i in range(1, nbsamples):
        _rnd = np.random.randint(0, number, size=number)
        index[_i] = _rnd
    return index

def main():
    # parse infile
    if len(sys.argv) < 2:
        ens = ana.LatticeEnsemble.parse("kk_I1_TP0_A40.24.ini")
    else:
        ens = ana.LatticeEnsemble.parse(sys.argv[1])
    corr_in ="k_charged_p0"
    # get data from input file
    prefix = ens.get_data("path")
    print prefix
    lat = ens.name()
    nboot = ens.get_data("nboot")
    datadir = ens.get_data("datadir")
    plotdir = ens.get_data("plotdir")
    gmax = ens.get_data("gmax")
    d2 = ens.get_data("d2")
    try:
        debug = ens.get_data("debug")
    except KeyError:
        debug = 0
    L = ens.L()
    T = ens.T()
    T2 = ens.T2()
    addT = np.ones((nboot,)) * T
    addT2 = np.ones((nboot,)) * T2

    # read correlation function
    print("read single particle corrs")
    files = ["%s/%s.dat" % (datadir,corr_in)]
    k_corr = ana.Correlators(files, matrix=False)
    print(k_corr.data.shape[0])
    bindex = bootstrap_indices(k_corr,1500)
    print(bindex)
    np.savetxt('%s/bs_indices.txt'%(datadir),bindex,fmt='%d')

if __name__ == '__main__':
    try:
        print("starting")
        main()
    except KeyboardInterrupt:
        pass
