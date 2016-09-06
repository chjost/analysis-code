#!/hadron/knippsch/Enthought/Canopy_64bit/User/bin/python

import os
import numpy as np
import multiprocessing
import matplotlib
matplotlib.use('Agg') # has to be imported before the next lines
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import analysis2 as ana

from pipi_globalfit import read_pi_data, read_pipi_data

def get_levels(datafolder, plotfolder, lattices, irreps, mpicut):
    debug = 0
    # reading switches
    readmpi = True
    readpipi = True
    readplotdata = False

    # read in pion data
    if readmpi:
        fname = "/".join((datafolder, "globalfit", "mpi.npy"))
        mpi = np.load(fname)
    else:
        mpi = read_pi_data(lattices, datafolder)
        fname = "/".join((datafolder, "globalfit", "mpi.npy"))
        np.save(fname, mpi)
    #print(mpi.shape)
    #print("pi data")
    #print(mpi[:,0])

    # calculate the cut
    cut = 2 * np.sqrt(1. + mpicut)* mpi[:,0]
    #print(cut)

    # read in pipi data and do the cut
    if readpipi:
        fname = "/".join((datafolder, "globalfit", "Epipi.npz"))
        fi = np.load(fname)
        Epipi = fi["Epipi"]
        levels = fi["levels"]
        cor = fi["cor"]
    else:
        Epipi, levels, cor = read_pipi_data(datafolder, lattices, irreps, cut)
        fname = "/".join((datafolder, "globalfit", "Epipi.npz"))
        np.savez(fname, Epipi=Epipi, levels=levels, cor=cor)
    #print(Epipi.shape)
    #print(Epipi[:,0])
    #print(cor.shape)
    #print(levels)

    resname = os.path.join(datafolder, "globalfit", "gfitdata.npz")
    with np.load(resname) as f:
        res = f["res"]
        chi2 = f["chi2"]

    #print(mpi.shape)
    #print(Epipi.shape)

    if readplotdata:
        fname = "/".join((datafolder, "globalfit", "plot_data.npz"))
        with np.load(fname) as f:
            Ecalc = f["Ecalc"]
    else:
        Ecalc = []
        pool = multiprocessing.Pool(4)
        
        for j, r in enumerate(res[:2]):
            data = Epipi[:,j]
            m = mpi[:,j]
            _par = np.zeros(4)
            if r.size < 4:
                _par[:r.size] = r
            else:
                _par = r[:4]
            Wroot = np.zeros(np.sum([len(t[-1]) for t in levels]))
            # calc all different roots parallel
            tmp = np.empty((len(levels),), dtype=object)
            args = [[_par, m, x] for x in levels]
            if debug > 2:
                print("args for the pool")
                print(args)
            tmp = pool.map(ana.get_Wroot, args)
            for t,s in zip(tmp, levels):
                count = t[0]
                Wtmp = t[1]
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
            Ecalc.append(Wroot)
        # close pool
        pool.close()
        pool.terminate()
        pool.join()

        Ecalc = np.asarray(Ecalc)
        #print(Ecalc)
        #print(Ecalc.shape)
        # save data
        fname = "/".join((datafolder, "globalfit", "plot_data.npz"))
        np.savez(fname, Ecalc=Ecalc)

    Epipi = Epipi.T
    #print(Epipi[0])
    #print(Epipi.shape)
    Ecalcm = ana.mean_std(Ecalc)
    Epipim = ana.mean_std(Epipi)
    # generate labels for x axis
    xlabels = []
    for l in levels:
        tmpstring = "%d,%2s,TP%d" % (l[0], l[2], l[3])
        for i in l[4]:
            xlabels.append(tmpstring)
    xvals = np.arange(len(xlabels))

    print("info, data, fit")
    for l, Epm, Eps, Ecm, Ecs in zip(xlabels, Epipim[0], Epipim[1], Ecalcm[0], Ecalcm[1]):
        print("%s: (%.3f +- %.3f), (%.3f +- %.3f)" % (l, Epm, Eps, Ecm, Ecs))
    
    #print(xvals)
    #print(Epipim[0])
    #print(Epipim[1])
    #print(Ecalcm[0])
    #print(Ecalcm[1])
    plotter = ana.LatticePlot("%s/globalfit_energy_levels.pdf" % (plotfolder))
    plt.errorbar(xvals, Epipim[0], yerr=Epipim[1], fmt="or", label="data", alpha=0.6)
    plt.errorbar(xvals, Ecalcm[0], yerr=Ecalcm[1], fmt="ob", label="fit", alpha=0.6)
    # manage labels
    plt.xticks(xvals, xlabels, rotation="vertical")
    plt.margins(0.1)
    plt.subplots_adjust(bottom=0.25)
    plt.ylabel("E/a")
    plt.title("Energy overview")

    plt.legend(numpoints=1, loc="best")
    plotter.save()
    del plotter

        
def main():
    # the lattices to work on
    lattices = ["A40.32", "A40.24", "A40.20"]
    # the irreps and momentum squared to work on
    irreps  = [["A1", "E", "T2"], ["A1"], ["A1"], ["A1"]]
    #irreps  = [["A1", "E", "T2"], ["A1"], ["A1"]]
    # the path to the data folders
    datafolder = "./data/I2/"
    # the folder for plots
    plotfolder = "./plots/I2/globalfit/"
    # a upper cut on the data
    mpicut = 3.
    # execute
    get_levels(datafolder, plotfolder, lattices, irreps, mpicut)
    return

# make this script importable, according to the Google Python Style Guide
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
