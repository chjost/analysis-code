#!/hadron/knippsch/Enthought/Canopy_64bit/User/bin/python

# Script plotting fit parameter in dependance of fit rangess

import matplotlib
matplotlib.use('Agg') # has to be imported before the next lines
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
import sys
import numpy as np
import analysis2 as ana

def cut_data(fitres, ranges, par=1):
  """ Function to cut data in FitResult object to certain fit ranges

  Parameters:
  -----------
  fitres : A FitResult object containing the parameter of interest for all fit
           ranges
  ranges : A list of fit range indices to take
  par : the parameter of the fit result

  Returns:
  --------
  fitres_cut : A truncated FitResult object containing only, data, chi2 and pvals for the fit
  ranges of interst
  """
  # Create an empty correlator object for easier plotting
  fitres_cut = ana.FitResult('delta E', derived = True)
  # shape for 1 Correlator, data and pvalues
  shape_dE = (fitres.data[0].shape[0], len(ranges))
  shape_pval = (fitres.data[0].shape[0], len(ranges))
  shape1 = [shape_dE for dE in fitres.data]
  shape2 = [shape_pval for p in fitres.pval]
  fitres_cut.create_empty(shape1, shape2, 1)

  # Get data for error calculation
  dat = fitres.data[0][:,par,0,ranges]
  pval = fitres.pval[0][:,0,ranges]
  chi2 = fitres.chi2[0][:,0,ranges]
  # add data to FitResult
  fitres_cut.add_data((0,range(len(ranges))),dat,chi2,pval)

  return fitres_cut

def main():
####################################################
# parse the input file and setup parameters
#####################################################
    if len(sys.argv) < 2:
        ens = ana.LatticeEnsemble.parse("kk_I1_TP0_A40.24.ini")
    else:
        ens = ana.LatticeEnsemble.parse(sys.argv[1])

    # read settings
    readsinglefit = False
    plotsingle = False
    readtwofit = True
    plottwo = True

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
    T = ens.T()
    T2 = ens.T2()
    addT = np.ones((nboot,)) * T
    addT2 = np.ones((nboot,)) * T2


#######################################################################
# Begin calculation
#######################################################################

    # Read in ratiofit
    ratiofit = ana.FitResult.read("%s/fit_kk_TP%d_%s.npz" % (datadir,
        d2, lat))
    # get fit_ranges
    range_r, r_r_shape = ratiofit.get_ranges()
    #print(range_r[0])
    t_low = [10,11,12,13]
    t_hi = [24,26,28,30,32]
    fr = len(t_low)*len(t_hi)
    res = np.zeros((fr, 6))
    j = 0
    for l in t_low:
      for h in t_hi:
        list1=[]
        idx=[]
        for s,i in enumerate(range_r[0]):
          if i[0] >= l and i[1] <= h:
            if i[1] - i[0] >= 10:
              list1.append(i)
              idx.append(s)
            else:
              continue
        # get data for plotting
        deltaE = cut_data(ratiofit, idx)
        eshift = deltaE.data_for_plot()
        print eshift
        res[j,:] = np.asarray([l, h, eshift[0], eshift[1],
          eshift[1]+eshift[2],eshift[1]+eshift[3]])
        j += 1
    x = np.linspace(0,fr,fr,endpoint=False)
    plot = PdfPages("%s/sys_err_deltaE_fr%d-%d.pdf" % (plotdir,t_low[0],t_hi[-1]))
    # plot original delta E with errors
    lbl = [r'Energy shift $\delta E$ for fit ranges',r'$fr_{min}$',
         r'$\delta E$', lat]
    ann = [" " for v in zip(res[:,0],res[:,1])]
    fig, ax = plt.subplots()
    for i in x:
      # lower annotation
      yup=res[i,2]-res[i,4]
      ax.annotate(str(int(res[i,0])), xy=(i, yup), xycoords='data',
                        xytext=(-7.5,-12), textcoords='offset points', alpha=0.6)
      # upper annotation
      ydn=res[i,2]+res[i,5]
      ax.annotate(str(int(res[i,1])), xy=(i, ydn), xycoords='data',
                        xytext=(-7.5,10), textcoords='offset points', alpha=0.6)
    ax.set_xticklabels(ann)
    plt.axhline(y=np.mean(res[:,2]), ls='solid', color='red', label='avg.')
    plt.errorbar(x, res[:,2], yerr=[res[:,4],res[:,5]], fmt='o', color='blue')
    plt.errorbar(x, res[:,2], yerr=res[:,3], fmt='o',color='blue', label=lbl[3])
    plt.title(lbl[0])
    plt.grid()
    plt.xlim((-1,fr+1))
    plt.ylim((0.0028,0.0045))
    plt.xlabel(lbl[1])
    plt.ylabel(lbl[2])
    plt.legend()
    plot.savefig()
    plot.close()
if __name__ == '__main__':
    try:
        print("starting")
        main()
    except KeyboardInterrupt:
        pass
