#!/usr/bin/python
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # has to be imported before the next lines
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
import chiron as chi

def get_beta_name(b):
    if b == 1.90:
        return 'A'
    elif b == 1.95:
        return 'B'
    elif b == 2.10:
        return 'D'
    else:
        print('bet not known')

def get_mul_name(l):
    return l*10**4
def get_mus_name(s):
    if s in [0.0115,0.013,0.0185,0.016]:
        return 'lo'
    elif s in [0.015,0.0186,0.0225]:
        return 'mi'
    elif s in [0.018,0.021,0.02464]:
        return 'hi'
    else:
                print('mu_s not known')
def ensemblenames(ix_values):
    ensemblelist = []
    for i,e in enumerate(ix_values):
        b = get_beta_name(e[0])
        mul = get_mul_name(e[1])
        mus = get_mus_name(e[2])
        string = '%s%d %s'%(b,mul,mus)
        ensemblelist.append(string)
    return np.asarray(ensemblelist)

def main():
    #load data
    plotdir = "/hiskp4/helmes/analysis/scattering/pi_k/I_32_blocked/plots/"
    resdir = "/hiskp4/helmes/analysis/scattering/pi_k/I_32_blocked/results/"
    proc_id="pi_K_I32_fixms_M1B.h5"
    fitres = pd.read_hdf(resdir+proc_id)
    fitres.info()
    means = chi.bootstrap_means(fitres,['beta','mu_l','mu_s'],['rel.dev.']) 
    # get xdata as ensemblenames 
    print(means.index.values)
    x = ensemblenames(means.index.values)
    print(x)
    y = means.values[:,0]
    yerr = means.values[:,1]
    print(y,yerr)


    with PdfPages(plotdir+'/rel_deviation_fixms_M1B.pdf') as pdf:
        plt.figure(figsize=(25,10))
        plt.xticks(np.arange(x.shape[0]),x,rotation=90)
        plt.ylabel(r'$(aM_{K,FSE}^2-aM_K^2(\mu_\ell)/aM_{K,FSE}^2$')
        plt.xlabel(r'Ensemble')
        plt.axhline(y=0,linewidth=1,color='k')
        plt.errorbar(np.arange(x.shape[0]),y,yerr=yerr,fmt='^r',
                label=r'$\beta$-dependent $\mu_s$ parameter')
        plt.legend()
        pdf.savefig()
        plt.close()
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
