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
        #string = '%s%d %s'%(b,mul,mus)
        string = '%s%d'%(b,mul)
        ensemblelist.append(string)
    return np.asarray(ensemblelist[0::3])

def plot_deviation(dataframe,label,shift=0.,fmt='ok'):
    dataframe.info()
    means = chi.bootstrap_means(dataframe,['beta','mu_l','mu_s'],['rel.dev.']) 
    # get xdata as ensemblenames 
    print(means.index.values)
    #x = ensemblenames(means.index.values)
    y = ensemblenames(means.index.values)
    print(y)
    #y = means.values[:,0]
    #yerr = means.values[:,1]
    x = means.values[:,0]
    xerr = means.values[:,1]
    print(x,xerr)
    plt.yticks(np.arange(1,y.shape[0]*3+1,3),y)
    plt.xticks(np.arange(-0.006,0.007,0.004))
    #plt.xticks(np.arange(1,y.shape[0]+1,3),y)
    #plt.errorbar(np.arange(y.shape[0])+shift,y,yerr=yerr,fmt=fmt,
    #        label=label)
    plt.errorbar(x,np.arange(3*y.shape[0])+shift,xerr=xerr,fmt=fmt,
            label=label)

def main():
    #load data
    plotdir = "/hiskp4/helmes/analysis/scattering/pi_k/I_32_blocked/plots/"
    resdir = "/hiskp4/helmes/analysis/scattering/pi_k/I_32_blocked/results/"
    proc_id="pi_K_I32_fixms_M1B.h5"
    with PdfPages(plotdir+'/rel_deviation_fixms_M1B_mismatch.pdf') as pdf:
        plt.figure(figsize=(10,25))
        #plt.xlabel(r'$(aM_{K,FSE}^2-aM_K^2(\mu_\ell))/aM_{K,FSE}^2$')
        plt.xlabel(r'rel.dev. $M_{K,FSE}^2$')
        plt.ylabel(r'Ensemble')
        plt.axvline(x=0,linewidth=1,color='k')
        fitres = pd.read_hdf(resdir+proc_id,key='Fitresults')
        #plot_deviation(fitres,r'$P_{\mu}(\beta)$',fmt='^r')
        fitres = pd.read_hdf(resdir+proc_id,key='Fitresults_sigma')
        plot_deviation(fitres,r'$P_{\mu}(\beta,\mu_{\sigma})$',shift=0,fmt='ob')
        plt.legend()
        pdf.savefig()
        plt.close()
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
