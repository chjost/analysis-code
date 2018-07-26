#!/usr/bin/python

import sys
import numpy as np
import analysis2 as ana
import re

def error(data,mean):
    var = np.nansum(np.square(data-mean))
    return np.sqrt(var/len(data))

def main():
#######################################################################
# Begin calculation
#######################################################################
    # single particle correlator
    read_samples=False
    print("read single particle corrs")
    datadir = '/hiskp4/helmes/analysis/scattering/pi_k/I_32_final/data/A40.24'
    mu_s = ['amu_s_185','amu_s_225','amu_s_2464']
    #files = ["%s/pi_corr_p%d.dat" % (prefix, d) for d in range(4)]
    files = ["%s/%s/k_charged_p0.dat" % (datadir, s) for s in mu_s]
    samples = 2000
    #bl = [1,2,3]
    bl = [1,4,10]
    bl.reverse()
    T=48
    T2=T/2
    addT = np.ones((samples,)) * T
    addT2 = np.ones((samples,)) * T2
    plotter=ana.LatticePlot('./hist_ck_mus2464.pdf',join=True)
    plotter.set_env(grid=False,title=r'Histogram, $C_K$ $a\mu_s = 0.02464$')
    #corr = ana.Correlators(files, matrix=False,conf_col = 2)
    #corr.sym_and_boot(samples,blocking=False,bl=1,method='naive')
    #k_fit = ana.LatticeFit(9,dt_f=-1,dt_i=1,dt=7,correlated=True)
    #k_fitres = k_fit.fit([2.,0.1],corr,[16,22],add=addT)
    #k_fitres.print_details()
    #plotter.plot(corr,['Correlator','t','C(t)','BL=0'],fitresult=k_fitres,
    #             fitfunc=k_fit,add=addT,ploterror=True)
    t = 20
    for nb in bl:
        #corr.bootstrap(samples,blocking=False,bl=nb,method='stationary')
        if read_samples is False:
            corr = ana.Correlators(files, matrix=False,conf_col = 2)
            corr.sym_and_boot(samples,blocking=False,bl=nb,method='stationary')
            corr.save('./%s_bl%d'%('A40.24_k',nb))
        else:
            corr = ana.Correlators.read('./%s_bl%d.npz'%(mu_s[0],nb)) 
        k_fit = ana.LatticeFit(0,dt_f=-1,dt_i=1,dt=7,correlated=True,debug=2)
        k_fitres = k_fit.fit(None,corr,[16,22],add=addT)
        k_fitres.print_details()
        plotter.bs_hist(corr.data[:,t,0],['bl=%d'%nb,r'$C_K(%d)$'%(t+1)],nb_bins=75)
        #plotter.plot(corr,['Correlator','t','C(t)','BL=%d'%nb],fitresult=k_fitres,
        #             fitfunc=k_fit,add=addT,ploterror=True)
        #corr.mass()
        #meff_fit = ana.LatticeFit(2,-1,1,7)
        #meff = meff_fit.fit([0.1],corr,[[16,22]],add=None)
        #meff.print_data(par=0)
        #plotter.plot(corr,['Correlator','t','m_effK(t)','BL = %d'%nb],
        #             fitresult=meff,fitfunc=meff_fit,ploterror=True)
        #print("\nblocklength: %d\tsamples: %d\n" %(nb,samples))
        #data_arr = np.asarray([k_fitres.data[i][:,0,0] for i in range(len(k_fitres.data))])
        #print("\ncorrelation coefficients:\n")
        #np.set_printoptions(formatter={'float':lambda x: '%06f'%x})
        #print(re.sub('[ ]+', ' ', re.sub(' *[\\[\\]] *', '', np.array_str(np.corrcoef(data_arr)))))
        #print(np.corrcoef(data_arr))
        #print("\ncovariance matrix\n")
        #np.set_printoptions()
        #print(re.sub('[ ]+', ' ', re.sub(' *[\\[\\]] *', '', np.array_str(np.cov(data_arr)))))
        #print(np.cov(data_arr))
        #print("\n\n")
    plotter.save()
    del plotter
if __name__ == '__main__':
    try:
        print("starting")
        main()
    except KeyboardInterrupt:
        pass

