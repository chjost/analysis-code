#!/usr/bin/python

# Plot the results of a chipt extrapolation
# 3 x 2 plots are generated:
# per fitrange
#   1) mu_piK_a32 vs. mu_piK/fpi with errors and LO ChPT curve
#   2) relative deviation between the data points for mu_piK_a32 and the fitted
# function per ensemble

import argparse
import matplotlib
matplotlib.use('pgf') # has to be imported before the next lines
import matplotlib.pyplot as plt
#plt.style.use('paper_standalone')
plt.style.use('paper_side_by_side')
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import sys

import analysis2 as ana
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
        l=int(e[1])
        mul = get_mul_name(e[2])
        #string = '%s%d %s'%(b,mul,mus)
        string = '%s%s.%d'%(b,mul,l)
        ensemblelist.append(string)
    return np.asarray(ensemblelist)

def mu_by_fpi_phys(cont):
    mk = cont.get('mk')
    mpi = cont.get('mpi_0')
    fpi = cont.get('fpi')
    mupik = mk*mpi/(mk+mpi)
    return mupik/fpi

def mua32_phys(df,cont):
    """Calculate the continuum value of mu_piK * a_32
    """
    meta = cont.get('meta')
    mpi = cont.get('mpi_0')
    mk = cont.get('mk')
    fpi = cont.get('fpi')
    #TODO: let that look nicer
    p = df[['L_piK','L_5']].head(n=meta.shape[0]).values.T
    print(p.shape)
    return ana.pik_I32_chipt_nlo_cont(mpi,mk,fpi,p,meta=meta)

#TODO: think about moving that to the extrapolation
def rel_dev_nlochpt(df):
    mua32 = df['mu_piK_a32'].values
    p = df[['L_piK','L_5']].values.T
    mpi = df['M_pi'].values
    mk = df['M_K'].values
    fpi = df['fpi'].values
    meta=df['M_eta'].values
    reldev=(mua32-ana.pik_I32_chipt_nlo(mpi,mk,fpi,p,meta=meta))/mua32
    dev_series = pd.Series(reldev,index=df.index)
    return dev_series

def func(p,x):
    return -2.*p[0]*x+p[1]

def main():
    pd.set_option('display.width',1000)
################################################################################
#                   set up objects                                             #
################################################################################
    # choose fitrange ms_fixing and epik-extraction per command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile",help="infile for paths",type=str,
                        required=True)
    parser.add_argument("--zp", help="Method of RC Z_P",type=int, required=True)
    parser.add_argument("--epik", help="Which method of fitting E_piK",type=int,
                        required=True)
    parser.add_argument("--msfix",help="Method for fixing ms",type=str,required=True)
    args = parser.parse_args()
    # Get presets from analysis input files
    ens = ana.LatticeEnsemble.parse(args.infile)
    nboot = ens.get_data("nboot")
    continuum_seeds = None
    if args.msfix == 'A':
        continuum_seeds=ens.get_data("continuum_seeds_a")
    elif args.msfix =='B':
        continuum_seeds=ens.get_data("continuum_seeds_b")
    else:
        print('ms fixing method unknown') 
    cont_dat = ana.ContDat(continuum_seeds,zp_meth=args.zp,nboot=nboot)
    # get data from input file
    datadir = ens.get_data("datadir") 
    plotdir = ens.get_data("plotdir") 
    resdir = ens.get_data("resultdir") 
    # Load the desired dataframe for a given method M,ms_fix,E_piK
    # build key for dataset
    filename = resdir+'/pi_K_I32_gamma_M%d%s.h5'%(args.zp, args.msfix)
    # calculate physical x-value
    x_phys = mu_by_fpi_phys(cont_dat)
    for fr in range(3):
        #key='/nlo_chpt/E%d/fr_%d'%(args.epik,fr)
        key='fse_true/gamma/E%d/fr_%d'%(args.epik,fr)
        fit_df = pd.read_hdf(filename,key=key)
        fit_df.info()
        y_phys = mua32_phys(fit_df,cont_dat)
        #merge values into fit_df
        #phys_point = {'sample':np.arange(nboot),'mu_piK/fpi_phys':x_phys,'mu_piK_a32_phys':y_phys}
        #phys_df = pd.DataFrame(phys_point)
        #phys_df.info()
        #fit_df = fit_df.merge(phys_df,on='sample')
        # Calculate the relative deviation between the data and the fit
        #fit_df['rel.dev.'] = rel_dev_nlochpt(fit_df)
        # for the plots we only need the y-values, the x-values and the function
        # evaluation, carry with us the identifiers beta mu_l and L
        plot_df = fit_df[['beta','L','mu_l','fr_bgn','fr_end',
                         'M_K/M_pi','Gamma']]
        groups = ['beta','L','mu_l']
        obs = ['M_K/M_pi','Gamma']
        plot_means = chi.bootstrap_means(plot_df,groups,obs)
        print(plot_means)
        # plot the data beta wise 
        plotname = plotdir+'/pi_K_I32_fse_true_gamma_M%d%s_E%d_fr%d'%(args.zp,
                            args.msfix,args.epik,fr) 
        fig = plt.figure()
        plt.xlabel(r'$M_K/M_\pi$',fontsize=11)
        plt.ylabel(r'$\Gamma$',fontsize=11)
        #bfc is for beta,format,colour
        beta = [1.90,1.95,2.1]
        fmt = ['^','v','o']
        col = ['r','b','g']
        plt.xlim((0.,2.2))
        plt.ylim((-0.018,0.008))
        for bfc in zip(beta,fmt,col):
            # get data
            try:
                x = plot_means.xs(bfc[0]).loc[:,[('M_K/M_pi','own_mean')]].values[:,0]
                xerr = plot_means.xs(bfc[0]).loc[:,[('M_K/M_pi','own_std')]].values[:,0]
                y = plot_means.xs(bfc[0]).loc[:,[('Gamma','own_mean')]].values[:,0]
                yerr = plot_means.xs(bfc[0]).loc[:,[('Gamma','own_std')]].values[:,0]
                if x is not None:
                    plt.errorbar(x,y,yerr=yerr,xerr=xerr,fmt=bfc[1]+bfc[2],
                                 label = r'$\beta=$%.2f'%bfc[0])
            except:
                pass
        # an errorband is derived from filling the maximal and minimal
        # curve
        p = fit_df[['L_piK','L_5']].values[0:1500].T
        print(p.shape)
        x = np.linspace(0,2.2,num=300)
        #original sample is at 0
        y = func(p[:,0],x)
        yext = np.asarray([func(p,m) for m in x])
        print(yext.shape)
        ymin = yext.min(axis=1)
        ymax = yext.max(axis=1)
        print(x.shape)
        print(y.shape)
        print(ymin.shape)
        print(ymax.shape)
        plt.errorbar(x,y,fmt='-k',label=r'linear fit')
        plt.fill_between(x,ymin,ymax,color='gray',alpha=0.4)
        #plt.errorbar(x,ymin,fmt='-.r')
        #plt.errorbar(x,ymax,fmt='--r')
        # physical point
        #x = plot_means.loc[:,[('mu_piK/fpi_phys','own_mean')]].values[0] 
        #xerr = plot_means.loc[:,[('mu_piK/fpi_phys','own_std')]].values[0]
        #y = plot_means.loc[:,[('mu_piK_a32_phys','own_mean')]].values[0]
        #yerr =plot_means.loc[:,[('mu_piK_a32_phys','own_std')]].values[0] 
        #plt.errorbar(x,y,xerr=xerr,yerr=yerr,fmt='d',color='darkgoldenrod',
        #             label=r'physical point')
        plt.legend()
        matplotlib.backends.backend_pgf.FigureCanvasPgf(fig).print_pdf(plotname+'.pdf')
        plt.savefig(plotname+'.pgf')
        plt.clf()

        #plot_dev_df = fit_df[['beta','L','mu_l','rel.dev.']]
        #rel_dev = chi.bootstrap_means(plot_dev_df,['beta','L','mu_l'],['rel.dev.'])
        #plt.xlabel(r'rel.dev. $\mu_{\pi K}a_0$')
        #y = ensemblenames(rel_dev.index.values)
        #x = rel_dev.values[:,0]
        #xerr = rel_dev.values[:,1]
        #plt.yticks(np.arange(y.shape[0]),y)
        ##plt.xticks()
        #plt.errorbar(x,np.arange(y.shape[0]),xerr=xerr,fmt='ob',label =
        #        r'M%d%s E%d fr%d'%(args.zp,args.msfix,args.epik,fr))
        #plt.axvline(x=0,linewidth=1,color='k')
        #plt.legend(frameon=True)
        #plt.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
