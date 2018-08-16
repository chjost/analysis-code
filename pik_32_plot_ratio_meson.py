#!/usr/bin/python
# Plot weights as a function of the final timeslice for E1 and E3
import argparse
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
plt.style.use('paper_side_by_side')
import numpy as np
import pandas as pd
import sys
# Christian's packages
sys.path.append('/hiskp4/helmes/projects/analysis-code/')
import chiron as chi
import analysis2 as ana
def df_eval_meson_exp(df):
    t = df['t'].values
    add=2*df['t'].max()
    p = df[['par0','par1']].values.T
    fit_eval = ana.functions.func_single_corr_bare(p,t,add)
    return fit_eval

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ens",help="directoryname",type=str,required=True)
    parser.add_argument("--mus",help="bare strange mass",type=str,required=True)
    args=parser.parse_args()
    # Define filenames
    path = '/hiskp4/helmes/analysis/scattering/pi_k/I_32_publish/'
    if args.mus =='pi':
        datadir = "%s/data/%s/pi/" %(path,args.ens)
        plotdir = "%s/plots/%s/pi/" %(path,args.ens)
        fname_corr = 'corr_pi_%s.h5'%args.ens
        fname_fitres = 'fit_pi_%s.h5'%args.ens
        label = 'pi'
    else:
        datadir = "%s/data/%s/amu_s_%s/" %(path,args.ens,args.mus)
        plotdir = "%s/plots/%s/amu_s_%s/" %(path,args.ens,args.mus)
        fname_corr = 'corr_k_%s.h5'%args.ens
        fname_fitres = 'fit_k_%s.h5'%args.ens
        label = 'k'
    # Load necessary data
    corr = pd.read_hdf('%s/%s'%(datadir,fname_corr),key='sym_sb')
    fitres = pd.read_hdf('%s/%s'%(datadir,fname_fitres),key='fit_%s'%label)
    #fitres_e1 = pd.read_hdf('%s/%s'%(datadir,fname_fitres),
    #                        key='fit_corr_e1_corr_false')
    corr['boot']=corr['sample'].astype(str)
    # duplicate original data
    orig_corr = corr.loc[corr['boot']=='0.0']
    # Hard code sample number
    nboot=1500
    corr_ratio = pd.DataFrame()
    for bs in np.arange(nboot):
        orig_corr['sample']=bs
        corr_ratio=corr_ratio.append(orig_corr)
    orig_corr['sample']=orig_corr['sample'].astype(float)
    print(orig_corr.info())
    ratio = corr_ratio.merge(fitres,on='sample',how='outer')
    ratio_result = df_eval_meson_exp(ratio)
    ratio['r'] = ratio['C(t)']/ratio_result
    result = ratio[['t_i','t_f','t','r','sample','par1','chi^2/dof']]
    result['fr']=result['t_i'].astype(str)+';'+result['t_f'].astype(str)
    result.set_index('sample',inplace=True)
    result.sample(n=20)
    plot_data = chi.bootstrap_means(result,['fr','t'],['r','par1','chi^2/dof'])
    plot_data.reset_index(inplace=True)
    fr = plot_data['fr'].unique()
     
    # Open pdfpages
    with PdfPages(plotdir+'ratio_%s.pdf'%label) as pdf:
    #with PdfPages(plotdir+'ratio_corr_false.pdf') as pdf:
	# TODO: use iterrows for that
        for i,r in enumerate(fr):
            print("plotting fitrange %s"% r)
            interval = r.split(';')
            t_i = float(interval[0])
            t_f = float(interval[1])
            tmp_data = plot_data.loc[plot_data['fr']==r]
            print(tmp_data)
            x = tmp_data['t'].values
            r = tmp_data.loc[:,[('r','own_mean')]].values
            dr = tmp_data.loc[:,[('r','own_std')]].values 
            #lbl_e1=r'E1 $E_{\pi K} = %.3f \pm %.3f$'%(epik_e1[0],epik_e1[1])
            lbl = r'%s'%label
            plt.errorbar(x,r,dr,fmt='o',capsize=1.,fillstyle='none',color='darkblue',label=lbl,lw=0.5)
            plt.ylim((0.9,1.10))
            plt.xlim((5,tmp_data.values.shape[0]))
            plt.xlabel(r'$t/a$',size=11)
            plt.ylabel(r'$C(t)/f(t)$',size=11)
            plt.legend(loc='lower left',frameon=False)
            plt.axvline(t_i,ls='--')
            plt.axvline(t_f,ls='--')
            plt.axhline(1.,alpha=0.6,c='k')
            pdf.savefig()
            plt.clf()
	plt.close()
if __name__ == '__main__':
    try:
	print("starting")
	main()
    except KeyboardInterrupt:
	pass


