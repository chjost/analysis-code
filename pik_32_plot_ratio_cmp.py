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
def df_eval_corr_e1(df):
    t = df['t'].values
    p = df[['par0','par1','par2']].values.T
    add=df[['medianE_pi','medianE_K','T']].values.T
    print(p.shape)
    print(add.shape)
    fit_eval = ana.functions.func_corr_shift_therm(p,t,add)
    return fit_eval
def df_eval_corr_e2(df):
    t = df['t'].values
    p = df[['par0','par1']].values.T
    add=df[['medianE_pi','medianE_K','T']].values.T
    print(p.shape)
    print(add.shape)
    fit_eval = ana.functions.func_corr_shift_poll_removal(p,t,add)
    return fit_eval

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ens",help="directoryname",type=str,required=True)
    parser.add_argument("--mus",help="bare strange mass",type=str,required=True)
    args=parser.parse_args()
    # Define filenames
    path = '/hiskp4/helmes/analysis/scattering/pi_k/I_32_publish/'
    datadir = "%s/data/%s/amu_s_%s/" %(path,args.ens,args.mus)
    plotdir = "%s/plots/%s/amu_s_%s/" %(path,args.ens,args.mus)
    fname_corr = 'corr_pik.h5'
    fname_fitres = 'fit_pik.h5'
    # Load necessary data
    corr_e1 = pd.read_hdf('%s/%s'%(datadir,fname_corr),key='ws_e1')
    corr_e2 = pd.read_hdf('%s/%s'%(datadir,fname_corr),key='ws_e2')
    fitres_e1 = pd.read_hdf('%s/%s'%(datadir,fname_fitres),key='fit_corr_e1')
    fitres_e2 = pd.read_hdf('%s/%s'%(datadir,fname_fitres),key='fit_corr_e2')
    #fitres_e1 = pd.read_hdf('%s/%s'%(datadir,fname_fitres),
    #                        key='fit_corr_e1_corr_false')
    #fitres_e2 = pd.read_hdf('%s/%s'%(datadir,fname_fitres),
    #                        key='fit_corr_e2_corr_false')
    fit_input = pd.read_hdf('%s/%s'%(datadir,'pik_fit_input.h5'),key='additional_input')
    # Massage data for E1
    fit_input = fit_input.reset_index().rename(index=str,columns={'index':'sample'})
    fit_input['sample']=fit_input['sample'].astype(float)
    fitres_e1 = fitres_e1.merge(fit_input,on='sample')
    # That is all for E1
    corr_e1['boot']=corr_e1['sample'].astype(str)
    # duplicate original data
    orig_corr_e1 = corr_e1.loc[corr_e1['boot']=='0.0']
    # Hard code sample number
    nboot=1500
    corr_e1 = pd.DataFrame()
    for bs in np.arange(nboot):
        orig_corr_e1['sample']=bs
        corr_e1=corr_e1.append(orig_corr_e1)
    orig_corr_e1['sample']=orig_corr_e1['sample'].astype(float)
    print(orig_corr_e1.info())
    ratio_e1 = corr_e1.merge(fitres_e1,on='sample',how='outer')
    ratio_result_e1 = df_eval_corr_e1(ratio_e1)
    ratio_e1['r'] = ratio_e1['C(t)']/ratio_result_e1
    result_e1 = ratio_e1[['t_i','t_f','t','r','sample','par1','chi^2/dof']]
    result_e1['fr']=result_e1['t_i'].astype(str)+';'+result_e1['t_f'].astype(str)
    result_e1.set_index('sample',inplace=True)
    result_e1.sample(n=20)
    plot_data_e1 = chi.bootstrap_means(result_e1,['fr','t'],['r','par1','chi^2/dof'])
    plot_data_e1.reset_index(inplace=True)
    fr = plot_data_e1['fr'].unique()
     
    fitres_e2 = fitres_e2.merge(fit_input,on='sample')
    corr_e2['boot']=corr_e2['sample'].astype(str)
    # duplicate original data
    orig_corr_e2 = corr_e2.loc[corr_e2['boot']=='0.0']
    # Hard code sample number
    nboot=1500
    corr_e2 = pd.DataFrame()
    for bs in np.arange(nboot):
        orig_corr_e2['sample']=bs
        corr_e2=corr_e2.append(orig_corr_e2)
    orig_corr_e2['sample']=orig_corr_e2['sample'].astype(float)
    print(orig_corr_e2.info())
    ratio_e2 = corr_e2.merge(fitres_e2,on='sample',how='outer')
    ratio_result_e2 = df_eval_corr_e2(ratio_e2)
    ratio_e2['r'] = ratio_e2['C(t)']/ratio_result_e2
    result_e2 = ratio_e2[['t_i','t_f','t','r','sample','par1','chi^2/dof']]
    result_e2['fr']=result_e2['t_i'].astype(str)+';'+result_e2['t_f'].astype(str)
    result_e2.set_index('sample',inplace=True)
    result_e2.sample(n=20)
    plot_data_e2 = chi.bootstrap_means(result_e2,['fr','t'],['r','par1','chi^2/dof'])
    plot_data_e2.reset_index(inplace=True)
    #fr = plot_data_e1['fr'].unique()
    # Open pdfpages
    with PdfPages(plotdir+'ratio.pdf') as pdf:
    #with PdfPages(plotdir+'ratio_corr_false.pdf') as pdf:
	# TODO: use iterrows for that
        for i,r in enumerate(fr):
            print("plotting fitrange %s"% r)
            interval = r.split(';')
            t_i = float(interval[0])
            t_f = float(interval[1])
            tmp_data_e1 = plot_data_e1.loc[plot_data_e1['fr']==r]
            tmp_data_e2 = plot_data_e2.loc[plot_data_e2['fr']==r]
            print(tmp_data_e1)
            x = tmp_data_e1['t'].values
            r_e1 = tmp_data_e1.loc[:,[('r','own_mean')]].values
            dr_e1 = tmp_data_e1.loc[:,[('r','own_std')]].values 
            r_e2 = tmp_data_e2.loc[:,[('r','own_mean')]].values
            dr_e2 = tmp_data_e2.loc[:,[('r','own_std')]].values
            chi2_e1 = tmp_data_e1.values[0,6]
            chi2_e2 = tmp_data_e2.values[0,6]
            epik_e1 = (tmp_data_e1.values[0,4],tmp_data_e1.values[0,5])
            epik_e2 = (tmp_data_e2.values[0,4],tmp_data_e2.values[0,5])
            #lbl_e1=r'E1 $E_{\pi K} = %.3f \pm %.3f$'%(epik_e1[0],epik_e1[1])
            #lbl_e2=r'E2 $E_{\pi K} = %.3f \pm %.3f$'%(epik_e2[0],epik_e2[1])
            lbl_e1 = r'E1'
            lbl_e2 = r'E2'
            plt.errorbar(x,r_e1,dr_e1,fmt='o',capsize=1.,fillstyle='none',color='darkblue',label=lbl_e1,lw=0.5)
            plt.errorbar(x,r_e2,dr_e2,fmt='+',capsize=1.,color='firebrick',lw=0.5,label=lbl_e2)
            plt.ylim((0.9,1.10))
            plt.xlim((5,tmp_data_e1.values.shape[0]))
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


