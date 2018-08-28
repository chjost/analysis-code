#!/usr/bin/python
import matplotlib
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
plt.style.use('paper_standalone')
import numpy as np
import pandas as pd

import chiron as chi
def plot_epik_cmp(df,eid):
    # prep data for errorbar plot
    # x is a fitrange identifier
    
    x = df[('weight_Epik_covtrue','own_mean')].values
    yt = df[('Epik_covtrue','own_mean')].values
    yterr=df[('Epik_covtrue','own_std')].values
    yf = df[('Epik_covfalse','own_mean')].values
    yferr=df[('Epik_covfalse','own_std')].values
    plt.xlabel(r'weight')
    plt.ylabel(r'$E_{\pi K}$')
    plt.errorbar(x,yt,yerr=yterr,label=r'cov true',fmt='o',color='darkblue',fillstyle='none',capsize=0.75)
    plt.errorbar(x,yf,yerr=yferr,label=r'cov false',fmt='+r',fillstyle='none',capsize=0.75,alpha=0.5)
    #plt.xscale('log')
    #plt.xticks(x,df['fr'].values,rotation=90,fontsize=8)
    #plt.xlim(-0.04,-0.01)
    plt.legend()
    plt.title(eid)
def get_weights_ens_all(df,obs):                                                 
    df['ensemble'] = chi.ensemblenames(df[['beta','L','mu_l']].values)           
    df['eid'] = df['ensemble']+' '+df['mu_s'].astype(str)                        
    # Loop over ensembles
    df_weights=pd.DataFrame()
    for eid in df['eid'].unique():                                               
        ens_result=df.loc[df['eid']==eid]           
        # get weights per ensemble
        ens_result=get_weights(ens_result,obs)
	print(ens_result.head())
        df_weights = df_weights.append(ens_result)                                
    return df_weights                                                          
    
# define a function that calculates the weights from fitresults
def get_weights(df,obs,rel=False):
    # weights are calculated per fitrange
    # prepare dataframe for queries
    df['fr'] = df['t_i'].astype(str)+','+df['t_f'].astype(str)
    df['boot'] = df['sample'].astype(str)
    wcolname = 'weight_'+obs
    # get statistics of obs with fitranges as index, collect all data for weights there
    stats = chi.bootstrap_means(df,['fr'],obs)
    stats.reset_index(inplace=True)
    #print('\nStatistics for observable:%s over fitranges' %obs)
    #print(stats)
    # Reset index also resets column multilevel
    if rel is True:
        stats['error'] = stats['own_std']/stats['own_mean']
    else:
        # this option uses the distribution mean for calculating the statistical error.
        # I think that this is implemented in the analysis as well
        # TODO: implement that as an option?
        #stats['error'] = df.groupby('fr').std().reset_index()[obs]
        stats['error'] = stats['own_std']
    print('\nStatistics for observable:%s over fitranges' %obs)
    print(stats)
    min_err = stats['error'].min()
    #print("\nMinimal (relative) error over fitranges")
    #print(min_err)
    # get pvalues of original data
    pvals = df[['fr','p-val']].loc[df['boot']=='0.0']
    #print("\nP-values of the fits")
    #print(pvals)
    stats = stats.set_index('fr').join(pvals.set_index('fr')).reset_index()
    # calculate weights
    stats[wcolname]=((1.-2.*np.abs(stats['p-val']-0.5))*min_err/stats['error'])**2
    #print("\nreturned dataframe with weights")
    #print(stats)
    df = df.merge(stats[['fr',wcolname]],left_on='fr',right_on='fr')
    return df

def main():
    collect_data =False
    datadir ='/hiskp4/helmes/analysis/scattering/pi_k/I_32_publish/data' 
    if collect_data is True:
        fit_e1 = pd.read_hdf('%s/%s'%(datadir,'fit_pik_collect.h5'),
                             key='fit_corr_e1')
        fit_e1_corr_false = pd.read_hdf('%s/%s'%(datadir,'fit_pik_collect.h5'),
                                        key='fit_corr_e1_corr_false')
        fit_e1.info()
    	fit_e1['fr'] = fit_e1['t_i'].astype(str)+','+fit_e1['t_f'].astype(str)                   
    	fit_e1_corr_false['fr'] = fit_e1['t_i'].astype(str)+','+fit_e1['t_f'].astype(str)                   
        # calculate weights for fit E1 with correlation
        fit_e1 = fit_e1.rename(index=str,columns={'par1':'Epik_covtrue'})
        fit_e1 = fit_e1.replace(to_replace=0.001,value=0.01)
        fit_e1_corr_false = fit_e1_corr_false.replace(to_replace=0.001,value=0.01)
        fit_e1 = get_weights_ens_all(fit_e1,'Epik_covtrue')
        fit_e1_corr_false = fit_e1_corr_false.rename(index=str,columns={'par1':'Epik_covfalse'})
        fit_e1 = fit_e1.set_index(['beta','L','mu_l','mu_s','fr','sample'])
        fit_e1_corr_false = fit_e1_corr_false.set_index(['beta','L','mu_l','mu_s','fr','sample'])
        fit_e1_cmp = fit_e1.join(fit_e1_corr_false,lsuffix='covtrue',rsuffix='covfalse')
        fit_e1_cmp = fit_e1_cmp.reset_index()
        overview = chi.bootstrap_means(fit_e1_cmp,['beta','L','mu_l','mu_s','fr'],
                                       ['Epik_covtrue','Epik_covfalse','weight_Epik_covtrue'])
        print(overview.sample(n=20))
        fit_e1_cmp.to_hdf('%s/%s'%(datadir,'epik_cov_comparison.h5'),key='e1_cmp')
    else:
        fit_e1_cmp = pd.read_hdf('%s/%s'%(datadir,'epik_cov_comparison.h5'),key='e1_cmp')
    # correct wrong entry for A100

    epik_name = "%s/epik_overview.txt"%(datadir)
    epik_overview = pd.DataFrame.from_csv(epik_name,sep='\s+')
    # get mu_s as string
    epik_overview['mu_s'] = epik_overview['mu_s'].map(lambda x: float("0.0%s"%(x.split('_')[-1])))
    epik_overview['eid'] = epik_overview['ensemble']+' '+epik_overview['mu_s'].astype(str)
    print(epik_overview)

        #cut out fitrange 13:37 from d30
    idx1 = fit_e1_cmp.loc[(fit_e1_cmp['ensemble']=='D30.48')&
                          (fit_e1_cmp['mu_s']==0.0115)&
                          (fit_e1_cmp['fr']=='13.0,37.0')].index
    idx2 = fit_e1_cmp.loc[(fit_e1_cmp['ensemble']=='D30.48')&
                          (fit_e1_cmp['mu_s']==0.015)&
                         (fit_e1_cmp['fr']=='13.0,26.0')].index
    idx3 = fit_e1_cmp.loc[(fit_e1_cmp['ensemble']=='D30.48')&
                          (fit_e1_cmp['mu_s']==0.018) &
                         (fit_e1_cmp['fr']=='14.0,31.0')].index
    fit_e1_cmp = fit_e1_cmp.drop(idx1)
    fit_e1_cmp = fit_e1_cmp.drop(idx2)
    fit_e1_cmp = fit_e1_cmp.drop(idx3)
    d30 = fit_e1_cmp.loc[fit_e1_cmp['ensemble']=='D30.48']
    pd.set_option('display.max_rows',800)
    pd.set_option('display.width',1000)
    #print(chi.bootstrap_means(d30,['mu_s','fr'],['Epik_covtrue','Epik_covfalse','weight_Epik_covtrue']))
    # Make plots for every ensemble and strange mass comparing epik for each
    # fitrange
    plotname = 'cov_cmp.pdf'
    with PdfPages('%s/%s'%(datadir,plotname)) as pdf:
        for eid in fit_e1_cmp['eid'].unique():
            ensemble_fit = fit_e1_cmp.loc[fit_e1_cmp['eid']==eid]
	    plot_df = chi.bootstrap_means(ensemble_fit,['beta','L','mu_l','mu_s','fr'],
                                                       ['Epik_covtrue','Epik_covfalse',
                                                        'weight_Epik_covtrue'])
            xmin = plot_df[('weight_Epik_covtrue','own_mean')].min() 
            xmax = plot_df[('weight_Epik_covtrue','own_mean')].max()
            print(xmin,xmax)
            epik_dat = epik_overview.loc[epik_overview['eid']==eid]
            print(epik_dat)
            v = epik_dat['epik'].values[0]
            sd = epik_dat['d(epik)'].values[0]
            sup = epik_dat['sup(epik)'].values[0]
            sdn = epik_dat['sdn(epik)'].values[0]
            plt.fill_between(np.array((xmin,xmax)),v+np.sqrt(sd**2+sup**2),v-np.sqrt(sd**2+sdn**2),alpha=0.3,color='deepskyblue',label='sys')
            plt.fill_between(np.array((xmin,xmax)),v+sd,v-sd,alpha=0.3,color='darkblue',label='stat.')
            plt.axhline(v,color='black',label='weighted median')
            plot_epik_cmp(plot_df,eid)
            pdf.savefig()
            plt.clf()
if __name__ == "__main__":
    try:
        main()
    except(KeyboardInterrupt):
        print(KeyboardInterrupt)
