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
    
    x = df[('weight_Ek_covtrue','own_mean')].values
    yt = df[('Ek_covtrue','own_mean')].values
    yterr=df[('Ek_covtrue','own_std')].values
    #yf = df[('Ek_covfalse','own_mean')].values
    #yferr=df[('Ek_covfalse','own_std')].values
    plt.xlabel(r'weight')
    plt.ylabel(r'$E_{\pi K}$')
    plt.errorbar(x,yt,yerr=yterr,label=r'cov true',fmt='o',color='darkblue',fillstyle='none',capsize=0.75)
    #plt.errorbar(x,yf,yerr=yferr,label=r'cov false',fmt='+r',fillstyle='none',capsize=0.75,alpha=0.5)
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
        fit_k = pd.read_hdf('%s/%s'%(datadir,'fit_k_collect.h5'),
                             key='fit_k')
        fit_k.info()
    	fit_k['fr'] = fit_k['t_i'].astype(str)+','+fit_k['t_f'].astype(str)                   
        # calculate weights for fit E1 with correlation
        fit_k = fit_k.rename(index=str,columns={'par1':'Ek_covtrue'})
        fit_k = fit_k.replace(to_replace=0.001,value=0.01)
        fit_k = get_weights_ens_all(fit_k,'Ek_covtrue')
        overview = chi.bootstrap_means(fit_k,['beta','L','mu_l','mu_s','fr'],
                                       ['Ek_covtrue','weight_Ek_covtrue'])
        print(overview.sample(n=20))
        fit_k.to_hdf('%s/%s'%(datadir,'ek_weighted.h5'),key='ek')
    else:
        fit_k = pd.read_hdf('%s/%s'%(datadir,'ek_weighted.h5'),key='ek')
    # correct wrong entry for A100

    ek_name = "%s/ek_overview.txt"%(datadir)
    ek_overview = pd.DataFrame.from_csv(ek_name,sep='\s+')
    #et mu_s as string
    ek_overview['mu_s'] = ek_overview['mu_s'].map(lambda x: float("0.0%s"%(x.split('_')[-1])))
    ek_overview['eid'] = ek_overview['ensemble']+' '+ek_overview['mu_s'].astype(str)
    print(ek_overview)

        #cut out fitrange 13:37 from d30
    #idx1 = fit_k.loc[(fit_k['ensemble']=='D30.48')&
    #                      (fit_k['mu_s']==0.0115)&
    #                      (fit_k['fr']=='13.0,37.0')].index
    #idx2 = fit_k.loc[(fit_k['ensemble']=='D30.48')&
    #                      (fit_k['mu_s']==0.015)&
    #                     (fit_k['fr']=='13.0,26.0')].index
    #idx3 = fit_k.loc[(fit_k['ensemble']=='D30.48')&
    #                      (fit_k['mu_s']==0.018) &
    #                     (fit_k['fr']=='14.0,31.0')].index
    #fit_k = fit_k.drop(idx1)
    #fit_k = fit_k.drop(idx2)
    #fit_k = fit_k.drop(idx3)
    #d30 = fit_k.loc[fit_k['ensemble']=='D30.48']
    #pd.set_option('display.max_rows',800)
    #pd.set_option('display.width',1000)
    #print(chi.bootstrap_means(d30,['mu_s','fr'],['Ek_covtrue','Ek_covfalse','weight_Ek_covtrue']))
    # Make plots for every ensemble and strange mass comparing epik for each
    # fitrange
    plotname = 'cov_cmp.pdf'
    with PdfPages('%s/%s'%(datadir,plotname)) as pdf:
        for eid in fit_k['eid'].unique():
            ensemble_fit = fit_k.loc[fit_k['eid']==eid]
	    plot_df = chi.bootstrap_means(ensemble_fit,['beta','L','mu_l','mu_s','fr'],
                                                       ['Ek_covtrue',
                                                        'weight_Ek_covtrue'])
            xmin = plot_df[('weight_Ek_covtrue','own_mean')].min() 
            xmax = plot_df[('weight_Ek_covtrue','own_mean')].max()
            print(xmin,xmax)
            ek_dat = ek_overview.loc[ek_overview['eid']==eid]
            print(ek_dat)
            v = ek_dat['ek'].values[0]
            sd = ek_dat['d(ek)'].values[0]
            sup = ek_dat['sup(ek)'].values[0]
            sdn = ek_dat['sdn(ek)'].values[0]
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
