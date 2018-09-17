import pandas as pd
import numpy as np
import syseffos_io as io
import syseffos_info as info
import boot_statistics as bstats

def bootstrap_means(frame,names,observables):
    if names is None:
        frame['dummy_key'] = observables
        names='dummy_key'
    bstrap_frame=frame.groupby(names)[observables].agg([bstats.own_mean,
                                                   bstats.own_std])
    #info.print_si_format(bstrap_frame)
    return bstrap_frame

def bootstrap_means_key(frame,groups,observables,loc=None):
    # mean over fitranges, needs group by sample
    mean_frame=frame.groupby(groups+['sample']).mean()
    # undo grouping by samples
    mean_frame=frame.reset_index(level=['sample'])
    mean_frame=get_at_loc(mean_frame,groups,observables,loc_tuple=loc)
    return mean_frame


def get_at_loc(frame,groups,observables,loc_tuple=None):
    if loc_tuple is not None:
        mean_frame=bootstrap_means(frame,groups,observables).loc[loc_tuple,:]
    else:
        mean_frame=bootstrap_means(frame,groups,observables)
    return mean_frame

def average_all_methods(frame,agg,agg_keys=None,fixed=None,index=None,value=None,drop=None):
    """Take average over all specified methods
    
    DataFrame gets averaged over all methods specified in agg
    """

    # need only lattice artefact 'None' and fit_end 2.5
    # TODO: I cannot shake off the feeling that this should work simpler
    # Look at Lattice artefact None and largest fitrange
    if index is None and value is None:
        index,value,drop=default_index()
    else:
        drop=index+drop
    filtered_frame=choose_data(frame,index,value,drop)
    return method_average(filtered_frame,agg,agg_keys=agg_keys)


def average_methods(frame,agg1,agg2,agg_keys=None,fixed=None,index=None,value=None,drop=None):
    """Take systematic 

    """
    # need only lattice artefact 'None' and fit_end 2.5
    # TODO: I cannot shake off the feeling that this should work simpler
    # Look at Lattice artefact None and largest fitrange
    if index is None and value is None:
        index,value,drop=default_index()
    else:
        drop=index+drop
    filtered_frame=choose_data(frame,index,value,drop)
    result_agg1 = method_average(filtered_frame, agg1,agg_keys)
    result_agg2 = method_average(filtered_frame, agg2,agg_keys)
    return pd.concat((result_agg1,result_agg2),keys=fixed)

def default_index():
    index=['Lattice Artefact','fit_end']
    value=['None',2.50]
    drop=index+['c','fit_start']
    return index,value,drop

def choose_data(frame,index,value,drop):
    """
    Parameters
    ----------
    frame: DataFrame from which data gets chosen
    index: list, columnnames to choose from
    values: list, corresponding values to choose
    drop: list, columnnames to drop

    Returns
    -------
    chosen: Dataframe without NaN-entries and matching values for corresponding
    columns
    """
    for i,v in zip(index,value):
        frame=frame.where(frame[i] == v)
    chosen=frame.drop(drop,1).dropna()
    return chosen
    
def method_average(filtered,agg,agg_keys=None):
    """Take average over keys in agg
    """
    if agg_keys is not None:
        if len(agg) != len(agg_keys):
            print("keys and values have different lenghts")
        for i,n in enumerate(agg_keys):
            filtered  = filtered.where((filtered[n].isin(agg[i]))).dropna()
        method_filtered = filtered
    else:
        method_filtered = filtered.where((filtered['method'].isin(agg))).dropna()
    average = method_filtered.groupby('sample').mean().reset_index()
    mean = average.apply(bstats.own_mean).drop('sample',0)
    std = average.apply(bstats.own_std).drop('sample',0)
    result = pd.concat((mean,std),keys=['mean','std'],axis=1)
    return result

def get_weights(df,obs,err_meth='abs',rel=True):
    """ Get weights for a dataframe comprising only one set of observations
    """
    # weights are calculated per fitrange
    # prepare dataframe for queries
    df['fr'] = df['fr_bgn'].astype(str)+','+df['fr_end'].astype(str)
    df['boot'] = df['sample'].astype(str)
    # get statistics of obs with fitranges as index, collect all data for weights there
    stats = bootstrap_means(df,['fr'],obs)
    stats.reset_index(inplace=True)
    # Reset index also resets column multilevel
    if rel is True:
        stats['error'] = stats['own_std']/stats['own_mean']
    else:
        # this option uses the distribution mean for calculating the statistical error.
        # I think that this is implemented in the analysis as well
        # TODO: implement that as an option?
        #stats['error'] = df.groupby('fr').std().reset_index()[obs]
        stats['error'] = stats['own_std']
    min_err = stats['error'].abs().min()
    # get pvalues of original data
    pvals = df[['fr','p-val']].loc[df['boot']=='0']
    stats = stats.set_index('fr').join(pvals.set_index('fr')).reset_index()
    # calculate weights
    stats['weights']=((1.-2.*np.abs(stats['p-val']-0.5))*min_err/stats['error'])**2
    df = df.merge(stats[['fr','weights']],left_on='fr',right_on='fr')
    return df

def weighted_mean_sample(df,obs):
    # \bar{x} = \frac{\sum_{i=1}^N w_i*x_i}/\sum_{i}^N w_i
    wm=pd.DataFrame()
    wm[obs] = df[obs]*df['weights']/df['weights'].sum()
    return wm.sum()
#def weighted_average(df,obs,groups):
#    """Compute a weighted average of observables over groups
#    """
#    weight_avg = pd.DataFrame()
#    # Observables are columns
#    for o in obs:
#	obs_avg=pd.DataFrame()
#	obs_avg = df.groupby(groups)[[o,'sample','fr_bgn','fr_end','ChPT','poll']].agg(get_weights)
#	weight_avg = weight_avg.merge(obs_avg)
#    return weight_avg

def whole_average(frame,obs_keys=None):
    if obs_keys is None:
        method_means = frame.groupby('sample')['L_piK','mu_a32_phys',
                                               'Mpi_a32','Mpi_a12','tau'].median().reset_index()
    else:
        method_means = frame.groupby('sample')[obs_keys].mean().reset_index()
    mean = method_means.apply(bstats.own_mean).drop('sample',0)
    std = method_means.apply(bstats.own_std).drop('sample',0)
    result = pd.concat((mean,std),keys=['mean','std'],axis=1)
    return result
    
def partial_average(frame,fixed,obs_keys=None):
    """Take average over all but one method

    Parameters
    ----------
    frame: DataFrame, the raw data
    fixed: string, column name to fix

    Returns
    -------
    concatenated dataframe 
    """
    # need mean over all methods but 
    method_filtered = frame.groupby([fixed,'sample']).mean().reset_index()
    if obs_keys is None:
        result = bootstrap_means(method_filtered,[fixed],['L_piK','mu_a32_phys', 'Mpi_a32','Mpi_a12','tau'])
    else:
        result = bootstrap_means(method_filtered,[fixed],obs_keys)
    return result

def combine_averages(mean_frame,fixed_frame_list):
    """Combine one frame with mean values from other frames
    """
    new_columns = pd.MultiIndex.from_tuples([('original','mean'),('original','std')])
    orig_final = pd.DataFrame(mean_frame.values,index=mean_frame.index,columns=new_columns)
    for f in fixed_frame_list:
        f_T = f.transpose().xs('own_mean',level=1)
        tmp_columns = pd.MultiIndex.from_product([[f_T.columns.name],f_T.columns.values])
        orig_final[tmp_columns] = f_T
    return orig_final

def spreads(mean_frame):
    # The resulting frame should contain the mean value the standard deviation
    # and one multicolumn for each entry in fixed_frame_list
    # Loop over multiindex columns
    spread = mean_frame
    spread.columns=['-'.join(col) for col in spread.columns]
    spread = spread.sub(spread['original-mean'],axis=0)
    spread['original-mean']=mean_frame['original-mean']
    spread['original-std']=mean_frame['original-std']
    my_tuples = [i.split('-') for i in spread.columns]
    spread.columns = pd.MultiIndex.from_tuples(my_tuples)
    return spread

def get_systematics(data,source_keys,observable_keys=None):
    """Get a dataframe with systematic uncertainties from different methods

    Take median over all methods as mean-value and use samples of mean for
    statistical error.
    For every key in source_keys take the partial average over the remaining
    methods. The spread between the median and the partial median determines the
    systematic uncertainty. This is then combined in the end

    Parameters:
    -----------
    data:   DataFrame, raw data taken into account for estimate
    source_keys:    list of strings, labels of the sources of systematic errors
                    (Must not contain dashes ('-') in names)
    observable_keys: list of strings, labels of the observables to estimate
                     systematic effects for
    Returns:
    --------
    estimate: DataFrame, the first two columns are mean and statistical error,
              following columns are the differences to the partial medians
    """
    median_over_all = whole_average(data,observable_keys)
    partial_median_list = []
    for k in source_keys:
        partial_median_list.append(partial_average(data,k,observable_keys))
    combined = combine_averages(median_over_all,partial_median_list)
    estimate = spreads(combined)
    return estimate

def combine(main,sys=None,names=None):
    
    orig_columns = pd.MultiIndex.from_tuples([('original','mean'),('original','std')])
    tmp_columns = pd.MultiIndex.from_product([names,['+','-']])
    fix_ms_up = main['mean']-sys[0].loc['Zp1']['mean']
    fix_ms_dn = main['mean']-sys[0].loc['Zp2']['mean']
    zp_up = main['mean']-sys[1].loc['A']['mean']
    zp_dn = main['mean']-sys[1].loc['B']['mean']
    orig_final = pd.DataFrame(main.values,index=main.index,columns=orig_columns)
    final = pd.concat((fix_ms_up,fix_ms_dn,zp_up,zp_dn), axis=1, keys=tmp_columns)
    return orig_final.merge(final,left_index=True,right_index=True) 
    #average =
    #return average
