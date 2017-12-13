import pandas as pd
import syseffos_io as io
import syseffos_info as info
import boot_statistics as bstats

def bootstrap_means(frame,names,observables):
    bstrap_frame=frame.groupby(names)[observables].agg([bstats.own_mean,
                                                   bstats.own_std])
    info.print_si_format(bstrap_frame)
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

def average_all_methods(frame,agg):
    # need only lattice artefact 'None' and fit_end 2.5
    # TODO: I cannot shake off the feeling that this should work simpler
    # Look at Lattice artefact None and largest fitrange
    filtered_frame=frame.where((frame['Lattice Artefact'] == 'None') &
                               (frame['fit_end'] == 2.50)).drop(['Lattice Artefact',
                                'c', 'fit_start','fit_end'],1).dropna()
    return method_average(filtered_frame,agg)


def average_methods(frame,agg1,agg2,fixed=None):
    """Take systematic 

    """
    # need only lattice artefact 'None' and fit_end 2.5
    # TODO: I cannot shake off the feeling that this should work simpler
    # Look at Lattice artefact None and largest fitrange
    filtered_frame=frame.where((frame['Lattice Artefact'] == 'None') &
                               (frame['fit_end'] == 2.50)).drop(['Lattice Artefact',
                                'c', 'fit_start','fit_end'],1).dropna()
    result_agg1 = method_average(filtered_frame, agg1)
    result_agg2 = method_average(filtered_frame, agg2)
    return pd.concat((result_agg1,result_agg2),keys=fixed)

    
def method_average(filtered,agg):
    method_filtered = filtered.where((filtered['method'].isin(agg))).dropna()
    average = method_filtered.groupby('sample').mean().reset_index()
    mean = average.apply(bstats.own_mean).drop('sample',0)
    std = average.apply(bstats.own_std).drop('sample',0)
    result = pd.concat((mean,std),keys=['mean','std'],axis=1)
    return result

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
