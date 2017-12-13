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

def average_systematics(frame,agg1,agg2):
    """Take systematic 

    """
    # need only lattice artefact 'None' and fit_end 2.5
    # TODO: I cannot shake off the feeling that this should work simpler
    # Look at Lattice artefact None and largest fitrange
    filtered_frame=frame.where((frame['Lattice Artefact'] == 'None') &
                               (frame['fit_end'] == 2.50)).drop(['Lattice Artefact',
                                'c', 'fit_start','fit_end'],1).dropna()
    print(filtered_frame.sample(n=25))
    # take methods of agg1
    result_agg1 = method_average(filtered_frame, agg1)
    result_agg2 = method_average(filtered_frame, agg2)
    print(result_agg1)
    print(result_agg2)

    
def method_average(filtered,agg):
    method_filtered = filtered.where((filtered['method'] == agg[0]) |
                                     (filtered['method'] == agg[1])).dropna()
    average=method_filtered.groupby('sample').mean().reset_index()
    mean=average.apply(bstats.own_mean).drop('sample',0)
    std=average.apply(bstats.own_std).drop('sample',0)
    result=pd.concat((mean,std),keys=['mean','std'],axis=1)
    return result

    #average =
    #return average
