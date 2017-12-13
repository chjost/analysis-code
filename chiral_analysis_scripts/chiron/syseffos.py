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
