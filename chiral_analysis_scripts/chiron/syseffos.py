import pandas as pd
import syseffos_io as io
import syseffos_info as info
import boot_statistics as bstats

class SysEffos:
    def __init__(self,directory=None,filenames=None,debug=0):
        self.debug=debug
        if filenames is not None and directory is not None:
            self.data=io.get_dataframe_disk(directory,filenames)
            self.add_index_as_column('sample')
            if self.debug > 0:
                self.data.info()

    def add_index_as_column(self,name):
        # Sooner or later we need the sample_number as variable
        self.data[name]=self.data.index
    
    def bootstrap_means(self,names,observables):
        bstrap_frame=self.data.groupby(names)[observables].agg([bstats.own_mean,
                                                       bstats.own_std])
        info.print_si_format(bstrap_frame)
        return bstrap_frame
    
    def bootstrap_means_key(self,groups,observables):
        # mean over fitranges, needs group by sample
        self.key_mean_frame=self.data.groupby(groups+['sample']).mean()
        # undo grouping by samples
        self.key_mean_frame=self.data.reset_index(level=['sample'])
        self.mean_frame=self.bootstrap_means(groups,observables)

#    def result_weights():
#        self.data.pipe(bstats.own_weight)
