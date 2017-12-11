import pandas as pd
import syseffos_io as io
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
        self.mean_frame=self.data.groupby(names)[observables].agg([bstats.own_mean,
                                                       bstats.own_std])
    
    def bootstrap_means_key(groups,observables,mean_key=None):
        if mean_key is not None:
           # mean over fitranges, needs group by sample
           self.key_mean_frame=self.data.groupby(groups+'sample').mean()
           # undo grouping by samples

        else:
            print("no key given, returning bootstrap_means instead")
            bootstrap_means
#    def result_weights():
#        self.data.pipe(bstats.own_weight)
