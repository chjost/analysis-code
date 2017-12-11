import pandas as pd
import numpy as np

def own_std(series,level=None):
    # 0th bootstrapsample poses mean value of measurement.
    #if level is not None:
    mean = series.iloc[0]
    difference = np.diff(series-mean)
    variance = np.nansum(np.square(difference))/series.size
    std = np.sqrt(variance)
    return std

def own_mean(series):
    return series.iloc[0]

def systematic_average(series):
    print(series.loc[0].size)
    mean=np.sum(series.loc[0])/series.loc[0].size
    return mean

def own_weight(table):
    """Compute the weight for the dataframe results
    weighting function is given by
    
    """ 
    # TODO: At the moment this is a dummy variable
    # Get table length
    length = len(table.index)
    print("Table has %d rows" %length)
    table['weight'] = pd.Series(np.ones((length,)), index=table.index)
    

def systematics_by_methods(results,keys,column):
    system_list=[]
    for k in keys:
        sys=results[results['method'].isin(k)]
        sys=keywise_mean(sys,column)
        print(sys)
        system_list.append(sys)
    return system_list

def keywise_mean(frame,keys):
    key_mean=frame.groupby('nb').mean()[keys].agg([own_mean,own_std])
    return key_mean

