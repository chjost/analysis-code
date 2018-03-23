
import pandas as pd

splitter = lambda x: x.split('/')[1].split('.')[0].rsplit('_')[-2]

def get_dataframe_disk(directory,filenames):
    method_ids, results_as_list = read_unfiltered(directory,filenames)
    result_frame = build_dataframe(results_as_list,method_ids)
    return result_frame

def read_unfiltered(directory,filenames):
    print(ending(filenames))
    if ending(filenames) == "pkl":
        method_ids, results_as_list = read_unfiltered_pickle(directory,filenames)
    elif ending(filenames) == "h5":
        method_ids, results_as_list = read_unfiltered_hdf(directory,filenames)
    else:
        print("File-extension %s unknown" % ending(filenames))
        method_ids, results_as_list = None, None
    return method_ids, results_as_list

def ending(filenames):
    return filenames[0].split('.')[-1]

def read_unfiltered_pickle(directory,filenames):
    # get method names from list of filenames, TODO: place somewhere else
    fix_ms_methods = mqfix_methods_as_list(filenames)
    results_list = pickle_list_from_disk(directory,filenames)
    return fix_ms_methods, results_list

def read_unfiltered_hdf(directory,filenames):
    # get method names from list of filenames, TODO: place somewhere else
    fix_ms_methods = mqfix_methods_as_list(filenames)
    results_list = hdf_list_from_disk(directory,filenames)
    return fix_ms_methods, results_list

def mqfix_methods_as_list(filenames):
    method_list=[splitter(f) for f in filenames]
    return method_list

def pickle_list_from_disk(directory,filenames):
    dataframe_list=[ pd.read_pickle(directory+f) for f in filenames ]
    return dataframe_list

def hdf_list_from_disk(directory,filenames):
    dataframe_list=[ pd.read_hdf(directory+f) for f in filenames ]
    return dataframe_list

#TODO: function too long, split that up somehow
def build_dataframe(results_list,fix_ms_methods):
    results_dict = { key:value for key,value in zip(fix_ms_methods,
                     results_list) }
    # New column for method used
    for k,v in zip(results_dict.keys(),results_dict.values()):
        v.loc[:,'method'] = pd.Series(k,index=v.index)
    # make a great table with all data.
    combined_methods = pd.concat(results_dict.values())
    combined_methods['sample'] = combined_methods.index
    return combined_methods

