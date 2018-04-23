import pandas as pd
import numpy as np
def extend_data(dictionary,pandas_index):
    _repeats = len(pandas_index)
    for _k in dictionary.keys():
        dictionary[_k] = np.repeat(np.asarray(dictionary[_k]),_repeats)
    return dictionary
