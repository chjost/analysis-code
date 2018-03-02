import format_val_err
import pandas as pd

def si_format(frame):
    #print(frame)
    formatted=[format_val_err.unitprint(a,b) for a,b in
            zip(frame.own_mean,frame.own_std)]
    return formatted

def print_si_format(frame):
    # new dataframe
    formatted=pd.DataFrame(index=frame.index,
                            columns=frame.columns.levels[0])
    for k in frame.columns.levels[0]:
        formatted[k]=si_format(frame[k])
    pd.set_option('display.width',None)
    print(formatted)
    return formatted

