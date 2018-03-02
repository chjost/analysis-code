"""
Debug information printed to screen:
"""

import numpy as np
import re
from statistics import compute_error

def print_info_data(x,y,dy=None,dx=None):
    """ print info on data for a plot
    """
    print("\n--------------------------------------------------------------------------------")
    print("Data for plotting")
    np.set_printoptions(precision=4)
    if dx is not None and dy is not None:
        print("#x\tdx\ty\tdy:")
        data=np.column_stack((x,dx,y,dy))
    elif dx is not None and dy is None:
        print("#x\tdx\ty:")
        data=np.column_stack((x,dx,y))
    elif dx is None and dy is not None:
        print("#x\ty\tdy:")
        data=np.column_stack((x,y,dy))
    else:
        print("#x\ty:")
        data=np.column_stack((x,y))
    print(re.sub('[\[\]]', '', np.array_str(data)))
    print("--------------------------------------------------------------------------------\n")

def init_summ_arrays(extrapol,lyt_x,lyt_y):
    """
    Initialize empty data arrays for summary
    """
    # x and y are raw data arrays of shape [range,dim,nsamples]
    # calculate mean values and error for each
    # Initialize an array for x values and x-errors
    # calc number of ensembles
    nb_ens = np.sum(lyt_x[1])
    x_shape = (nb_ens*lyt_x[2],lyt_x[3])
    y_shape = (nb_ens*lyt_y[2],lyt_y[3])
    _x = np.zeros(x_shape)
    _dx = np.zeros_like(_x)
    _y = np.zeros(y_shape)
    _dy = np.zeros_like(_y)
    # values for matching
    _mu = np.zeros_like(_y)
    _dmu = np.zeros_like(_y)
    return _x, _y, _mu, _dx, _dy, _dmu

def fill_arr_summ(raw, arr, darr, lyt_raw):
    """
    Fill a data array from raw chiral analysis data

    Parameters
    ----------
    raw: list of ndarrays, raw data of a ChirAna object
    arr: ndarray, the summary data gets stored here
    darr: ndarray, the error of the summary data gets stored here
    lyt_raw: list, the layout of raw
    """
    #TODO: Code coubling, bad style
    ens = lyt_raw[1]
    if (len(ens) >= 1) and (ens[0] > 0):
        #A Ensembles
        for i in range(ens[0]):
            for j in range(lyt_raw[2]):
                for p in range(lyt_raw[3]):
                    arr[lyt_raw[2]*i+j,p], darr[lyt_raw[2]*i+j,p] = compute_error(raw[0][i,j,p])
                    #print(lyt_raw[2]*i+j,i,j,p, arr[i+j,p])

    if (len(ens) >= 2) and (ens[1] > 0):
        #B Ensembles
        # calculate offset
        off1 = ens[0]*lyt_raw[2]
        for i in range(ens[1]):
            for j in range(lyt_raw[2]):
                for p in range(lyt_raw[3]):
                    arr[off1+lyt_raw[2]*i+j,p], darr[off1+lyt_raw[2]*i+j,p] = compute_error(raw[1][i,j,p])
    if (len(ens) >= 3) and (ens[2] > 0):
        #D Ensembles
        # calculate offset
        off = (ens[0]+ens[1])*lyt_raw[2]
        for i in range(ens[2]):
            for j in range(lyt_raw[2]):
                for p in range(lyt_raw[3]):
                    arr[off+lyt_raw[2]*i+j,p], darr[off+lyt_raw[2]*i+j,p] = compute_error(raw[2][i,j,p])

def print_line_latex(lat, dx, dy, dm=None, prec=1e4):
  """Print summary line.

  Parameter
  ---------
  lat : str
      The lattice name
  d : tuple, list
      The data
  """
  if dx.shape[0] == 2:
    if dm is None:
      print("%9s & NA & $%.4f(%1.0f)(^{+%1.0f}_{-%1.0f})$ &$%.4f(%1.0f)(^{+%1.0f}_{-%1.0f})$ & $%.4f(%1.0f)(^{+%1.0f}_{-%1.0f})$ \\\\" % 
                  (lat, dx[0][0], dx[0][1]*prec, dx[0][2]*prec, dx[0][3]*prec,
                    dx[1][0], dx[1][1]*prec, dx[1][2]*prec, dx[1][3]*prec,
                    dy[0], dy[1]*prec, dy[2]*prec, dy[3]*prec))
    else:
      print("%9s & $%.4f(%1.0f)$ & $%.4f(%1.0f)(^{+%1.0f}_{-%1.0f})$ &$%.4f(%1.0f)(^{+%1.0f}_{-%1.0f})$ & $%.4f(%1.0f)(^{+%1.0f}_{-%1.0f})$ \\\\" % 
                  (lat, dm[0],dm[1]*prec,dx[0][0], dx[0][1]*prec, dx[0][2]*prec, dx[0][3]*prec,
                    dx[1][0], dx[1][1]*prec, dx[1][2]*prec, dx[1][3]*prec,
                    dy[0], dy[1]*prec, dy[2]*prec, dy[3]*prec))

  else:
    print("%9s & NA & $%.4f(%1.0f)(^{+%1.0f}_{-%1.0f})$  & $%.4f(%1.0f)(^{+%1.0f}_{-%1.0f})$ \\\\" % 
                (lat, dx[0][0], dx[0][1]*prec, dx[0][2]*prec, dx[0][3]*prec,
                  dy[0], dy[1]*prec, dy[2]*prec, dy[3]*prec))

def cus_str(x): return '%.4f' %x

def print_summary(extrapol,header,mul,mus):
    """This function should print a summary of the whole chiral analysis,
    preferably in latex format
    """
    _lyt_x = extrapol.x_shape
    _lyt_y = extrapol.y_shape
    # Initialize empty data arrays
    _x, _y, _mu, _dx, _dy, _dmu = init_summ_arrays(extrapol,_lyt_x,_lyt_y)

    # Fill the data arrays
    fill_arr_summ(extrapol.x_data,_x,_dx,_lyt_x)         
    fill_arr_summ(extrapol.y_data,_y,_dy,_lyt_y)
    #extrapol.amu_matched_to = None
    if extrapol.match is True:
        fill_arr_summ(extrapol.amu_matched_to,_mu,_dmu,_lyt_y)
    # put light and strange quark masses before it
    _mul = np.repeat(np.hstack(mul.values()),_lyt_x[2])
    _tmp = []
    # start with ascii format
    # print the header first
    print(' '.join(header))
    for l in range(_x.shape[0]):
        _line=[]
        #for each line build up interlace of _x,_dx,_y,_dy
        _line.append(_mul[l])
        #_line.append(_mus[l])
        for c in range(_x.shape[1]):
            _line.append(_x[l,c])
            _line.append(_dx[l,c])
        for c1 in range(_y.shape[1]):
            _line.append(_y[l,c1])
            _line.append(_dy[l,c1])
        for c2 in range(_y.shape[1]):
            _line.append(_mu[l,c1])
            _line.append(_dmu[l,c1])
        print(' '.join(map(cus_str,_line)))
    
    # print parameters of fitresult
    if extrapol.fitres is not None:
        # number of parameters
        npar = extrapol.fitres.data[0].shape[1]
        print("\n#--------------")
        print("# Summary of fit parameters:" )
        for p in range(npar):
            extrapol.fitres.print_data(par=p, tex=False)
        if extrapol.fit_stats is not None:
            for i,d in enumerate(extrapol.fit_stats):
                # chi^2/dof
                print("Chi^2/d.o.f:\t%.03f\t(%.02f/%d)"%(d[1]/d[0],d[1],d[0]))
                if extrapol.correlated is True:
                    print("p-val.:\t%.03f"%d[2])
                
    # print physical point result
    if extrapol.phys_point is not None:
        print("\n#--------------")
        print("# Physical Point result:")
        print("x: %f +/- %f" %(extrapol.phys_point[0][0],extrapol.phys_point[0][1]))
        print("y %f +/- %f\n" %(extrapol.phys_point[1][0],extrapol.phys_point[1][1]))

