import numpy as np
# Calculate number of bins for histogram from array of fitranges
def get_hist_bins(fitranges_shape):
    nbins1 = fitranges_shape[-1][0]/4
    print(nbins1)
    if nbins1 >= 0:
        if nbins1 > 10:
          nbins = nbins1
        else:
          nbins = 10
    return nbins

if __name__ == "__main__":
  pass
