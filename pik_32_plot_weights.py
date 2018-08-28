#!/usr/bin/python
# Plot weights as a function of the final timeslice for E1 and E3
import argparse
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
plt.style.use('paper_standalone')
import numpy as n
import pandas as pd
import sys
# Christian's packages
sys.path.append('/hiskp4/helmes/projects/analysis-code/')
import analysis2 as ana
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ens",help="directoryname",type=str,required=True)
    parser.add_argument("--mus",help="bare strange mass",type=str,required=True)
    args=parser.parse_args()
    path = '/hiskp4/helmes/analysis/scattering/pi_k/I_32_blocked/'
    datadir = "%sdata/%s/amu_s_%s/" %(path,args.ens,args.mus)
    plotdir = "%splots/%s/amu_s_%s/" %(path,args.ens,args.mus)
    storer=pd.HDFStore(datadir+'fit_pik.h5')
    result = storer.get('summary')
    data = result[result['sample']==0]
    print(data)
    t_initial = data['t_i'].unique()
    # Open pdfpages
    with PdfPages(plotdir+'weights.pdf') as pdf:
	    # TODO: use iterrows for that
	    for t_i in t_initial:
	        t_i_data = data[data['t_i']==t_i]
		print(t_i_data)
	        for p in ['E1','E3']:
	            poll_data = t_i_data[t_i_data['poll']==p]
	            x = poll_data['t_f']
	            y = poll_data['weight']
	            plt.errorbar(x,y,label=p,fmt='o')
		plt.title(r't_i = %d'%t_i)
		plt.legend()
		plt.xlabel(r't_f')
		plt.ylabel(r'weight')
		pdf.savefig()
		plt.close()
if __name__ == '__main__':
    try:
        print("starting")
        main()
    except KeyboardInterrupt:
        pass

