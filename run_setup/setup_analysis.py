#!/usr/bin/python
import os
import subprocess as sub

def create_flds(beta,ens_dict,mus_dict,folders,exec_dir):
    ensembles = ens_dict[beta]
    strange_parameters = mus_dict[beta]
    for f in folders:
        for e in ensembles:
            for s in strange_parameters:
                foldername = exec_dir+'/'+f+'/'+e+'/'+s
                os.makedirs(foldername)
def main():
    exec_dir = "/hiskp4/helmes/analysis/scattering/pi_k/I_32_cov_false"
    
    folders = ["data","plots","results","runs"]

    ens ={'A':["A30.32","A40.20", "A40.24", "A40.32",
             "A60.24", "A80.24", "A100.24"],
          'B':["B35.32","B55.32","B85.24"],
          'D':["D45.32"], 'D_var':["D30.48"]}

    mus_fld = {'A':["amu_s_185","amu_s_225","amu_s_2464","pi"],
               'B':["amu_s_16","amu_s_186","amu_s_21","pi"],
               'D':["amu_s_13","amu_s_15","amu_s_18","pi"],
               'D_var': ["amu_s_115","amu_s_15","amu_s_18","pi"]}
# Set up all that is needed for an analysis project
# Create folder structure
    create_flds('A',ens,mus_fld,folders,exec_dir)
    create_flds('B',ens,mus_fld,folders,exec_dir)
    create_flds('D',ens,mus_fld,folders,exec_dir)
    create_flds('D_var',ens,mus_fld,folders,exec_dir)
# Copy templates to run folders
    src = "/hiskp4/helmes/projects/analysis-code/run_setup/templates"
    dst = exec_dir+"/runs/"
    sub.call(["rsync", "-av", src, dst])
# Copy init scripts content to run folder
    src = "/hiskp4/helmes/projects/analysis-code/run_setup/init_scripts/"
    dst = exec_dir+"/runs/"
    sub.call(["rsync", "-av", src, dst])
    
# make this script importable, according to the Google Python Style Guide
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
