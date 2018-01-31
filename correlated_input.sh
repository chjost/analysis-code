#!/bin/bash

./strange_all_in_hdf5.py $1
./omit_outliers $1
./plot_corrs.py $1
