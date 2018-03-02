#!/bin/bash

./strange_all_in.py $1
./omit_outliers $1
./plot_corrs.py $1
