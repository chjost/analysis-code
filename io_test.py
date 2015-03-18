#!/usr/bin/python
################################################################################
#
# Author: Christopher Helmes (helmes@hiskp.uni-bonn.de)
# Date:   Februar 2015
#
# Copyright (C) 2015 Christopher Helmes
# 
# This program is free software: you can redistribute it and/or modify it under 
# the terms of the GNU General Public License as published by the Free Software 
# Foundation, either version 3 of the License, or (at your option) any later 
# version.
# 
# This program is distributed in the hope that it will be useful, but WITHOUT 
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS 
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with tmLQCD. If not, see <http://www.gnu.org/licenses/>.
#
################################################################################
#
# Function: At the moment this is a test program, this file will change a lot
#
# For informations on input parameters see the description of the function.
#
################################################################################
import bootstrap
import fit
import input_output
#import corr_matrix

def concatenate(*lists):
      return chain(lists)

def main():
  # Test extraction of ascii dump
  correl = input_output.extract_ascii_corr_fct("/hiskp2/helmes/correlators/test/A40.24/corrs.",1870,4,15,1,7,48)
  #print correl[0]
  boot = bootstrap.sym_and_boot(correl[0],48,15,1000)
  print len(boot)
  fit.fitting()
# make this script importable, according to the Google Python Style Guide
if __name__ == '__main__':
    main()

