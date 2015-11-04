#!/usr/bin/python
################################################################################
#
# Author: Christian Jost (jost@hiskp.uni-bonn.de),
#         Christopher Helmes (helmes@hiskp.uni-bonn.de)
# Date:   October 2015
#
# Copyright (C) 2015 Christian Jost, Christopher Helmes
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
# Function: Parse the inputfiles of contraction code to build filenames
#
###############################################################################

import configparser as cp

def inputnames(conf_file, corr_string):
    """ Function to build Correlator input names conformal with B. Knippschilds
    naming scheme
        
    The function invokes a config parser which reads the config file (config.ini) regarding the
    specified correlation functions and returns a list of input names

    Args:
        conf_file: string with path to the configuration file 
        corr_string: array of identifiers for correlation functions to be
                    read in
    Returns:
        A list of inputnames

    """
    # invoke config parser and read config file
    config = cp.ConfigParser()
    config.read(conf_file)
    # set section names for get operations, have to be the same in configfile
    quarks = config['quarks']
    operators = config['operator_lists']
    corr_list = config['correlator_lists']
    # result list
    inputnames = []
    # Loop over correlation function names
    for c in corr_list:
        # get value of each key splited by ':'
        c0 = corr_list.get(c).split(sep=':')
        # read only functions in corr_string, sort them into q and op arrays
        if c0[0] in corr_string:
            q_list = []
            op_list = []
            for val in c0[1:]:
                if val[0] == 'Q':
                    q_list.append(quarks.get(val))
                elif val[0] == 'O':
                    op_list.append(operators.get(val))
                else:
                    print("Identifier not found")
            # TODO: expand entrys in operator lists
            # join single lists together for filename
            join_q = ''.join(q_list)
            join_op = '.'.join(op_list)
            # build the filename
            corrname = c0[0]+"/"+c0[0]+"_"+join_q+"_"+join_op+".dat" 
            inputnames.append(corrname)
    return inputnames

def main():
    f_string = ['C2+','C4+C','C4+D']
    in_name = inputnames('config.ini',f_string)
    print(in_name)
# make this script importable, according to the Google Python Style Guide
if __name__ == '__main__':
  try:
    main()
  except KeyboardInterrupt:
    print("Keyboard Interrupt")
