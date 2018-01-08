#!/bin/bash

infile="/hiskp2/helmes/projects/analysis-code/ini/pi_K/I_32/chiral_analysis_mua0_zp1.ini"
./pik_I32_nlo_chipt ${infile} A > pik_I32_nlo_chipt_A1.log 
./pik_I32_nlo_chipt ${infile} B > pik_I32_nlo_chipt_B1.log
./pik_I32_gamma ${infile} A > pik_I32_gamma_A1.log
./pik_I32_gamma ${infile} B > pik_I32_gamma_B1.log
infile="/hiskp2/helmes/projects/analysis-code/ini/pi_K/I_32/chiral_analysis_mua0_zp2.ini"
./pik_I32_nlo_chipt ${infile} A > pik_I32_nlo_chipt_A2.log
./pik_I32_nlo_chipt ${infile} B > pik_I32_nlo_chipt_B2.log
./pik_I32_gamma ${infile} A > pik_I32_gamma_A2.log
./pik_I32_gamma ${infile} B > pik_I32_gamma_B2.log

