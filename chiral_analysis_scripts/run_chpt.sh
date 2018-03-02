#!/bin/bash
# e is one of '', '_fp' or '_div' corresponding to E1, E2 and E3
e='_div'
meth=( A B )
zp=( 1 2 )
for m in "${meth[@]}"; do
    for z in "${zp[@]}"; do
        infile="../ini/pi_K/I_32${e}/chiral_analysis_mua0_zp${z}.ini"
        ./pik_I32_nlo_chipt ${infile} ${m} > ./logfiles/pik_I32_nlo_chipt_${m}${z}${e}.log
        ./pik_I32_gamma ${infile} ${m} > ./logfiles/pik_I32_gamma_${m}${z}${e}.log
    done
done
#infile="/hiskp4/helmes/projects/analysis-code/ini/pi_K/I_32/chiral_analysis_mua0_zp1.ini"
#./pik_I32_nlo_chipt ${infile} A > pik_I32_nlo_chipt_A1_fp.log 
#./pik_I32_nlo_chipt ${infile} B > pik_I32_nlo_chipt_B1_fp.log
#./pik_I32_gamma ${infile} A > pik_I32_gamma_A1_fp.log
#./pik_I32_gamma ${infile} B > pik_I32_gamma_B1_fp.log
#infile="/hiskp4/helmes/projects/analysis-code/ini/pi_K/I_32/chiral_analysis_mua0_zp2.ini"
#./pik_I32_nlo_chipt ${infile} A > pik_I32_nlo_chipt_A2_fp.log
#./pik_I32_nlo_chipt ${infile} B > pik_I32_nlo_chipt_B2_fp.log
#./pik_I32_gamma ${infile} A > pik_I32_gamma_A2_fp.log
#./pik_I32_gamma ${infile} B > pik_I32_gamma_B2_fp.log

