#!/bin/bash
# e is one of '', '_fp' or '_div' corresponding to E1, E2 and E3
# e is '_final'
e='_cov_false'
# We have two methods for fixing the strange quark mass, labelled A and B
meth=( A )
# Furthermore there are two values for Z_P around (choose 1, 2)
zp=( 1 )
# Lastly we employ three methods for extracting E_piK ( E1 E2 E3 )
# only needed in interpolation
epik=( E1 E2 )

# Fix strange quark mass for B

#for z in "${zp[@]}"; do
#    infile="../ini/pi_K/I_32${e}/chiral_analysis_mua0_zp${z}.ini"
#    ./pik_I32_B_fixms.py ${infile} > ./logfiles/fixms_B_${z}${e}.log
#    #./pik_I32_B_fixms_wopmu.py ${infile} > ./logfiles/fixms_B_wopmu_${z}${e}.log
#done

for m in "${meth[@]}"; do
    for z in "${zp[@]}"; do
        for e_meth in "${epik[@]}"; do
            infile="../ini/pi_K/I_32${e}/chiral_analysis_mua0.ini"
            sed  -i "s/.*epik_meth.*/epik_meth=${e_meth}/" ${infile}
            ./pik_I32_${m}_interpolate.py ${infile} > ./logfiles/interpolate_${m}_${z}_${e_meth}${e}.log
        done
    done
done
           
