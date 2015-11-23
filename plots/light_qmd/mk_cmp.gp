set title "Kaon mass comparison"
set xlabel "M_pi^2 in (GeV)^2"
set ylabel "M_k in MeV"

file='mk_cmp.dat'

set terminal postscript solid colour landscape lw 1.5
set output 'mk.eps'
set grid

plot file i 0 u (($2*197.97*1e-3/0.086)**2):($5*197.97/0.086):($6*197.97/0.086) w yerr t 'amu_s = 0.225',\
     file i 1 u (($2*197.97*1e-3/0.086)**2):($5*197.97/0.086):($6*197.97/0.086) w yerr t 'amu_s = 0.2464',\
     file i 2 u (($2*197.97*1e-3/0.125)**2):($5*197.97/0.125):($6*197.97/0.125) w yerr t 'NPLQCD'

#replot file i 0 u ($10/$8):($4*$12) t 'amu_s = 0.0185',\
#       file i 1 u ($10/$8):($4*$12) t 'amu_s = 0.0225',\
#       file i 2 u ($10/$8):($4*$12) t 'amu_s = 0.02464'

#plot file i 0 u ($10/$8):22:23 w yerr t 'A40.24'
#plot file i 1 u ($10/$8):22:23 w yerr t 'A60.24'
#plot file i 2 u ($10/$8):22:23 w yerr t 'A80.24'
#plot file i 3 u ($10/$8):22:23 w yerr t 'A100.24'
