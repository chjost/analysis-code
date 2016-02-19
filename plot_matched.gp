#set title "K K Scattering length"
set xlabel "(r_0 M_{Pi})^2"
set ylabel "M_Ka_0"
set xrange [0:1.5]
set yrange [-0.5:-0.15]
#set grid
r0_A2=5.31**2
r0_B2=5.77**2
r0_D2=7.60**2
mpi_r0_phys = 0.3525029

set style line 1 lc rgb "red" pt 18
set style line 2 lc rgb "blue" pt 22
set style line 3 lc rgb "dark-green" pt 20
set style line 4 lc rgb "black" pt 17
#plot statistical errors first (they are larger)
set terminal pdfcairo enhanced solid color size 7,5 lw 2 font ",16"
set out './kk_scat_AB_match_to_D.pdf'
#plot 'test.dat' u 2:3:1 w labels point offset character 0,character 1
#set arrow from (mpi_r0_phys**2),-0.42 to (mpi_r0_phys**2),-0.24 nohead
# Plot overview only A ensembles 
#plot './kk_beta_matched.dat' i 0 u ($2**2)*r0_A2:9:($9-$10-$12):($9+$10+$11) ls 1 w errorbars notitle, \
# './kk_beta_matched.dat' i 0 u ($2**2)*r0_A2:9:13 w labels point offset 0,-3 notitle, \
# './kk_beta_matched.dat' i 0 u ($2**2)*r0_A2:9:10 w yerr ls 1 t 'b1.90, \mu_s = 0.02125 (int.)', \
# './kk_beta_matched.dat.bkp2' i 0 u ($2**2)*r0_A2:9:($9-$10-$12):($9+$10+$11) ls 2 w errorbars notitle, \
# './kk_beta_matched.dat.bkp2' i 0 u ($2**2)*r0_A2:9:10 w yerr ls 2 t 'b1.90, old \mu_s = 0.02125 (int.)'
#
#plot './kk_beta_matched.dat' i 1 u ($2**2)*r0_A2:9:($9-$10-$12):($9+$10+$11) ls 1 w errorbars notitle, \
# './kk_beta_matched.dat' i 1 u ($2**2)*r0_A2:9:13 w labels point offset 0,-3 notitle, \
# './kk_beta_matched.dat' i 1 u ($2**2)*r0_A2:9:10 w yerr ls 1 t 'b1.90, \mu_s = 0.02125 (ext.)', \
# './kk_beta_matched.dat.bkp2' i 1 u ($2**2)*r0_A2:9:($9-$10-$12):($9+$10+$11) ls 2 w errorbars notitle, \
# './kk_beta_matched.dat.bkp2' i 1 u ($2**2)*r0_A2:9:10 w yerr ls 2 t 'b1.90, old \mu_s = 0.02125 (ext.)'

#set yrange [0:0.15]
#set ylabel "rel. err. (M_Ka_0)"
#plot './kk_beta_matched.dat' i 0 u ($2**2)*r0_A2:($10*(-1)/$9):13 w labels point offset 0,1 notitle, \
# './kk_beta_matched.dat' i 0 u ($2**2)*r0_A2:($10*(-1)/$9)  ls 1 t 'b1.90, \mu_s = 0.02125 (int.)', \
# './kk_beta_matched.dat.bkp2' i 0 u ($2**2)*r0_A2:($10*(-1)/$9) ls 2 t 'b1.90, old \mu_s = 0.02125 (int.)'
#
#plot './kk_beta_matched.dat' i 1 u ($2**2)*r0_A2:($10*(-1)/$9):13 w labels point offset 0,-1 notitle, \
# './kk_beta_matched.dat' i 1 u ($2**2)*r0_A2:($10*(-1)/$9) ls 1 t 'b1.90, \mu_s = 0.02125 (ext.)', \
# './kk_beta_matched.dat.bkp2' i 1 u ($2**2)*r0_A2:($10*(-1)/$9)ls 2 t 'b1.90, old \mu_s = 0.02125 (ext.)'

#plot './kk_beta_matched.dat' i 0 u ($2**2)*r0_A2:9:($9-$10-$11):($9+$10+$12) ls 1 w errorbars notitle, \
# './kk_beta_matched.dat' i 1 u ($2**2)*r0_A2:9:($9-$10-$11):($9+$10+$12) ls 4 w errorbars notitle, \
# './kk_beta_matched.dat' i 2 u ($2**2)*r0_B2:9:($9-$10-$11):($9+$10+$12) ls 2 w errorbars notitle, \
# './kk_beta_matched.dat' i 3 u ($2**2)*r0_D2:9:($9-$10-$11):($9+$10+$12) ls 3 w errorbars notitle, \
# './kk_beta_matched.dat' i 1 u ($2**2)*r0_A2:9:13 w labels point offset 0,-3 notitle, \
# './kk_beta_matched.dat' i 2 u ($2**2)*r0_B2:9:13 w labels point offset 0,2 notitle, \
# './kk_beta_matched.dat' i 3 u ($2**2)*r0_D2:9:13 w labels point offset 0,2 notitle, \
# './kk_beta_matched.dat' i 0 u ($2**2)*r0_A2:9:10 w yerr ls 1 t 'b1.90, \mu_s = 0.02125 (int.)', \
# './kk_beta_matched.dat' i 1 u ($2**2)*r0_A2:9:10 w yerr ls 4 t 'b1.90, \mu_s = 0.02125 (ext.)', \
# './kk_beta_matched.dat' i 2 u ($2**2)*r0_B2:9:10 w yerr ls 2 t 'b1.95, \mu_s = 0.0186', \
# './kk_beta_matched.dat' i 3 u ($2**2)*r0_D2:9:10 w yerr ls 3 t 'b2.10, \mu_s = 0.0150'
# # Spare D-Ensemble

## Plot interpolation
#plot './kk_beta_matched.dat' i 0 u ($2**2)*r0_A2:9:($9-$10-$11):($9+$10+$12) ls 1 w errorbars notitle, \
# './kk_beta_matched.dat' i 2 u ($2**2)*r0_B2:9:($9-$10-$11):($9+$10+$12) ls 2 w errorbars notitle, \
# './kk_beta_matched.dat' i 3 u ($2**2)*r0_D2:9:($9-$10-$11):($9+$10+$12) ls 3 w errorbars notitle, \
# './kk_beta_matched.dat' i 0 u ($2**2)*r0_A2:9:10 w yerr ls 1 t 'b1.90, \mu_s = 0.02125 (int.)', \
# './kk_beta_matched.dat' i 2 u ($2**2)*r0_B2:9:10 w yerr ls 2 t 'b1.95, \mu_s = 0.0186', \
# './kk_beta_matched.dat' i 3 u ($2**2)*r0_D2:9:10 w yerr ls 3 t 'b2.10, \mu_s = 0.0150'
 # Spare D-Ensemble

# Plot extrapolation
plot './kk_beta_matched.dat' i 1 u ($2**2)*r0_A2:9:($9-$10-$11):($9+$10+$12) ls 1 w errorbars notitle, \
 './kk_beta_matched.dat' i 3 u ($2**2)*r0_B2:9:($9-$10-$11):($9+$10+$12) ls 2 w errorbars notitle, \
 './kk_beta_matched.dat' i 4 u ($2**2)*r0_D2:9:($9-$10-$11):($9+$10+$12) ls 3 w errorbars notitle, \
 './kk_beta_matched.dat' i 1 u ($2**2)*r0_A2:9:10 w yerr ls 1 t 'A-Ensembles', \
 './kk_beta_matched.dat' i 3 u ($2**2)*r0_B2:9:10 w yerr ls 2 t 'B-Ensembles', \
 './kk_beta_matched.dat' i 4 u ($2**2)*r0_D2:9:10 w yerr ls 3 t 'D-Ensemble'


 #'./kk_beta_matched.dat' i 4 u ($2**2)*r0_D2:9:10 w yerr ls 3 t 'b2.10, \mu_s = 0.0150'
 #'./kk_beta_matched.dat' i 3 u ($2**2)*r0_B2:9:10 w yerr ls 2 t 'b1.95, \mu_s = 0.0198 (ext.)', \
 #'./kk_beta_matched.dat' i 1 u ($2**2)*r0_A2:9:10 w yerr ls 1 t 'b1.90, \mu_s = 0.02261 (int.)', \

set out 'kk_scat_A_match_to_B.pdf'

# Plot extrapolation
plot './kk_beta_matched.dat' i 0 u ($2**2)*r0_A2:9:($9-$10-$11):($9+$10+$12) ls 1 w errorbars notitle, \
 './kk_beta_matched.dat' i 2 u ($2**2)*r0_B2:9:($9-$10-$11):($9+$10+$12) ls 2 w errorbars notitle, \
 './kk_beta_matched.dat' i 0 u ($2**2)*r0_A2:9:10 w yerr ls 1 t 'A-Ensembles', \
 './kk_beta_matched.dat' i 2 u ($2**2)*r0_B2:9:10 w yerr ls 2 t 'B-Ensembles'


 #'./kk_beta_matched.dat' i 0 u ($2**2)*r0_A2:9:10 w yerr ls 1 t 'b1.90, \mu_s = 0.02125 (ext.)', \
 #'./kk_beta_matched.dat' i 2 u ($2**2)*r0_B2:9:10 w yerr ls 2 t 'b1.95, \mu_s = 0.0186'

## Plot overview
#plot './kk_beta_matched.dat' i 0 u ($5)*r0_A2:9:($9-$10-$12):($9+$10+$11) ls 1 w errorbars notitle, \
# './kk_beta_matched.dat' i 1 u ($5)*r0_A2:9:($9-$10-$12):($9+$10+$11) ls 4 w errorbars notitle, \
# './kk_beta_matched.dat' i 2 u ($5**2)*r0_B2:9:($9-$10-$12):($9+$10+$11) ls 2 w errorbars notitle, \
# './kk_beta_matched.dat' i 1 u ($5)*r0_A2:9:13 w labels point offset 0,3 notitle, \
# './kk_beta_matched.dat' i 2 u ($5**2)*r0_B2:9:13 w labels point offset 0,-3 notitle, \
# './kk_beta_matched.dat' i 0 u ($5)*r0_A2:9:10 w yerr ls 1 t 'b1.90, \mu_s = 0.02125 (int.)', \
# './kk_beta_matched.dat' i 1 u ($5)*r0_A2:9:10 w yerr ls 4 t 'b1.90, \mu_s = 0.02125 (ext.)', \
# './kk_beta_matched.dat' i 2 u ($5**2)*r0_B2:9:10 w yerr ls 2 t 'b1.95, \mu_s = 0.0186'
 # Spare D-Ensemble
 #'./kk_beta_matched.dat' i 3 u ($5**2)*r0_D2:9:($9-$10-$12):($9+$10+$11) ls 3 w errorbars notitle, \
 #'./kk_beta_matched.dat' i 3 u ($5**2)*r0_D2:9:13 w labels point offset 0,-3 notitle, \
 #'./kk_beta_matched.dat' i 3 u ($5**2)*r0_D2:9:10 w yerr ls 3 t 'b2.10, \mu_s = 0.0150'

## Plot interpolation
#plot './kk_beta_matched.dat' i 0 u ($5)*r0_A2:9:($9-$10-$12):($9+$10+$11) ls 1 w errorbars notitle, \
# './kk_beta_matched.dat' i 2 u ($5**2)*r0_B2:9:($9-$10-$12):($9+$10+$11) ls 2 w errorbars notitle, \
# './kk_beta_matched.dat' i 0 u ($5)*r0_A2:9:10 w yerr ls 1 t 'b1.90, \mu_s = 0.02125 (int.)', \
# './kk_beta_matched.dat' i 2 u ($5**2)*r0_B2:9:10 w yerr ls 2 t 'b1.95, \mu_s = 0.0186'
# # Spare D-Ensemble
# #'./kk_beta_matched.dat' i 3 u ($5**2)*r0_D2:9:($9-$10-$12):($9+$10+$11) ls 3 w errorbars notitle, \
## './kk_beta_matched.dat' i 3 u ($5**2)*r0_D2:9:10 w yerr ls 3 t 'b2.10, \mu_s = 0.0150'
#
## Plot extrapolation
#plot './kk_beta_matched.dat' i 1 u ($5)*r0_A2:9:($9-$10-$12):($9+$10+$11) ls 4 w errorbars notitle, \
# './kk_beta_matched.dat' i 2 u ($5**2)*r0_B2:9:($9-$10-$12):($9+$10+$11) ls 2 w errorbars notitle, \
# './kk_beta_matched.dat' i 1 u ($5)*r0_A2:9:10 w yerr ls 4 t 'b1.90, \mu_s = 0.02125 (ext.)', \
# './kk_beta_matched.dat' i 2 u ($5**2)*r0_B2:9:10 w yerr ls 2 t 'b1.95, \mu_s = 0.0186'
# # Spare D-Ensemble
# #'./kk_beta_matched.dat' i 3 u ($5**2)*r0_D2:9:($9-$10-$12):($9+$10+$11) ls 3 w errorbars notitle, \
# #'./kk_beta_matched.dat' i 3 u ($5**2)*r0_D2:9:10 w yerr ls 3 t 'b2.10, \mu_s = 0.0150'

