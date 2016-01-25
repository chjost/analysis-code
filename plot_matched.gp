set title "Kaon Kaon Scattering length"
set xlabel "(r_0 M_K)^2"
set ylabel "M_Ka_0"
set xrange [1.5:2.4]
set yrange [-0.42:-0.26]
set grid
r0_A2=5.31**2
r0_B2=5.77**2
r0_D2=7.60**2
set style line 1 lc rgb "red" pt 18
set style line 2 lc rgb "blue" pt 22
set style line 3 lc rgb "dark-green" pt 20
#plot statistical errors first (they are larger)
#plot './kk_beta_matched.dat' i 1 u ($5**2)*r0_A2:9:($9-$10-$12):($9+$10+$11) ls 1 w errorbars notitle
#replot './kk_beta_matched.dat' i 2 u ($5**2)*r0_B2:9:($9-$10-$12):($9+$10+$11) ls 2 w errorbars notitle
#replot './kk_beta_matched.dat' i 3 u ($5**2)*r0_D2:9:($9-$10-$12):($9+$10+$11) ls 3 w errorbars notitle
#replot './kk_beta_matched.dat' i 1 u ($5**2)*r0_A2:9:10 w yerr ls 1 t 'b1.90, \mu_s = 0.02125 (ext.)'
#replot './kk_beta_matched.dat' i 2 u ($5**2)*r0_B2:9:10 w yerr ls 2 t 'b1.95, \mu_s = 0.0186'
#replot './kk_beta_matched.dat' i 3 u ($5**2)*r0_D2:9:10 w yerr ls 3 t 'b2.10, \mu_s = 0.0150'
#
#plot './kk_beta_matched.dat' i 0 u ($5**2)*r0_A2:9:($9-$10-$12):($9+$10+$11) ls 1 w errorbars notitle
#replot './kk_beta_matched.dat' i 2 u ($5**2)*r0_B2:9:($9-$10-$12):($9+$10+$11) ls 2 w errorbars notitle
#replot './kk_beta_matched.dat' i 3 u ($5**2)*r0_D2:9:($9-$10-$12):($9+$10+$11) ls 3 w errorbars notitle
#replot './kk_beta_matched.dat' i 0 u ($5**2)*r0_A2:9:10 w yerr ls 1 t 'b1.90, \mu_s = 0.02125 (int.)'
#replot './kk_beta_matched.dat' i 2 u ($5**2)*r0_B2:9:10 w yerr ls 2 t 'b1.95, \mu_s = 0.0186'
#replot './kk_beta_matched.dat' i 3 u ($5**2)*r0_D2:9:10 w yerr ls 3 t 'b2.10, \mu_s = 0.0150'
set terminal pdfcairo enhanced solid color size 7,5 lw 2 font ",16"
set out './kk_scat_matched.pdf'

plot './kk_beta_matched.dat' i 1 u ($5**2)*r0_A2:9:($9-$10-$12):($9+$10+$11) ls 1 w errorbars notitle, \
 './kk_beta_matched.dat' i 2 u ($5**2)*r0_B2:9:($9-$10-$12):($9+$10+$11) ls 2 w errorbars notitle, \
 './kk_beta_matched.dat' i 3 u ($5**2)*r0_D2:9:($9-$10-$12):($9+$10+$11) ls 3 w errorbars notitle, \
 './kk_beta_matched.dat' i 1 u ($5**2)*r0_A2:9:10 w yerr ls 1 t 'b1.90, \mu_s = 0.02125 (ext.)', \
 './kk_beta_matched.dat' i 2 u ($5**2)*r0_B2:9:10 w yerr ls 2 t 'b1.95, \mu_s = 0.0186', \
 './kk_beta_matched.dat' i 3 u ($5**2)*r0_D2:9:10 w yerr ls 3 t 'b2.10, \mu_s = 0.0150'

plot './kk_beta_matched.dat' i 0 u ($5**2)*r0_A2:9:($9-$10-$12):($9+$10+$11) ls 1 w errorbars notitle, \
'./kk_beta_matched.dat' i 2 u ($5**2)*r0_B2:9:($9-$10-$12):($9+$10+$11) ls 2 w errorbars notitle, \
'./kk_beta_matched.dat' i 3 u ($5**2)*r0_D2:9:($9-$10-$12):($9+$10+$11) ls 3 w errorbars notitle, \
'./kk_beta_matched.dat' i 0 u ($5**2)*r0_A2:9:10 w yerr ls 1 t 'b1.90, \mu_s = 0.02125 (int.)', \
'./kk_beta_matched.dat' i 2 u ($5**2)*r0_B2:9:10 w yerr ls 2 t 'b1.95, \mu_s = 0.0186', \
'./kk_beta_matched.dat' i 3 u ($5**2)*r0_D2:9:10 w yerr ls 3 t 'b2.10, \mu_s = 0.0150'
