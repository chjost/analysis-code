set title "L-dependence of relative error"
set xlabel "L/a"
set ylabel "Delta dE/dE"
set xrange [10:40]
set yrange [0.1:1.2]
set grid
set style line 2 lc rgb "blue" pt 22
set style line 3 lc rgb "dark-green" pt 20
set style line 4 lc rgb "red" pt 17
set terminal pdfcairo enhanced solid color size 7,5 lw 2 font ",16"
set out "./rel_err_E_vs_L.pdf"
early=0.00107922
late=0.0017508
plot './delta_E_vs_L.dat' u 1:($2/early) ls 2 title "kk-data, A30.32, A40.24, t = 10.5", \
     './delta_E_vs_L.dat' u 1:($3/late) ls 4 title "kk-data, A30.32, A40.24, t = 15.5"
     #'./delta_E_vs_L.dat' u 1:($3/norm2) title "pi-pi-data, A30.32, A40.24, t = "
