
set term png
set output "mse.png"

set title "Mean Squared Error (MSE)"
set xlabel "Iteration #"
set ylabel "MSE"
set grid
plot "mse.dat" title "" with lines
