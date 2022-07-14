/*
    Neural network perceptron (single layer network) with data
    Copyright (C) 2022  Lester Vecsey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

#include <stdint.h>

#include "linmath.h"

#include "set_cluster.h"

#include "sigmoid.h"

int set_rndcluster(int rnd_fd, double *range, vec3 *values, long int num_values) {

  long int valno;

  double span;

  ssize_t bytes_read;

  uint64_t rnds[3];
  
  span = range[1] - range[0];
  
  for (valno = 0; valno < num_values; valno++) {

    bytes_read = read(rnd_fd, rnds, sizeof(uint64_t) * 3);
    if (bytes_read != sizeof(uint64_t) * 3) {
      perror("read");
      return -1;
    }
    
    values[valno][0] = range[0] + (span * rnds[0]) / 18446744073709551615.0;
    values[valno][1] = range[0] + (span * rnds[1]) / 18446744073709551615.0;
    values[valno][2] = range[0] + (span * rnds[2]) / 18446744073709551615.0;
    
  }
  
  return 0;
  
}

double basic_poly(double x) {

  return 3.2 * x * x - 7.4 * x;

}

int set_polysweep(double (*polyfunc)(double x), double *range, double *values, long int num_values) {

  long int valno;

  double span;

  double x;

  double v;
  
  span = range[1] - range[0];
  
  for (valno = 0; valno < num_values; valno++) {

    x = range[0] + (span * valno) / (num_values - 1);

    v = sigmoid(polyfunc(x));
    
    values[valno] = range[0] + (v * span);
    
  }
  
  return 0;
  
}
