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

#ifndef SET_CLUSTER_H
#define SET_CLUSTER_H

#include "linmath.h"

int set_rndcluster(int rnd_fd, double *range, vec3 *values, long int num_values);

double basic_poly(double x);

int set_polysweep(double (*polyfunc)(double x), double *range, double *values, long int num_values);

#endif
