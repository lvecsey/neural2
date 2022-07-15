#ifndef FORWARD2_H
#define FORWARD2_H

#include "linmath.h"

double neural_forward2(double *inputs, long int num_neurons, double *weights, double bias_weight, double *hidden);

#include "nf_ret.h"

#include "squashed_nodes.h"

nf_ret neural_forwarderr2(double *inputs, double output, long int num_neurons, double *weights, double bias_weight, squashed_nodes *sq_cache);

#endif
  
