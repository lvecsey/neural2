#ifndef FORWARD_H
#define FORWARD_H

#include "linmath.h"

double neural_forward(double *inputs, long int num_neurons, double *weights, double bias_weight);

#include "nf_ret.h"

#include "squashed_nodes.h"

nf_ret neural_forwarderr(double *inputs, double output, long int num_neurons, double *weights, double bias_weight, squashed_nodes *sq_cache);

#endif


  
