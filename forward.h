#ifndef FORWARD_H
#define FORWARD_H

#include "linmath.h"

double neural_forward(vec3 input, long int num_neurons, vec3 *weights);

typedef struct {

  double current_output;

  double mse;
  
} nf_ret;

nf_ret neural_forwarderr(vec3 input, double output, long int num_neurons, vec3 *weights);

nf_ret neural_forwarderr2(vec3 input, double output, long int num_neurons, vec3 *weights, double *hidden);

#endif


  
