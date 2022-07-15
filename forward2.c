
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

#include "linmath.h"

#include "sigmoid.h"

#include "forward2.h"

double neural_forward2(double *inputs, long int num_neurons, double *weights, double bias_weight, double *hidden) {

  double current_output;
  
  long int ono;

  long int nodesum;

  double sum;

  double moided;
  
  for (ono = 0; ono < num_neurons; ono++) {

    moided = sigmoid(weights[ono] * inputs[0]);
    moided += sigmoid(weights[ono] * inputs[1]);
    moided += sigmoid(weights[ono] * inputs[2]);    

    hidden[ono] = moided;
    
  }

  weights += num_neurons;

  sum = 0.0;

  moided = 0.0;
  
  for (ono = 0; ono < num_neurons; ono++) {

    moided += sigmoid(weights[ono] * hidden[ono]);

  }

  {

    moided += sigmoid(bias_weight * 1.0);

  }

  current_output = moided;

  return current_output;

}

nf_ret neural_forwarderr2(double *inputs, double output, long int num_neurons, double *weights, double bias_weight, squashed_nodes *sq_cache) {

  nf_ret nf;
  
  double sum;

  long int ono;

  double nodesum;

  double err;

  double moided;
  
  for (ono = 0; ono < num_neurons; ono++) {

    moided = sigmoid(weights[ono] * inputs[0]);
    moided += sigmoid(weights[ono] * inputs[1]);
    moided += sigmoid(weights[ono] * inputs[2]);    

    sq_cache->hidden_squashed[ono] = moided;

  }

  weights += num_neurons;

  nf.mse = 0.0;

  sum = 0.0;

  moided = 0.0;
  
  for (ono = 0; ono < num_neurons; ono++) {

    moided += sigmoid(weights[ono] * sq_cache->hidden_squashed[ono]);

    // sq_cache->output_nodesums[ono] = nodesum;

  }

  {

    moided += sigmoid(bias_weight * 1.0);

  }

  err = (output - moided);
    
  nf.mse += (err * err);
  
  nf.current_output = moided;
  
  nf.mse /= (num_neurons + 1);

  return nf;
  
}
