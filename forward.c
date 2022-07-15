
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

#include "linmath.h"

#include "sigmoid.h"

#include "forward.h"

double neural_forward(double *inputs, long int num_neurons, double *weights, double bias_weight) {

  double current_output;
  
  long int ono;

  double nodesum;

  double sum;
  
  sum = 0.0;
  
  for (ono = 0; ono < num_neurons; ono++) {
    
    nodesum = (weights[3*ono+0] * inputs[0]);
    nodesum += (weights[3*ono+1] * inputs[1]);
    nodesum += (weights[3*ono+2] * inputs[2]);

    sum += nodesum;
      
  }

  {

    nodesum = (bias_weight * 1.0);

    sum += nodesum;
    
  }

  current_output = sigmoid(sum);
    
  return current_output;

}

nf_ret neural_forwarderr(double *inputs, double output, long int num_neurons, double *weights, double bias_weight, squashed_nodes *sq_cache) {

  nf_ret nf;
  
  long int ono;

  double err;

  double nodesum;

  double moided;
  
  double sum;
  
  err = 0.0;

  sum = 0.0;

  nf.mse = 0.0;
  
  for (ono = 0; ono < num_neurons; ono++) {

    nodesum = (weights[3*ono+0] * inputs[0]);
    nodesum += (weights[3*ono+1] * inputs[1]);
    nodesum += (weights[3*ono+2] * inputs[2]);

    moided = sigmoid(nodesum);
    
    sq_cache->hidden_nodesums[ono] = nodesum;

    sq_cache->hidden_squashed[ono] = moided;

    sum += nodesum;

    err = (output - moided);
    
    nf.mse += (err * err);
    
  }

  {

    nodesum = (bias_weight * 1.0);

    moided = sigmoid(nodesum);
    
    sum += nodesum;

    err = (output - moided);
    
    nf.mse += (err * err);
    
  }

  nf.mse /= (num_neurons + 1);
  
  nf.current_output = sigmoid(sum);
  
  return nf;

}

