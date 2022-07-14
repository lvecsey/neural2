
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

#include "linmath.h"

#include "sigmoid.h"

#include "forward.h"

double neural_forward(vec3 input, long int num_neurons, vec3 *weights) {

  double current_output;
  
  long int ono;

  double moided;

  long int nodesum;

  double output;
  
  moided = 0.0;

  output = 0.0;
  
  for (ono = 0; ono < num_neurons; ono++) {

    nodesum = weights[ono][0] * input[0] + weights[ono][1] * input[1] + weights[ono][2] * input[2];
      
    output += nodesum;
      
  }
    
  current_output = sigmoid(output);

  return current_output;

}

nf_ret neural_forwarderr(vec3 input, double output, long int num_neurons, vec3 *weights) {

  nf_ret nf;
  
  long int ono;

  double err;

  long int nodesum;

  double sum;
  
  err = 0.0;

  nf.mse = 0.0;

  sum = 0.0;
  
  for (ono = 0; ono < num_neurons; ono++) {

    nodesum = weights[ono][0] * input[0] + weights[ono][1] * input[1] + weights[ono][2] * input[2];

    sum += nodesum;
    
    err = (output - nodesum);

    nf.mse += (err * err);
    
  }
    
  nf.current_output = sigmoid(sum);

  nf.mse /= num_neurons;
  
  return nf;

}

nf_ret neural_forwarderr2(vec3 input, double output, long int num_neurons, vec3 *weights, double *hidden) {

  nf_ret nf;
  
  double sum;

  long int ono;

  double nodesum;

  double err;
  
  sum = 0.0;

  for (ono = 0; ono < num_neurons; ono++) {

    nodesum = weights[ono][0] * input[0] + weights[ono][1] * input[1] + weights[ono][2] * input[2];

    sum += nodesum;
    
    hidden[ono] = sigmoid(nodesum);
      
  }
    
  {

    double sum;

    long int wno;
    
    sum = 0.0;

    nf.mse = 0.0;

    for (wno = 0; wno < num_neurons; wno++) {

      nodesum = weights[wno + num_neurons][0] * hidden[wno] + weights[wno + num_neurons][1] * hidden[wno] + weights[wno + num_neurons][2] * hidden[wno];

      sum += nodesum;
      
      err = (output - nodesum);
      
      nf.mse += (err * err);
      
    }

    nf.current_output = sigmoid(sum);

    nf.mse /= num_neurons;
    
  }

  return nf;
  
}
