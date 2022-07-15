
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <errno.h>

#include <rpc/xdr.h>

#include "linmath.h"

#include "sigmoid.h"

#include "relu.h"

#include "forward.h"

#include "forward2.h"

#include "neural_cfg.h"

#define def_filename "neural_net.xdr"

int main(int argc, char *argv[]) {

  long int num_neurons;

  double *weights;

  double bias_weight;
  
  double inputs[3];

  double *hidden;
  
  char *filename;

  double output;
  
  double current_output;

  long int num_weights;

  long int num_layers;

  num_layers = def_layers;
  
  num_neurons = def_neurons;
  
  {
	
    filename = argc>1 ? argv[1] : def_filename;
	
    inputs[0] = argc>2 ? strtod(argv[2],NULL) : 0.5;
    inputs[1] = argc>3 ? strtod(argv[3],NULL) : 0.5;
    inputs[2] = argc>4 ? strtod(argv[4],NULL) : 0.5;

    printf("Processing input value %g %g %g\n", inputs[0], inputs[1], inputs[2]);
  }

  {

    XDR xdrs;
    
    FILE *fp;

    fp = fopen(filename, "r");
    if (fp == NULL) {
      perror("fopen");
      return -1;
    }
    
    {

      xdrstdio_create(&xdrs, fp, XDR_DECODE);

      xdr_long(&xdrs, &num_weights);

      weights = malloc(sizeof(double) * num_weights);
      if (weights == NULL) {
	perror("malloc");
	return -1;
      }

      xdr_vector(&xdrs, weights, num_weights, sizeof(double), xdr_double);

      xdr_vector(&xdrs, &bias_weight, 1, sizeof(double), xdr_double);
      
      xdr_destroy(&xdrs);

      /*
      retval = fclose(fp);
      if (retval == -1) {
	perror("fclose");
	return -1;
      }
      */
      
    }

  }

  hidden = malloc(num_neurons * sizeof(double));
  if (hidden == NULL) {
    perror("malloc");
    return -1;
  }
  
  switch(num_layers) {

  case 1:

    current_output = neural_forward(inputs, num_neurons, weights, bias_weight);
    
    break;

  case 2:
    
    current_output = neural_forward2(inputs, num_neurons, weights, bias_weight, hidden);

    break;

  }
  
  free(weights);
  free(hidden);

  printf("Current output (final) %.04g\n", current_output);
  
  return 0;

}

  
