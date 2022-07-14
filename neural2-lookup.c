
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <errno.h>

#include <rpc/xdr.h>

#include "linmath.h"

#include "dotproduct.h"

#include "sigmoid.h"

#include "relu.h"

#include "forward.h"

#include "neural_cfg.h"

#define def_filename "neural_net.xdr"

int main(int argc, char *argv[]) {

  XDR xdrs;

  long int num_neurons;

  vec3 *weights;
  
  vec3 input;

  double *hidden;
  
  char *filename;

  double output;
  
  double outputs_cache;

  double current_output;

  int retval;

  long int num_weights;
  
  num_neurons = def_neurons;
  
  {
	
    filename = argc>1 ? argv[1] : def_filename;
	
    input[0] = argc>2 ? strtod(argv[2],NULL) : 0.5;
    input[1] = argc>3 ? strtod(argv[3],NULL) : 0.5;
    input[2] = argc>4 ? strtod(argv[4],NULL) : 0.5;

    printf("Processing input value %g %g %g\n", input[0], input[1], input[2]);
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

      weights = malloc(sizeof(vec3) * num_weights);
      if (weights == NULL) {
	perror("malloc");
	return -1;
      }

      xdr_vector(&xdrs, weights, 3 * num_weights, sizeof(float), xdr_float);

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

  /*
  hidden = malloc(num_neurons * sizeof(double));
  if (hidden == NULL) {
    perror("malloc");
    return -1;
  }
  */
  
  current_output = neural_forward(input, num_neurons, weights);

  // free(hidden);

  printf("Current output (final) %.04g\n", current_output);
  
  return 0;

}

  
