
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <rpc/xdr.h>

#include <errno.h>

#include "set_cluster.h"

int set_rndclusterd(int rnd_fd, double *range, double *values, long int num_values) {

  long int valno;

  double span;

  ssize_t bytes_read;

  uint64_t rnd;
  
  span = (range[1] - range[0]);
  
  for (valno = 0; valno < num_values; valno++) {

    bytes_read = read(rnd_fd, &rnd, sizeof(uint64_t));
    if (bytes_read != sizeof(uint64_t)) {
      perror("read");
      return -1;
    }
    
    values[valno] = range[0] + (span * rnd) / 18446744073709551615.0;
    
  }
  
  return 0;
  
}

double basic_poly2(double x) {

  return -7.532 * x * x + 3.75 * x;

}

int main(int argc, char *argv[]) {

  long int num_rows;

  double *inputs;

  double *outputs;

  long int input_cols;
  long int output_cols;
  
  num_rows = 1200;

  input_cols = 3;
  output_cols = 1;

  inputs = malloc(sizeof(double) * input_cols * num_rows);
  if (inputs == NULL) {
    perror("malloc");
    return -1;
  }
  
  outputs = malloc(sizeof(double) * output_cols * num_rows);
  if (outputs == NULL) {
    perror("malloc");
    return -1;
  }

  {

    long int rowno;

    int rnd_fd;

    double range[2];

    long int num_values;

    long int offset;
    
    rnd_fd = open("/dev/urandom", O_RDONLY);
    if (rnd_fd == -1) {
      perror("open");
      return -1;
    }
    
    num_values = (num_rows / 2);

    fprintf(stderr, "Range A values (input)\n");
    
    {

      range[0] = 0.2;
      range[1] = 0.4;
      offset = 0 * num_rows + 0;
      set_rndclusterd(rnd_fd, range, inputs + offset, num_values);
      
      range[0] = 0.7;
      range[1] = 0.9;
      offset = 1 * num_rows + 0;
      set_rndclusterd(rnd_fd, range, inputs + offset, num_values);
      
      range[0] = -1.0;
      range[1] = 1.0;
      offset = 2 * num_rows + 0;
      set_rndclusterd(rnd_fd, range, inputs + offset, num_values);
      
    }

    fprintf(stderr, "Range B values (input)\n");
    
    {

      range[0] = -1.0;
      range[1] = 1.0;
      offset = 0 * num_rows + num_values;
      set_rndclusterd(rnd_fd, range, inputs + offset, num_values);
      
      range[0] = -0.4;
      range[1] = -0.2;
      offset = 1 * num_rows + num_values;
      set_rndclusterd(rnd_fd, range, inputs + offset, num_values);
      
      range[0] = 0.6;
      range[1] = 0.8;
      offset = 2 * num_rows + num_values;
      set_rndclusterd(rnd_fd, range, inputs + offset, num_values);
      
    }

    fprintf(stderr, "Corresponding output values.\n");
    
    {

      range[0] = 0.95;
      range[1] = 1.0;
      offset = 0;
      set_polysweep(basic_poly, range, outputs + offset, num_values);
      
      range[0] = 0.0;
      range[1] = 0.05;
      offset = num_values;
      set_polysweep(basic_poly2, range, outputs + offset, num_values);

    }
    
  }
  
  {

    XDR xdrs;

    xdrstdio_create(&xdrs, stdout, XDR_ENCODE);

    xdr_long(&xdrs, &num_rows);
    xdr_long(&xdrs, &input_cols);
    xdr_long(&xdrs, &output_cols);
    
    xdr_vector(&xdrs, inputs, input_cols * num_rows, sizeof(double), xdr_double);

    xdr_vector(&xdrs, outputs, output_cols * num_rows, sizeof(double), xdr_double);

    xdr_destroy(&xdrs);
    
  }

  free(inputs);
  free(outputs);
  
  return 0;

}
  
  
  
  
