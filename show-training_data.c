
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <rpc/xdr.h>

#include <errno.h>

int main(int argc, char *argv[]) {

  long int num_rows;

  double *inputs;

  double *outputs;

  long int input_cols;
  long int output_cols;

  char *filename;

  filename = argc>1 ? argv[1] : "training_data.xdr";
  
  {

    XDR xdrs;

    long int rowno;

    FILE *fp;

    fp = fopen(filename, "r");
    if (fp == NULL) {
      perror("fopen");
      return -1;
    }
    
    xdrstdio_create(&xdrs, fp, XDR_DECODE);

    xdr_long(&xdrs, &num_rows);
    xdr_long(&xdrs, &input_cols);
    xdr_long(&xdrs, &output_cols);

    inputs = malloc(sizeof(double) * input_cols * num_rows);
    outputs = malloc(sizeof(double) * output_cols * num_rows);
    
    xdr_vector(&xdrs, inputs, input_cols * num_rows, sizeof(double), xdr_double);

    xdr_vector(&xdrs, outputs, output_cols * num_rows, sizeof(double), xdr_double);

    xdr_destroy(&xdrs);
    
    printf("Input (first 10 rows)\n");
    
    for (rowno = 0; rowno < 10; rowno++) {
      printf("%g %g %g\n", inputs[0*num_rows+3*rowno+0], inputs[0*num_rows+3*rowno+1], inputs[0*num_rows+3*rowno+2]);
    }

    printf("Input (last 10 rows)\n");

    for (rowno = num_rows - 10; rowno < num_rows; rowno++) {
      printf("%g %g %g\n", inputs[0*num_rows+3*rowno+0], inputs[0*num_rows+3*rowno+1], inputs[0*num_rows+3*rowno+2]);
    }

    printf("Output (first 10 rows)\n");
    
    for (rowno = 0; rowno < 10; rowno++) {
      printf("%g\n", outputs[0*num_rows+rowno+0]);
    }

    printf("Output (last 10 rows)\n");

    for (rowno = num_rows - 10; rowno < num_rows; rowno++) {
      printf("%g\n", outputs[0*num_rows+rowno+0]);
    }

    
    
  }
  
  return 0;

}
  
  
  
  
