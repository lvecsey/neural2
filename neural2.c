/*
    Neural network perceptron (single layer network) with data
    Copyright (C) 2022  Lester Vecsey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
				
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <stdint.h>
#include <errno.h>

#include <rpc/xdr.h>

#include "linmath.h"

#include "dotproduct.h"

#include "relu.h"

#include "relu_deriv.h"

#include "norm.h"

#include "set_cluster.h"

#include "sigmoid.h"

#include "sigmoid_deriv.h"

#include "forward.h"

#include "neural_cfg.h"

int set_norm(vec3 *values, long int num_values) {

  long int valno;

  for (valno = 0; valno < num_values; valno++) {
    norm(values[valno]);
  }

  return 0;

}

int show_vector(vec3 *values, long int num_values) {

  long int valno;

  for (valno = 0; valno <  num_values; valno++) {
    printf("%g %g %g\n", values[valno][0], values[valno][1], values[valno][2]);
  }

  return 0;

}

int set_range(vec3 *values, long int num_values) {

  long int valno;

  for (valno = 0; valno < num_values; valno++) {

    values[valno][0] = -1.0 + (2.0 * valno) / (num_values - 1);
    values[valno][1] = -1.0 + (2.0 * valno) / (num_values - 1);
    values[valno][2] = -1.0 + (2.0 * valno) / (num_values - 1);
    
  }
  
  return 0;
  
}

int set_random(int rnd_fd, vec3 *values, long int num_values) {

  long int valno;

  ssize_t bytes_read;

  uint64_t rnds[3];
  
  for (valno = 0; valno < num_values; valno++) {

    bytes_read = read(rnd_fd, rnds, sizeof(uint64_t) * 3);
    if (bytes_read != sizeof(uint64_t) * 3) {
      perror("read");
      return -1;
    }

    values[valno][0] = -1.0 + (2.0 * rnds[0]) / 18446744073709551615.0;
    values[valno][1] = -1.0 + (2.0 * rnds[1]) / 18446744073709551615.0;
    values[valno][2] = -1.0 + (2.0 * rnds[2]) / 18446744073709551615.0;
    
  }

  return 0;
  
}

bool_t xdr_vec3 (XDR *xdrs, void *extra) {

  vec3 *nup;

  nup = (vec3*) extra;
  
  if (!xdr_float(xdrs, nup[0])) {
    return FALSE;
  }

  if (!xdr_float(xdrs, nup[1])) {
    return FALSE;
  }

  if (!xdr_float(xdrs, nup[2])) {
    return FALSE;
  }

  return (TRUE);
}

int main(int argc, char *argv[]) {

  vec3 *weights;

  long int num_neurons;

  double *training_inputs;
  double *training_outputs;

  long int num_rows;
  long int input_cols;
  long int output_cols;
  
  vec3 *dataset_inputs;
  
  vec3 *dataset_outputs;
  
  vec3 *inputs;

  vec3 *outputs;

  double bias_weight;
  
  double *hidden;

  double *outputs_cache;

  long int wno;
  
  int rnd_fd;

  long int iterno;

  long int num_iters;

  long int num_inputs;

  long int num_outputs;
  
  long int ono;
  
  double percent_completed;

  long int num_weights;

  double sum;

  vec3 input;

  vec3 output;

  uint64_t rnds[3];
	  
  ssize_t bytes_read;

  int retval;

  char *env_VERBOSE;

  double learning_rate;

  long int progress_meter;

  char *env_PROGRESS;

  FILE *log_msefn;

  char *nn_xdrfn;

  long int num_dataset;
  
  long int num_accumulated;

  nf_ret nf;
  
  env_VERBOSE = getenv("VERBOSE");

  env_PROGRESS = getenv("PROGRESS");

  if (env_PROGRESS == NULL) {
    progress_meter = 1;
  }
  else {
    progress_meter = strtol(env_PROGRESS, NULL, 10);
  }
  
  learning_rate = 0.00125;
  
  num_neurons = argc>1 ? strtol(argv[1],NULL,10) : def_neurons;
  
  num_iters = argc>2 ? strtol(argv[2],NULL,10) : def_iters;

  input[0] = argc>3 ? strtod(argv[3],NULL) : 0.37;
  input[1] = argc>4 ? strtod(argv[4],NULL) : 0.75;
  input[2] = argc>5 ? strtod(argv[5],NULL) : 0.21;

  rnd_fd = open("/dev/urandom", O_RDONLY);
  if (rnd_fd == -1) {
    perror("open");
    return -1;
  }

  num_inputs = 3;

  num_outputs = 1;
  
  num_weights = num_neurons;
  
  {

    XDR xdrs;

    FILE *fp;

    char *filename;

    filename = "training_data.xdr";
    
    fp = fopen(filename, "r");
    if (fp == NULL) {
      perror("fopen");
      return -1;
    }
    
    xdrstdio_create(&xdrs, fp, XDR_DECODE);

    xdr_long(&xdrs, &num_rows);
    xdr_long(&xdrs, &input_cols);
    xdr_long(&xdrs, &output_cols);

    training_inputs = malloc(sizeof(double) * input_cols * num_rows);
    if (training_inputs == NULL) {
      perror("malloc");
      return -1;
    }
    
    training_outputs = malloc(sizeof(double) * output_cols * num_rows);
    if (training_outputs == NULL) {
      perror("malloc");
      return -1;
    }
    
    xdr_vector(&xdrs, (char*) training_inputs, input_cols * num_rows, sizeof(double), xdr_double);

    xdr_vector(&xdrs, (char*) training_outputs, output_cols * num_rows, sizeof(double), xdr_double);

    xdr_destroy(&xdrs);

    /*
    retval = fclose(fp);
    if (retval == -1) {
      perror("fclose");
      return -1;
    }
    */
    
  }

  {

    num_dataset = (4 * num_rows) / 5;
	
    dataset_inputs = malloc(sizeof(vec3) * num_dataset);
    if (dataset_inputs == NULL) {
      perror("malloc");
      return -1;
    }

    dataset_outputs = malloc(sizeof(vec3) * num_dataset);
    if (dataset_outputs == NULL) {
      perror("malloc");
    return -1;
    }  

  }

  {

    long int rowno;

    long int acumno;

    double training_portion;

    for (rowno = 0; rowno < num_dataset; rowno++) {
      dataset_inputs[rowno][0] = training_inputs[0*num_rows+rowno];
      dataset_inputs[rowno][1] = training_inputs[1*num_rows+rowno];
      dataset_inputs[rowno][2] = training_inputs[2*num_rows+rowno];
      dataset_outputs[rowno][0] = training_outputs[rowno];
      dataset_outputs[rowno][1] = 0.0;
      dataset_outputs[rowno][2] = 0.0;
    }

    free(training_inputs);
    free(training_outputs);

    inputs = malloc(sizeof(vec3) * num_dataset);
    if (inputs == NULL) {
      perror("malloc");
      return -1;
    }

    outputs = malloc(sizeof(vec3) * num_dataset);
    if (outputs == NULL) {
      perror("malloc");
      return -1;
    }
    
    training_portion = 0.80;
    
    acumno = 0;
    
    for (rowno = 0; rowno < num_dataset; rowno++) {

      double training_sample;
      
      bytes_read = read(rnd_fd, rnds, sizeof(uint64_t));
      if (bytes_read != sizeof(uint64_t)) {
	perror("read");
	return -1;
      }

      training_sample = (rnds[0] / 18446744073709551615.0);

      if (training_sample < training_portion) {
      
	inputs[acumno][0] = dataset_inputs[rowno][0];
	inputs[acumno][1] = dataset_inputs[rowno][1];
	inputs[acumno][2] = dataset_inputs[rowno][2];

	outputs[acumno][0] = dataset_outputs[rowno][0];
	outputs[acumno][1] = dataset_outputs[rowno][1];
	outputs[acumno][2] = dataset_outputs[rowno][2];

	acumno++;
	
      }

    }

    num_accumulated = acumno;
    
  }

  printf("Processed %ld rows of training data.\n", num_accumulated);

  printf("Dataset inputs: \n");
  show_vector(dataset_inputs, 10);

  printf("Dataset outputs: \n");
  show_vector(dataset_outputs, 10);

  printf("Allocating weights and rest of neural network.\n");

  {

    printf("Allocating %ld weights.\n", num_weights);
    weights = malloc(sizeof(vec3) * num_weights);
    if (weights == NULL) {
      perror("malloc");
      return -1;
    }

    /*
    printf("Allocating %ld hidden nodes.\n", num_neurons);    
    hidden = malloc(sizeof(double) * num_neurons);
    if (hidden == NULL) {
      perror("malloc");
      return -1;
    }
    */
    
    printf("Allocating %ld output nodes.\n", num_neurons);    
    outputs_cache = malloc(sizeof(double) * num_neurons);
    if (outputs_cache == NULL) {
      perror("malloc");
      return -1;
    }

  }
  
  {

    XDR xdrs;
    
    FILE *fp;

    nn_xdrfn = "neural_net.xdr";
    
    fp = fopen(nn_xdrfn, "r");
    if (fp != NULL) {

      bool_t bret;
      
      printf("Reading cached weights from %s.\n", nn_xdrfn);
      
      xdrstdio_create(&xdrs, fp, XDR_DECODE);

      bret = xdr_long(&xdrs, &num_weights);
      if (!bret) {
	printf("Trouble retrieving num_neurons from %s.\n", nn_xdrfn);
	return -1;
      }

      if (num_weights < 0) {
	fprintf(stderr, "weights file has an invalid number of weights.\n");
	return -1;
      }

      bret = xdr_vector(&xdrs, weights, 3 * num_weights, sizeof(float), xdr_float);
      if (!bret) {
	printf("Trouble retrieving weights from %s.\n", nn_xdrfn);
	return -1;
      }
      
      xdr_destroy(&xdrs);

      /*
      retval = fclose(fp);
      if (retval == -1) {
	perror("fclose");
	return -1;
      }
      */
      
    }

    else {

      printf("Cached neural network (trained) file not found.\n");

      set_random(rnd_fd, weights, num_weights);

      for (wno = 0; wno < num_weights; wno++) {

	weights[wno][0] *= 0.95;
	weights[wno][1] *= 0.95;
	weights[wno][2] *= 0.95;

	weights[wno][0] += 1.05;
	weights[wno][1] += 1.05;
	weights[wno][2] += 1.05;	
	
      }

      bias_weight = (2.0 - 0.1) * 0.5;
      
      for (ono = 0; ono < num_neurons; ono++) {
	outputs_cache[ono] = 0.0;
      }

    }

    {

      long int rowno;
      
      for (rowno = 0; rowno < num_dataset; rowno++) {
	inputs[rowno][0] = dataset_inputs[rowno][0];
	inputs[rowno][1] = dataset_inputs[rowno][1];
	inputs[rowno][2] = dataset_inputs[rowno][2];
	outputs[rowno][0] = dataset_outputs[rowno][0];
	outputs[rowno][1] = dataset_outputs[rowno][0];
	outputs[rowno][2] = dataset_outputs[rowno][0];
      }

    }

  }

  {

    double actual_y;
    
    long int rowno;

    printf("Num iterations %ld\n", num_iters);

    log_msefn = fopen("mse.dat", "w+");
    if (log_msefn == NULL) {
      perror("fopen");
      return -1;
    }

    rowno = 0;
    
    for (iterno = 0; iterno < num_iters; iterno++) {

      percent_completed = iterno; percent_completed /= (num_iters - 1);

      {

	double nodesum;
	
	sum = 0.0;

	input[0] = inputs[rowno][0];
	input[1] = inputs[rowno][1];
	input[2] = inputs[rowno][2];

	output[0] = outputs[rowno][0];
	output[1] = outputs[rowno][1];
	output[2] = outputs[rowno][2];

	rowno++;

	rowno %= num_accumulated;

	nf = neural_forwarderr(input, output[0], num_neurons, weights);
	
	if (log_msefn != NULL) {
	  fprintf(log_msefn, "%g\n", nf.mse);
	}

	{

	  double z;
	  vec3 delta;
	  double delta_target;
	  
	  actual_y = output[0];

	  delta_target = (actual_y - nf.current_output);

	  /*
	  for (wno = 0; wno < num_neurons; wno++) {

	    double z;
	    
	    delta[0] = weights[num_neurons + wno][0] * delta_target * sigmoid_deriv(hidden[wno]);
	    delta[1] = weights[num_neurons + wno][1] * delta_target * sigmoid_deriv(hidden[wno]);
	    delta[2] = weights[num_neurons + wno][2] * delta_target * sigmoid_deriv(hidden[wno]);
	
	    z = hidden[wno];
	
	    weights[num_neurons + wno][0] -= -learning_rate * (delta[0] * z);
	    weights[num_neurons + wno][1] -= -learning_rate * (delta[1] * z);
	    weights[num_neurons + wno][2] -= -learning_rate * (delta[2] * z);
	
	  }	  
	  */
	  
	  for (wno = 0; wno < num_neurons; wno++) {
	    
	    vec3 z;
	    
	    delta[0] = weights[wno][0] * delta_target * sigmoid_deriv(input[0]);
	    delta[1] = weights[wno][1] * delta_target * sigmoid_deriv(input[1]);
	    delta[2] = weights[wno][2] * delta_target * sigmoid_deriv(input[2]);
	
	    z[0] = input[0];
	    z[1] = input[1];
	    z[2] = input[2];
	
	    weights[wno][0] -= -learning_rate * (delta[0] * z[0]);
	    weights[wno][1] -= -learning_rate * (delta[1] * z[1]);
	    weights[wno][2] -= -learning_rate * (delta[2] * z[2]);
	
	  }

	  /*
	  {
	    double z;
	    bias_weight -= -learning_rate * (delta[0] * z);
	  }
	  */
	  
	}
	  
	if (progress_meter) {

	  wno = (iterno % num_weights);
	
	  printf("[%ld] Percent completed %.04g%% (Sample weights %.03f %.03f %.03f and cached output %.06g) Current output %g mse %g    \r", iterno, 100.0 * percent_completed, weights[wno][0], weights[wno][1], weights[wno][2], outputs_cache[wno], nf.current_output, nf.mse);
	
	}

      }

    }
      
    retval = fclose(log_msefn);
    if (retval == -1) {
      perror("fclose");
      return -1;
    }

    if (progress_meter > 0) {
      putchar('\n');
    }

    printf("Final mse %g\n", nf.mse);   

    {

      XDR xdrs;
    
      FILE *fp;

      printf("Writing %ld weights to %s\n", num_weights, nn_xdrfn);
      
      fp = fopen(nn_xdrfn, "w");
      if (fp != NULL) {

	bool_t bret;

	unsigned int size;

	unsigned int maxsize;
	
	xdrstdio_create(&xdrs, fp, XDR_ENCODE);

	bret = xdr_long(&xdrs, &num_weights);
	if (!bret) {
	  printf("Trouble writing number of weights.\n");
	  return -1;
	}

	bret = xdr_vector(&xdrs, weights, 3 * num_weights, sizeof(float), xdr_float);

	/*
	size = num_weights;
	maxsize = num_weights;
	bret = xdr_array(&xdrs, &weights, &size, maxsize, sizeof(vec3), xdr_vec3);
	*/
	
	if (!bret) {
	  printf("Trouble writing weights to %s.\n", nn_xdrfn);
	  return -1;
	}
	
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

  }

  free(weights);
  free(hidden);
  free(outputs_cache);

  free(inputs);
  free(outputs);
  
  printf("Completed\n");

  return 0;

}
