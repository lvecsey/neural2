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

#include "relu.h"

#include "relu_deriv.h"

#include "norm.h"

#include "set_cluster.h"

#include "sigmoid.h"

#include "sigmoid_deriv.h"

#include "forward.h"

#include "forward2.h"

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

int set_random(int rnd_fd, double *values, long int num_values) {

  long int valno;

  ssize_t bytes_read;

  uint64_t rndval;
  
  for (valno = 0; valno < num_values; valno++) {

    bytes_read = read(rnd_fd, &rndval, sizeof(uint64_t));
    if (bytes_read != sizeof(uint64_t)) {
      perror("read");
      return -1;
    }

    values[valno] = -1.0 + (2.0 * rndval) / 18446744073709551615.0;
    
  }

  return 0;
  
}

long int assign_trainingrows(int rnd_fd, long int num_rows, double *training_inputs, double *training_outputs, long int num_dataset, vec3 *dataset_inputs, vec3 *dataset_outputs) {

  long int rowno;

  long int acumno;

  double training_portion;

  long int num_accumulated;

  ssize_t bytes_read;

  uint64_t rndval;
  
  training_portion = 0.80;
    
  acumno = 0;
    
  for (rowno = 0; rowno < num_dataset; rowno++) {

    double training_sample;
      
    bytes_read = read(rnd_fd, &rndval, sizeof(uint64_t));
    if (bytes_read != sizeof(uint64_t)) {
      perror("read");
      return -1;
    }

    training_sample = (rndval / 18446744073709551615.0);

    if (training_sample < training_portion) {

      dataset_inputs[acumno][0] = training_inputs[0*num_rows+rowno];
      dataset_inputs[acumno][1] = training_inputs[1*num_rows+rowno];
      dataset_inputs[acumno][2] = training_inputs[2*num_rows+rowno];
      dataset_outputs[acumno][0] = training_outputs[rowno];
      dataset_outputs[acumno][1] = 0.0;
      dataset_outputs[acumno][2] = 0.0;

      acumno++;

    }

  }

  num_accumulated = acumno;
  
  return num_accumulated;
    
}



int main(int argc, char *argv[]) {

  double *weights;

  long int num_neurons;

  double *training_inputs;
  double *training_outputs;

  long int num_rows;
  long int input_cols;
  long int output_cols;
  
  vec3 *dataset_inputs;
  
  vec3 *dataset_outputs;
  
  double *inputs;

  double *outputs;

  double bias_weight;

  double *hidden_squashed;

  double *hidden_nodesums;

  double *output_nodesums;
  
  squashed_nodes sq_cache;
  
  long int wno;
  
  int rnd_fd;

  long int iterno;

  long int num_iters;

  long int num_inputs;

  long int num_outputs;
  
  double percent_completed;

  long int num_weights;

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

  long int num_layers;

  num_layers = def_layers;
  
  env_VERBOSE = getenv("VERBOSE");

  env_PROGRESS = getenv("PROGRESS");

  if (env_PROGRESS == NULL) {
    progress_meter = 1;
  }
  else {
    progress_meter = strtol(env_PROGRESS, NULL, 10);
  }
  
  learning_rate = 0.00125;

  num_inputs = 3;

  num_outputs = 1;

  inputs = malloc(sizeof(double) * num_inputs);
  if (inputs == NULL) {
    perror("malloc");
    return -1;
  }

  outputs = malloc(sizeof(double) * num_outputs);
  if (outputs == NULL) {
    perror("malloc");
    return -1;
  }
  
  num_neurons = argc>1 ? strtol(argv[1],NULL,10) : def_neurons;
  
  num_iters = argc>2 ? strtol(argv[2],NULL,10) : def_iters;

  inputs[0] = argc>3 ? strtod(argv[3],NULL) : 0.37;
  inputs[1] = argc>4 ? strtod(argv[4],NULL) : 0.75;
  inputs[2] = argc>5 ? strtod(argv[5],NULL) : 0.21;

  rnd_fd = open("/dev/urandom", O_RDONLY);
  if (rnd_fd == -1) {
    perror("open");
    return -1;
  }

  switch(num_layers) {
  case 1:
    num_weights = 3 * num_neurons;
    break;

  case 2:
    num_weights = (3 * num_neurons) + num_neurons;
    break;
  }
      
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

  num_accumulated = assign_trainingrows(rnd_fd, num_rows, training_inputs, training_outputs, num_dataset, dataset_inputs, dataset_outputs);
  
  printf("Processed %ld rows of training data.\n", num_accumulated);

  printf("Dataset inputs: \n");
  show_vector(dataset_inputs, 10);

  printf("Dataset outputs: \n");
  show_vector(dataset_outputs, 10);

  printf("Allocating weights and rest of neural network.\n");

  {

    printf("Allocating %ld weights.\n", num_weights);
    weights = malloc(sizeof(double) * num_weights);
    if (weights == NULL) {
      perror("malloc");
      return -1;
    }

    printf("Allocating %ld hidden (squashed) nodes.\n", num_neurons);    
    hidden_squashed = malloc(sizeof(double) * num_neurons);
    if (hidden_squashed == NULL) {
      perror("malloc");
      return -1;
    }
    
    printf("Allocating %ld hidden nodesums.\n", num_neurons);    
    hidden_nodesums = malloc(sizeof(double) * num_neurons);
    if (hidden_nodesums == NULL) {
      perror("malloc");
      return -1;
    }

    printf("Allocating %ld output nodesums.\n", num_neurons);    
    output_nodesums = malloc(sizeof(double) * num_neurons);
    if (output_nodesums == NULL) {
      perror("malloc");
      return -1;
    }
    
  }

  {

    sq_cache.hidden_squashed = hidden_squashed;
    
    sq_cache.hidden_nodesums = hidden_nodesums;

    sq_cache.output_nodesums = output_nodesums;

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

      bret = xdr_vector(&xdrs, weights, num_weights, sizeof(double), xdr_double);
      if (!bret) {
	printf("Trouble retrieving weights from %s.\n", nn_xdrfn);
	return -1;
      }

      bret = xdr_vector(&xdrs, &bias_weight, 1, sizeof(double), xdr_double);

      if (!bret) {
	printf("Trouble retrieving bias weight from %s.\n", nn_xdrfn);
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

	weights[wno] *= 0.95;

	weights[wno] += 1.05;
	
      }

      bias_weight = 1.0; 

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

    nf.mse = 0.0;
    
    rowno = 0;
    
    for (iterno = 0; iterno < num_iters; iterno++) {

      long int unique_trainings;
      
      percent_completed = iterno; percent_completed /= (num_iters - 1);

      unique_trainings = 10;
      
      {

	if (!rowno && (!(iterno % (num_iters / unique_trainings)))) {

	  num_accumulated = assign_trainingrows(rnd_fd, num_rows, training_inputs, training_outputs, num_dataset, dataset_inputs, dataset_outputs);

	}
	
	inputs[0] = dataset_inputs[rowno][0];
	inputs[1] = dataset_inputs[rowno][1];
	inputs[2] = dataset_inputs[rowno][2];

	outputs[0] = dataset_outputs[rowno][0];

	rowno++;

	rowno %= num_accumulated;
	
	switch(num_layers) {
	case 1:

	  nf = neural_forwarderr(inputs, outputs[0], num_neurons, weights, bias_weight, &sq_cache);
	  
	  break;

	case 2:
	
	  nf = neural_forwarderr2(inputs, outputs[0], num_neurons, weights, bias_weight, &sq_cache);

	  break;

	}
	  
	if (log_msefn != NULL) {
	  fprintf(log_msefn, "%g\n", nf.mse);
	}

	if (num_layers == 1) {

	  double z;
	  double delta[3];
	  double delta_target;

	  actual_y = outputs[0];

	  delta_target = (actual_y - nf.current_output);
	  
	  for (wno = 0; wno < num_neurons; wno++) {

	    z = sq_cache.hidden_nodesums[wno];
	    
	    delta[0] = weights[3*wno+0] * delta_target * sigmoid_deriv(sq_cache.hidden_nodesums[wno]);
	    delta[1] = weights[3*wno+1] * delta_target * sigmoid_deriv(sq_cache.hidden_nodesums[wno]);
	    delta[2] = weights[3*wno+2] * delta_target * sigmoid_deriv(sq_cache.hidden_nodesums[wno]);
	    
	    weights[3*wno+0] -= -learning_rate * (delta[0] * z);
	    weights[3*wno+1] -= -learning_rate * (delta[1] * z);
	    weights[3*wno+2] -= -learning_rate * (delta[2] * z);	    
	
	  }	  

	  {
	    z = 1.0;
	    delta[0] = bias_weight * delta_target * sigmoid_deriv(z);
	    bias_weight -= -learning_rate * (delta[0] * z);
	  }
	  
	}
	
	if (num_layers == 2) {

	  double z;
	  double delta;
	  double delta_target;

	  double *hidden_weights;
	  double *output_weights;

	  {
	  
	    hidden_weights = weights;
	    output_weights = (hidden_weights + num_neurons);

	  }
	    
	  actual_y = outputs[0];

	  delta_target = (actual_y - nf.current_output);
	  
	  for (wno = 0; wno < num_neurons; wno++) {

	    delta = output_weights[wno] * delta_target * sigmoid_deriv(sq_cache.hidden_nodesums[wno]);
	
	    z = sq_cache.hidden_squashed[wno];
	    
	    output_weights[wno] -= -learning_rate * (delta * z);
	
	  }	  

	  {
	    z = 1.0;
	    delta = bias_weight * delta_target * sigmoid_deriv(z);
	    bias_weight -= -learning_rate * (delta * z);
	  }

	  {

	    double delta_1;

	    for (wno = 0; wno < num_neurons; wno++) {

	      z = inputs[0];
	      delta_1 = (sq_cache.hidden_nodesums[wno] - inputs[0]);
	      delta = hidden_weights[wno] * delta_1 * sigmoid_deriv(inputs[0]);
	      hidden_weights[wno] -= -learning_rate * (delta * z);

	      z = inputs[1];
	      delta_1 = (sq_cache.hidden_nodesums[wno] - inputs[1]);
	      delta = hidden_weights[wno] * delta_1 * sigmoid_deriv(inputs[1]);
	      hidden_weights[wno] -= -learning_rate * (delta * z);

	      z = inputs[2];	      
	      delta_1 = (sq_cache.hidden_nodesums[wno] - inputs[2]);
	      delta = hidden_weights[wno] * delta_1 * sigmoid_deriv(inputs[2]);
	      hidden_weights[wno] -= -learning_rate * (delta * z);	      
	
	    }

	  }
	    
	}
	  
	if (progress_meter && (!(iterno % num_accumulated))) {

	  wno = (iterno % num_neurons);
	
	  printf("[%ld] Percent completed %.04g%% (Sample weights %.03g %.03g %.03g) Current output %g mse %g    \r", iterno, 100.0 * percent_completed, weights[3*wno+0], weights[3*wno+1], weights[3*wno+2], nf.current_output, nf.mse);
	
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

	xdrstdio_create(&xdrs, fp, XDR_ENCODE);

	bret = xdr_long(&xdrs, &num_weights);
	if (!bret) {
	  printf("Trouble writing number of weights.\n");
	  return -1;
	}

	bret = xdr_vector(&xdrs, weights, num_weights, sizeof(double), xdr_double);

	if (!bret) {
	  printf("Trouble writing weights to %s.\n", nn_xdrfn);
	  return -1;
	}

	bret = xdr_vector(&xdrs, &bias_weight, 1, sizeof(double), xdr_double);

	if (!bret) {
	  printf("Trouble writing bias weight to %s.\n", nn_xdrfn);
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

  free(dataset_inputs);
  free(dataset_outputs);

  free(training_inputs);
  free(training_outputs);
  
  free(weights);
  free(hidden_squashed);
  free(hidden_nodesums);
  free(output_nodesums);

  free(inputs);
  free(outputs);
  
  printf("Completed\n");

  return 0;

}
