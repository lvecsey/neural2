
Set a number of weights for a single layer neural network to match
the expected output. Input values are between -1..1 and the training data has output values that are 0..1

Give an input value to the program and after running it, the result should say whether it is out of range (close to zero), or it should give a value close to 1 signifying that a match was made.

In the default configuration there are 1200 rows of training data, 80% of which will be presented to the neural network at any given time during training (across the 64 neuron nodes, with the addition of one bias node)

*Dataset*

For the training data, there are 3 input columns of various values -1..1

Half of the rows have a mapping such that if the first input is between 0.2 to 0.4,
and the second input is between 0.7 and 0.9 then the output value (or column) is between 0.95 and 1.0.

The remaining rows have a second input value between -0.4 and -0.2, and the third input value is between 0.6 and 0.8; the output column is also between 0.95 and 1.0

Output values fall within the ranges described however they aren't distributed randomly within that range, rather there is a type of smoothing function to generate the data.

This is specified more completely in training_data.c

*Configuration*

Edit neural2.c to adjust the number of neurons, number of iterations (defaults) represented as two defines.

*Compiling*

```console
make training_data.xdr
make neural2
```

*Running*

```console
./neural2
./neural2-lookup neural_net.xdr 0.34 0.76 0.21
./neural2-lookup neural_net.xdr 0.57 -0.37 0.79
```

Each time you run ./neural2 it will run the weights through an additional set of iterations. It re-saves the weights at the end of the run.

Once completed just perform a lookup with ./neural2-lookup using a set of input values, to see a quick output response from the neural network.

To start again, you can remove the neural network weights. 

```console
rm neural_net.xdr
./neural2
```

*Test results - What to look for*

```console
make test-values
```

This will run four ./neural2-lookup tests with input data in valid ranges, similar to the training data. The first two lookups should return an output final that is close to 1. The last two lookups should be near 0

*Verbose mode*

You can run a low amount of iterations with the progress meter off, to also see line by line that the mean squared error (MSE) is reducing.

```console
VERBOSE=1 PROGRESS=0 ./neural2 64 150
```

*Silent mode*

If you have a sense for how long your runtime will be, you can run with minimal output (no progress meter)

```console
PROGRESS=0 ./neural2 64 1250000
```

*Plotting the error*

Use the mse.plot file which uses the outputted mse.dat file (single file lines of values) to generate a png file.

```console
make neural2
rm neural_net.xdr
./neural2 64 1250000
gnuplot mse.plot
feh mse.png
```

*Developer notes*

training_inputs and training_outputs are the values imported through the training_data.xdr

dataset_inputs and dataset_outputs are just in a different representation (vec3)

inputs and outputs variables are used for the actual training routine of the neural network

input variable is used for neural network lookup

