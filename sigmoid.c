
#include <math.h>

#include "sigmoid.h"

double sigmoid(double x) {
  return 1.0 / (1.0 + exp(-x));
}

