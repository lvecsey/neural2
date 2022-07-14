
#include <math.h>

#include "sigmoid.h"

#include "sigmoid_deriv.h"

double sigmoid_deriv(double x) {

  double s;

  s = sigmoid(x);

  return 2 * s * (1.0 - s);
  
}
