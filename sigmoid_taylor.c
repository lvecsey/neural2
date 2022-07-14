
#include <math.h>

#include "sigmoid_taylor.h"

double sigmoid_taylor(double x) {

  double b;

  b = 2.0;
  
  if (x >= 0.0) {
    return 1.0 / (b - x + 0.5 * x * x);
  }

  else {
    return 1.0 - (1.0 / (b + x + 0.5 * x * x));
  }

}

