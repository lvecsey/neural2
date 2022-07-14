
#include <math.h>

#include "linmath.h"

void norm(vec3 a) {

  double mag;

  mag = sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);

  a[0] /= mag;
  a[1] /= mag;
  a[2] /= mag;

}
