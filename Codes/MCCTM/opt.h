
#ifndef OPT_H_
#define OPT_H_

#include "MCCTM.h"
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <gsl/gsl_multimin.h>


void optimize_eps(gsl_vector * x, void * data, int n, gsl_vector * xx);
void my_optimize_eps(mcctm_epsopt * epsopt, int n);


#endif /* OPT_H_ */
