/*
 * opt.c
 *
 *  Created on: May 22, 2015
 *      Author: studadmin
 */

#include "opt.h"

//static gsl_vector thalpha_evaluate(void *instance, const gsl_vector *x,
//		gsl_vector *grad, const int n, const gsl_vector step);
double my_f (const gsl_vector *v, void *params);
void my_df (const gsl_vector *v, void *params, gsl_vector *df);
void my_fdf (const gsl_vector *v, void *params, double *f, gsl_vector *df);


void optimize_alpha(gsl_vector * x, void * data, int n, gsl_vector * x2){

	size_t iter = 0;
	int status, j;

	const gsl_multimin_fdfminimizer_type *T;
	gsl_multimin_fdfminimizer *s;

	gsl_multimin_function_fdf my_func;

	my_func.n = n;
	my_func.f = my_f;
	my_func.df = my_df;
	my_func.fdf = my_fdf;
	my_func.params = data;

	T = gsl_multimin_fdfminimizer_conjugate_fr;
	s = gsl_multimin_fdfminimizer_alloc (T, n);
	gsl_multimin_fdfminimizer_set (s, &my_func, x, 0.01, 1e-3);

	do{
		iter++;
		status = gsl_multimin_fdfminimizer_iterate (s);
		//printf ("status = %s\n", gsl_strerror (status));
		if (status){
			if (iter == 1){
				for (j = 0; j < n; j++){
					gsl_vector_set(x2, j, gsl_vector_get(x, j));
				}
			}
			break;
		}

		status = gsl_multimin_test_gradient (s->gradient, 1e-3);

		if ((isnan(s->f)) || (isinf(s->f)))
			break;
		for (j = 0; j < n; j++){
			gsl_vector_set(x2, j, gsl_vector_get(s->x, j));
		}

	}while (status == GSL_CONTINUE && iter < 100);

	gsl_multimin_fdfminimizer_free (s);

}


double my_f (const gsl_vector *v, void *params)
{

	cclda_alphaopt * alphaopt=(cclda_alphaopt *) params;
	double f = 0.0;
	double sumalpha, sumpsialpha;
	int j, c;

	f = 0.0;
	c = alphaopt->c;
	sumalpha = 0.0;
	sumpsialpha = 0.0;
	for (j = 0; j < alphaopt->m; j++){
		alphaopt->alpha[j] = exp(gsl_vector_get(v, j));
		sumalpha += alphaopt->alpha[j];
		f += (alphaopt->alpha[j] - 1)*alphaopt->ss2[j][c];
		sumpsialpha += gsl_sf_psi(alphaopt->alpha[j]);
	}
	f += alphaopt->ss1[c]*(gsl_sf_psi(sumalpha) - sumpsialpha);

	f = -f;

	return(f);
}



void my_df (const gsl_vector *v, void *params, gsl_vector *df)
{

	cclda_alphaopt * alphaopt=(cclda_alphaopt *) params;
	double f = 0.0;
	double sumalpha, sumpsialpha, temp;
	int j, c;

	f = 0.0;
	c = alphaopt->c;
	sumalpha = 0.0;
	sumpsialpha = 0.0;
	for (j = 0; j < alphaopt->m; j++){
		alphaopt->alpha[j] = exp(gsl_vector_get(v, j));
		sumalpha += alphaopt->alpha[j];
		f += (alphaopt->alpha[j] - 1)*alphaopt->ss2[j][c];
		temp = gsl_sf_psi(alphaopt->alpha[j]);
		sumpsialpha += temp;

		alphaopt->grad[j] = alphaopt->ss2[j][c] -alphaopt->ss1[c]*temp;
	}
	f += alphaopt->ss1[c]*(gsl_sf_psi(sumalpha) - sumpsialpha);
	for (j = 0; j < alphaopt->m; j++){

		alphaopt->grad[j] += gsl_sf_psi(sumalpha)*alphaopt->ss1[c];
		gsl_vector_set(df, j, -alphaopt->alpha[j]*alphaopt->grad[j]);
	}
	f *= -1;

}


void my_fdf (const gsl_vector *v, void *params, double *f, gsl_vector *df)
{

	cclda_alphaopt * alphaopt=(cclda_alphaopt *) params;
	double sumalpha, sumpsialpha, temp;
	int j, c;

	*f = 0.0;
	c = alphaopt->c;
	sumalpha = 0.0;
	sumpsialpha = 0.0;
	for (j = 0; j < alphaopt->m; j++){
		alphaopt->alpha[j] = exp(gsl_vector_get(v, j));
		sumalpha += alphaopt->alpha[j];
		*f += (alphaopt->alpha[j] - 1)*alphaopt->ss2[j][c];
		temp = gsl_sf_psi(alphaopt->alpha[j]);
		sumpsialpha += temp;

		alphaopt->grad[j] = alphaopt->ss2[j][c] -alphaopt->ss1[c]*temp;
	}
	*f += alphaopt->ss1[c]*(gsl_sf_psi(sumalpha) - sumpsialpha);
	for (j = 0; j < alphaopt->m; j++){

		alphaopt->grad[j] += gsl_sf_psi(sumalpha)*alphaopt->ss1[c];
		gsl_vector_set(df, j, -alphaopt->alpha[j]*alphaopt->grad[j]);
	}
	*f = -(*f);

}
