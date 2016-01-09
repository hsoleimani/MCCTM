
#include "opt.h"

double my_f (const gsl_vector *v, void *params);
void my_df (const gsl_vector *v, void *params, gsl_vector *df);
void my_fdf (const gsl_vector *v, void *params, double *f, gsl_vector *df);

double myopt_f (mcctm_epsopt * epsopt, double t);
double myopt_fdf (mcctm_epsopt * epsopt);

void my_optimize_eps(mcctm_epsopt * epsopt, int n){

	double alpha, beta, t;
	double f, newf;
	int j, abort;

	beta = 0.5;
	alpha = 0.01;

	do{
		abort = 0;
		t = 1.0;

		f = myopt_fdf(epsopt);
		newf = myopt_f(epsopt, t);
		while(newf > f + alpha*t*epsopt->grad_dx)
		{
			t *= beta;
			newf = myopt_f(epsopt, t);
			if ((t < 1e-6) && (newf > f)){
				abort = 1;
				break;
			}
		}
		if (abort == 1)
			break;

		//accept new x
		for (j = 0; j < n; j++){
			epsopt->x[j] = epsopt->xnew[j];
		}

		//check convergence
		if (epsopt->stpchk < 1e-4)
			break;

	}while(1);
}

double myopt_fdf (mcctm_epsopt * epsopt)
{

	double tempgrad, lambda, temp, f;
	double a, b, d, c11, c22, c12, det; //hessian terms
	int j;

	f = 0.0;
	epsopt->grad_dx = 0.0;
	for (j = 0; j < epsopt->m; j++){
		lambda = epsopt->x[2*j];
		epsopt->nu[j] = exp(epsopt->x[2*j+1]);
		temp = exp(epsopt->alpha[j] + lambda + 0.5*epsopt->nu[j]);

		f += -(pow(lambda,2.0) + epsopt->nu[j])/(2*epsopt->sigma2)
				+ lambda*epsopt->sumphi[j] -
				epsopt->etainv*epsopt->nd*temp +
				0.5*log(epsopt->nu[j]);

		tempgrad = -lambda/epsopt->sigma2 + epsopt->sumphi[j] -
				epsopt->nd*epsopt->etainv*temp;
		epsopt->grad[2*j] = -tempgrad;

		tempgrad = -0.5/epsopt->sigma2 - 0.5*epsopt->nd*epsopt->etainv*temp + 0.5/epsopt->nu[j];
		epsopt->grad[2*j + 1] = -epsopt->nu[j]*tempgrad;

		a = 1.0/epsopt->sigma2 + epsopt->nd*epsopt->etainv*temp; //*-1
		d = epsopt->nu[j]*(-tempgrad + epsopt->nu[j]*0.25*epsopt->nd*epsopt->etainv*temp) + 0.5;
		b = epsopt->nu[j]*0.5*epsopt->nd*epsopt->etainv*temp;
		det = a*d - b*b;
		if (det < 1e-10){
			epsopt->dx[2*j] = -epsopt->grad[2*j];
			epsopt->dx[2*j+1] = -epsopt->grad[2*j+1];
		}else{
			det = 1.0/det;
			c11 = d*det;
			c22 = a*det;
			c12 = -b*det;
			epsopt->dx[2*j] = -(c11*epsopt->grad[2*j] + c12*epsopt->grad[2*j+1]);
			epsopt->dx[2*j+1] = -(c12*epsopt->grad[2*j] + c22*epsopt->grad[2*j+1]);
		}

		epsopt->grad_dx += epsopt->dx[2*j]*epsopt->grad[2*j] + epsopt->dx[2*j+1]*epsopt->grad[2*j+1];
	}
	epsopt->stpchk = -epsopt->grad_dx;

	f = -f;

	return(f);
}

double myopt_f (mcctm_epsopt * epsopt, double t)
{

	double lambda, temp, f;
	int j;

	f = 0.0;
	epsopt->grad_dx = 0.0;
	for (j = 0; j < epsopt->m; j++){
		epsopt->xnew[2*j] = epsopt->x[2*j] + t*epsopt->dx[2*j];
		lambda = epsopt->xnew[2*j];
		epsopt->xnew[2*j+1] = epsopt->x[2*j+1] + t*epsopt->dx[2*j+1];
		epsopt->nu[j] = exp(epsopt->xnew[2*j+1]);
		temp = exp(epsopt->alpha[j] + lambda + 0.5*epsopt->nu[j]);

		f += -(pow(lambda,2.0) + epsopt->nu[j])/(2*epsopt->sigma2)
				+ lambda*epsopt->sumphi[j] -
				epsopt->etainv*epsopt->nd*temp +
				0.5*log(epsopt->nu[j]);
	}

	f = -f;

	return(f);
}


void optimize_eps(gsl_vector * x, void * data, int n, gsl_vector * x2){

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
	//T = gsl_multimin_fdfminimizer_vector_bfgs2;
	s = gsl_multimin_fdfminimizer_alloc (T, n);

	//gsl_multimin_fdfminimizer_set (s, &my_func, x, 0.1, 0.01);
	gsl_multimin_fdfminimizer_set (s, &my_func, x, 0.01, 1e-4);

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
		//print_state (iter, s);
		//printf ("status = %s\n", gsl_strerror (status));
		/*if (status == GSL_SUCCESS)
		printf ("Minimum found at:\n");

		printf ("%5d %.5f %.5f %10.5f\n", iter,
			  gsl_vector_get (s->x, 0),
			  gsl_vector_get (s->x, 1),
			  s->f);*/
		if ((isnan(s->f)) || (isinf(s->f)))
			break;
		for (j = 0; j < n; j++){
			gsl_vector_set(x2, j, gsl_vector_get(s->x, j));
		}

	}while (status == GSL_CONTINUE && iter < 100);

	gsl_multimin_fdfminimizer_free (s);
	//gsl_vector_free (x);

	//return 0;
}


//static gsl_vector theps_evaluate(void *instance, const gsl_vector *x,
//		gsl_vector *grad, const int m, const gsl_vector step)
double my_f (const gsl_vector *v, void *params)
{

	mcctm_epsopt * epsopt=(mcctm_epsopt *) params;
	double f = 0.0;
	double lambda;
	int j;

	f = 0.0;

	for (j = 0; j < epsopt->m; j++){
		lambda = gsl_vector_get(v, 2*j);
		epsopt->nu[j] = exp(gsl_vector_get(v, 2*j+1));
		f += -(pow(lambda,2.0) + epsopt->nu[j])/(2*epsopt->sigma2)
				+ lambda*epsopt->sumphi[j] -
				epsopt->etainv*epsopt->nd*
					exp(epsopt->alpha[j] + lambda + 0.5*epsopt->nu[j]) +
					0.5*log(epsopt->nu[j]);
	}


	f = -f;

	return(f);
}



void my_df (const gsl_vector *v, void *params, gsl_vector *df)
{

	mcctm_epsopt * epsopt=(mcctm_epsopt *) params;
	//double f = 0.0;
	double lambda, temp, tempgrad;
	int j;

	//*f = 0.0;
	for (j = 0; j < epsopt->m; j++){
		lambda = gsl_vector_get(v, 2*j);
		epsopt->nu[j] = exp(gsl_vector_get(v, 2*j+1));
		temp = exp(epsopt->alpha[j] + lambda + 0.5*epsopt->nu[j]);

		tempgrad = -lambda/epsopt->sigma2 + epsopt->sumphi[j] -
				epsopt->nd*epsopt->etainv*temp;
		gsl_vector_set(df, 2*j, -tempgrad);

		tempgrad = -0.5/epsopt->sigma2 - 0.5*epsopt->nd*epsopt->etainv*temp + 0.5/epsopt->nu[j];
		gsl_vector_set(df, 2*j+1, -epsopt->nu[j]*tempgrad);

	}



}


void my_fdf (const gsl_vector *v, void *params, double *f, gsl_vector *df)
{


	mcctm_epsopt * epsopt=(mcctm_epsopt *) params;
	//double f = 0.0;
	double tempgrad, temp, lambda;
	int j;

	*f = 0.0;
	for (j = 0; j < epsopt->m; j++){

		lambda = gsl_vector_get(v, 2*j);
		epsopt->nu[j] = exp(gsl_vector_get(v, 2*j+1));
		temp = exp(epsopt->alpha[j] + lambda + 0.5*epsopt->nu[j]);

		*f += -(pow(lambda,2.0) + epsopt->nu[j])/(2*epsopt->sigma2)
				+ lambda*epsopt->sumphi[j] -
				epsopt->etainv*epsopt->nd*temp +
				0.5*log(epsopt->nu[j]);;

		tempgrad = -lambda/epsopt->sigma2 + epsopt->sumphi[j] -
				epsopt->nd*epsopt->etainv*temp;
		gsl_vector_set(df, 2*j, -tempgrad);

		tempgrad = -0.5/epsopt->sigma2 - 0.5*epsopt->nd*epsopt->etainv*temp + 0.5/epsopt->nu[j];
		gsl_vector_set(df, 2*j+1, -epsopt->nu[j]*tempgrad);
	}

	*f = -(*f);
}
