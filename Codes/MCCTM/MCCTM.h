
#ifndef mcctm_H_
#define mcctm_H_

//#include <gsl/gsl_vector.h>
//#include <gsl/gsl_matrix.h>
#include <time.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf_psi.h>


#define NUM_INIT 20
#define SEED_INIT_SMOOTH 1.0
#define EPS 1e-40
#define PI 3.14159265359
#define max(a,b) ({ __typeof__ (a) _a = (a);  __typeof__ (b) _b = (b); _a > _b ? _a : _b; })



typedef struct
{
	int* words;
	int* counts;
	int length;
	int total;
	int label;
	int gt_label;
	double lkh; //likelihood of each doc on the word space
	double lkh_givenC;
} document;


typedef struct
{
	document* docs;
	int nterms;
	int ndocs;
} mcctm_corpus;


typedef struct mcctm_model
{
	int c; // # classes
	int m; // # topics
	int D; // # docs
	int n; // # terms
	double sigma2;
	double* delta; //delta[c]
	double** alpha; // alpha[j][c]
	double** beta; // beta[j][n]
	double** logbeta; // beta[j][n]
} mcctm_model;


typedef struct mcctm_var
{
	double*** phi; //phi[n][j][c];
	double* eta; //eta[c]
	double* theta;
	double** lambda; //lambda[j][c]
	double** nu;//nu[j][c]
	double** nuhat; //nuhat[j][c]
	double* mu; //mu[c]
	double* Qc;
	double** sumphi; //sumphi[j][c]
	double* oldphi; //oldphi[j]

} mcctm_var;

typedef struct mcctm_ss
{
	double* delta; //delta[c]
	double** beta;
	double* sumbeta;
	double** alpha1; //[j][c]
	double** alpha2; //[j][c]
} mcctm_ss;


typedef struct mcctm_epsopt
{
	double* nu; //nu[j]
	double* sumphi; //sumphi[j]
	double* alpha; //alpha[j]
	double nd;
	double etainv;
	double sigma2;
	int m;
	double* x;
	double* xnew;
	double* grad;
	double* dx;
	double stpchk;
	double grad_dx;
} mcctm_epsopt;

#endif /* mcctm_H_ */
