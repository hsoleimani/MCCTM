
#ifndef cclda_H_
#define cclda_H_

//#include <gsl/gsl_vector.h>
//#include <gsl/gsl_matrix.h>
#include <time.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf_psi.h>


#define NUM_INIT 20
#define SEED_INIT_SMOOTH 1.0
#define EPS 1e-50
#define PI 3.14159265359
#define max(a,b) ({ __typeof__ (a) _a = (a);  __typeof__ (b) _b = (b); _a > _b ? _a : _b; })



typedef struct
{
	int* words;
	int* counts;
	int length;
	int total;
	int label;
	double lkh; //likelihood of each doc on the word space
} document;


typedef struct
{
	document* docs;
	int nterms;
	int ndocs;
} cclda_corpus;


typedef struct cclda_model
{
	int c; // # classes
	int m; // # topics
	int D; // # docs
	int n; // # terms
	double** alpha; // [j][c]
	double** alphahat; // [j][c]
	double* sumalpha; //[c]
	double** beta; // beta[j][n]
	double** logbeta; // beta[j][n]
} cclda_model;


typedef struct cclda_var
{

	double* gamma; // gamma[j]
	double sumgamma;
	double** phi; // phi[n][j]
	double* sumphi; //sumphi[j]
	double* oldphi; //oldphi[j]

} cclda_var;

typedef struct cclda_ss
{
	double** beta;
	double* sumbeta;
} cclda_ss;


typedef struct cclda_alphaopt
{
	double* alpha;
	double* grad;
	double* ss1; //[c]
	double** ss2; //[j][c]
	int c;
	int m;
} cclda_alphaopt;

#endif /* cclda_H_ */
