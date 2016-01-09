
#ifndef MAIN_H_
#define MAIN_H_

#include "opt.h"
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>
//#include <gsl/gsl_rng.h>
#include <math.h>
#include "ccLDA.h"


#define myrand() (double) (((unsigned long) randomMT()) / 4294967296.)
const gsl_rng_type * T;
gsl_rng * r;
double TAU;
double CONVERGED;
int MAXITER;
int NUMC;
int MCSIZE;
int NUMINIT;
int BatchSize;
double Kappa;

cclda_ss * new_cclda_ss(cclda_model* model);
cclda_model* new_cclda_model(int ntopics, int nclasses, int nterms);
cclda_var * new_cclda_var(cclda_model* model, int nmax);

void train(char* dataset, char* lblfile, int nclasses, int ntopics, int nterms,
		char* start, char* dir, char* model_name);

void mstep(cclda_corpus* corpus, cclda_model* model, cclda_ss* ss, cclda_var* var, double* lhood,
		cclda_alphaopt* etaopt, gsl_vector* x, gsl_vector* x2);

void hdp_lda_est(cclda_corpus* corpus, cclda_model* model,
		cclda_var* var, double** theta, int nmax);

int main(int argc, char* argv[]);
void write_cclda_model(cclda_model * model, cclda_var* var, char * root,cclda_corpus * corpus);
void corpus_initialize_model(cclda_model* model, cclda_corpus* corpus, cclda_ss* ss, cclda_var* var);
int max_corpus_length(cclda_corpus* c);

double doc_estep(document* doc, cclda_model* model,
		cclda_ss* ss, cclda_var* var, int d, int test, int cd, cclda_alphaopt* alphaopt);

cclda_model* load_model(char* model_root);
void write_pred_time(cclda_corpus* corpus, char * filename);
cclda_corpus* read_data(const char* data_filename, int lblchck, const char* lbl_filename);

void test(char* dataset, char* lblfile, char* model_name, char* dir);
void random_initialize_model(cclda_model * model, cclda_corpus* corpus, cclda_ss* ss, cclda_var* var);
void write_word_assignment(cclda_corpus* c,char * filename, cclda_model* model);
double log_sum(double log_a, double log_b);

#endif /* MAIN_H_ */
