
#ifndef MAIN_H_
#define MAIN_H_

#include "MCCTM.h"
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

mcctm_ss * new_mcctm_ss(mcctm_model* model);
mcctm_model* new_mcctm_model(int ntopics, int nclasses, int nterms, double lambda);
mcctm_var * new_mcctm_var(mcctm_model* model, int nmax);

void train(char* dataset, char* lblfile, int nclasses, int ntopics, double lambda,
		char* start, char* dir, char* model_name);

void mstep(mcctm_corpus* corpus, mcctm_model* model, mcctm_ss* ss, mcctm_var* var, double* lhood);


int main(int argc, char* argv[]);
void write_mcctm_model(mcctm_model * model, mcctm_var* var, char * root,mcctm_corpus * corpus, double** mu);
void corpus_initialize_model(mcctm_model* model, mcctm_corpus* corpus, mcctm_ss* ss, mcctm_var* var);
int max_corpus_length(mcctm_corpus* c);

double doc_estep(mcctm_corpus* corpus, mcctm_model* model,
		mcctm_ss* ss, mcctm_var* var, int d, mcctm_epsopt* etaopt, int test, int comp_wrdlkh);

mcctm_epsopt* new_mcctm_epsopt(mcctm_model* model);

mcctm_model* load_model(char* model_root);
void write_pred_time(mcctm_corpus* corpus, char * filename);
mcctm_corpus* read_data(const char* data_filename, int lblchck, const char* lbl_filename);

//void corpus_initialize_model(mcctm_var* alpha, mcctm_model* model, mcctm_corpus* c);//, gsl_rng * r);
void test(char* dataset, char* model_name, char* dir);
void random_initialize_model(mcctm_model * model, mcctm_corpus* corpus, mcctm_ss* ss, mcctm_var* var);
void write_word_assignment(mcctm_corpus* c,char * filename, mcctm_model* model);
double log_sum(double log_a, double log_b);

#endif /* MAIN_H_ */
