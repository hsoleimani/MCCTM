#include "main.h"



int main(int argc, char* argv[])
{

	char task[40];
	char dir[400];
	char corpus_file[400];
	char label_file[400];
	char model_name[400];
	char init[400];
	int num_classes, num_topics, num_terms;
	long int seed;


	seed = atoi(argv[1]);

	gsl_rng_env_setup();

	T = gsl_rng_default;
	r = gsl_rng_alloc (T);
	gsl_rng_set (r, seed);

	printf("SEED = %ld\n", seed);

	MAXITER = 1000;
	CONVERGED = 5e-4;
	NUMINIT = 10;
	TAU = 10.0;

	strcpy(task,argv[2]);
	strcpy(corpus_file,argv[3]);
	strcpy(label_file,argv[4]);

	if (argc > 1){
		if (strcmp(task, "train")==0)
		{
			num_classes = atoi(argv[5]);
			num_topics = atoi(argv[6]);
			num_terms = atoi(argv[7]);
			strcpy(init,argv[8]);
			strcpy(dir,argv[9]);
			if (strcmp(init,"load")==0)
					strcpy(model_name, argv[10]);
			train(corpus_file, label_file, num_classes, num_topics, num_terms, init, dir, model_name);

			gsl_rng_free (r);
			return(0);
		}
		if (strcmp(task, "test")==0)
		{
			strcpy(model_name,argv[5]);
			strcpy(dir,argv[6]);
			test(corpus_file, label_file, model_name, dir);

			gsl_rng_free (r);
			return(0);
		}
    }
    return(0);
}


void train(char* dataset, char* lblfile, int nclasses, int ntopics, int nterms,
		char* start, char* dir, char* model_name)
{
	FILE* lhood_fptr;
	//FILE* fp;
	char string[100];
	char filename[100];
	int iteration, nmax;
	double lhood, prev_lhood, conv, doclkh, wrdlkh;
	//double y;
	int d, j, c;
	cclda_corpus* corpus;
	cclda_model *model = NULL;
	cclda_ss* ss = NULL;
	cclda_var* var = NULL;
	time_t t1,t2;
	gsl_vector *x;
	gsl_vector *x2;
	cclda_alphaopt * alphaopt = NULL;

	corpus = read_data(dataset, 1, lblfile);

	// nmax
	nmax = max_corpus_length(corpus);
	corpus->nterms = nterms;

	mkdir(dir, S_IRUSR|S_IWUSR|S_IXUSR);

	// set up the log likelihood log file

	sprintf(string, "%s/likelihood.dat", dir);
	lhood_fptr = fopen(string, "w");

	model = new_cclda_model(ntopics, nclasses, corpus->nterms);
	ss = new_cclda_ss(model);
	var =  new_cclda_var(model, nmax);

	if (((strcmp(start, "seeded")==0) && (corpus->ndocs < NUM_INIT*ntopics)) || (strcmp(start, "random")==0)){
		printf("random\n");
		random_initialize_model(model, corpus, ss, var);
	}
	else if (strcmp(start, "seeded")==0){
		printf("seeded\n");
		corpus_initialize_model(model, corpus, ss, var);
	}
	else if (strcmp(start, "load")==0){
		printf("load\n");
		int n;
		FILE* fileptr;
		sprintf(filename, "%s.beta", model_name);
		printf("loading %s\n", filename);
		fileptr = fopen(filename, "r");
		for (n = 0; n < corpus->nterms; n++){
			for (j = 0; j < ntopics; j++){
				fscanf(fileptr, " %lf", &model->logbeta[j][n]);
				model->beta[j][n] = exp(model->logbeta[j][n]);
			}
		}
		fclose(fileptr);

		//sprintf(filename, "%s.alpha", model_name);
		//printf("loading %s\n", filename);
		//fileptr = fopen(filename, "r");
		for (c = 0; c < model->c; c++){
			model->sumalpha[c] = 0.0;
			for (j = 0; j < ntopics; j++){
				//fscanf(fileptr, " %lf", &model->alphahat[j][c]);
				model->alphahat[j][c] = log(0.1);
				model->alpha[j][c] = exp(model->alphahat[j][c]);
				model->sumalpha[c] += model->alpha[j][c];
			}
		}
		//fclose(fileptr);

	}


	x = gsl_vector_alloc(model->m);
	x2 = gsl_vector_alloc(model->m);

	alphaopt = malloc(sizeof(cclda_alphaopt));
	alphaopt->c = model->c;
	alphaopt->m = model->m;
	alphaopt->alpha = malloc(sizeof(double)*model->m);
	alphaopt->grad = malloc(sizeof(double)*model->m);
	alphaopt->ss1 = malloc(sizeof(double)*model->c);
	alphaopt->ss2 = malloc(sizeof(double*)*model->m);
	for (j = 0; j < model->m; j++){
		alphaopt->ss2[j] = malloc(sizeof(double)*model->c);
		alphaopt->alpha[j] = 0.0;
		alphaopt->grad[j] = 0.0;
		for (c = 0; c < model->c; c++){
			alphaopt->ss2[j][c] = 0.0;
		}
	}
	for (c = 0; c < model->c; c++){
		alphaopt->ss1[c] = 0.0;
	}



	iteration = 0;
	sprintf(filename, "%s/%03d", dir,iteration);
	printf("%s\n",filename);
	write_cclda_model(model, var, filename, corpus);

	time(&t1);
	prev_lhood = -1e100;


	do{

		printf("***** VB ITERATION %d *****\n", iteration);
		lhood = 0.0;
		wrdlkh = 0.0;

		for (d = 0; d < corpus->ndocs; d++){

			doclkh = doc_estep(&(corpus->docs[d]), model, ss, var, d, 0, corpus->docs[d].label, alphaopt);

			lhood += doclkh;
			wrdlkh += corpus->docs[d].lkh;

		}

		// m-step
		mstep(corpus, model, ss, var, &lhood, alphaopt, x, x2);

		conv = fabs(prev_lhood - lhood)/fabs(prev_lhood);

		if (prev_lhood > lhood){
			printf("Oops, likelihood is decreasing! \n");
		}
		time(&t2);
		prev_lhood = lhood;

		sprintf(filename, "%s/%03d", dir,1);
		write_cclda_model(model, var, filename, corpus);

		printf("lkh = %5.5e, Conv = %5.5e, wrd-lkh = %5.5e, Time = %d\n", lhood, conv, wrdlkh, (int)(t2-t1));

		fprintf(lhood_fptr, "%d %5.5e %5.5e %5ld %5.5e\n",iteration, lhood, conv, (int)t2-t1, wrdlkh);
		fflush(lhood_fptr);
		iteration ++;

	}while((iteration < MAXITER) && (conv > CONVERGED));



	//last run to compute wrd-lkh
	lhood = 0.0;
	wrdlkh = 0.0;

	FILE* gammafp;
	sprintf(filename, "%s/final.gamma", dir);
	gammafp = fopen(filename, "w");
	for (d = 0; d < corpus->ndocs; d++){

		doclkh = doc_estep(&(corpus->docs[d]), model, ss, var, d, 1, corpus->docs[d].label, alphaopt);
		lhood += doclkh;
		wrdlkh += corpus->docs[d].lkh;
		for(j = 0; j < model->m; j++){
			fprintf(gammafp, "%lf ", var->gamma[j]);
		}
		fprintf(gammafp, "\n");

	}
	fclose(gammafp);

	printf("lkh = %5.5e, Conv = %5.5e, wrd-lkh = %5.5e, Time = %d\n", lhood, conv, wrdlkh, (int)(t2-t1));

	fprintf(lhood_fptr, "%d %5.5e %5.5e %5ld %5.5e\n",iteration, lhood, conv, (int)t2-t1, wrdlkh);
	fflush(lhood_fptr);
	fclose(lhood_fptr);

	sprintf(filename, "%s/final", dir);

	write_cclda_model(model, var, filename, corpus);

}



double doc_estep(document* doc, cclda_model* model,
		cclda_ss* ss, cclda_var* var, int d, int test, int cd, cclda_alphaopt* alphaopt){

	int n, variter, w, j;
	double normsum, cnt, temp;
	double varlkh, prev_varlkh, conv;
	//double cwrdlkh;

	prev_varlkh = -1e100;
	conv = 0.0;
	variter = 0;

	var->sumgamma = 0.0;
	for (j = 0; j < model->m; j++){
		var->sumphi[j] = ((double)doc->total)/((double)model->m);
		var->gamma[j] = model->alpha[j][cd] + var->sumphi[j];
		var->sumgamma += var->gamma[j];
		for (n = 0; n < doc->length; n++){
			var->phi[n][j] = 1.0/((double)model->m);
		}
	}


	doc->lkh= 0.0;

	prev_varlkh = -1e100;
	conv = 0.0;
	variter = 0;
	do{
		varlkh = 0.0;

		for (n = 0; n < doc->length; n++){
			w = doc->words[n];
			cnt = (double) doc->counts[n];

			normsum = 0.0;
			for (j = 0; j < model->m; j++){
				var->oldphi[j] = var->phi[n][j];

				var->phi[n][j] = gsl_sf_psi(var->gamma[j]) + model->logbeta[j][w];
				 if (j > 0)
					normsum = log_sum(normsum, var->phi[n][j]);
				else
					normsum = var->phi[n][j];

			}

			for (j = 0; j < model->m; j++){

				var->phi[n][j] = exp(var->phi[n][j] - normsum);

				temp = cnt*(var->phi[n][j] - var->oldphi[j]);
				var->gamma[j] += temp;
				var->sumgamma += temp;
				var->sumphi[j] += temp;

				if (var->phi[n][j] > 0){
					varlkh += cnt*var->phi[n][j]*(model->logbeta[j][w]-log(var->phi[n][j]));
				}
			}
		}
		varlkh += lgamma(model->sumalpha[cd]) - lgamma(var->sumgamma);
		for (j = 0; j < model->m; j++){
			varlkh += lgamma(var->gamma[j]) - lgamma(model->alpha[j][cd]);
		}
		conv = fabs(prev_varlkh - varlkh)/fabs(prev_varlkh);
		if ((prev_varlkh > varlkh) && (conv > 1e-10)){
			printf("Oops, likelihood of doc %d class %d is decreasing!\n", d, cd);
			printf("ooops class %d, %lf %lf, %5.10e %d\n", cd, varlkh, prev_varlkh, conv, variter);
		}
		prev_varlkh = varlkh;
		variter ++;

	}while((variter < MAXITER) && (conv > CONVERGED));



	if (test == 0){
		alphaopt->ss1[cd] += 1.0;
		for (j = 0; j < model->m; j++){
			alphaopt->ss2[j][cd] += gsl_sf_psi(var->gamma[j]) - gsl_sf_psi(var->sumgamma);

			for (n = 0; n < doc->length; n++){
				w = doc->words[n];
				cnt = (double) doc->counts[n];
				temp = cnt*var->phi[n][j];
				ss->beta[j][w] += temp;
				ss->sumbeta[j] += temp;
			}
		}

	}

	// compute likelihood on wrd space
	doc->lkh = 0.0;
	if (test == 1){
		for (j = 0; j < model->m; j++){
			var->oldphi[j] = var->gamma[j]/var->sumgamma; //(pseudo) topic proportions
		}
		doc->lkh = 0.0;
		for (n = 0; n < doc->length; n++){
			w = doc->words[n];
			cnt = (double) doc->counts[n];
			temp = 0.0;
			for (j = 0; j < model->m; j++){
				temp += var->oldphi[j]*model->beta[j][w];
			}
			doc->lkh += cnt*log(temp);
		}
	}

	return(varlkh);

}


void mstep(cclda_corpus* corpus, cclda_model* model, cclda_ss* ss, cclda_var* var, double* lhood,
		cclda_alphaopt* alphaopt, gsl_vector* x, gsl_vector* x2){

	int j, c, n;
	//double lkh;
	//lkh = 0.0;

	for (j = 0; j < model->m; j++){
		for (n = 0; n < model->n; n++){
			model->beta[j][n] = ss->beta[j][n]/ss->sumbeta[j];
			if (model->beta[j][n] == 0)
				model->beta[j][n] = EPS;
			model->logbeta[j][n] = log(model->beta[j][n]);

			//lkh += ss->beta[j][n]*model->logbeta[j][n];

			ss->beta[j][n] = 0.0;
		}
		ss->sumbeta[j] = 0.0;
	}

	for (c = 0; c < model->c; c++){
		alphaopt->c = c;
		for (j = 0; j < model->m; j++){
			gsl_vector_set(x, j, model->alphahat[j][c]);
			gsl_vector_set(x2, j, model->alphahat[j][c]);
		}

		optimize_alpha(x, (void *)alphaopt, model->m, x2);

		model->sumalpha[c] = 0.0;
		alphaopt->ss1[c] = 0.0;
		for (j = 0; j < model->m; j++){
			model->alphahat[j][c] = gsl_vector_get(x2, j);
			model->alpha[j][c] = exp(model->alphahat[j][c]);
			model->sumalpha[c] += model->alpha[j][c];
			alphaopt->ss2[j][c] = 0.0;

		}
	}

}


void test(char* dataset, char* lblfile, char* model_name, char* dir)
{

	FILE* lhood_fptr;
	//FILE* fp;
	char string[100];
	char filename[100];
	int iteration;
	int d, doclkh, nmax, c, j;
	double lhood, wrdlkh;
	double* mu;

	cclda_corpus* corpus;
	cclda_model *model = NULL;
	cclda_ss* ss = NULL;
	cclda_var* var = NULL;
	time_t t1,t2;


	corpus = read_data(dataset, 1, lblfile); //change to tmax, kmax later
	nmax = max_corpus_length(corpus);

	mkdir(dir, S_IRUSR|S_IWUSR|S_IXUSR);

	// set up the log likelihood log file
	sprintf(string, "%s/test-lhood.dat", dir);
	lhood_fptr = fopen(string, "w");

	model = load_model(model_name);

	ss = new_cclda_ss(model);
	var =  new_cclda_var(model, nmax);

	mu = malloc(sizeof(double)*model->c);
	for (c = 0; c < model->c; c++){
		mu[c] = 0.0;
	}

	cclda_alphaopt* alphaopt = NULL; //don't really need this ...
	alphaopt = malloc(sizeof(cclda_alphaopt));
	alphaopt->c = model->c;
	alphaopt->m = model->m;
	alphaopt->alpha = malloc(sizeof(double)*model->m);
	alphaopt->grad = malloc(sizeof(double)*model->m);
	alphaopt->ss1 = malloc(sizeof(double)*model->c);
	alphaopt->ss2 = malloc(sizeof(double*)*model->m);
	for (j = 0; j < model->m; j++){
		alphaopt->ss2[j] = malloc(sizeof(double)*model->c);
		alphaopt->alpha[j] = 0.0;
		alphaopt->grad[j] = 0.0;
		for (c = 0; c < model->c; c++){
			alphaopt->ss2[j][c] = 0.0;
		}
	}
	for (c = 0; c < model->c; c++){
		alphaopt->ss1[c] = 0.0;
	}


	iteration = 0;
    /*sprintf(filename, "%s/test%03d", dir,iteration);
    printf("%s\n",filename);
	write_cclda_model(model, var, filename, corpus, theta);*/

	lhood = 0.0;
	wrdlkh = 0.0;

	time(&t1);

	double normsum;
	FILE* gammafp;
	FILE* mufp;
	sprintf(filename, "%s/testfinal.gamma", dir);
	gammafp = fopen(filename, "w");
	sprintf(filename, "%s/testfinal.mu", dir);
	mufp = fopen(filename, "w");

	for (d = 0; d < corpus->ndocs; d++){

		for (c = 0; c < model->c; c++){

			doclkh = doc_estep(&(corpus->docs[d]), model, ss, var, d, 1, c, alphaopt);

			lhood += doclkh;
			if (c == corpus->docs[d].label){
				wrdlkh += corpus->docs[d].lkh;
			}
			mu[c] = doclkh;
			if(c > 0)
				normsum = log_sum(normsum, doclkh);
			else
				normsum = doclkh;

			if (c == corpus->docs[d].label){
				for(j = 0; j < model->m; j++){
					fprintf(gammafp, "%lf ", var->gamma[j]);
				}
				fprintf(gammafp, "\n");
			}
		}
		for (c = 0; c < model->c; c++){
			mu[c] = exp(mu[c] - normsum);
			fprintf(mufp, "%lf ", mu[c]);
		}
		fprintf(mufp, "\n");

	}
	fclose(gammafp);
	fclose(mufp);

	time(&t2);

	fprintf(lhood_fptr, "%d %5.5e %5.5e %5ld %5.5e \n",iteration, lhood, 0.0, (int)t2-t1, wrdlkh);
	fflush(lhood_fptr);
	fclose(lhood_fptr);
	//*************************************


	sprintf(filename, "%s/testfinal", dir);
	write_cclda_model(model, var, filename, corpus);

}

cclda_model* new_cclda_model(int ntopics, int nclasses, int nterms)
{
	int n, j, c;

	cclda_model* model = malloc(sizeof(cclda_model));
	model->c = nclasses;
	model->m = ntopics;
	model->n = nterms;

	model->beta = malloc(sizeof(double*)*ntopics);
	model->logbeta = malloc(sizeof(double*)*ntopics);
	model->alpha = malloc(sizeof(double*)*ntopics);
	model->alphahat = malloc(sizeof(double*)*ntopics);
	for (j = 0; j < ntopics; j++){
		model->beta[j] = malloc(sizeof(double)*nterms);
		model->logbeta[j] = malloc(sizeof(double)*nterms);
		for (n = 0; n < nterms; n++){
			model->beta[j][n] = 0.0;
			model->logbeta[j][n] = 0.0;
		}
		model->alpha[j] = malloc(sizeof(double)*nclasses);
		model->alphahat[j] = malloc(sizeof(double)*nclasses);
		for (c = 0; c < nclasses; c++){
			model->alpha[j][c] = 0.0;
			model->alphahat[j][c] = 0.0;
		}
	}
	model->sumalpha = malloc(sizeof(double)*nclasses);
	for (c = 0; c < nclasses; c++){
		model->sumalpha[c] = 0.0;
	}

	return(model);
}

cclda_var * new_cclda_var(cclda_model* model, int nmax){

	int n, j;

	cclda_var * var;
	var = malloc(sizeof(cclda_var));

	var->gamma = malloc(sizeof(double)*model->m);
	var->sumphi = malloc(sizeof(double)*model->m);
	var->oldphi = malloc(sizeof(double)*model->m);
	for (j = 0; j < model->m; j++){
		var->gamma[j] = 0.0;
		var->sumphi[j] = 0.0;
		var->oldphi[j] = 0.0;
	}

	var->phi = malloc(sizeof(double*)*nmax);
	for (n = 0; n < nmax; n++){
		var->phi[n] = malloc(sizeof(double)*model->m);
		for (j = 0; j < model->m; j++){
			var->phi[n][j] = 0.0;
		}
	}

	return(var);
}


cclda_ss * new_cclda_ss(cclda_model* model)
{
	int j, n;
	cclda_ss * ss;
	ss = malloc(sizeof(cclda_ss));

	ss->beta = malloc(sizeof(double*)*model->m);
	ss->sumbeta = malloc(sizeof(double)*model->m);
	for (j = 0; j < model->m; j++){
		ss->sumbeta[j] = 0.0;
		ss->beta[j] = malloc(sizeof(double)*model->n);
		for (n = 0; n < model->n; n++){
			ss->beta[j][n] = 0.0;
		}
	}
	return(ss);
}



cclda_corpus* read_data(const char* data_filename, int lblchck, const char* lbl_filename)
{
	FILE *fileptr;
	int length, count, word, n, nd, nw, lbl;
	cclda_corpus* c;

	printf("reading data from %s\n", data_filename);
	c = malloc(sizeof(cclda_corpus));
	c->docs = 0;
	c->nterms = 0;
	c->ndocs = 0;
	fileptr = fopen(data_filename, "r");
	nd = 0; nw = 0;
	while ((fscanf(fileptr, "%10d", &length) != EOF)){
		c->docs = (document*) realloc(c->docs, sizeof(document)*(nd+1));
		c->docs[nd].length = length;
		c->docs[nd].total = 0;
		c->docs[nd].label = -1;
		c->docs[nd].lkh = 0.0;
		c->docs[nd].words = malloc(sizeof(int)*length);
		c->docs[nd].counts = malloc(sizeof(int)*length);
		for (n = 0; n < length; n++){
			fscanf(fileptr, "%10d:%10d", &word, &count);
			c->docs[nd].words[n] = word;
			c->docs[nd].counts[n] = count;
			c->docs[nd].total += count;
			if (word >= nw) { nw = word + 1; }
		}
		nd++;
	}
	fclose(fileptr);
	c->ndocs = nd;
	c->nterms = nw;
	printf("number of docs    : %d\n", nd);
	printf("number of terms   : %d\n", nw);

	if (lblchck == 1){
		printf("reading data from %s\n", lbl_filename);
		fileptr = fopen(lbl_filename, "r");
		for (nd = 0; nd < c->ndocs; nd++){
			fscanf(fileptr, "%d, %d",&nw, &lbl);
			c->docs[nd].label = lbl;
		}
	}

	return(c);
}

int max_corpus_length(cclda_corpus* c)
{
    int n, max = 0;
    for (n = 0; n < c->ndocs; n++)
	if (c->docs[n].length > max) max = c->docs[n].length;
    return(max);
}


void write_cclda_model(cclda_model * model, cclda_var* var, char * root,cclda_corpus * corpus)
{
	char filename[200];
	FILE* fileptr;
	int n, j, c;

	//beta
	sprintf(filename, "%s.beta", root);
	fileptr = fopen(filename, "w");
	for (n = 0; n < model->n; n++){
		for (j = 0; j < model->m; j++){
			fprintf(fileptr, "%.10lf ",model->logbeta[j][n]);
		}
		fprintf(fileptr, "\n");
	}
	fclose(fileptr);


	//a,b
	sprintf(filename, "%s.alpha", root);
	fileptr = fopen(filename, "w");
	for (c = 0; c < model->c; c++){
		for (j = 0; j < model->m; j++)
			fprintf(fileptr, "%.10lf ",model->alphahat[j][c]);
		fprintf(fileptr, "\n");
	}
	fclose(fileptr);

	sprintf(filename, "%s.other", root);
	fileptr = fopen(filename, "w");
	fprintf(fileptr,"M %d \n",model->m);
	fprintf(fileptr,"C %d \n",model->c);
	fprintf(fileptr,"num_terms %d \n",model->n);
	fprintf(fileptr,"num_docs %d \n",corpus->ndocs);
	fclose(fileptr);

}

cclda_model* load_model(char* model_root){

	char filename[100];
	FILE* fileptr;
	int j, n, c, num_terms, num_docs, ntopics, nclasses;
	//float x;
	double y;

	cclda_model* model;
	sprintf(filename, "%s.other", model_root);
	printf("loading %s\n", filename);
	fileptr = fopen(filename, "r");
	fscanf(fileptr, "M %d\n", &ntopics);
	fscanf(fileptr, "C %d\n", &nclasses);
	fscanf(fileptr, "num_terms %d\n", &num_terms);
	fscanf(fileptr, "num_docs %d\n", &num_docs);
	fclose(fileptr);

	model  = new_cclda_model(ntopics, nclasses, num_terms);

	sprintf(filename, "%s.beta", model_root);
	printf("loading %s\n", filename);
	fileptr = fopen(filename, "r");
	for (n = 0; n < num_terms; n++){
		for (j = 0; j < ntopics; j++){
			fscanf(fileptr, " %lf", &y);
			model->logbeta[j][n] = y;
			model->beta[j][n] = exp(model->logbeta[j][n]);
		}
	}
	fclose(fileptr);

	sprintf(filename, "%s.alpha", model_root);
	printf("loading %s\n", filename);
	fileptr = fopen(filename, "r");
	for (c = 0; c < model->c; c++){
		model->sumalpha[c] = 0.0;
		for (j = 0; j < model->m; j++){
			fscanf(fileptr, "%lf ", &y);
			model->alphahat[j][c] = y;
			model->alpha[j][c] = exp(y);
			model->sumalpha[c] += model->alpha[j][c];
		}
	}
	fclose(fileptr);


	return(model);
}


void corpus_initialize_model(cclda_model* model, cclda_corpus* corpus, cclda_ss* ss, cclda_var* var)
{

	int j, n, num_init, i, d, c;
	double normsum;
	document* doc;

	if (corpus->ndocs > model->m*NUM_INIT)
		num_init = 1;
	else
		num_init = NUM_INIT;

	for (c = 0; c < model->c; c++){
		model->sumalpha[c] = 0.0;
	}

	for (j = 0; j < model->m; j++){

		normsum = 0.0;
		for (n = 0; n < model->n; n++){
			model->beta[j][n] = 1e-5;
			normsum += 1e-5;
		}

		for (i = 0; i < num_init; i++){

			d = floor(gsl_rng_uniform(r) * corpus->ndocs);
			doc = &(corpus->docs[d]);
			for (n = 0; n < doc->length; n++){
				model->beta[j][doc->words[n]] += (double)doc->counts[n];
				normsum += (double)doc->counts[n];
			}
		}
		for (n = 0; n < model->n; n++){
			model->beta[j][n] /= normsum;
			model->logbeta[j][n] = log(model->beta[j][n]);
		}

		for (c = 0; c < model->c; c++){

			model->alphahat[j][c] = 0.0;
			model->alpha[j][c] = exp(model->alpha[j][c]);
			model->sumalpha[c] += model->alpha[j][c];
		}
	}

}



void random_initialize_model(cclda_model * model, cclda_corpus* corpus, cclda_ss* ss, cclda_var* var){

	int n, j, c;
	double temp;

	for (c = 0; c < model->c; c++){
		model->sumalpha[c] = 0.0;
	}

	for (j = 0; j < model->m; j++){

		temp = 0.0;
		for (n = 0; n < model->n; n++){
			model->beta[j][n] = gsl_rng_uniform(r);
			temp += model->beta[j][n];
		}
		for (n = 0; n < model->n; n++){
			model->beta[j][n] /= temp;
			model->logbeta[j][n] = log(model->beta[j][n]);
		}

		for (c = 0; c < model->c; c++){

			model->alphahat[j][c] = gsl_ran_gaussian(r, 1);
			model->alpha[j][c] = exp(model->alpha[j][c]);
			model->sumalpha[c] += model->alpha[j][c];
		}

	}
}


double log_sum(double log_a, double log_b)
{
	double v;

	if (log_a < log_b)
		v = log_b+log(1 + exp(log_a-log_b));
	else
		v = log_a+log(1 + exp(log_b-log_a));
	return(v);
}

