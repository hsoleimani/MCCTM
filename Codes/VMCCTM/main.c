#include "main.h"



int main(int argc, char* argv[])
{

	char task[40];
	char dir[400];
	char corpus_file[400];
	char label_file[400];
	char model_name[400];
	char init[400];
	int num_classes, num_topics;
	double sigma2, nu;
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

	strcpy(task,argv[2]);
	strcpy(corpus_file,argv[3]);

	if (argc > 1){
		if (strcmp(task, "train")==0){
			strcpy(label_file,argv[4]);
			num_classes = atoi(argv[5]);
			num_topics = atoi(argv[6]);
			sigma2 = atof(argv[7]);
			nu = atof(argv[8]);
			strcpy(init,argv[9]);
			strcpy(dir,argv[10]);
			if (strcmp(init, "load") == 0)
				strcpy(model_name, argv[11]);
			train(corpus_file, label_file, num_classes, num_topics, sigma2, nu, init, dir, model_name);

			gsl_rng_free (r);
			return(0);
		}
		if (strcmp(task, "test")==0){
			strcpy(model_name,argv[4]);
			strcpy(dir,argv[5]);
			test(corpus_file, model_name, dir);

			gsl_rng_free (r);
			return(0);
		}
	}
	return(0);
}


void train(char* dataset, char* lblfile, int nclasses, int ntopics, double sigma2, double nu,
		char* start, char* dir, char* model_name)
{
	FILE* lhood_fptr;
	FILE* fileptr;
	char string[100];
	char filename[100];
	int iteration, nmax;
	double lhood, prev_lhood, conv, doclkh, wrdlkh;
	double y;
	int d, n, j, c;
	double** mu;

	mcctm_corpus* corpus;
	mcctm_model *model = NULL;
	mcctm_ss* ss = NULL;
	mcctm_var* var = NULL;
	mcctm_epsopt * epsopt = NULL;
	time_t t1,t2;

	corpus = read_data(dataset, 1, lblfile);

	// nmax
	nmax = max_corpus_length(corpus);

	mkdir(dir, S_IRUSR|S_IWUSR|S_IXUSR);

	// set up the log likelihood log file
	sprintf(string, "%s/likelihood.dat", dir);
	lhood_fptr = fopen(string, "w");

	if (strcmp(start, "seeded")==0){  //not updated
		printf("seeded\n");
		model = new_mcctm_model(ntopics, nclasses, corpus->nterms, sigma2 ,nu);
		ss = new_mcctm_ss(model);
		var =  new_mcctm_var(model, nmax);
		corpus_initialize_model(model, corpus, ss, var);
	}
	else if (strcmp(start, "random")==0){
		printf("random\n");
		model = new_mcctm_model(ntopics, nclasses, corpus->nterms, sigma2, nu);
		ss = new_mcctm_ss(model);
		var =  new_mcctm_var(model, nmax);
		random_initialize_model(model, corpus, ss, var);
	}
	else if (strcmp(start, "load")==0){
		printf("load\n");
		model = new_mcctm_model(ntopics, nclasses, corpus->nterms, sigma2, nu);
		model->D = corpus->ndocs;
		ss = new_mcctm_ss(model);
		var =  new_mcctm_var(model, nmax);

		random_initialize_model(model, corpus, ss, var);

		//load beta
		for (j = 0; j < ntopics; j++){
			model->sumgamma[j] = 0.0;
		}
		sprintf(filename, "%s.beta", model_name);
		printf("loading %s\n", filename);
		fileptr = fopen(filename, "r");
		for (n = 0; n < model->n; n++){
			for (j = 0; j < ntopics; j++){
				fscanf(fileptr, " %lf", &y);
				model->gamma[j][n] = y;
				model->sumgamma[j] += model->gamma[j][n];
			}
		}
		fclose(fileptr);
		// compute Elogbeta
		for (j = 0; j < ntopics; j++){
			y = gsl_sf_psi(model->sumgamma[j]);
			for (n = 0; n < model->n; n++){
				model->Elogbeta[j][n] = gsl_sf_psi(model->gamma[j][n]) - y;
				model->expElogbeta[j][n] = exp(model->Elogbeta[j][n]);
			}
		}

		//load alpha
		sprintf(filename, "%s.alpha", model_name);
		printf("loading %s\n", filename);
		fileptr = fopen(filename, "r");
		for (j = 0; j < ntopics; j++){
			for (c = 0; c < model->c; c++){
				fscanf(fileptr, " %lf", &y);
				model->alpha[j][c] = y + EPS;
			}
		}
		fclose(fileptr);

	}

	mu = malloc(sizeof(double*)*corpus->ndocs);
	for (d = 0; d < corpus->ndocs; d++){
		mu[d] = malloc(sizeof(double)*model->c);
		for (c = 0; c < model->c; c++){
			mu[d][c] = 0.0;
		}
	}

	mcctm_nuopt* nuopt = malloc(sizeof(mcctm_nuopt));
	nuopt->nterms = model->n;
	nuopt->ntopics = model->m;

	epsopt = new_mcctm_epsopt(model);

    iteration = 0;
    //sprintf(filename, "%s/%03d", dir,iteration);
    //printf("%s\n",filename);
	//write_mcctm_model(model, var, filename, corpus, mu);

    time(&t1);
	prev_lhood = -1e100;
	double* phi = malloc(sizeof(double)*model->m);

	do{

		printf("***** VB ITERATION %d *****\n", iteration);

		lhood = 0.0;
		wrdlkh = 0.0;
		for (d = 0; d < corpus->ndocs; d++){

			doclkh = doc_estep(&(corpus->docs[d]), model, ss, var, d, epsopt, 0, 0, phi);

			for (c = 0; c < model->c; c++){
				mu[d][c] = var->mu[c];
			}

			lhood += doclkh;
		}

		// m-step
		mstep(corpus, model, ss, var, nuopt, &lhood);

		conv = fabs(prev_lhood - lhood)/fabs(prev_lhood);

		if (prev_lhood > lhood){
			printf("Oops, likelihood is decreasing! \n");
		}
		time(&t2);
		prev_lhood = lhood;

		//sprintf(filename, "%s/%03d", dir,1);
		//write_mcctm_model(model, var, filename, corpus, mu);

		printf("lkh = %5.5e, Conv = %5.5e, wrd-lkh = %5.5e, Time = %d\n", lhood, conv, wrdlkh, (int)(t2-t1));
		fprintf(lhood_fptr, "%d %5.5e %5.5e %5ld %5.5e\n",iteration, lhood, conv, (int)t2-t1, wrdlkh);
		fflush(lhood_fptr);

		iteration ++;

	}while((iteration < MAXITER) && (conv > CONVERGED));


	FILE* doclkhFP;
	sprintf(string, "%s/doclkh.dat", dir);
	doclkhFP = fopen(string, "w");
	//last run to compute wrd-lkh
	lhood = 0.0;
	wrdlkh = 0.0;
	for (d = 0; d < corpus->ndocs; d++){

		doclkh = doc_estep(&(corpus->docs[d]), model, ss, var, d, epsopt, 1, 1, phi);
		//printf("%d %lf\n",d,doclkh);
		for (c = 0; c < model->c; c++){
			mu[d][c] = var->mu[c];
		}
		lhood += doclkh;
		wrdlkh += corpus->docs[d].lkh;
		fprintf(doclkhFP, "%5.5e\n", corpus->docs[d].lkh);
	}
	printf("lkh = %5.5e, Conv = %5.5e, wrd-lkh = %5.5e, Time = %d\n", lhood, conv, wrdlkh, (int)(t2-t1));
	fclose(doclkhFP);

	fprintf(lhood_fptr, "%d %5.5e %5.5e %5ld %5.5e\n",iteration, lhood, conv, (int)t2-t1, wrdlkh);
	fflush(lhood_fptr);
	fclose(lhood_fptr);

	sprintf(filename, "%s/final", dir);

	write_mcctm_model(model, var, filename, corpus, mu);

}



double doc_estep(document* doc, mcctm_model* model,
		mcctm_ss* ss, mcctm_var* var, int d, mcctm_epsopt* epsopt, int test, int comp_wrdlkh, double* phi){

	int n, variter, variter2, w, j, c, suminit;
	double normsum, cnt, temp, musum, maxval, wrdlkh;
	double varlkh, prev_varlkh, conv;
	double varlkh2, prev_varlkh2, conv2;

	prev_varlkh = -1e100;
	conv = 0.0;
	variter = 0;

	// init vars for this doc
	epsopt->nd = (double) doc->total;
	for (c = 0; c < model->c; c++){

		if ((doc->label != -1) && (doc->label != c))
			continue;

		var->mu[c] = 1.0/((double)model->c);
		var->eta[c] = 0.0;
		for (j = 0; j < model->m; j++){
			var->lambda[j][c] = 0.0;
			var->nu[j][c] = model->sigma2;
			var->nuhat[j][c] = log(model->sigma2);

			var->eta[c] += exp(model->alpha[j][c] + var->lambda[j][c] +
					0.5*var->nu[j][c]);

			var->sumphi[j][c] = ((double)doc->total)/((double)model->m);
			for (n = 0; n < doc->length; n++){
				var->phi[n][j][c] = 1.0/((double)model->m);
			}
		}
	}

	do{
		varlkh = 0.0;
		musum = 0;
		doc->lkh= 0.0;

		suminit = 0;

		for (c = 0; c < model->c; c++){

			if ((doc->label != -1) && (doc->label != c)){
				var->mu[c] = 0.0;
				continue;
			}

			if (var->mu[c] < 1e-15){
				var->mu[c] = -1;
				continue;
			}

			//compute theta
			//maxval = -1e50;
			for (j = 0; j < model->m; j++){
				var->theta[j] = model->alpha[j][c] + var->lambda[j][c];
				//if (var->theta[j] > maxval)
				//	maxval = var->theta[j];
			}
			//for (j = 0; j < model->m; j++){
			//	var->theta[j] = exp(var->theta[j] - maxval);
			//}//don't need to normalize this

			prev_varlkh2 = -1e100;
			conv2 = 0.0;
			variter2 = 0;

			do{
				varlkh2 = 0.0;

				for (n = 0; n < doc->length; n++){
					w = doc->words[n];
					cnt = (double) doc->counts[n];

					maxval = -1e100;
					normsum = 0.0;
					for (j = 0; j < model->m; j++){
						var->oldphi[j] = var->phi[n][j][c];
						phi[j] = var->theta[j] + model->Elogbeta[j][w];//var->theta[j]*model->expElogbeta[j][w];
						//normsum += var->phi[n][j][c];
						if (phi[j] > maxval) maxval = phi[j];
					}
					normsum = 0;
					for (j = 0; j < model->m; j++){
						phi[j] = exp(phi[j] - maxval);
						normsum += phi[j];
					}
					for (j = 0; j < model->m; j++){

						phi[j] /= normsum;
						var->sumphi[j][c] += cnt*(phi[j] - var->oldphi[j]);
						var->phi[n][j][c] = phi[j];

						if (phi[j] > 0){
							varlkh2 += cnt*phi[j]*(model->alpha[j][c] + model->Elogbeta[j][w]-log(phi[j]));
						}
					}
				}
				// update lambda and nu
				epsopt->etainv = 1.0/var->eta[c];
				for (j = 0; j < model->m; j++){
					epsopt->x[2*j] = var->lambda[j][c];
					epsopt->x[2*j +1 ] = var->nuhat[j][c];
					epsopt->alpha[j] = model->alpha[j][c];
					epsopt->sumphi[j] = var->sumphi[j][c];
				}
				my_optimize_eps(epsopt, model->m*2);
				
				var->eta[c] = 0.0;
				maxval = -1e50;
				for (j = 0; j < model->m; j++){
					var->lambda[j][c] = epsopt->x[2*j];
					var->nuhat[j][c] = epsopt->x[2*j+1];
					var->nu[j][c] = exp(var->nuhat[j][c]);
					var->eta[c] += exp(model->alpha[j][c] + var->lambda[j][c] +
							0.5*var->nu[j][c]);

					varlkh2 += -(pow(var->lambda[j][c],2.0) + var->nu[j][c])/(2*model->sigma2)
							+ var->lambda[j][c]*var->sumphi[j][c] + 0.5*log(var->nu[j][c]);

					var->theta[j] = model->alpha[j][c] + var->lambda[j][c];
					//if (var->theta[j] > maxval)
					//	maxval = var->theta[j];
				}
				varlkh2 -= epsopt->nd*log(var->eta[c]);
				//for (j = 0; j < model->m; j++){
				//	var->theta[j] = exp(var->theta[j] - maxval);
				//}//don't need to normalize this

				conv2 = fabs(prev_varlkh2 - varlkh2)/fabs(prev_varlkh2);
				/*if ((prev_varlkh2 > varlkh2) && (conv2 > 1e-7)){
					printf("Oops, likelihood of doc %d class %d is decreasing!\n", d, c);
					printf("ooops class %d, %lf %lf, %5.10e %d\n", c, varlkh2, prev_varlkh2, conv2, variter2);
				}*/
				prev_varlkh2 = varlkh2;
				variter2 ++;

			}while((variter2 < MAXITER) && (conv2 > CONVERGED));

			var->Qc[c] = varlkh2 + log(model->delta[c]);

			if (suminit != 0)	musum = log_sum(musum, var->Qc[c]);
			else{
				musum = var->Qc[c];
				suminit = 1;
			}

		}

		for (c = 0; c < model->c; c++){

			if (var->mu[c] == -1){
				var->mu[c] = 0;
				continue;
			}
			if ((doc->label != -1) && (doc->label != c)){
				var->mu[c] = 0.0;
				continue;
			}

			var->mu[c] = exp(var->Qc[c] - musum);

			if (var->mu[c] > 0){
				varlkh += var->mu[c]*(var->Qc[c] - log(var->mu[c]));
			}
		}

		conv = fabs(prev_varlkh - varlkh)/fabs(prev_varlkh);
		/*if ((prev_varlkh > varlkh) && (conv > 1e-7)){
			printf("Oops, likelihood of doc %d is decreasing!\n", d);
			printf("ooops doc %d, %lf %lf, %5.10e %d\n", d, varlkh, prev_varlkh, conv, variter);
		}*/
		prev_varlkh = varlkh;
		variter ++;

	}while((variter < MAXITER) && (conv > CONVERGED));


	if (test == 0){
		for (c = 0; c < model->c; c++){

			if ((doc->label != -1) && (doc->label != c)){
				var->mu[c] = 0.0;
				continue;
			}
			if (var->mu[c] == 0)
				continue;

			ss->delta[c] += var->mu[c];
			for (j = 0; j < model->m; j++){
				ss->alpha1[j][c] += var->mu[c]*var->sumphi[j][c];
				ss->alpha2[j][c] += epsopt->nd*var->mu[c]*exp(var->lambda[j][c]+
						0.5*var->nu[j][c])/var->eta[c];
				for (n = 0; n < doc->length; n++){
					w = doc->words[n];
					cnt = (double) doc->counts[n];
					temp = cnt*var->mu[c]*var->phi[n][j][c];
					ss->beta[j][w] += temp;
					ss->sumbeta[j] += temp;
					//subtract gamma parts
					varlkh -= temp*model->Elogbeta[j][w];
				}
			}
		}

	}

	if (comp_wrdlkh == 1){
		doc->lkh = 0.0;

		suminit = 0;
		for (c = 0; c < model->c; c++){
			wrdlkh = 0.0;

			if ((doc->label != -1) && (doc->label != c)){
			//if (doc->gt_label != c){
				continue;
			}
			//if (var->mu[c] == 0)
			//	continue;

			//compute topic proportions
			normsum = 0.0;
			for (j = 0; j < model->m; j++){
				var->theta[j] = model->alpha[j][c] + var->lambda[j][c] +
						0.5*var->nu[j][c];
				if (j > 0)	normsum = log_sum(normsum, var->theta[j]);
				else normsum = var->theta[j];
			}
			for (j = 0; j < model->m; j++){
				var->theta[j] = exp(var->theta[j] - normsum);
			}

			for (n = 0; n < doc->length; n++){
				temp = 0.0;
				for (j = 0; j < model->m; j++){
					temp += var->theta[j]*model->expElogbeta[j][doc->words[n]];
				}
				wrdlkh += (double)doc->counts[n]*log(temp);
			}
			//doc->lkh = wrdlkh;
			wrdlkh += log(model->delta[c]);

			if (suminit != 0)
				doc->lkh = log_sum(doc->lkh, wrdlkh);
			else{
				doc->lkh = wrdlkh;
				suminit = 1;
			}
		}
	}


	return(varlkh);

}


void mstep(mcctm_corpus* corpus, mcctm_model* model, mcctm_ss* ss, mcctm_var* var,
		mcctm_nuopt* nuopt, double* lhood){

	int j, c, n;
	double lkh, normsum, temp;
	lkh = 0.0;

	gsl_vector *x = gsl_vector_alloc(1);
	gsl_vector *x2 = gsl_vector_alloc(1);

	nuopt->nuss = 0.0;
	for (j = 0; j < model->m; j++){
		model->sumgamma[j] = model->nu*model->n + ss->sumbeta[j];
		temp = gsl_sf_psi(model->sumgamma[j]);
		for (n = 0; n < model->n; n++){
			model->gamma[j][n] = model->nu + ss->beta[j][n];
			model->Elogbeta[j][n] = gsl_sf_psi(model->gamma[j][n]) - temp;
			model->expElogbeta[j][n] = exp(model->Elogbeta[j][n]);

			lkh += (ss->beta[j][n]-model->gamma[j][n])*model->Elogbeta[j][n];
			nuopt->nuss += model->Elogbeta[j][n];

			ss->beta[j][n] = 0.0;
			lkh += lgamma(model->gamma[j][n]);
		}

		lkh -= lgamma(model->sumgamma[j]);
		ss->sumbeta[j] = 0.0;
	}


	normsum = 0.0;
	for (c = 0; c < model->c; c++){
		model->delta[c] = ss->delta[c];
		normsum += model->delta[c];

		for (j = 0; j < model->m; j++){
			if (ss->alpha1[j][c] > 0)
				model->alpha[j][c] = log((ss->alpha1[j][c]+EPS)/(ss->alpha2[j][c])+EPS) + EPS;
			else
				model->alpha[j][c] = log(EPS);
			if (isnan(model->alpha[j][c]))
			printf("%lf %lf %lf\n", ss->alpha1[j][c],ss->alpha2[j][c], model->alpha[j][c]);
			ss->alpha1[j][c] = 0.0;
			ss->alpha2[j][c] = 0.0;
		}

	}
	for (c = 0; c < model->c; c++){
		model->delta[c] /= normsum;
		if (model->delta[c] == 0)	model->delta[c] = EPS;

		//lkh += model->theta[c]*ss->theta[c];
		ss->delta[c] = 0.0;
	}

	//optimize nu
	gsl_vector_set(x, 0, model->nuhat);
	gsl_vector_set(x2, 0, model->nuhat);
	optimize_nu(x, nuopt, 1, x2);
	model->nuhat = gsl_vector_get(x2,0);
	model->nu = exp(model->nuhat);
	if (model->nu < 1e-10){
		model->nu = 1e-10;
		model->nuhat = log(model->nu);
	}

	//lkh terms related to nu
	lkh += model->m*(lgamma(model->nu*model->n) - model->n*lgamma(model->nu));
	lkh += model->nu*nuopt->nuss;// +  (log(model->nu) -model->nu);

	*lhood += lkh;

	gsl_vector_free(x);
	gsl_vector_free(x2);
}


void test(char* dataset, char* model_name, char* dir)
{

	FILE* lhood_fptr;
	FILE* fp;
	char string[100];
	char filename[100];
	char lblfile[100];
	int iteration;
	int d, doclkh, nmax, c;
	double lhood, wrdlkh;
	double** mu;

	mcctm_corpus* corpus;
	mcctm_model *model = NULL;
	mcctm_epsopt *epsopt = NULL;
	mcctm_ss* ss = NULL;
	mcctm_var* var = NULL;
	time_t t1,t2;

	corpus = read_data(dataset, 0, lblfile);
	nmax = max_corpus_length(corpus);

	mkdir(dir, S_IRUSR|S_IWUSR|S_IXUSR);

	// set up the log likelihood log file
	sprintf(string, "%s/test-lhood.dat", dir);
	lhood_fptr = fopen(string, "w");

	model = load_model(model_name);
	model->D = corpus->ndocs;

	ss = new_mcctm_ss(model);
	var =  new_mcctm_var(model, nmax);

	mu = malloc(sizeof(double*)*corpus->ndocs);
	for (d = 0; d < corpus->ndocs; d++){
		mu[d] = malloc(sizeof(double)*model->c);
		for (c = 0; c < model->c; c++){
			mu[d][c] = 0.0;
		}
	}

	epsopt =  new_mcctm_epsopt(model);

	iteration = 0;

	lhood = 0.0;
	wrdlkh = 0.0;

	time(&t1);
	double* phi = malloc(sizeof(double)*model->m);
	for (d = 0; d < corpus->ndocs; d++){

		doclkh = doc_estep(&(corpus->docs[d]), model, ss, var, d, epsopt, 1, 1, phi);

		for (c = 0; c < model->c; c++){
			mu[d][c] = var->mu[c];
		}
		lhood += doclkh;
		wrdlkh += corpus->docs[d].lkh;

	}


	time(&t2);

	fprintf(lhood_fptr, "%d %5.5e %5.5e %5ld %5.5e \n",iteration, lhood, 0.0, (int)t2-t1, wrdlkh);
	fflush(lhood_fptr);
	fclose(lhood_fptr);
	//*************************************

	sprintf(filename, "%s/testfinal.mu", dir);
	//write_mcctm_model(model, var, filename, corpus, mu);
	//mu
	fp = fopen(filename, "w");
	for (d = 0; d < corpus->ndocs; d++){
		for (c = 0; c < model->c; c++){
			fprintf(fp, "%5.10lf ", mu[d][c]);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);

}

mcctm_model* new_mcctm_model(int ntopics, int nclasses, int nterms, double sigma2, double nu)
{
	int n, j, c;

	mcctm_model* model = malloc(sizeof(mcctm_model));
	model->c = nclasses;
	model->m = ntopics;
	model->D = 0;
	model->n = nterms;
	model->sigma2 = sigma2;
	model->nu = nu;
	model->nuhat = log(model->nu);
	model->Elogbeta = malloc(sizeof(double*)*ntopics);
	model->expElogbeta = malloc(sizeof(double*)*ntopics);
	model->gamma = malloc(sizeof(double*)*ntopics);
	model->sumgamma = malloc(sizeof(double)*ntopics);
	model->alpha = malloc(sizeof(double*)*ntopics);
	for (j = 0; j < ntopics; j++){
		model->sumgamma[j] = 0.0;
		model->Elogbeta[j] = malloc(sizeof(double)*nterms);
		model->expElogbeta[j] = malloc(sizeof(double)*nterms);
		model->gamma[j] = malloc(sizeof(double)*nterms);
		for (n = 0; n < nterms; n++){
			model->Elogbeta[j][n] = 0.0;
			model->expElogbeta[j][n] = 0.0;
			model->gamma[j][n] = 0.0;
		}
		model->alpha[j] = malloc(sizeof(double)*nclasses);
		for (c = 0; c < nclasses; c++){
			model->alpha[j][c] = 0.0;
		}
	}
	model->delta = malloc(sizeof(double)*nclasses);
	for (c = 0; c < nclasses; c++){
		model->delta[c] = 0.0;
	}

	return(model);
}

mcctm_var * new_mcctm_var(mcctm_model* model, int nmax){

	int n, j, c;

	mcctm_var * var;
	var = malloc(sizeof(mcctm_var));

	var->sumphi = malloc(sizeof(double*)*model->m);
	var->oldphi = malloc(sizeof(double)*model->m);
	var->nu = malloc(sizeof(double*)*model->m);
	var->nuhat = malloc(sizeof(double*)*model->m);
	var->lambda = malloc(sizeof(double*)*model->m);
	var->theta = malloc(sizeof(double)*model->m);
	for (j = 0; j < model->m; j++){
		var->theta[j] = 0.0;
		var->sumphi[j] = malloc(sizeof(double)*model->c);
		var->lambda[j] = malloc(sizeof(double)*model->c);
		var->nu[j] = malloc(sizeof(double)*model->c);
		var->nuhat[j] = malloc(sizeof(double)*model->c);
		for (c = 0; c < model->c; c++){
			var->sumphi[j][c] = 0.0;
			var->lambda[j][c] = 0.0;
			var->nuhat[j][c] = 0.0;
			var->nu[j][c] = 0.0;
			/*var->nuhat[j][c] = malloc(sizeof(double)*model->D);
			var->nu[j][c] = malloc(sizeof(double)*model->D);
			var->lambda[j][c] = malloc(sizeof(double)*model->D);
			for (d = 0; d < model->D; d++){
				var->nuhat[j][c][d] = 0.0;
				var->nu[j][c][d] = 0.0;
				var->lambda[j][c][d] = 0.0;
			}*/
		}
		var->oldphi[j] = 0.0;
	}

	var->Qc = malloc(sizeof(double)*model->c);
	var->mu = malloc(sizeof(double)*model->c);
	var->eta = malloc(sizeof(double)*model->c);
	for (c = 0; c < model->c; c++){
		var->mu[c] = 0.0;
		var->Qc[c] = 0.0;
		var->eta[c] = 0.0;
	}

	var->phi = malloc(sizeof(double**)*nmax);
	for (n = 0; n < nmax; n++){
		var->phi[n] = malloc(sizeof(double*)*model->m);
		for (j = 0; j < model->m; j++){
			var->phi[n][j] = malloc(sizeof(double)*model->c);
			for (c = 0; c < model->c; c++){
				var->phi[n][j][c] = 0.0;
			}
		}
	}

	return(var);
}


mcctm_ss * new_mcctm_ss(mcctm_model* model)
{
	int c, j, n;
	mcctm_ss * ss;
	ss = malloc(sizeof(mcctm_ss));

	ss->delta = malloc(sizeof(double)*model->c);
	for (c = 0; c < model->c; c++){
		ss->delta[c] = 0.0;
	}
	ss->alpha1 = malloc(sizeof(double*)*model->m);
	ss->alpha2 = malloc(sizeof(double*)*model->m);
	ss->beta = malloc(sizeof(double*)*model->m);
	ss->sumbeta = malloc(sizeof(double)*model->m);
	for (j = 0; j < model->m; j++){
		ss->alpha1[j] = malloc(sizeof(double)*model->c);
		ss->alpha2[j] = malloc(sizeof(double)*model->c);
		for (c = 0; c < model->c; c++){
			ss->alpha1[j][c] = 0.0;
			ss->alpha2[j][c] = 0.0;
		}
		ss->sumbeta[j] = 0.0;
		ss->beta[j] = malloc(sizeof(double)*model->n);
		for (n = 0; n < model->n; n++){
			ss->beta[j][n] = 0.0;
		}
	}
	return(ss);
}



mcctm_corpus* read_data(const char* data_filename, int lblchck, const char* lbl_filename)
{
	FILE *fileptr;
	int length, count, word, n, nd, nw, lbl, gt_lbl;
	mcctm_corpus* c;

	printf("reading data from %s\n", data_filename);
	c = malloc(sizeof(mcctm_corpus));
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
			fscanf(fileptr, "%d, %d", &lbl,&gt_lbl);
			c->docs[nd].label = lbl;
			c->docs[nd].gt_label = gt_lbl;
		}
	}

	return(c);
}

int max_corpus_length(mcctm_corpus* c)
{
	int n, max = 0;
	for (n = 0; n < c->ndocs; n++)
	if (c->docs[n].length > max) max = c->docs[n].length;
	return(max);
}


void write_mcctm_model(mcctm_model * model, mcctm_var* var, char * root,mcctm_corpus * corpus, double** mu)
{
	char filename[200];
	FILE* fileptr;
	int n, j, d, c;

	//beta
	sprintf(filename, "%s.beta", root);
	fileptr = fopen(filename, "w");
	for (n = 0; n < model->n; n++){
		for (j = 0; j < model->m; j++){
			fprintf(fileptr, "%.10lf ",model->gamma[j][n]);
		}
		fprintf(fileptr, "\n");
	}
	fclose(fileptr);

	//mu
	sprintf(filename, "%s.mu", root);
	fileptr = fopen(filename, "w");
	for (d = 0; d < corpus->ndocs; d++){
		for (c = 0; c < model->c; c++){
			fprintf(fileptr, "%5.10lf ", mu[d][c]);
			//fprintf(fileptr, "%5.10e ", mu[d][c]);
		}
		fprintf(fileptr, "\n");
	}
	fclose(fileptr);


	//a,b
	sprintf(filename, "%s.alpha", root);
	fileptr = fopen(filename, "w");
	for (j = 0; j < model->m; j++){
		for (c = 0; c < model->c; c++){
			fprintf(fileptr, "%.10lf ",model->alpha[j][c]);
		}
		fprintf(fileptr, "\n");
	}
	fclose(fileptr);

	sprintf(filename, "%s.delta", root);
	fileptr = fopen(filename, "w");
	for (c = 0; c < model->c; c++){
		fprintf(fileptr, "%.10lf ", log(model->delta[c]));
	}
	fclose(fileptr);

	sprintf(filename, "%s.other", root);
	fileptr = fopen(filename, "w");
	fprintf(fileptr,"M %d \n",model->m);
	fprintf(fileptr,"C %d \n",model->c);
	fprintf(fileptr,"sigma2 %lf \n",model->sigma2);
	fprintf(fileptr,"nu %lf \n",model->nu);
	fprintf(fileptr,"num_terms %d \n",model->n);
	fprintf(fileptr,"num_docs %d \n",corpus->ndocs);
	fclose(fileptr);

}

mcctm_model* load_model(char* model_root){

	char filename[100];
	FILE* fileptr;
	int j, n, c, num_terms, num_docs, ntopics, nclasses;
	//float x;
	double y, sigma2, nu;

	mcctm_model* model;
	sprintf(filename, "%s.other", model_root);
	printf("loading %s\n", filename);
	fileptr = fopen(filename, "r");
	fscanf(fileptr, "M %d\n", &ntopics);
	fscanf(fileptr, "C %d\n", &nclasses);
	fscanf(fileptr, "sigma2 %lf\n", &sigma2);
	fscanf(fileptr, "nu %lf\n", &nu);
	fscanf(fileptr, "num_terms %d\n", &num_terms);
	fscanf(fileptr, "num_docs %d\n", &num_docs);
	fclose(fileptr);

	model  = new_mcctm_model(ntopics, nclasses, num_terms, sigma2, nu);

	sprintf(filename, "%s.beta", model_root);
	printf("loading %s\n", filename);
	fileptr = fopen(filename, "r");
	for (j = 0; j < ntopics; j++){
		model->sumgamma[j] = 0.0;
	}
	for (n = 0; n < num_terms; n++){
		for (j = 0; j < ntopics; j++){
			fscanf(fileptr, " %lf", &y);
			model->gamma[j][n] = y;
			model->sumgamma[j] += y;
		}
	}
	fclose(fileptr);
	for (j = 0; j < ntopics; j++){
		y = gsl_sf_psi(model->sumgamma[j]);
		for (n = 0; n < num_terms; n++){
			model->Elogbeta[j][n] = gsl_sf_psi(model->gamma[j][n]) - y;
			model->expElogbeta[j][n] = exp(model->Elogbeta[j][n]);
		}
	}

	sprintf(filename, "%s.alpha", model_root);
	printf("loading %s\n", filename);
	fileptr = fopen(filename, "r");
	for (j = 0; j < model->m; j++){
		for (c = 0; c < model->c; c++){
			fscanf(fileptr, "%lf ", &y);
			model->alpha[j][c] = y;
		}
	}
	fclose(fileptr);

	sprintf(filename, "%s.delta", model_root);
	printf("loading %s\n", filename);
	fileptr = fopen(filename, "r");
	for (c = 0; c < model->c; c++){
		fscanf(fileptr, "%lf ", &y);
		model->delta[c] = exp(y);
	}
	fclose(fileptr);


	return(model);
}


void corpus_initialize_model(mcctm_model* model, mcctm_corpus* corpus, mcctm_ss* ss, mcctm_var* var)
{

	/*int n, j, d, i, count, c, argmaxlkh;
	double temp, normsum, maxlkh;

	int* sdocs = malloc(sizeof(int)*corpus->ndocs);
	for (d = 0; d < corpus->ndocs; d++){
		sdocs[d] = -1;
	}

	if (NUMINIT*model->m >= corpus->ndocs){
		NUMINIT = (int)(corpus->ndocs/model->m/2.0);
		if (NUMINIT < 3)	NUMINIT = 3;
	}

	//init topics
	for (j = 0; j < model->m; j++){
		normsum = 0.0;
		for (n = 0; n < model->n; n++){
			model->beta[j][n] = 1e-5;
			normsum += model->beta[j][n];
		}
		for (i = 0; i < NUMINIT; i++){
			//choose a doc from this tim and init
			count = 0;
			while (1){
				d = floor(gsl_rng_uniform(r) * corpus->ndocs);
				if(sdocs[d] != -1){
					count ++;
					continue;
				}
				else{
					sdocs[d] = j;
					break;
				}
			}

			for (n = 0; n < corpus->docs[d].length; n++){
				model->beta[j][corpus->docs[d].words[n]] += (double) corpus->docs[d].counts[n];
				normsum += (double) corpus->docs[d].counts[n];
			}
		}

		for (n = 0; n < model->n; n++){
			model->beta[j][n] /= normsum;
			model->logbeta[j][n] = log(model->beta[j][n]);
		}
	}


	for (c = 0; c < model->c; c++){
		model->delta[c] = 0.0;
		for (j = 0; j < model->m; j++){
			model->alpha[j][c] = 0.1;
		}
	}
	normsum = 0.0;
	for (d = 0; d < corpus->ndocs; d++){
		c = corpus->docs[d].label;
		if (c == -1)
			continue;

		maxlkh = -1e50;
		argmaxlkh = 0;
		for (j = 0; j < model->m; j++){
			temp = 0.0;
			for (n = 0; n < corpus->docs[d].length; n++){
				temp += (double)corpus->docs[d].counts[n]*model->logbeta[j][corpus->docs[d].words[n]];
			}
			if (temp > maxlkh){
				argmaxlkh = j;
				maxlkh = temp;
			}
		}
		model->alpha[argmaxlkh][c] += 1.0;
		model->delta[c] += 1.0;
		normsum += 1.0;
	}
	for (c = 0; c < model->c; c++){
		for (j = 0; j < model->m; j++){
			model->alpha[j][c] *= 10.0/model->delta[c];
		}
		model->delta[c] /= normsum;
	}


  	free(sdocs);*/

}



void random_initialize_model(mcctm_model * model, mcctm_corpus* corpus, mcctm_ss* ss, mcctm_var* var){

	int n, j, c;
	double temp;
	double exp_par = (double)corpus->ndocs*100.0/((double)model->m*model->n);

	for (j = 0; j < model->m; j++){


		model->sumgamma[j] = 0.0;
		for (n = 0; n < model->n; n++){
			model->gamma[j][n] = model->nu + gsl_ran_exponential(r, exp_par);
			model->sumgamma[j] += model->gamma[j][n];
		}
		temp = gsl_sf_psi(model->sumgamma[j]);
		for (n = 0; n < model->n; n++){
			model->Elogbeta[j][n] = gsl_sf_psi(model->gamma[j][n]) - temp;
			model->expElogbeta[j][n] = exp(model->Elogbeta[j][n]);
		}
	}
	temp = 0.0;
	for (c = 0; c < model->c; c++){

		//model->theta[c] = gsl_rng_uniform(r);
		model->delta[c] = 1.0;
		temp += model->delta[c];

		//model->sumeta[c] = 0.0;
		/*for (j = 0; j < model->m; j++){
			model->etahat[j][c] = gsl_rng_uniform(r)*10.0;
			//model->sumeta[c] += model->eta[j][c];
		}*/
	}
	for (c = 0; c < model->c; c++){
		model->delta[c] /= temp;
		for (j = 0; j < model->m; j++){
			model->alpha[j][c] = gsl_rng_uniform(r);
		}
	}

}

mcctm_epsopt* new_mcctm_epsopt(mcctm_model* model)
{
	mcctm_epsopt * epsopt;
	int j;

    epsopt = malloc(sizeof(mcctm_epsopt));
    epsopt->m = model->m;
    epsopt->sigma2 = model->sigma2;
    epsopt->etainv = 0.0;
    epsopt->nd = 0.0;
    epsopt->nu = malloc(sizeof(double)*model->m);
    epsopt->sumphi = malloc(sizeof(double)*model->m);
    epsopt->alpha = malloc(sizeof(double)*model->m);
    for (j = 0; j < model->m; j++){
    	epsopt->alpha[j] = 0.0;
        epsopt->sumphi[j] = 0.0;
        epsopt->alpha[j] = 0.0;
    }
    epsopt->x = malloc(sizeof(double)*2*model->m);
    epsopt->xnew = malloc(sizeof(double)*2*model->m);
    epsopt->grad = malloc(sizeof(double)*2*model->m);
    epsopt->dx = malloc(sizeof(double)*2*model->m);
    epsopt->stpchk = 0.0;
    for (j = 0; j < 2*model->m; j++){
    	epsopt->x[j] = 0.0;
    	epsopt->xnew[j] = 0.0;
    	epsopt->grad[j] = 0.0;
    	epsopt->dx[j] = 0.0;
    }

    return(epsopt);
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

