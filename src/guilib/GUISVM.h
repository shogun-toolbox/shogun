#ifndef _GUISVM_H__
#define _GUISVM_H__ 

#define SVMMPI

#include "svm/SVM.h"
#include "svm/SVM_light.h"
#include "svm_cplex/SVM_cplex.h"
#include "svm_mpi/mpi_svm.h"

class CGUI ;

class CGUISVM
{

public:
	CGUISVM(CGUI*);
	~CGUISVM();

	bool new_svm(char* param);
	bool train(char* param);
	bool test(char* param);
	bool set_kernel();
	bool get_kernel();
	bool set_preproc();
	bool get_preproc();
	bool load_svm();
	bool save_svm();
	bool set_C(char* param);
 protected:
	CGUI* gui ;
	double C ;
	CSVM* svm ;
};
#endif

#if 0
	if (lambda)
	{
	    if (neg)
		delete neg;

	    neg=lambda;
	    neg->set_observations(NULL);
	    delete lambda_train;
	    
	    lambda=NULL;
	    lambda_train=NULL;
	else
	   CIO::message("create model first\n");
    } 

      if (lambda)
	{
	  if (test)
	    delete test;
	  
	  test=lambda;
	  test->set_observations(NULL);
	  delete lambda_train;
	  
	    lambda=NULL;
	    lambda_train=NULL;
	    
#warning	    CHMM::invalidate_top_feature_cache(CHMM::INVALID);
	}
      else
	CIO::message("create model first\n");
	if (lambda)
	  {
	    if (pos)
	      delete pos;
	    
	    pos=lambda;
	    pos->set_observations(NULL);
	    delete lambda_train;
	    
	    lambda=NULL;
	    lambda_train=NULL;
	  }
	else
	  CIO::message("create model first\n");
	for (i=strlen(N_SAVE_MODEL_BIN); isspace(input[i]); i++);

	if (lambda)
	{
	    FILE* file=fopen(&input[i], "w");

	    if ((!file) ||	(!lambda->save_model_bin(file)))
		printf("writing to file %s failed!\n", &input[i]);
	    else
		printf("successfully written model into \"%s\" !\n", &input[i]);
	    if (file)
		fclose(file);
	}
	else
	   CIO::message("create model first\n");
	double value;
	for (i=strlen(N_CHOP); isspace(input[i]); i++);

	if (sscanf(&input[i], "%le", &value) == 1)
	{
	    if ( (lambda) && (lambda_train) )
	    {
		lambda->chop(value);
		lambda_train->chop(value);
	    }
	}
	else
	   CIO::message("see help for parameters/create model first\n");
	for (i=strlen(N_LOAD_DEFINITIONS); isspace(input[i]); i++);

	if ((!lambda) || (!lambda_train)) 
	   CIO::message("load or create model first\n");
	else
	{
	    char file_name[1024]="" ;
	    int initialize ;
	    int num_parm=sscanf(&input[i],"%s %d", file_name, &initialize) ;
	    if (num_parm<2)
		initialize=1 ;

	    FILE* def_file=fopen(file_name, "r");

	    if (def_file)
	    {
		bool ok=lambda->load_definitions(def_file,true,(initialize!=0));

		rewind(def_file);
		ok=ok && lambda_train->load_definitions(def_file,false,(initialize!=0)) ;

		if (ok)
		   CIO::message("file successfully read\n");

		fclose(def_file);
	    }
	    else
		printf("opening file %s failed\n", file_name);
	}
	for (i=strlen(N_SAVE_PATH); isspace(input[i]); i++);

	FILE* file=fopen(&input[i], "w");

	if (file)
	{
	    lambda->save_path(file);
	    fclose(file);
	}
	else
	   CIO::message("opening file %s for writing failed\n", &input[i]);
    } 
	for (i=strlen(N_SAVE_LIKELIHOOD_BIN); isspace(input[i]); i++);

	FILE* file=fopen(&input[i], "w");

	if (file)
	{
	    lambda->save_likelihood_bin(file);
	    fclose(file);
	}
	else
	   CIO::message("opening file %s for writing failed\n", &input[i]);
	for (i=strlen(N_SAVE_LIKELIHOOD); isspace(input[i]); i++);

	FILE* file=fopen(&input[i], "w");

	if (file)
	{
	    lambda->save_likelihood(file);
	    fclose(file);
	}
	else
	   CIO::message("opening file %s for writing failed\n", &input[i]);
		int j;
		for (i=strlen(N_SAVE_TOP_FEATURES); isspace(input[i]); i++);
		for (j=i; j<(int)strlen(input) && !isspace(input[j]); j++);
		input[j]='\0';
		FILE* file=fopen(&input[i], "w");
		if (file)
		{
			if (pos && neg)
			{
				if (obs_postrain && obs_negtrain)
				{
					CObservation* obs=new CObservation(obs_postrain, obs_negtrain);

					CObservation* old_pos=pos->get_observations();
					CObservation* old_neg=neg->get_observations();

					pos->set_observations(obs);
					neg->set_observations(obs);

					unsigned int endian=0x12345678;
					unsigned char* e=(unsigned char*) &endian;
									 
					CIO::message("we got %i pos samples and %i neg samples,\nwriting order is pos first\n endian check: 0x12345678 is seen as 0x%x%x%x%x -> its %s\n", obs_postrain->get_DIMENSION(), obs_negtrain->get_DIMENSION(), e[0],e[1],e[2],e[3], ( e[0]==0x12 && e[1]==0x34 && e[2]==0x56 && e[3]==0x78) ? "BIG ENDIAN" : "LITTLE ENDIAN"); 
					fprintf(stderr,"Sorry not implemented\n") ;
#warning					CHMM::save_top_features(pos,neg,file);
					fclose(file);
					CIO::message("successfully written top_features into \"%s\" !\n", &input[i]);

					pos->set_observations(old_pos);
					neg->set_observations(old_neg);

					delete obs;
				}
				else
					CIO::message("assign postrain and negtrain observations first!\n");
			}
			else
				CIO::message("assign positive and negative models first!\n");
		}
		else
			CIO::message("opening file %s for writing failed\n", &input[i]);
		char filename[1024];
		char kern[1024];
		int kern_type=6;
		for (i=strlen(N_SAVE_KERNEL); isspace(input[i]); i++);

		if ((sscanf(&input[i], "%s %s", filename, kern))==2)
		{
			FILE* file=fopen(filename, "w");

			if (kern[0]=='F')
				kern_type=7;

			if (file)
			{
				if (pos && neg)
				{
					if (obs_postrain && obs_negtrain)
					{
						CObservation* obs=new CObservation(obs_postrain, obs_negtrain);

						CObservation* old_pos=pos->get_observations();
						CObservation* old_neg=neg->get_observations();

						pos->set_observations(obs);
						neg->set_observations(obs);

						fprintf(stderr,"Sorry not implemented") ;
#warning						save_kernel(file, obs, kern_type);
						fclose(file);
						CIO::message("successfully written top_kernel into \"%s\" !\n", filename);


						pos->set_observations(old_pos);
						neg->set_observations(old_neg);

						delete obs;
					}
					else
						CIO::message("assign postrain and negtrain observations first!\n");
				}
				else
					CIO::message("assign positive and negative models first!\n");

			}
			else
				CIO::message("opening file %s for writing failed\n", filename);
		}
		else
			CIO::message("see help for parameters\n");
	int j;

	for (i=strlen(N_SAVE_PATH_DERIVATIVES_BIN); isspace(input[i]); i++);
	for (j=i; j<(int)strlen(input) && !isspace(input[j]); j++);
	input[j]='\0';
	FILE* file=fopen(&input[i], "w");
	if (file)
	{
	    lambda->save_path_derivatives_bin(file);
	    fclose(file);
	   CIO::message("successfully written vit_derivatives into \"%s\" !\n", &input[i]);
	}
	else
	   CIO::message("opening file %s for writing failed\n", &input[i]);
	int j;
	for (i=strlen(N_SAVE_PATH_DERIVATIVES); isspace(input[i]); i++);
	for (j=i; j<(int)strlen(input) && !isspace(input[j]); j++);
	input[j]='\0';
	FILE* file=fopen(&input[i], "w");
	if (file)
	{
	    lambda->save_path_derivatives(file);
	    fclose(file);
	   CIO::message("successfully written vit_derivatives into \"%s\" !\n", &input[i]);
	} 
	else
	   CIO::message("opening file %s for writing failed\n", &input[i]);
	int j;
	for (i=strlen(N_SAVE_MODEL_DERIVATIVES_BIN); isspace(input[i]); i++);
	for (j=i; j<(int)strlen(input) && !isspace(input[j]); j++);
	input[j]='\0';
	FILE* file=fopen(&input[i], "w");

	if (file)
	{
	    lambda->save_model_derivatives_bin(file);
	    fclose(file);
	   CIO::message("successfully written bw_derivatives into \"%s\" !\n", &input[i]);
	} 
	else
	   CIO::message("opening file %s for writing failed\n", &input[i]);
	int j;
	for (i=strlen(N_SAVE_MODEL_DERIVATIVES); isspace(input[i]); i++);
	for (j=i; j<(int)strlen(input) && !isspace(input[j]); j++);
	input[j]='\0';
	FILE* file=fopen(&input[i], "w");
	if (file)
	{
	    lambda->save_model_derivatives(file);
	    fclose(file);
	   CIO::message("successfully written bw_derivatives into \"%s\" !\n", &input[i]);
	}
	else
	   CIO::message("opening file %s for writing failed\n", &input[i]);
	for (i=strlen(N_FIX_POS_STATE); isspace(input[i]); i++);

	int pos,state,value;
	if (sscanf(&input[i], "%d %d %d", &pos, &state, &value) == 3)
	{
	    if ((lambda) && (lambda_train))
	    {
		bool ok=lambda->set_fix_pos_state(pos,state,value) ;
		ok= ok && lambda_train->set_fix_pos_state(pos,state,value) ;
		if (!ok)
		   CIO::message("%s failed\n",N_FIX_POS_STATE);
	    }
	    else
		printf("create model first!\n");
	}
	else
	   CIO::message("see help for parameters\n");
	for (i=strlen(N_SET_MAX_DIM); isspace(input[i]); i++);
	char target[1024];
	int dim;

	sscanf(&input[i], "%d %s", &dim, target);
	CObservation* obs=NULL;

	if (strcmp(target,"POSTRAIN")==0)
	{
		obs=obs_postrain;
	}
	else if (strcmp(target,"NEGTRAIN")==0)
	{
		obs=obs_negtrain;
	}
	else if (strcmp(target,"POSTEST")==0)
	{
		obs=obs_postest;
	}
	else if (strcmp(target,"NEGTEST")==0)
	{
		obs=obs_negtest;
	}
	else if (strcmp(target,"TEST")==0)
	{
		obs=obs_test;
	}
	else
		printf("target POSTRAIN|NEGTRAIN|POSTEST|NEGTEST|TEST missing\n");

	if (sscanf(&input[i], "%d", &dim) == 1)
	{
	    if (obs)
	    {
			obs->set_dimension(dim) ;
	    }
	    else
			printf("load observation first!\n");
	}
	else
	   CIO::message("see help for parameters\n");
	delete lambda;
	delete lambda_train;
	delete pos;
	delete neg;
	delete obs_postrain;
	delete obs_negtrain;
	delete obs_postest;
	delete obs_negtest;
	delete obs_test;
	lambda=NULL;
	lambda_train=NULL;
	pos=NULL;
	neg=NULL;
	obs_postrain=NULL;
	obs_negtrain=NULL;
	obs_postest=NULL;
	obs_negtest=NULL;
	obs_test=NULL;

	printf("cleared.\n");

	for (i=strlen(N_NEW); isspace(input[i]); i++);

	double pseudo;
	for (i=strlen(N_PSEUDO); isspace(input[i]); i++);
	if (sscanf(&input[i], "%le", &pseudo) == 1)
	{
	    PSEUDO=pseudo ;
	    if ((lambda!=NULL) & (lambda_train!=NULL))
	    {
		lambda->set_pseudo(PSEUDO) ;
		lambda_train->set_pseudo(PSEUDO) ;
	    }
	}
	else
	    if ((lambda!=NULL) & (lambda_train!=NULL))
		printf("see help for parameters. current setting: pseudo=%e (%e,%e)\n",
			(double) PSEUDO, (double) lambda->get_pseudo(), (double) lambda_train->get_pseudo());
	    else
		printf("see help for parameters. current setting: pseudo=%e\n", PSEUDO);

	for (i=strlen(N_ALPHABET); isspace(input[i]); i++);
	alphabet= DNA;

	char obs_type[1024];
	char target[1024];

	sscanf(&input[i], "%s %s", target, obs_type);
	CObservation* obs=NULL;

	if (strcmp(target,"POSTRAIN")==0)
	{
		obs=obs_postrain;
	}
	else if (strcmp(target,"NEGTRAIN")==0)
	{
		obs=obs_negtrain;
	}
	else if (strcmp(target,"POSTEST")==0)
	{
		obs=obs_postest;
	}
	else if (strcmp(target,"NEGTEST")==0)
	{
		obs=obs_negtest;
	}
	else if (strcmp(target,"TEST")==0)
	{
		obs=obs_test;
	}
	else
		printf("target POSTRAIN|NEGTRAIN|POSTEST|NEGTEST|TEST missing\n");

	if (obs_type[0]=='P' || obs_type[0]=='D' || obs_type[0]=='A' || obs_type[0]=='C')
	{
	    if (obs_type[0]=='P')
		alphabet=PROTEIN;
	    else if (obs_type[0]=='A')
		alphabet=ALPHANUM;
	    else if (obs_type[0]=='D')
		alphabet=DNA;
	    else if (obs_type[0]=='C')
		alphabet=CUBE;
	}
	else
	{
	    if (obs)
		printf("see help for parameters. current setting: alphabet=%s\n",
			(obs->get_alphabet()==DNA) ?  "DNA":  (obs->get_alphabet()==PROTEIN) ? "PROTEIN" : (obs->get_alphabet()==CUBE) ? "CUBE":"ALPHANUM" );		    
	}
	int j=100;
	double f=0.001;

	for (i=strlen(N_CONVERGENCE_CRITERIA); isspace(input[i]); i++);

	if (sscanf(&input[i], "%d %le", &j, &f) == 2)
	{
	    ITERATIONS=j;
	    EPSILON=f;
	}
	else
	   CIO::message("see help for parameters. current setting: iterations=%i, epsilon=%e\n",ITERATIONS,EPSILON);
	double start_pseudo=1000 ;
	double act_pseudo ;
	double step ;
	int in_steps=ITERATIONS ;

	for (i=strlen(N_VITERBI_TRAIN_DEFINED_ANNEALED); isspace(input[i]); i++);
	int numpar=sscanf(&input[i], "%le %d", &act_pseudo, &in_steps) ;
	if (numpar<=0)
	    act_pseudo=start_pseudo ;
	if (numpar<2)
	    in_steps=ITERATIONS ;
	step=exp(log(PSEUDO/act_pseudo)/in_steps) ;

	printf("\nAnnealed optimization: pseudo_start=%e, pseudo_end=%e, step=%e, in_steps=%i\n",act_pseudo,PSEUDO,step,in_steps) ;

	if ((lambda) && (lambda_train)) 
	{
	    PSEUDO=lambda->get_pseudo() ;
	    lambda->set_pseudo(act_pseudo) ;
	    lambda_train->set_pseudo(act_pseudo) ;
	    iteration_count=ITERATIONS ;
	    while (!converge(lambda->best_path(-1), lambda_train->best_path(-1)))
	    {
		switch_model(&lambda, &lambda_train);
		lambda->estimate_model_viterbi_defined(lambda_train);
		act_pseudo*=step ;
		lambda->set_pseudo(act_pseudo) ;
		lambda_train->set_pseudo(act_pseudo) ;
		printf("   pseudo=%e",act_pseudo) ; 
	    }
	}
	else
	   CIO::message("create model first\n");
	double act_pseudo ;
	double step ;
	double eps_add ;

	for (i=strlen(N_VITERBI_TRAIN_DEFINED_ADDIABATIC); isspace(input[i]); i++);
	int numpar=sscanf(&input[i], "%le %le %le", &act_pseudo, &step, &eps_add) ;
	if (numpar<1)
	    act_pseudo=1000 ;
	if (numpar<2)
	    step=exp(log(PSEUDO/act_pseudo)/(ITERATIONS/10)) ;
	if (numpar<3)
	    eps_add=100*EPSILON ;

	printf("\nAddiabatic annealed optimization: pseudo_start=%e, step=%e, eps_add=%e\n",act_pseudo,step,eps_add) ;

	if ((lambda) && (lambda_train)) 
	{
	    double prob, prob_train ;
	    PSEUDO=lambda->get_pseudo() ;
	    lambda->set_pseudo(act_pseudo) ;
	    lambda_train->set_pseudo(act_pseudo) ;
	    iteration_count=ITERATIONS ;
	   CIO::message("pseudo=%e  \n",act_pseudo) ; 
	    prob=lambda->best_path(-1) ;
	    prob_train=lambda_train->best_path(-1) ;
	    while ((iteration_count>0) && (act_pseudo>PSEUDO))
	    {
		switch_model(&lambda, &lambda_train);
		lambda->estimate_model_viterbi_defined(lambda_train);
		prob=lambda->best_path(-1) ;
		prob_train=lambda_train->best_path(-1) ;
		if (fabs(prob-prob_train)>EPSILON)
		    converge(prob, prob_train) ;
		if (fabs(prob-prob_train)<=eps_add)
		{
		    act_pseudo*=step ;
		    lambda->set_pseudo(act_pseudo) ;
		    lambda_train->set_pseudo(act_pseudo) ;
		   CIO::message("   pseudo=%e",act_pseudo) ; 
		} ;
	    }
	}
	else
	   CIO::message("create model first\n");
	if ((lambda) && (lambda_train)) 
	{
	    char templname[30]="/tmp/vit_def_model_XXXXXX" ;
	    mkstemp(templname);
	    double prob=0.0, prob_train=0.0 ;
	    iteration_count=ITERATIONS ;
	    while (!converge(prob, prob_train))
	    {
		switch_model(&lambda, &lambda_train);
		prob=prob_train ;
		lambda->estimate_model_viterbi_defined(lambda_train);
		prob_train=lambda_train->best_path(-1) ;
		FILE* file=fopen(templname, "w");
		if (prob>prob_train)
		{
		   CIO::message("\nsaving model with filename %s ... ", templname) ;
		    lambda->save_model(file) ;
		    fclose(file) ;
		   CIO::message("done.") ;
		}
		else
		   CIO::message("\nskipping TMP_SAVE. model got worse.");
	    }
	}
	else
	   CIO::message("create model first\n");
	if ((lambda) && (lambda_train)) 
	{
	    char templname[30]="/tmp/vit_model_XXXXXX" ;
	    mkstemp(templname);
	    double prob=0.0,prob_train=0.0 ;
	    iteration_count=ITERATIONS ;
	    while (!converge(prob, prob_train)) 
	    {
		switch_model(&lambda, &lambda_train);
		prob=prob_train ;
		lambda->estimate_model_viterbi(lambda_train);
		prob_train=lambda_train->best_path(-1) ;
		FILE* file=fopen(templname, "w");
		if (prob>prob_train)
		{
		   CIO::message("\nsaving model with filename %s ... ", templname) ;
		    lambda->save_model(file) ;
		    fclose(file) ;
		   CIO::message("done.") ;
		}
		else
		   CIO::message("\nskipping TMP_SAVE. model got worse.");
	    }
	}
	else
	   CIO::message("create model first\n");
	char templname[30]="/tmp/bw_def_model_XXXXXX" ;
	mkstemp(templname);
	char templname_best[40] ;
	sprintf(templname_best, "%s_best", templname) ;
	double prob_max=-CMath::INFTY ;
	iteration_count=ITERATIONS ;
	if ((lambda) && (lambda_train)) 
	{
		if (lambda->get_observations() && lambda_train->get_observations())
		{
			double prob_train=math.ALMOST_NEG_INFTY, prob = -math.INFTY ;
			while (!converge(prob, prob_train))
			{
			switch_model(&lambda, &lambda_train);
			prob=prob_train ;
			lambda->estimate_model_baum_welch_defined(lambda_train);
			prob_train=lambda_train->model_probability();
			if (prob_max<prob_train)
			{
				prob_max=prob_train ;
				FILE* file=fopen(templname_best, "w");
				printf("\nsaving best model with filename %s ... ", templname_best) ;
				lambda_train->save_model(file) ;
				fclose(file) ;
				printf("done.") ;		      
			} 
			else
			{
				FILE* file=fopen(templname, "w");
				printf("\nsaving model with filename %s ... ", templname) ;
				lambda->save_model(file) ;
				fclose(file) ;
				printf("done.") ;
			}
	    }
		}
		else
			printf("assign observation first\n");

	}
	else
	   CIO::message("create model first\n");
	if ((lambda) && (lambda_train)) 
	{
	    lambda->output_model_sequence(false);
	}
	else
	   CIO::message("create model first\n");
	if ((lambda) && (lambda_train)) 
	{
	    lambda->output_model(false);
	}
	else
	   CIO::message("create model first\n");
	if (lambda)
	{
	    lambda->output_model_defined(true);
	}
	else
	   CIO::message("create model first\n");
	int from, to ;
	for (i=strlen(N_OUTPUT_PATH); isspace(input[i]); i++);

	if (sscanf(&input[i], "%d %d", &from, &to) != 2)
	{
	    from=0; 
	    to=10 ;
	}

	if (lambda)
	{
	    lambda->output_model_sequence(true,from,to);
	}
	else
	   CIO::message("create model first\n");
	if (lambda)
	{
	    lambda->output_gene_positions(true);
	}
	else
	   CIO::message("create model first\n");
	if (lambda)
	{
	    lambda->output_model(true);
	    lambda_train->output_model(true);
	}
	else
	   CIO::message("create model first\n");
	for (i=strlen(N_EXEC); isspace(input[i]); i++);

	FILE* file=fopen(&input[i], "r");

	if (!file)
	{
	   CIO::message("error opening/reading file: \"%s\"",&input[i]);
	    return true;
	}
	else
	{
	    while(!feof(file) && get_line(file));
	    fclose(file);
	}

	return true;
	lambda->check_path_derivatives() ;
	for (i=strlen(N_LINEAR_LIKELIHOOD); isspace(input[i]); i++);

	int WIDTH=-1,UPTO=-1;
	char fname[1024];
	sscanf(&input[i], "%s %d %d", fname, &WIDTH, &UPTO);

	if (lambda)
	{
	    FILE* file=fopen(fname, "r");

	    if (file) 
	    {
		if (WIDTH < 0 || UPTO < 0 )
		{
		    char buf[1024];
		    if ( (fread(buf, sizeof (unsigned char), sizeof(buf), file)) == sizeof(buf))
		    {
			for (int i=0; i<(int)sizeof(buf); i++)
			{
			    if (buf[i]=='\n')
			    {
				WIDTH=i+1;
				UPTO=i;
				printf("detected WIDTH=%d UPTO=%d\n",WIDTH, UPTO);
				break;
			    }
			}
			fseek(file,0,SEEK_SET);
		    }
		    else
			return false;

		    if (UPTO==lambda->get_N())
		    {	  
				CObservation* obs=new CObservation(TRAIN, alphabet, (BYTE)ceil(log(lambda->get_M())/log(2)), lambda->get_M(), ORDER);
			lambda->set_observation_nocache(obs);
			printf("log(Pr[O|model])=%e, #states: %i, #observation symbols: %i\n", 
				(double)lambda->linear_likelihood(file, WIDTH, UPTO), lambda->get_N(), lambda->get_M());
			lambda->set_observation_nocache(NULL);
			delete obs;
		    }
		    else
			printf("model has wrong size\n");
		}

		fclose(file);
	    }
	    else
		printf("opening file %s failed!\n", fname);

	}
	else
	   CIO::message("create model first!\n");


		int WIDTH=-1,UPTO=-1;
		char srcname[1024];
		char dstname[1024];

		for (i=strlen(N_SAVE_LINEAR_LIKELIHOOD); isspace(input[i]); i++);
		sscanf(&input[i], "%s %s %d %d", srcname, dstname, &WIDTH, &UPTO);

		if (lambda)
		{
			FILE* srcfile=fopen(srcname, "r");
			FILE* dstfile=fopen(dstname, "w");

			if (srcfile && dstfile) 
			{
			if (WIDTH < 0 || UPTO < 0 )
			{
				char buf[1024];
				if ( (fread(buf, sizeof (unsigned char), sizeof(buf), srcfile)) == sizeof(buf))
				{
				for (int i=0; i<(int)sizeof(buf); i++)
				{
					if (buf[i]=='\n')
					{
					WIDTH=i+1;
					UPTO=i;
					printf("detected WIDTH=%d UPTO=%d\n",WIDTH, UPTO);
					break;
					}
				}
				fseek(srcfile,0,SEEK_SET);
				}
				else
				return false;

				if (UPTO==lambda->get_N())
				{
					CObservation* obs=new CObservation(TRAIN, alphabet, (BYTE)ceil(log(lambda->get_M())/log(2)), lambda->get_M(), ORDER);
				    lambda->set_observation_nocache(obs);
				    lambda->save_linear_likelihood(srcfile, dstfile, WIDTH, UPTO);
				    lambda->set_observation_nocache(NULL);
				    delete obs;
				}
				else
				printf("model has wrong size\n");
			}

			fclose(srcfile);
			fclose(dstfile);
			}
			else
			printf("opening files %s or %s failed!\n", srcname, dstname);

		}
		else
			printf("create model first!\n");


	int WIDTH=-1,UPTO=-1;
	char srcname[1024];
	char dstname[1024];

	for (i=strlen(N_SAVE_LINEAR_LIKELIHOOD_BIN); isspace(input[i]); i++);
	sscanf(&input[i], "%s %s %d %d", srcname, dstname, &WIDTH, &UPTO);

	if (lambda)
	{
	    FILE* srcfile=fopen(srcname, "r");
	    FILE* dstfile=fopen(dstname, "w");

	    if (srcfile && dstfile) 
	    {
		if (WIDTH < 0 || UPTO < 0 )
		{
		    char buf[1024];
		    if ( (fread(buf, sizeof (unsigned char), sizeof(buf), srcfile)) == sizeof(buf))
		    {
			for (int i=0; i<(int)sizeof(buf); i++)
			{
			    if (buf[i]=='\n')
			    {
				WIDTH=i+1;
				UPTO=i;
				printf("detected WIDTH=%d UPTO=%d\n",WIDTH, UPTO);
				break;
			    }
			}
			fseek(srcfile,0,SEEK_SET);
		    }
		    else
			return false;
			
		    if (UPTO==lambda->get_N())
		    {
				CObservation* obs=new CObservation(TRAIN, alphabet, (BYTE)ceil(log(lambda->get_M())/log(2)), lambda->get_M(), ORDER);
			lambda->set_observation_nocache(obs);
			lambda->save_linear_likelihood_bin(srcfile, dstfile, WIDTH, UPTO);
			lambda->set_observation_nocache(NULL);
			delete obs;
		    }
		    else
			printf("model has wrong size\n");
		}

		fclose(srcfile);
		fclose(dstfile);
	    }
	    else
		printf("opening files %s or %s failed!\n", srcname, dstname);

	}
	else
	   CIO::message("create model first!\n");
	char name[1024];

	for (i=strlen(N_SVM_TRAIN); isspace(input[i]); i++);
	if (sscanf(&input[i], "%s", name) == 0)
	  strcpy(name,"") ;
	if (pos && neg)
	  {
	    if (obs_postrain && obs_negtrain)
	      {
		CObservation* obs=new CObservation(obs_postrain, obs_negtrain);
		
		CObservation* old_pos=pos->get_observations();
		CObservation* old_neg=neg->get_observations();
		
		pos->set_observations(obs);
		neg->set_observations(obs);
	
#warning extract this to extra guifunction select_features TOP|BLA|BLA
		CTOPFeatures* topfeatures=new CTOPFeatures(pos, neg);
		topfeatures->set_feature_matrix();
		
		if (trainfeatures)
			delete[] trainfeatures;
		trainfeatures=topfeatures;
		trainfeatures->set_preproc(preproc);
		trainfeatures->preproc_feature_matrix();

		svm->set_kernel(kernel);
		svm->set_C(C);

		svm->svm_train(trainfeatures);

		if (strlen(name)>0)
		  {
		    FILE * fd=fopen(name,"w+") ;
		    svm->save_svm(fd) ;
		    fclose(fd) ;
		  } 
		
		pos->set_observations(old_pos);
		neg->set_observations(old_neg);
		
		delete obs;
	      }
	    else
	     CIO::message("assign postrain and negtrain observations first!\n");
	  }
	else
	 CIO::message("assign positive and negative models first!\n");
	char name[1024];

	for (i=strlen(N_LINEAR_SVM_TRAIN); isspace(input[i]); i++);
	if (sscanf(&input[i], "%s", name) == 0)
	  strcpy(name,"") ;
	
	if (pos && neg)
	  {
	    if (obs_postrain && obs_negtrain)
	      {
		CObservation* obs=new CObservation(obs_postrain, obs_negtrain);
		
		CObservation* old_pos=pos->get_observations();
		CObservation* old_neg=neg->get_observations();
		
		pos->set_observation_nocache(obs);
		neg->set_observation_nocache(obs);
		
		CTOPFeatures* topfeatures=new CTOPFeatures(pos, neg);
		topfeatures->set_feature_matrix();
		
		if (trainfeatures)
			delete[] trainfeatures;
		trainfeatures=topfeatures;
		trainfeatures->set_preproc(preproc);
		trainfeatures->preproc_feature_matrix();

		svm->set_kernel(kernel);
		svm->set_C(C);

		svm->svm_train(trainfeatures);

		if (strlen(name)>0)
		  {
		    FILE * fd=fopen(name,"w+") ;
		    svm->save_svm(fd) ;
		    fclose(fd) ;
		  } 
		
		pos->set_observations(old_pos);
		neg->set_observations(old_neg);
		
		delete obs;
	      }
	    else
	     CIO::message("assign postrain and negtrain observations first!\n");
	  }
	else
	 CIO::message("assign positive and negative models first!\n");
	char svmname[1024];
	char outputname[1024];
	char rocfname[1024];
	FILE* outputfile=stdout;
	FILE* rocfile=NULL;
	int numargs=-1;

#warning	CHMM::invalidate_top_feature_cache(CHMM::SV_INVALID);
	for (i=strlen(N_SVM_TEST); isspace(input[i]); i++);
	numargs=sscanf(&input[i], "%s %s %s", svmname, outputname, rocfname);
	if (numargs >= 1)
	{
	    FILE* svm_file=fopen(svmname, "r");
	    if (svm_file)
	    {
		if (numargs>=2)
		{
		    outputfile=fopen(outputname, "w");

		    if (!outputfile)
		    {
			CIO::message(stderr,"ERROR: could not open %s\n",outputname);
			return false;
		    }
		   
		    if (numargs==3) 
		    {
			rocfile=fopen(rocfname, "w");

			if (!rocfile)
			{
			    CIO::message(stderr,"ERROR: could not open %s\n",rocfname);
			    return false;
			}
		    }
		}

		printf("testing\n");
		
		if (pos && neg)
		{
		    if (obs_postest && obs_negtest)
		    {
			CObservation* obs=new CObservation(obs_postest, obs_negtest);
			svm->load_svm(svm_file) ;

			CObservation* old_pos=pos->get_observations();
			CObservation* old_neg=neg->get_observations();

			pos->set_observations(obs);
			neg->set_observations(obs);
#warning handle {train,test}features separately		
			CTOPFeatures* topfeatures=new CTOPFeatures(pos, neg);
			topfeatures->set_feature_matrix();
		
			if (testfeatures)
				delete[] testfeatures;

			testfeatures=topfeatures;
			testfeatures->set_preproc(preproc);
			testfeatures->preproc_feature_matrix();

			//svm->svm_test(features, outputfile, rocfile);
			svm->set_kernel(kernel);
			double* output=svm->svm_test(testfeatures, trainfeatures);

#warning do something with the outputs here			

			pos->set_observations(old_pos);
			neg->set_observations(old_neg);

			delete obs;
		    }
		    else
			printf("assign postrain and negtrain observations first!\n");
		    
		    if ( numargs>=2 )
		    {
			if (outputfile)
			    fclose(outputfile);

			if (rocfile)
			    fclose(rocfile);
		    }
		}
		else
		   CIO::message("assign positive and negative models first!\n");

		fclose(svm_file);
	    }
	    else
		printf("could not open svm model\n");
	}
	else
	   CIO::message("see help for parameters\n");
	char posname[1024];
	char negname[1024];
	char outputname[1024];
	char rocfname[1024];
	FILE* outputfile=stdout;
	FILE* rocfile=NULL;
	int numargs=-1;
	double tresh=0.5;
	
	int WIDTH=-1,UPTO=-1;

	for (i=strlen(N_ONE_CLASS_LINEAR_HMM_TEST); isspace(input[i]); i++);

	numargs=sscanf(&input[i], "%s %s %le %s %s %d %d", negname, posname, &tresh, outputname, rocfname, &WIDTH,&UPTO);

	if (numargs >= 2)
	{
	    if (numargs>=4)
	    {
		outputfile=fopen(outputname, "w");

		if (!outputfile)
		{
		    CIO::message(stderr,"ERROR: could not open %s\n",outputname);
		    return false;
		}

		if (numargs>=5) 
		{
		    rocfile=fopen(rocfname, "w");

		    if (!rocfile)
		    {
			CIO::message(stderr,"ERROR: could not open %s\n",rocfname);
			return false;
		    }
		}
	    }
	    if (test)
	    {
		FILE* posfile=fopen(posname, "r");
		FILE* negfile=fopen(negname, "r");

		if (posfile && negfile)
		{
		   CIO::message("opened %s and %s\n",posname,negname);
		    if (WIDTH < 0 || UPTO < 0 )
		    {
			char buf[1024];
			if ( (fread(buf, sizeof (unsigned char), sizeof(buf), posfile)) == sizeof(buf))
			{
			    for (int i=0; i<(int)sizeof(buf); i++)
			    {
				if (buf[i]=='\n')
				{
				    WIDTH=i+1;
				    UPTO=i;
				   CIO::message("detected WIDTH=%d UPTO=%d\n",WIDTH, UPTO);
				   
				    break;
				}
			    }
			    fseek(posfile,0,SEEK_SET);
			}
			else
			    return false;

			if (UPTO==test->get_N())
			{
			    fseek(posfile,0,SEEK_END);
			    int posfsize=ftell(posfile);
			    fseek(posfile,0,SEEK_SET);

			    fseek(negfile,0,SEEK_END);
			    int negfsize=ftell(negfile);
			    fseek(negfile,0,SEEK_SET);

			    if ( ((posfsize/WIDTH)*WIDTH!=posfsize) || ((negfsize/WIDTH)*WIDTH!=negfsize))
			    {
				CIO::message(stderr,"ERROR: file has wrong size");
				return false;
			    }
			    	    
			    int possize=posfsize/WIDTH;
			    int negsize=negfsize/WIDTH;
			    int total=possize+negsize;

			   CIO::message("p:%d,n:%d,t:%d\n",possize,negsize,total);
			    REAL* output = new REAL[total];	
			    int* label= new int[total];	

			    for (int dim=0; dim<total; dim++)
			    {
				if (dim<negsize)
				{
				    output[dim]=test->linear_likelihood(negfile, WIDTH, UPTO,true)-tresh;
				    label[dim]=-1;
				    
				    if (output[dim] < 0)
					fprintf(outputfile,"%+.8g (%+d)\n",(double) output[dim], label[dim]);
				    else
					fprintf(outputfile,"%+.8g (%+d)(*)\n",(double) output[dim], label[dim]);
				}
				else
				{
				    output[dim]=test->linear_likelihood(posfile, WIDTH, UPTO,true)-tresh;
				    label[dim]=+1;
				    
				    if (output[dim] > 0)
					fprintf(outputfile,"%+.8g (%+d)\n",(double) output[dim], label[dim]);
				    else
					fprintf(outputfile,"%+.8g (%+d)(*)\n",(double) output[dim], label[dim]);
				}
			    }

			    REAL* fp= new REAL[total];	
			    REAL* tp= new REAL[total];	

			    int pointeven=math.calcroc(fp, tp, output, label, total, possize, negsize, rocfile);

			    double correct=possize*tp[pointeven]+(1-fp[pointeven])*negsize;
			    double fpo=fp[pointeven]*negsize;
			    double fne=(1-tp[pointeven])*possize;

			   CIO::message("classified:\n");
			   CIO::message("\tcorrect:%i\n", int (correct));
			   CIO::message("\twrong:%i (fp:%i,fn:%i)\n", int(fpo+fne), int (fpo), int (fne));
			   CIO::message("of %i samples (c:%f,w:%f,fp:%f,tp:%f)\n",total, correct/total, 1-correct/total, fp[pointeven], tp[pointeven]);
			    delete[] fp;
			    delete[] tp;
			    delete[] output;
			    delete[] label;

			}
			else
			   CIO::message("model has wrong size\n");
		    }

		}
		else
		   CIO::message("assign postrain and negtrain observations first!\n");
		
		if (posfile)
		    fclose(posfile);
		if (negfile)
		    fclose(negfile);
	    }
	    else
		printf("assign positive and negative models first!\n");
	}
	else
	   CIO::message("see help for parameters\n");
	char posname[1024];
	char negname[1024];
	char outputname[1024];
	char rocfname[1024];
	FILE* outputfile=stdout;
	FILE* rocfile=NULL;
	int numargs=-1;
	
	int WIDTH=-1,UPTO=-1;

	for (i=strlen(N_LINEAR_HMM_TEST); isspace(input[i]); i++);

	numargs=sscanf(&input[i], "%s %s %s %s %d %d", negname, posname, outputname, rocfname, &WIDTH,&UPTO);

	if (numargs >= 2)
	{
	    if (numargs>=3)
	    {
		outputfile=fopen(outputname, "w");

		if (!outputfile)
		{
		    CIO::message(stderr,"ERROR: could not open \"%s\"\n",outputname);
		    return false;
		}

		if (numargs>=4) 
		{
		    rocfile=fopen(rocfname, "w");

		    if (!rocfile)
		    {
			CIO::message(stderr,"ERROR: could not open %s\n",rocfname);
			return false;
		    }
		}
	    }

	    if (pos && neg)
	    {
		FILE* posfile=fopen(posname, "r");
		FILE* negfile=fopen(negname, "r");

		if (posfile && negfile)
		{
		   CIO::message("opened %s and %s\n",posname,negname);
		    if (WIDTH < 0 || UPTO < 0 )
		    {
			char buf[1024];
			if ( (fread(buf, sizeof (unsigned char), sizeof(buf), posfile)) == sizeof(buf))
			{
			    for (int i=0; i<(int)sizeof(buf); i++)
			    {
				if (buf[i]=='\n')
				{
				    WIDTH=i+1;
				    UPTO=i;
				   CIO::message("detected WIDTH=%d UPTO=%d\n",WIDTH, UPTO);
				   
				    break;
				}
			    }
			    fseek(posfile,0,SEEK_SET);
			}
			else
			    return false;

			if (UPTO==pos->get_N())
			{
			    fseek(posfile,0,SEEK_END);
			    int posfsize=ftell(posfile);
			    fseek(posfile,0,SEEK_SET);

			    fseek(negfile,0,SEEK_END);
			    int negfsize=ftell(negfile);
			    fseek(negfile,0,SEEK_SET);

			    if ( ((posfsize/WIDTH)*WIDTH!=posfsize) || ((negfsize/WIDTH)*WIDTH!=negfsize))
			    {
				CIO::message(stderr,"ERROR: file has wrong size");
				return false;
			    }

			    int possize=posfsize/WIDTH;
			    int negsize=negfsize/WIDTH;
			    int total=possize+negsize;

				CObservation* obs=new CObservation(TRAIN, alphabet, (BYTE)ceil(log(pos->get_M())/log(2)), pos->get_M(), ORDER);
			    pos->set_observation_nocache(obs);
			    neg->set_observation_nocache(obs);

			    CIO::message("p:%d,n:%d,t:%d\n",possize,negsize,total);
			    REAL* output = new REAL[total];	
			    int* label= new int[total];	

			    for (int dim=0; dim<total; dim++)
			    {
				if (dim<negsize)
				{
				    int fileptr=ftell(negfile);
				    output[dim]=pos->linear_likelihood(negfile, WIDTH, UPTO,true);
				    fseek(negfile, fileptr, SEEK_SET);
				    output[dim]-=neg->linear_likelihood(negfile, WIDTH, UPTO,true);
				    label[dim]=-1;

				    if (output[dim] < 0)
					fprintf(outputfile,"%+.8g (%+d)\n",(double) output[dim], label[dim]);
				    else
					fprintf(outputfile,"%+.8g (%+d)(*)\n",(double) output[dim], label[dim]);
				}
				else
				{
				    int fileptr=ftell(posfile);
				    output[dim]=pos->linear_likelihood(posfile, WIDTH, UPTO,true);
				    fseek(posfile, fileptr, SEEK_SET);
				    output[dim]-=neg->linear_likelihood(posfile, WIDTH, UPTO,true);
				    label[dim]=+1;

				    if (output[dim] > 0)
					fprintf(outputfile,"%+.8g (%+d)\n",(double) output[dim], label[dim]);
				    else
					fprintf(outputfile,"%+.8g (%+d)(*)\n",(double) output[dim], label[dim]);
				}
			    }

			    REAL* fp= new REAL[total];	
			    REAL* tp= new REAL[total];	

			    int pointeven=math.calcroc(fp, tp, output, label, total, possize, negsize, rocfile);

			    double correct=possize*tp[pointeven]+(1-fp[pointeven])*negsize;
			    double fpo=fp[pointeven]*negsize;
			    double fne=(1-tp[pointeven])*possize;

			    CIO::message("classified:\n");
			    CIO::message("\tcorrect:%i\n", int (correct));
			    CIO::message("\twrong:%i (fp:%i,fn:%i)\n", int(fpo+fne), int (fpo), int (fne));
			    CIO::message("of %i samples (c:%f,w:%f,fp:%f,tp:%f)\n",total, correct/total, 1-correct/total, fp[pointeven], tp[pointeven]);

			    pos->set_observation_nocache(NULL);
			    neg->set_observation_nocache(NULL);
			    delete obs;

			    delete[] fp;
			    delete[] tp;
			    delete[] output;
			    delete[] label;

			}
			else
			   CIO::message("model has wrong size\n");
		    }

		}
		else
		   CIO::message("assign postrain and negtrain observations first!\n");
		
		if (posfile)
		    fclose(posfile);
		if (negfile)
		    fclose(negfile);
	    }
	    else
		printf("assign positive and negative models first!\n");
	}
	else
	   CIO::message("see help for parameters\n");
	char outputname[1024];
	char rocfname[1024];
	FILE* outputfile=stdout;
	FILE* rocfile=NULL;
	int numargs=-1;
	double tresh=0.5;

	for (i=strlen(N_ONE_CLASS_HMM_TEST); isspace(input[i]); i++);

	numargs=sscanf(&input[i], "%le %s %s", &tresh, outputname, rocfname);

	CIO::message("Tresholding at:%f\n",tresh);
	if (numargs>=2)
	{
	    outputfile=fopen(outputname, "w");

	    if (!outputfile)
	    {
		CIO::message(stderr,"ERROR: could not open %s\n",outputname);
		return false;
	    }

	    if (numargs==3) 
	    {
		rocfile=fopen(rocfname, "w");

		if (!rocfile)
		{
		    CIO::message(stderr,"ERROR: could not open %s\n",rocfname);
		    return false;
		}
	    }
	}

	if (test)
	{
	    if (obs_postest && obs_negtest)
	    {
		CObservation* obs=new CObservation(obs_postest, obs_negtest);

		CObservation* old_test=test->get_observations();
		test->set_observations(obs);

		int total=obs->get_DIMENSION();

		REAL* output = new REAL[total];	
		int* label= new int[total];	

		REAL* fp= new REAL[total];	
		REAL* tp= new REAL[total];	

		for (int dim=0; dim<total; dim++)
		{
		    output[dim]=test->model_probability(dim)-tresh;
		    label[dim]= obs->get_label(dim);

		    if (math.sign((REAL) output[dim])==label[dim])
			fprintf(outputfile,"%+.8g (%+d)\n",(double) output[dim], label[dim]);
		    else
			fprintf(outputfile,"%+.8g (%+d)(*)\n",(double) output[dim], label[dim]);
		}

		int possize,negsize;
		int pointeven=math.calcroc(fp, tp, output, label, total, possize, negsize, rocfile);

		double correct=possize*tp[pointeven]+(1-fp[pointeven])*negsize;
		double fpo=fp[pointeven]*negsize;
		double fne=(1-tp[pointeven])*possize;

		printf("classified:\n");
		printf("\tcorrect:%i\n", int (correct));
		printf("\twrong:%i (fp:%i,fn:%i)\n", int(fpo+fne), int (fpo), int (fne));
		printf("of %i samples (c:%f,w:%f,fp:%f,tp:%f)\n",total, correct/total, 1-correct/total, (double) fp[pointeven], (double) tp[pointeven]);

		delete[] fp;
		delete[] tp;
		delete[] output;
		delete[] label;

		test->set_observations(old_test);

		delete obs;
	    }
	    else
		printf("assign posttest and negtest observations first!\n");
	}
	else
	   CIO::message("assign test model first!\n");
	char outputname[1024];
	char rocfname[1024];
	FILE* outputfile=stdout;
	FILE* rocfile=NULL;
	int numargs=-1;

	for (i=strlen(N_HMM_TEST); isspace(input[i]); i++);

	numargs=sscanf(&input[i], "%s %s", outputname, rocfname);

	if (numargs>=1)
	{
	    outputfile=fopen(outputname, "w");

	    if (!outputfile)
	    {
		CIO::message(stderr,"ERROR: could not open %s\n",outputname);
		return false;
	    }

	    if (numargs==2) 
	    {
		rocfile=fopen(rocfname, "w");

		if (!rocfile)
		{
		    CIO::message(stderr,"ERROR: could not open %s\n",rocfname);
		    return false;
		}
	    }
	}

	if (pos && neg)
	{
	    if (obs_postest && obs_negtest)
	    {
		CObservation* obs=new CObservation(obs_postest, obs_negtest);

		CObservation* old_pos=pos->get_observations();
		CObservation* old_neg=neg->get_observations();

		pos->set_observations(obs);
		neg->set_observations(obs);

		int total=obs->get_DIMENSION();

		REAL* output = new REAL[total];	
		int* label= new int[total];	

		REAL* fp= new REAL[total];	
		REAL* tp= new REAL[total];	

		for (int dim=0; dim<total; dim++)
		{
		    output[dim]=pos->model_probability(dim)-neg->model_probability(dim);
		    label[dim]= obs->get_label(dim);

		    if (math.sign((REAL) output[dim])==label[dim])
			fprintf(outputfile,"%+.8g (%+d)\n",(double) output[dim], label[dim]);
		    else
			fprintf(outputfile,"%+.8g (%+d)(*)\n",(double) output[dim], label[dim]);
		}

		int possize,negsize;
		int pointeven=math.calcroc(fp, tp, output, label, total, possize, negsize, rocfile);

		double correct=possize*tp[pointeven]+(1-fp[pointeven])*negsize;
		double fpo=fp[pointeven]*negsize;
		double fne=(1-tp[pointeven])*possize;

		printf("classified:\n");
		printf("\tcorrect:%i\n", int (correct));
		printf("\twrong:%i (fp:%i,fn:%i)\n", int(fpo+fne), int (fpo), int (fne));
		printf("of %i samples (c:%f,w:%f,fp:%f,tp:%f)\n",total, correct/total, 1-correct/total, (double) fp[pointeven], (double) tp[pointeven]);

		delete[] fp;
		delete[] tp;
		delete[] output;
		delete[] label;

		pos->set_observations(old_pos);
		neg->set_observations(old_neg);

		delete obs;
	    }
	    else
		printf("assign postest and negtest observations first!\n");
	}
	else
	   CIO::message("assign positive and negative models first!\n");
	    if (lambda)
	    {
		    for (i=strlen(N_APPEND_MODEL); isspace(input[i]); i++);

		    FILE* model_file=fopen(&input[i], "r");

		    if (model_file)
		    {
			    CHMM* h=new CHMM(model_file,PSEUDO);
			    if (h && h->get_status())
			    {
				    printf("file successfully read\n");
				    fclose(model_file);

				    REAL cur_o[]= {0, -1000, -1000, -1000};
				    REAL app_o[]= {-1000, -1000, 0, -1000};

				    //REAL cur_o[]= {0, -math.INFTY, -math.INFTY, -math.INFTY};
				    //REAL app_o[]= {-math.INFTY, -math.INFTY, 0, -math.INFTY};

				    lambda->append_model(h, cur_o, app_o);
				    lambda_train->append_model(h, cur_o, app_o);
				    CIO::message("new model has %i states\n", lambda->get_N());
				    delete h;
			    }
			    else
				    CIO::message("reading file %s failed\n", &input[i]);
		    }
		    else
			    CIO::message("opening file %s failed\n", &input[i]);
	    }
	    else
		    CIO::message("create model first\n");

	if (lambda && lambda_train)
	{
		int states=1;
		double value=0;

		for (i=strlen(N_ADD_STATES); isspace(input[i]); i++);
		sscanf(&input[i], "%i %le", &states, &value);
		CIO::message("adding %i states\n", states);
		lambda->add_states(states, value);
		lambda_train->add_states(states, value);
		CIO::message("new model has %i states\n", lambda->get_N());
	}
	else
	   CIO::message("create model first\n");
	
	double new_C;
	for (i=strlen(N_C); isspace(input[i]); i++);
	if (sscanf(&input[i], "%le", &new_C) == 1)
	{
	    C=new_C;
	}
	else
		printf("current setting: C=%e\n", C);
		for (i=strlen(N_SET_ORDER); isspace(input[i]); i++);
		
		if (sscanf(&input[i], "%d", &ORDER)==1)
			CIO::message("setting ORDER to %d\n", ORDER);
		else
			CIO::message("current ORDER is set to %d, see help for parameters.\n", ORDER);
*/
#endif
