
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <ctype.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#ifndef _WIN32
//#include <termios.h>
#endif

#include "hmm/HMM.h"
#include "lib/Observation.h"
#include "lib/Mathmatics.h"
#include "svm/SVM.h"

//names of menu commands
static const char* N_SET_POS_MODEL=			"set_pos_model";
static const char* N_SET_NEG_MODEL=			"set_neg_model";
static const char* N_LOAD_MODEL=			"load_model";
static const char* N_SAVE_MODEL=			"save_model";
static const char* N_SAVE_MODEL_BIN=			"save_model_bin";
static const char* N_LOAD_DEFINITIONS=		        "load_defs";
#ifndef NOVIT
static const char* N_SAVE_PATH=				"save_path";
static const char* N_SAVE_PATH_DERIVATIVES=	        "save_vit_deriv";
static const char* N_SAVE_PATH_DERIVATIVES_BIN=	        "save_vit_deriv_bin";
static const char* N_BEST_PATH=			        "best_path";
static const char* N_OUTPUT_PATH=			"output_path";
static const char* N_OUTPUT_GENES=			"output_genes";
static const char* N_VITERBI_TRAIN=		        "vit";
static const char* N_VITERBI_TRAIN_DEFINED=             "vit_def";
static const char* N_VITERBI_TRAIN_DEFINED_ANNEALED=    "vit_def_ann";
static const char* N_VITERBI_TRAIN_DEFINED_ADDIABATIC=  "vit_def_add";
#endif // NOVIT
static const char* N_LINEAR_TRAIN=			"linear_train";
static const char* N_LINEAR_LIKELIHOOD=			"linear_likelihood";
static const char* N_SAVE_LINEAR_LIKELIHOOD=		"save_linear_likelihood";
static const char* N_SAVE_LINEAR_LIKELIHOOD_BIN=	"save_linear_likelihood_bin";

static const char* N_SAVE_MODEL_DERIVATIVES=            "save_bw_deriv";
static const char* N_SAVE_MODEL_DERIVATIVES_BIN=        "save_bw_deriv_bin";
static const char* N_SAVE_LIKELIHOOD=                   "save_likelihood";
static const char* N_SAVE_LIKELIHOOD_BIN=               "save_likelihood_bin";
static const char* N_LOAD_OBSERVATIONS=		        "load_obs";
static const char* N_ASSIGN_OBSERVATION=	"assign_obs";
static const char* N_NEW=				"new";
static const char* N_CLEAR=				"clear";
static const char* N_CHOP=				"chop";
static const char* N_CONVERGENCE_CRITERIA=	        "convergence_criteria";
static const char* N_PSEUDO=				"pseudo";
static const char* N_BAUM_WELCH_TRAIN=		        "bw";
static const char* N_BAUM_WELCH_TRAIN_DEFINED=		"bw_def";
static const char* N_LIKELIHOOD=			"likelihood";
static const char* N_ALPHABET=			        "alphabet";
static const char* N_OUTPUT_MODEL=			"output_model";
static const char* N_OUTPUT_MODEL_DEFINED=              "output_model_defined";
static const char* N_QUIT=				"quit";
static const char* N_EXEC=				"exec";
static const char* N_EXIT=				"exit";
static const char* N_HELP=				"help";
static const char* N_SYSTEM=				"!";
#ifdef FIX_POS
static const char* N_FIX_POS_STATE=			"fix_pos_state";
#endif // FIX_POS
static const char* N_SET_MAX_DIM=			"max_dim";
#ifdef DEBUG
static const char* N_TEST=				"test";
#endif // DEBUG
static const char* N_LINEAR_SVM_TRAIN=			"linear_svm_train";
static const char* N_SVM_TRAIN=				"svm_train";
static const char* N_SVM_TEST=				"svm_test";
static const char* N_HMM_TEST=				"hmm_test";
static const char* N_LINEAR_HMM_TEST=			"linear_hmm_test";

clock_t current_time;

/*		-format specs: out_file 
		train messages
		learned model (looks like model_file)
		state sequence (looks like in_file)
 */
FILE* out_file=stdout;

// initial parameters
int    ITERATIONS = 100;
double EPSILON    = 1e-6;
double PSEUDO     = 1e-3 ;
int ORDER=1;
int M=4;

CHMM* lambda=NULL;
CHMM* lambda_train=NULL;	//model and training model

E_OBS_ALPHABET alphabet=DNA;
CObservation* obs_postrain=NULL;//observations
CObservation* obs_negtrain=NULL;//observations
CObservation* obs_postest=NULL; //observations
CObservation* obs_negtest=NULL; //observations
CObservation* obs_test=NULL;	//observations

CHMM* pos=NULL;	//positive model 
CHMM* neg=NULL;	//negative model
CSVM svm;	//support vector machine

double* theta; //full parameter vector

static int iteration_count=ITERATIONS ;
static int conv_it=5 ;

void error(int line, char* str)
{
    if (line)
	fprintf(out_file,"error in line %d %s\n", line, str);
    else
	fprintf(out_file,"error %s\n", str);

    fflush(out_file);

    //	exit(1);
}

//convergence criteria  -tobeadjusted-
bool converge(double x, double y)
{
    printf("\n #%03d\tbest result so far: %G (eps: %f", iteration_count, y, (float) y-x);
    if (y-x<0)
	printf(" ***") ;
    printf(")") ;
    if (fabs(x-y)<EPSILON)
	conv_it-- ;
    if (iteration_count-- == 0 || (fabs(x-y)<EPSILON && conv_it<=0))
    {
	iteration_count=ITERATIONS;
	printf("...finished\n");
	conv_it=5 ;
	return true;
    }
    else
	return false;
}


//switch model and train model
static void switch_model(CHMM** m1, CHMM** m2)
{
    CHMM* dummy= *m1;

    *m1= *m2;
    *m2= dummy;
}

//some initialization
static void initialize()
{
    lambda=NULL ;
    lambda_train=NULL ;
    fprintf(stdout,"Learning uses %i threads\n", NUM_PARALLEL) ;
    fflush(stdout);
}

//cleanup
static void cleanup()
{
    if (pos)
	delete pos;
    if (neg)
	delete neg;
    if (lambda)
	delete lambda;
    if (lambda_train)
	delete lambda_train;
}

static void help()
{
    printf("\n[LOAD]\n");
    printf("%s <filename>\t- load hmm\n",N_LOAD_MODEL);
    printf("%s <filename> [initialize=1]\t- load hmm defs\n",N_LOAD_DEFINITIONS);
    printf("%s <filename>\t- load observed data\n",N_LOAD_OBSERVATIONS);
    printf("\n[SAVE]\n");
    printf("%s <filename>\t- save hmm\n",N_SAVE_MODEL);
    printf("%s <filename>\t- save hmm in binary format\n",N_SAVE_MODEL_BIN);
#ifndef NOVIT
    printf("%s <filename>\t- save state sequence of viterbi path\n",N_SAVE_PATH);
    printf("%s <filename>\t- save derivatives of log P[O,Q_best|model]\n",N_SAVE_PATH_DERIVATIVES);
    printf("%s <filename>\t- save derivatives of log P[O,Q_best|model] in binary format\n",N_SAVE_PATH_DERIVATIVES_BIN);
#endif // NOVIT
    printf("%s <filename>\t- save log derivatives of P[O|model]\n",N_SAVE_MODEL_DERIVATIVES);
    printf("%s <filename>\t- save log derivatives of P[O|model] in binary format\n",N_SAVE_MODEL_DERIVATIVES_BIN);
    printf("%s <filename>\t- save P[O|model]\n",N_SAVE_LIKELIHOOD);
    printf("%s <filename>\t- save P[O|model]\n",N_SAVE_LIKELIHOOD_BIN);
    printf("%s <srcname> <destname> [<width> <upto>]\t\t- saves likelihood for linear model from file\n",N_SAVE_LINEAR_LIKELIHOOD);
    printf("%s <srcname> <destname> [<width> <upto>]\t\t- saves likelihood for linear model from file\n",N_SAVE_LINEAR_LIKELIHOOD_BIN);
    printf("\n[MODEL]\n");
    printf("%s - frees all models and observations\n",N_CLEAR);
    printf("%s #states #oberservations #order\t- frees previous model and creates an empty new one\n",N_NEW);
    printf("%s <POSTRAIN|NEGTRAIN|POSTEST|NEGTEST|TEST> - assign observation to current model\n",N_ASSIGN_OBSERVATION);
    printf("%s - make current model the positive model; then free current model \n",N_SET_POS_MODEL);
    printf("%s - make current model the negative model; then free current model \n",N_SET_NEG_MODEL);
    printf("%s <value>\t\t\t- chops likelihood of all parameters 0<value<1\n", N_CHOP);
    printf("%s [pseudovalue]\t\t\t- changes pseudo value\n", N_PSEUDO);
    printf("%s [PROTEIN|DNA|ALPHANUM]\t\t\t- changes alphabet type\n", N_ALPHABET);
    printf("%s [maxiterations] [maxallowedchange]\t- defines the convergence criteria for all train algorithms (%i,%e)\n",N_CONVERGENCE_CRITERIA,ITERATIONS,EPSILON);
#ifdef FIX_POS
    printf("%s position state\t- sets the state which has to be passed at a certain position\n",N_FIX_POS_STATE);
#endif
    printf("%s <max_dim>\t - set maximum number of patterns\n",N_SET_MAX_DIM);
    printf("\n[TRAIN]\n");
    printf("%s <filename> [<width> <upto>]\t\t- obtains new linear model from file\n",N_LINEAR_TRAIN);
    printf("%s <filename> [<width> <upto>]\t\t- computes likelihood for linear model from file\n",N_LINEAR_LIKELIHOOD);
#ifndef NOVIT
    printf("%s\t\t- does viterbi training on the current model\n",N_VITERBI_TRAIN);
    printf("%s\t\t- does viterbi training only on defined transitions etc\n",N_VITERBI_TRAIN_DEFINED);
    printf("%s [pseudo_start [in_steps]]\t\t- does viterbi training only on defined transitions with annealing\n",N_VITERBI_TRAIN_DEFINED_ANNEALED);
    printf("%s [pseudo_start [step [eps_add]]]\t\t- does viterbi training only on defined transitions with addiabatic annealing\n",N_VITERBI_TRAIN_DEFINED_ADDIABATIC);
#endif //NOVIT
    printf("%s\t\t- does baum welch training on current model\n",N_BAUM_WELCH_TRAIN);
    printf("%s\t\t- does baum welch training only on defined transitions etc.\n",N_BAUM_WELCH_TRAIN_DEFINED);
#ifndef NOVIT
    printf("%s\t- find the best path using viterbi\n",N_BEST_PATH);
#endif //NOVIT
    printf("%s\t- find model likelihood\n",N_LIKELIHOOD);
    printf("%s [maxiterations] [maxallowedchange]\t- defines the convergence criteria for all train algorithms (%i,%e)\n",N_CONVERGENCE_CRITERIA,ITERATIONS,EPSILON);
    printf("\n[OUTPUT]\n");
#ifndef NOVIT
    printf("%s [from to]\t- outputs best path\n",N_OUTPUT_PATH);
#endif //NOVIT
    printf("%s\t- output whole model\n",N_OUTPUT_MODEL);
    printf("\n[HMM-classification]\n");
    printf("%s\t\t\t\t- calculate output from obs using current HMMs\n",N_HMM_TEST);
    printf("%s <negtest> <postest> [<width> <upto>]\t- calculate svm output from obs using linear model\n",N_LINEAR_HMM_TEST);
    printf("\n[Hybrid HMM-<TOP-Kernel>-SVM]\n");
    printf("%s <dstsvm>\t\t- obtains svm from POS/NEGTRAIN using pos/neg HMM\n",N_SVM_TRAIN);
    printf("%s <srcsvm> <output>\t\t- calculate [linear_]svm output from obs using current HMM\n",N_SVM_TEST);
    printf("%s <dstsvm> \t\t- obtains svm from pos/neg linear models\n",N_LINEAR_SVM_TRAIN);
    printf("\n[SYSTEM]\n");
    printf("%s <filename>\t- load and execute a script\n",N_EXEC);
    printf("%s\t- exit genfinder\n",N_QUIT);
    printf("%s\t- exit genfinder\n",N_EXIT);
    printf("%s\t- this message\n",N_HELP);
    printf("%s <commands>\t- execute system functions \n",N_SYSTEM);
    fflush(stdout);
}


static bool prompt(FILE* infile=stdin)
{
    int i;
    char input[2000];

    printf("genefinder >> ");fflush(stdout);
    char* b=fgets(input, sizeof(input), infile);
    input[strlen(input)-1]='\0';
    if (!strlen(input) || (b==NULL))
	return true;
    if (infile!=stdin)
	printf("%s\n",input) ;

    if (!strncmp(input, N_LOAD_MODEL, strlen(N_LOAD_MODEL)))
    {
	for (i=strlen(N_LOAD_MODEL); isspace(input[i]); i++);
	if (lambda)
	    delete lambda;
	if (lambda_train)
	    delete lambda_train;
	lambda=NULL ;
	lambda_train=NULL ;

	FILE* model_file=fopen(&input[i], "r");

	if (model_file)
	{
	    lambda=new CHMM(model_file,PSEUDO);
	    rewind(model_file);
	    lambda_train= new CHMM(model_file,PSEUDO);
	    rewind(model_file);

	    if ((lambda) && (lambda_train) && 
		    (lambda->get_status()) && (lambda_train->get_status()))
		printf("file successfully read\n");

		ORDER=lambda->get_ORDER();
		M=lambda->get_M();
	    fclose(model_file);
	}
	else
	    printf("opening file %s failed\n", &input[i]);
    } 
    else if (!strncmp(input, N_SET_NEG_MODEL, strlen(N_SET_NEG_MODEL)))
    {
	if (lambda)
	{
	    if (neg)
		delete neg;

	    neg=lambda;
		neg->set_observations(NULL);
		delete lambda_train;
	    
	    lambda=NULL;
	    lambda_train=NULL;
	}
	else
	    printf("create model first\n");
    } 
    else if (!strncmp(input, N_SET_POS_MODEL, strlen(N_SET_POS_MODEL)))
    {
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
	    printf("create model first\n");
    } 
    else if (!strncmp(input, N_SAVE_MODEL_BIN, strlen(N_SAVE_MODEL_BIN)))
    {
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
	    printf("create model first\n");
    } 
    else if (!strncmp(input, N_CHOP, strlen(N_CHOP)))
    {
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
	    printf("see help for parameters/create model first\n");
    } 
    else if (!strncmp(input, N_SAVE_MODEL, strlen(N_SAVE_MODEL)))
    {
	for (i=strlen(N_SAVE_MODEL); isspace(input[i]); i++);

	if (lambda)
	{
	    FILE* file=fopen(&input[i], "w");

	    if ((!file) ||	(!lambda->save_model(file)))
		printf("writing to file %s failed!\n", &input[i]);
	    else
		printf("successfully written model into \"%s\" !\n", &input[i]);
	    if (file)
		fclose(file);
	}
	else
	    printf("create model first\n");
    } 
    else if (!strncmp(input, N_LOAD_DEFINITIONS, strlen(N_LOAD_DEFINITIONS)))
    {
	for (i=strlen(N_LOAD_DEFINITIONS); isspace(input[i]); i++);

	if ((!lambda) || (!lambda_train)) 
	    printf("load or create model first\n");
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
		    printf("file successfully read\n");

		fclose(def_file);
	    }
	    else
		printf("opening file %s failed\n", file_name);
	}
    } 
    else if (!strncmp(input, N_ASSIGN_OBSERVATION, strlen(N_ASSIGN_OBSERVATION)))
    {
	for (i=strlen(N_ASSIGN_OBSERVATION); isspace(input[i]); i++);
	char target[1024];

	if ((sscanf(&input[i], "%s", target))==1)
	{
		if (lambda && lambda_train)
		{
			if (strcmp(target,"POSTRAIN")==0)
			{
				lambda->set_observations(obs_postrain);
				lambda_train->set_observations(obs_postrain);//,lambda);
			}
			else if (strcmp(target,"NEGTRAIN")==0)
			{
				lambda->set_observations(obs_negtrain);
				lambda_train->set_observations(obs_negtrain);//,lambda);
			}
			else if (strcmp(target,"POSTEST")==0)
			{
				lambda->set_observations(obs_postest);
				lambda_train->set_observations(obs_postest);//,lambda);
			}
			else if (strcmp(target,"NEGTEST")==0)
			{
				lambda->set_observations(obs_negtest);
				lambda_train->set_observations(obs_negtest);//,lambda);
			}
			else if (strcmp(target,"TEST")==0)
			{
				lambda->set_observations(obs_test);
				lambda_train->set_observations(obs_test);//,lambda);
			}
			else
				printf("target POSTRAIN|NEGTRAIN|POSTEST|NEGTEST|TEST missing\n");
		}
		else
			printf("create model first!\n");
	}
	else
		printf("target POSTRAIN|NEGTRAIN|POSTEST|NEGTEST|TEST missing\n");
    }
    else if (!strncmp(input, N_LOAD_OBSERVATIONS, strlen(N_LOAD_OBSERVATIONS)))
    {
	for (i=strlen(N_LOAD_OBSERVATIONS); isspace(input[i]); i++);
	char filename[1024];
	char target[1024];

	if ((sscanf(&input[i], "%s %s", filename, target))==2)
	{
	    FILE* trn_file=fopen(filename, "r");

	    if (trn_file)
	    {
		if (strcmp(target,"POSTRAIN")==0)
		{
			delete obs_postrain;
		    obs_postrain= new CObservation(trn_file, POSTRAIN, alphabet, 8*sizeof(T_OBSERVATIONS), M, ORDER);
		}
		else if (strcmp(target,"NEGTRAIN")==0)
		{
			delete obs_negtrain;
			obs_negtrain= new CObservation(trn_file, NEGTRAIN, alphabet, 8*sizeof(T_OBSERVATIONS), M, ORDER);
		}
		else if (strcmp(target,"POSTEST")==0)
		{
			delete obs_postest;
			obs_postest= new CObservation(trn_file, POSTEST, alphabet, 8*sizeof(T_OBSERVATIONS), M, ORDER);
		}
		else if (strcmp(target,"NEGTEST")==0)
		{
			delete obs_negtest;
			obs_negtest= new CObservation(trn_file, NEGTEST, alphabet, 8*sizeof(T_OBSERVATIONS), M, ORDER);
		}
		else if (strcmp(target,"TEST")==0)
		{
			delete obs_test;
			obs_test= new CObservation(trn_file, TEST, alphabet, 8*sizeof(T_OBSERVATIONS), M, ORDER);
		}
		else
		    printf("target POSTRAIN|NEGTRAIN|POSTEST|NEGTEST|TEST missing\n");
	
		fclose(trn_file);
	    }
	    else
		printf("opening file %s failed\n", filename);
	}
	else
	    printf("target POSTRAIN|NEGTRAIN|POSTEST|NEGTEST|TEST missing\n");
    }
#ifndef NOVIT
    else if (!strncmp(input, N_SAVE_PATH, strlen(N_SAVE_PATH)))
    {
	for (i=strlen(N_SAVE_PATH); isspace(input[i]); i++);

	FILE* file=fopen(&input[i], "w");

	if (file)
	{
	    lambda->save_path(file);
	    fclose(file);
	}
	else
	    printf("opening file %s for writing failed\n", &input[i]);
    } 
#endif // NOVIT
    else if (!strncmp(input, N_SAVE_LIKELIHOOD_BIN, strlen(N_SAVE_LIKELIHOOD_BIN)))
    {
	for (i=strlen(N_SAVE_LIKELIHOOD_BIN); isspace(input[i]); i++);

	FILE* file=fopen(&input[i], "w");

	if (file)
	{
	    lambda->save_likelihood_bin(file);
	    fclose(file);
	}
	else
	    printf("opening file %s for writing failed\n", &input[i]);
    } 
    else if (!strncmp(input, N_SAVE_LIKELIHOOD, strlen(N_SAVE_LIKELIHOOD)))
    {
	for (i=strlen(N_SAVE_LIKELIHOOD); isspace(input[i]); i++);

	FILE* file=fopen(&input[i], "w");

	if (file)
	{
	    lambda->save_likelihood(file);
	    fclose(file);
	}
	else
	    printf("opening file %s for writing failed\n", &input[i]);
    } 
#ifndef NOVIT
    else if (!strncmp(input, N_SAVE_PATH_DERIVATIVES_BIN, strlen(N_SAVE_PATH_DERIVATIVES_BIN)))
    {
	int j;

	for (i=strlen(N_SAVE_PATH_DERIVATIVES_BIN); isspace(input[i]); i++);
	for (j=i; j<(int)strlen(input) && !isspace(input[j]); j++);
	input[j]='\0';
	FILE* file=fopen(&input[i], "w");
	if (file)
	{
	    lambda->save_path_derivatives_bin(file);
	    fclose(file);
	    printf("successfully written vit_derivatives into \"%s\" !\n", &input[i]);
	}
	else
	    printf("opening file %s for writing failed\n", &input[i]);
    } 
    else if (!strncmp(input, N_SAVE_PATH_DERIVATIVES, strlen(N_SAVE_PATH_DERIVATIVES)))
    {
	int j;
	for (i=strlen(N_SAVE_PATH_DERIVATIVES); isspace(input[i]); i++);
	for (j=i; j<(int)strlen(input) && !isspace(input[j]); j++);
	input[j]='\0';
	FILE* file=fopen(&input[i], "w");
	if (file)
	{
	    lambda->save_path_derivatives(file);
	    fclose(file);
	    printf("successfully written vit_derivatives into \"%s\" !\n", &input[i]);
	} 
	else
	    printf("opening file %s for writing failed\n", &input[i]);
    } 
#endif // NOVIT
    else if (!strncmp(input, N_SAVE_MODEL_DERIVATIVES_BIN, strlen(N_SAVE_MODEL_DERIVATIVES_BIN)))
    {
	int j;
	for (i=strlen(N_SAVE_MODEL_DERIVATIVES_BIN); isspace(input[i]); i++);
	for (j=i; j<(int)strlen(input) && !isspace(input[j]); j++);
	input[j]='\0';
	FILE* file=fopen(&input[i], "w");

	if (file)
	{
	    lambda->save_model_derivatives_bin(file);
	    fclose(file);
	    printf("successfully written bw_derivatives into \"%s\" !\n", &input[i]);
	} 
	else
	    printf("opening file %s for writing failed\n", &input[i]);
    } 
    else if (!strncmp(input, N_SAVE_MODEL_DERIVATIVES, strlen(N_SAVE_MODEL_DERIVATIVES)))
    {
	int j;
	for (i=strlen(N_SAVE_MODEL_DERIVATIVES); isspace(input[i]); i++);
	for (j=i; j<(int)strlen(input) && !isspace(input[j]); j++);
	input[j]='\0';
	FILE* file=fopen(&input[i], "w");
	if (file)
	{
	    lambda->save_model_derivatives(file);
	    fclose(file);
	    printf("successfully written bw_derivatives into \"%s\" !\n", &input[i]);
	}
	else
	    printf("opening file %s for writing failed\n", &input[i]);
    } 
#ifdef FIX_POS
    else if (!strncmp(input, N_FIX_POS_STATE, strlen(N_FIX_POS_STATE)))
    {
	for (i=strlen(N_FIX_POS_STATE); isspace(input[i]); i++);

	int pos,state,value;
	if (sscanf(&input[i], "%d %d %d", &pos, &state, &value) == 3)
	{
	    if ((lambda) && (lambda_train))
	    {
		bool ok=lambda->set_fix_pos_state(pos,state,value) ;
		ok= ok && lambda_train->set_fix_pos_state(pos,state,value) ;
		if (!ok)
		    printf("%s failed\n",N_FIX_POS_STATE);
	    }
	    else
		printf("create model first!\n");
	}
	else
	    printf("see help for parameters\n");
    } 
#endif
    else if (!strncmp(input, N_SET_MAX_DIM, strlen(N_SET_MAX_DIM)))
    {
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
	    printf("see help for parameters\n");
    } 
    else if (!strncmp(input, N_CLEAR, strlen(N_CLEAR)))
    {
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
    } 
    else if (!strncmp(input, N_NEW, strlen(N_NEW)))
    {

	for (i=strlen(N_NEW); isspace(input[i]); i++);

	int n,m,order;
	if (sscanf(&input[i], "%d %d %d", &n, &m, &order) == 3)
	{
	    delete lambda;
	    delete lambda_train;
	    lambda=new CHMM(n,m,order,NULL,PSEUDO);
	    lambda_train=new CHMM(n,m,order,NULL,PSEUDO);
		ORDER=order;
		M=m;
	}
	else
	    printf("see help for parameters\n");
    } 
    else if (!strncmp(input, N_PSEUDO, strlen(N_PSEUDO)))
    {
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
			PSEUDO,lambda->get_pseudo(),lambda_train->get_pseudo());
	    else
		printf("see help for parameters. current setting: pseudo=%e\n", PSEUDO);

    } 
    else if (!strncmp(input, N_ALPHABET, strlen(N_ALPHABET)))
    {
	for (i=strlen(N_ALPHABET); isspace(input[i]); i++);
	alphabet= DNA;

	char obs_type[1024];
	char target[1024];

	sscanf(&input[i], "%s %s", obs_type, target);
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

	if (obs_type[0]=='P' || obs_type[0]=='D' || obs_type[0]=='A')
	{
	    if (obs_type[0]=='P')
		alphabet=PROTEIN;
	    if (obs_type[0]=='A')
		alphabet=ALPHANUM;
	    else if (obs_type[0]=='D')
		alphabet=DNA;
	}
	else
	{
	    if (obs)
			printf("see help for parameters. current setting: alphabet=%s\n",(obs->get_alphabet()==0) ? "DNA":(obs->get_alphabet()==1 ? "PROTEIN":"ALPHANUM"));		    
	}
    } 
    else if (!strncmp(input, N_CONVERGENCE_CRITERIA, strlen(N_CONVERGENCE_CRITERIA)))
    {
	int j=100;
	double f=0.001;

	for (i=strlen(N_CONVERGENCE_CRITERIA); isspace(input[i]); i++);

	if (sscanf(&input[i], "%d %le", &j, &f) == 2)
	{
	    ITERATIONS=j;
	    EPSILON=f;
	}
	else
	    printf("see help for parameters. current setting: iterations=%i, epsilon=%e\n",ITERATIONS,EPSILON);
    } 
#ifndef NOVIT
    else if (!strncmp(input, N_VITERBI_TRAIN_DEFINED_ANNEALED, strlen(N_VITERBI_TRAIN_DEFINED_ANNEALED)))
    {
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
	    printf("create model first\n");
    } 
    else if (!strncmp(input, N_VITERBI_TRAIN_DEFINED_ADDIABATIC, strlen(N_VITERBI_TRAIN_DEFINED_ADDIABATIC)))
    {
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
	    printf("pseudo=%e  \n",act_pseudo) ; 
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
		    printf("   pseudo=%e",act_pseudo) ; 
		} ;
	    }
	}
	else
	    printf("create model first\n");
    } 
    else if (!strncmp(input, N_VITERBI_TRAIN_DEFINED, strlen(N_VITERBI_TRAIN_DEFINED)))
    {
	if ((lambda) && (lambda_train)) 
	{
#ifdef TMP_SAVE
	    char templname[30]="/tmp/vit_def_model_XXXXXX" ;
	    mkstemp(templname);
#endif
	    double prob=0.0, prob_train=0.0 ;
	    iteration_count=ITERATIONS ;
	    while (!converge(prob, prob_train))
	    {
		switch_model(&lambda, &lambda_train);
		prob=prob_train ;
		lambda->estimate_model_viterbi_defined(lambda_train);
		prob_train=lambda_train->best_path(-1) ;
#ifdef TMP_SAVE
		FILE* file=fopen(templname, "w");
		if (prob>prob_train)
		{
		    printf("\nsaving model with filename %s ... ", templname) ;
		    lambda->save_model(file) ;
		    fclose(file) ;
		    printf("done.") ;
		}
		else
		    printf("\nskipping TMP_SAVE. model got worse.");
#endif
	    }
	}
	else
	    printf("create model first\n");
    } 
    else if (!strncmp(input, N_VITERBI_TRAIN, strlen(N_VITERBI_TRAIN)))
    {
	if ((lambda) && (lambda_train)) 
	{
#ifdef TMP_SAVE
	    char templname[30]="/tmp/vit_model_XXXXXX" ;
	    mkstemp(templname);
#endif
	    double prob=0.0,prob_train=0.0 ;
	    iteration_count=ITERATIONS ;
	    while (!converge(prob, prob_train)) 
	    {
		switch_model(&lambda, &lambda_train);
		prob=prob_train ;
		lambda->estimate_model_viterbi(lambda_train);
		prob_train=lambda_train->best_path(-1) ;
#ifdef TMP_SAVE
		FILE* file=fopen(templname, "w");
		if (prob>prob_train)
		{
		    printf("\nsaving model with filename %s ... ", templname) ;
		    lambda->save_model(file) ;
		    fclose(file) ;
		    printf("done.") ;
		}
		else
		    printf("\nskipping TMP_SAVE. model got worse.");
#endif
	    }
	}
	else
	    printf("create model first\n");
    }
#endif // NOVIT
    else if (!strncmp(input, N_BAUM_WELCH_TRAIN_DEFINED, strlen(N_BAUM_WELCH_TRAIN_DEFINED)))
    {
#ifdef TMP_SAVE
	char templname[30]="/tmp/bw_def_model_XXXXXX" ;
	mkstemp(templname);
	char templname_best[40] ;
	sprintf(templname_best, "%s_best", templname) ;
	double prob_max=-CMath::INFTY ;
#endif
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
#ifdef TMP_SAVE
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
#endif
	    }
		}
		else
			printf("assign observation first\n");

	}
	else
	    printf("create model first\n");
    } 
    else if (!strncmp(input, N_BAUM_WELCH_TRAIN, strlen(N_BAUM_WELCH_TRAIN)))
    {
#ifdef TMP_SAVE
	char templname[30]="/tmp/bw_model_XXXXXX" ;
	mkstemp(templname);
	char templname_best[40] ;
	sprintf(templname_best, "%s_best", templname) ;
	double prob_max=-CMath::INFTY ;
#endif
	iteration_count=ITERATIONS ;
	if ((lambda) && (lambda_train)) 
	{
		if (lambda->get_observations() && lambda_train->get_observations())
		{
			double prob_train=math.ALMOST_NEG_INFTY, prob = -math.INFTY ;

			while (!converge(prob,prob_train))
			{
			switch_model(&lambda, &lambda_train);
			prob=prob_train ;
			lambda->estimate_model_baum_welch(lambda_train);
			prob_train=lambda_train->model_probability();
	#ifdef TMP_SAVE
			if (prob_max<prob_train)
			{
				prob_max=prob_train ;
				FILE* file=fopen(templname_best, "w");
				printf("\nsaving best model with filename %s ... ", templname_best) ;
				lambda->save_model(file) ;
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
			} ;
	#endif
			}
		}
		else
			printf("assign observation first\n");
	}
	else
	    printf("create model first\n");
    } 
#ifndef NOVIT
    else if (!strncmp(input, N_BEST_PATH, strlen(N_BEST_PATH)))
    {
	if ((lambda) && (lambda_train)) 
	{
	    lambda->output_model_sequence(false);
	}
	else
	    printf("create model first\n");
    } 
#endif //NOVIT
    else if (!strncmp(input, N_LIKELIHOOD, strlen(N_LIKELIHOOD)))
    {
	if ((lambda) && (lambda_train)) 
	{
	    lambda->output_model(false);
	}
	else
	    printf("create model first\n");
    } 
    else if (!strncmp(input, N_OUTPUT_MODEL_DEFINED, strlen(N_OUTPUT_MODEL_DEFINED)))
    {
	if (lambda)
	{
	    lambda->output_model_defined(true);
	}
	else
	    printf("create model first\n");
    } 
#ifndef NOVIT
    else if (!strncmp(input, N_OUTPUT_PATH, strlen(N_OUTPUT_PATH)))
    {
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
	    printf("create model first\n");
    } 
    else if (!strncmp(input, N_OUTPUT_GENES, strlen(N_OUTPUT_GENES)))
    {
	if (lambda)
	{
	    lambda->output_gene_positions(true);
	}
	else
	    printf("create model first\n");
    } 
#endif //NOVIT
    else if (!strncmp(input, N_OUTPUT_MODEL, strlen(N_OUTPUT_MODEL)))
    {
	if (lambda)
	{
	    lambda->output_model(true);
	    lambda_train->output_model(true);
	}
	else
	    printf("create model first\n");
    } 
    else if (!strncmp(input, N_EXEC, strlen(N_EXEC)))
    {
	for (i=strlen(N_EXEC); isspace(input[i]); i++);

	FILE* file=fopen(&input[i], "r");

	if (!file)
	{
	    printf("error opening/reading file: \"%s\"",&input[i]);
	    return true;
	}
	else
	{
	    while(!feof(file) && prompt(file));
	    fclose(file);
	}

	return true;
    } 
    else if (!strncmp(input, N_EXIT, strlen(N_EXIT)))
    {
	return false;
    } 
    else if (!strncmp(input, N_QUIT, strlen(N_QUIT)))
    {
	return false;
    } 
#ifdef DEBUG
    else if (!strncmp(input, N_TEST, strlen(N_TEST)))
    {
	lambda->check_path_derivatives() ;
    } 
#endif // DEBUG
    else if (!strncmp(input, N_HELP, strlen(N_HELP)))
    {
	help();
    }
    else if (!strncmp(input, N_SYSTEM, strlen(N_SYSTEM)))
    {
	for (i=strlen(N_SYSTEM); isspace(input[i]); i++);
	system(&input[i]);
    } 
    else if (!strncmp(input, N_LINEAR_TRAIN, strlen(N_LINEAR_TRAIN)))
    {
	for (i=strlen(N_LINEAR_TRAIN); isspace(input[i]); i++);

	int WIDTH=-1,UPTO=-1;
	char fname[1024];

	sscanf(&input[i], "%s %d %d", fname, &WIDTH, &UPTO);
	
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
	    }

	    if (WIDTH >0 && UPTO >0)
	    {	  

		alphabet= DNA;
		ORDER=1;
		M=4;

		CObservation* obs=new CObservation(TRAIN, alphabet, 8*sizeof(T_OBSERVATIONS), M, ORDER);

		if (lambda && obs)
		{
		    alphabet=obs->get_alphabet();
		    ORDER=lambda->get_ORDER();
		    delete(lambda_train);
		    delete(lambda);
		}

		switch (alphabet)
		{
		    case DNA:
			M=4;
			break;
		    case PROTEIN:
			M=26;
			break;
		    case ALPHANUM:
			M=36;
			break;
		    default:
			M=4;
			break;
		};


		lambda=new CHMM(UPTO,M,ORDER,NULL,PSEUDO);
		lambda_train=new CHMM(UPTO,M,ORDER,NULL,PSEUDO);

		lambda->set_observation_nocache(obs);
		lambda_train->set_observation_nocache(obs);

		if (lambda && lambda_train)
		{
		    lambda->linear_train(file, WIDTH, UPTO);
		    printf("done.\n");
		}
		else
		    printf("model creation failed\n");
	    }

	    fclose(file);
	}
	else
	    printf("opening file %s failed!\n", fname);

    } 
    else if (!strncmp(input, N_LINEAR_LIKELIHOOD, strlen(N_LINEAR_LIKELIHOOD)))
    {
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
			printf("log(Pr[O|model])=%e, #states: %i, #observation symbols: %i\n", 
				(double)lambda->linear_likelihood(file, WIDTH, UPTO), lambda->get_N(), lambda->get_M());
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
	    printf("create model first!\n");

    } 
    else if (!strncmp(input, N_SAVE_LINEAR_LIKELIHOOD_BIN, strlen(N_SAVE_LINEAR_LIKELIHOOD_BIN)))
    {

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
			lambda->save_linear_likelihood_bin(srcfile, dstfile, WIDTH, UPTO);
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
    } 
    else if (!strncmp(input, N_SAVE_LINEAR_LIKELIHOOD, strlen(N_SAVE_LINEAR_LIKELIHOOD)))
    {

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
				lambda->save_linear_likelihood(srcfile, dstfile, WIDTH, UPTO);
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
    } 
    else if (!strncmp(input, N_SVM_TRAIN, strlen(N_SVM_TRAIN)))
    {
	char name[1024];

	for (i=strlen(N_SVM_TRAIN); isspace(input[i]); i++);
	if (sscanf(&input[i], "%s", name) == 1)
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

		    theta=new double[pos->get_N()*(pos->get_N()+2)+pos->get_N()*pos->get_M() + neg->get_N()*(neg->get_N()+2)+neg->get_N()*neg->get_M()];
		    svm.svm_train(name,obs, 4);
		    delete theta;
		    theta=NULL;

		    pos->set_observations(old_pos);
		    neg->set_observations(old_neg);

		    delete obs;
		}
		else
		    printf("assign postrain and negtrain observations first!\n");
	    }
	    else
		printf("assign positive and negative models first!\n");
	}
	else
	    printf("see help for parameters\n");
    } 
    else if (!strncmp(input, N_LINEAR_SVM_TRAIN, strlen(N_LINEAR_SVM_TRAIN)))
    {
	char name[1024];

	for (i=strlen(N_LINEAR_SVM_TRAIN); isspace(input[i]); i++);
	if (sscanf(&input[i], "%s", name) == 1)
	{
	    if (pos && neg)
	    {
		if (obs_postrain && obs_negtrain)
		{
		    CObservation* obs=new CObservation(obs_postrain, obs_negtrain);

		    CObservation* old_pos=pos->get_observations();
		    CObservation* old_neg=neg->get_observations();

		    pos->set_observation_nocache(obs);
		    neg->set_observation_nocache(obs);

		    theta=new double[pos->get_N()*pos->get_M() + neg->get_N()*neg->get_M()];
		    svm.svm_train(name,obs, 5);
		    delete theta;
		    theta=NULL;

		    pos->set_observations(old_pos);
		    neg->set_observations(old_neg);

		    delete obs;
		}
		else
		    printf("assign postrain and negtrain observations first!\n");
	    }
	    else
		printf("assign positive and negative models first!\n");
	}
	else
	    printf("see help for parameters\n");
    } 
    else if (!strncmp(input, N_SVM_TEST, strlen(N_SVM_TEST)))
    {
	char svmname[1024];
	char outputname[1024];
	FILE* outputfile=stdout;
	int numargs=-1;

	for (i=strlen(N_SVM_TEST); isspace(input[i]); i++);
	numargs=sscanf(&input[i], "%s %s", svmname, outputname);
	if (numargs >= 1)
	{
	    FILE* svm_file=fopen(svmname, "r");
	    if (svm_file)
	    {
		if (numargs==2)
		{
		    outputfile=fopen(outputname, "w");

		    if (!outputfile)
		    {
			fprintf(stderr,"ERROR: could not open %s\n",outputname);
			return false;
		    }
		}

		if (pos && neg)
		{
		    if (obs_postest && obs_negtest)
		    {
			CObservation* obs=new CObservation(obs_postest, obs_negtest);
			svm.load_svm(svm_file, obs);

			CObservation* old_pos=pos->get_observations();
			CObservation* old_neg=neg->get_observations();

			pos->set_observations(obs);
			neg->set_observations(obs,pos);

			theta=new double[pos->get_N()*(pos->get_N()+2)+pos->get_N()*pos->get_M() + neg->get_N()*(neg->get_N()+2)+neg->get_N()*neg->get_M()];
			svm.svm_test(obs, outputfile);
			delete theta;
			theta=NULL;

			pos->set_observations(old_pos);
			neg->set_observations(old_neg);

			delete obs;
		    }
		    else
			printf("assign postrain and negtrain observations first!\n");
		}
		else
		    printf("assign positive and negative models first!\n");
	    }
	    else
		printf("see help for parameters\n");
	    fclose(svm_file);
	}
	else
	    printf("could not open svm model\n");
    } 
    else if (!strncmp(input, N_LINEAR_HMM_TEST, strlen(N_LINEAR_HMM_TEST)))
    {
	char posname[1024];
	char negname[1024];
	
	int WIDTH=-1,UPTO=-1;

	for (i=strlen(N_LINEAR_HMM_TEST); isspace(input[i]); i++);

	if (sscanf(&input[i], "%s %s %d %d", negname, posname,&WIDTH,&UPTO) >= 2)
	{
	    if (pos && neg)
	    {
		FILE* posfile=fopen(posname, "r");
		FILE* negfile=fopen(negname, "r");

		if (posfile && negfile)
		{
		    printf("opened %s and %s\n",posname,negname);
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
				    printf("detected WIDTH=%d UPTO=%d\n",WIDTH, UPTO);
				    fflush(stdout);
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
				fprintf(stderr,"ERROR: file has wrong size");
				return false;
			    }
			    	    
			    int possize=posfsize/WIDTH;
			    int negsize=negfsize/WIDTH;
			    int total=possize+negsize;

			    printf("p:%d,n:%d,t:%d\n",possize,negsize,total);
			    double *output = new double[total];	
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
				}
				else
				{
				    int fileptr=ftell(posfile);
				    output[dim]=pos->linear_likelihood(posfile, WIDTH, UPTO,true);
				    fseek(posfile, fileptr, SEEK_SET);
				    output[dim]-=neg->linear_likelihood(posfile, WIDTH, UPTO,true);
				    label[dim]=+1;
				}
			    }

			    double *fp= new double[total];	
			    double *tp= new double[total];	

			    int pointeven=math.calcroc(fp, tp, output, label, total, possize, negsize);

			    double correct=possize*tp[pointeven]+(1-fp[pointeven])*negsize;
			    double fpo=fp[pointeven]*negsize;
			    double fne=(1-tp[pointeven])*possize;

			    printf("classified:\n");
			    printf("\tcorrect:%i\n", int (correct));
			    printf("\twrong:%i (fp:%i,fn:%i)\n", int(fpo+fne), int (fpo), int (fne));
			    printf("of %i samples (c:%f,w:%f,fp:%f,tp:%f)\n",total, correct/total, 1-correct/total, fp[pointeven], tp[pointeven]);
			    delete[] fp;
			    delete[] tp;
			    delete[] output;
			    delete[] label;

			}
			else
			    printf("model has wrong size\n");
		    }

		}
		else
		    printf("assign postrain and negtrain observations first!\n");
		
		if (posfile)
		    fclose(posfile);
		if (negfile)
		    fclose(negfile);
	    }
	    else
		printf("assign positive and negative models first!\n");
	}
	else
	    printf("see help for parameters\n");
    } 
    else if (!strncmp(input, N_HMM_TEST, strlen(N_HMM_TEST)))
    {
	//char name[1024];

	for (i=strlen(N_HMM_TEST); isspace(input[i]); i++);
	//if (sscanf(&input[i], "%s", name) == 1)
	//{
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

		    double *output = new double[total];	
		    int* label= new int[total];	

		    for (int dim=0; dim<total; dim++)
		    {
			output[dim]=pos->model_probability(dim)-neg->model_probability(dim);
			label[dim]=obs->get_label(dim);
		    }

		    double *fp= new double[total];	
		    double *tp= new double[total];	

		    for (int dim=0; dim<total; dim++)
		    {
			output[dim]=pos->model_probability(dim)-neg->model_probability(dim);
			label[dim]= obs->get_label(dim);
		    }
		 
		    int possize,negsize;
		    int pointeven=math.calcroc(fp, tp, output, label, total, possize, negsize);

		    double correct=possize*tp[pointeven]+(1-fp[pointeven])*negsize;
		    double fpo=fp[pointeven]*negsize;
		    double fne=(1-tp[pointeven])*possize;

		    printf("classified:\n");
		    printf("\tcorrect:%i\n", int (correct));
		    printf("\twrong:%i (fp:%i,fn:%i)\n", int(fpo+fne), int (fpo), int (fne));
		    printf("of %i samples (c:%f,w:%f,fp:%f,tp:%f)\n",total, correct/total, 1-correct/total, fp[pointeven], tp[pointeven]);

		    delete[] fp;
		    delete[] tp;
		    delete[] output;
		    delete[] label;

		    pos->set_observations(old_pos);
		    neg->set_observations(old_neg);

		    delete obs;
		}
		else
		    printf("assign postrain and negtrain observations first!\n");
	    }
	    else
		printf("assign positive and negative models first!\n");
	//}
	//else
	//    printf("see help for parameters\n");
    } 
    else
	printf("unrecognized command. type help for options\n");

    return true;
}
//------------------------------------------------------------------------------------//

int main(int argc, char* argv[])
{
    initialize() ;
    if (argc<=1)
	while (prompt());
    else
    {
	if (argc>=2)
	{
	    if ( argc>2 || !strcmp(argv[1], "-h") || !strcmp(argv[1], "/?") || !strcmp(argv[1], "--help"))
	    {
		printf("usage: genfinder [ <script> ]\n\n");
		printf("if no options are given genfinder enters interactive mode\n");
		printf("if <script> is specified the commands will be executed");
		return 1;
	    }
	    else
	    {
		FILE* file=fopen(argv[1], "r");

		if (!file)
		{
		    printf("error opening/reading file: \"%s\"",argv[1]);
		    return 1;
		}
		else
		{
		    while(!feof(file) && prompt(file));
		    fclose(file);
		}
	    }
	}
    }
    cleanup() ;
    return 0;
}
