#include "gui/TextGUI.h"
#include "lib/io.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

CTextGUI gui;

//names of menu commands
static const char* N_SET_TEST_MODEL=		"set_test_model";
static const char* N_SET_POS_MODEL=			"set_pos_model";
static const char* N_SET_NEG_MODEL=			"set_neg_model";
static const char* N_LOAD_MODEL=			"load_model";
static const char* N_SAVE_MODEL=			"save_model";
static const char* N_SAVE_MODEL_BIN=		"save_model_bin";
static const char* N_LOAD_DEFINITIONS=		"load_defs";
static const char* N_SAVE_KERNEL=			"save_kernel";
static const char* N_SAVE_TOP_FEATURES=		"save_top_features";
#ifndef NOVIT
static const char* N_SAVE_PATH=						"save_path";
static const char* N_SAVE_PATH_DERIVATIVES=	        "save_vit_deriv";
static const char* N_SAVE_PATH_DERIVATIVES_BIN=	    "save_vit_deriv_bin";
static const char* N_BEST_PATH=			        	"best_path";
static const char* N_OUTPUT_PATH=					"output_path";
static const char* N_OUTPUT_GENES=					"output_genes";
static const char* N_VITERBI_TRAIN=		        	"vit";
static const char* N_VITERBI_TRAIN_DEFINED=             "vit_def";
static const char* N_VITERBI_TRAIN_DEFINED_ANNEALED=    "vit_def_ann";
static const char* N_VITERBI_TRAIN_DEFINED_ADDIABATIC=  "vit_def_add";
#endif // NOVIT
static const char* N_LINEAR_TRAIN=					"linear_train";
static const char* N_LINEAR_LIKELIHOOD=				"linear_likelihood";
static const char* N_SAVE_LINEAR_LIKELIHOOD=		"save_linear_likelihood";
static const char* N_SAVE_LINEAR_LIKELIHOOD_BIN=	"save_linear_likelihood_bin";
static const char* N_SAVE_MODEL_DERIVATIVES=        "save_bw_deriv";
static const char* N_SAVE_MODEL_DERIVATIVES_BIN=    "save_bw_deriv_bin";
static const char* N_SAVE_LIKELIHOOD=               "save_likelihood";
static const char* N_SAVE_LIKELIHOOD_BIN=           "save_likelihood_bin";
static const char* N_LOAD_OBSERVATIONS=		        "load_obs";
static const char* N_ASSIGN_OBSERVATION=			"assign_obs";
static const char* N_NEW=							"new";
static const char* N_CLEAR=							"clear";
static const char* N_CHOP=							"chop";
static const char* N_CONVERGENCE_CRITERIA=	        "convergence_criteria";
static const char* N_PSEUDO=						"pseudo";
static const char* N_C=								"c";
static const char* N_ADD_STATES=					"add_states";
static const char* N_APPEND_MODEL=					"append_model";
static const char* N_BAUM_WELCH_TRAIN=		        "bw";
static const char* N_BAUM_WELCH_TRAIN_DEFINED=		"bw_def";
static const char* N_LIKELIHOOD=					"likelihood";
static const char* N_ALPHABET=			       	 	"alphabet";
static const char* N_OUTPUT_MODEL=					"output_model";
static const char* N_OUTPUT_MODEL_DEFINED=          "output_model_defined";
static const char* N_QUIT=				"quit";
static const char* N_EXEC=				"exec";
static const char* N_EXIT=				"exit";
static const char* N_HELP=				"help";
static const char* N_SYSTEM=				"!";
static const char N_COMMENT1=				'#';
static const char N_COMMENT2=				'%';
static const char* N_FIX_POS_STATE=			"fix_pos_state";
static const char* N_SET_MAX_DIM=			"max_dim";
static const char* N_TEST=				"test";
static const char* N_LINEAR_SVM_TRAIN=			"linear_svm_train";
static const char* N_SVM_TRAIN=					"svm_train";
static const char* N_SVM_TEST=					"svm_test";
static const char* N_SET_SVM_LIGHT=			 	"set_svm_light";
static const char* N_SET_SVM_CPLEX=			 	"set_svm_cplex";
static const char* N_ONE_CLASS_HMM_TEST=		"one_class_hmm_test";
static const char* N_ONE_CLASS_LINEAR_HMM_TEST=	"one_class_linear_hmm_test";
static const char* N_HMM_TEST=					"hmm_test";
static const char* N_LINEAR_HMM_TEST=			"linear_hmm_test";
static const char* N_SET_ORDER=					"set_order";


CTextGUI::CTextGUI()
{
    CIO::message("Learning uses %i threads\n", NUM_PARALLEL) ;
   
#ifdef SVMCPLEX
    libmmfileInitialize() ;
#endif
}

CTextGUI::~CTextGUI()
{
#ifdef SVMCPLEX
  libmmfileTerminate() ;
#endif
}

void CTextGUI::print_help()
{
   CIO::message("\n[LOAD]\n");
   CIO::message("%s <filename>\t- load hmm\n",N_LOAD_MODEL);
   CIO::message("%s <filename> [initialize=1]\t- load hmm defs\n",N_LOAD_DEFINITIONS);
   CIO::message("%s <filename>\t- load observed data\n",N_LOAD_OBSERVATIONS);
   CIO::message("\n[SAVE]\n");
   CIO::message("%s <filename>\t- save hmm\n",N_SAVE_MODEL);
   CIO::message("%s <filename>\t- save hmm in binary format\n",N_SAVE_MODEL_BIN);
#ifndef NOVIT
   CIO::message("%s <filename>\t- save state sequence of viterbi path\n",N_SAVE_PATH);
   CIO::message("%s <filename>\t- save derivatives of log P[O,Q_best|model]\n",N_SAVE_PATH_DERIVATIVES);
   CIO::message("%s <filename>\t- save derivatives of log P[O,Q_best|model] in binary format\n",N_SAVE_PATH_DERIVATIVES_BIN);
#endif // NOVIT
   CIO::message("%s <filename>\t- save log derivatives of P[O|model]\n",N_SAVE_MODEL_DERIVATIVES);
   CIO::message("%s <filename>\t- save log derivatives of P[O|model] in binary format\n",N_SAVE_MODEL_DERIVATIVES_BIN);
   CIO::message("%s <filename> <TOP|FK>\t- save kernel in binary format\n",N_SAVE_KERNEL);
   CIO::message("%s <filename>\t- save top features for all train obs,neg first\n",N_SAVE_TOP_FEATURES);
   CIO::message("%s <filename>\t- save P[O|model]\n",N_SAVE_LIKELIHOOD);
   CIO::message("%s <filename>\t- save P[O|model]\n",N_SAVE_LIKELIHOOD_BIN);
   CIO::message("%s <srcname> <destname> [<width> <upto>]\t\t- saves likelihood for linear model from file\n",N_SAVE_LINEAR_LIKELIHOOD);
   CIO::message("%s <srcname> <destname> [<width> <upto>]\t\t- saves likelihood for linear model from file\n",N_SAVE_LINEAR_LIKELIHOOD_BIN);
   CIO::message("\n[MODEL]\n");
   CIO::message("%s - frees all models and observations\n",N_CLEAR);
   CIO::message("%s #states #oberservations #order\t- frees previous model and creates an empty new one\n",N_NEW);
   CIO::message("%s <POSTRAIN|NEGTRAIN|POSTEST|NEGTEST|TEST> - assign observation to current model\n",N_ASSIGN_OBSERVATION);
   CIO::message("%s - make current model the test model; then free current model \n",N_SET_TEST_MODEL);
   CIO::message("%s - make current model the positive model; then free current model \n",N_SET_POS_MODEL);
   CIO::message("%s - make current model the negative model; then free current model \n",N_SET_NEG_MODEL);
   CIO::message("%s <value>\t\t\t- chops likelihood of all parameters 0<value<1\n", N_CHOP);
   CIO::message("%s <<num> [<value>]>\t\t\t- add num (def 1) states,initialize with value (def rnd)", N_ADD_STATES);
   CIO::message("%s <filename> <[ACGT][ACGT]>\t\t\t- append model <filename> to current model", N_ADD_STATES);
   CIO::message("%s [pseudovalue]\t\t\t- changes pseudo value\n", N_PSEUDO);
   CIO::message("%s <POSTRAIN|NEGTRAIN|POSTEST|NEGTEST|TEST> [PROTEIN|DNA|ALPHANUM|CUBE]\t\t\t- changes alphabet type\n", N_ALPHABET);
   CIO::message("%s [maxiterations] [maxallowedchange]\t- defines the convergence criteria for all train algorithms (%i,%e)\n",N_CONVERGENCE_CRITERIA,ITERATIONS,EPSILON);
#ifdef FIX_POS
   CIO::message("%s position state\t- sets the state which has to be passed at a certain position\n",N_FIX_POS_STATE);
#endif
   CIO::message("%s <max_dim>\t - set maximum number of patterns\n",N_SET_MAX_DIM);
   CIO::message("%s <ORDER>\t - set order of linear HMMs\n",N_SET_ORDER);
   CIO::message("\n[TRAIN]\n");
   CIO::message("%s <filename> [<width> <upto>]\t\t- obtains new linear model from file\n",N_LINEAR_TRAIN);
   CIO::message("%s <filename> [<width> <upto>]\t\t- computes likelihood for linear model from file\n",N_LINEAR_LIKELIHOOD);
#ifndef NOVIT
   CIO::message("%s\t\t- does viterbi training on the current model\n",N_VITERBI_TRAIN);
   CIO::message("%s\t\t- does viterbi training only on defined transitions etc\n",N_VITERBI_TRAIN_DEFINED);
   CIO::message("%s [pseudo_start [in_steps]]\t\t- does viterbi training only on defined transitions with annealing\n",N_VITERBI_TRAIN_DEFINED_ANNEALED);
   CIO::message("%s [pseudo_start [step [eps_add]]]\t\t- does viterbi training only on defined transitions with addiabatic annealing\n",N_VITERBI_TRAIN_DEFINED_ADDIABATIC);
#endif //NOVIT
   CIO::message("%s\t\t- does baum welch training on current model\n",N_BAUM_WELCH_TRAIN);
   CIO::message("%s\t\t- does baum welch training only on defined transitions etc.\n",N_BAUM_WELCH_TRAIN_DEFINED);
#ifndef NOVIT
   CIO::message("%s\t- find the best path using viterbi\n",N_BEST_PATH);
#endif //NOVIT
   CIO::message("%s\t- find model likelihood\n",N_LIKELIHOOD);
   CIO::message("%s [maxiterations] [maxallowedchange]\t- defines the convergence criteria for all train algorithms (%i,%e)\n",N_CONVERGENCE_CRITERIA,ITERATIONS,EPSILON);
   CIO::message("\n[OUTPUT]\n");
#ifndef NOVIT
   CIO::message("%s [from to]\t- outputs best path\n",N_OUTPUT_PATH);
#endif //NOVIT
   CIO::message("%s\t- output whole model\n",N_OUTPUT_MODEL);
   CIO::message("\n[HMM-classification]\n");
   CIO::message("%s[<treshhold> [<output> [<rocfile>]]]\t\t\t\t- calculate output from obs using test HMM\n",N_ONE_CLASS_HMM_TEST);
   CIO::message("%s[<output> [<rocfile>]]\t\t\t\t- calculate output from obs using current HMMs\n",N_HMM_TEST);
   CIO::message("%s <negtest> <postest> [<treshhold> [<output> [<rocfile>]]]\t\t\t\t- calculate output from obs using test HMM\n",N_ONE_CLASS_LINEAR_HMM_TEST);
   CIO::message("%s <negtest> <postest> [<output> [<rocfile> [<width> <upto>]]]\t- calculate hmm output from obs using linear model\n",N_LINEAR_HMM_TEST);
   CIO::message("\n[Hybrid HMM-<TOP-Kernel>-SVM]\n");
   CIO::message("%s [c-value]\t\t\t- changes svm_c value\n", N_C);
   CIO::message("%s <dstsvm>\t\t- obtains svm from POS/NEGTRAIN using pos/neg HMM\n",N_SVM_TRAIN);
   CIO::message("%s <srcsvm> [<output> [<rocfile>]]\t\t- calculate [linear_]svm output from obs using current HMM\n",N_SVM_TEST);
   CIO::message("%s <dstsvm> \t\t- obtains svm from pos/neg linear models\n",N_LINEAR_SVM_TRAIN);
   CIO::message("%s - enables SVM Light \n",N_SET_SVM_LIGHT);
#ifdef SVMCPLEX
   CIO::message("%s - enables SVM CPLEX \n",N_SET_SVM_CPLEX);
#endif
   CIO::message("\n[SYSTEM]\n");
   CIO::message("%s <filename>\t- load and execute a script\n",N_EXEC);
   CIO::message("%s\t- exit genfinder\n",N_QUIT);
   CIO::message("%s\t- exit genfinder\n",N_EXIT);
   CIO::message("%s\t- this message\n",N_HELP);
   CIO::message("%s <commands>\t- execute system functions \n",N_SYSTEM);
   
}

void CTextGUI::print_prompt()
{
   CIO::message("genefinder >> ");
}

bool CTextGUI::get_line(FILE* infile)
{
    int i;
    char input[2000];

	print_prompt();

    char* b=fgets(input, sizeof(input), infile);
    if ((b==NULL) || !strlen(input) || (input[0]==N_COMMENT1) || (input[0]==N_COMMENT2) || (input[0]=='\n'))
	return true;
    
    input[strlen(input)-1]='\0';
    if (infile!=stdin)
	printf("%s\n",input) ;

    if (!strncmp(input, N_LOAD_MODEL, strlen(N_LOAD_MODEL)))
    {
    } 
    else if (!strncmp(input, N_SET_NEG_MODEL, strlen(N_SET_NEG_MODEL)))
    {
	}
    else if (!strncmp(input, N_SET_TEST_MODEL, strlen(N_SET_TEST_MODEL)))
    {
    } 
    else if (!strncmp(input, N_SET_POS_MODEL, strlen(N_SET_POS_MODEL)))
    {
    } 
    else if (!strncmp(input, N_SAVE_MODEL_BIN, strlen(N_SAVE_MODEL_BIN)))
    {
    } 
    else if (!strncmp(input, N_CHOP, strlen(N_CHOP)))
    {
    } 
    else if (!strncmp(input, N_SAVE_MODEL, strlen(N_SAVE_MODEL)))
    {
    } 
    else if (!strncmp(input, N_LOAD_DEFINITIONS, strlen(N_LOAD_DEFINITIONS)))
    {
    } 
    else if (!strncmp(input, N_ASSIGN_OBSERVATION, strlen(N_ASSIGN_OBSERVATION)))
    {
    }
    else if (!strncmp(input, N_LOAD_OBSERVATIONS, strlen(N_LOAD_OBSERVATIONS)))
    {
    }
    else if (!strncmp(input, N_SAVE_PATH, strlen(N_SAVE_PATH)))
    {
	}
    else if (!strncmp(input, N_SAVE_LIKELIHOOD_BIN, strlen(N_SAVE_LIKELIHOOD_BIN)))
    {
    } 
    else if (!strncmp(input, N_SAVE_LIKELIHOOD, strlen(N_SAVE_LIKELIHOOD)))
    {
    } 
    else if (!strncmp(input, N_SAVE_TOP_FEATURES, strlen(N_SAVE_TOP_FEATURES)))
	{
	} 
    else if (!strncmp(input, N_SAVE_KERNEL, strlen(N_SAVE_KERNEL)))
	{
	} 
    else if (!strncmp(input, N_SAVE_PATH_DERIVATIVES_BIN, strlen(N_SAVE_PATH_DERIVATIVES_BIN)))
    {
    } 
    else if (!strncmp(input, N_SAVE_PATH_DERIVATIVES, strlen(N_SAVE_PATH_DERIVATIVES)))
    {
    } 
    else if (!strncmp(input, N_SAVE_MODEL_DERIVATIVES_BIN, strlen(N_SAVE_MODEL_DERIVATIVES_BIN)))
    {
    } 
    else if (!strncmp(input, N_SAVE_MODEL_DERIVATIVES, strlen(N_SAVE_MODEL_DERIVATIVES)))
    {
    } 
    else if (!strncmp(input, N_FIX_POS_STATE, strlen(N_FIX_POS_STATE)))
    {
    } 
    else if (!strncmp(input, N_SET_MAX_DIM, strlen(N_SET_MAX_DIM)))
    {
    } 
    else if (!strncmp(input, N_CLEAR, strlen(N_CLEAR)))
    {
    } 
    else if (!strncmp(input, N_NEW, strlen(N_NEW)))
    {
    } 
    else if (!strncmp(input, N_PSEUDO, strlen(N_PSEUDO)))
    {
    } 
    else if (!strncmp(input, N_ALPHABET, strlen(N_ALPHABET)))
    {
    } 
    else if (!strncmp(input, N_CONVERGENCE_CRITERIA, strlen(N_CONVERGENCE_CRITERIA)))
    {
    } 
    else if (!strncmp(input, N_VITERBI_TRAIN_DEFINED_ANNEALED, strlen(N_VITERBI_TRAIN_DEFINED_ANNEALED)))
    {
    } 
    else if (!strncmp(input, N_VITERBI_TRAIN_DEFINED_ADDIABATIC, strlen(N_VITERBI_TRAIN_DEFINED_ADDIABATIC)))
    {
    } 
    else if (!strncmp(input, N_VITERBI_TRAIN_DEFINED, strlen(N_VITERBI_TRAIN_DEFINED)))
    {
    } 
    else if (!strncmp(input, N_VITERBI_TRAIN, strlen(N_VITERBI_TRAIN)))
    {
    }
    else if (!strncmp(input, N_BAUM_WELCH_TRAIN_DEFINED, strlen(N_BAUM_WELCH_TRAIN_DEFINED)))
    {
    } 
    else if (!strncmp(input, N_BAUM_WELCH_TRAIN, strlen(N_BAUM_WELCH_TRAIN)))
    {
    } 
    else if (!strncmp(input, N_BEST_PATH, strlen(N_BEST_PATH)))
    {
    } 
    else if (!strncmp(input, N_LIKELIHOOD, strlen(N_LIKELIHOOD)))
    {
    } 
    else if (!strncmp(input, N_OUTPUT_MODEL_DEFINED, strlen(N_OUTPUT_MODEL_DEFINED)))
    {
    } 
    else if (!strncmp(input, N_OUTPUT_PATH, strlen(N_OUTPUT_PATH)))
    {
    } 
    else if (!strncmp(input, N_OUTPUT_GENES, strlen(N_OUTPUT_GENES)))
    {
    } 
    else if (!strncmp(input, N_OUTPUT_MODEL, strlen(N_OUTPUT_MODEL)))
    {
    } 
    else if (!strncmp(input, N_EXEC, strlen(N_EXEC)))
    {
    } 
    else if (!strncmp(input, N_EXIT, strlen(N_EXIT)))
    {
	return false;
    } 
    else if (!strncmp(input, N_QUIT, strlen(N_QUIT)))
    {
	return false;
    } 
    else if (!strncmp(input, N_TEST, strlen(N_TEST)))
    {
    } 
    else if (!strncmp(input, N_HELP, strlen(N_HELP)))
    {
		print_help();
    }
    else if (!strncmp(input, N_SYSTEM, strlen(N_SYSTEM)))
    {
		for (i=strlen(N_SYSTEM); isspace(input[i]); i++);
		system(&input[i]);
    } 
    else if (!strncmp(input, N_LINEAR_TRAIN, strlen(N_LINEAR_TRAIN)))
    {
    } 
    else if (!strncmp(input, N_LINEAR_LIKELIHOOD, strlen(N_LINEAR_LIKELIHOOD)))
    {
    } 
    else if (!strncmp(input, N_SAVE_LINEAR_LIKELIHOOD_BIN, strlen(N_SAVE_LINEAR_LIKELIHOOD_BIN)))
    {
    } 
    else if (!strncmp(input, N_SAVE_LINEAR_LIKELIHOOD, strlen(N_SAVE_LINEAR_LIKELIHOOD)))
    {
    } 
    else if (!strncmp(input, N_SVM_TRAIN, strlen(N_SVM_TRAIN)))
    {
    } 
    else if (!strncmp(input, N_SET_SVM_LIGHT, strlen(N_SET_SVM_LIGHT)))
	{
	}
    else if (!strncmp(input, N_SET_SVM_CPLEX, strlen(N_SET_SVM_CPLEX)))
	{
	}
    else if (!strncmp(input, N_LINEAR_SVM_TRAIN, strlen(N_LINEAR_SVM_TRAIN)))
    {
    } 
    else if (!strncmp(input, N_SVM_TEST, strlen(N_SVM_TEST)))
    {
    } 
    else if (!strncmp(input, N_ONE_CLASS_LINEAR_HMM_TEST, strlen(N_ONE_CLASS_LINEAR_HMM_TEST)))
    {
    } 
    else if (!strncmp(input, N_LINEAR_HMM_TEST, strlen(N_LINEAR_HMM_TEST)))
    {
    } 
    else if (!strncmp(input, N_ONE_CLASS_HMM_TEST, strlen(N_ONE_CLASS_HMM_TEST)))
    {
    } 
    else if (!strncmp(input, N_HMM_TEST, strlen(N_HMM_TEST)))
    {
    } 
    else if (!strncmp(input, N_APPEND_MODEL, strlen(N_APPEND_MODEL)))
    {
    } 
    else if (!strncmp(input, N_ADD_STATES, strlen(N_ADD_STATES)))
    {
    } 
    else if (!strncmp(input, N_C, strlen(N_C)))
    {
    } 
    else if (!strncmp(input, N_SET_ORDER, strlen(N_SET_ORDER)))
	{
	}
    else
		CIO::message("unrecognized command. type help for options\n");
    return true;
}

//// main - the one and only ///
int main(int argc, const char* argv[])
{
	if (argc<=1)
		while (gui.get_line());
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
					CIO::message("error opening/reading file: \"%s\"",argv[1]);
					return 1;
				}
				else
				{
					while(!feof(file) && gui.get_line(file));
					fclose(file);
				}
			}
		}
	}
	return 0;
}
