#include "gui/TextGUI.h"

#include <ctype.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <string.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include "lib/io.h"
#include "lib/common.h"
#include "distributions/histogram/Histogram.h"

#include <readline/readline.h>
#include <readline/history.h>



CTextGUI* gui=NULL;
CHistogram* h;

#ifdef WITHMATLAB
#include <libmmfile.h>
#endif // WITHMATLAB

#ifdef SVMMPI
#include "svm_mpi/mpi_base.h"
#endif

//names of menu commands
static const CHAR* N_NEW_HMM=			"new_hmm";
static const CHAR* N_NEW_SVM=			"new_svm";
static const CHAR* N_NEW_KNN=			"new_knn";
static const CHAR* N_NEW_PLUGIN_ESTIMATOR="new_plugin_estimator";
static const CHAR* N_TRAIN_ESTIMATOR=	"train_estimator";
static const CHAR* N_TEST_ESTIMATOR=	"test_estimator";
static const CHAR* N_TRAIN_KNN=	"train_knn";
static const CHAR* N_TEST_KNN=	"test_knn";
static const CHAR* N_SET_NUM_TABLES=	"set_num_tables";
static const CHAR* N_LOAD_PREPROC=		"load_preproc";
static const CHAR* N_SAVE_PREPROC=		"save_preproc";
static const CHAR* N_LOAD_HMM=			"load_hmm";
static const CHAR* N_SAVE_HMM=			"save_hmm";
static const CHAR* N_LOAD_SVM=			"load_svm";
static const CHAR* N_SAVE_SVM=			"save_svm";
static const CHAR* N_LOAD_KERNEL_INIT=	"load_kernel_init";
static const CHAR* N_SAVE_KERNEL_INIT=	"save_kernel_init";
static const CHAR* N_LOAD_LABELS=		"load_labels";
static const CHAR* N_LOAD_FEATURES=		"load_features";
static const CHAR* N_SAVE_FEATURES=		"save_features";
static const CHAR* N_RESHAPE=			"reshape";
static const CHAR* N_LOAD_DEFINITIONS=	"load_defs";
static const CHAR* N_SAVE_KERNEL=		"save_kernel";
static const CHAR* N_SET_HMM_AS=		"set_hmm_as";
static const CHAR* N_NORMALIZE=			"normalize_hmm";
static const CHAR* N_RELATIVE_ENTROPY=	"relative_entropy";
static const CHAR* N_ENTROPY=			"entropy";
static const CHAR* N_PERMUTATION_ENTROPY="permutation_entropy";
static const CHAR* N_SET_KERNEL=		"set_kernel";
static const CHAR* N_ADD_PREPROC=		"add_preproc";
static const CHAR* N_DEL_PREPROC=		"del_preproc";
static const CHAR* N_PREPROCESS=		"preprocess";
static const CHAR* N_INIT_KERNEL=		"init_kernel";
static const CHAR* N_DELETE_KERNEL_TREE=		"delete_kernel_tree";
static const CHAR* N_INIT_KERNEL_TREE=		"init_kernel_tree";
static const CHAR* N_SAVE_PATH=			"save_hmm_path";
static const CHAR* N_SAVE_LIKELIHOOD=	"save_hmm_likelihood";
static const CHAR* N_BEST_PATH=			"best_path";
static const CHAR* N_OUTPUT_PATH=      		"output_hmm_path";
static const CHAR* N_VITERBI_TRAIN=	       	"vit";
static const CHAR* N_VITERBI_TRAIN_DEFINED=     "vit_def";
static const CHAR* N_LINEAR_TRAIN=       	"linear_train";
static const CHAR* N_CLEAR=			"clear";
static const CHAR* N_CHOP=			"chop";
static const CHAR* N_CONVERGENCE_CRITERIA=	"convergence_criteria";
static const CHAR* N_PSEUDO=			"pseudo";
static const CHAR* N_CONVERT_TO_SPARSE=	"convert_to_sparse";
static const CHAR* N_CONVERT_TO_DENSE=	"convert_to_dense";
static const CHAR* N_CONVERT_CHAR_TO_WORD=	"convert_char_to_word";
static const CHAR* N_CONVERT_CHAR_TO_SHORT=	"convert_char_to_short";
static const CHAR* N_CONVERT=	"convert";
static const CHAR* N_C=			     	"c";
static const CHAR* N_ADD_STATES=	        "add_states";
static const CHAR* N_APPEND_HMM=		"append_hmm";
static const CHAR* N_BAUM_WELCH_TRAIN=	        "bw";
static const CHAR* N_BAUM_WELCH_TRANS_TRAIN=	"bw_trans";
static const CHAR* N_BAUM_WELCH_TRAIN_DEFINED=	"bw_def";
static const CHAR* N_LIKELIHOOD=	       	"likelihood";
static const CHAR* N_ALPHABET=			"alphabet";
static const CHAR* N_OUTPUT_HMM=		"output_hmm";
static const CHAR* N_OUTPUT_HMM_DEFINED=        "output_hmm_defined";
static const CHAR* N_QUIT=			"quit";
static const CHAR* N_EXEC=			"exec";
static const CHAR* N_EXIT=			"exit";
static const CHAR* N_HELP=			"help";
static const CHAR* N_SYSTEM=			"!";
static const CHAR N_COMMENT1=			'#';
static const CHAR N_COMMENT2=			'%';
static const CHAR* N_SET_MAX_DIM=		"max_dim";
static const CHAR* N_TEST=			"test";
static const CHAR* N_SET_THRESHOLD=			"set_threshold";
static const CHAR* N_SVM_TRAIN=			"svm_train";
static const CHAR* N_SVM_TEST=			"svm_test";
static const CHAR* N_ONE_CLASS_HMM_TEST=	"one_class_hmm_test";
static const CHAR* N_HMM_TEST=			"hmm_test";
static const CHAR* N_HMM_CLASSIFY=		"hmm_classify";
static const CHAR* N_SET_OUTPUT=		"set_output";
static const CHAR* N_GRADIENT_STEP=		"do_grad_step";
static const CHAR* N_ALIGN_CHAR=                "align_char" ;
static const CHAR* N_SET_REF_FEAT=              "set_ref_features" ;
static const CHAR* N_TIC=              "tic" ;
static const CHAR* N_TOC=              "toc" ;

CTextGUI::CTextGUI(INT argc, const CHAR** argv)
: CGUI(argc, argv), out_file(NULL)
{
#ifdef WITHMATLAB
	libmmfileInitialize() ;
#endif

#ifdef SVMMPI
	//CIO::message("Initializing MPI\n");
	CMPIBase::svm_mpi_init(argc, argv) ;
#else
	CIO::message("undef'd MPI\n");
#endif

	CIO::message("HMM uses %i separate tables\n", guihmm.get_number_of_tables()) ;
}

CTextGUI::~CTextGUI()
{
	if (out_file)
		fclose(out_file);
#ifdef WITHMATLAB
	libmmfileTerminate() ;
#endif

#ifdef SVMMPI
#if  defined(HAVE_MPI) && !defined(DISABLE_MPI)
	CMPIBase::svm_mpi_destroy() ;
#endif
#endif
}

void CTextGUI::print_help()
{
	CIO::message("\n[LOAD]\n");
	CIO::message("\033[1;31m%s\033[0m <filename>\t- load hmm\n",N_LOAD_HMM);
	CIO::message("\033[1;31m%s\033[0m <filename> <LINEAR|MPI|CPLEX>\t- load svm\n",N_LOAD_SVM);
	CIO::message("\033[1;31m%s\033[0m <filename>\t- load kernel init data\n",N_LOAD_KERNEL_INIT);
	CIO::message("\033[1;31m%s\033[0m <filename> <SIMPLE|SPARSE|STRING> <REAL|SHORT|BYTE|CHAR> <TRAIN|TEST> [<CACHE SIZE> [0|1]]\t- load features\n",N_LOAD_FEATURES);
	CIO::message("\033[1;31m%s\033[0m <filename> <TRAIN|TEST> \t- load labels\n",N_LOAD_LABELS);
	CIO::message("\033[1;31m%s\033[0m <filename>\t- load preproc init data\n",N_LOAD_PREPROC);
	CIO::message("\033[1;31m%s\033[0m <filename> [initialize=1]\t- load hmm defs\n",N_LOAD_DEFINITIONS);
	CIO::message("\n[SAVE]\n");
	CIO::message("\033[1;31m%s\033[0m <filename> [<0|1>]\t- save hmm in [binary] format\n",N_SAVE_HMM);
	CIO::message("\033[1;31m%s\033[0m <filename>\t- save svm\n",N_SAVE_SVM);
	CIO::message("\033[1;31m%s\033[0m <filename>\t- save kernel init data\n",N_SAVE_KERNEL_INIT);
	CIO::message("\033[1;31m%s\033[0m <filename>\t- save preproc init data\n",N_SAVE_PREPROC);
	CIO::message("\033[1;31m%s\033[0m <filename> <REAL|...> <TRAIN|TEST> \t- save features\n",N_SAVE_FEATURES);
	CIO::message("\033[1;31m%s\033[0m <filename>\t- save likelihood for each sequence\n",N_SAVE_LIKELIHOOD);
	CIO::message("\033[1;31m%s\033[0m <filename>\t- save state sequence of viterbi path\n",N_SAVE_PATH);
	CIO::message("\n[HMM]\n");
	CIO::message("\033[1;31m%s\033[0m - frees all HMMs and observations\n",N_CLEAR);
	CIO::message("\033[1;31m%s\033[0m #states #oberservations\t- frees previous HMM and creates an empty new one\n",N_NEW_HMM);
	CIO::message("\033[1;31m%s\033[0m <POS|NEG|TEST>- make current HMM the POS,NEG or TEST HMM; then free current HMM \n",N_SET_HMM_AS);
	CIO::message("\033[1;31m%s\033[0m <value>\t\t\t- chops likelihood of all parameters 0<value<1\n", N_CHOP);
	CIO::message("\033[1;31m%s\033[0m [<keep_dead_states>]\t\t\t- normalizes HMM params to be sum = 1\n", N_NORMALIZE);
	CIO::message("\033[1;31m%s\033[0m \t\t\t- returns the relative entropy for each position (requires lin. HMMS)\n", N_RELATIVE_ENTROPY);
	CIO::message("\033[1;31m%s\033[0m \t\t\t- returns the entropy for each position (requires lin. HMM)\n", N_ENTROPY);
	CIO::message("\033[1;31m%s\033[0m <width> <num>\t\t\t- returns the permutation entropy for sequence <num>\n", N_PERMUTATION_ENTROPY);
	CIO::message("\033[1;31m%s\033[0m <<num> [<value>]>\t\t\t- add num (def 1) states,initialize with value (def rnd)\n", N_ADD_STATES);
	CIO::message("\033[1;31m%s\033[0m <filename> [<INT> <INT>]\t\t\t- append HMM <filename> to current HMM\n", N_APPEND_HMM);
	CIO::message("\033[1;31m%s\033[0m [pseudovalue]\t\t\t- changes pseudo value\n", N_PSEUDO);
	CIO::message("\033[1;31m%s\033[0m <PROTEIN|DNA|ALPHANUM|CUBE>\t\t\t- changes alphabet type\n", N_ALPHABET);
	CIO::message("\033[1;31m%s\033[0m [maxiterations] [maxallowedchange]\t- defines the convergence criteria for all train algorithms\n",N_CONVERGENCE_CRITERIA);
	CIO::message("\033[1;31m%s\033[0m <max_dim>\t - set maximum number of patterns\n",N_SET_MAX_DIM);
	CIO::message("\033[1;31m%s\033[0m <num>\t - set number of forw/backw.-tables\n",N_SET_NUM_TABLES);
	CIO::message("\n[TRAIN]\n");
	CIO::message("\033[1;31m%s\033[0m [<width> <upto>]\t\t- obtains new linear HMM\n",N_LINEAR_TRAIN);
	CIO::message("\033[1;31m%s\033[0m\t\t- does viterbi training on the current HMM\n",N_VITERBI_TRAIN);
	CIO::message("\033[1;31m%s\033[0m\t\t- does viterbi training only on defined transitions etc\n",N_VITERBI_TRAIN_DEFINED);
	CIO::message("\033[1;31m%s\033[0m\t\t- does baum welch training on current HMM\n",N_BAUM_WELCH_TRAIN);
	CIO::message("\033[1;31m%s\033[0m\t\t- does baum welch training only on defined transitions etc.\n",N_BAUM_WELCH_TRAIN_DEFINED);
	CIO::message("\033[1;31m%s\033[0m\t- find the best path using viterbi\n",N_BEST_PATH);
	CIO::message("\033[1;31m%s\033[0m\t- find HMM likelihood\n",N_LIKELIHOOD);
	CIO::message("\n[HMM-OUTPUT]\n");
	CIO::message("\033[1;31m%s\033[0m [from to]\t- outputs best path\n",N_OUTPUT_PATH);
	CIO::message("\033[1;31m%s\033[0m\t- output whole HMM\n",N_OUTPUT_HMM);
	CIO::message("\033[1;31m%s\033[0m\t- output whole HMM\n",N_OUTPUT_HMM_DEFINED);
	CIO::message("\n[FEATURES]\n");
	CIO::message("\033[1;31m%s\033[0m <PCACUT|NORMONE|PRUNEVARSUBMEAN|LOGPLUSONE>\t\t\t- add preprocessor of type\n", N_ADD_PREPROC);
	CIO::message("\033[1;31m%s\033[0m <NUM>\t\t\t- delete preprocessor\n", N_DEL_PREPROC);
	CIO::message("\033[1;31m%s\033[0m <TRAIN|TEST> <NUM_FEAT> <NUM_VEC>\t\t\t- reshape feature matrix for simple features\n", N_RESHAPE);
	CIO::message("\033[1;31m%s\033[0m  <TRAIN|TEST>\t\t\t- convert dense features to sparse feature matrix\n", N_CONVERT_TO_SPARSE);
	CIO::message("\033[1;31m%s\033[0m  <TRAIN|TEST>\t\t\t- convert sparse features to dense feature matrix\n", N_CONVERT_TO_DENSE);
	CIO::message("\033[1;31m%s\033[0m  <TRAIN|TEST> <DNA|..> <ORDER> <START>\t\t\t- convert CHAR to word features, order remapping\n", N_CONVERT_CHAR_TO_WORD);
	CIO::message("\033[1;31m%s\033[0m  <TRAIN|TEST> <DNA|..> <ORDER> <START>\t\t\t- convert CHAR to short features, order remapping\n", N_CONVERT_CHAR_TO_SHORT);
	CIO::message("\033[1;31m%s\033[0m  <TRAIN|TEST> <STRING|SPARSE|SIMPLE> <REAL|CHAR|WORD|..> <STRING|...> <REAL|TOP..>\t\t\t- convert from feature class/type to class/type\n", N_CONVERT);
	CIO::message("\033[1;31m%s\033[0m\t <TRAIN|TEST> [<0|1>] - preprocesses the feature_matrix, 1 to force preprocessing of already processed\n",N_PREPROCESS);
	CIO::message("\n[CLASSIFIER]\n");
	CIO::message("\033[1;31m%s\033[0m\t <LIGHT|CPLEX|MPI> - creates SVM of type LIGHT,CPLEX or MPI\n",N_NEW_SVM);
	CIO::message("\033[1;31m%s\033[0m [c-value]\t\t\t- changes svm_c value\n", N_C);
	CIO::message("\033[1;31m%s\033[0m <LINEAR|GAUSSIAN|POLY|...> <REAL|BYTE|SPARSEREAL|SLIK> [<CACHESIZE> [OPTS]]\t\t\t- set kernel type\n", N_SET_KERNEL);
	CIO::message("\033[1;31m%s\033[0m\t\t- obtains svm from TRAINFEATURES\n",N_SVM_TRAIN);
	CIO::message("\033[1;31m%s\033[0m\t <TRAIN|TEST> - init kernel for training/testingn\n",N_INIT_KERNEL);
	CIO::message("\033[1;31m%s\033[0m\t - creates Plugin Estimator using Linear HMMs\n",N_NEW_PLUGIN_ESTIMATOR);
	CIO::message("\033[1;31m%s\033[0m\t- creates new KNN classifier\n",N_NEW_KNN);
	CIO::message("\033[1;31m%s\033[0m<k>\t- trains KNN classifier\n",N_TRAIN_KNN);
	CIO::message("\033[1;31m%s\033[0m\t [<pos_pseudo> [neg_pseudo]]- train the Estimator\n",N_TRAIN_ESTIMATOR);
	CIO::message("\n[CLASSIFICATION]\n");
	CIO::message("\033[1;31m%s\033[0m<threshold>\t\t\t\t- set classification threshold\n",N_SET_THRESHOLD);
	CIO::message("\033[1;31m%s\033[0m[[<output> [<rocfile>]]]\t\t\t\t- calculate output from obs using test HMM\n",N_ONE_CLASS_HMM_TEST);
	CIO::message("\033[1;31m%s\033[0m[[<output> [<rocfile>]]]\t\t\t\t- calculate output from obs using current HMMs\n",N_HMM_TEST);
	CIO::message("\033[1;31m%s\033[0m[<output>]\t\t\t\t- classify unknown examples using current HMMs\n",N_HMM_CLASSIFY);
	CIO::message("\033[1;31m%s\033[0m[[<output> [<rocfile>]]]\t\t- calculate svm output on TESTFEATURES\n",N_SVM_TEST);
	CIO::message("\033[1;31m%s\033[0m[[<output> [<rocfile>]]]\t\t- calculate estimator output on TESTFEATURES\n",N_TEST_ESTIMATOR);
	CIO::message("\033[1;31m%s\033[0m<k>\t- tests KNN classifier\n",N_TEST_KNN);
	CIO::message("\n[SYSTEM]\n");
	CIO::message("\033[1;31m%s\033[0m <STDERR|STDOUT|filename>\t- make std-output go to e.g file\n",N_SET_OUTPUT);
	CIO::message("\033[1;31m%s\033[0m <filename>\t- load and execute a script\n",N_EXEC);
	CIO::message("\033[1;31m%s\033[0m\t- exit genfinder\n",N_QUIT);
	CIO::message("\033[1;31m%s\033[0m\t- exit genfinder\n",N_EXIT);
	CIO::message("\033[1;31m%s\033[0m\t- this message\n",N_HELP);
	CIO::message("\033[1;31m%s\033[0m <commands>\t- execute system functions \n",N_SYSTEM);
}

void CTextGUI::print_prompt()
{
	CIO::message("\033[1;34mgenefinder\033[0m >> ");
	//CIO::message("genefinder >> ");
}

CHAR* CTextGUI::get_line(FILE* infile, bool show_prompt)
{
	memset(input, 0, sizeof(input));

	//guihmm.debug();
	//if (show_prompt)
	//	print_prompt();

	if (feof(infile))
		return NULL;

	char* in=readline("\033[1;34mgenefinder\033[0m >> ");
	if (in)
	{
		strncpy(input, in, sizeof(input));
		add_history(in);
		free(in);
	}

	if (in==NULL || (!strlen(input)))
		return NULL;
	//if ( (fgets(input, sizeof(input), infile)==NULL) || (!strlen(input)) )
		//return NULL;
	else
		return input;
}

bool CTextGUI::parse_line(CHAR* input)
{
	INT i;
	if (!input)
		return false;

	if ((input[0]==N_COMMENT1) || (input[0]==N_COMMENT2) || (input[0]=='\0') || (input[0]=='\n'))
		return true;

	//remove linebreaks if there are any
	if (strlen(input)>=1 && input[strlen(input)-1]=='\n')
		input[strlen(input)-1]='\0';

	CIO::message("%s\n",input) ;

	if (!strncmp(input, N_NEW_HMM, strlen(N_NEW_HMM)))
	{
		guihmm.new_hmm(input+strlen(N_NEW_HMM));
	} 
	else if (!strncmp(input, N_SET_NUM_TABLES, strlen(N_SET_NUM_TABLES)))
	{
		guihmm.set_num_hmm_tables(input+strlen(N_SET_NUM_TABLES));
	} 
	else if (!strncmp(input, N_NEW_SVM, strlen(N_NEW_SVM)))
	{
		guisvm.new_svm(input+strlen(N_NEW_SVM));
	} 
	else if (!strncmp(input, N_NEW_KNN, strlen(N_NEW_KNN)))
	{
		guiknn.new_knn(input+strlen(N_NEW_KNN));
	} 
	else if (!strncmp(input, N_NEW_PLUGIN_ESTIMATOR, strlen(N_NEW_PLUGIN_ESTIMATOR)))
	{
		guipluginestimate.new_estimator(input+strlen(N_NEW_PLUGIN_ESTIMATOR));
	} 
	else if (!strncmp(input, N_TRAIN_KNN, strlen(N_TRAIN_KNN)))
	{
		guiknn.train(input+strlen(N_TRAIN_KNN));
	} 
	else if (!strncmp(input, N_TEST_KNN, strlen(N_TEST_KNN)))
	{
		guiknn.test(input+strlen(N_TEST_KNN));
	} 
	else if (!strncmp(input, N_TRAIN_ESTIMATOR, strlen(N_TRAIN_ESTIMATOR)))
	{
		guipluginestimate.train(input+strlen(N_TRAIN_ESTIMATOR));
	} 
	else if (!strncmp(input, N_TEST_ESTIMATOR, strlen(N_TEST_ESTIMATOR)))
	{
		guipluginestimate.test(input+strlen(N_TEST_ESTIMATOR));
	} 
	else if (!strncmp(input, N_LOAD_HMM, strlen(N_LOAD_HMM)))
	{
		guihmm.load(input+strlen(N_LOAD_HMM));
	} 
	else if (!strncmp(input, N_LOAD_PREPROC, strlen(N_LOAD_PREPROC)))
	{
		guipreproc.load(input+strlen(N_LOAD_PREPROC)) ;
	} 
	else if (!strncmp(input, N_LOAD_LABELS, strlen(N_LOAD_LABELS)))
	{
		guilabels.load(input+strlen(N_LOAD_LABELS));
	} 
	else if (!strncmp(input, N_LOAD_FEATURES, strlen(N_LOAD_FEATURES)))
	{
		guifeatures.load(input+strlen(N_LOAD_FEATURES));
	} 
	else if (!strncmp(input, N_SAVE_FEATURES, strlen(N_SAVE_FEATURES)))
	{
		guifeatures.save(input+strlen(N_SAVE_FEATURES));
	} 
	else if (!strncmp(input, N_RESHAPE, strlen(N_RESHAPE)))
	{
		guifeatures.reshape(input+strlen(N_RESHAPE));
	} 
	else if (!strncmp(input, N_CONVERT_TO_SPARSE, strlen(N_CONVERT_TO_SPARSE)))
	{
		guifeatures.convert_full_to_sparse(input+strlen(N_CONVERT_TO_SPARSE));
	} 
	else if (!strncmp(input, N_CONVERT_TO_DENSE, strlen(N_CONVERT_TO_DENSE)))
	{
		guifeatures.convert_sparse_to_full(input+strlen(N_CONVERT_TO_DENSE));
	} 
	else if (!strncmp(input, N_CONVERT_CHAR_TO_WORD, strlen(N_CONVERT_CHAR_TO_WORD)))
	{
		guifeatures.convert_char_to_word(input+strlen(N_CONVERT_CHAR_TO_WORD));
	} 
	else if (!strncmp(input, N_SET_REF_FEAT, strlen(N_SET_REF_FEAT)))
        {
	        guifeatures.set_ref_features(input+strlen(N_SET_REF_FEAT));
	} 
	else if (!strncmp(input, N_ALIGN_CHAR, strlen(N_ALIGN_CHAR)))
        {
	        guifeatures.alignment_char(input+strlen(N_ALIGN_CHAR));
	} 
	else if (!strncmp(input, N_CONVERT_CHAR_TO_SHORT, strlen(N_CONVERT_CHAR_TO_SHORT)))
	{
		guifeatures.convert_char_to_word(input+strlen(N_CONVERT_CHAR_TO_SHORT));
	} 
	else if (!strncmp(input, N_CONVERT, strlen(N_CONVERT)))
	{
		guifeatures.convert(input+strlen(N_CONVERT));
	} 
	else if (!strncmp(input, N_LOAD_SVM, strlen(N_LOAD_SVM)))
	{
		guisvm.load(input+strlen(N_LOAD_SVM));
	} 
	else if (!strncmp(input, N_DELETE_KERNEL_TREE, strlen(N_DELETE_KERNEL_TREE)))
	{
		guikernel.delete_kernel_tree(input+strlen(N_DELETE_KERNEL_TREE));
	} 
	else if (!strncmp(input, N_INIT_KERNEL_TREE, strlen(N_INIT_KERNEL_TREE)))
	{
		guikernel.init_kernel_tree(input+strlen(N_INIT_KERNEL_TREE));
	} 
	else if (!strncmp(input, N_INIT_KERNEL, strlen(N_INIT_KERNEL)))
	{
		guikernel.init_kernel(input+strlen(N_INIT_KERNEL));
	} 
	else if (!strncmp(input, N_LOAD_KERNEL_INIT, strlen(N_LOAD_KERNEL_INIT)))
	{
		guikernel.load_kernel_init(input+strlen(N_LOAD_KERNEL_INIT));
	} 
	else if (!strncmp(input, N_SET_HMM_AS, strlen(N_SET_HMM_AS)))
	{
		guihmm.set_hmm_as(input+strlen(N_SET_HMM_AS));
	}
	else if (!strncmp(input, N_CHOP, strlen(N_CHOP)))
	{
		guihmm.chop(input+strlen(N_CHOP));
	} 
	else if (!strncmp(input, N_SAVE_LIKELIHOOD, strlen(N_SAVE_LIKELIHOOD)))
	{
		guihmm.save_likelihood(input+strlen(N_SAVE_LIKELIHOOD));
	}
	else if (!strncmp(input, N_SAVE_PATH, strlen(N_SAVE_PATH)))
	{
		guihmm.save_path(input+strlen(N_SAVE_PATH));
	}
	else if (!strncmp(input, N_SAVE_HMM, strlen(N_SAVE_HMM)))
	{
		guihmm.save(input+strlen(N_SAVE_HMM)) ;
	} 
	else if (!strncmp(input, N_SAVE_PREPROC, strlen(N_SAVE_PREPROC)))
	{
		guipreproc.save(input+strlen(N_SAVE_PREPROC)) ;
	} 
	else if (!strncmp(input, N_SAVE_SVM, strlen(N_SAVE_SVM)))
	{
		guisvm.save(input+strlen(N_SAVE_SVM)) ;
	} 
	else if (!strncmp(input, N_SAVE_KERNEL_INIT, strlen(N_SAVE_KERNEL_INIT)))
	{
		guikernel.save_kernel_init(input+strlen(N_SAVE_KERNEL_INIT)) ;
	} 
	else if (!strncmp(input, N_LOAD_DEFINITIONS, strlen(N_LOAD_DEFINITIONS)))
	{
		guihmm.load_defs(input+strlen(N_LOAD_DEFINITIONS));
	} 
	else if (!strncmp(input, N_SAVE_KERNEL, strlen(N_SAVE_KERNEL)))
	{
		guikernel.save_kernel(input+strlen(N_SAVE_KERNEL));
	} 
	else if (!strncmp(input, N_CLEAR, strlen(N_CLEAR)))
	{
		const CHAR** argv=gui->argv;
		INT argc=gui->argc;
		delete gui;
		gui=new CTextGUI(argc, argv);
	} 
	else if (!strncmp(input, N_PSEUDO, strlen(N_PSEUDO)))
	{
		guihmm.set_pseudo(input+strlen(N_PSEUDO));
	} 
	else if (!strncmp(input, N_SET_THRESHOLD, strlen(N_SET_THRESHOLD)))
	{
		guimath.set_threshold(input+strlen(N_SET_THRESHOLD));
	} 
	//else if (!strncmp(input, N_ALPHABET, strlen(N_ALPHABET)))
	//{
		//guiobs.set_alphabet(input+strlen(N_ALPHABET)) ;
	//} 
	else if (!strncmp(input, N_CONVERGENCE_CRITERIA, strlen(N_CONVERGENCE_CRITERIA)))
	{
		guihmm.convergence_criteria(input+strlen(N_CONVERGENCE_CRITERIA)) ;
	} 
	else if (!strncmp(input, N_VITERBI_TRAIN_DEFINED, strlen(N_VITERBI_TRAIN_DEFINED)))
	{
		guihmm.viterbi_train_defined(input+strlen(N_VITERBI_TRAIN_DEFINED));
	} 
	else if (!strncmp(input, N_VITERBI_TRAIN, strlen(N_VITERBI_TRAIN)))
	{
		guihmm.viterbi_train(input+strlen(N_VITERBI_TRAIN));
	}
	else if (!strncmp(input, N_BAUM_WELCH_TRAIN_DEFINED, strlen(N_BAUM_WELCH_TRAIN_DEFINED)))
	{
		CIO::not_implemented() ;
	} 
	else if (!strncmp(input, N_BAUM_WELCH_TRANS_TRAIN, strlen(N_BAUM_WELCH_TRANS_TRAIN)))
	{
		guihmm.baum_welch_trans_train(input+strlen(N_BAUM_WELCH_TRANS_TRAIN));
	} 
	else if (!strncmp(input, N_BAUM_WELCH_TRAIN, strlen(N_BAUM_WELCH_TRAIN)))
	{
		guihmm.baum_welch_train(input+strlen(N_BAUM_WELCH_TRAIN));
	} 
	else if (!strncmp(input, N_BEST_PATH, strlen(N_BEST_PATH)))
	{
		guihmm.best_path(input+strlen(N_BEST_PATH));
	} 
	else if (!strncmp(input, N_LIKELIHOOD, strlen(N_LIKELIHOOD)))
	{
		guihmm.likelihood(input+strlen(N_LIKELIHOOD));
	} 
	else if (!strncmp(input, N_OUTPUT_HMM_DEFINED, strlen(N_OUTPUT_HMM_DEFINED)))
	{
		guihmm.output_hmm_defined(input+strlen(N_OUTPUT_HMM_DEFINED));
	} 
	else if (!strncmp(input, N_OUTPUT_PATH, strlen(N_OUTPUT_PATH)))
	{
		guihmm.output_hmm_path(input+strlen(N_OUTPUT_PATH));
	} 
	else if (!strncmp(input, N_OUTPUT_HMM, strlen(N_OUTPUT_HMM)))
	{
		guihmm.output_hmm(input+strlen(N_OUTPUT_HMM));
	} 
	else if (!strncmp(input, N_SET_OUTPUT, strlen(N_SET_OUTPUT)))
	{
		for (i=strlen(N_SET_OUTPUT); isspace(input[i]); i++);
		CHAR* param=&input[i];

		if (out_file)
			fclose(out_file);

		out_file=NULL;

		CIO::message("setting out_target to: %s\n", param);

		if (strcmp(param, "STDERR")==0)
			CIO::set_target(stderr);
		else if(strcmp(param, "STDOUT")==0)
			CIO::set_target(stdout);
		else
		{
			out_file=fopen(param, "w");
			if (!out_file)
				CIO::message("error opening out_target \"%s\"", param);
			CIO::set_target(out_file);
		}
	}
	else if (!strncmp(input, N_EXEC, strlen(N_EXEC)))
	{
		for (i=strlen(N_EXEC); isspace(input[i]); i++);

		FILE* file=fopen(&input[i], "r");

		if (file)
		{
			while(!feof(file) && gui->parse_line((gui->get_line(file,false))));
			fclose(file);
		}
		else
			CIO::message("error opening/reading file: \"%s\"",argv[1]);
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
		CIO::not_implemented() ;
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
		guihmm.linear_train(input+strlen(N_LINEAR_TRAIN));
	} 
	else if (!strncmp(input, N_SVM_TRAIN, strlen(N_SVM_TRAIN)))
	{
		guisvm.train(input+strlen(N_SVM_TRAIN));
	} 
	else if (!strncmp(input, N_SET_KERNEL, strlen(N_SET_KERNEL)))
	{
		guikernel.set_kernel(input+strlen(N_SET_KERNEL));
	} 
	else if (!strncmp(input, N_DEL_PREPROC, strlen(N_DEL_PREPROC)))
	{
		guipreproc.del_preproc(input+strlen(N_DEL_PREPROC));
	} 
	else if (!strncmp(input, N_ADD_PREPROC, strlen(N_ADD_PREPROC)))
	{
		guipreproc.add_preproc(input+strlen(N_ADD_PREPROC));
	} 
	else if (!strncmp(input, N_PREPROCESS, strlen(N_PREPROCESS)))
	{
		guifeatures.preprocess(input+strlen(N_PREPROCESS));
	} 
	else if (!strncmp(input, N_SVM_TEST, strlen(N_SVM_TEST)))
	{
		guisvm.test(input+strlen(N_SVM_TEST));
	} 
	else if (!strncmp(input, N_ONE_CLASS_HMM_TEST, strlen(N_ONE_CLASS_HMM_TEST)))
	{
		guihmm.one_class_test(input+strlen(N_ONE_CLASS_HMM_TEST));
	} 
	else if (!strncmp(input, N_HMM_CLASSIFY, strlen(N_HMM_CLASSIFY)))
	{
		guihmm.hmm_classify(input+strlen(N_HMM_CLASSIFY));
	}
	else if (!strncmp(input, N_HMM_TEST, strlen(N_HMM_TEST)))
	{
		guihmm.hmm_test(input+strlen(N_HMM_TEST));
	}
	else if (!strncmp(input, N_NORMALIZE, strlen(N_NORMALIZE)))
	{
		guihmm.normalize(input+strlen(N_NORMALIZE));
	} 
	else if (!strncmp(input, N_RELATIVE_ENTROPY, strlen(N_RELATIVE_ENTROPY)))
	{
		guihmm.relative_entropy(input+strlen(N_RELATIVE_ENTROPY));
	} 
	else if (!strncmp(input, N_ENTROPY, strlen(N_ENTROPY)))
	{
		guihmm.entropy(input+strlen(N_ENTROPY));
	} 
	else if (!strncmp(input, N_PERMUTATION_ENTROPY, strlen(N_PERMUTATION_ENTROPY)))
	{
		guihmm.permutation_entropy(input+strlen(N_PERMUTATION_ENTROPY));
	} 
	else if (!strncmp(input, N_APPEND_HMM, strlen(N_APPEND_HMM)))
	{
		guihmm.append_model(input+strlen(N_APPEND_HMM));
	} 
	else if (!strncmp(input, N_ADD_STATES, strlen(N_ADD_STATES)))
	{
		guihmm.add_states(input+strlen(N_ADD_STATES));
	} 
	else if (!strncmp(input, N_C, strlen(N_C)))
	{
		guisvm.set_C(input+strlen(N_C));
	} 
	else if (!strncmp(input, N_GRADIENT_STEP, strlen(N_GRADIENT_STEP)))
	{
     	        guihmm.gradient_step(input+strlen(N_GRADIENT_STEP)) ;
	} 
	else if (!strncmp(input, N_TIC, strlen(N_TIC)))
	{
     	        guitime.start();
	} 
	else if (!strncmp(input, N_TOC, strlen(N_TOC)))
	{
     	        guitime.stop();
	} 
	else
		CIO::message("unrecognized command. type help for options\n");

	return true;
}

//// main - the one and only ///
int main(int argc, const CHAR* argv[])
{
	gui=new CTextGUI(argc, argv) ;

	if (argc<=1)
		while (gui->parse_line(gui->get_line()));
	else
	{
		if (argc>=2)
		{
			if ( argc>2 || !strcmp(argv[1], "-h") || !strcmp(argv[1], "/?") || !strcmp(argv[1], "--help"))
			{
				CIO::message("usage: genefinder [ <-h|--help|/?|-i|<script> ]\n\n");
				CIO::message("if no options are given genfinder enters interactive mode\n");
				CIO::message("if <script> is specified the commands will be executed");
				CIO::message("if -i is specified genefinder will listen on port 6766 (==hex(gf), *dangerous* as commands from any source are accepted");
				return 1;
			}
			else if ( argc>2 || !strcmp(argv[1], "-i") || !strcmp(argv[1], "/?") || !strcmp(argv[1], "--help"))
			{
				int s=socket(AF_INET, SOCK_STREAM, 0);
				struct sockaddr_in sa;
				sa.sin_family=AF_INET;
				sa.sin_port=htons(6766);
				sa.sin_addr.s_addr=INADDR_ANY;
				bzero(&(sa.sin_zero), 8);

				bind(s, (sockaddr*) (&sa), sizeof(sockaddr_in));
				listen(s, 1);
				int s2=accept(s, NULL, NULL);
				CIO::message("accepting connection\n");

				CHAR input[2000];
				do
				{
					bzero(input, sizeof(input));
					int length=read(s2, input, sizeof(input));
					if (length>0 && length<(int) sizeof(input))
						input[length]='\0';
					else
					{
						CIO::message("error reading cmdline\n");
						return 1;
					}
				}
				while(gui->parse_line(input));
				return 0;
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
					while(!feof(file) && gui->parse_line(gui->get_line(file, false)));
					fclose(file);
				}
			}
		}
	}

	CIO::message("quitting...\n");
	delete gui ;

	return 0;
}
