/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2006 Fabio De Bona
 * Written (W) 2006 Soeren Sonnenburg
 * Copyright (C) 2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/config.h"

#if defined(HAVE_R) && !defined(HAVE_SWIG)
#include <stdio.h>
#include <string.h>
#include <Rdefines.h>

#include "lib/common.h"
#include "lib/io.h"

#include "gui/TextGUI.h"
#include "guilib/GUIR.h"
#include "gui/GUI.h"

static CGUI_R sg_R;
extern CTextGUI* gui;

static const CHAR* N_SEND_COMMAND=		"send_command";
static const CHAR* N_HELP=		        "help";
static const CHAR* N_CRC=			"crc";
//static const CHAR* N_TRANSLATE_STRING=			"translate_string";
static const CHAR* N_GET_HMM=			"get_hmm";
static const CHAR* N_GET_VITERBI_PATH=			"get_viterbi_path";
static const CHAR* N_GET_SVM=			"get_svm";
static const CHAR* N_GET_SVM_OBJECTIVE=		"get_svm_objective";
//static const CHAR* N_GET_KERNEL_INIT=	        "get_kernel_init";
static const CHAR* N_GET_KERNEL_MATRIX=	        "get_kernel_matrix";
static const CHAR* N_HMM_LIKELIHOOD=	        "hmm_likelihood";
//static const CHAR* N_GET_KERNEL_OPTIMIZATION=	        "get_kernel_optimization";
//static const CHAR* N_COMPUTE_BY_SUBKERNELS=	        "compute_by_subkernels";
//static const CHAR* N_SET_SUBKERNEL_WEIGHTS=	        "set_subkernel_weights";
//static const CHAR* N_SET_LAST_SUBKERNEL_WEIGHTS=	        "set_last_subkernel_weights";
//static const CHAR* N_SET_WD_POS_WEIGHTS=	        "set_WD_position_weights";
static const CHAR* N_GET_SUBKERNEL_WEIGHTS=	        "get_subkernel_weights";
//static const CHAR* N_GET_LAST_SUBKERNEL_WEIGHTS=	        "get_last_subkernel_weights";
//static const CHAR* N_GET_WD_POS_WEIGHTS=	        "get_WD_position_weights";
static const CHAR* N_GET_FEATURES=		"get_features";
static const CHAR* N_GET_LABELS=		"get_labels";
static const CHAR* N_GET_VERSION=		"get_version";
//static const CHAR* N_GET_PREPROC_INIT=	        "get_preproc_init";
//static const CHAR* N_GET_HMM_DEFS=		"get_hmm_defs";
static const CHAR* N_SET_HMM=			"set_hmm";
//static const CHAR* N_MODEL_PROB_NO_B_TRANS=			"model_prob_no_b_trans";
//static const CHAR* N_BEST_PATH_NO_B_TRANS=			"best_path_no_b_trans";
//static const CHAR* N_BEST_PATH_TRANS=			"best_path_trans";
//static const CHAR* N_BEST_PATH_2STRUCT=			"best_path_2struct";
//static const CHAR* N_BEST_PATH_TRANS_SIMPLE=			"best_path_trans_simple";
//static const CHAR* N_BEST_PATH_NO_B=			"best_path_no_b";
static const CHAR* N_APPEND_HMM=			"append_hmm";
static const CHAR* N_SET_SVM=			"set_svm";
static const CHAR* N_SET_CUSTOM_KERNEL=	        "set_custom_kernel";
//static const CHAR* N_SET_KERNEL_INIT=	        "set_kernel_init";
static const CHAR* N_SET_FEATURES=		"set_features";
static const CHAR* N_ADD_FEATURES=		"add_features";
static const CHAR* N_SET_LABELS=		"set_labels";
//static const CHAR* N_SET_PREPROC_INIT=	        "set_preproc_init";
//static const CHAR* N_SET_HMM_DEFS=		"set_hmm_defs";
static const CHAR* N_ONE_CLASS_HMM_CLASSIFY=		"one_class_hmm_classify";
//static const CHAR* N_ONE_CLASS_LINEAR_HMM_CLASSIFY=		"one_class_linear_hmm_classify";
static const CHAR* N_HMM_CLASSIFY=		"hmm_classify";
static const CHAR* N_ONE_CLASS_HMM_CLASSIFY_EXAMPLE=		"one_class_hmm_classify_example";
static const CHAR* N_HMM_CLASSIFY_EXAMPLE=	"hmm_classify_example";
static const CHAR* N_SVM_CLASSIFY=		"svm_classify";
static const CHAR* N_CLASSIFY=		"classify";
static const CHAR* N_SVM_CLASSIFY_EXAMPLE=	"svm_classify_example";
static const CHAR* N_CLASSIFY_EXAMPLE=	"classify_example";
//static const CHAR* N_GET_PLUGIN_ESTIMATE=	"get_plugin_estimate";
//static const CHAR* N_SET_PLUGIN_ESTIMATE=	"set_plugin_estimate";
//static const CHAR* N_PLUGIN_ESTIMATE_CLASSIFY=	"plugin_estimate_classify";
//static const CHAR* N_PLUGIN_ESTIMATE_CLASSIFY_EXAMPLE=	"plugin_estimate_classify_example";

extern "C" {

static void successful(SEXP& result, bool v)
{
	PROTECT(result);
	*LOGICAL(result)=v;
	UNPROTECT(1);
}
   
static SEXP sg_helper(SEXP args)
{
	SG_SDEBUG("length of args %d\n", length(args));


	SEXP result=NEW_LOGICAL(1);
	PROTECT(result);
	*LOGICAL(result)=FALSE;
	UNPROTECT(1);

	int cmd_len = length(args)-1;
	if ( cmd_len >= 1 )
	{
		args = CDR(args); /* pop "sg" out of list */
		CHAR* action=NULL;
		
		if (TYPEOF(CAR(args)) == STRSXP)
		{
			action=CHAR(VECTOR_ELT(CAR(args), 0));
			SG_SDEBUG("action is %s\n", action);
		}

		args = CDR(args); /* pop action out of list */

		if (action)
		{
			if (!strncmp(action, N_SEND_COMMAND, strlen(N_SEND_COMMAND)))
			{
				if (cmd_len==2)
				{
					CHAR* cmd=CHAR(STRING_ELT(CAR(args),0));
					SG_SDEBUG("command is %s\n", cmd);
					sg_R.send_command(cmd);
				}
				else
					SG_SERROR( "usage is sg('send_command', 'cmdline')");
			}
			else if (!strncmp(action, N_HELP, strlen(N_HELP)))
			{
				if (cmd_len==1)
				{
					sg_R.send_command("help");
				}
				else
					SG_SERROR( "usage is sg('help')");
			}
			else if (!strncmp(action, N_GET_VERSION, strlen(N_GET_VERSION)))
			{
				return sg_R.get_version();
			}
			else if (!strncmp(action, N_GET_SVM_OBJECTIVE, strlen(N_GET_SVM_OBJECTIVE)))
			{
				return sg_R.get_svm_objective();
			}
			else if (!strncmp(action, N_GET_SVM, strlen(N_GET_SVM)))
			{
				return sg_R.get_svm();
			}
			else if (!strncmp(action, N_SET_CUSTOM_KERNEL, strlen(N_SET_CUSTOM_KERNEL)))
			{
				SG_SERROR( "Not implemented yet");
				return R_NilValue;
			}
			else if (!strncmp(action, N_SVM_CLASSIFY_EXAMPLE, strlen(N_SVM_CLASSIFY_EXAMPLE)) ||
					!strncmp(action, N_CLASSIFY_EXAMPLE, strlen(N_CLASSIFY_EXAMPLE)))
			{
				if (cmd_len==2)
				{
					if (TYPEOF(args) == REALSXP)
						return sg_R.classify_example((INT) REAL(args)[0]);
					else
						SG_SERROR( "usage is [result]=sg('classify_example', feature_vector_index)");
				}
				else
					SG_SERROR( "usage is [result]=sg('classify_example', feature_vector_index)");
			}
			else if (!strncmp(action, N_GET_HMM, strlen(N_GET_HMM)))
			{
				return sg_R.get_hmm();
				//SG_ERROR( "usage is [p,q,a,b]=sg('get_hmm')");
			}
			else if (!strncmp(action, N_GET_VITERBI_PATH, strlen(N_GET_VITERBI_PATH)))
			{
				if (cmd_len == 2)
				{
					args = CAR(args);
					if (TYPEOF(args) == REALSXP)
						return sg_R.best_path((INT) REAL(args)[0]);
					else
						SG_SERROR( "usage is [path, lik]=sg('get_viterbi_path',dim)");
				}
				else
					SG_SERROR( "usage is [path, lik]=sg('get_viterbi_path',dim)");
			}
			else if (!strncmp(action, N_HMM_LIKELIHOOD, strlen(N_HMM_LIKELIHOOD)))
			{
				return sg_R.hmm_likelihood();
			}
			else if (!strncmp(action, N_ONE_CLASS_HMM_CLASSIFY_EXAMPLE, strlen(N_ONE_CLASS_HMM_CLASSIFY_EXAMPLE)))
			{
				if (cmd_len == 2)
				{
					if (TYPEOF(args) == REALSXP)
						sg_R.one_class_hmm_classify_example((INT) REAL(args)[0] );
					else
						SG_SERROR( "usage is [result]=sg('hmm_classify_example', feature_vector_index)");
				}
				else
					SG_SERROR( "usage is [result]=sg('one_class_hmm_classify_example', feature_vector_index)");
			}
			else if (!strncmp(action, N_ONE_CLASS_HMM_CLASSIFY, strlen(N_ONE_CLASS_HMM_CLASSIFY)))
			{
				return sg_R.one_class_hmm_classify();
			}
			else if (!strncmp(action, N_SVM_CLASSIFY, strlen(N_SVM_CLASSIFY)) ||
					!strncmp(action, N_CLASSIFY, strlen(N_CLASSIFY)))
			{
				return sg_R.classify();
			}
			else if (!strncmp(action, N_HMM_CLASSIFY, strlen(N_HMM_CLASSIFY)))
			{
				return sg_R.hmm_classify();
			}
			else if (!strncmp(action, N_HMM_CLASSIFY_EXAMPLE, strlen(N_HMM_CLASSIFY_EXAMPLE)))
			{
				if (cmd_len==1)
				{
					if (TYPEOF(args) == REALSXP)
						sg_R.hmm_classify_example((INT) REAL(args)[0] );
					else
						SG_SERROR( "usage is [result]=sg('hmm_classify_example', feature_vector_index)");
				}
				else
					SG_SERROR( "usage is [result]=sg('hmm_classify_example', feature_vector_index)");
			}
			else if (!strncmp(action, N_CRC, strlen(N_CRC)))
			{
				//if ((nrhs==2) && (nlhs==1))
				//{
				//CHAR* target=CHAR(STRING_ELT(CAR(args),0));
				//    CHAR* string=CGUIMatlab::get_mxString(prhs[1]);
				//    UINT sl = strlen(string) ;

				//    BYTE* bstring = new BYTE[sl] ;
				//    for (UINT i=0; i<sl; i++)
				// 	   bstring[i] = string[i] ;
				//    UINT res = CMath::crc32(bstring, sl) ;
				//    plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
				//    DREAL * p=mxGetPr(plhs[0]) ;
				//    *p = res ;
				//    delete[] bstring ;
				//    mxFree(string) ;
				//}
				//else
				//	   SG_SERROR( "usage is crc32=sg('crc', string)");

			}
			else if (!strncmp(action, N_ADD_FEATURES, strlen(N_ADD_FEATURES)))
			{
				//	if (cmd_len>=3) 
				// {
				CHAR* target=CHAR(STRING_ELT(CAR(args),0));
				args = CDR(args); /* pop target out of list */

				if ( (!strncmp(target, "TRAIN", strlen("TRAIN"))) || 
						(!strncmp(target, "TEST", strlen("TEST"))) ) 
				{

					SEXP features_mat = CAR(args); // Maybe results in NULL pointer 
					args = CDR(args); /* pop features out of list */
					SEXP alphabet = CAR(args); // Maybe results in NULL pointer 
					args = CDR(args); /* pop features out of list */

					CFeatures* features= sg_R.set_features(features_mat,alphabet);

					if (features)
					{
						if (!strncmp(target, "TRAIN", strlen("TRAIN")))
						{
							SG_SDEBUG("Adding features.\n");
							gui->guifeatures.add_train_features(features);
						}
						else if (!strncmp(target, "TEST", strlen("TEST")))
						{
							gui->guifeatures.add_test_features(features);
						}
					}
					else
						SG_SERROR( "usage is sg('add_features', 'TRAIN|TEST', features, ...)");
				}
				else
					SG_SERROR( "usage is sg('add_features', 'TRAIN|TEST', features, ...)");
			}
			else if (!strncmp(action, N_SET_FEATURES, strlen(N_SET_FEATURES)))
			{
				if (cmd_len>=3)
				{
					CHAR* target=CHAR(STRING_ELT(CAR(args),0));
					args = CDR(args); /* pop target out of list */

					if ( (!strncmp(target, "TRAIN", strlen("TRAIN"))) || 
							(!strncmp(target, "TEST", strlen("TEST"))) ) 
					{
						SEXP features_mat = CAR(args); // Maybe results in NULL pointer 
						args = CDR(args); /* pop features out of list */
						SEXP alphabet = CAR(args); // Maybe results in NULL pointer 
						args = CDR(args); /* pop features out of list */

						CFeatures* features = sg_R.set_features(features_mat,alphabet);

						if (features)
						{
							if (!strncmp(target, "TRAIN", strlen("TRAIN")))
								successful(result, gui->guifeatures.set_train_features(features));
							else if (!strncmp(target, "TEST", strlen("TEST")))
								successful(result, gui->guifeatures.set_test_features(features));
						}
						else
							SG_SERROR( "usage is sg('set_features', 'TRAIN|TEST', features, ...)");
					}
					else
						SG_SERROR( "usage is sg('set_features', 'TRAIN|TEST', features, ...)");
				}
				else
					SG_SERROR( "usage is sg('set_features', 'TRAIN|TEST', features, ...)");
				SG_SINFO( "done\n");
			}
			else if (!strncmp(action, N_SET_HMM, strlen(N_SET_HMM)))
			{
				if (cmd_len==2)
					successful(result, sg_R.set_hmm(CAR(args)));
				else
					SG_SERROR( "usage is sg('set_hmm',hmm$[p,q,a,b])");
			}
		else if (!strncmp(action, N_APPEND_HMM, strlen(N_APPEND_HMM)))
		{
			if (cmd_len==1+4)
				sg_R.append_hmm(args);
			else
				SG_SERROR( "usage is sg('append_hmm',[p,q,a,b])");
		}
			else if (!strncmp(action, N_SET_SVM, strlen(N_SET_SVM)))
			{
				if (cmd_len==3)
					successful(result, sg_R.set_svm(args));
				else
					SG_SERROR( "usage is sg('set_svm', [ b, alphas])");
			}
			else if (!strncmp(action, N_GET_FEATURES, strlen(N_GET_FEATURES)))
			{
				CFeatures* features=NULL;
				CHAR* target=CHAR(STRING_ELT(CAR(args),0));
				args = CDR(args); /* pop target out of list */

				if (!strncmp(target, "TRAIN", strlen("TRAIN")))
				{
					features=gui->guifeatures.get_train_features();
				}
				else if (!strncmp(target, "TEST", strlen("TEST")))
				{
					features=gui->guifeatures.get_test_features();
				}
				else
					SG_SERROR( "usage is [features]=sg('get_features', 'TRAIN|TEST')");

				if (features)
					return sg_R.get_features(features);
				else
					SG_SERROR( "no features set\n");
			}
			/*
			 * This action returns the either the TEST or TRAIN labels 
			 * which were registered earlier.
			 *
			 */
			else if (!strncmp(action, N_GET_LABELS, strlen(N_GET_LABELS)))
			{
				CLabels* labels=NULL;
				CHAR* target=CHAR(STRING_ELT(CAR(args),0));

				if (!strncmp(target, "TRAIN", strlen("TRAIN")))
				{
					labels=gui->guilabels.get_train_labels();
				}
				else if (!strncmp(target, "TEST", strlen("TEST")))
				{
					labels=gui->guilabels.get_test_labels();
				}

				if (labels)
					return sg_R.get_labels(labels);
				else
					SG_SERROR( "usage is [lab]=sg('get_labels', 'TRAIN|TEST')");
			}
			else if (!strncmp(action, N_SET_LABELS, strlen(N_SET_LABELS)))
			{
				if (cmd_len==3)
				{ 
					CHAR* target=CHAR(STRING_ELT(CAR(args),0));
					// pop target out of arglist
					args = CDR(args);

					if ( (!strncmp(target, "TRAIN", strlen("TRAIN"))) || 
							(!strncmp(target, "TEST", strlen("TEST"))) )
					{

						SEXP labels_vec = CAR(args); // Maybe results in NULL pointer 
						args = CDR(args); /* pop labels out of list */
						CLabels* labels=sg_R.set_labels(labels_vec);

						if (labels && target)
						{
							if (!strncmp(target, "TRAIN", strlen("TRAIN")))
								successful(result, gui->guilabels.set_train_labels(labels));
							else if (!strncmp(target, "TEST", strlen("TEST")))
								successful(result, gui->guilabels.set_test_labels(labels));
						}
						else
							SG_SERROR( "usage is sg('set_labels', 'TRAIN|TEST', labels)");
					}
					else
						SG_SERROR( "usage is sg('set_labels', 'TRAIN|TEST', labels)");
				}
				else
					SG_SERROR( "usage is sg('set_labels', 'TRAIN|TEST', labels)");
			}
			else if (!strncmp(action, N_GET_SUBKERNEL_WEIGHTS, strlen(N_GET_SUBKERNEL_WEIGHTS)))
			{
				return sg_R.get_subkernel_weights();
			}
			else if (!strncmp(action, N_GET_KERNEL_MATRIX, strlen(N_GET_KERNEL_MATRIX)))
			{
				return sg_R.get_kernel_matrix();
			}
			else
				SG_SERROR( "unrecognized command. type help for options\n");
		}
		else
			SG_SERROR("No input arguments supplied.");

	} // function sg
	else
		SG_SERROR("No input arguments supplied.");
	return result;
}

/* The main function of the shogun R interface. All commands from the R command line
 * to the shogun backend are passed using the syntax:
 * .External("sg", "func", ... ) 
 * where '...' is a number of arguments passed to the shogun function 'func'. */

SEXP sg(SEXP args)
{
	/* The SEXP (Simple Expression) args is a list of arguments of the .External call. 
	 * it consists of "sg", "func" and additional arguments.
	 * */


#ifndef WIN32
    CSignal::set_handler();
#endif

	if (!gui)
		gui=new CTextGUI(0, NULL);

	if (!gui)
		SG_SERROR("gui could not be initialized.");

	SEXP result=sg_helper(args);
#ifndef WIN32
    CSignal::unset_handler();
#endif
	return result;
}

} // extern C


/* This method is called by R when the shogun module is loaded into R
 * via dyn.load('sg.so'). */

void R_init_sg(DllInfo *info) { 
   
   /* There are four different external language call mechanisms available in R, namely:
    *    .C
    *    .Call
    *    .Fortran
    *    .External
    *
    * Currently shogun uses only the .External interface. */

   R_CMethodDef cMethods[] = { {NULL, NULL, 0} };
   R_FortranMethodDef fortranMethods[] = { {NULL, NULL, 0} };
   R_ExternalMethodDef externalMethods[] = { {NULL, NULL, 0} };

   R_CallMethodDef callMethods[] = {
      {"sg", (void*(*)()) &sg, 1},
      {NULL, NULL, 0} };

   /* Register the routines saved in the callMethods structure so that they are available under R. */
   R_registerRoutines(info, cMethods, callMethods, (R_FortranMethodDef*) fortranMethods, (R_ExternalMethodDef*) externalMethods);

}


/* This method is called form within R when the current module is unregistered.
 * Note that R does not allow unregistering of single symbols. */

void R_unload_sg(DllInfo *info) { }

#endif //HAVE_SWIG
