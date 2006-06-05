#include "lib/config.h"

#ifdef HAVE_R
#include <stdio.h>
#include <string.h>

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
static const CHAR* N_TRANSLATE_STRING=			"translate_string";
static const CHAR* N_GET_HMM=			"get_hmm";
static const CHAR* N_GET_VITERBI_PATH=			"get_viterbi_path";
static const CHAR* N_GET_SVM=			"get_svm";
static const CHAR* N_GET_SVM_OBJECTIVE=		"get_svm_objective";
static const CHAR* N_GET_KERNEL_INIT=	        "get_kernel_init";
static const CHAR* N_GET_KERNEL_MATRIX=	        "get_kernel_matrix";
static const CHAR* N_GET_KERNEL_OPTIMIZATION=	        "get_kernel_optimization";
static const CHAR* N_COMPUTE_BY_SUBKERNELS=	        "compute_by_subkernels";
static const CHAR* N_SET_SUBKERNEL_WEIGHTS=	        "set_subkernel_weights";
static const CHAR* N_SET_LAST_SUBKERNEL_WEIGHTS=	        "set_last_subkernel_weights";
static const CHAR* N_SET_WD_POS_WEIGHTS=	        "set_WD_position_weights";
static const CHAR* N_GET_SUBKERNEL_WEIGHTS=	        "get_subkernel_weights";
static const CHAR* N_GET_LAST_SUBKERNEL_WEIGHTS=	        "get_last_subkernel_weights";
static const CHAR* N_GET_WD_POS_WEIGHTS=	        "get_WD_position_weights";
static const CHAR* N_GET_FEATURES=		"get_features";
static const CHAR* N_GET_LABELS=		"get_labels";
static const CHAR* N_GET_VERSION=		"get_version";
static const CHAR* N_GET_PREPROC_INIT=	        "get_preproc_init";
static const CHAR* N_GET_HMM_DEFS=		"get_hmm_defs";
static const CHAR* N_SET_HMM=			"set_hmm";
static const CHAR* N_MODEL_PROB_NO_B_TRANS=			"model_prob_no_b_trans";
static const CHAR* N_BEST_PATH_NO_B_TRANS=			"best_path_no_b_trans";
static const CHAR* N_BEST_PATH_TRANS=			"best_path_trans";
static const CHAR* N_BEST_PATH_2STRUCT=			"best_path_2struct";
static const CHAR* N_BEST_PATH_TRANS_SIMPLE=			"best_path_trans_simple";
static const CHAR* N_BEST_PATH_NO_B=			"best_path_no_b";
static const CHAR* N_APPEND_HMM=			"append_hmm";
static const CHAR* N_SET_SVM=			"set_svm";
static const CHAR* N_SET_KERNEL_PARAMETERS=	        "set_kernel_parameters";
static const CHAR* N_SET_CUSTOM_KERNEL=	        "set_custom_kernel";
static const CHAR* N_SET_KERNEL_INIT=	        "set_kernel_init";
static const CHAR* N_SET_FEATURES=		"set_features";
static const CHAR* N_ADD_FEATURES=		"add_features";
static const CHAR* N_SET_LABELS=		"set_labels";
static const CHAR* N_SET_PREPROC_INIT=	        "set_preproc_init";
static const CHAR* N_SET_HMM_DEFS=		"set_hmm_defs";
static const CHAR* N_ONE_CLASS_HMM_CLASSIFY=		"one_class_hmm_classify";
static const CHAR* N_ONE_CLASS_LINEAR_HMM_CLASSIFY=		"one_class_linear_hmm_classify";
static const CHAR* N_HMM_CLASSIFY=		"hmm_classify";
static const CHAR* N_ONE_CLASS_HMM_CLASSIFY_EXAMPLE=		"one_class_hmm_classify_example";
static const CHAR* N_HMM_CLASSIFY_EXAMPLE=	"hmm_classify_example";
static const CHAR* N_CLASSIFY=		"classify";
static const CHAR* N_SVM_CLASSIFY=		"svm_classify";
static const CHAR* N_SVM_CLASSIFY_EXAMPLE=	"svm_classify_example";
static const CHAR* N_GET_PLUGIN_ESTIMATE=	"get_plugin_estimate";
static const CHAR* N_SET_PLUGIN_ESTIMATE=	"set_plugin_estimate";
static const CHAR* N_PLUGIN_ESTIMATE_CLASSIFY=	"plugin_estimate_classify";
static const CHAR* N_PLUGIN_ESTIMATE_CLASSIFY_EXAMPLE=	"plugin_estimate_classify_example";

extern "C" {
   
SEXP test(SEXP args) {
   Rprintf("command vector length is %d\n", length(args));

   int idx=0;
   int argLength = length(args);
   for(;idx <argLength; idx++) {
      args = CDR(args);

      if( TYPEOF(CAR(args)) == STRSXP ) {
         Rprintf("element number %i of args is %s\n", idx, CHAR(STRING_ELT(CAR(args),0)));
      }
      
      if( TYPEOF(CAR(args)) == REALSXP ) {
         for(int j=0; j<length(CAR(args)); j++) {
            Rprintf("element number %i of args element %i is %f\n", j, idx, REAL(CAR(args))[j] );
         }
      }
      
      if( TYPEOF(CAR(args)) == NILSXP ) {
         Rprintf("end of list");
      }
      
      Rprintf("type of elem %d is %d\n", idx, TYPEOF(CAR(args)));
   }
   return(R_NilValue);
}


SEXP matrix_test(SEXP args) {
   Rprintf("command vector length is %d\n", length(args));
   
   int argLength = length(args);
   Rprintf("argument length is %i\n", argLength);

   // pop matrix_test out
   args = CDR(args);

   int size = length(CAR(args));
   Rprintf("size is %i\n", size);
   int r = Rf_nrows(CAR(args));
   int c = Rf_ncols(CAR(args));
   Rprintf("int: num rows is %i\n", r);
   Rprintf("int: num cols is %i\n", c);
   //args = CDR(args);

   return(R_NilValue);
}

SEXP return_test(SEXP args) {
   SEXP ans, mat1, mat2, dim1, dim2;
   
   PROTECT( mat1 = allocMatrix(REALSXP, 10, 10) );
   PROTECT( dim1 = allocVector(INTSXP, 2) );
   INTEGER(dim1)[0] = 10;
   INTEGER(dim1)[1] = 10;
   setAttrib(mat1, R_DimSymbol, dim1);

   PROTECT( mat2 = allocMatrix(REALSXP, 5, 5) );
   PROTECT( dim2 = allocVector(INTSXP, 2) );
   INTEGER(dim2)[0] = 5;
   INTEGER(dim2)[1] = 5;
   setAttrib(mat2, R_DimSymbol, dim2);
   
   for(int i=0; i<10; i++) {
      for(int j=0; j<10; j++) {
         if( i < 5 && j < 5)
            REAL(mat2)[i+5*j] = i+j;
         REAL(mat1)[i+10*j] = i+j;
      }
   }

   ans = R_NilValue;
   PROTECT(ans);
   ans = CONS(mat1, ans);
   SET_TAG(ans, install("alphas"));
   //UNPROTECT(1);

   // go to the next element in the list
   
   //PROTECT(ans);
   ans = CONS(mat2, ans);
   SET_TAG(ans, install("b"));
   UNPROTECT(1);

   UNPROTECT(4);

   Rprintf("matrix created!\n");
   return ans;
}

SEXP sg_helper(SEXP args)
{
#ifdef DEBUG
	Rprintf("length of args %d\n", length(args));
#endif

	int cmd_len = length(args)-1;
	if ( cmd_len < 1 )
		CIO::message(M_ERROR,"No input arguments supplied.");

	args = CDR(args); /* pop "sg" out of list */
	CHAR* action=CHAR(VECTOR_ELT(CAR(args), 0));

#ifdef DEBUG
	Rprintf("action is %s\n", action);
#endif

	args = CDR(args); /* pop action out of list */

	if (action)
	{
		if (!strncmp(action, N_SEND_COMMAND, strlen(N_SEND_COMMAND)))
		{
			if (cmd_len==2)
			{
				CHAR* cmd=CHAR(STRING_ELT(CAR(args),0));
#ifdef DEBUG
				Rprintf("command is %s\n", cmd);
#endif
				sg_R.send_command(cmd);
			}
			else
				CIO::message(M_ERROR, "usage is sg('send_command', 'cmdline')");
		}

		else if (!strncmp(action, N_HELP, strlen(N_HELP)))
		{
			if (cmd_len==1)
			{
				sg_R.send_command("help");
			}
			else
				CIO::message(M_ERROR, "usage is sg('help')");
		}

		else if (!strncmp(action, N_GET_VERSION, strlen(N_GET_VERSION)))
		{
			return sg_R.get_version();
		}

		else if (!strncmp(action, N_GET_SVM, strlen(N_GET_SVM)))
		{
			return sg_R.get_svm();
		}

		/*
		   else if (!strncmp(action, N_SVM_CLASSIFY_EXAMPLE, strlen(N_SVM_CLASSIFY_EXAMPLE)))
		   {
		   if (cmd_len==2)
		   {
		   if ( isDouble(VECTOR_ELT,(VECTOR_ELT(args, 1)),1) )
		   {
		   double* idx=REALSXP(VECTOR_ELT(args, 1));

		   mxGetPr(prhs[1]);

		   if (!sg_R.svm_classify_example(plhs, (int) (*idx) ))
		   CIO::message(M_ERROR, "svm_classify_example failed");
		   }
		   else
		   CIO::message(M_ERROR, "usage is [result]=sg('svm_classify_example', feature_vector_index)");
		   }
		   else
		   CIO::message(M_ERROR, "usage is [result]=sg('svm_classify_example', feature_vector_index)");
		   }
		   */
		   else if (!strncmp(action, N_GET_HMM, strlen(N_GET_HMM)))
		   {
			   return sg_R.get_hmm();
			   //CIO::message(M_ERROR, "usage is [p,q,a,b]=sg('get_hmm')");
		   }

		   else if (!strncmp(action, N_SVM_CLASSIFY, strlen(N_SVM_CLASSIFY)))
		   {
			   return sg_R.svm_classify();
		   }

		   else if (!strncmp(action, N_HMM_CLASSIFY, strlen(N_HMM_CLASSIFY)))
		   {
			   return sg_R.hmm_classify();
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

				   CFeatures* features= sg_R.set_features(features_mat);

				   if (features)
				   {
					   if (!strncmp(target, "TRAIN", strlen("TRAIN")))
					   {
#ifdef DEBUG
						   Rprintf("Adding features.\n");
#endif
						   gui->guifeatures.add_train_features(features);
					   }
					   else if (!strncmp(target, "TEST", strlen("TEST")))
					   {
						   gui->guifeatures.add_test_features(features);
					   }
				   }
				   else
					   CIO::message(M_ERROR, "usage is sg('add_features', 'TRAIN|TEST', features, ...)");
			   }
			   else
				   CIO::message(M_ERROR, "usage is sg('add_features', 'TRAIN|TEST', features, ...)");
			   /*
				  }
				  else 
				  CIO::message(M_ERROR, "usage is sg('add_features', 'TRAIN|TEST', features, ...)");
				  CIO::message(M_INFO, "done\n");
				  */
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
					   CFeatures* features = sg_R.set_features(features_mat);

					   if (features)
					   {
						   if (!strncmp(target, "TRAIN", strlen("TRAIN")))
						   {
#ifdef DEBUG
							   Rprintf("Setting features.\n");
#endif
							   gui->guifeatures.set_train_features(features);
						   }
						   else if (!strncmp(target, "TEST", strlen("TEST")))
						   {
							   gui->guifeatures.set_test_features(features);
						   }
					   }
					   else
						   CIO::message(M_ERROR, "usage is sg('set_features', 'TRAIN|TEST', features, ...)");
				   }
				   else
					   CIO::message(M_ERROR, "usage is sg('set_features', 'TRAIN|TEST', features, ...)");
			   }
			   else
				   CIO::message(M_ERROR, "usage is sg('set_features', 'TRAIN|TEST', features, ...)");
			   CIO::message(M_INFO, "done\n");
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
			   //delete[] target;

			   if (features)
				   return sg_R.get_features(features);
			   else
				   CIO::message(M_ERROR, "usage is [features]=sg('get_features', 'TRAIN|TEST')");
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
			   //delete[] target;

			   if (labels)
				   return sg_R.get_labels(labels);
			   else
				   CIO::message(M_ERROR, "usage is [lab]=sg('get_labels', 'TRAIN|TEST')");
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
						   {
							   gui->guilabels.set_train_labels(labels);
						   }
						   else if (!strncmp(target, "TEST", strlen("TEST")))
						   {
							   gui->guilabels.set_test_labels(labels);
						   }
						   //delete[] target;
					   }
					   else
						   CIO::message(M_ERROR, "usage is sg('set_labels', 'TRAIN|TEST', labels)");
				   }
				   else
					   CIO::message(M_ERROR, "usage is sg('set_labels', 'TRAIN|TEST', labels)");
			   }
			   else
				   CIO::message(M_ERROR, "usage is sg('set_labels', 'TRAIN|TEST', labels)");
		   }

		   else if (!strncmp(action, N_GET_SUBKERNEL_WEIGHTS, strlen(N_GET_SUBKERNEL_WEIGHTS)))
		   {
			   return sg_R.get_subkernel_weights();
		   }



		   else if (!strncmp(action, N_GET_KERNEL_MATRIX, strlen(N_GET_KERNEL_MATRIX)))
		   {
			   return sg_R.get_kernel_matrix();
		   }

	}

	return(R_NilValue);
} // function sg

/* The main function of the shogun R interface. All commands from the R command line
 * to the shogun backend are passed using the syntax:
 * .External("sg", "func", ... ) 
 * where '...' is a number of arguments passed to the shogun function 'func'. */

SEXP sg(SEXP args)
{
	/* The SEXP (Simple Expression) args is a list of arguments of the .External call. 
	 * it consists of "sg", "func" and additional arguments.
	 * */

	CSignal::set_handler();

	if (!gui)
		gui=new CTextGUI(0, NULL);

	if (!gui)
		CIO::message(M_ERROR,"gui could not be initialized.");

	SEXP result=sg_helper(args);
	CSignal::unset_handler();
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
      {"test", (void*(*)()) &test, 1},
      {"matrix_test", (void*(*)()) &matrix_test, 1},
      {"return_test", (void*(*)()) &return_test, 1},
      {NULL, NULL, 0} };

   
   /* Register the routines saved in the callMethods structure so that they are available under R. */
   R_registerRoutines(info, cMethods, callMethods, (R_FortranMethodDef*) fortranMethods, (R_ExternalMethodDef*) externalMethods);

}


/* This method is called form within R when the current module is unregistered.
 * Note that R does not allow unregistering of single symbols. */

void R_unload_sg(DllInfo *info) { }

#endif
