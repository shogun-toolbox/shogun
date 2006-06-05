#include "lib/config.h"

#ifdef HAVE_MATLAB
#include <stdio.h>
#include <string.h>

#include "lib/common.h"
#include "lib/io.h"
#include <mex.h>

#include "guilib/GUIMatlab.h"
#include "gui/TextGUI.h"
#include "gui/GUI.h"

static CGUIMatlab sg_matlab;
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
static const CHAR* N_RELATIVE_ENTROPY =	"relative_entropy";
static const CHAR* N_ENTROPY =	"entropy";
static const CHAR* N_CLASSIFY=		"classify";
static const CHAR* N_SVM_CLASSIFY=		"svm_classify";
static const CHAR* N_SVM_CLASSIFY_EXAMPLE=	"svm_classify_example";
static const CHAR* N_GET_PLUGIN_ESTIMATE=	"get_plugin_estimate";
static const CHAR* N_SET_PLUGIN_ESTIMATE=	"set_plugin_estimate";
static const CHAR* N_PLUGIN_ESTIMATE_CLASSIFY=	"plugin_estimate_classify";
static const CHAR* N_PLUGIN_ESTIMATE_CLASSIFY_EXAMPLE=	"plugin_estimate_classify_example";

void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{
	if (!gui)
		gui=new CTextGUI(0, NULL);

	CSignal::set_handler();

	if (!gui)
		CIO::message(M_ERROR,"gui could not be initialized.");

	if (!nrhs)
		CIO::message(M_ERROR,"No input arguments supplied.");

	CHAR* action=CGUIMatlab::get_mxString(prhs[0]);

	if (action)
	{
		if (!strncmp(action, N_SEND_COMMAND, strlen(N_SEND_COMMAND)))
		{
			if (nrhs==2)
			{
				CHAR* cmd=CGUIMatlab::get_mxString(prhs[1]);
				sg_matlab.send_command(cmd);
				delete[] cmd;
			}
			else
				CIO::message(M_ERROR, "usage is sg('send_command', 'cmdline')");
		}
		else if (!strncmp(action, N_HELP, strlen(N_HELP)))
		{
			if (nrhs==1)
			{
				sg_matlab.send_command("help");
			}
			else
				CIO::message(M_ERROR, "usage is sg('help')");
		}
		else if (!strncmp(action, N_GET_HMM, strlen(N_GET_HMM)))
		{
			if (nlhs==4)
			{
				sg_matlab.get_hmm(plhs);
			}
			else
				CIO::message(M_ERROR, "usage is [p,q,a,b]=sg('get_hmm')");
		}
		else if (!strncmp(action, N_GET_VITERBI_PATH, strlen(N_GET_VITERBI_PATH)))
		{
			if ((nlhs==2) && (nrhs == 2))
			{
				if (mxIsDouble(prhs[1]))
				{
					double* dim=mxGetPr(prhs[1]);
					sg_matlab.best_path(plhs, (int) *dim);
				}
				else
					CIO::message(M_ERROR, "usage is [path, lik]=sg('get_viterbi_path',dim)");
			}
			else
				CIO::message(M_ERROR, "usage is [path, lik]=sg('get_viterbi_path',dim)");
		}
		else if (!strncmp(action, N_ONE_CLASS_HMM_CLASSIFY_EXAMPLE, strlen(N_ONE_CLASS_HMM_CLASSIFY_EXAMPLE)))
		  {
		    if (nlhs==1 && nrhs==2)
		      {
			if (mxIsDouble(prhs[1]))
			  {
			    double* idx=mxGetPr(prhs[1]);
			    sg_matlab.one_class_hmm_classify_example(plhs, (int) (*idx) );
			  }
			else
			  CIO::message(M_ERROR, "usage is [result]=sg('hmm_classify_example', feature_vector_index)");
		      }
		    else
		      CIO::message(M_ERROR, "usage is [result]=sg('one_class_hmm_classify_example', feature_vector_index)");
		}
		else if (!strncmp(action, N_ONE_CLASS_HMM_CLASSIFY, strlen(N_ONE_CLASS_HMM_CLASSIFY)))
		{
			if (nlhs==1)
				sg_matlab.one_class_hmm_classify(plhs, false);
			else
				CIO::message(M_ERROR, "usage is [result]=sg('hmm_classify')");
		}
		else if (!strncmp(action, N_ONE_CLASS_LINEAR_HMM_CLASSIFY, strlen(N_ONE_CLASS_HMM_CLASSIFY)))
		{
			if (nlhs==1)
				sg_matlab.one_class_hmm_classify(plhs, true);
			else
				CIO::message(M_ERROR, "usage is [result]=sg('hmm_classify')");
		}
		else if (!strncmp(action, N_HMM_CLASSIFY_EXAMPLE, strlen(N_HMM_CLASSIFY_EXAMPLE)))
		{
			if (nlhs==1 && nrhs==2)
			{
				if (mxIsDouble(prhs[1]))
				{
					double* idx=mxGetPr(prhs[1]);
					sg_matlab.hmm_classify_example(plhs, (int) (*idx) );
				}
				else
					CIO::message(M_ERROR, "usage is [result]=sg('hmm_classify_example', feature_vector_index)");
			}
			else
				CIO::message(M_ERROR, "usage is [result]=sg('hmm_classify_example', feature_vector_index)");
		}
		else if (!strncmp(action, N_RELATIVE_ENTROPY, strlen(N_RELATIVE_ENTROPY)))
		{
			if (nlhs==1 && nrhs==1)
			{
					sg_matlab.relative_entropy(plhs);
			}
			else
				CIO::message(M_ERROR, "usage is [result]=sg('relative_entropy')");
		}
		else if (!strncmp(action, N_ENTROPY, strlen(N_ENTROPY)))
		{
			if (nlhs==1 && nrhs==1)
			{
					sg_matlab.entropy(plhs);
			}
			else
				CIO::message(M_ERROR, "usage is [result]=sg('permuation_entropy')");
		}
		else if (!strncmp(action, N_HMM_CLASSIFY, strlen(N_HMM_CLASSIFY)))
		{
			if (nlhs==1)
				sg_matlab.hmm_classify(plhs);
			else
				CIO::message(M_ERROR, "usage is [result]=sg('hmm_classify')");
		}
		else if (!strncmp(action, N_GET_SVM_OBJECTIVE, strlen(N_GET_SVM_OBJECTIVE)))
		{
			if (nlhs==1)
			{
				sg_matlab.get_svm_objective(plhs);
			}
			else
				CIO::message(M_ERROR, "usage is [obj]=sg('get_svm_objective')");
		}
		else if (!strncmp(action, N_GET_SVM, strlen(N_GET_SVM)))
		{
			if (nlhs==2)
			{
				sg_matlab.get_svm(plhs);
			}
			else
				CIO::message(M_ERROR, "usage is [b,alphas]=sg('get_svm')");
		}
		else if (!strncmp(action, N_SET_SVM, strlen(N_SET_SVM)))
		{
			if (nrhs==3)
			{
				sg_matlab.set_svm(prhs);
			}
			else
				CIO::message(M_ERROR, "usage is sg('set_svm', [ b, alphas])");
		}
		else if (!strncmp(action, N_SVM_CLASSIFY_EXAMPLE, strlen(N_SVM_CLASSIFY_EXAMPLE)))
		{
			if (nlhs==1 && nrhs==2)
			{
				if (mxIsDouble(prhs[1]))
				{
				  double* idx=mxGetPr(prhs[1]);
				  if (!sg_matlab.svm_classify_example(plhs, (int) (*idx) ))
				    CIO::message(M_ERROR, "svm_classify_example failed");
				}
				else
				CIO::message(M_ERROR, "usage is [result]=sg('svm_classify_example', feature_vector_index)");
			}
			else
				CIO::message(M_ERROR, "usage is [result]=sg('svm_classify_example', feature_vector_index)");
		}
		else if (!strncmp(action, N_CLASSIFY, strlen(N_CLASSIFY)))
		{
			if (nlhs==1)
			  {
			    if (!sg_matlab.classify(plhs))
			      CIO::message(M_ERROR, "classify failed");
			  }
			else
				CIO::message(M_ERROR, "usage is [result]=sg('classify')");
		}
		else if (!strncmp(action, N_SVM_CLASSIFY, strlen(N_SVM_CLASSIFY)))
		{
			if (nlhs==1)
			  {
			    if (!sg_matlab.svm_classify(plhs))
			      CIO::message(M_ERROR, "svm_classify failed");
			  }
			else
				CIO::message(M_ERROR, "usage is [result]=sg('svm_classify')");
		}
		else if (!strncmp(action, N_GET_PLUGIN_ESTIMATE, strlen(N_GET_PLUGIN_ESTIMATE)))
		{
			if (nlhs==2)
			{
				sg_matlab.get_plugin_estimate(plhs);
			}
			else
				CIO::message(M_ERROR, "usage is [emission_probs, model_sizes]=sg('get_plugin_estimate')");
		}
		else if (!strncmp(action, N_SET_PLUGIN_ESTIMATE, strlen(N_SET_PLUGIN_ESTIMATE)))
		{
			if (nrhs==3)
			{
				sg_matlab.set_plugin_estimate(prhs);
			}
			else
				CIO::message(M_ERROR, "usage is sg('set_plugin_estimate', emission_probs, model_sizes)");
		}
		else if (!strncmp(action, N_PLUGIN_ESTIMATE_CLASSIFY_EXAMPLE, strlen(N_PLUGIN_ESTIMATE_CLASSIFY_EXAMPLE)))
		{
			if (nlhs==1 && nrhs==2)
			{
				if (mxIsDouble(prhs[1]))
				{
					double* idx=mxGetPr(prhs[1]);
					sg_matlab.plugin_estimate_classify_example(plhs, (int) (*idx) );
				}
				else
				CIO::message(M_ERROR, "usage is [result]=sg('plugin_estimate_classify_example', feature_vector_index)");
			}
			else
				CIO::message(M_ERROR, "usage is [result]=sg('plugin_estimate_classify_example', feature_vector_index)");
		}
		else if (!strncmp(action, N_PLUGIN_ESTIMATE_CLASSIFY, strlen(N_PLUGIN_ESTIMATE_CLASSIFY)))
		{
			if (nlhs==1)
			{
				if (!sg_matlab.plugin_estimate_classify(plhs))
					CIO::message(M_ERROR, "error executing command");
			}
			else
				CIO::message(M_ERROR, "usage is [result]=sg('plugin_estimate_classify')");
		}
		else if (!strncmp(action, N_GET_KERNEL_OPTIMIZATION, strlen(N_GET_KERNEL_OPTIMIZATION)))
		{
			if ((nlhs==1) && (nrhs==1))
			{
				if (!sg_matlab.get_kernel_optimization(plhs))
					CIO::message(M_ERROR, "error executing command");
			}
			else
				CIO::message(M_ERROR, "usage is W=sg('get_kernel_optimization')");
		}
		else if (!strncmp(action, N_COMPUTE_BY_SUBKERNELS, strlen(N_COMPUTE_BY_SUBKERNELS)))
		{
			if ((nlhs==1) && (nrhs==1))
			{
				if (!sg_matlab.compute_by_subkernels(plhs))
					CIO::message(M_ERROR, "error executing command");
			}
			else
				CIO::message(M_ERROR, "usage is W=sg('compute_by_subkernels')");
		}
		else if (!strncmp(action, N_GET_LAST_SUBKERNEL_WEIGHTS, strlen(N_GET_LAST_SUBKERNEL_WEIGHTS)))
		{
			if ((nlhs==1) && (nrhs==1))
			{
				if (!sg_matlab.get_last_subkernel_weights(plhs))
					CIO::message(M_ERROR, "error executing command");
			}
			else
				CIO::message(M_ERROR, "usage is W=sg('get_last_subkernel_weights')");
		}
		else if (!strncmp(action, N_GET_SUBKERNEL_WEIGHTS, strlen(N_GET_SUBKERNEL_WEIGHTS)))
		{
			if ((nlhs==1) && (nrhs==1))
			{
				if (!sg_matlab.get_subkernel_weights(plhs))
					CIO::message(M_ERROR, "error executing command");
			}
			else
				CIO::message(M_ERROR, "usage is W=sg('get_subkernel_weights')");
		}
		else if (!strncmp(action, N_GET_WD_POS_WEIGHTS, strlen(N_GET_WD_POS_WEIGHTS)))
		{
			if ((nlhs==1) && (nrhs==1))
			{
				if (!sg_matlab.get_WD_position_weights(plhs))
					CIO::message(M_ERROR, "error executing command");
			}
			else
				CIO::message(M_ERROR, "usage is W=sg('get_WD_position_weights')");
		}
		else if (!strncmp(action, N_SET_LAST_SUBKERNEL_WEIGHTS, strlen(N_SET_LAST_SUBKERNEL_WEIGHTS)))
		{
			if ((nlhs==0) && (nrhs==2))
			{
				if (!sg_matlab.set_last_subkernel_weights(prhs[1]))
					CIO::message(M_ERROR, "error executing command");
			}
			else
				CIO::message(M_ERROR, "usage is sg('set_last_subkernel_weights', W)");
		}
		else if (!strncmp(action, N_SET_SUBKERNEL_WEIGHTS, strlen(N_SET_SUBKERNEL_WEIGHTS)))
		{
			if ((nlhs==0) && (nrhs==2))
			{
				if (!sg_matlab.set_subkernel_weights(prhs[1]))
					CIO::message(M_ERROR, "error executing command");
			}
			else
				CIO::message(M_ERROR, "usage is sg('set_subkernel_weights', W)");
		}
		else if (!strncmp(action, N_SET_WD_POS_WEIGHTS, strlen(N_SET_WD_POS_WEIGHTS)))
		{
			if ((nlhs==0) && (nrhs==2))
			{
				if (!sg_matlab.set_WD_position_weights(prhs[1]))
					CIO::message(M_ERROR, "error executing command");
			}
			else
				CIO::message(M_ERROR, "usage is sg('set_WD_position_weights', W)");
		}
		else if (!strncmp(action, N_GET_KERNEL_MATRIX, strlen(N_GET_KERNEL_MATRIX)))
		{
			if ((nlhs==1) && (nrhs==1))
				sg_matlab.get_kernel_matrix(plhs);
			else
				CIO::message(M_ERROR, "usage is K=sg('get_kernel_matrix')");
		}
		else if (!strncmp(action, N_GET_KERNEL_INIT, strlen(N_GET_KERNEL_INIT)))
		{
		}
		else if (!strncmp(action, N_GET_FEATURES, strlen(N_GET_FEATURES)))
		{
			if (nrhs==2 && nlhs==1)
			{
				CFeatures* features=NULL;
				CHAR* target=CGUIMatlab::get_mxString(prhs[1]);

				if (!strncmp(target, "TRAIN", strlen("TRAIN")))
				{
					features=gui->guifeatures.get_train_features();
				}
				else if (!strncmp(target, "TEST", strlen("TEST")))
				{
					features=gui->guifeatures.get_test_features();
				}
				delete[] target;

				if (features)
					sg_matlab.get_features(plhs,features);
				else
					CIO::message(M_ERROR, "usage is [features]=sg('get_features', 'TRAIN|TEST')");
			}
			else
				CIO::message(M_ERROR, "usage is [features]=sg('get_features', 'TRAIN|TEST')");
		}
		else if (!strncmp(action, N_GET_LABELS, strlen(N_GET_LABELS)))
		{
			if (nrhs==2 && nlhs==1)
			{
				CLabels* labels=NULL;
				CHAR* target=CGUIMatlab::get_mxString(prhs[1]);

				if (!strncmp(target, "TRAIN", strlen("TRAIN")))
				{
					labels=gui->guilabels.get_train_labels();
				}
				else if (!strncmp(target, "TEST", strlen("TEST")))
				{
					labels=gui->guilabels.get_test_labels();
				}
				delete[] target;

				if (labels)
					sg_matlab.get_labels(plhs,labels);
				else
					CIO::message(M_ERROR, "usage is [lab]=sg('get_labels', 'TRAIN|TEST')");
			}
			else
				CIO::message(M_ERROR, "usage is [lab]=sg('get_labels', 'TRAIN|TEST')");
		}
		else if (!strncmp(action, N_GET_VERSION, strlen(N_GET_VERSION)))
		{
			if (nrhs==1 && nlhs==1)
			{
					sg_matlab.get_version(plhs);
			}
			else
				CIO::message(M_ERROR, "usage is [lab]=sg('get_version')");
		}
		else if (!strncmp(action, N_GET_PREPROC_INIT, strlen(N_GET_PREPROC_INIT)))
		{
		}
		else if (!strncmp(action, N_GET_HMM_DEFS, strlen(N_GET_HMM_DEFS)))
		{
		}
		else if (!strncmp(action, N_BEST_PATH_NO_B_TRANS, strlen(N_BEST_PATH_NO_B_TRANS)))
		{
			if ((nrhs==1+5) & (nlhs==2))
			{
				sg_matlab.best_path_no_b_trans(prhs,plhs);
			}
			else
				CIO::message(M_ERROR, "usage is [prob,path]=sg('best_path_no_b_trans',p,q,a_trans,max_iter,nbest)");
		}
		else if (!strncmp(action, N_BEST_PATH_TRANS_SIMPLE, strlen(N_BEST_PATH_TRANS_SIMPLE)))
		{
			if ((nrhs==1+5) & (nlhs==2))
			{
				sg_matlab.best_path_trans_simple(prhs,plhs);
			}
			else
				CIO::message(M_ERROR, "usage is [prob,path]=sg('best_path_trans_simple', p, q, a_trans, seq, nbest)");
		}
		else if (!strncmp(action, N_BEST_PATH_TRANS, strlen(N_BEST_PATH_TRANS)))
		{
			if ((nrhs==1+12) & (nlhs==5))
			{
				sg_matlab.best_path_trans(prhs,plhs);
			}
			else
				CIO::message(M_ERROR, "usage is [prob,path,pos,PEN_values, PEN_input_values]=sg('best_path_trans',p,q,a_trans,seq,pos,orf_info, genestr, penalties, penalty_info, nbest, dict_weights, use_orf)");
		}
		else if (!strncmp(action, N_BEST_PATH_2STRUCT, strlen(N_BEST_PATH_2STRUCT)))
		{
			if ((nrhs==1+11) & (nlhs==5))
			{
				sg_matlab.best_path_2struct(prhs,plhs);
			}
			else
				CIO::message(M_ERROR, "usage is [prob,path,pos,PEN_values, PEN_input_values]=sg('best_path_2struct',p,q,a_trans,seq,pos, genestr, penalties, penalty_info, nbest, dict_weights, segment_sum_weights)");
		}
		else if (!strncmp(action, N_MODEL_PROB_NO_B_TRANS, strlen(N_MODEL_PROB_NO_B_TRANS)))
		{
			if ((nrhs==1+4) & (nlhs==1))
			{
				sg_matlab.model_prob_no_b_trans(prhs,plhs);
			}
			else
				CIO::message(M_ERROR, "usage is probs=sg('model_prob_no_b_trans',p,q,a_trans,max_iter)");
		}
		else if (!strncmp(action, N_BEST_PATH_NO_B, strlen(N_BEST_PATH_NO_B)))
		{
			if ((nrhs==1+4) & (nlhs==2))
			{
				sg_matlab.best_path_no_b(prhs,plhs);
			}
			else
				CIO::message(M_ERROR, "usage is [prob,path]=sg('best_path_no_b',p,q,a,max_iter)");
		}
		else if (!strncmp(action, N_SET_HMM, strlen(N_SET_HMM)))
		{
			if (nrhs==1+4)
			{
				sg_matlab.set_hmm(prhs);
			}
			else
				CIO::message(M_ERROR, "usage is sg('set_hmm',[p,q,a,b])");
		}
		else if (!strncmp(action, N_APPEND_HMM, strlen(N_APPEND_HMM)))
		{
			if (nrhs==1+4)
			{
				sg_matlab.append_hmm(prhs);
			}
			else
				CIO::message(M_ERROR, "usage is sg('append_hmm',[p,q,a,b])");
		}
		else if (!strncmp(action, N_SET_SVM, strlen(N_SET_SVM)))
		{
			if (nrhs==1+2)
			{
				sg_matlab.set_svm(prhs);
			}
			else
				CIO::message(M_ERROR, "usage is sg('set_svm',[b,alphas])");
		}
		else if (!strncmp(action, N_SET_KERNEL_PARAMETERS, strlen(N_SET_KERNEL_PARAMETERS)))
		{
			if (nlhs!=0 || nrhs!=2 || !sg_matlab.set_kernel_parameters(prhs[1]))
				CIO::message(M_ERROR, "usage is sg('set_kernel_parameters',[parm])");
		}
		else if (!strncmp(action, N_SET_CUSTOM_KERNEL, strlen(N_SET_CUSTOM_KERNEL)))
		{
			if (nlhs==0 && nrhs==3)
			{
				CHAR* target=CGUIMatlab::get_mxString(prhs[2]);

				if ( (!strncmp(target, "DIAG", strlen("DIAG"))) || 
						(!strncmp(target, "FULL", strlen("FULL"))) ) 
				{
					if (!strncmp(target, "FULL2DIAG", strlen("FULL2DIAG")))
					{
						sg_matlab.set_custom_kernel(prhs, false, true);
					}
					else if (!strncmp(target, "FULL", strlen("FULL")))
					{
						sg_matlab.set_custom_kernel(prhs, false, false);
					}
					else if (!strncmp(target, "DIAG", strlen("DIAG")))
					{
						sg_matlab.set_custom_kernel(prhs, true, true);
					}
				}
				else
					CIO::message(M_ERROR, "usage is sg('set_custom_kernel',[kernelmatrix, is_upperdiag])");
				delete[] target;
			}
			else
				CIO::message(M_ERROR, "usage is sg('set_custom_kernel',[kernelmatrix, is_upperdiag])");
		}
		else if (!strncmp(action, N_SET_KERNEL_INIT, strlen(N_SET_KERNEL_INIT)))
		{
		}
		else if (!strncmp(action, N_SET_FEATURES, strlen(N_SET_FEATURES)))
		{
			if (nrhs>=3)
			{
				CHAR* target=CGUIMatlab::get_mxString(prhs[1]);

				if ( (!strncmp(target, "TRAIN", strlen("TRAIN"))) || 
						(!strncmp(target, "TEST", strlen("TEST"))) ) 
				{
					CFeatures* features=sg_matlab.set_features(prhs, nrhs);

					if (features && target)
					{
						if (!strncmp(target, "TRAIN", strlen("TRAIN")))
						{
							gui->guifeatures.set_train_features(features);
						}
						else if (!strncmp(target, "TEST", strlen("TEST")))
						{
							gui->guifeatures.set_test_features(features);
						}
						delete[] target;
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
		else if (!strncmp(action, N_ADD_FEATURES, strlen(N_ADD_FEATURES)))
		{
			if (nrhs>=3)
			{
				CHAR* target=CGUIMatlab::get_mxString(prhs[1]);

				if ( (!strncmp(target, "TRAIN", strlen("TRAIN"))) || 
						(!strncmp(target, "TEST", strlen("TEST"))) ) 
				{
					CFeatures* features=sg_matlab.set_features(prhs, nrhs);

					if (features && target)
					{
						if (!strncmp(target, "TRAIN", strlen("TRAIN")))
						{
							gui->guifeatures.add_train_features(features);
						}
						else if (!strncmp(target, "TEST", strlen("TEST")))
						{
							gui->guifeatures.add_test_features(features);
						}
						delete[] target;
						target=NULL ;
					}
					else
						CIO::message(M_ERROR, "usage is sg('add_features', 'TRAIN|TEST', features, ...)");
				}
				else
					CIO::message(M_ERROR, "usage is sg('add_features', 'TRAIN|TEST', features, ...)");
				delete[] target;
				target=NULL ;
			}
			else
				CIO::message(M_ERROR, "usage is sg('add_features', 'TRAIN|TEST', features, ...)");
			CIO::message(M_INFO, "done\n");
		}
		else if (!strncmp(action, N_TRANSLATE_STRING, strlen(N_TRANSLATE_STRING)))
		{
			if ((nrhs==4) && (nlhs==1))
			{
				DREAL* string=mxGetPr(prhs[1]);
				int len = mxGetN(prhs[1]) ;
				if (mxGetM(prhs[1])!=1 || mxGetN(prhs[2])!=1 || mxGetM(prhs[2])!=1 ||
					mxGetN(prhs[3])!=1 || mxGetM(prhs[3])!=1)
					CIO::message(M_ERROR, "usage2 is translation=sg('translate_string', string, order, start)");
				DREAL *p_order = mxGetPr(prhs[2]) ;
				DREAL *p_start = mxGetPr(prhs[3]) ;
				INT order = (INT)p_order[0] ;
				INT start = (INT)p_start[0] ;
				const INT max_val = 2 ; /* DNA->2bits */
				
				plhs[0] = mxCreateDoubleMatrix(1, len, mxREAL);
				DREAL* real_obs = mxGetPr(plhs[0]) ;
				
				WORD* obs=new WORD[len] ;
				
				INT i,j;
				for (i=0; i<len; i++)
					switch ((char)string[i])
					{
					case 'A': obs[i]=0 ; break ;
					case 'C': obs[i]=1 ; break ;
					case 'G': obs[i]=2 ; break ;
					case 'T': obs[i]=3 ; break ;
					case 'a': obs[i]=0 ; break ;
					case 'c': obs[i]=1 ; break ;
					case 'g': obs[i]=2 ; break ;
					case 't': obs[i]=3 ; break ;
					default: CIO::message(M_ERROR, "wrong letter") ;
					}
				//mxFree(string) ;
				
				for (i=len-1; i>= ((int) order)-1; i--)	//convert interval of size T
				{
					WORD value=0;
					for (j=i; j>=i-((int) order)+1; j--)
						value= (value >> max_val) | ((obs[j]) << (max_val * (order-1)));
					
					obs[i]= (WORD) value;
				}
				
				for (i=order-2;i>=0;i--)
				{
					WORD value=0;
					for (j=i; j>=i-order+1; j--)
					{
						value= (value >> max_val);
						if (j>=0)
							value|=(obs[j]) << (max_val * (order-1));
					}
					obs[i]=value;
				}
				for (i=start; i<len; i++)	
					real_obs[i-start]=(DREAL)obs[i];

				delete[] obs ;
			}
			else
				CIO::message(M_ERROR, "usage is translation=sg('translate_string', string, order, start)");
			
		}
		else if (!strncmp(action, N_CRC, strlen(N_CRC)))
		{
			if ((nrhs==2) && (nlhs==1))
			{
				CHAR* string=CGUIMatlab::get_mxString(prhs[1]);
				UINT sl = strlen(string) ;
				
				BYTE* bstring = new BYTE[sl] ;
				for (UINT i=0; i<sl; i++)
					bstring[i] = string[i] ;
				UINT res = CMath::crc32(bstring, sl) ;
				plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
				DREAL * p=mxGetPr(plhs[0]) ;
				*p = res ;
				delete[] bstring ;
				mxFree(string) ;
			}
			else
				CIO::message(M_ERROR, "usage is crc32=sg('crc', string)");
			
		}
		else if (!strncmp(action, N_SET_LABELS, strlen(N_SET_LABELS)))
		{
			if (nrhs==3)
			{ 
				CHAR* target=CGUIMatlab::get_mxString(prhs[1]);
				if ( (!strncmp(target, "TRAIN", strlen("TRAIN"))) || 
						(!strncmp(target, "TEST", strlen("TEST"))) )
				{
					CLabels* labels=sg_matlab.set_labels(prhs);

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
						delete[] target;
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
		else if (!strncmp(action, N_SET_PREPROC_INIT, strlen(N_SET_PREPROC_INIT)))
		{
		}
		else if (!strncmp(action, N_SET_HMM_DEFS, strlen(N_SET_HMM_DEFS)))
		{
		}
		else
		{
			CIO::message(M_ERROR, "action not defined");
		}

		delete[] action;
	}
	else
		CIO::message(M_ERROR, "string expected as first argument");

	CSignal::unset_handler();
}
#endif
