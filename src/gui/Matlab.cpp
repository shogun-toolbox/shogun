#include "lib/config.h"

#ifdef HAVE_MATLAB
#include <stdio.h>
#include <string.h>

#include "lib/common.h"
#include "lib/io.h"
#include "mex.h"

#include "guilib/GUIMatlab.h"
#include "gui/TextGUI.h"
#include "gui/GUI.h"

static CGUIMatlab gf_matlab;
extern CTextGUI* gui;

static const CHAR* N_SEND_COMMAND=		"send_command";
static const CHAR* N_HELP=		        "help";
static const CHAR* N_CRC=			"crc";
static const CHAR* N_TRANSLATE_STRING=			"translate_string";
static const CHAR* N_GET_HMM=			"get_hmm";
static const CHAR* N_GET_VITERBI_PATH=			"get_viterbi_path";
static const CHAR* N_GET_SVM=			"get_svm";
static const CHAR* N_GET_KERNEL_INIT=	        "get_kernel_init";
static const CHAR* N_GET_KERNEL_MATRIX=	        "get_kernel_matrix";
static const CHAR* N_GET_KERNEL_OPTIMIZATION=	        "get_kernel_optimization";
static const CHAR* N_COMPUTE_WD_BY_LEVELS=	        "compute_WD_by_levels";
static const CHAR* N_GET_FEATURES=		"get_features";
static const CHAR* N_GET_LABELS=		"get_labels";
static const CHAR* N_GET_VERSION=		"get_version";
static const CHAR* N_GET_PREPROC_INIT=	        "get_preproc_init";
static const CHAR* N_GET_HMM_DEFS=		"get_hmm_defs";
static const CHAR* N_SET_HMM=			"set_hmm";
static const CHAR* N_MODEL_PROB_NO_B_TRANS=			"model_prob_no_b_trans";
static const CHAR* N_BEST_PATH_NO_B_TRANS=			"best_path_no_b_trans";
static const CHAR* N_BEST_PATH_TRANS=			"best_path_trans";
static const CHAR* N_BEST_PATH_NO_B=			"best_path_no_b";
static const CHAR* N_APPEND_HMM=			"append_hmm";
static const CHAR* N_SET_SVM=			"set_svm";
static const CHAR* N_SET_KERNEL_PARAMETERS=	        "set_kernel_parameters";
static const CHAR* N_SET_KERNEL_INIT=	        "set_kernel_init";
static const CHAR* N_SET_FEATURES=		"set_features";
static const CHAR* N_ADD_FEATURES=		"add_features";
static const CHAR* N_CLEAN_FEATURES=		"clean_features";
static const CHAR* N_SET_LABELS=		"set_labels";
static const CHAR* N_SET_PREPROC_INIT=	        "set_preproc_init";
static const CHAR* N_SET_HMM_DEFS=		"set_hmm_defs";
static const CHAR* N_ONE_CLASS_HMM_CLASSIFY=		"one_class_hmm_classify";
static const CHAR* N_ONE_CLASS_LINEAR_HMM_CLASSIFY=		"one_class_linear_hmm_classify";
static const CHAR* N_HMM_CLASSIFY=		"hmm_classify";
static const CHAR* N_ONE_CLASS_HMM_CLASSIFY_EXAMPLE=		"one_class_hmm_classify_example";
static const CHAR* N_HMM_CLASSIFY_EXAMPLE=	"hmm_classify_example";
static const CHAR* N_SVM_CLASSIFY=		"svm_classify";
static const CHAR* N_SVM_CLASSIFY_EXAMPLE=	"svm_classify_example";
static const CHAR* N_GET_PLUGIN_ESTIMATE=	"get_plugin_estimate";
static const CHAR* N_SET_PLUGIN_ESTIMATE=	"set_plugin_estimate";
static const CHAR* N_PLUGIN_ESTIMATE_CLASSIFY=	"plugin_estimate_classify";
static const CHAR* N_PLUGIN_ESTIMATE_CLASSIFY_EXAMPLE=	"plugin_estimate_classify_example";

void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{
  //  fprintf(stderr,"gui=%ld\n", gui) ;
        if (!gui)
		gui=new CTextGUI(0, NULL);
	
	assert(gui);
	if (!nrhs)
	{
		//add some more text
		mexErrMsgTxt("No input arguments supplied.");
	} 

	CHAR* action=CGUIMatlab::get_mxString(prhs[0]);

	if (action)
	{
		if (!strncmp(action, N_SEND_COMMAND, strlen(N_SEND_COMMAND)))
		{
			if (nrhs==2)
			{
				CHAR* cmd=CGUIMatlab::get_mxString(prhs[1]);
				gf_matlab.send_command(cmd);
				delete[] cmd;
			}
			else
				mexErrMsgTxt("usage is gf('send_command', 'cmdline')");
		}
		else if (!strncmp(action, N_HELP, strlen(N_HELP)))
		{
			if (nrhs==1)
			{
				gf_matlab.send_command("help");
			}
			else
				mexErrMsgTxt("usage is gf('help')");
		}
		else if (!strncmp(action, N_GET_HMM, strlen(N_GET_HMM)))
		{
			if (nlhs==4)
			{
				gf_matlab.get_hmm(plhs);
			}
			else
				mexErrMsgTxt("usage is [p,q,a,b]=gf('get_hmm')");
		}
		else if (!strncmp(action, N_GET_VITERBI_PATH, strlen(N_GET_VITERBI_PATH)))
		{
			if ((nlhs==2) && (nrhs == 2))
			{
				if (mxIsDouble(prhs[1]))
				{
					double* dim=mxGetPr(prhs[1]);
					gf_matlab.best_path(plhs, (int) *dim);
				}
				else
					mexErrMsgTxt("usage is [path, lik]=gf('get_viterbi_path',dim)");
			}
			else
				mexErrMsgTxt("usage is [path, lik]=gf('get_viterbi_path',dim)");
		}
		else if (!strncmp(action, N_ONE_CLASS_HMM_CLASSIFY_EXAMPLE, strlen(N_ONE_CLASS_HMM_CLASSIFY_EXAMPLE)))
		  {
		    if (nlhs==1 && nrhs==2)
		      {
			if (mxIsDouble(prhs[1]))
			  {
			    double* idx=mxGetPr(prhs[1]);
			    gf_matlab.one_class_hmm_classify_example(plhs, (int) (*idx) );
			  }
			else
			  mexErrMsgTxt("usage is [result]=gf('hmm_classify_example', feature_vector_index)");
		      }
		    else
		      mexErrMsgTxt("usage is [result]=gf('one_class_hmm_classify_example', feature_vector_index)");
		}
		else if (!strncmp(action, N_ONE_CLASS_HMM_CLASSIFY, strlen(N_ONE_CLASS_HMM_CLASSIFY)))
		{
			if (nlhs==1)
				gf_matlab.one_class_hmm_classify(plhs, false);
			else
				mexErrMsgTxt("usage is [result]=gf('hmm_classify')");
		}
		else if (!strncmp(action, N_ONE_CLASS_LINEAR_HMM_CLASSIFY, strlen(N_ONE_CLASS_HMM_CLASSIFY)))
		{
			if (nlhs==1)
				gf_matlab.one_class_hmm_classify(plhs, true);
			else
				mexErrMsgTxt("usage is [result]=gf('hmm_classify')");
		}
		else if (!strncmp(action, N_HMM_CLASSIFY_EXAMPLE, strlen(N_HMM_CLASSIFY_EXAMPLE)))
		{
			if (nlhs==1 && nrhs==2)
			{
				if (mxIsDouble(prhs[1]))
				{
					double* idx=mxGetPr(prhs[1]);
					gf_matlab.hmm_classify_example(plhs, (int) (*idx) );
				}
				else
					mexErrMsgTxt("usage is [result]=gf('hmm_classify_example', feature_vector_index)");
			}
			else
				mexErrMsgTxt("usage is [result]=gf('hmm_classify_example', feature_vector_index)");
		}
		else if (!strncmp(action, N_HMM_CLASSIFY, strlen(N_HMM_CLASSIFY)))
		{
			if (nlhs==1)
				gf_matlab.hmm_classify(plhs);
			else
				mexErrMsgTxt("usage is [result]=gf('hmm_classify')");
		}
		else if (!strncmp(action, N_GET_SVM, strlen(N_GET_SVM)))
		{
			if (nlhs==2)
			{
				gf_matlab.get_svm(plhs);
			}
			else
				mexErrMsgTxt("usage is [b,alphas]=gf('get_svm')");
		}
		else if (!strncmp(action, N_SET_SVM, strlen(N_SET_SVM)))
		{
			if (nrhs==3)
			{
				gf_matlab.set_svm(prhs);
			}
			else
				mexErrMsgTxt("usage is gf('set_svm', [ b, alphas])");
		}
		else if (!strncmp(action, N_SVM_CLASSIFY_EXAMPLE, strlen(N_SVM_CLASSIFY_EXAMPLE)))
		{
			if (nlhs==1 && nrhs==2)
			{
				if (mxIsDouble(prhs[1]))
				{
				  double* idx=mxGetPr(prhs[1]);
				  if (!gf_matlab.svm_classify_example(plhs, (int) (*idx) ))
				    mexErrMsgTxt("svm_classify_example failed");
				}
				else
				mexErrMsgTxt("usage is [result]=gf('svm_classify_example', feature_vector_index)");
			}
			else
				mexErrMsgTxt("usage is [result]=gf('svm_classify_example', feature_vector_index)");
		}
		else if (!strncmp(action, N_SVM_CLASSIFY, strlen(N_SVM_CLASSIFY)))
		{
			if (nlhs==1)
			  {
			    if (!gf_matlab.svm_classify(plhs))
			      mexErrMsgTxt("svm_classify failed");
			  }
			else
				mexErrMsgTxt("usage is [result]=gf('svm_classify')");
		}
		else if (!strncmp(action, N_GET_PLUGIN_ESTIMATE, strlen(N_GET_PLUGIN_ESTIMATE)))
		{
			if (nlhs==2)
			{
				gf_matlab.get_plugin_estimate(plhs);
			}
			else
				mexErrMsgTxt("usage is [emission_probs, model_sizes]=gf('get_plugin_estimate')");
		}
		else if (!strncmp(action, N_SET_PLUGIN_ESTIMATE, strlen(N_SET_PLUGIN_ESTIMATE)))
		{
			if (nrhs==3)
			{
				gf_matlab.set_plugin_estimate(prhs);
			}
			else
				mexErrMsgTxt("usage is gf('set_plugin_estimate', emission_probs, model_sizes)");
		}
		else if (!strncmp(action, N_PLUGIN_ESTIMATE_CLASSIFY_EXAMPLE, strlen(N_PLUGIN_ESTIMATE_CLASSIFY_EXAMPLE)))
		{
			if (nlhs==1 && nrhs==2)
			{
				if (mxIsDouble(prhs[1]))
				{
					double* idx=mxGetPr(prhs[1]);
					gf_matlab.plugin_estimate_classify_example(plhs, (int) (*idx) );
				}
				else
				mexErrMsgTxt("usage is [result]=gf('plugin_estimate_classify_example', feature_vector_index)");
			}
			else
				mexErrMsgTxt("usage is [result]=gf('plugin_estimate_classify_example', feature_vector_index)");
		}
		else if (!strncmp(action, N_PLUGIN_ESTIMATE_CLASSIFY, strlen(N_PLUGIN_ESTIMATE_CLASSIFY)))
		{
			if (nlhs==1)
				gf_matlab.plugin_estimate_classify(plhs);
			else
				mexErrMsgTxt("usage is [result]=gf('plugin_estimate_classify')");
		}
		else if (!strncmp(action, N_GET_KERNEL_OPTIMIZATION, strlen(N_GET_KERNEL_OPTIMIZATION)))
		{
			if ((nlhs==1) && (nrhs==1))
				gf_matlab.get_kernel_optimization(plhs);
			else
				mexErrMsgTxt("usage is W=gf('get_kernel_optimization')");
		}
		else if (!strncmp(action, N_COMPUTE_WD_BY_LEVELS, strlen(N_COMPUTE_WD_BY_LEVELS)))
		{
			if ((nlhs==1) && (nrhs==1))
				gf_matlab.compute_WD_by_levels(plhs);
			else
				mexErrMsgTxt("usage is W=gf('compute_WD_by_levels')");
		}
		else if (!strncmp(action, N_GET_KERNEL_MATRIX, strlen(N_GET_KERNEL_MATRIX)))
		{
			if ((nlhs==1) && (nrhs==2))
			{
				CHAR* target=CGUIMatlab::get_mxString(prhs[1]);
				CFeatures* features=NULL;

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
					gf_matlab.get_kernel_matrix(plhs, features);
				else
					mexErrMsgTxt("usage is K=gf('get_kernel_matrix','TRAIN|TEST')");
			}
			else
				mexErrMsgTxt("usage is K=gf('get_kernel_matrix','TRAIN|TEST')");
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
					gf_matlab.get_features(plhs,features);
				else
					mexErrMsgTxt("usage is [features]=gf('get_features', 'TRAIN|TEST')");
			}
			else
				mexErrMsgTxt("usage is [features]=gf('get_features', 'TRAIN|TEST')");
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
					gf_matlab.get_labels(plhs,labels);
				else
					mexErrMsgTxt("usage is [lab]=gf('get_labels', 'TRAIN|TEST')");
			}
			else
				mexErrMsgTxt("usage is [lab]=gf('get_labels', 'TRAIN|TEST')");
		}
		else if (!strncmp(action, N_GET_VERSION, strlen(N_GET_VERSION)))
		{
			if (nrhs==1 && nlhs==1)
			{
					gf_matlab.get_version(plhs);
			}
			else
				mexErrMsgTxt("usage is [lab]=gf('get_version')");
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
				gf_matlab.best_path_no_b_trans(prhs,plhs);
			}
			else
				mexErrMsgTxt("usage is [prob,path]=gf('best_path_no_b_trans',p,q,a_trans,max_iter,nbest)");
		}
		else if (!strncmp(action, N_BEST_PATH_TRANS, strlen(N_BEST_PATH_TRANS)))
		{
			if ((nrhs==1+11) & (nlhs==3))
			{
				gf_matlab.best_path_trans(prhs,plhs);
			}
			else
				mexErrMsgTxt("usage is [prob,path,pos]=gf('best_path_trans',p,q,a_trans,seq,pos,orf_info, genestr, penalties, penalty_info, nbest, dict_weights)");
		}
		else if (!strncmp(action, N_MODEL_PROB_NO_B_TRANS, strlen(N_MODEL_PROB_NO_B_TRANS)))
		{
			if ((nrhs==1+4) & (nlhs==1))
			{
				gf_matlab.model_prob_no_b_trans(prhs,plhs);
			}
			else
				mexErrMsgTxt("usage is probs=gf('model_prob_no_b_trans',p,q,a_trans,max_iter)");
		}
		else if (!strncmp(action, N_BEST_PATH_NO_B, strlen(N_BEST_PATH_NO_B)))
		{
			if ((nrhs==1+4) & (nlhs==2))
			{
				gf_matlab.best_path_no_b(prhs,plhs);
			}
			else
				mexErrMsgTxt("usage is [prob,path]=gf('best_path_no_b',p,q,a,max_iter)");
		}
		else if (!strncmp(action, N_SET_HMM, strlen(N_SET_HMM)))
		{
			if (nrhs==1+4)
			{
				gf_matlab.set_hmm(prhs);
			}
			else
				mexErrMsgTxt("usage is gf('set_hmm',[p,q,a,b])");
		}
		else if (!strncmp(action, N_APPEND_HMM, strlen(N_APPEND_HMM)))
		{
			if (nrhs==1+4)
			{
				gf_matlab.append_hmm(prhs);
			}
			else
				mexErrMsgTxt("usage is gf('append_hmm',[p,q,a,b])");
		}
		else if (!strncmp(action, N_SET_SVM, strlen(N_SET_SVM)))
		{
			if (nrhs==1+2)
			{
				gf_matlab.set_svm(prhs);
			}
			else
				mexErrMsgTxt("usage is gf('set_svm',[b,alphas])");
		}
		else if (!strncmp(action, N_SET_KERNEL_PARAMETERS, strlen(N_SET_KERNEL_PARAMETERS)))
		{
			if (nlhs!=0 || nrhs!=2 || !gf_matlab.set_kernel_parameters(prhs[1]))
				mexErrMsgTxt("usage is gf('set_kernel_parameters',[parm])");

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
					CFeatures* features=gf_matlab.set_features(prhs, nrhs);

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
						mexErrMsgTxt("usage is gf('set_features', 'TRAIN|TEST', features, ...)");
				}
				else
					mexErrMsgTxt("usage is gf('set_features', 'TRAIN|TEST', features, ...)");
			}
			else
				mexErrMsgTxt("usage is gf('set_features', 'TRAIN|TEST', features, ...)");
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
					CFeatures* features=gf_matlab.set_features(prhs, nrhs);

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
					}
					else
						mexErrMsgTxt("usage is gf('add_features', 'TRAIN|TEST', features, ...)");
				}
				else
					mexErrMsgTxt("usage is gf('add_features', 'TRAIN|TEST', features, ...)");
			}
			else
				mexErrMsgTxt("usage is gf('set_features', 'TRAIN|TEST', features, ...)");
			CIO::message(M_INFO, "done\n");
		}
		else if (!strncmp(action, N_CLEAN_FEATURES, strlen(N_CLEAN_FEATURES)))
		{
			if (nrhs==2)
			{
				CHAR* target=CGUIMatlab::get_mxString(prhs[1]);

				if ( (!strncmp(target, "TRAIN", strlen("TRAIN"))) || 
						(!strncmp(target, "TEST", strlen("TEST"))) ) 
				{
					if (target)
					{
						if (!strncmp(target, "TRAIN", strlen("TRAIN")))
							gui->guifeatures.set_train_features(NULL);
						else if (!strncmp(target, "TEST", strlen("TEST")))
							gui->guifeatures.set_test_features(NULL);
						delete[] target;
					}
					else
						mexErrMsgTxt("usage is gf('clean_features', 'TRAIN|TEST')");
				}
				else
					mexErrMsgTxt("usage is gf('clean_features', 'TRAIN|TEST')");
			}
			else
				mexErrMsgTxt("usage is gf('clean_features', 'TRAIN|TEST')");
			CIO::message(M_INFO, "done\n");
		}
		else if (!strncmp(action, N_TRANSLATE_STRING, strlen(N_TRANSLATE_STRING)))
		{
			if ((nrhs==4) && (nlhs==1))
			{
				REAL* string=mxGetPr(prhs[1]);
				int len = mxGetN(prhs[1]) ;
				if (mxGetM(prhs[1])!=1 || mxGetN(prhs[2])!=1 || mxGetM(prhs[2])!=1 ||
					mxGetN(prhs[3])!=1 || mxGetM(prhs[3])!=1)
					mexErrMsgTxt("usage2 is translation=gf('translate_string', string, order, start)");
				REAL *p_order = mxGetPr(prhs[2]) ;
				REAL *p_start = mxGetPr(prhs[3]) ;
				INT order = (INT)p_order[0] ;
				INT start = (INT)p_start[0] ;
				const INT max_val = 2 ; /* DNA->2bits */
				
				plhs[0] = mxCreateDoubleMatrix(1, len, mxREAL);
				REAL* real_obs = mxGetPr(plhs[0]) ;
				
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
					default: mexErrMsgTxt("wrong letter") ;
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
					real_obs[i-start]=(REAL)obs[i];

				delete[] obs ;
			}
			else
				mexErrMsgTxt("usage is translation=gf('translate_string', string, order, start)");
			
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
				REAL * p=mxGetPr(plhs[0]) ;
				*p = res ;
				delete[] bstring ;
				mxFree(string) ;
			}
			else
				mexErrMsgTxt("usage is crc32=gf('crc', string)");
			
		}
		else if (!strncmp(action, N_SET_LABELS, strlen(N_SET_LABELS)))
		{
			if (nrhs==3)
			{ 
				CHAR* target=CGUIMatlab::get_mxString(prhs[1]);
				if ( (!strncmp(target, "TRAIN", strlen("TRAIN"))) || 
						(!strncmp(target, "TEST", strlen("TEST"))) )
				{
					CLabels* labels=gf_matlab.set_labels(prhs);

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
						mexErrMsgTxt("usage is gf('set_labels', 'TRAIN|TEST', labels)");
				}
				else
					mexErrMsgTxt("usage is gf('set_labels', 'TRAIN|TEST', labels)");
			}
			else
				mexErrMsgTxt("usage is gf('set_labels', 'TRAIN|TEST', labels)");
		}
		else if (!strncmp(action, N_SET_PREPROC_INIT, strlen(N_SET_PREPROC_INIT)))
		{
		}
		else if (!strncmp(action, N_SET_HMM_DEFS, strlen(N_SET_HMM_DEFS)))
		{
		}
		else
		{
			mexErrMsgTxt("action not defined");
		}

		delete[] action;
	}
	else
		mexErrMsgTxt("string expected as first argument");
}
#endif
