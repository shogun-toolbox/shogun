#include "lib/config.h"

#ifdef HAVE_OCTAVE
#include <stdio.h>
#include <string.h>

#include "lib/common.h"
#include "lib/io.h"

#include <octave/config.h>

#include <octave/defun-dld.h>
#include <octave/error.h>
#include <octave/oct-obj.h>
#include <octave/pager.h>
#include <octave/symtab.h>
#include <octave/variables.h>

#include "guilib/GUIOctave.h"
#include "gui/TextGUI.h"
#include "gui/GUI.h"

static CGUIOctave gf_octave;
extern CTextGUI* gui;

static const CHAR* N_SEND_COMMAND=		"send_command";
static const CHAR* N_HELP=		        "help";
static const CHAR* N_CRC=			"crc";
static const CHAR* N_TRANSLATE_STRING=			"translate_string";
static const CHAR* N_GET_HMM=			"get_hmm";
static const CHAR* N_GET_SVM=			"get_svm";
static const CHAR* N_GET_KERNEL_INIT=	        "get_kernel_init";
static const CHAR* N_GET_KERNEL_MATRIX=	        "get_kernel_matrix";
static const CHAR* N_GET_KERNEL_OPTIMIZATION=	        "get_kernel_optimization";
static const CHAR* N_GET_FEATURES=		"get_features";
static const CHAR* N_GET_LABELS=		"get_labels";
static const CHAR* N_GET_PREPROC_INIT=	        "get_preproc_init";
static const CHAR* N_GET_HMM_DEFS=		"get_hmm_defs";
static const CHAR* N_SET_HMM=			"set_hmm";
static const CHAR* N_MODEL_PROB_NO_B_TRANS=			"model_prob_no_b_trans";
static const CHAR* N_BEST_PATH_NO_B_TRANS=			"best_path_no_b_trans";
static const CHAR* N_BEST_PATH_TRANS=			"best_path_trans";
static const CHAR* N_BEST_PATH_NO_B=			"best_path_no_b";
static const CHAR* N_APPEND_HMM=			"append_hmm";
static const CHAR* N_SET_SVM=			"set_svm";
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

DEFUN_DLD (gf, prhs, nlhs, "genefinder.")
{
	int nrhs = prhs.length();
	octave_value_list plhs;
	//int nlhs = plhs.length();

	CHAR* action = NULL;

	if (!gui)
		gui=new CTextGUI(0, NULL);

	assert(gui);
	if (!nrhs)
		CIO::message(M_ERROR, "No input arguments supplied.");
	else if (!prhs(0).is_string())
		CIO::message(M_ERROR, "input should be string.");
	else
		action= CGUIOctave::get_octaveString(prhs(0).string_value());

	if (action)
	{
		if (!strncmp(action, N_SEND_COMMAND, strlen(N_SEND_COMMAND)))
		{
			if (nrhs==2)
			{
				CHAR* cmd=CGUIOctave::get_octaveString(prhs(1).string_value());
				gf_octave.send_command(cmd);
				delete[] cmd;
			}
			else
				CIO::message(M_ERROR, "usage is gf('send_command', 'cmdline')");
		}
		else if (!strncmp(action, N_HELP, strlen(N_HELP)))
		{
			if (nrhs==1)
			{
				gf_octave.send_command("help");
			}
			else
				CIO::message(M_ERROR, "usage is gf('help')");
		}
		else if (!strncmp(action, N_GET_HMM, strlen(N_GET_HMM)))
		{
			if (nlhs==4)
			{
				gf_octave.get_hmm(plhs);
			}
			else
				CIO::message(M_ERROR, "usage is [p,q,a,b]=gf('get_hmm')");
		}
		else if (!strncmp(action, N_ONE_CLASS_HMM_CLASSIFY_EXAMPLE, strlen(N_ONE_CLASS_HMM_CLASSIFY_EXAMPLE)))
		{
			if (nlhs==1 && nrhs==2)
			{
				if (prhs(1).is_real_scalar())
				{
					int idx = prhs(1).int_value();
					gf_octave.one_class_hmm_classify_example(plhs, idx);
				}
				else
					CIO::message(M_ERROR, "usage is [result]=gf('hmm_classify_example', feature_vector_index)");
			}
			else
				CIO::message(M_ERROR, "usage is [result]=gf('one_class_hmm_classify_example', feature_vector_index)");
		}
		else if (!strncmp(action, N_ONE_CLASS_HMM_CLASSIFY, strlen(N_ONE_CLASS_HMM_CLASSIFY)))
		{
			if (nlhs==1)
				gf_octave.one_class_hmm_classify(plhs, false);
			else
				CIO::message(M_ERROR, "usage is [result]=gf('hmm_classify')");
		}
		else if (!strncmp(action, N_ONE_CLASS_LINEAR_HMM_CLASSIFY, strlen(N_ONE_CLASS_HMM_CLASSIFY)))
		{
			if (nlhs==1)
				gf_octave.one_class_hmm_classify(plhs, true);
			else
				CIO::message(M_ERROR, "usage is [result]=gf('hmm_classify')");
		}
		else if (!strncmp(action, N_HMM_CLASSIFY_EXAMPLE, strlen(N_HMM_CLASSIFY_EXAMPLE)))
		{
			if (nlhs==1 && nrhs==2)
			{
				if (prhs(1).is_real_scalar())
				{
					int idx=prhs(1).int_value();
					gf_octave.hmm_classify_example(plhs, idx );
				}
				else
					CIO::message(M_ERROR, "usage is [result]=gf('hmm_classify_example', feature_vector_index)");
			}
			else
				CIO::message(M_ERROR, "usage is [result]=gf('hmm_classify_example', feature_vector_index)");
		}
		else if (!strncmp(action, N_HMM_CLASSIFY, strlen(N_HMM_CLASSIFY)))
		{
			if (nlhs==1)
				gf_octave.hmm_classify(plhs);
			else
				CIO::message(M_ERROR, "usage is [result]=gf('hmm_classify')");
		}
		else if (!strncmp(action, N_GET_SVM, strlen(N_GET_SVM)))
		{
			if (nlhs==2)
			{
				gf_octave.get_svm(plhs);
			}
			else
				CIO::message(M_ERROR, "usage is [b,alphas]=gf('get_svm')");
		}
		else if (!strncmp(action, N_SET_SVM, strlen(N_SET_SVM)))
		{
			if (nrhs==3)
			{
				gf_octave.set_svm(prhs);
			}
			else
				CIO::message(M_ERROR, "usage is gf('set_svm', [ b, alphas])");
		}
		else if (!strncmp(action, N_SVM_CLASSIFY_EXAMPLE, strlen(N_SVM_CLASSIFY_EXAMPLE)))
		{
			if (nlhs==1 && nrhs==2)
			{
				if (prhs(1).is_real_scalar())
				{
					int idx = prhs(1).int_value();
					if (!gf_octave.svm_classify_example(plhs, idx ))
						CIO::message(M_ERROR, "svm_classify_example failed");
				}
				else
					CIO::message(M_ERROR, "usage is [result]=gf('svm_classify_example', feature_vector_index)");
			}
			else
				CIO::message(M_ERROR, "usage is [result]=gf('svm_classify_example', feature_vector_index)");
		}
		else if (!strncmp(action, N_SVM_CLASSIFY, strlen(N_SVM_CLASSIFY)))
		{
			if (nlhs==1)
			{
				if (!gf_octave.svm_classify(plhs))
					CIO::message(M_ERROR, "svm_classify failed");
			}
			else
				CIO::message(M_ERROR, "usage is [result]=gf('svm_classify')");
		}
		else if (!strncmp(action, N_GET_PLUGIN_ESTIMATE, strlen(N_GET_PLUGIN_ESTIMATE)))
		{
			if (nlhs==2)
			{
				gf_octave.get_plugin_estimate(plhs);
			}
			else
				CIO::message(M_ERROR, "usage is [emission_probs, model_sizes]=gf('get_plugin_estimate')");
		}
		else if (!strncmp(action, N_SET_PLUGIN_ESTIMATE, strlen(N_SET_PLUGIN_ESTIMATE)))
		{
			if (nrhs==3)
			{
				gf_octave.set_plugin_estimate(prhs);
			}
			else
				CIO::message(M_ERROR, "usage is gf('set_plugin_estimate', emission_probs, model_sizes)");
		}
		else if (!strncmp(action, N_PLUGIN_ESTIMATE_CLASSIFY_EXAMPLE, strlen(N_PLUGIN_ESTIMATE_CLASSIFY_EXAMPLE)))
		{
			if (nlhs==1 && nrhs==2)
			{
				if (prhs(1).is_real_scalar())
				{
					int idx = prhs(1).int_value();
					gf_octave.plugin_estimate_classify_example(plhs, idx);
				}
				else
					CIO::message(M_ERROR, "usage is [result]=gf('plugin_estimate_classify_example', feature_vector_index)");
			}
			else
				CIO::message(M_ERROR, "usage is [result]=gf('plugin_estimate_classify_example', feature_vector_index)");
		}
		else if (!strncmp(action, N_PLUGIN_ESTIMATE_CLASSIFY, strlen(N_PLUGIN_ESTIMATE_CLASSIFY)))
		{
			if (nlhs==1)
				gf_octave.plugin_estimate_classify(plhs);
			else
				CIO::message(M_ERROR, "usage is [result]=gf('plugin_estimate_classify')");
		}
		else if (!strncmp(action, N_GET_KERNEL_OPTIMIZATION, strlen(N_GET_KERNEL_OPTIMIZATION)))
		{
			if ((nlhs==1) && (nrhs==1))
				gf_octave.get_kernel_optimization(plhs);
			else
				CIO::message(M_ERROR, "usage is W=gf('get_kernel_optimization')");
		}
		else if (!strncmp(action, N_GET_KERNEL_MATRIX, strlen(N_GET_KERNEL_MATRIX)))
		{
			if ((nlhs==1) && (nrhs==1))
				gf_octave.get_kernel_matrix(plhs);
			else
				CIO::message(M_ERROR, "usage is K=gf('get_kernel_matrix')");
		}
		else if (!strncmp(action, N_GET_KERNEL_INIT, strlen(N_GET_KERNEL_INIT)))
		{
		}
		else if (!strncmp(action, N_GET_FEATURES, strlen(N_GET_FEATURES)))
		{
			if (nrhs==2 && nlhs==1)
			{
				CFeatures* features=NULL;
				CHAR* target=CGUIOctave::get_octaveString(prhs(1).string_value());

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
					gf_octave.get_features(plhs, features);
				else
					CIO::message(M_ERROR, "usage is [features]=gf('get_features', 'TRAIN|TEST')");
			}
			else
				CIO::message(M_ERROR, "usage is [features]=gf('get_features', 'TRAIN|TEST')");
		}
		else if (!strncmp(action, N_GET_LABELS, strlen(N_GET_LABELS)))
		{
			if (nrhs==2 && nlhs==1)
			{
				CLabels* labels=NULL;
				CHAR* target=CGUIOctave::get_octaveString(prhs(1).string_value());

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
					gf_octave.get_labels(plhs,labels);
				else
					CIO::message(M_ERROR, "usage is [lab]=gf('get_labels', 'TRAIN|TEST')");
			}
			else
				CIO::message(M_ERROR, "usage is [lab]=gf('get_labels', 'TRAIN|TEST')");
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
				gf_octave.best_path_no_b_trans(prhs,plhs);
			}
			else
				CIO::message(M_ERROR, "usage is [prob,path]=gf('best_path_no_b_trans',p,q,a_trans,max_iter,nbest)");
		}
		else if (!strncmp(action, N_BEST_PATH_TRANS, strlen(N_BEST_PATH_TRANS)))
		{
			if ((nrhs==1+11) & (nlhs==3))
			{
				gf_octave.best_path_trans(prhs,plhs);
			}
			else
				CIO::message(M_ERROR, "usage is [prob,path,pos]=gf('best_path_trans',p,q,a_trans,seq,pos,orf_info, genestr, penalties, penalty_info, nbest, dict_weights)");
		}
		else if (!strncmp(action, N_MODEL_PROB_NO_B_TRANS, strlen(N_MODEL_PROB_NO_B_TRANS)))
		{
			if ((nrhs==1+4) & (nlhs==1))
			{
				gf_octave.model_prob_no_b_trans(prhs,plhs);
			}
			else
				CIO::message(M_ERROR, "usage is probs=gf('model_prob_no_b_trans',p,q,a_trans,max_iter)");
		}
		else if (!strncmp(action, N_BEST_PATH_NO_B, strlen(N_BEST_PATH_NO_B)))
		{
			if ((nrhs==1+4) & (nlhs==2))
			{
				gf_octave.best_path_no_b(prhs,plhs);
			}
			else
				CIO::message(M_ERROR, "usage is [prob,path]=gf('best_path_no_b',p,q,a,max_iter)");
		}
		else if (!strncmp(action, N_SET_HMM, strlen(N_SET_HMM)))
		{
			if (nrhs==1+4)
			{
				gf_octave.set_hmm(prhs);
			}
			else
				CIO::message(M_ERROR, "usage is gf('set_hmm',[p,q,a,b])");
		}
		else if (!strncmp(action, N_APPEND_HMM, strlen(N_APPEND_HMM)))
		{
			if (nrhs==1+4)
			{
				gf_octave.append_hmm(prhs);
			}
			else
				CIO::message(M_ERROR, "usage is gf('append_hmm',[p,q,a,b])");
		}
		else if (!strncmp(action, N_SET_SVM, strlen(N_SET_SVM)))
		{
			if (nrhs==1+2)
			{
				gf_octave.set_svm(prhs);
			}
			else
				CIO::message(M_ERROR, "usage is gf('set_svm',[b,alphas])");
		}
		else if (!strncmp(action, N_SET_KERNEL_INIT, strlen(N_SET_KERNEL_INIT)))
		{
		}
		else if (!strncmp(action, N_SET_FEATURES, strlen(N_SET_FEATURES)))
		{
			if (nrhs==3)
			{
				CHAR* target=CGUIOctave::get_octaveString(prhs(1).string_value());

				if ( (!strncmp(target, "TRAIN", strlen("TRAIN"))) || 
						(!strncmp(target, "TEST", strlen("TEST"))) ) 
				{
					CFeatures* features=gf_octave.set_features(prhs);

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
						CIO::message(M_ERROR, "usage is gf('set_features', 'TRAIN|TEST', features)");
				}
				else
					CIO::message(M_ERROR, "usage is gf('set_features', 'TRAIN|TEST', features)");
			}
			else
				CIO::message(M_ERROR, "usage is gf('set_features', 'TRAIN|TEST', features)");
			CIO::message(M_INFO, "done\n");
		}
		else if (!strncmp(action, N_ADD_FEATURES, strlen(N_ADD_FEATURES)))
		{
			if (nrhs==3)
			{
				CHAR* target=CGUIOctave::get_octaveString(prhs(1).string_value());

				if ( (!strncmp(target, "TRAIN", strlen("TRAIN"))) || 
						(!strncmp(target, "TEST", strlen("TEST"))) ) 
				{
					CFeatures* features=gf_octave.set_features(prhs);

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
						CIO::message(M_ERROR, "usage is gf('add_features', 'TRAIN|TEST', features)");
				}
				else
					CIO::message(M_ERROR, "usage is gf('add_features', 'TRAIN|TEST', features)");
			}
			else
				CIO::message(M_ERROR, "usage is gf('set_features', 'TRAIN|TEST', features)");
			CIO::message(M_INFO, "done\n");
		}
		else if (!strncmp(action, N_CLEAN_FEATURES, strlen(N_CLEAN_FEATURES)))
		{
			if (nrhs==2)
			{
				CHAR* target=CGUIOctave::get_octaveString(prhs(1).string_value());

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
						CIO::message(M_ERROR, "usage is gf('clean_features', 'TRAIN|TEST')");
				}
				else
					CIO::message(M_ERROR, "usage is gf('clean_features', 'TRAIN|TEST')");
			}
			else
				CIO::message(M_ERROR, "usage is gf('clean_features', 'TRAIN|TEST')");
			CIO::message(M_INFO, "done\n");
		}
		else if (!strncmp(action, N_TRANSLATE_STRING, strlen(N_TRANSLATE_STRING)))
		{
			if ((nrhs==4) && (nlhs==1))
			{
				RowVector string = prhs(1).row_vector_value();
				int len = string.cols();
				int order = prhs(2).int_value();
				int start = prhs(3).int_value();

				if (order==0 || start<0 || len <=0)
					CIO::message(M_ERROR, "usage2 is translation=gf('translate_string', string, order, start)");
				const INT max_val = 2 ; // DNA->2bits

				RowVector real_obs = RowVector(len);

				WORD* obs=new WORD[len] ;

				INT i,j;
				for (i=0; i<len; i++)
					switch ((char) string(i))
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
					real_obs(i-start)=(REAL)obs[i];

				delete[] obs ;
				plhs(0) = real_obs;
			}
			else
				CIO::message(M_ERROR, "usage is translation=gf('translate_string', string, order, start)");

		}
		else if (!strncmp(action, N_CRC, strlen(N_CRC)))
		{
			if ((nrhs==2) && (nlhs==1))
			{
				CHAR* string=CGUIOctave::get_octaveString(prhs(1).string_value());
				UINT sl = strlen(string) ;

				BYTE* bstring = new BYTE[sl] ;

				for (UINT i=0; i<sl; i++)
					bstring[i] = string[i];

				UINT res = math.crc32(bstring, sl) ;
				plhs(0) = (double) res;

				delete[] bstring;
				free(string);
			}
			else
				CIO::message(M_ERROR, "usage is crc32=gf('crc', string)");

		}
		else if (!strncmp(action, N_SET_LABELS, strlen(N_SET_LABELS)))
		{
			if (nrhs==3)
			{ 
				CHAR* target=CGUIOctave::get_octaveString(prhs(1).string_value());
				if ( (!strncmp(target, "TRAIN", strlen("TRAIN"))) || 
						(!strncmp(target, "TEST", strlen("TEST"))) )
				{
					CLabels* labels=gf_octave.set_labels(prhs);

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
						CIO::message(M_ERROR, "usage is gf('set_labels', 'TRAIN|TEST', labels)");
				}
				else
					CIO::message(M_ERROR, "usage is gf('set_labels', 'TRAIN|TEST', labels)");
			}
			else
				CIO::message(M_ERROR, "usage is gf('set_labels', 'TRAIN|TEST', labels)");
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

	return plhs;
}
#endif
