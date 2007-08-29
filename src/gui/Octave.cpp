/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/config.h"

#if defined(HAVE_OCTAVE) && !defined(HAVE_SWIG)
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

#include "guilib/GUICommands.h"
#include "guilib/GUIOctave.h"
#include "gui/TextGUI.h"
#include "gui/GUI.h"

static CGUIOctave sg_octave;
extern CTextGUI* gui;

DEFUN_DLD (sg, prhs, nlhs, "shogun.")
{
	int nrhs = prhs.length();
	octave_value_list plhs;
	//int nlhs = plhs.length();

	CHAR* action = NULL;

	if (!gui)
		gui=new CTextGUI(0, NULL);

#ifndef WIN32
    CSignal::set_handler();
#endif

	ASSERT(gui);
	if (!nrhs)
		SG_SERROR( "No input arguments supplied.");
	else if (!prhs(0).is_string())
		SG_SERROR( "input should be string.");
	else
		action= CGUIOctave::get_octaveString(prhs(0).string_value());

	if (action)
	{
		if (!strncmp(action, N_SEND_COMMAND, strlen(N_SEND_COMMAND)))
		{
			if (nrhs==2)
			{
				CHAR* cmd=CGUIOctave::get_octaveString(prhs(1).string_value());
				sg_octave.send_command(cmd);
				delete[] cmd;
			}
			else
				SG_SERROR( "usage is sg('send_command', 'cmdline')");
		}
		else if (!strncmp(action, N_HELP, strlen(N_HELP)))
		{
			if (nrhs==1)
			{
				sg_octave.send_command("help");
			}
			else
				SG_SERROR( "usage is sg('help')");
		}
		else if (!strncmp(action, N_GET_HMM, strlen(N_GET_HMM)))
		{
			if (nlhs==4)
			{
				sg_octave.get_hmm(plhs);
			}
			else
				SG_SERROR( "usage is [p,q,a,b]=sg('get_hmm')");
		}
		else if (!strncmp(action, N_GET_VITERBI_PATH, strlen(N_GET_VITERBI_PATH)))
		{
			if ((nlhs==2) && (nrhs == 2))
			{
				if (prhs(1).is_real_scalar())
				{
					int dim = prhs(1).int_value();
					sg_octave.best_path(plhs, dim);
				}
				else
					SG_SERROR( "usage is [path, lik]=sg('get_viterbi_path',dim)");
			}
			else
				SG_SERROR( "usage is [path, lik]=sg('get_viterbi_path',dim)");
		}
		else if (!strncmp(action, N_HMM_LIKELIHOOD, strlen(N_HMM_LIKELIHOOD)))
		{
			if ( !((nlhs==1) && (nrhs == 1) && sg_octave.hmm_likelihood(plhs)) )
				SG_SERROR( "usage is [lik]=sg('hmm_likelihood')");
		}
		else if (!strncmp(action, N_ONE_CLASS_HMM_CLASSIFY_EXAMPLE, strlen(N_ONE_CLASS_HMM_CLASSIFY_EXAMPLE)))
		{
			if (nlhs==1 && nrhs==2)
			{
				if (prhs(1).is_real_scalar())
				{
					int idx = prhs(1).int_value();
					sg_octave.one_class_hmm_classify_example(plhs, idx);
				}
				else
					SG_SERROR( "usage is [result]=sg('hmm_classify_example', feature_vector_index)");
			}
			else
				SG_SERROR( "usage is [result]=sg('one_class_hmm_classify_example', feature_vector_index)");
		}
		else if (!strncmp(action, N_ONE_CLASS_HMM_CLASSIFY, strlen(N_ONE_CLASS_HMM_CLASSIFY)))
		{
			if (nlhs==1)
				sg_octave.one_class_hmm_classify(plhs, false);
			else
				SG_SERROR( "usage is [result]=sg('hmm_classify')");
		}
		else if (!strncmp(action, N_ONE_CLASS_LINEAR_HMM_CLASSIFY, strlen(N_ONE_CLASS_HMM_CLASSIFY)))
		{
			if (nlhs==1)
				sg_octave.one_class_hmm_classify(plhs, true);
			else
				SG_SERROR( "usage is [result]=sg('hmm_classify')");
		}
		else if (!strncmp(action, N_HMM_CLASSIFY_EXAMPLE, strlen(N_HMM_CLASSIFY_EXAMPLE)))
		{
			if (nlhs==1 && nrhs==2)
			{
				if (prhs(1).is_real_scalar())
				{
					int idx=prhs(1).int_value();
					sg_octave.hmm_classify_example(plhs, idx );
				}
				else
					SG_SERROR( "usage is [result]=sg('hmm_classify_example', feature_vector_index)");
			}
			else
				SG_SERROR( "usage is [result]=sg('hmm_classify_example', feature_vector_index)");
		}
		else if (!strncmp(action, N_HMM_CLASSIFY, strlen(N_HMM_CLASSIFY)))
		{
			if (nlhs==1)
				sg_octave.hmm_classify(plhs);
			else
				SG_SERROR( "usage is [result]=sg('hmm_classify')");
		}
		else if (!strncmp(action, N_GET_SVM, strlen(N_GET_SVM)))
		{
			if (nlhs==2)
			{
				sg_octave.get_svm(plhs);
			}
			else
				SG_SERROR( "usage is [b,alphas]=sg('get_svm')");
		}
		else if (!strncmp(action, N_SET_SVM, strlen(N_SET_SVM)))
		{
			if (nrhs==3)
			{
				sg_octave.set_svm(prhs);
			}
			else
				SG_SERROR( "usage is sg('set_svm', [ b, alphas])");
		}
		else if (!strncmp(action, N_SVM_CLASSIFY_EXAMPLE, strlen(N_SVM_CLASSIFY_EXAMPLE)))
		{
			if (nlhs==1 && nrhs==2)
			{
				if (prhs(1).is_real_scalar())
				{
					int idx = prhs(1).int_value();
					if (!sg_octave.classify_example(plhs, idx ))
						SG_SERROR( "svm_classify_example failed");
				}
				else
					SG_SERROR( "usage is [result]=sg('svm_classify_example', feature_vector_index)");
			}
			else
				SG_SERROR( "usage is [result]=sg('svm_classify_example', feature_vector_index)");
		}
		else if (!strncmp(action, N_SVM_CLASSIFY, strlen(N_SVM_CLASSIFY)))
		{
			if (nlhs==1)
			{
				if (!sg_octave.svm_classify(plhs))
					SG_SERROR( "svm_classify failed");
			}
			else
				SG_SERROR( "usage is [result]=sg('svm_classify')");
		}
		else if (!strncmp(action, N_CLASSIFY, strlen(N_CLASSIFY)))
		{
			if (nlhs==1)
			{
				if (!sg_octave.classify(plhs))
					SG_SERROR( "classify failed");
			}
			else
				SG_SERROR( "usage is [result]=sg('classify')");
		}
		else if (!strncmp(action, N_GET_PLUGIN_ESTIMATE, strlen(N_GET_PLUGIN_ESTIMATE)))
		{
			if (nlhs==2)
			{
				sg_octave.get_plugin_estimate(plhs);
			}
			else
				SG_SERROR( "usage is [emission_probs, model_sizes]=sg('get_plugin_estimate')");
		}
		else if (!strncmp(action, N_SET_PLUGIN_ESTIMATE, strlen(N_SET_PLUGIN_ESTIMATE)))
		{
			if (nrhs==3)
			{
				sg_octave.set_plugin_estimate(prhs);
			}
			else
				SG_SERROR( "usage is sg('set_plugin_estimate', emission_probs, model_sizes)");
		}
		else if (!strncmp(action, N_PLUGIN_ESTIMATE_CLASSIFY_EXAMPLE, strlen(N_PLUGIN_ESTIMATE_CLASSIFY_EXAMPLE)))
		{
			if (nlhs==1 && nrhs==2)
			{
				if (prhs(1).is_real_scalar())
				{
					int idx = prhs(1).int_value();
					sg_octave.plugin_estimate_classify_example(plhs, idx);
				}
				else
					SG_SERROR( "usage is [result]=sg('plugin_estimate_classify_example', feature_vector_index)");
			}
			else
				SG_SERROR( "usage is [result]=sg('plugin_estimate_classify_example', feature_vector_index)");
		}
		else if (!strncmp(action, N_PLUGIN_ESTIMATE_CLASSIFY, strlen(N_PLUGIN_ESTIMATE_CLASSIFY)))
		{
			if (nlhs==1)
				sg_octave.plugin_estimate_classify(plhs);
			else
				SG_SERROR( "usage is [result]=sg('plugin_estimate_classify')");
		}
		else if (!strncmp(action, N_GET_KERNEL_OPTIMIZATION, strlen(N_GET_KERNEL_OPTIMIZATION)))
		{
			if ((nlhs==1) && (nrhs==1))
				sg_octave.get_kernel_optimization(plhs);
			else
				SG_SERROR( "usage is W=sg('get_kernel_optimization')");
		}
		else if (!strncmp(action, N_GET_KERNEL_MATRIX, strlen(N_GET_KERNEL_MATRIX)))
		{
			if ((nlhs==1) && (nrhs==1))
				sg_octave.get_kernel_matrix(plhs);
			else
				SG_SERROR( "usage is K=sg('get_kernel_matrix')");
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
					sg_octave.get_features(plhs, features);
				else
					SG_SERROR( "usage is [features]=sg('get_features', 'TRAIN|TEST')");
			}
			else
				SG_SERROR( "usage is [features]=sg('get_features', 'TRAIN|TEST')");
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
					sg_octave.get_labels(plhs,labels);
				else
					SG_SERROR( "usage is [lab]=sg('get_labels', 'TRAIN|TEST')");
			}
			else
				SG_SERROR( "usage is [lab]=sg('get_labels', 'TRAIN|TEST')");
		}
		else if (!strncmp(action, N_GET_PREPROC_INIT, strlen(N_GET_PREPROC_INIT)))
		{
		}
		else if (!strncmp(action, N_GET_HMM_DEFS, strlen(N_GET_HMM_DEFS)))
		{
		}
		else if (!strncmp(action, N_SET_HMM, strlen(N_SET_HMM)))
		{
			if (nrhs==1+4)
			{
				sg_octave.set_hmm(prhs);
			}
			else
				SG_SERROR( "usage is sg('set_hmm',[p,q,a,b])");
		}
		else if (!strncmp(action, N_APPEND_HMM, strlen(N_APPEND_HMM)))
		{
			if (nrhs==1+4)
			{
				sg_octave.append_hmm(prhs);
			}
			else
				SG_SERROR( "usage is sg('append_hmm',[p,q,a,b])");
		}
		else if (!strncmp(action, N_SET_SVM, strlen(N_SET_SVM)))
		{
			if (nrhs==1+2)
			{
				sg_octave.set_svm(prhs);
			}
			else
				SG_SERROR( "usage is sg('set_svm',[b,alphas])");
		}
		else if (!strncmp(action, N_SET_KERNEL_INIT, strlen(N_SET_KERNEL_INIT)))
		{
		}
		else if (!strncmp(action, N_SET_FEATURES, strlen(N_SET_FEATURES)))
		{
			if (nrhs>=3)
			{
				CHAR* target=CGUIOctave::get_octaveString(prhs(1).string_value());

				if ( (!strncmp(target, "TRAIN", strlen("TRAIN"))) || 
						(!strncmp(target, "TEST", strlen("TEST"))) ) 
				{
					CFeatures* features=sg_octave.set_features(prhs);

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
						SG_SERROR( "usage is sg('set_features', 'TRAIN|TEST', features)");
				}
				else
					SG_SERROR( "usage is sg('set_features', 'TRAIN|TEST', features)");
			}
			else
				SG_SERROR( "usage is sg('set_features', 'TRAIN|TEST', features)");
			SG_SINFO( "done\n");
		}
		else if (!strncmp(action, N_ADD_FEATURES, strlen(N_ADD_FEATURES)))
		{
			if (nrhs>=3)
			{
				CHAR* target=CGUIOctave::get_octaveString(prhs(1).string_value());

				if ( (!strncmp(target, "TRAIN", strlen("TRAIN"))) || 
						(!strncmp(target, "TEST", strlen("TEST"))) ) 
				{
					CFeatures* features=sg_octave.set_features(prhs);

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
						SG_SERROR( "usage is sg('add_features', 'TRAIN|TEST', features)");
				}
				else
					SG_SERROR( "usage is sg('add_features', 'TRAIN|TEST', features)");
			}
			else
				SG_SERROR( "usage is sg('set_features', 'TRAIN|TEST', features)");
			SG_SINFO( "done\n");
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
					SG_SERROR( "usage2 is translation=sg('translate_string', string, order, start)");
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
						default: SG_SERROR( "wrong letter") ;
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
					real_obs(i-start)=(DREAL)obs[i];

				delete[] obs ;
				plhs(0) = real_obs;
			}
			else
				SG_SERROR( "usage is translation=sg('translate_string', string, order, start)");

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

				UINT res = CMath::crc32(bstring, sl) ;
				plhs(0) = (double) res;

				delete[] bstring;
				delete[] string;
			}
			else
				SG_SERROR( "usage is crc32=sg('crc', string)");

		}
		else if (!strncmp(action, N_SET_LABELS, strlen(N_SET_LABELS)))
		{
			if (nrhs==3)
			{ 
				CHAR* target=CGUIOctave::get_octaveString(prhs(1).string_value());
				if ( (!strncmp(target, "TRAIN", strlen("TRAIN"))) || 
						(!strncmp(target, "TEST", strlen("TEST"))) )
				{
					CLabels* labels=sg_octave.set_labels(prhs);

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
						SG_SERROR( "usage is sg('set_labels', 'TRAIN|TEST', labels)");
				}
				else
					SG_SERROR( "usage is sg('set_labels', 'TRAIN|TEST', labels)");
			}
			else
				SG_SERROR( "usage is sg('set_labels', 'TRAIN|TEST', labels)");
		}
		else if (!strncmp(action, N_SET_PREPROC_INIT, strlen(N_SET_PREPROC_INIT)))
		{
		}
		else if (!strncmp(action, N_SET_HMM_DEFS, strlen(N_SET_HMM_DEFS)))
		{
		}
		else
		{
			SG_SERROR( "action not defined");
		}

		delete[] action;
	}
	else
		SG_SERROR( "string expected as first argument");

#ifndef WIN32
    CSignal::unset_handler();
#endif

	return plhs;
}
#endif //HAVE_SWIG
