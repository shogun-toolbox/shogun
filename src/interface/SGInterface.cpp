#include "lib/config.h"

#if !defined(HAVE_SWIG)

#include <string.h>
#include <stdlib.h>

#include "interface/SGInterface.h"
#include "lib/ShogunException.h"
#include "lib/Mathematics.h"
#include "guilib/GUICommands.h"
#include "gui/TextGUI.h"
#include "gui/GUI.h"
#include "classifier/svm/SVM.h"
#include "kernel/WeightedDegreePositionStringKernel.h"
#include "kernel/CommWordStringKernel.h"
#include "kernel/LinearKernel.h"
#include "kernel/SparseLinearKernel.h"

CSGInterface* interface=NULL;
extern CTextGUI* gui;

#define USAGE(method) "sg('" method "')"
#define USAGE_I(method, in) "sg('" method "', " in ")"
#define USAGE_O(method, out) "[" out "]=sg('" method "')"
#define USAGE_IO(method, in, out) "[" out "]=sg('" method "', " in ")"

static CSGInterfaceMethod sg_methods[] =
{
	{
		(CHAR*) N_GET_KERNEL_OPTIMIZATION,
		(&CSGInterface::a_get_kernel_optimization), 0,
		(CHAR*) USAGE_O(N_GET_KERNEL_OPTIMIZATION, "W")
	},
	{
		(CHAR*) N_PLUGIN_ESTIMATE_CLASSIFY_EXAMPLE,
		(&CSGInterface::a_plugin_estimate_classify_example), 0,
		(CHAR*) USAGE_IO(N_PLUGIN_ESTIMATE_CLASSIFY_EXAMPLE, "feature_vector_index", "result")
	},
	{
		(CHAR*) N_PLUGIN_ESTIMATE_CLASSIFY,
		(&CSGInterface::a_plugin_estimate_classify), 0,
		(CHAR*) USAGE_O(N_PLUGIN_ESTIMATE_CLASSIFY, "result")
	},
	{
		(CHAR*) N_SET_PLUGIN_ESTIMATE,
		(&CSGInterface::a_set_plugin_estimate), 0,
		(CHAR*) USAGE_I(N_SET_PLUGIN_ESTIMATE, "emission_probs, model_sizes")
	},
	{
		(CHAR*) N_GET_PLUGIN_ESTIMATE,
		(&CSGInterface::a_get_plugin_estimate), 0,
		(CHAR*) USAGE_O(N_GET_PLUGIN_ESTIMATE, "emission_probs, model_sizes")
	},
	{
		(CHAR*) N_CLASSIFY,
		(&CSGInterface::a_classify), 0,
		(CHAR*) USAGE_O(N_CLASSIFY, "result")
	},
	{
		(CHAR*) N_CLASSIFY_EXAMPLE,
		(&CSGInterface::a_classify_example), 0,
		(CHAR*) USAGE_IO(N_CLASSIFY_EXAMPLE, "feature_vector_index", "result")
	},
	{
		(CHAR*) N_SVM_CLASSIFY_EXAMPLE,
		(&CSGInterface::a_classify_example), 0,
		(CHAR*) USAGE_IO(N_SVM_CLASSIFY_EXAMPLE, "feature_vector_index", "result")
	},
	{
		(CHAR*) N_GET_CLASSIFIER,
		(&CSGInterface::a_get_classifier), 0,
		(CHAR*) USAGE_O(N_GET_CLASSIFIER, "bias, weights")
	},
	{
		(CHAR*) N_GET_SVM,
		(&CSGInterface::a_get_svm), 0,
		(CHAR*) USAGE_O(N_GET_SVM, "bias, alphas")
	},
	{
		(CHAR*) N_SET_SVM,
		(&CSGInterface::a_set_svm), 0,
		(CHAR*) USAGE_I(N_SET_SVM, "bias, alphas")
	},
	{
		(CHAR*) N_GET_SVM_OBJECTIVE,
		(&CSGInterface::a_get_svm_objective), 0,
		(CHAR*) USAGE_O(N_GET_SVM_OBJECTIVE, "objective")
	},
	{
		(CHAR*) N_RELATIVE_ENTROPY,
		(&CSGInterface::a_relative_entropy), 0,
		(CHAR*) USAGE_O(N_RELATIVE_ENTROPY, "result")
	},
	{
		(CHAR*) N_ENTROPY,
		(&CSGInterface::a_entropy), 0,
		(CHAR*) USAGE_O(N_ENTROPY, "result")
	},
	{
		(CHAR*) N_HMM_CLASSIFY,
		(&CSGInterface::a_hmm_classify), 0,
		(CHAR*) USAGE_O(N_HMM_CLASSIFY, "result")
	},
	{
		(CHAR*) N_ONE_CLASS_LINEAR_HMM_CLASSIFY,
		(&CSGInterface::a_one_class_linear_hmm_classify), 0,
		(CHAR*) USAGE_O(N_ONE_CLASS_LINEAR_HMM_CLASSIFY, "result")
	},
	{
		(CHAR*) N_ONE_CLASS_HMM_CLASSIFY,
		(&CSGInterface::a_one_class_hmm_classify), 0,
		(CHAR*) USAGE_O(N_ONE_CLASS_HMM_CLASSIFY, "result")
	},
	{
		(CHAR*) N_ONE_CLASS_HMM_CLASSIFY_EXAMPLE,
		(&CSGInterface::a_one_class_hmm_classify_example), 0,
		(CHAR*) USAGE_IO(N_ONE_CLASS_HMM_CLASSIFY_EXAMPLE, "feature_vector_inde", "result")
	},
	{
		(CHAR*) N_HMM_CLASSIFY_EXAMPLE,
		(&CSGInterface::a_hmm_classify_example), 0,
		(CHAR*) USAGE_IO(N_HMM_CLASSIFY_EXAMPLE, "feature_vector_index", "result")},
	{
		(CHAR*) N_HMM_LIKELIHOOD,
		(&CSGInterface::a_hmm_likelihood), 0,
		(CHAR*) USAGE_O(N_HMM_LIKELIHOOD, "likelihood")
	},
	{
		(CHAR*) N_GET_VITERBI_PATH,
		(&CSGInterface::a_get_viterbi_path), 0,
		(CHAR*) USAGE_IO(N_GET_VITERBI_PATH, "dim", "path, likelihood")
	},
	{
		(CHAR*) N_GET_HMM,
		(&CSGInterface::a_get_hmm), 0,
		(CHAR*) USAGE_O(N_GET_HMM, "p, q, a, b")
	},
	{
		(CHAR*) N_HELP,
		(&CSGInterface::a_help), 0,
		(CHAR*) USAGE(N_HELP)
	},
	{
		(CHAR*) "test",
		(&CSGInterface::a_test), 0,
		(CHAR*) USAGE_I("test", "arg")
	},
	{NULL, NULL, 0, NULL}        /* Sentinel */
};


CSGInterface::CSGInterface()
{
	m_lhs_counter=0;
	m_rhs_counter=0;
	m_nlhs=0;
	m_nrhs=0;
}

CSGInterface::~CSGInterface()
{
}

////////////////////////////////////////////////////////////////////////////
// simple get helper
////////////////////////////////////////////////////////////////////////////

INT CSGInterface::get_int_from_string()
{
	INT len=0;
	CHAR* str=get_string(len);
	return strtol(str, NULL, 10);
}

DREAL CSGInterface::get_real_from_string()
{
	INT len=0;
	CHAR* str=get_string(len);
	return strtod(str, NULL);
}

bool CSGInterface::get_bool_from_string()
{
	INT len=0;
	CHAR* str=get_string(len);
	return strtol(str, NULL, 10)!=0;
}


////////////////////////////////////////////////////////////////////////////
// handler
////////////////////////////////////////////////////////////////////////////

bool CSGInterface::handle()
{
	INT len=0;
	bool success=false;

#ifndef WIN32
	CSignal::set_handler();
#endif

	if (!gui)
		gui=new CTextGUI(0, NULL);
	if (!gui)
		SG_SERROR("GUI could not be initialized.\n");

	CHAR* action=NULL;
	try
	{
		action=interface->get_action(len);
	}
	catch (ShogunException e)
	{
		SG_SERROR("String expected as first argument: %s\n", e.get_exception_string());
	}

	SG_PRINT("action: %s, nlhs %d, nrhs %d\n", action, m_nlhs, m_nrhs);
	INT i=0;
	while (sg_methods[i].action)
	{
		if (strmatch(action, len, sg_methods[i].action))
		{
			if (!(interface->*(sg_methods[i].method))())
				SG_SERROR("Usage: %s\n", sg_methods[i].usage);
			else
			{
				success=true;
				break;
			}
		}
		i++;
	}

	// FIXME: invoke old interface
	if(!success && strmatch(action, len, N_SEND_COMMAND))
	{
		//parse_args(2, 0);
		CHAR* cmd=interface->get_string(len);
		SG_PRINT("cmd:%s\n", cmd);
		gui->parse_line(cmd);

		delete[] cmd;
		delete gui;
		success=true;
	}

#ifndef WIN32
	CSignal::unset_handler();
#endif

	delete[] action;
	return success;
}


////////////////////////////////////////////////////////////////////////////
// actions
////////////////////////////////////////////////////////////////////////////

bool CSGInterface::a_get_kernel_optimization()
{
	if (m_nlhs!=1 || m_nrhs<1)
		return false;

	CKernel* kernel=gui->guikernel.get_kernel();
	if (!kernel)
		SG_ERROR("No kernel defined.\n");

	switch (kernel->get_kernel_type())
	{
		case K_WEIGHTEDDEGREEPOS:
		{
			if (m_nrhs!=2)
				SG_ERROR("parameter missing\n");

			INT max_order=(INT) get_real();
			if ((max_order<1) || (max_order>12))
			{
				SG_WARNING( "max_order out of range 1..12 (%d). setting to 1\n", max_order);
				max_order=1;
			}

			CWeightedDegreePositionStringKernel* k = (CWeightedDegreePositionStringKernel *) kernel;
			CSVM* svm=(CSVM*) gui->guiclassifier.get_classifier();
			if (!svm)
				SG_ERROR("No SVM defined.\n");

			INT num_suppvec=svm->get_num_support_vectors();
			INT* sv_idx=new INT[num_suppvec];
			DREAL* sv_weight=new DREAL[num_suppvec];
			INT num_feat=-1;
			INT num_sym=-1;

			for (INT i=0; i<num_suppvec; i++)
			{
				sv_idx[i]=svm->get_support_vector(i);
				sv_weight[i]=svm->get_alpha(i);
			}

			DREAL* position_weights=k->extract_w(max_order, num_feat,
				num_sym, NULL, num_suppvec, sv_idx, sv_weight);
			set_real_matrix(position_weights, num_sym, num_feat);

			delete[] sv_idx;
			delete[] sv_weight;
			delete[] position_weights;
			return true;
		}

		case K_COMMWORDSTRING:
		case K_WEIGHTEDCOMMWORDSTRING:
		{
			CCommWordStringKernel* k=(CCommWordStringKernel*) kernel;
			INT len=0;
			DREAL* weights;
			k->get_dictionary(len, weights);
			set_real_matrix(weights, len, 1);

			delete[] weights;
			return true;
		}
		case K_LINEAR:
		{
			CLinearKernel* k=(CLinearKernel*) kernel;
			INT len=0;
			const double* weights=k->get_normal(len);

			set_real_matrix(weights, len, 1);

			return true;
		}
		case K_SPARSELINEAR:
		{
			CSparseLinearKernel* k=(CSparseLinearKernel*) kernel;
			INT len=0;
			const double* weights=k->get_normal(len);

			set_real_matrix(weights, len, 1);

			return true;
		}
		default:
			break;
	}

	return true;
}

bool CSGInterface::a_plugin_estimate_classify_example()
{
	if (m_nlhs!=1 || m_nrhs!=2)
		return false;

	INT idx=(INT) get_real();
	DREAL result=gui->guipluginestimate.classify_example(idx);

	set_real_matrix(&result, 1, 1);
	return true;
}

bool CSGInterface::a_plugin_estimate_classify()
{
	if (m_nlhs!=1 || m_nrhs!=1)
		return false;

	CFeatures* feat=gui->guifeatures.get_test_features();
	if (!feat)
		SG_ERROR("No features found.\n");

	INT num_vec=feat->get_num_vectors();
	DREAL* result=new DREAL[num_vec];
	ASSERT(result);
	CLabels* labels=gui->guipluginestimate.classify();

	for (INT i=0; i<num_vec; i++)
		result[i]=labels->get_label(i);

	set_real_matrix(result, 1, num_vec);

	delete labels;
	delete[] result;
	return true;
}

bool CSGInterface::a_set_plugin_estimate()
{
	if (m_nlhs!=0 || m_nrhs!=3)
		return false;

	DREAL* emission_probs;
	INT num_probs;
	INT num_vec;
	get_real_matrix(emission_probs, num_probs, num_vec);

	if (num_vec!=2)
	{
		delete[] emission_probs;
		SG_ERROR("Need at least 1 set of positive and 1 set of negative params.\n");
	}

	DREAL* pos_params=emission_probs;
	DREAL* neg_params=&(emission_probs[num_probs]);

	DREAL* model_sizes;
	INT len;
	get_real_vector(model_sizes, len);

	INT seq_length=(INT) model_sizes[0];
	INT num_symbols=(INT) model_sizes[1];
	if (num_probs!=seq_length*num_symbols)
	{
		delete[] emission_probs;
		delete[] model_sizes;
		SG_ERROR("Mismatch in number of emission probs and sequence length * number of symbols.\n");
	}

	gui->guipluginestimate.get_estimator()->set_model_params(
		pos_params, neg_params, seq_length, num_symbols);

	delete[] emission_probs;
	delete[] model_sizes;
	return true;
}

bool CSGInterface::a_get_plugin_estimate()
{
	if (m_nlhs!=2 || m_nrhs!=1)
		return false;

	DREAL* pos_params;
	DREAL* neg_params;
	INT num_params=0;
	INT seq_length=0;
	INT num_symbols=0;

	if (!gui->guipluginestimate.get_estimator()->get_model_params(
		pos_params, neg_params, seq_length, num_symbols))
		return false;

	num_params=seq_length*num_symbols;

	DREAL* result=new DREAL[num_params*2];
	ASSERT(result);
	for (INT i=0; i<num_params; i++)
		result[i]=pos_params[i];
	for (INT i=0; i<num_params; i++)
		result[i+num_params]=neg_params[i];
	set_real_matrix(result, num_params, 2);
	delete[] result;

	DREAL model_sizes[2];
	model_sizes[0]=(DREAL) seq_length;
	model_sizes[1]=(DREAL) num_symbols;
	set_real_matrix(model_sizes, 1, 2);

	return true;
}

bool CSGInterface::a_classify()
{
	if (m_nlhs!=1 || m_nrhs!=1)
		return false;

	CFeatures* feat=gui->guifeatures.get_test_features();
	if (!feat)
		SG_ERROR("No features found.\n");

	INT num_vec=feat->get_num_vectors();
	CLabels* labels=gui->guiclassifier.classify();
	if (!labels)
		SG_ERROR("Classify failed\n");

	DREAL* result=new DREAL[num_vec];
	ASSERT(result);
	for (INT i=0; i<num_vec; i++)
		result[i]=labels->get_label(i);

	set_real_matrix(result, 1, num_vec);

	delete labels;
	delete[] result;
	return true;
}

bool CSGInterface::a_classify_example()
{
	if (m_nlhs!=1 || m_nrhs!=2)
		return false;

	INT idx=(INT) get_real();
	DREAL result;

	if (!gui->guiclassifier.classify_example(idx, result))
		SG_ERROR("Classify_example failed.\n");

	set_real_matrix(&result, 1, 1);

	return true;
}

bool CSGInterface::a_get_classifier()
{
	if (m_nlhs!=2 || m_nrhs!=1)
		return false;

	DREAL* bias=NULL;
	DREAL* weights=NULL;
	INT rows=0;
	INT cols=0;
	INT brows=0;
	INT bcols=0;

	if (!gui->guiclassifier.get_trained_classifier(weights, rows, cols, bias, brows, bcols))
		return false;

	set_real_matrix(bias, brows, bcols);
	delete[] bias;
	set_real_matrix(weights, rows, cols);
	delete[] weights;

	return true;
}

bool CSGInterface::a_get_svm()
{
	return a_get_classifier();
}

bool CSGInterface::a_set_svm()
{
	if (m_nlhs!=0 || m_nrhs!=3)
		return false;

	DREAL bias=get_real();
	DREAL* alphas;
	INT num_feat_alphas;
	INT num_vec_alphas;
	get_real_matrix(alphas, num_feat_alphas, num_vec_alphas);

	if (!alphas)
		SG_ERROR("No proper alphas given.\n");
	if (num_vec_alphas!=2)
	{
		delete[] alphas;
		SG_ERROR("Not 2 vectors in alphas.\n");
	}

	CSVM* svm=(CSVM*) gui->guiclassifier.get_classifier();
	if (!svm)
		SG_ERROR("No SVM object available.\n");

	svm->create_new_model(num_feat_alphas);
	svm->set_bias(bias);

	INT num_support_vectors=svm->get_num_support_vectors();
	for (INT i=0; i<num_support_vectors; i++)
	{
		svm->set_alpha(i, alphas[i]);
		svm->set_support_vector(i, (INT) alphas[i+num_support_vectors]);
	}

	delete[] alphas;
	return true;
}

bool CSGInterface::a_get_svm_objective()
{
	if (m_nlhs!=1 || m_nrhs!=1)
		return false;

	CSVM* svm=(CSVM*) gui->guiclassifier.get_classifier();
	if (!svm)
		SG_ERROR("No SVM set.\n");

	DREAL objective=svm->get_objective();
	set_real_matrix(&objective, 1, 1);

	return true;
}

bool CSGInterface::a_relative_entropy()
{
	if (m_nlhs!=1 || m_nrhs!=1)
		return false;

	CHMM* pos=gui->guihmm.get_pos();
	CHMM* neg=gui->guihmm.get_neg();
	if (!pos || !neg)
		//return false;
		SG_ERROR("Set pos and neg HMM first!\n");

	INT pos_N=pos->get_N();
	INT neg_N=neg->get_N();
	INT pos_M=pos->get_M();
	INT neg_M=neg->get_M();
	if (pos_M!=neg_M || pos_N!=neg_N)
		//return false;
		SG_ERROR("Pos and neg HMM's differ in number of emissions or states.\n");

	DREAL* p=new DREAL[pos_M];
	ASSERT(p);
	DREAL* q=new DREAL[neg_M];
	ASSERT(q);
	DREAL* entropy=new DREAL[pos_N];
	ASSERT(entropy);

	for (INT i=0; i<pos_N; i++)
	{
		for (INT j=0; j<pos_M; j++)
		{
			p[j]=pos->get_b(i, j);
			q[j]=neg->get_b(i, j);
		}

		entropy[i]=CMath::relative_entropy(p, q, pos_M);
	}

	delete[] p;
	delete[] q;
	set_real_matrix(entropy, 1, pos_N);

	delete[] entropy;
	return true;
}

bool CSGInterface::a_entropy()
{
	if (m_nlhs!=1 || m_nrhs!=1)
		return false;

	CHMM* current=gui->guihmm.get_current();
	if (!current)
		//return false;
		SG_ERROR("Create HMM first!\n");

	INT N=current->get_N();
	INT M=current->get_M();
	DREAL* p=new DREAL[M];
	ASSERT(p);
	DREAL* entropy=new DREAL[N];
	ASSERT(entropy);

	for (INT i=0; i<N; i++)
	{
		for (INT j=0; j<M; j++)
			p[j]=current->get_b(i, j);

		entropy[i]=CMath::entropy(p, M);
	}

	delete[] p;
	set_real_matrix(entropy, 1, N);

	delete[] entropy;
	return true;
}

bool CSGInterface::a_hmm_classify()
{
	return do_hmm_classify(false, false);
}

bool CSGInterface::a_one_class_hmm_classify()
{
	return do_hmm_classify(false, true);
}

bool CSGInterface::a_one_class_linear_hmm_classify()
{
	return do_hmm_classify(true, true);
}

bool CSGInterface::do_hmm_classify(bool linear, bool one_class)
{
	if (m_nlhs!=1 || m_nrhs>1)
		return false;

	CFeatures* feat=gui->guifeatures.get_test_features();
	if (!feat)
		return false;

	INT num_vec=feat->get_num_vectors();
	CLabels* labels=NULL;

	if (linear) // must be one_class as well
	{
		labels=gui->guihmm.linear_one_class_classify();
	}
	else
	{
		if (one_class)
			labels=gui->guihmm.one_class_classify();
		else
			labels=gui->guihmm.classify();
	}
	if (!labels)
		return false;

	DREAL* result=new DREAL[num_vec];
	ASSERT(result);
	for (INT i=0; i<num_vec; i++)
		result[i]=labels->get_label(i);
	set_real_matrix(result, 1, num_vec);

	delete labels;
	delete[] result;
	return true;
}

bool CSGInterface::a_one_class_hmm_classify_example()
{
	return do_hmm_classify_example(true);
}

bool CSGInterface::a_hmm_classify_example()
{
	return do_hmm_classify_example(false);
}

bool CSGInterface::do_hmm_classify_example(bool one_class)
{
	if (m_nlhs!=1 || m_nrhs!=2)
		return false;

	INT idx=(INT) get_real();

	DREAL result;
	if (one_class)
		result=gui->guihmm.one_class_classify_example(idx);
	else
		result=gui->guihmm.classify_example(idx);
	set_real_matrix(&result, 1, 1);

	return true;
}

bool CSGInterface::a_hmm_likelihood()
{
	if (m_nlhs!=1 || m_nrhs!=1)
		return false;

	CHMM* h=gui->guihmm.get_current();
	if (!h)
		return false;

	DREAL likelihood=h->model_probability();
	set_real_matrix(&likelihood, 1, 1);

	return true;
}

bool CSGInterface::a_get_viterbi_path()
{
	if (m_nlhs!=2 || m_nrhs!=2)
		return false;

	INT dim=(INT) get_real(); // less hassle than requiring int from outside?
	SG_DEBUG("dim: %f\n", dim);

	CHMM* h=gui->guihmm.get_current();
	if (!h)
		return false;

	CFeatures* feat=gui->guifeatures.get_test_features();
	if (!feat || (feat->get_feature_class()!=C_STRING) ||
			(feat->get_feature_type()!=F_WORD))
		return false;

	h->set_observations((CStringFeatures<WORD>*) feat);

	INT num_feat;
	WORD* vec=((CStringFeatures<WORD>*) feat)->get_feature_vector(dim, num_feat);
	if (!vec || num_feat<=0)
		return false;

	SG_DEBUG( "computing viterbi path for vector %d (length %d)\n", dim, num_feat);
	DREAL likelihood;
	DREAL* path=(DREAL*) h->get_path(dim, likelihood);
	set_real_matrix(path, 1, num_feat);
	delete[] path;
	set_real_matrix(&likelihood, 1, 1);

	return true;
}

bool CSGInterface::a_get_hmm()
{
	if (m_nlhs!=4)
		return false;

	CHMM* h=gui->guihmm.get_current();
	if (!h)
		return false;

	INT N=h->get_N();
	INT M=h->get_M();
	INT i, j;

	DREAL* p=new DREAL[N];
	ASSERT(p);
	DREAL* q=new DREAL[N];
	ASSERT(q);
	for (i=0; i<N; i++)
	{
		p[i]=h->get_p(i);
		q[i]=h->get_q(i);
	}
	set_real_matrix(p, 1, N);
	delete[] p;
	set_real_matrix(q, 1, N);
	delete[] q;

	DREAL* a=new DREAL[N*N];
	ASSERT(a);
	for (i=0; i<N; i++)
		for (j=0; j<N; j++)
			a[i+j*N]=h->get_a(i, j);
	set_real_matrix(a, N, N);
	delete[] a;

	DREAL* b=new DREAL[N*M];
	ASSERT(b);
	for (i=0; i<N; i++)
		for (j=0; j<M; j++)
			b[i+j*N]=h->get_b(i, j);
	set_real_matrix(b, N, M);
	delete[] b;

	return true;
}

bool CSGInterface::a_help()
{
	if (m_nrhs!=1 || m_nlhs!=0)
		return false;

	gui->print_help();

	return true;
}

bool CSGInterface::a_test()
{
	if (m_nrhs<2)
		return false;

	/*
	   DREAL* vector;
	   INT len;

	   get_real_vector(vector, len);
	   for (INT i=0; i<len; i++) SG_PRINT("data %d: %f\n", i, vector[i]);
	   set_real_vector(vector, len);
	   delete[] vector;
	   */

	/*
	   TSparse<DREAL>* matrix;
	   INT num_feat, num_vec;

	   get_real_sparsematrix(matrix, num_feat, num_vec);
	   for (INT i=0; i<num_vec; i++)
	   {
	   for (INT j=0; j<num_feat; j++)
	   {
	   SG_PRINT("data %d, %d, %f\n", i, j, matrix[i].features[j].entry);
	   }
	   }

	   set_real_sparsematrix(matrix, num_feat, num_vec);
	   delete[] matrix;
	   */

	T_STRING<CHAR>* list;
	INT num_str;
	get_string_list(list, num_str);
	set_string_list(list, num_str);
	delete[] list;

	return true;
}
#endif // !HAVE_SWIG
