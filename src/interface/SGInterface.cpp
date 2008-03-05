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
#include "kernel/WeightedDegreeStringKernel.h"
#include "kernel/CommWordStringKernel.h"
#include "kernel/WeightedCommWordStringKernel.h"
#include "kernel/LinearKernel.h"
#include "kernel/SparseLinearKernel.h"
#include "kernel/CombinedKernel.h"

CSGInterface* interface=NULL;
extern CTextGUI* gui;

#define USAGE(method) "sg('" method "')"
#define USAGE_I(method, in) "sg('" method "', " in ")"
#define USAGE_O(method, out) "[" out "]=sg('" method "')"
#define USAGE_IO(method, in, out) "[" out "]=sg('" method "', " in ")"

static CSGInterfaceMethod sg_methods[]=
{
	{
		(CHAR*) N_GET_VERSION,
		(&CSGInterface::a_get_version),
		(CHAR*) USAGE_O(N_GET_VERSION, "version")
	},
	{
		(CHAR*) N_GET_LABELS,
		(&CSGInterface::a_get_labels),
		(CHAR*) USAGE_IO(N_GET_LABELS, "TRAIN|TEST", "labels")
	},
	{
		(CHAR*) N_GET_FEATURES,
		(&CSGInterface::a_get_features),
		(CHAR*) USAGE_IO(N_GET_FEATURES, "TRAIN|TEST", "features")
	},
	{
		(CHAR*) N_GET_DISTANCE_MATRIX,
		(&CSGInterface::a_get_distance_matrix),
		(CHAR*) USAGE_O(N_GET_DISTANCE_MATRIX, "D")
	},
	{
		(CHAR*) N_GET_KERNEL_MATRIX,
		(&CSGInterface::a_get_kernel_matrix),
		(CHAR*) USAGE_O(N_GET_KERNEL_MATRIX, "K")
	},
	{
		(CHAR*) N_SET_WD_POS_WEIGHTS,
		(&CSGInterface::a_set_WD_position_weights),
		(CHAR*) USAGE_I(N_SET_WD_POS_WEIGHTS, "W[, 'TRAIN|TEST']")
	},
	{
		(CHAR*) N_SET_SUBKERNEL_WEIGHTS,
		(&CSGInterface::a_set_subkernel_weights),
		(CHAR*) USAGE_I(N_SET_SUBKERNEL_WEIGHTS, "W")
	},
	{
		(CHAR*) N_SET_SUBKERNEL_WEIGHTS_COMBINED,
		(&CSGInterface::a_set_subkernel_weights_combined),
		(CHAR*) USAGE_I(N_SET_SUBKERNEL_WEIGHTS_COMBINED, "W, idx")
	},
	{
		(CHAR*) N_SET_LAST_SUBKERNEL_WEIGHTS,
		(&CSGInterface::a_set_last_subkernel_weights),
		(CHAR*) USAGE_I(N_SET_LAST_SUBKERNEL_WEIGHTS, "W")
	},
	{
		(CHAR*) N_GET_SPEC_CONSENSUS,
		(&CSGInterface::a_get_SPEC_consensus),
		(CHAR*) USAGE_O(N_GET_SPEC_CONSENSUS, "W")
	},
	{
		(CHAR*) N_GET_SPEC_SCORING,
		(&CSGInterface::a_get_SPEC_scoring),
		(CHAR*) USAGE_IO(N_GET_SPEC_SCORING, "max_order", "W")
	},
	{
		(CHAR*) N_GET_WD_CONSENSUS,
		(&CSGInterface::a_get_WD_consensus),
		(CHAR*) USAGE_O(N_GET_WD_CONSENSUS, "W")
	},
	{
		(CHAR*) N_COMPUTE_POIM_WD,
		(&CSGInterface::a_compute_POIM_WD),
		(CHAR*) USAGE_IO(N_COMPUTE_POIM_WD, "max_order, distribution", "W")
	},
	{
		(CHAR*) N_GET_WD_SCORING,
		(&CSGInterface::a_get_WD_scoring),
		(CHAR*) USAGE_IO(N_GET_WD_SCORING, "max_order", "W")
	},
	{
		(CHAR*) N_GET_WD_POS_WEIGHTS,
		(&CSGInterface::a_get_WD_position_weights),
		(CHAR*) USAGE_O(N_GET_WD_POS_WEIGHTS, "W")
	},
	{
		(CHAR*) N_GET_LAST_SUBKERNEL_WEIGHTS,
		(&CSGInterface::a_get_last_subkernel_weights),
		(CHAR*) USAGE_O(N_GET_LAST_SUBKERNEL_WEIGHTS, "W")
	},
	{
		(CHAR*) N_COMPUTE_BY_SUBKERNELS,
		(&CSGInterface::a_compute_by_subkernels),
		(CHAR*) USAGE_O(N_COMPUTE_BY_SUBKERNELS, "W")
	},
	{
		(CHAR*) N_GET_KERNEL_OPTIMIZATION,
		(&CSGInterface::a_get_kernel_optimization),
		(CHAR*) USAGE_O(N_GET_KERNEL_OPTIMIZATION, "W")
	},
	{
		(CHAR*) N_PLUGIN_ESTIMATE_CLASSIFY_EXAMPLE,
		(&CSGInterface::a_plugin_estimate_classify_example),
		(CHAR*) USAGE_IO(N_PLUGIN_ESTIMATE_CLASSIFY_EXAMPLE, "feature_vector_index", "result")
	},
	{
		(CHAR*) N_PLUGIN_ESTIMATE_CLASSIFY,
		(&CSGInterface::a_plugin_estimate_classify),
		(CHAR*) USAGE_O(N_PLUGIN_ESTIMATE_CLASSIFY, "result")
	},
	{
		(CHAR*) N_SET_PLUGIN_ESTIMATE,
		(&CSGInterface::a_set_plugin_estimate),
		(CHAR*) USAGE_I(N_SET_PLUGIN_ESTIMATE, "emission_probs, model_sizes")
	},
	{
		(CHAR*) N_GET_PLUGIN_ESTIMATE,
		(&CSGInterface::a_get_plugin_estimate),
		(CHAR*) USAGE_O(N_GET_PLUGIN_ESTIMATE, "emission_probs, model_sizes")
	},
	{
		(CHAR*) N_CLASSIFY,
		(&CSGInterface::a_classify),
		(CHAR*) USAGE_O(N_CLASSIFY, "result")
	},
	{
		(CHAR*) N_CLASSIFY_EXAMPLE,
		(&CSGInterface::a_classify_example),
		(CHAR*) USAGE_IO(N_CLASSIFY_EXAMPLE, "feature_vector_index", "result")
	},
	{
		(CHAR*) N_SVM_CLASSIFY_EXAMPLE,
		(&CSGInterface::a_classify_example),
		(CHAR*) USAGE_IO(N_SVM_CLASSIFY_EXAMPLE, "feature_vector_index", "result")
	},
	{
		(CHAR*) N_GET_CLASSIFIER,
		(&CSGInterface::a_get_classifier),
		(CHAR*) USAGE_O(N_GET_CLASSIFIER, "bias, weights")
	},
	{
		(CHAR*) N_GET_SVM,
		(&CSGInterface::a_get_svm),
		(CHAR*) USAGE_O(N_GET_SVM, "bias, alphas")
	},
	{
		(CHAR*) N_SET_SVM,
		(&CSGInterface::a_set_svm),
		(CHAR*) USAGE_I(N_SET_SVM, "bias, alphas")
	},
	{
		(CHAR*) N_GET_SVM_OBJECTIVE,
		(&CSGInterface::a_get_svm_objective),
		(CHAR*) USAGE_O(N_GET_SVM_OBJECTIVE, "objective")
	},
	{
		(CHAR*) N_RELATIVE_ENTROPY,
		(&CSGInterface::a_relative_entropy),
		(CHAR*) USAGE_O(N_RELATIVE_ENTROPY, "result")
	},
	{
		(CHAR*) N_ENTROPY,
		(&CSGInterface::a_entropy),
		(CHAR*) USAGE_O(N_ENTROPY, "result")
	},
	{
		(CHAR*) N_HMM_CLASSIFY,
		(&CSGInterface::a_hmm_classify),
		(CHAR*) USAGE_O(N_HMM_CLASSIFY, "result")
	},
	{
		(CHAR*) N_ONE_CLASS_LINEAR_HMM_CLASSIFY,
		(&CSGInterface::a_one_class_linear_hmm_classify),
		(CHAR*) USAGE_O(N_ONE_CLASS_LINEAR_HMM_CLASSIFY, "result")
	},
	{
		(CHAR*) N_ONE_CLASS_HMM_CLASSIFY,
		(&CSGInterface::a_one_class_hmm_classify),
		(CHAR*) USAGE_O(N_ONE_CLASS_HMM_CLASSIFY, "result")
	},
	{
		(CHAR*) N_ONE_CLASS_HMM_CLASSIFY_EXAMPLE,
		(&CSGInterface::a_one_class_hmm_classify_example),
		(CHAR*) USAGE_IO(N_ONE_CLASS_HMM_CLASSIFY_EXAMPLE, "feature_vector_inde", "result")
	},
	{
		(CHAR*) N_HMM_CLASSIFY_EXAMPLE,
		(&CSGInterface::a_hmm_classify_example),
		(CHAR*) USAGE_IO(N_HMM_CLASSIFY_EXAMPLE, "feature_vector_index", "result")},
	{
		(CHAR*) N_HMM_LIKELIHOOD,
		(&CSGInterface::a_hmm_likelihood),
		(CHAR*) USAGE_O(N_HMM_LIKELIHOOD, "likelihood")
	},
	{
		(CHAR*) N_GET_VITERBI_PATH,
		(&CSGInterface::a_get_viterbi_path),
		(CHAR*) USAGE_IO(N_GET_VITERBI_PATH, "dim", "path, likelihood")
	},
	{
		(CHAR*) N_GET_HMM,
		(&CSGInterface::a_get_hmm),
		(CHAR*) USAGE_O(N_GET_HMM, "p, q, a, b")
	},
	{
		(CHAR*) N_HELP,
		(&CSGInterface::a_help),
		(CHAR*) USAGE(N_HELP)
	},
	{
		(CHAR*) "test",
		(&CSGInterface::a_test),
		(CHAR*) USAGE_I("test", "arg")
	},
	{NULL, NULL, NULL}        /* Sentinel */
};


CSGInterface::CSGInterface()
 : m_lhs_counter(0), m_rhs_counter(0), m_nlhs(0), m_nrhs(0)
{
}

CSGInterface::~CSGInterface()
{
}

////////////////////////////////////////////////////////////////////////////
// actions
////////////////////////////////////////////////////////////////////////////

bool CSGInterface::a_get_version()
{
	if (m_nlhs!=1 || m_nrhs!=1)
		return false;

	DREAL* ver=(DREAL*) version.get_version_revision();
	set_real_vector(ver, 1);

	return true;
}

bool CSGInterface::a_get_labels()
{
	if (m_nlhs!=1 || m_nrhs!=2)
		return false;

	INT tlen=0;
	CHAR* target=get_string(tlen);
	CLabels* labels=NULL;

	if (strmatch(target, tlen, "TRAIN"))
		labels=gui->guilabels.get_train_labels();
	else if (strmatch(target, tlen, "TEST"))
		labels=gui->guilabels.get_test_labels();
	else
	{
		delete[] target;
		SG_ERROR("Unknown target, neither TRAIN nor TEST.\n");
	}
	delete[] target;

	if (!labels)
		SG_ERROR("No labels.\n");

	INT num_labels=labels->get_num_labels();
	DREAL* lab=new DREAL[num_labels];

	for (INT i=0; i<num_labels ; i++)
		lab[i]=labels->get_label(i);

	set_real_matrix(lab, 1, num_labels);
	delete[] lab;

	return true;
}

bool CSGInterface::a_get_features()
{
	if (m_nlhs!=1 || m_nrhs!=2)
		return false;

	INT tlen=0;
	CHAR* target=get_string(tlen);
	CFeatures* feat=NULL;

	if (strmatch(target, tlen, "TRAIN"))
		feat=gui->guifeatures.get_train_features();
	else if (strmatch(target, tlen, "TEST"))
		feat=gui->guifeatures.get_test_features();
	else
	{
		delete[] target;
		SG_ERROR("Unknown target, neither TRAIN nor TEST.\n");
	}
	delete[] target;

	switch (feat->get_feature_class())
	{
		case C_SIMPLE:
			switch (feat->get_feature_type())
			{
				case F_DREAL:
				{
					CRealFeatures* realfeat=(CRealFeatures*) feat;
					INT num_feat=realfeat->get_num_features();
					INT num_vec=realfeat->get_num_vectors();
					DREAL* result=new DREAL[num_feat*num_vec];
					ASSERT(result);

					for (INT i=0; i<num_vec; i++)
					{
						INT num_vfeat=0;
						bool free_vec=true;
						DREAL* vec=realfeat->get_feature_vector(
							i, num_vfeat, free_vec);
						ASSERT(num_vfeat==num_feat);

						for (INT j=0; j<num_vfeat; j++)
							result[num_feat*i+j]=vec[j];
						realfeat->free_feature_vector(vec, i, free_vec);
					}

					set_real_matrix(result, num_feat, num_vec);
					delete[] result;

					break;
				}

				case F_WORD:
				{
					CWordFeatures* wordfeat=(CWordFeatures*) feat;
					INT num_feat=wordfeat->get_num_features();
					INT num_vec=wordfeat->get_num_vectors();
					WORD* result=new WORD[num_feat*num_vec];
					ASSERT(result);

					for (INT i=0; i<num_vec; i++)
					{
						INT num_vfeat=0;
						bool free_vec=true;
						WORD* vec=wordfeat->get_feature_vector(i, num_vfeat, free_vec);
						ASSERT(num_vfeat==num_feat);

						for (INT j=0; j<num_vfeat; j++)
							result[num_feat*i+j]=vec[j];
						wordfeat->free_feature_vector(vec, i, free_vec);
					}

					set_word_matrix(result, num_feat, num_vec);
					delete[] result;

					break;
				}
				
				case F_SHORT:
				{
					CShortFeatures* shortfeat=(CShortFeatures*) feat;
					INT num_feat=shortfeat->get_num_features();
					INT num_vec=shortfeat->get_num_vectors();
					SHORT* result=new SHORT[num_feat*num_vec];
					ASSERT(result);

					for (INT i=0; i<num_vec; i++)
					{
						INT num_vfeat=0;
						bool free_vec=true;
						SHORT* vec=shortfeat->get_feature_vector(i, num_vfeat, free_vec);
						ASSERT(num_vfeat==num_feat);

						for (INT j=0; j<num_vfeat; j++)
							result[num_feat*i+j]=vec[j];
						shortfeat->free_feature_vector(vec, i, free_vec);
					}

					set_short_matrix(result, num_feat, num_vec);
					delete[] result;

					break;
				}
				
				case F_CHAR:
				{
					CCharFeatures* charfeat=(CCharFeatures*) feat;
					INT num_feat=charfeat->get_num_features();
					INT num_vec=charfeat->get_num_vectors();
					CHAR* result=new CHAR[num_feat*num_vec];
					ASSERT(result);

					for (INT i=0; i<num_vec; i++)
					{
						INT num_vfeat=0;
						bool free_vec=true;
						CHAR* vec=charfeat->get_feature_vector(i, num_vfeat, free_vec);
						ASSERT(num_vfeat==num_feat);

						for (INT j=0; j<num_vfeat; j++)
							result[num_feat*i+j]=vec[j];
						charfeat->free_feature_vector(vec, i, free_vec);
					}

					set_char_matrix(result, num_feat, num_vec);
					delete[] result;

					break;
				}
				
				case F_BYTE:
				{
					CByteFeatures* bytefeat=(CByteFeatures*) feat;
					INT num_feat=bytefeat->get_num_features();
					INT num_vec=bytefeat->get_num_vectors();
					BYTE* result=new BYTE[num_feat*num_vec];
					ASSERT(result);

					for (INT i=0; i<num_vec; i++)
					{
						INT num_vfeat=0;
						bool free_vec=true;
						BYTE* vec=bytefeat->get_feature_vector(i, num_vfeat, free_vec);
						ASSERT(num_vfeat==num_feat);

						for (INT j=0; j<num_vfeat; j++)
							result[num_feat*i+j]=vec[j];
						bytefeat->free_feature_vector(vec, i, free_vec);
					}

					set_byte_matrix(result, num_feat, num_vec);
					delete[] result;

					break;
				}
				
				default:
					SG_ERROR("%s not implemented.\n", feat->get_feature_type());
			}
		break;

		case C_SPARSE:
			switch (feat->get_feature_type())
			{
				case F_DREAL:
				{
					LONG nnz=((CSparseFeatures<DREAL>*) feat)->
						get_num_nonzero_entries();
					INT num_vec=feat->get_num_vectors();
					INT num_feat=
						((CSparseFeatures<DREAL>*) feat)->get_num_features();

					SG_DEBUG("sparse matrix has %d rows, %d cols and %d nnz elemements\n", num_feat, num_vec, nnz);

					TSparse<DREAL>* result=new TSparse<DREAL>[num_vec];
					ASSERT(result);

					for (INT i=0; i<num_vec; i++)
					{
						INT len=0;
						bool dofree=false;
						TSparseEntry<DREAL>* vec=
							((CSparseFeatures<DREAL>*) feat)->
								get_sparse_feature_vector(i, len, dofree);
						result[i].features=new TSparseEntry<DREAL>[len];

						for (INT j=0; j<len; j++)
						{
							result[i].features[j].entry=vec[j].entry;
							result[i].features[j].feat_index=vec[j].feat_index;
						}
						((CSparseFeatures<DREAL>*) feat)->
							free_feature_vector(vec, len, dofree);
					}

					set_real_sparsematrix(result, num_feat, num_vec);
					delete[] result;
				}
				break;

				default:
					SG_ERROR("not implemented\n");
			}
		break;

		case C_STRING:
			switch (feat->get_feature_type())
			{
				case F_CHAR:
				{
					INT num_vec=feat->get_num_vectors();
					T_STRING<CHAR>* list=new T_STRING<CHAR>[num_vec];
					for (INT i=0; i<num_vec; i++)
					{
						INT len=0;
						CHAR* vec=((CStringFeatures<CHAR>*) feat)->
							get_feature_vector(i, len);

						if (len>0)
							list[i].string=vec;
						else
							list[i].string=NULL;
					}

					set_string_list(list, num_vec);
					delete[] list;
				}
				break;

				case F_WORD:
				{
					INT num_vec=feat->get_num_vectors();
					T_STRING<WORD>* list=new T_STRING<WORD>[num_vec];
					for (INT i=0; i<num_vec; i++)
					{
						INT len=0;
						WORD* vec=((CStringFeatures<WORD>*) feat)->
							get_feature_vector(i, len);

						if (len>0)
							list[i].string=vec;
						else
							list[i].string=NULL;
					}

					set_string_list(list, num_vec);
					delete[] list;
				}
				break;

				default:
					SG_ERROR("not implemented\n");
			}
		break;

		default:
			SG_ERROR( "not implemented\n");
	}

	return true;
}

bool CSGInterface::a_get_distance_matrix()
{
	if (m_nlhs!=1 || m_nrhs!=1)
		return false;

	CDistance* distance=gui->guidistance.get_distance();
	if (!distance || !distance->get_rhs() || !distance->get_lhs())
		SG_ERROR("No distance defined.\n");

	INT num_vec1=distance->get_lhs()->get_num_vectors();
	INT num_vec2=distance->get_rhs()->get_num_vectors();
	DREAL* dmatrix=NULL;
	distance->get_distance_matrix_real(num_vec1, num_vec2, dmatrix);

	set_real_matrix(dmatrix, num_vec1, num_vec2);
	delete[] dmatrix;

	return true;
}

bool CSGInterface::a_get_kernel_matrix()
{
	if (m_nlhs!=1 || m_nrhs!=1)
		return false;

	CKernel* kernel=gui->guikernel.get_kernel();
	if (!kernel || !kernel->get_rhs() || !kernel->get_lhs())
		SG_ERROR("No kernel defined.\n");

	INT num_vec1=kernel->get_lhs()->get_num_vectors();
	INT num_vec2=kernel->get_rhs()->get_num_vectors();
	DREAL* kmatrix=NULL;
	kernel->get_kernel_matrix_real(num_vec1, num_vec2, kmatrix);

	set_real_matrix(kmatrix, num_vec1, num_vec2);
	delete[] kmatrix;

	return true;
}

bool CSGInterface::a_set_WD_position_weights()
{
	if (m_nlhs!=0 || m_nrhs<2 || m_nrhs>3)
		return false;

	CKernel* kernel=gui->guikernel.get_kernel();
	if (!kernel)
		SG_ERROR("No kernel.\n");
	if (kernel->get_kernel_type()!=K_COMBINED)
		SG_ERROR("Only works for combined kernels.\n");

	kernel=((CCombinedKernel*) kernel)->get_last_kernel();
	if (!kernel)
		SG_ERROR("No last kernel.\n");

	EKernelType ktype=kernel->get_kernel_type();
	if (ktype!=K_WEIGHTEDDEGREE && ktype!=K_WEIGHTEDDEGREEPOS)
		SG_ERROR("Unsupported kernel.\n");

	bool success=false;
	DREAL* weights=NULL;
	INT dim=0;
	INT len=0;
	get_real_matrix(weights, dim, len);

	if (ktype==K_WEIGHTEDDEGREE)
	{
		CWeightedDegreeStringKernel* k=
			(CWeightedDegreeStringKernel*) kernel;

		if (dim!=1 & len>0)
		{
			delete[] weights;
			SG_ERROR("Dimension mismatch (should be 1 x seq_length or 0x0\n");
		}

		success=k->set_position_weights(weights, len);
	}
	else
	{
		CWeightedDegreePositionStringKernel* k=
			(CWeightedDegreePositionStringKernel*) kernel;
		CHAR* target=NULL;
		bool is_train=true;

		if (m_nrhs==3)
		{
			INT tlen=0;
			target=get_string(tlen);
			if (!target)
			{
				delete[] weights;
				SG_ERROR("Couldn't find second argument to method.\n");
			}

			if (!strmatch(target, tlen, "TRAIN") && !strmatch(target, tlen, "TEST"))
			{
				delete[] weights;
				delete[] target;
				SG_ERROR("Second argument none of TRAIN or TEST.\n");
			}

			if (strmatch(target, tlen, "TEST"))
				is_train=false;
		}

		if (dim!=1 & len>0)
		{
			delete[] weights;
			delete[] target;
			SG_ERROR("Dimension mismatch (should be 1 x seq_length or 0x0\n");
		}

		if (dim==0 & len==0)
		{
			if (m_nlhs==3)
			{
				if (is_train)
					success=k->delete_position_weights_lhs();
				else
					success=k->delete_position_weights_rhs();
			}
			else
				success=k->delete_position_weights();
		}
		else
		{
			if (m_nlhs==3)
			{
				if (is_train)
					success=k->set_position_weights_lhs(weights, dim, len);
				else
					success=k->set_position_weights_rhs(weights, dim, len);
			}
			else
				success=k->set_position_weights(weights, len);
		}

		delete[] target;
	}

	delete[] weights;
	return success;
}

bool CSGInterface::a_set_subkernel_weights()
{
	if (m_nlhs!=0 || m_nrhs!=2)
		return false;

	CKernel* kernel=gui->guikernel.get_kernel();
	if (!kernel)
		SG_ERROR("No kernel.\n");

	bool success=false;
	DREAL* weights=NULL;
	INT dim=0;
	INT len=0;
	get_real_matrix(weights, dim, len);

	EKernelType ktype=kernel->get_kernel_type();
	if (ktype==K_WEIGHTEDDEGREE)
	{
		CWeightedDegreeStringKernel* k=
			(CWeightedDegreeStringKernel*) kernel;
		INT degree=k->get_degree();
		if (dim!=degree || len<1)
		{
			delete[] weights;
			SG_ERROR("Dimension mismatch (should be de(seq_length | 1) x degree)\n");
		}

		if (len==1)
			len=0;

		success=k->set_weights(weights, dim, len);
	}
	else if (ktype==K_WEIGHTEDDEGREEPOS)
	{
		CWeightedDegreePositionStringKernel* k=
			(CWeightedDegreePositionStringKernel*) kernel;
		INT degree=k->get_degree();
		if (dim!=degree || len<1)
		{
			delete[] weights;
			SG_ERROR("Dimension mismatch (should be de(seq_length | 1) x degree)\n");
		}

		if (len==1)
			len=0;

		success=k->set_weights(weights, dim, len);
	}
	else // all other kernels
	{
		INT num_subkernels=kernel->get_num_subkernels();
		if (dim!=1 || len!=num_subkernels)
		{
			delete[] weights;
			SG_ERROR("Dimension mismatch (should be 1 x num_subkernels)\n");
		}

		kernel->set_subkernel_weights(weights, len);
		success=true;
	}

	delete[] weights;
	return success;
}

bool CSGInterface::a_set_subkernel_weights_combined()
{
	if (m_nlhs!=0 || m_nrhs!=3)
		return false;

	CKernel* kernel=gui->guikernel.get_kernel();
	if (!kernel)
		SG_ERROR("No kernel.\n");
	if (kernel->get_kernel_type()!=K_COMBINED)
		SG_ERROR("Only works for combined kernels.\n");

	bool success=false;
	DREAL* weights=NULL;
	INT dim=0;
	INT len=0;
	get_real_matrix(weights, dim, len);

	INT idx=(INT) get_real();
	SG_DEBUG("using kernel_idx=%i\n", idx);

	kernel=((CCombinedKernel*) kernel)->get_kernel(idx);
	if (!kernel)
	{
		delete[] weights;
		SG_ERROR("No subkernel at idx %d.\n", idx);
	}

	EKernelType ktype=kernel->get_kernel_type();
	if (ktype==K_WEIGHTEDDEGREE)
	{
		CWeightedDegreeStringKernel* k=
			(CWeightedDegreeStringKernel*) kernel;
		INT degree=k->get_degree();
		if (dim!=degree || len<1)
		{
			delete[] weights;
			SG_ERROR("Dimension mismatch (should be de(seq_length | 1) x degree)\n");
		}

		if (len==1)
			len=0;

		success=k->set_weights(weights, dim, len);
	}
	else if (ktype==K_WEIGHTEDDEGREEPOS)
	{
		CWeightedDegreePositionStringKernel* k=
			(CWeightedDegreePositionStringKernel*) kernel;
		INT degree=k->get_degree();
		if (dim!=degree || len<1)
		{
			delete[] weights;
			SG_ERROR("Dimension mismatch (should be de(seq_length | 1) x degree)\n");
		}

		if (len==1)
			len=0;

		success=k->set_weights(weights, dim, len);
	}
	else // all other kernels
	{
		INT num_subkernels=kernel->get_num_subkernels();
		if (dim!=1 || len!=num_subkernels)
		{
			delete[] weights;
			SG_ERROR("Dimension mismatch (should be 1 x num_subkernels)\n");
		}

		kernel->set_subkernel_weights(weights, len);
		success=true;
	}

	delete[] weights;
	return success;
}

bool CSGInterface::a_set_last_subkernel_weights()
{
	if (m_nlhs!=0 || m_nrhs!=2)
		return false;

	CKernel* kernel=gui->guikernel.get_kernel();
	if (!kernel)
		SG_ERROR("No kernel.\n");
	if (kernel->get_kernel_type()!=K_COMBINED)
		SG_ERROR("Only works for Combined kernels.\n");

	kernel=((CCombinedKernel*) kernel)->get_last_kernel();
	if (!kernel)
		SG_ERROR("No last kernel.\n");

	bool success=false;
	DREAL* weights=NULL;
	INT dim=0;
	INT len=0;
	get_real_matrix(weights, dim, len);

	EKernelType ktype=kernel->get_kernel_type();
	if (ktype==K_WEIGHTEDDEGREE)
	{
		CWeightedDegreeStringKernel* k=(CWeightedDegreeStringKernel*) kernel;
		if (dim!=k->get_degree() || len<1)
		{
			delete[] weights;
			SG_ERROR("Dimension mismatch (should be de(seq_length | 1) x degree)\n");
		}

		if (len==1)
			len=0;

		success=k->set_weights(weights, dim, len);
	}
	else if (ktype==K_WEIGHTEDDEGREEPOS)
	{
		CWeightedDegreePositionStringKernel* k=
			(CWeightedDegreePositionStringKernel*) kernel;
		if (dim!=k->get_degree() || len<1)
		{
			delete[] weights;
			SG_ERROR("Dimension mismatch (should be de(seq_length | 1) x degree)\n");
		}

		if (len==1)
			len=0;

		success=k->set_weights(weights, dim, len);
	}
	else // all other kernels
	{
		INT num_subkernels=kernel->get_num_subkernels();
		if (dim!=1 || len!=num_subkernels)
		{
			delete[] weights;
			SG_ERROR("Dimension mismatch (should be 1 x num_subkernels)\n");
		}

		kernel->set_subkernel_weights(weights, len);
		success=true;
	}

	delete[] weights;
	return success;
}

bool CSGInterface::a_get_SPEC_consensus()
{
	if (m_nlhs!=1 || m_nrhs!=1)
		return false;

	CKernel* kernel=gui->guikernel.get_kernel();
	if (!kernel)
		SG_ERROR("No kernel.\n");
	if (kernel->get_kernel_type()!=K_COMMWORDSTRING)
		SG_ERROR("Only works for CommWordString kernels.\n");

	CSVM* svm=(CSVM*) gui->guiclassifier.get_classifier();
	ASSERT(svm);
	INT num_suppvec=svm->get_num_support_vectors();
	INT* sv_idx=new INT[num_suppvec];
	DREAL* sv_weight=new DREAL[num_suppvec];
	INT num_feat=0;

	for (INT i=0; i<num_suppvec; i++)
	{
		sv_idx[i]=svm->get_support_vector(i);
		sv_weight[i]=svm->get_alpha(i);
	}

	CHAR* consensus=((CCommWordStringKernel*) kernel)->compute_consensus(
		num_feat, num_suppvec, sv_idx, sv_weight);
	delete[] sv_idx;
	delete[] sv_weight;

	set_char_matrix(consensus, 1, num_feat);
	delete[] consensus;

	return true;
}

bool CSGInterface::a_get_SPEC_scoring()
{
	if (m_nlhs!=1 || m_nrhs!=2)
		return false;

	INT max_order=(INT) get_real();
	CKernel* kernel=gui->guikernel.get_kernel();
	if (!kernel)
		SG_ERROR("No kernel.\n");

	EKernelType ktype=kernel->get_kernel_type();
	if (ktype!=K_COMMWORDSTRING && ktype!=K_WEIGHTEDCOMMWORDSTRING)
		SG_ERROR("Only works for (Weighted) CommWordString kernels.\n");

	CSVM* svm=(CSVM*) gui->guiclassifier.get_classifier();
	ASSERT(svm);
	INT num_suppvec=svm->get_num_support_vectors();
	INT* sv_idx=new INT[num_suppvec];
	DREAL* sv_weight=new DREAL[num_suppvec];
	INT num_feat=0;
	INT num_sym=0;

	for (INT i=0; i<num_suppvec; i++)
	{
		sv_idx[i]=svm->get_support_vector(i);
		sv_weight[i]=svm->get_alpha(i);
	}

	if ((max_order<1) || (max_order>8))
	{
		SG_WARNING( "max_order out of range 1..8 (%d). setting to 1\n", max_order);
		max_order=1;
	}

	DREAL* position_weights=NULL;
	if (ktype==K_COMMWORDSTRING)
		position_weights=((CCommWordStringKernel*) kernel)->compute_scoring(
			max_order, num_feat, num_sym, NULL,
			num_suppvec, sv_idx, sv_weight);
	else
		position_weights=((CWeightedCommWordStringKernel*) kernel)->compute_scoring(
			max_order, num_feat, num_sym, NULL,
			num_suppvec, sv_idx, sv_weight);
	delete[] sv_idx;
	delete[] sv_weight;

	set_real_matrix(position_weights, num_sym, num_feat);
	delete[] position_weights;

	return true;
}

bool CSGInterface::a_get_WD_consensus()
{
	if (m_nlhs!=1 || m_nrhs!=1)
		return false;

	CKernel* kernel=gui->guikernel.get_kernel();
	if (!kernel)
		SG_ERROR("No kernel.\n");
	if (kernel->get_kernel_type()!=K_WEIGHTEDDEGREEPOS)
		SG_ERROR("Only works for Weighted Degree Position kernels.\n");

	CSVM* svm=(CSVM*) gui->guiclassifier.get_classifier();
	ASSERT(svm);
	INT num_suppvec=svm->get_num_support_vectors();
	INT* sv_idx=new INT[num_suppvec];
	DREAL* sv_weight=new DREAL[num_suppvec];
	INT num_feat=0;

	for (INT i=0; i<num_suppvec; i++)
	{
		sv_idx[i]=svm->get_support_vector(i);
		sv_weight[i]=svm->get_alpha(i);
	}

	CHAR* consensus=((CWeightedDegreePositionStringKernel*) kernel)->compute_consensus(
			num_feat, num_suppvec, sv_idx, sv_weight);
	delete[] sv_idx;
	delete[] sv_weight;

	set_char_matrix(consensus, 1, num_feat);
	delete[] consensus;

	return true;
}

bool CSGInterface::a_compute_POIM_WD()
{
	if (m_nlhs!=1 || m_nrhs!=3)
		return false;

	INT max_order=(INT) get_real();
	DREAL* distribution=NULL;
	INT num_dfeat=0;
	INT num_dvec=0;

	get_real_matrix(distribution, num_dfeat, num_dvec);
	if (!distribution)
		SG_ERROR("Wrong distribution.\n");

	CKernel* kernel=gui->guikernel.get_kernel();
	if (!kernel)
	{
		delete[] distribution;
		SG_ERROR("No Kernel.\n");
	}
	if (kernel->get_kernel_type()!=K_WEIGHTEDDEGREEPOS)
	{
		delete[] distribution;
		SG_ERROR("Only works for Weighted Degree Position kernels.\n");
	}

	INT seqlen=0;
	INT num_sym=0;
	CStringFeatures<CHAR>* sfeat=(CStringFeatures<CHAR>*)
		(((CWeightedDegreePositionStringKernel*) kernel)->get_lhs());
	ASSERT(sfeat);
	seqlen=sfeat->get_max_vector_length();
	num_sym=(INT) sfeat->get_num_symbols();

	if (num_dvec!=seqlen || num_dfeat!=num_sym)
	{
		delete[] distribution;
		SG_ERROR("distribution should have (seqlen x num_sym) elements"
				"(seqlen: %d vs. %d symbols: %d vs. %d)\n", seqlen,
				num_dvec, num_sym, num_dfeat);
	}

	CSVM* svm=(CSVM*) gui->guiclassifier.get_classifier();
	ASSERT(svm);
	INT num_suppvec=svm->get_num_support_vectors();
	INT* sv_idx=new INT[num_suppvec];
	ASSERT(sv_idx);
	DREAL* sv_weight=new DREAL[num_suppvec];
	ASSERT(sv_weight);

	for (INT i=0; i<num_suppvec; i++)
	{
		sv_idx[i]=svm->get_support_vector(i);
		sv_weight[i]=svm->get_alpha(i);
	}

	/*
	if ((max_order < 1) || (max_order > 12))
	{
		SG_WARNING( "max_order out of range 1..12 (%d). setting to 1.\n", max_order);
		max_order=1;
	}
	*/

	DREAL* position_weights;
	position_weights=((CWeightedDegreePositionStringKernel*) kernel)->compute_POIM(
			max_order, seqlen, num_sym, NULL,
			num_suppvec, sv_idx, sv_weight, distribution);
	delete[] sv_idx;
	delete[] sv_weight;
	delete[] distribution;

	set_real_matrix(position_weights, num_sym, seqlen);
	delete[] position_weights;

	return true;
}

bool CSGInterface::a_get_WD_scoring()
{
	if (m_nlhs!=1 || m_nrhs!=2)
		return false;

	INT max_order=(INT) get_real();

	CKernel* kernel=gui->guikernel.get_kernel();
	if (!kernel)
		SG_ERROR("No kernel.\n");
	if (kernel->get_kernel_type()!=K_WEIGHTEDDEGREEPOS)
		SG_ERROR("Only works for Weighted Degree Position kernels.\n");

	CSVM* svm=(CSVM*) gui->guiclassifier.get_classifier();
	ASSERT(svm);
	INT num_suppvec=svm->get_num_support_vectors();
	INT* sv_idx=new INT[num_suppvec];
	DREAL* sv_weight=new DREAL[num_suppvec];
	INT num_feat=0;
	INT num_sym=0;

	for (INT i=0; i<num_suppvec; i++)
	{
		sv_idx[i]=svm->get_support_vector(i);
		sv_weight[i]=svm->get_alpha(i);
	}

	if ((max_order<1) || (max_order>12))
	{
		SG_WARNING("max_order out of range 1..12 (%d). setting to 1\n", max_order);
		max_order=1;
	}

	DREAL* position_weights=
		((CWeightedDegreePositionStringKernel*) kernel)->compute_scoring(
			max_order, num_feat, num_sym, NULL, num_suppvec, sv_idx, sv_weight);
	delete[] sv_idx;
	delete[] sv_weight;

	set_real_matrix(position_weights, num_sym, num_feat);
	delete[] position_weights;

	return true;
}

bool CSGInterface::a_get_WD_position_weights()
{
	if (m_nlhs!=1 || m_nrhs!=1)
		return false;

	CKernel* kernel=gui->guikernel.get_kernel();
	if (!kernel)
		SG_ERROR("No kernel.\n");
	if (kernel->get_kernel_type()!=K_COMBINED)
		SG_ERROR("Only works for Combined kernels.\n");

	kernel=((CCombinedKernel*) kernel)->get_last_kernel();
	if (!kernel)
		SG_ERROR("Couldn't find last kernel.\n");

	if (kernel->get_kernel_type()!=K_WEIGHTEDDEGREE &&
		kernel->get_kernel_type()!=K_WEIGHTEDDEGREEPOS)
		SG_ERROR("Wrong subkernel type.\n");

	INT len=0;
	const DREAL* position_weights;

	if (kernel->get_kernel_type()==K_WEIGHTEDDEGREE)
		position_weights=((CWeightedDegreeStringKernel*) kernel)->get_position_weights(len);
	else
		position_weights=((CWeightedDegreePositionStringKernel*) kernel)->get_position_weights(len);

	if (position_weights==NULL)
		set_real_matrix(position_weights, 1, 0);
	else
		set_real_matrix(position_weights, 1, len);

	return true;
}

bool CSGInterface::a_get_last_subkernel_weights()
{
	if (m_nlhs!=1 || m_nrhs!=1)
		return false;

	CKernel* kernel=gui->guikernel.get_kernel();
	EKernelType ktype=kernel->get_kernel_type();
	if (!kernel)
		SG_ERROR("No kernel.\n");
	if (ktype!=K_COMBINED)
		SG_ERROR("Only works for Combined kernels.\n");

	kernel=((CCombinedKernel*) kernel)->get_last_kernel();
	if (!kernel)
		SG_ERROR("Couldn't find last kernel.\n");

	INT degree=0;
	INT len=0;

	if (ktype==K_COMBINED)
	{
		INT num_weights=0;
		const DREAL* weights=
			((CCombinedKernel*) kernel)->get_subkernel_weights(num_weights);
		set_real_matrix(weights, 1, num_weights);
		return true;
	}

	const DREAL* weights=NULL;
	if (ktype==K_WEIGHTEDDEGREE)
		weights=((CWeightedDegreeStringKernel*) kernel)->
			get_degree_weights(degree, len);
	else if (ktype==K_WEIGHTEDDEGREEPOS)
		weights=((CWeightedDegreePositionStringKernel*) kernel)->
			get_degree_weights(degree, len);
	else
		SG_ERROR("Only works for Weighted Degree (Position) kernels.\n");

	if (len==0)
		len=1;

	set_real_matrix(weights, degree, len);

	return true;
}

bool CSGInterface::a_compute_by_subkernels()
{
	if (m_nlhs!=1 || m_nrhs!=1)
		return false;

	CKernel* kernel=gui->guikernel.get_kernel();
	if (!kernel)
		SG_ERROR("No kernel.\n");
	if (!kernel->get_rhs())
		SG_ERROR("No rhs.\n");

	INT num_vec=kernel->get_rhs()->get_num_vectors();
	INT degree=0;
	INT len=0;
	EKernelType ktype=kernel->get_kernel_type();

	// it would be nice to have a common base class for the WD kernels
	if (ktype==K_WEIGHTEDDEGREE)
	{
		CWeightedDegreeStringKernel* k=(CWeightedDegreeStringKernel*) kernel;
		k->get_degree_weights(degree, len);
		if (!k->is_tree_initialized())
			SG_ERROR("Kernel optimization not initialized.\n");
	}
	else if (ktype==K_WEIGHTEDDEGREEPOS)
	{
		CWeightedDegreePositionStringKernel* k=
			(CWeightedDegreePositionStringKernel*) kernel;
		k->get_degree_weights(degree, len);
		if (!k->is_tree_initialized())
			SG_ERROR("Kernel optimization not initialized.\n");
	}
	else
		SG_ERROR("Only works for Weighted Degree (Position) kernels.\n");

	if (len==0)
		len=1;

	INT num_feat=degree*len;
	INT num=num_feat*num_vec;
	DREAL* result=new DREAL[num];
	ASSERT(result);

	for (INT i=0; i<num; i++)
		result[i]=0;

	if (ktype==K_WEIGHTEDDEGREE)
	{
		CWeightedDegreeStringKernel* k=(CWeightedDegreeStringKernel*) kernel;
		for (INT i=0; i<num_vec; i++)
			k->compute_by_tree(i, &result[i*num_feat]);
	}
	else
	{
		CWeightedDegreePositionStringKernel* k=
			(CWeightedDegreePositionStringKernel*) kernel;
		for (INT i=0; i<num_vec; i++)
			k->compute_by_tree(i, &result[i*num_feat]);
	}

	set_real_matrix(result, num_feat, num_vec);
	delete[] result;

	return true;
}

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

			CWeightedDegreePositionStringKernel* k=(CWeightedDegreePositionStringKernel*) kernel;
			CSVM* svm=(CSVM*) gui->guiclassifier.get_classifier();
			if (!svm)
				SG_ERROR("No SVM defined.\n");

			INT num_suppvec=svm->get_num_support_vectors();
			INT* sv_idx=new INT[num_suppvec];
			DREAL* sv_weight=new DREAL[num_suppvec];
			INT num_feat=0;
			INT num_sym=0;

			for (INT i=0; i<num_suppvec; i++)
			{
				sv_idx[i]=svm->get_support_vector(i);
				sv_weight[i]=svm->get_alpha(i);
			}

			DREAL* position_weights=k->extract_w(max_order, num_feat,
				num_sym, NULL, num_suppvec, sv_idx, sv_weight);
			delete[] sv_idx;
			delete[] sv_weight;

			set_real_matrix(position_weights, num_sym, num_feat);
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
			SG_ERROR("Unsupported kernel %s.\n", kernel->get_name());
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
	delete labels;

	set_real_matrix(result, 1, num_vec);
	delete[] result;

	return true;
}

bool CSGInterface::a_set_plugin_estimate()
{
	if (m_nlhs!=0 || m_nrhs!=3)
		return false;

	DREAL* emission_probs=NULL;
	INT num_probs=0;
	INT num_vec=0;
	get_real_matrix(emission_probs, num_probs, num_vec);

	if (num_vec!=2)
	{
		delete[] emission_probs;
		SG_ERROR("Need at least 1 set of positive and 1 set of negative params.\n");
	}

	DREAL* pos_params=emission_probs;
	DREAL* neg_params=&(emission_probs[num_probs]);

	DREAL* model_sizes=NULL;
	INT len=0;
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

	DREAL* pos_params=NULL;
	DREAL* neg_params=NULL;
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
	delete labels;

	set_real_matrix(result, 1, num_vec);
	delete[] result;

	return true;
}

bool CSGInterface::a_classify_example()
{
	if (m_nlhs!=1 || m_nrhs!=2)
		return false;

	INT idx=(INT) get_real();
	DREAL result=0;

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
	DREAL* alphas=NULL;
	INT num_feat_alphas=0;
	INT num_vec_alphas=0;
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
	delete labels;

	set_real_matrix(result, 1, num_vec);
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
	DREAL result=0;

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

	INT num_feat=0;
	WORD* vec=((CStringFeatures<WORD>*) feat)->get_feature_vector(dim, num_feat);
	if (!vec || num_feat<=0)
		return false;

	SG_DEBUG( "computing viterbi path for vector %d (length %d)\n", dim, num_feat);
	DREAL likelihood=0;
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
	INT i=0;
	INT j=0;
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

	T_STRING<CHAR>* list=NULL;
	INT num_str=0;
	get_string_list(list, num_str);
	set_string_list(list, num_str);
	delete[] list;

	return true;
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

#endif // !HAVE_SWIG
