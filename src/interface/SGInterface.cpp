#include "lib/config.h"

#if !defined(HAVE_SWIG)

#include <string.h>
#include <stdlib.h>

#include "interface/SGInterface.h"
#include "lib/ShogunException.h"
#include "lib/Mathematics.h"
#include "structure/DynProg.h"
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
#include "kernel/CustomKernel.h"
#include "features/ByteFeatures.h"
#include "features/CharFeatures.h"
#include "features/IntFeatures.h"
#include "features/RealFeatures.h"
#include "features/ShortFeatures.h"
#include "features/ShortRealFeatures.h"
#include "features/WordFeatures.h"

CSGInterface* interface=NULL;
extern CTextGUI* gui;

#define USAGE(method) "sg('" method "')"
#define USAGE_I(method, in) "sg('" method "', " in ")"
#define USAGE_O(method, out) "[" out "]=sg('" method "')"
#define USAGE_IO(method, in, out) "[" out "]=sg('" method "', " in ")"

static CSGInterfaceMethod sg_methods[]=
{
	{ (CHAR*) "Features", NULL, NULL },
	{
		(CHAR*) N_LOAD_FEATURES,
		(&CSGInterface::cmd_load_features),
		(CHAR*) USAGE_I(N_LOAD_FEATURES, "'filename, fclass, type, target[, size[, comp_features]]'")
	},
	{
		(CHAR*) N_SAVE_FEATURES,
		(&CSGInterface::cmd_save_features),
		(CHAR*) USAGE_I(N_SAVE_FEATURES, "'filename, type, target'")
	},
	{
		(CHAR*) N_CLEAN_FEATURES,
		(&CSGInterface::cmd_clean_features),
		(CHAR*) USAGE_I(N_CLEAN_FEATURES, "'TRAIN|TEST'")
	},
	{
		(CHAR*) N_GET_FEATURES,
		(&CSGInterface::cmd_get_features),
		(CHAR*) USAGE_IO(N_GET_FEATURES, "'TRAIN|TEST'", "features")
	},
	{
		(CHAR*) N_ADD_FEATURES,
		(&CSGInterface::cmd_add_features),
		(CHAR*) USAGE_I(N_ADD_FEATURES,
			"'TRAIN|TEST', features[, DNABINFILE|<ALPHABET>]")
	},
	{
		(CHAR*) N_SET_FEATURES,
		(&CSGInterface::cmd_set_features),
		(CHAR*) USAGE_I(N_SET_FEATURES,
			"'TRAIN|TEST', features[, DNABINFILE|<ALPHABET>]")
	},
	{
		(CHAR*) N_SET_REF_FEAT,
		(&CSGInterface::cmd_set_reference_features),
		(CHAR*) USAGE_I(N_SET_REF_FEAT, "'TRAIN|TEST'")
	},
	{
		(CHAR*) N_CONVERT,
		(&CSGInterface::cmd_convert),
		(CHAR*) USAGE_I(N_CONVERT, "'TRAIN|TEST', from_class, from_type, to_class, to_type'")
	},
	{
		(CHAR*) N_FROM_POSITION_LIST,
		(&CSGInterface::cmd_obtain_from_position_list),
		(CHAR*) USAGE_I(N_FROM_POSITION_LIST, "'TRAIN|TEST', winsize, shift[, skip]")
	},
	{
		(CHAR*) N_SLIDE_WINDOW,
		(&CSGInterface::cmd_obtain_by_sliding_window),
		(CHAR*) USAGE_I(N_SLIDE_WINDOW, "'TRAIN|TEST, winsize, shift[, skip]'")
	},
	{
		(CHAR*) N_RESHAPE,
		(&CSGInterface::cmd_reshape),
		(CHAR*) USAGE_I(N_RESHAPE, "'TRAIN|TEST, num_feat, num_vec'")
	},
	{
		(CHAR*) N_LOAD_LABELS,
		(&CSGInterface::cmd_load_labels),
		(CHAR*) USAGE_I(N_LOAD_LABELS, "'filename, TRAIN|TARGET'")
	},
	{
		(CHAR*) N_SET_LABELS,
		(&CSGInterface::cmd_set_labels),
		(CHAR*) USAGE_I(N_SET_LABELS, "'TRAIN|TEST', labels")
	},
	{
		(CHAR*) N_GET_LABELS,
		(&CSGInterface::cmd_get_labels),
		(CHAR*) USAGE_IO(N_GET_LABELS, "'TRAIN|TEST'", "labels")
	},


	{ (CHAR*) "Kernel", NULL, NULL },
	{
		(CHAR*) N_SET_KERNEL,
		(&CSGInterface::cmd_set_kernel),
		(CHAR*) USAGE_I(N_SET_KERNEL, "'type, size[, kernel-specific parameters]'")
	},
	{
		(CHAR*) N_ADD_KERNEL,
		(&CSGInterface::cmd_add_kernel),
		(CHAR*) USAGE_I(N_ADD_KERNEL, "'weight, kernel-specific parameters'")
	},
	{
		(CHAR*) N_INIT_KERNEL,
		(&CSGInterface::cmd_init_kernel),
		(CHAR*) USAGE_I(N_INIT_KERNEL, "'TRAIN|TEST'")
	},
	{
		(CHAR*) N_CLEAN_KERNEL,
		(&CSGInterface::cmd_clean_kernel),
		(CHAR*) USAGE(N_CLEAN_KERNEL)
	},
	{
		(CHAR*) N_SAVE_KERNEL,
		(&CSGInterface::cmd_save_kernel),
		(CHAR*) USAGE_I(N_SAVE_KERNEL, "'filename'")
	},
	{
		(CHAR*) N_LOAD_KERNEL_INIT,
		(&CSGInterface::cmd_load_kernel_init),
		(CHAR*) USAGE_I(N_LOAD_KERNEL_INIT, "'filename'")
	},
	{
		(CHAR*) N_SAVE_KERNEL_INIT,
		(&CSGInterface::cmd_save_kernel_init),
		(CHAR*) USAGE_I(N_SAVE_KERNEL_INIT, "'filename'")
	},
	{
		(CHAR*) N_GET_KERNEL_MATRIX,
		(&CSGInterface::cmd_get_kernel_matrix),
		(CHAR*) USAGE_O(N_GET_KERNEL_MATRIX, "K")
	},
	{
		(CHAR*) N_SET_CUSTOM_KERNEL,
		(&CSGInterface::cmd_set_custom_kernel),
		(CHAR*) USAGE_I(N_SET_CUSTOM_KERNEL, "kernelmatrix, 'DIAG|FULL|FULL2DIAG'")
	},
	{
		(CHAR*) N_SET_WD_POS_WEIGHTS,
		(&CSGInterface::cmd_set_WD_position_weights),
		(CHAR*) USAGE_I(N_SET_WD_POS_WEIGHTS, "W[, 'TRAIN|TEST']")
	},
	{
		(CHAR*) N_GET_SUBKERNEL_WEIGHTS,
		(&CSGInterface::cmd_get_subkernel_weights),
		(CHAR*) USAGE_O(N_GET_SUBKERNEL_WEIGHTS, "W")
	},
	{
		(CHAR*) N_SET_SUBKERNEL_WEIGHTS,
		(&CSGInterface::cmd_set_subkernel_weights),
		(CHAR*) USAGE_I(N_SET_SUBKERNEL_WEIGHTS, "W")
	},
	{
		(CHAR*) N_SET_SUBKERNEL_WEIGHTS_COMBINED,
		(&CSGInterface::cmd_set_subkernel_weights_combined),
		(CHAR*) USAGE_I(N_SET_SUBKERNEL_WEIGHTS_COMBINED, "W, idx")
	},
	{
		(CHAR*) N_SET_LAST_SUBKERNEL_WEIGHTS,
		(&CSGInterface::cmd_set_last_subkernel_weights),
		(CHAR*) USAGE_I(N_SET_LAST_SUBKERNEL_WEIGHTS, "W")
	},
	{
		(CHAR*) N_GET_WD_POS_WEIGHTS,
		(&CSGInterface::cmd_get_WD_position_weights),
		(CHAR*) USAGE_O(N_GET_WD_POS_WEIGHTS, "W")
	},
	{
		(CHAR*) N_GET_LAST_SUBKERNEL_WEIGHTS,
		(&CSGInterface::cmd_get_last_subkernel_weights),
		(CHAR*) USAGE_O(N_GET_LAST_SUBKERNEL_WEIGHTS, "W")
	},
	{
		(CHAR*) N_COMPUTE_BY_SUBKERNELS,
		(&CSGInterface::cmd_compute_by_subkernels),
		(CHAR*) USAGE_O(N_COMPUTE_BY_SUBKERNELS, "W")
	},
	{
		(CHAR*) N_INIT_KERNEL_OPTIMIZATION,
		(&CSGInterface::cmd_init_kernel_optimization),
		(CHAR*) USAGE(N_INIT_KERNEL_OPTIMIZATION)
	},
	{
		(CHAR*) N_GET_KERNEL_OPTIMIZATION,
		(&CSGInterface::cmd_get_kernel_optimization),
		(CHAR*) USAGE_O(N_GET_KERNEL_OPTIMIZATION, "W")
	},
	{
		(CHAR*) N_DELETE_KERNEL_OPTIMIZATION,
		(&CSGInterface::cmd_delete_kernel_optimization),
		(CHAR*) USAGE(N_DELETE_KERNEL_OPTIMIZATION)
	},
	{
		(CHAR*) N_SET_KERNEL_OPTIMIZATION_TYPE,
		(&CSGInterface::cmd_set_kernel_optimization_type),
		(CHAR*) USAGE_I(N_SET_KERNEL_OPTIMIZATION_TYPE, "'FASTBUTMEMHUNGRY|SLOWBUTMEMEFFICIENT'")
	},
#ifdef USE_SVMLIGHT
	{
		(CHAR*) N_RESIZE_KERNEL_CACHE,
		(&CSGInterface::cmd_resize_kernel_cache),
		(CHAR*) USAGE_I(N_RESIZE_KERNEL_CACHE, "'size'")
	},
#endif //USE_SVMLIGHT


	{ (CHAR*) "Distance", NULL, NULL },
	{
		(CHAR*) N_SET_DISTANCE,
		(&CSGInterface::cmd_set_distance),
		(CHAR*) USAGE_I(N_SET_DISTANCE, "'type, data type[, distance-specific parameters]'")
	},
	{
		(CHAR*) N_INIT_DISTANCE,
		(&CSGInterface::cmd_init_distance),
		(CHAR*) USAGE_I(N_INIT_DISTANCE, "'TRAIN|TEST'")
	},
	{
		(CHAR*) N_GET_DISTANCE_MATRIX,
		(&CSGInterface::cmd_get_distance_matrix),
		(CHAR*) USAGE_O(N_GET_DISTANCE_MATRIX, "D")
	},


	{ (CHAR*) "SVM & Other Classifier", NULL, NULL },
	{
		(CHAR*) N_CLASSIFY,
		(&CSGInterface::cmd_classify),
		(CHAR*) USAGE_O(N_CLASSIFY, "result")
	},
	{
		(CHAR*) N_SVM_CLASSIFY,
		(&CSGInterface::cmd_classify),
		(CHAR*) USAGE_O(N_SVM_CLASSIFY, "result")
	},
	{
		(CHAR*) N_CLASSIFY_EXAMPLE,
		(&CSGInterface::cmd_classify_example),
		(CHAR*) USAGE_IO(N_CLASSIFY_EXAMPLE, "feature_vector_index", "result")
	},
	{
		(CHAR*) N_SVM_CLASSIFY_EXAMPLE,
		(&CSGInterface::cmd_classify_example),
		(CHAR*) USAGE_IO(N_SVM_CLASSIFY_EXAMPLE, "feature_vector_index", "result")
	},
	{
		(CHAR*) N_GET_CLASSIFIER,
		(&CSGInterface::cmd_get_classifier),
		(CHAR*) USAGE_O(N_GET_CLASSIFIER, "bias, weights")
	},
	{
		(CHAR*) N_NEW_SVM,
		(&CSGInterface::cmd_new_classifier),
		(CHAR*) USAGE_I(N_NEW_SVM, "'LIBSVM_ONECLASS|LIBSVM_MULTICLASS|LIBSVM|SVMLIGHT|LIGHT|SVMLIN|GPBTSVM|MPDSVM|GNPPSVM|GMNPSVM|SUBGRADIENTSVM|WDSVMOCAS|SVMOCAS|SVMSGD|SVMBMRM|SVMPERF|SVRLIGHT|LIBSVR|KERNELPERCEPTRON|PERCEPTRON|LIBLINEAR_LR|LIBLINEAR_L2|LDA|LPM|LPBOOST|SUBGRADIENTLPM|KNN|KMEANS|HIERARCHICAL'")
	},
	{
		(CHAR*) N_NEW_CLASSIFIER,
		(&CSGInterface::cmd_new_classifier),
		(CHAR*) USAGE_I(N_NEW_CLASSIFIER, "'LIBSVM_ONECLASS|LIBSVM_MULTICLASS|LIBSVM|SVMLIGHT|LIGHT|SVMLIN|GPBTSVM|MPDSVM|GNPPSVM|GMNPSVM|SUBGRADIENTSVM|WDSVMOCAS|SVMOCAS|SVMSGD|SVMBMRM|SVMPERF|SVRLIGHT|LIBSVR|KERNELPERCEPTRON|PERCEPTRON|LIBLINEAR_LR|LIBLINEAR_L2|LDA|LPM|LPBOOST|SUBGRADIENTLPM|KNN|KMEANS|HIERARCHICAL'")
	},
	{
		(CHAR*) N_LOAD_SVM,
		(&CSGInterface::cmd_load_classifier),
		(CHAR*) USAGE_O(N_LOAD_SVM, "'filename, type'")
	},
	{
		(CHAR*) N_GET_SVM,
		(&CSGInterface::cmd_get_svm),
		(CHAR*) USAGE_O(N_GET_SVM, "bias, alphas")
	},
	{
		(CHAR*) N_SET_SVM,
		(&CSGInterface::cmd_set_svm),
		(CHAR*) USAGE_I(N_SET_SVM, "bias, alphas")
	},
	{
		(CHAR*) N_GET_SVM_OBJECTIVE,
		(&CSGInterface::cmd_get_svm_objective),
		(CHAR*) USAGE_O(N_GET_SVM_OBJECTIVE, "objective")
	},
	{
		(CHAR*) N_DO_AUC_MAXIMIZATION,
		(&CSGInterface::cmd_do_auc_maximization),
		(CHAR*) USAGE_I(N_DO_AUC_MAXIMIZATION, "'auc'")
	},
	{
		(CHAR*) N_SET_PERCEPTRON_PARAMETERS,
		(&CSGInterface::cmd_set_perceptron_parameters),
		(CHAR*) USAGE_I(N_SET_PERCEPTRON_PARAMETERS, "'learnrate, maxiter'")
	},
	{
		(CHAR*) N_TRAIN_CLASSIFIER,
		(&CSGInterface::cmd_train_classifier),
		(CHAR*) USAGE(N_TRAIN_CLASSIFIER)
	},
	{
		(CHAR*) N_SVM_TRAIN,
		(&CSGInterface::cmd_train_classifier),
		(CHAR*) USAGE(N_SVM_TRAIN)
	},
	{
		(CHAR*) N_SVM_TEST,
		(&CSGInterface::cmd_test_svm),
		(CHAR*) USAGE(N_SVM_TEST)
	},
	{
		(CHAR*) N_SVMQPSIZE,
		(&CSGInterface::cmd_set_svm_qpsize),
		(CHAR*) USAGE_I(N_SVMQPSIZE, "'size'")
	},
	{
		(CHAR*) N_SVMMAXQPSIZE,
		(&CSGInterface::cmd_set_svm_max_qpsize),
		(CHAR*) USAGE_I(N_SVMMAXQPSIZE, "'size'")
	},
	{
		(CHAR*) N_SVMBUFSIZE,
		(&CSGInterface::cmd_set_svm_bufsize),
		(CHAR*) USAGE_I(N_SVMBUFSIZE, "'size'")
	},
	{
		(CHAR*) N_C,
		(&CSGInterface::cmd_set_svm_C),
		(CHAR*) USAGE_I(N_C, "'C1, C2'")
	},
	{
		(CHAR*) N_SVM_EPSILON,
		(&CSGInterface::cmd_set_svm_epsilon),
		(CHAR*) USAGE_I(N_SVM_EPSILON, "'epsilon'")
	},
	{
		(CHAR*) N_SVR_TUBE_EPSILON,
		(&CSGInterface::cmd_set_svr_tube_epsilon),
		(CHAR*) USAGE_I(N_SVR_TUBE_EPSILON, "'tube_epsilon'")
	},
	{
		(CHAR*) N_SVM_ONE_CLASS_NU,
		(&CSGInterface::cmd_set_svm_one_class_nu),
		(CHAR*) USAGE_I(N_SVM_ONE_CLASS_NU, "'nu'")
	},
	{
		(CHAR*) N_MKL_PARAMETERS,
		(&CSGInterface::cmd_set_svm_mkl_parameters),
		(CHAR*) USAGE_I(N_MKL_PARAMETERS, "'weight_epsilon, C_MKL'")
	},
	{
		(CHAR*) N_SVM_MAX_TRAIN_TIME,
		(&CSGInterface::cmd_set_max_train_time),
		(CHAR*) USAGE_I(N_SVM_MAX_TRAIN_TIME, "'max_train_time'")
	},
	{
		(CHAR*) N_USE_PRECOMPUTE,
		(&CSGInterface::cmd_set_svm_precompute_enabled),
		(CHAR*) USAGE_I(N_USE_PRECOMPUTE, "'enable_precompute'")
	},
	{
		(CHAR*) N_USE_MKL,
		(&CSGInterface::cmd_set_svm_mkl_enabled),
		(CHAR*) USAGE_I(N_USE_MKL, "'enable_mkl'")
	},
	{
		(CHAR*) N_USE_SHRINKING,
		(&CSGInterface::cmd_set_svm_shrinking_enabled),
		(CHAR*) USAGE_I(N_USE_SHRINKING, "'enable_shrinking'")
	},
	{
		(CHAR*) N_USE_BATCH_COMPUTATION,
		(&CSGInterface::cmd_set_svm_batch_computation_enabled),
		(CHAR*) USAGE_I(N_USE_BATCH_COMPUTATION, "'enable_batch_computation'")
	},
	{
		(CHAR*) N_USE_LINADD,
		(&CSGInterface::cmd_set_svm_linadd_enabled),
		(CHAR*) USAGE_I(N_USE_LINADD, "'enable_linadd'")
	},
	{
		(CHAR*) N_SVM_USE_BIAS,
		(&CSGInterface::cmd_set_svm_bias_enabled),
		(CHAR*) USAGE_I(N_SVM_USE_BIAS, "'enable_bias'")
	},


	{ (CHAR*) "Preprocessors", NULL, NULL },
	{
		(CHAR*) N_ADD_PREPROC,
		(&CSGInterface::cmd_add_preproc),
		(CHAR*) USAGE_I(N_ADD_PREPROC, "'preproc[, preproc-specific parameters]'")
	},
	{
		(CHAR*) N_DEL_PREPROC,
		(&CSGInterface::cmd_del_preproc),
		(CHAR*) USAGE(N_DEL_PREPROC)
	},
	{
		(CHAR*) N_LOAD_PREPROC,
		(&CSGInterface::cmd_load_preproc),
		(CHAR*) USAGE_I(N_LOAD_PREPROC, "'filename'")
	},
	{
		(CHAR*) N_SAVE_PREPROC,
		(&CSGInterface::cmd_save_preproc),
		(CHAR*) USAGE_I(N_SAVE_PREPROC, "'filename'")
	},
	{
		(CHAR*) N_ATTACH_PREPROC,
		(&CSGInterface::cmd_attach_preproc),
		(CHAR*) USAGE_I(N_ATTACH_PREPROC, "'TRAIN|TEST, force'")
	},
	{
		(CHAR*) N_CLEAN_PREPROC,
		(&CSGInterface::cmd_clean_preproc),
		(CHAR*) USAGE(N_CLEAN_PREPROC)
	},


	{ (CHAR*) "HMM", NULL, NULL },
	{
		(CHAR*) N_NEW_HMM,
		(&CSGInterface::cmd_new_hmm),
		(CHAR*) USAGE_I(N_NEW_HMM, "'N, M'")
	},
	{
		(CHAR*) N_LOAD_HMM,
		(&CSGInterface::cmd_load_hmm),
		(CHAR*) USAGE_I(N_LOAD_HMM, "'filename'")
	},
	{
		(CHAR*) N_SAVE_HMM,
		(&CSGInterface::cmd_save_hmm),
		(CHAR*) USAGE_I(N_SAVE_HMM, "'filename[, save_binary]'")
	},
	{
		(CHAR*) N_GET_HMM,
		(&CSGInterface::cmd_get_hmm),
		(CHAR*) USAGE_O(N_GET_HMM, "p, q, a, b")
	},
	{
		(CHAR*) N_APPEND_HMM,
		(&CSGInterface::cmd_append_hmm),
		(CHAR*) USAGE_I(N_APPEND_HMM, "p, q, a, b")
	},
	{
		(CHAR*) N_SET_HMM,
		(&CSGInterface::cmd_set_hmm),
		(CHAR*) USAGE_I(N_SET_HMM, "p, q, a, b")
	},
	{
		(CHAR*) N_SET_HMM_AS,
		(&CSGInterface::cmd_set_hmm_as),
		(CHAR*) USAGE_I(N_SET_HMM_AS, "'POS|NEG|TEST'")
	},
	{
		(CHAR*) N_CHOP,
		(&CSGInterface::cmd_set_chop),
		(CHAR*) USAGE_I(N_CHOP, "'chop'")
	},
	{
		(CHAR*) N_PSEUDO,
		(&CSGInterface::cmd_set_pseudo),
		(CHAR*) USAGE_I(N_PSEUDO, "'pseudo'")
	},
	{
		(CHAR*) N_LOAD_DEFINITIONS,
		(&CSGInterface::cmd_load_definitions),
		(CHAR*) USAGE_I(N_LOAD_DEFINITIONS, "'filename, init'")
	},
	{
		(CHAR*) N_HMM_CLASSIFY,
		(&CSGInterface::cmd_hmm_classify),
		(CHAR*) USAGE_O(N_HMM_CLASSIFY, "result")
	},
	{
		(CHAR*) N_HMM_TEST,
		(&CSGInterface::cmd_hmm_test),
		(CHAR*) USAGE_I(N_HMM_TEST, "'output name[, ROC filename[, neglinear[, poslinear]]]'")
	},
	{
		(CHAR*) N_ONE_CLASS_LINEAR_HMM_CLASSIFY,
		(&CSGInterface::cmd_one_class_linear_hmm_classify),
		(CHAR*) USAGE_O(N_ONE_CLASS_LINEAR_HMM_CLASSIFY, "result")
	},
	{
		(CHAR*) N_ONE_CLASS_HMM_TEST,
		(&CSGInterface::cmd_one_class_hmm_test),
		(CHAR*) USAGE_I(N_ONE_CLASS_HMM_TEST, "'output name[, ROC filename[, linear]]'")
	},
	{
		(CHAR*) N_ONE_CLASS_HMM_CLASSIFY,
		(&CSGInterface::cmd_one_class_hmm_classify),
		(CHAR*) USAGE_O(N_ONE_CLASS_HMM_CLASSIFY, "result")
	},
	{
		(CHAR*) N_ONE_CLASS_HMM_CLASSIFY_EXAMPLE,
		(&CSGInterface::cmd_one_class_hmm_classify_example),
		(CHAR*) USAGE_IO(N_ONE_CLASS_HMM_CLASSIFY_EXAMPLE, "feature_vector_inde", "result")
	},
	{
		(CHAR*) N_HMM_CLASSIFY_EXAMPLE,
		(&CSGInterface::cmd_hmm_classify_example),
		(CHAR*) USAGE_IO(N_HMM_CLASSIFY_EXAMPLE, "feature_vector_index", "result")
	},
	{
		(CHAR*) N_OUTPUT_HMM,
		(&CSGInterface::cmd_output_hmm),
		(CHAR*) USAGE(N_OUTPUT_HMM)
	},
	{
		(CHAR*) N_OUTPUT_HMM_DEFINED,
		(&CSGInterface::cmd_output_hmm_defined),
		(CHAR*) USAGE(N_OUTPUT_HMM_DEFINED)
	},
	{
		(CHAR*) N_HMM_LIKELIHOOD,
		(&CSGInterface::cmd_hmm_likelihood),
		(CHAR*) USAGE_O(N_HMM_LIKELIHOOD, "likelihood")
	},
	{
		(CHAR*) N_LIKELIHOOD,
		(&CSGInterface::cmd_likelihood),
		(CHAR*) USAGE(N_LIKELIHOOD)
	},
	{
		(CHAR*) N_SAVE_LIKELIHOOD,
		(&CSGInterface::cmd_save_likelihood),
		(CHAR*) USAGE_I(N_SAVE_LIKELIHOOD, "'filename[, save_binary]'")
	},
	{
		(CHAR*) N_GET_VITERBI_PATH,
		(&CSGInterface::cmd_get_viterbi_path),
		(CHAR*) USAGE_IO(N_GET_VITERBI_PATH, "dim", "path, likelihood")
	},
	{
		(CHAR*) N_VITERBI_TRAIN_DEFINED,
		(&CSGInterface::cmd_viterbi_train_defined),
		(CHAR*) USAGE(N_VITERBI_TRAIN_DEFINED)
	},
	{
		(CHAR*) N_VITERBI_TRAIN,
		(&CSGInterface::cmd_viterbi_train),
		(CHAR*) USAGE(N_VITERBI_TRAIN)
	},
	{
		(CHAR*) N_BAUM_WELCH_TRAIN,
		(&CSGInterface::cmd_baum_welch_train),
		(CHAR*) USAGE(N_BAUM_WELCH_TRAIN)
	},
	{
		(CHAR*) N_BAUM_WELCH_TRANS_TRAIN,
		(&CSGInterface::cmd_baum_welch_trans_train),
		(CHAR*) USAGE(N_BAUM_WELCH_TRANS_TRAIN)
	},
	{
		(CHAR*) N_LINEAR_TRAIN,
		(&CSGInterface::cmd_linear_train),
		(CHAR*) USAGE(N_LINEAR_TRAIN)
	},
	{
		(CHAR*) N_SAVE_PATH,
		(&CSGInterface::cmd_save_path),
		(CHAR*) USAGE_I(N_SAVE_PATH, "'filename[, save_binary]'")
	},
	{
		(CHAR*) N_CONVERGENCE_CRITERIA,
		(&CSGInterface::cmd_convergence_criteria),
		(CHAR*) USAGE_I(N_CONVERGENCE_CRITERIA, "'j, f'")
	},
	{
		(CHAR*) N_NORMALIZE,
		(&CSGInterface::cmd_normalize),
		(CHAR*) USAGE_I(N_NORMALIZE, "'[keep_dead_states]'")
	},
	{
		(CHAR*) N_ADD_STATES,
		(&CSGInterface::cmd_add_states),
		(CHAR*) USAGE_I(N_ADD_STATES, "'states, value'")
	},
	{
		(CHAR*) N_PERMUTATION_ENTROPY,
		(&CSGInterface::cmd_permutation_entropy),
		(CHAR*) USAGE_I(N_PERMUTATION_ENTROPY, "'width, seqnum'")
	},
	{
		(CHAR*) N_RELATIVE_ENTROPY,
		(&CSGInterface::cmd_relative_entropy),
		(CHAR*) USAGE_O(N_RELATIVE_ENTROPY, "result")
	},
	{
		(CHAR*) N_ENTROPY,
		(&CSGInterface::cmd_entropy),
		(CHAR*) USAGE_O(N_ENTROPY, "result")
	},
	{
		(CHAR*) N_BEST_PATH,
		(&CSGInterface::cmd_best_path),
		(CHAR*) USAGE_I(N_BEST_PATH, "'from, to'")
	},
	{
		(CHAR*) N_BEST_PATH_2STRUCT,
		(&CSGInterface::cmd_best_path_2struct),
		(CHAR*) USAGE_IO(N_BEST_PATH_2STRUCT,
			"p, q, cmd_trans, seq, pos, genestr, penalties, penalty_info, nbest, dict_weights, segment_sum_weights",
			"prob, path, pos")
	},
	{
		(CHAR*) N_BEST_PATH_TRANS,
		(&CSGInterface::cmd_best_path_trans),
		(CHAR*) USAGE_IO(N_BEST_PATH_TRANS,
			"p, q, cmd_trans, seq, pos, orf_info, genestr, penalties, state_signals, penalty_info, nbest, dict_weights, use_orf, mod_words [, segment_loss, segmend_ids_mask]",
			"prob, path, pos")
	},
	{
		(CHAR*) N_BEST_PATH_TRANS_DERIV,
		(&CSGInterface::cmd_best_path_trans_deriv),
		(CHAR*) USAGE_IO(N_BEST_PATH_TRANS_DERIV,
			"my_path, my_pos, p, q, cmd_trans, seq, pos, genestr, penalties, state_signals, penalty_info, dict_weights, mod_words [, segment_loss, segmend_ids_mask]",
			"p_deriv, q_deriv, cmd_deriv, penalties_deriv, my_scores, my_loss")
	},
	{
		(CHAR*) N_BEST_PATH_NO_B,
		(&CSGInterface::cmd_best_path_no_b),
		(CHAR*) USAGE_IO(N_BEST_PATH_NO_B, "p, q, a, max_iter", "prob, path")
	},
	{
		(CHAR*) N_BEST_PATH_TRANS_SIMPLE,
		(&CSGInterface::cmd_best_path_trans_simple),
		(CHAR*) USAGE_IO(N_BEST_PATH_TRANS_SIMPLE,
			"p, q, cmd_trans, seq, nbest", "prob, path")
	},
	{
		(CHAR*) N_BEST_PATH_NO_B_TRANS,
		(&CSGInterface::cmd_best_path_no_b_trans),
		(CHAR*) USAGE_IO(N_BEST_PATH_NO_B_TRANS,
			"p, q, cmd_trans, max_iter, nbest", "prob, path")
	},
	{
		(CHAR*) N_NEW_PLUGIN_ESTIMATOR,
		(&CSGInterface::cmd_new_plugin_estimator),
		(CHAR*) USAGE_I(N_NEW_PLUGIN_ESTIMATOR, "'pos_pseudo, neg_pseudo'")
	},
	{
		(CHAR*) N_TRAIN_ESTIMATOR,
		(&CSGInterface::cmd_train_estimator),
		(CHAR*) USAGE(N_TRAIN_ESTIMATOR)
	},
	{
		(CHAR*) N_TEST_ESTIMATOR,
		(&CSGInterface::cmd_test_estimator),
		(CHAR*) USAGE(N_TEST_ESTIMATOR)
	},
	{
		(CHAR*) N_PLUGIN_ESTIMATE_CLASSIFY_EXAMPLE,
		(&CSGInterface::cmd_plugin_estimate_classify_example),
		(CHAR*) USAGE_IO(N_PLUGIN_ESTIMATE_CLASSIFY_EXAMPLE, "feature_vector_index", "result")
	},
	{
		(CHAR*) N_PLUGIN_ESTIMATE_CLASSIFY,
		(&CSGInterface::cmd_plugin_estimate_classify),
		(CHAR*) USAGE_O(N_PLUGIN_ESTIMATE_CLASSIFY, "result")
	},
	{
		(CHAR*) N_SET_PLUGIN_ESTIMATE,
		(&CSGInterface::cmd_set_plugin_estimate),
		(CHAR*) USAGE_I(N_SET_PLUGIN_ESTIMATE, "emission_probs, model_sizes")
	},
	{
		(CHAR*) N_GET_PLUGIN_ESTIMATE,
		(&CSGInterface::cmd_get_plugin_estimate),
		(CHAR*) USAGE_O(N_GET_PLUGIN_ESTIMATE, "emission_probs, model_sizes")
	},


	{ (CHAR*) "POIM", NULL, NULL },
	{
		(CHAR*) N_COMPUTE_POIM_WD,
		(&CSGInterface::cmd_compute_POIM_WD),
		(CHAR*) USAGE_IO(N_COMPUTE_POIM_WD, "max_order, distribution", "W")
	},
	{
		(CHAR*) N_GET_SPEC_CONSENSUS,
		(&CSGInterface::cmd_get_SPEC_consensus),
		(CHAR*) USAGE_O(N_GET_SPEC_CONSENSUS, "W")
	},
	{
		(CHAR*) N_GET_SPEC_SCORING,
		(&CSGInterface::cmd_get_SPEC_scoring),
		(CHAR*) USAGE_IO(N_GET_SPEC_SCORING, "max_order", "W")
	},
	{
		(CHAR*) N_GET_WD_CONSENSUS,
		(&CSGInterface::cmd_get_WD_consensus),
		(CHAR*) USAGE_O(N_GET_WD_CONSENSUS, "W")
	},
	{
		(CHAR*) N_GET_WD_SCORING,
		(&CSGInterface::cmd_get_WD_scoring),
		(CHAR*) USAGE_IO(N_GET_WD_SCORING, "max_order", "W")
	},


	{ (CHAR*) "Utility", NULL, NULL },
	{
		(CHAR*) N_CRC,
		(&CSGInterface::cmd_crc),
		(CHAR*) USAGE_IO(N_CRC, "string", "crc32")
	},
	{
		(CHAR*) N_SYSTEM,
		(&CSGInterface::cmd_system),
		(CHAR*) USAGE_I(N_SYSTEM, "'system_command'")
	},
	{
		(CHAR*) N_EXIT,
		(&CSGInterface::cmd_exit),
		(CHAR*) USAGE(N_EXIT)
	},
	{
		(CHAR*) N_QUIT,
		(&CSGInterface::cmd_exit),
		(CHAR*) USAGE(N_QUIT)
	},
	{
		(CHAR*) N_EXEC,
		(&CSGInterface::cmd_exec),
		(CHAR*) USAGE_I(N_EXEC, "'filename'")
	},
	{
		(CHAR*) N_SET_OUTPUT,
		(&CSGInterface::cmd_set_output),
		(CHAR*) USAGE_I(N_SET_OUTPUT, "'STDERR|STDOUT|filename'")
	},
	{
		(CHAR*) N_SET_THRESHOLD,
		(&CSGInterface::cmd_set_threshold),
		(CHAR*) USAGE_I(N_SET_THRESHOLD, "'threshold'")
	},
	{
		(CHAR*) N_THREADS,
		(&CSGInterface::cmd_set_num_threads),
		(CHAR*) USAGE_I(N_THREADS, "'num_threads'")
	},
	{
		(CHAR*) N_TRANSLATE_STRING,
		(&CSGInterface::cmd_translate_string),
		(CHAR*) USAGE_IO(N_TRANSLATE_STRING,
			"string, order, start", "translation")
	},
	{
		(CHAR*) N_CLEAR,
		(&CSGInterface::cmd_clear),
		(CHAR*) USAGE(N_CLEAR)
	},
	{
		(CHAR*) N_TIC,
		(&CSGInterface::cmd_tic),
		(CHAR*) USAGE(N_TIC)
	},
	{
		(CHAR*) N_TOC,
		(&CSGInterface::cmd_toc),
		(CHAR*) USAGE(N_TOC)
	},
	{
		(CHAR*) N_ECHO,
		(&CSGInterface::cmd_echo),
		(CHAR*) USAGE_I(N_ECHO, "'level'")
	},
	{
		(CHAR*) N_LOGLEVEL,
		(&CSGInterface::cmd_loglevel),
		(CHAR*) USAGE_I(N_LOGLEVEL, "'ALL|INFO|WARN|ERROR'")
	},
	{
		(CHAR*) N_GET_VERSION,
		(&CSGInterface::cmd_get_version),
		(CHAR*) USAGE_O(N_GET_VERSION, "version")
	},
	{
		(CHAR*) N_HELP,
		(&CSGInterface::cmd_help),
		(CHAR*) USAGE(N_HELP)
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
// commands
////////////////////////////////////////////////////////////////////////////

/* Features */

bool CSGInterface::cmd_load_features()
{
	return send_command(N_LOAD_FEATURES);
}

bool CSGInterface::cmd_save_features()
{
	return send_command(N_SAVE_FEATURES);
}

bool CSGInterface::cmd_clean_features()
{
	return send_command(N_CLEAN_FEATURES);
}

bool CSGInterface::cmd_get_features()
{
	if (m_nrhs!=2 || !create_return_values(1))
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

	ASSERT(feat);

	switch (feat->get_feature_class())
	{
		case C_SIMPLE:
		{
			INT num_feat=0;
			INT num_vec=0;

			switch (feat->get_feature_type())
			{
				case F_BYTE:
				{
					BYTE* fmatrix=((CByteFeatures *) feat)->get_feature_matrix(num_feat, num_vec);
					set_byte_matrix(fmatrix, num_feat, num_vec);
					break;
				}

				case F_CHAR:
				{
					CHAR* fmatrix=((CCharFeatures *) feat)->get_feature_matrix(num_feat, num_vec);
					set_char_matrix(fmatrix, num_feat, num_vec);
					break;
				}

				case F_DREAL:
				{
					DREAL* fmatrix=((CRealFeatures *) feat)->get_feature_matrix(num_feat, num_vec);
					set_real_matrix(fmatrix, num_feat, num_vec);
					break;
				}

				case F_INT:
				{
					INT* fmatrix=((CIntFeatures *) feat)->get_feature_matrix(num_feat, num_vec);
					set_int_matrix(fmatrix, num_feat, num_vec);
					break;
				}

				case F_SHORT:
				{
					SHORT* fmatrix=((CShortFeatures *) feat)->get_feature_matrix(num_feat, num_vec);
					set_short_matrix(fmatrix, num_feat, num_vec);
					break;
				}

				case F_SHORTREAL:
				{
					SHORTREAL* fmatrix=((CShortRealFeatures *) feat)->get_feature_matrix(num_feat, num_vec);
					set_shortreal_matrix(fmatrix, num_feat, num_vec);
					break;
				}

				case F_WORD:
				{
					WORD* fmatrix=((CWordFeatures *) feat)->get_feature_matrix(num_feat, num_vec);
					set_word_matrix(fmatrix, num_feat, num_vec);
					break;
				}

				default:
					SG_NOTIMPLEMENTED;
			}
			break;
		}

		case C_SPARSE:
		{
			switch (feat->get_feature_type())
			{
				case F_DREAL:
				{
					LONG nnz=((CSparseFeatures<DREAL>*) feat)->
						get_num_nonzero_entries();
					INT num_feat=0;
					INT num_vec=0;
					TSparse<DREAL>* fmatrix=((CSparseFeatures<DREAL>*) feat)->get_sparse_feature_matrix(num_feat, num_vec);
					SG_PRINT("sparse matrix has %d feats, %d vecs and %d nnz elemements\n", num_feat, num_vec, nnz);

					set_real_sparsematrix(fmatrix, num_feat, num_vec, nnz);
					break;
				}

				default:
					SG_NOTIMPLEMENTED;
			}
			break;
		}

		case C_STRING:
		{
			INT num_str=0;
			INT max_str_len=0;
			switch (feat->get_feature_type())
			{
				case F_BYTE:
				{
					T_STRING<BYTE>* fmatrix=((CStringFeatures<BYTE>*) feat)->get_features(num_str, max_str_len);
					set_byte_string_list(fmatrix, num_str);
					break;
				}

				case F_CHAR:
				{
					T_STRING<CHAR>* fmatrix=((CStringFeatures<CHAR>*) feat)->get_features(num_str, max_str_len);
					set_char_string_list(fmatrix, num_str);
					break;
				}

				case F_WORD:
				{
					T_STRING<WORD>* fmatrix=((CStringFeatures<WORD>*) feat)->get_features(num_str, max_str_len);
					set_word_string_list(fmatrix, num_str);
					break;
				}

				default:
					SG_NOTIMPLEMENTED;
			}
			break;
		}

		default:
			SG_NOTIMPLEMENTED;
	}

	return true;
}

bool CSGInterface::cmd_add_features()
{
	if ((m_nrhs!=3 && m_nrhs!=4) || !create_return_values(0))
		return false;

	return do_set_features(true);
}

bool CSGInterface::cmd_set_features()
{
	if ((m_nrhs!=3 && m_nrhs!=4) || !create_return_values(0))
		return false;

	return do_set_features(false);
}

bool CSGInterface::do_set_features(bool add)
{
	INT tlen=0;
	CHAR* target=get_string(tlen);
	if (!strmatch(target, tlen, "TRAIN") && !strmatch(target, tlen, "TEST"))
	{
		delete[] target;
		SG_ERROR("Unknown target, neither TRAIN nor TEST.\n");
	}

	CFeatures* feat=NULL;
	INT num_feat=0;
	INT num_vec=0;

	switch (get_argument_type())
	{
		case SPARSE_REAL:
		{
			TSparse<DREAL>* fmatrix=NULL;
			get_real_sparsematrix(fmatrix, num_feat, num_vec);

			feat=new CSparseFeatures<DREAL>(0);
			ASSERT(feat);
			((CSparseFeatures<DREAL>*) feat)->
				set_sparse_feature_matrix(fmatrix, num_feat, num_vec);
			break;
		}

		case DENSE_REAL:
		{
			DREAL* fmatrix=NULL;
			get_real_matrix(fmatrix, num_feat, num_vec);

			feat=new CRealFeatures(0);
			ASSERT(feat);
			((CRealFeatures*) feat)->
				set_feature_matrix(fmatrix, num_feat, num_vec);
			break;
		}

		case DENSE_INT:
		{
			INT* fmatrix=NULL;
			get_int_matrix(fmatrix, num_feat, num_vec);

			feat=new CIntFeatures(0);
			ASSERT(feat);
			((CIntFeatures*) feat)->
				set_feature_matrix(fmatrix, num_feat, num_vec);
			break;
		}

		case DENSE_SHORT:
		{
			SHORT* fmatrix=NULL;
			get_short_matrix(fmatrix, num_feat, num_vec);

			feat=new CShortFeatures(0);
			ASSERT(feat);
			((CShortFeatures*) feat)->
				set_feature_matrix(fmatrix, num_feat, num_vec);
			break;
		}

		case DENSE_WORD:
		{
			WORD* fmatrix=NULL;
			get_word_matrix(fmatrix, num_feat, num_vec);

			feat=new CWordFeatures(0);
			ASSERT(feat);
			((CWordFeatures*) feat)->
				set_feature_matrix(fmatrix, num_feat, num_vec);
			break;
		}

		case DENSE_SHORTREAL:
		{
			SHORTREAL* fmatrix=NULL;
			get_shortreal_matrix(fmatrix, num_feat, num_vec);

			feat=new CShortRealFeatures(0);
			ASSERT(feat);
			((CShortRealFeatures*) feat)->
				set_feature_matrix(fmatrix, num_feat, num_vec);
			break;
		}

		case STRING_CHAR:
		{
			if (m_nrhs!=4)
				SG_ERROR("Please specify alphabet!\n");

			INT num_str=0;
			INT max_str_len=0;
			T_STRING<CHAR>* fmatrix=NULL;
			get_char_string_list(fmatrix, num_str, max_str_len);

			INT alphabet_len=0;
			CHAR* alphabet_str=get_string(alphabet_len);
			ASSERT(alphabet_str);

			if (strmatch(alphabet_str, alphabet_len, "DNABINFILE"))
			{
				delete[] alphabet_str;

				ASSERT(fmatrix[0].string);
				feat=new CStringFeatures<BYTE>(DNA);
				ASSERT(feat);

				if (!((CStringFeatures<BYTE>*) feat)->load_dna_file(fmatrix[0].string))
				{
					delete feat;
					SG_ERROR("Couldn't load DNA features from file.\n");
				}
			}
			else
			{
				CAlphabet* alphabet=new CAlphabet(alphabet_str, alphabet_len);
				ASSERT(alphabet);
				delete[] alphabet_str;

				feat=new CStringFeatures<CHAR>(alphabet);
				ASSERT(feat);
				if (!((CStringFeatures<CHAR>*) feat)->set_features(fmatrix, num_str, max_str_len))
				{
					delete alphabet;
					delete feat;
					SG_ERROR("Couldnt set byte string features.\n");
				}

				delete alphabet;
			}
			break;
		}

		case STRING_BYTE:
		{
			if (m_nrhs!=4)
				SG_ERROR("Please specify alphabet!\n");

			INT num_str=0;
			INT max_str_len=0;
			T_STRING<BYTE>* fmatrix=NULL;
			get_byte_string_list(fmatrix, num_str, max_str_len);

			INT alphabet_len=0;
			CHAR* alphabet_str=get_string(alphabet_len);
			ASSERT(alphabet_str);
			CAlphabet* alphabet=new CAlphabet(alphabet_str, alphabet_len);
			ASSERT(alphabet);
			delete[] alphabet_str;

			feat=new CStringFeatures<BYTE>(alphabet);
			ASSERT(feat);
			if (!((CStringFeatures<BYTE>*) feat)->set_features(fmatrix, num_str, max_str_len))
			{
				delete alphabet;
				delete feat;
				SG_ERROR("Couldnt set byte string features.\n");
			}

			delete alphabet;
			break;
		}

		default:
			SG_ERROR("Wrong argument type %d.\n", get_argument_type());
	}

	if (strmatch(target, tlen, "TRAIN"))
	{
		if (add)
			gui->guifeatures.add_train_features(feat);
		else
			gui->guifeatures.set_train_features(feat);
	}
	else
	{
		if (add)
			gui->guifeatures.add_test_features(feat);
		else
			gui->guifeatures.set_test_features(feat);
	}

	delete[] target;

	return true;
}

bool CSGInterface::cmd_set_reference_features()
{
	return send_command(N_SET_REF_FEAT);
}

bool CSGInterface::cmd_convert()
{
	return send_command(N_CONVERT);
}

bool CSGInterface::cmd_obtain_from_position_list()
{
	if ((m_nrhs!=4 && m_nrhs!=5) || !create_return_values(0))
		return false;

	INT tlen=0;
	CHAR* target=get_string(tlen);
	if (!strmatch(target, tlen, "TRAIN") && !strmatch(target, tlen, "TEST"))
	{
		delete[] target;
		SG_ERROR("Unknown target, neither TRAIN nor TEST.\n");
	}

	INT winsize=get_int();

	INT* shifts=NULL;
	INT num_shift=0;
	get_int_vector(shifts, num_shift);

	INT skip=0;
	if (m_nrhs==5)
		skip=get_int();

	SG_DEBUG("winsize: %d num_shifts: %d skip: %d\n", winsize, num_shift, skip);

	CDynamicArray<INT> positions(num_shift+1);

	for (INT i=0; i<num_shift; i++)
		positions.set_element(shifts[i], i);

	CFeatures* features=NULL;
	if (strmatch(target, tlen, "TRAIN"))
	{
		gui->guifeatures.invalidate_train();
		features=gui->guifeatures.get_train_features();
	}
	else
	{
		gui->guifeatures.invalidate_test();
		features=gui->guifeatures.get_test_features();
	}
	delete[] target;

	if (!features)
		SG_ERROR("No features.\n");

	if (features->get_feature_class()==C_COMBINED)
	{
		features=((CCombinedFeatures*) features)->get_last_feature_obj();
		if (!features)
			SG_ERROR("No features from combined.\n");
	}

	if (features->get_feature_class()!=C_STRING)
		SG_ERROR("No string features.\n");

	bool success=false;
	switch (features->get_feature_type())
	{
		case F_CHAR:
		{
			success=(((CStringFeatures<CHAR>*) features)->
				obtain_by_position_list(winsize, &positions, skip)>0);
			break;
		}
		case F_BYTE:
		{
			success=(((CStringFeatures<BYTE>*) features)->
				obtain_by_position_list(winsize, &positions, skip)>0);
			break;
		}
		case F_WORD:
		{
			success=(((CStringFeatures<WORD>*) features)->
				obtain_by_position_list(winsize, &positions, skip)>0);
			break;
		}
		case F_ULONG:
		{
			success=(((CStringFeatures<ULONG>*) features)->
				obtain_by_position_list(winsize, &positions, skip)>0);
			break;
		}
		default:
			SG_ERROR("Unsupported string features type.\n");
	}

	return success;
}

bool CSGInterface::cmd_obtain_by_sliding_window()
{
	return send_command(N_SLIDE_WINDOW);
}

bool CSGInterface::cmd_reshape()
{
	return send_command(N_RESHAPE);
}

bool CSGInterface::cmd_load_labels()
{
	return send_command(N_LOAD_LABELS);
}

bool CSGInterface::cmd_set_labels()
{
	if (m_nrhs!=3 || !create_return_values(0))
		return false;

	INT tlen=0;
	CHAR* target=get_string(tlen);
	if (!strmatch(target, tlen, "TRAIN") && !strmatch(target, tlen, "TEST"))
	{
		delete target;
		SG_ERROR("Unknown target, neither TRAIN nor TEST.\n");
	}

	DREAL* lab=NULL;
	INT len=0;
	get_real_vector(lab, len);

	CLabels* labels=new CLabels(len);
	SG_INFO("num labels: %d\n", labels->get_num_labels());

	for (INT i=0; i<len; i++)
	{
		if (!labels->set_label(i, lab[i]))
			SG_ERROR("Couldn't set label %d (of %d): %f.\n", i, len, lab[i]);
	}

	if (strmatch(target, tlen, "TRAIN"))
		gui->guilabels.set_train_labels(labels);
	else if (strmatch(target, tlen, "TEST"))
		gui->guilabels.set_test_labels(labels);
	else
	{
		delete[] target;
		SG_ERROR("Unknown target, neither TRAIN nor TEST.\n");
	}
	delete[] target;

	return true;
}

bool CSGInterface::cmd_get_labels()
{
	if (m_nrhs!=2 || !create_return_values(1))
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

	set_real_vector(lab, num_labels);
	delete[] lab;

	return true;
}


/** Kernel */

bool CSGInterface::cmd_set_kernel()
{
	return send_command(N_SET_KERNEL);
}

bool CSGInterface::cmd_add_kernel()
{
	return send_command(N_ADD_KERNEL);
}

bool CSGInterface::cmd_init_kernel()
{
	return send_command(N_INIT_KERNEL);
}

bool CSGInterface::cmd_clean_kernel()
{
	return send_command(N_CLEAN_KERNEL);
}

bool CSGInterface::cmd_save_kernel()
{
	return send_command(N_SAVE_KERNEL);
}

bool CSGInterface::cmd_load_kernel_init()
{
	return send_command(N_LOAD_KERNEL_INIT);
}

bool CSGInterface::cmd_save_kernel_init()
{
	return send_command(N_SAVE_KERNEL_INIT);
}

bool CSGInterface::cmd_get_kernel_matrix()
{
	if (m_nrhs!=1 || !create_return_values(1))
		return false;

	CKernel* kernel=gui->guikernel.get_kernel();
	if (!kernel || !kernel->get_rhs() || !kernel->get_lhs())
		SG_ERROR("No kernel defined.\n");

	INT num_vec_lhs=0;
	INT num_vec_rhs=0;
	DREAL* kmatrix=NULL;
	kmatrix=kernel->get_kernel_matrix_real(num_vec_lhs, num_vec_rhs, kmatrix);

	set_real_matrix(kmatrix, num_vec_lhs, num_vec_rhs);
	delete[] kmatrix;

	return true;
}

bool CSGInterface::cmd_set_custom_kernel()
{
	if (m_nrhs!=3 || !create_return_values(0))
		return false;

	CCustomKernel* kernel=(CCustomKernel*) gui->guikernel.get_kernel();
	if (!kernel)
		SG_ERROR("No kernel defined.\n");

	if (kernel->get_kernel_type()==K_COMBINED)
	{
		SG_DEBUG("Identified combined kernel.\n");
		kernel=(CCustomKernel*) ((CCombinedKernel*) kernel)->
			get_last_kernel();
		if (!kernel)
			SG_ERROR("No last kernel defined.\n");
	}

	if (kernel->get_kernel_type()!=K_CUSTOM)
		SG_ERROR("Not a custom kernel.\n");

	DREAL* kmatrix=NULL;
	INT num_feat=0;
	INT num_vec=0;
	get_real_matrix(kmatrix, num_feat, num_vec);

	INT tlen=0;
	CHAR* type=get_string(tlen);

	if (!strmatch(type, tlen, "DIAG") && !strmatch(type, tlen, "FULL"))
	{
		delete[] type;
		SG_ERROR("Undefined type, not DIAG, FULL or FULL2DIAG.\n");
	}

	bool source_is_diag=false;
	bool dest_is_diag=false;
	if (strmatch(type, tlen, "FULL2DIAG"))
		dest_is_diag=true;
	else if (strmatch(type, tlen, "DIAG"))
	{
		source_is_diag=true;
		dest_is_diag=true;
	}
	// change nothing if FULL

	bool success=false;
	if (source_is_diag && dest_is_diag && num_vec==num_feat)
		success=kernel->set_triangle_kernel_matrix_from_triangle(
			kmatrix, num_vec);
	else if (!source_is_diag && dest_is_diag && num_vec==num_feat)
		success=kernel->set_triangle_kernel_matrix_from_full(
			kmatrix, num_feat, num_vec);
	else
		success=kernel->set_full_kernel_matrix_from_full(
			kmatrix, num_feat, num_vec);

	return success;
}

bool CSGInterface::cmd_set_WD_position_weights()
{
	if (m_nrhs<2 || m_nrhs>3 || !create_return_values(0))
		return false;

	CKernel* kernel=gui->guikernel.get_kernel();
	if (!kernel)
		SG_ERROR("No kernel.\n");

	if (kernel->get_kernel_type()==K_COMBINED)
	{
		kernel=((CCombinedKernel*) kernel)->get_last_kernel();
		if (!kernel)
			SG_ERROR("No last kernel.\n");

		EKernelType ktype=kernel->get_kernel_type();
		if (ktype!=K_WEIGHTEDDEGREE && ktype!=K_WEIGHTEDDEGREEPOS)
			SG_ERROR("Unsupported kernel.\n");
	}

	bool success=false;
	DREAL* weights=NULL;
	INT dim=0;
	INT len=0;
	get_real_matrix(weights, dim, len);

	if (kernel->get_kernel_type()==K_WEIGHTEDDEGREE)
	{
		CWeightedDegreeStringKernel* k=
			(CWeightedDegreeStringKernel*) kernel;

		if (dim!=1 & len>0)
			SG_ERROR("Dimension mismatch (should be 1 x seq_length or 0x0\n");

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
				delete[] target;
				SG_ERROR("Second argument none of TRAIN or TEST.\n");
			}

			if (strmatch(target, tlen, "TEST"))
				is_train=false;
		}

		if (dim!=1 && len>0)
		{
			delete[] target;
			SG_ERROR("Dimension mismatch (should be 1 x seq_length or 0x0\n");
		}

		if (dim==0 && len==0)
		{
			if (create_return_values(3))
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
			if (create_return_values(3))
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

	return success;
}

bool CSGInterface::cmd_get_subkernel_weights()
{
	if (m_nrhs!=1 || !create_return_values(1))
		return false;

	CKernel *kernel=gui->guikernel.get_kernel();
	if (!kernel)
		SG_ERROR("Invalid kernel.\n");

	EKernelType ktype=kernel->get_kernel_type();
	const DREAL* weights=NULL;

	if (ktype==K_COMBINED)
	{
		INT num_weights=-1;
		weights=((CCombinedKernel *) kernel)->get_subkernel_weights(num_weights);

		set_real_vector(weights, num_weights);
		return true;
	}

	INT degree=-1;
	INT length=-1;

	if (ktype==K_WEIGHTEDDEGREE)
	{
		weights=((CWeightedDegreeStringKernel *) kernel)->
			get_degree_weights(degree, length);
	}
	else if (ktype==K_WEIGHTEDDEGREEPOS)
	{
		weights=((CWeightedDegreePositionStringKernel *) kernel)->
			get_degree_weights(degree, length);
	}
	else
		SG_ERROR("Setting subkernel weights not supported on this kernel.\n");

	if (length==0)
		length=1;

	set_real_matrix(weights, degree, length);
	return true;
}

bool CSGInterface::cmd_set_subkernel_weights()
{
	if (m_nrhs!=2 || !create_return_values(0))
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
			SG_ERROR("Dimension mismatch (should be de(seq_length | 1) x degree)\n");

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
			SG_ERROR("Dimension mismatch (should be de(seq_length | 1) x degree)\n");

		if (len==1)
			len=0;

		success=k->set_weights(weights, dim, len);
	}
	else // all other kernels
	{
		INT num_subkernels=kernel->get_num_subkernels();
		if (dim!=1 || len!=num_subkernels)
			SG_ERROR("Dimension mismatch (should be 1 x num_subkernels)\n");

		kernel->set_subkernel_weights(weights, len);
		success=true;
	}

	return success;
}

bool CSGInterface::cmd_set_subkernel_weights_combined()
{
	if (m_nrhs!=3 || !create_return_values(0))
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

	INT idx=get_int();
	SG_DEBUG("using kernel_idx=%i\n", idx);

	kernel=((CCombinedKernel*) kernel)->get_kernel(idx);
	if (!kernel)
		SG_ERROR("No subkernel at idx %d.\n", idx);

	EKernelType ktype=kernel->get_kernel_type();
	if (ktype==K_WEIGHTEDDEGREE)
	{
		CWeightedDegreeStringKernel* k=
			(CWeightedDegreeStringKernel*) kernel;
		INT degree=k->get_degree();
		if (dim!=degree || len<1)
			SG_ERROR("Dimension mismatch (should be de(seq_length | 1) x degree)\n");

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
			SG_ERROR("Dimension mismatch (should be de(seq_length | 1) x degree)\n");

		if (len==1)
			len=0;

		success=k->set_weights(weights, dim, len);
	}
	else // all other kernels
	{
		INT num_subkernels=kernel->get_num_subkernels();
		if (dim!=1 || len!=num_subkernels)
			SG_ERROR("Dimension mismatch (should be 1 x num_subkernels)\n");

		kernel->set_subkernel_weights(weights, len);
		success=true;
	}

	return success;
}

bool CSGInterface::cmd_set_last_subkernel_weights()
{
	if (m_nrhs!=2 || !create_return_values(0))
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
			SG_ERROR("Dimension mismatch (should be de(seq_length | 1) x degree)\n");

		if (len==1)
			len=0;

		success=k->set_weights(weights, dim, len);
	}
	else if (ktype==K_WEIGHTEDDEGREEPOS)
	{
		CWeightedDegreePositionStringKernel* k=
			(CWeightedDegreePositionStringKernel*) kernel;
		if (dim!=k->get_degree() || len<1)
			SG_ERROR("Dimension mismatch (should be de(seq_length | 1) x degree)\n");

		if (len==1)
			len=0;

		success=k->set_weights(weights, dim, len);
	}
	else // all other kernels
	{
		INT num_subkernels=kernel->get_num_subkernels();
		if (dim!=1 || len!=num_subkernels)
			SG_ERROR("Dimension mismatch (should be 1 x num_subkernels)\n");

		kernel->set_subkernel_weights(weights, len);
		success=true;
	}

	return success;
}

bool CSGInterface::cmd_get_WD_position_weights()
{
	if (m_nrhs!=1 || !create_return_values(1))
		return false;

	CKernel* kernel=gui->guikernel.get_kernel();
	if (!kernel)
		SG_ERROR("No kernel.\n");

	if (kernel->get_kernel_type()==K_COMBINED)
	{
		kernel=((CCombinedKernel*) kernel)->get_last_kernel();
		if (!kernel)
			SG_ERROR("Couldn't find last kernel.\n");

		EKernelType ktype=kernel->get_kernel_type();
		if (ktype!=K_WEIGHTEDDEGREE && ktype!=K_WEIGHTEDDEGREEPOS)
			SG_ERROR("Wrong subkernel type.\n");
	}

	INT len=0;
	const DREAL* position_weights;

	if (kernel->get_kernel_type()==K_WEIGHTEDDEGREE)
		position_weights=((CWeightedDegreeStringKernel*) kernel)->get_position_weights(len);
	else
		position_weights=((CWeightedDegreePositionStringKernel*) kernel)->get_position_weights(len);

	if (position_weights==NULL)
		set_real_vector(position_weights, 0);
	else
		set_real_vector(position_weights, len);

	return true;
}

bool CSGInterface::cmd_get_last_subkernel_weights()
{
	if (m_nrhs!=1 || !create_return_values(1))
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

		set_real_vector(weights, num_weights);
		return true;
	}

	DREAL* weights=NULL;
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

bool CSGInterface::cmd_compute_by_subkernels()
{
	if (m_nrhs!=1 || !create_return_values(1))
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

bool CSGInterface::cmd_init_kernel_optimization()
{
	return send_command(N_INIT_KERNEL_OPTIMIZATION);
}

bool CSGInterface::cmd_get_kernel_optimization()
{
	if (m_nrhs<1 || !create_return_values(1))
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

			INT max_order=get_int();
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

			set_real_vector(weights, len);
			return true;
		}
		case K_LINEAR:
		{
			CLinearKernel* k=(CLinearKernel*) kernel;
			INT len=0;
			const DREAL* weights=k->get_normal(len);

			set_real_vector(weights, len);
			return true;
		}
		case K_SPARSELINEAR:
		{
			CSparseLinearKernel* k=(CSparseLinearKernel*) kernel;
			INT len=0;
			const DREAL* weights=k->get_normal(len);

			set_real_vector(weights, len);
			return true;
		}
		default:
			SG_ERROR("Unsupported kernel %s.\n", kernel->get_name());
	}

	return true;
}

bool CSGInterface::cmd_delete_kernel_optimization()
{
	return send_command(N_DELETE_KERNEL_OPTIMIZATION);
}

bool CSGInterface::cmd_set_kernel_optimization_type()
{
	return send_command(N_SET_KERNEL_OPTIMIZATION_TYPE);
}

#ifdef USE_SVMLIGHT
bool CSGInterface::cmd_resize_kernel_cache()
{
	return send_command(N_RESIZE_KERNEL_CACHE);
}
#endif //USE_SVMLIGHT


/** Distance */

bool CSGInterface::cmd_set_distance()
{
	return send_command(N_SET_DISTANCE);
}

bool CSGInterface::cmd_init_distance()
{
	return send_command(N_INIT_DISTANCE);
}

bool CSGInterface::cmd_get_distance_matrix()
{
	if (m_nrhs!=1 || !create_return_values(1))
		return false;

	CDistance* distance=gui->guidistance.get_distance();
	if (!distance || !distance->get_rhs() || !distance->get_lhs())
		SG_ERROR("No distance defined.\n");

	INT num_vec_lhs=0;
	INT num_vec_rhs=0;
	DREAL* dmatrix=NULL;
	dmatrix=distance->get_distance_matrix_real(num_vec_lhs, num_vec_rhs, dmatrix);

	set_real_matrix(dmatrix, num_vec_lhs, num_vec_rhs);
	delete[] dmatrix;

	return true;
}


/* POIM */

bool CSGInterface::cmd_get_SPEC_consensus()
{
	if (m_nrhs!=1 || !create_return_values(1))
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

	set_char_vector(consensus, num_feat);
	delete[] consensus;

	return true;
}

bool CSGInterface::cmd_get_SPEC_scoring()
{
	if (m_nrhs!=2 || !create_return_values(1))
		return false;

	INT max_order=get_int();
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

bool CSGInterface::cmd_get_WD_consensus()
{
	if (m_nrhs!=1 || !create_return_values(1))
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

	set_char_vector(consensus, num_feat);
	delete[] consensus;

	return true;
}

bool CSGInterface::cmd_compute_POIM_WD()
{
	if (m_nrhs!=3 || !create_return_values(1))
		return false;

	INT max_order=get_int();
	DREAL* distribution=NULL;
	INT num_dfeat=0;
	INT num_dvec=0;
	get_real_matrix(distribution, num_dfeat, num_dvec);

	if (!distribution)
		SG_ERROR("Wrong distribution.\n");

	CKernel* kernel=gui->guikernel.get_kernel();
	if (!kernel)
		SG_ERROR("No Kernel.\n");
	if (kernel->get_kernel_type()!=K_WEIGHTEDDEGREEPOS)
		SG_ERROR("Only works for Weighted Degree Position kernels.\n");

	INT seqlen=0;
	INT num_sym=0;
	CStringFeatures<CHAR>* sfeat=(CStringFeatures<CHAR>*)
		(((CWeightedDegreePositionStringKernel*) kernel)->get_lhs());
	ASSERT(sfeat);
	seqlen=sfeat->get_max_vector_length();
	num_sym=(INT) sfeat->get_num_symbols();

	if (num_dvec!=seqlen || num_dfeat!=num_sym)
	{
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

	set_real_matrix(position_weights, num_sym, seqlen);
	delete[] position_weights;

	return true;
}

bool CSGInterface::cmd_get_WD_scoring()
{
	if (m_nrhs!=2 || !create_return_values(1))
		return false;

	INT max_order=get_int();

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


/* Classifier */

bool CSGInterface::cmd_classify()
{
	if (m_nrhs!=1 || !create_return_values(1))
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

	set_real_vector(result, num_vec);
	delete[] result;

	return true;
}

bool CSGInterface::cmd_classify_example()
{
	if (m_nrhs!=2 || !create_return_values(1))
		return false;

	INT idx=get_int();
	DREAL result=0;

	if (!gui->guiclassifier.classify_example(idx, result))
		SG_ERROR("Classify_example failed.\n");

	set_real(result);

	return true;
}

bool CSGInterface::cmd_get_classifier()
{
	if (m_nrhs!=1 || !create_return_values(2))
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
	set_real_matrix(weights, rows, cols);

	return true;
}

bool CSGInterface::cmd_new_classifier()
{
	return send_command(N_NEW_CLASSIFIER);
}

bool CSGInterface::cmd_load_classifier()
{
	return send_command(N_LOAD_SVM); // FIXME: N_LOAD_CLASSIFIER
}

bool CSGInterface::cmd_get_svm()
{
	return cmd_get_classifier();
}

bool CSGInterface::cmd_set_svm()
{
	if (m_nrhs!=3 || !create_return_values(0))
		return false;

	DREAL bias=get_real();

	DREAL* alphas=NULL;
	INT num_feat_alphas=0;
	INT num_vec_alphas=0;
	get_real_matrix(alphas, num_feat_alphas, num_vec_alphas);

	if (!alphas)
		SG_ERROR("No proper alphas given.\n");
	if (num_vec_alphas!=2)
		SG_ERROR("Not 2 vectors in alphas.\n");

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

	return true;
}

bool CSGInterface::cmd_get_svm_objective()
{
	if (m_nrhs!=1 || !create_return_values(1))
		return false;

	CSVM* svm=(CSVM*) gui->guiclassifier.get_classifier();
	if (!svm)
		SG_ERROR("No SVM set.\n");

	set_real(svm->get_objective());

	return true;
}

bool CSGInterface::cmd_do_auc_maximization()
{
	return send_command(N_DO_AUC_MAXIMIZATION);
}

bool CSGInterface::cmd_set_perceptron_parameters()
{
	return send_command(N_SET_PERCEPTRON_PARAMETERS);
}

bool CSGInterface::cmd_train_classifier()
{
	return send_command(N_TRAIN_CLASSIFIER);
}

bool CSGInterface::cmd_test_svm()
{
	return send_command(N_SVM_TEST);
}

bool CSGInterface::cmd_set_svm_qpsize()
{
	return send_command(N_SVMQPSIZE);
}

bool CSGInterface::cmd_set_svm_max_qpsize()
{
	return send_command(N_SVMMAXQPSIZE);
}

bool CSGInterface::cmd_set_svm_bufsize()
{
	return send_command(N_SVMBUFSIZE);
}

bool CSGInterface::cmd_set_svm_C()
{
	return send_command(N_C);
}

bool CSGInterface::cmd_set_svm_epsilon()
{
	return send_command(N_SVM_EPSILON);
}

bool CSGInterface::cmd_set_svr_tube_epsilon()
{
	return send_command(N_SVR_TUBE_EPSILON);
}

bool CSGInterface::cmd_set_svm_one_class_nu()
{
	return send_command(N_SVM_ONE_CLASS_NU);
}

bool CSGInterface::cmd_set_svm_mkl_parameters()
{
	return send_command(N_MKL_PARAMETERS);
}

bool CSGInterface::cmd_set_max_train_time()
{
	return send_command(N_SVM_MAX_TRAIN_TIME);
}

bool CSGInterface::cmd_set_svm_precompute_enabled()
{
	return send_command(N_USE_PRECOMPUTE);
}

bool CSGInterface::cmd_set_svm_mkl_enabled()
{
	return send_command(N_USE_MKL);
}

bool CSGInterface::cmd_set_svm_shrinking_enabled()
{
	return send_command(N_USE_SHRINKING);
}

bool CSGInterface::cmd_set_svm_batch_computation_enabled()
{
	return send_command(N_USE_BATCH_COMPUTATION);
}

bool CSGInterface::cmd_set_svm_linadd_enabled()
{
	return send_command(N_USE_LINADD);
}

bool CSGInterface::cmd_set_svm_bias_enabled()
{
	return send_command(N_SVM_USE_BIAS);
}


/* Preproc */

bool CSGInterface::cmd_add_preproc()
{
	return send_command(N_ADD_PREPROC);
}

bool CSGInterface::cmd_del_preproc()
{
	return send_command(N_DEL_PREPROC);
}

bool CSGInterface::cmd_load_preproc()
{
	return send_command(N_LOAD_PREPROC);
}

bool CSGInterface::cmd_save_preproc()
{
	return send_command(N_SAVE_PREPROC);
}

bool CSGInterface::cmd_attach_preproc()
{
	return send_command(N_ATTACH_PREPROC);
}

bool CSGInterface::cmd_clean_preproc()
{
	return send_command(N_CLEAN_PREPROC);
}


/* HMM */

bool CSGInterface::cmd_new_plugin_estimator()
{
	return send_command(N_NEW_PLUGIN_ESTIMATOR);
}

bool CSGInterface::cmd_train_estimator()
{
	return send_command(N_TRAIN_ESTIMATOR);
}

bool CSGInterface::cmd_test_estimator()
{
	return send_command(N_TEST_ESTIMATOR);
}

bool CSGInterface::cmd_plugin_estimate_classify_example()
{
	if (m_nrhs!=2 || !create_return_values(1))
		return false;

	INT idx=get_int();
	DREAL result=gui->guipluginestimate.classify_example(idx);

	set_real_vector(&result, 1);
	return true;
}

bool CSGInterface::cmd_plugin_estimate_classify()
{
	if (m_nrhs!=1 || !create_return_values(1))
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

	set_real_vector(result, num_vec);
	delete[] result;

	return true;
}

bool CSGInterface::cmd_set_plugin_estimate()
{
	if (m_nrhs!=3 || !create_return_values(0))
		return false;

	DREAL* emission_probs=NULL;
	INT num_probs=0;
	INT num_vec=0;
	get_real_matrix(emission_probs, num_probs, num_vec);

	if (num_vec!=2)
		SG_ERROR("Need at least 1 set of positive and 1 set of negative params.\n");

	DREAL* pos_params=emission_probs;
	DREAL* neg_params=&(emission_probs[num_probs]);

	DREAL* model_sizes=NULL;
	INT len=0;
	get_real_vector(model_sizes, len);

	INT seq_length=(INT) model_sizes[0];
	INT num_symbols=(INT) model_sizes[1];
	if (num_probs!=seq_length*num_symbols)
		SG_ERROR("Mismatch in number of emission probs and sequence length * number of symbols.\n");

	gui->guipluginestimate.get_estimator()->set_model_params(
		pos_params, neg_params, seq_length, num_symbols);

	return true;
}

bool CSGInterface::cmd_get_plugin_estimate()
{
	if (m_nrhs!=1 || !create_return_values(2))
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
	set_real_vector(model_sizes, 2);

	return true;
}

bool CSGInterface::cmd_convergence_criteria()
{
	return send_command(N_CONVERGENCE_CRITERIA);
}

bool CSGInterface::cmd_normalize()
{
	return send_command(N_NORMALIZE);
}

bool CSGInterface::cmd_add_states()
{
	return send_command(N_ADD_STATES);
}

bool CSGInterface::cmd_permutation_entropy()
{
	return send_command(N_PERMUTATION_ENTROPY);
}

bool CSGInterface::cmd_relative_entropy()
{
	if (m_nrhs!=1 || !create_return_values(1))
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

	set_real_vector(entropy, pos_N);
	delete[] entropy;

	return true;
}

bool CSGInterface::cmd_entropy()
{
	if (m_nrhs!=1 || !create_return_values(1))
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

	set_real_vector(entropy, N);
	delete[] entropy;

	return true;
}

bool CSGInterface::cmd_hmm_classify()
{
	return do_hmm_classify(false, false);
}

bool CSGInterface::cmd_hmm_test()
{
	return send_command(N_HMM_TEST);
}

bool CSGInterface::cmd_one_class_hmm_test()
{
	return send_command(N_ONE_CLASS_HMM_TEST);
}

bool CSGInterface::cmd_one_class_hmm_classify()
{
	return do_hmm_classify(false, true);
}

bool CSGInterface::cmd_one_class_linear_hmm_classify()
{
	return do_hmm_classify(true, true);
}

bool CSGInterface::do_hmm_classify(bool linear, bool one_class)
{
	if (m_nrhs>1 || !create_return_values(1))
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

	set_real_vector(result, num_vec);
	delete[] result;

	return true;
}

bool CSGInterface::cmd_one_class_hmm_classify_example()
{
	return do_hmm_classify_example(true);
}

bool CSGInterface::cmd_hmm_classify_example()
{
	return do_hmm_classify_example(false);
}

bool CSGInterface::do_hmm_classify_example(bool one_class)
{
	if (m_nrhs!=2 || !create_return_values(1))
		return false;

	INT idx=get_int();
	DREAL result=0;

	if (one_class)
		result=gui->guihmm.one_class_classify_example(idx);
	else
		result=gui->guihmm.classify_example(idx);

	set_real(result);

	return true;
}

bool CSGInterface::cmd_output_hmm()
{
	return send_command(N_OUTPUT_HMM);
}

bool CSGInterface::cmd_output_hmm_defined()
{
	return send_command(N_OUTPUT_HMM_DEFINED);
}

bool CSGInterface::cmd_hmm_likelihood()
{
	if (m_nrhs!=1 || !create_return_values(1))
		return false;

	CHMM* h=gui->guihmm.get_current();
	if (!h)
		SG_ERROR("No HMM.\n");

	DREAL likelihood=h->model_probability();
	set_real(likelihood);

	return true;
}

bool CSGInterface::cmd_likelihood()
{
	return send_command(N_LIKELIHOOD);
}

bool CSGInterface::cmd_save_likelihood()
{
	return send_command(N_SAVE_LIKELIHOOD);
}

bool CSGInterface::cmd_get_viterbi_path()
{
	if (m_nrhs!=2 || !create_return_values(2))
		return false;

	INT dim=get_int();
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
	T_STATES* path=h->get_path(dim, likelihood);

	set_word_vector(path, num_feat);
	delete[] path;
	set_real(likelihood);

	return true;
}

bool CSGInterface::cmd_viterbi_train()
{
	return send_command(N_VITERBI_TRAIN);
}

bool CSGInterface::cmd_viterbi_train_defined()
{
	return send_command(N_VITERBI_TRAIN_DEFINED);
}

bool CSGInterface::cmd_baum_welch_train()
{
	return send_command(N_BAUM_WELCH_TRAIN);
}

bool CSGInterface::cmd_baum_welch_trans_train()
{
	return send_command(N_BAUM_WELCH_TRANS_TRAIN);
}

bool CSGInterface::cmd_linear_train()
{
	return send_command(N_LINEAR_TRAIN);
}

bool CSGInterface::cmd_save_path()
{
	return send_command(N_SAVE_PATH);
}

bool CSGInterface::cmd_append_hmm()
{
	if (m_nrhs!=5 || !create_return_values(0))
		return false;

	CHMM* old_h=gui->guihmm.get_current();
	if (!old_h)
		SG_ERROR("No current HMM set.\n");

	DREAL* p=NULL;
	INT N_p=0;
	get_real_vector(p, N_p);

	DREAL* q=NULL;
	INT N_q=0;
	get_real_vector(q, N_q);

	DREAL* a=NULL;
	INT M_a=0;
	INT N_a=0;
	get_real_matrix(a, M_a, N_a);
	INT N=N_a;

	DREAL* b=NULL;
	INT M_b=0;
	INT N_b=0;
	get_real_matrix(b, M_b, N_b);
	INT M=N_b;

	if (N_p!=N || N_q!=N || N_a!=N || M_a!=N || N_b!=M || M_b!=N)
	{
		SG_ERROR("Model matrices not matching in size.\n"
				"p:(%d) q:(%d) a:(%d,%d) b(%d,%d)\n",
				N_p, N_q, N_a, M_a, N_b, M_b);
	}

	CHMM* h=new CHMM(N, M, NULL, gui->guihmm.get_pseudo());
	ASSERT(h);
	INT i,j;

	for (i=0; i<N; i++)
	{
		h->set_p(i, p[i]);
		h->set_q(i, q[i]);
	}

	for (i=0; i<N; i++)
		for (j=0; j<N; j++)
			h->set_a(i,j, a[i+j*N]);

	for (i=0; i<N; i++)
		for (j=0; j<M; j++)
			h->set_b(i,j, b[i+j*N]);

	old_h->append_model(h);
	delete h;

	return true;
}

bool CSGInterface::cmd_new_hmm()
{
	return send_command(N_NEW_HMM);
}

bool CSGInterface::cmd_load_hmm()
{
	return send_command(N_LOAD_HMM);
}

bool CSGInterface::cmd_save_hmm()
{
	return send_command(N_SAVE_HMM);
}

bool CSGInterface::cmd_set_hmm()
{
	if (m_nrhs!=5 || !create_return_values(0))
		return false;

	DREAL* p=NULL;
	INT N_p=0;
	get_real_vector(p, N_p);

	DREAL* q=NULL;
	INT N_q=0;
	get_real_vector(q, N_q);

	DREAL* a=NULL;
	INT M_a=0;
	INT N_a=0;
	get_real_matrix(a, M_a, N_a);
	INT N=N_a;

	DREAL* b=NULL;
	INT M_b=0;
	INT N_b=0;
	get_real_matrix(b, M_b, N_b);
	INT M=N_b;

	if (N_p!=N || N_q!=N || N_a!=N || M_a!=N || N_b!=M || M_b!=N)
	{
		SG_ERROR("Model matrices not matching in size.\n"
				"p:(%d) q:(%d) a:(%d,%d) b(%d,%d)\n",
				N_p, N_q, N_a, M_a, N_b, M_b);
	}

	CHMM* current=gui->guihmm.get_current();
	if (!current)
		SG_ERROR("Need a previously created HMM.\n");

	INT i,j;

	for (i=0; i<N; i++)
	{
		current->set_p(i, p[i]);
		current->set_q(i, q[i]);
	}

	for (i=0; i<N; i++)
		for (j=0; j<N; j++)
			current->set_a(i,j, a[i+j*N]);

	for (i=0; i<N; i++)
		for (j=0; j<M; j++)
			current->set_b(i,j, b[i+j*N]);

	CStringFeatures<WORD>* sf = ((CStringFeatures<WORD>*) (gui->guifeatures.get_train_features()));
	current->set_observations(sf);

	return true;
}

bool CSGInterface::cmd_set_hmm_as()
{
	return send_command(N_SET_HMM_AS);
}

bool CSGInterface::cmd_set_chop()
{
	return send_command(N_CHOP);
}

bool CSGInterface::cmd_set_pseudo()
{
	return send_command(N_PSEUDO);
}

bool CSGInterface::cmd_load_definitions()
{
	return send_command(N_LOAD_DEFINITIONS);
}

bool CSGInterface::cmd_get_hmm()
{
	if (m_nrhs!=1 || !create_return_values(4))
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

	set_real_vector(p, N);
	delete[] p;
	set_real_vector(q, N);
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

bool CSGInterface::cmd_best_path()
{
	return send_command(N_BEST_PATH);
}

bool CSGInterface::cmd_best_path_2struct()
{
	if (m_nrhs!=12 || !create_return_values(3))
		return false;

	SG_ERROR("Sorry, this parameter list is awful!\n");
	
	return true;
}

bool CSGInterface::cmd_best_path_trans()
{
	if ((m_nrhs==15 || m_nrhs==17) || !create_return_values(3))
		return false;

	SG_ERROR("Sorry, this parameter list is awful!\n");
	
	return true;
}

bool CSGInterface::cmd_best_path_trans_deriv()
{
	if (!((m_nrhs==14 && create_return_values(5)) || (m_nrhs==16 && create_return_values(6))))
		return false;

	SG_ERROR("Sorry, this parameter list is awful!\n");

	return true;
}

bool CSGInterface::cmd_best_path_no_b()
{
	if (m_nrhs!=5 || !create_return_values(2))
		return false;

	DREAL* p=NULL;
	INT N_p=0;
	get_real_vector(p, N_p);

	DREAL* q=NULL;
	INT N_q=0;
	get_real_vector(q, N_q);

	DREAL* a=NULL;
	INT M_a=0;
	INT N_a=0;
	get_real_matrix(a, M_a, N_a);

	if (N_q!=N_p || N_a!=N_p || M_a!=N_p)
		SG_ERROR("Model matrices not matching in size.\n");

	INT max_iter=get_int();
	if (max_iter<1)
		SG_ERROR("max_iter < 1.\n");

	CDynProg* h=new CDynProg();
	ASSERT(h);
	h->set_N(N_p);
	h->set_p_vector(p, N_p);
	h->set_q_vector(q, N_p);
	h->set_a(a, N_p, N_p);

	INT* path=new INT[max_iter];
	ASSERT(path);
	INT best_iter=0;
	DREAL prob=h->best_path_no_b(max_iter, best_iter, path);
	delete h;

	set_real(prob);
	set_int_vector(path, best_iter+1);
	delete[] path;

	return true;
}

bool CSGInterface::cmd_best_path_trans_simple()
{
	if (m_nrhs!=6 || !create_return_values(2))
		return false;

	DREAL* p=NULL;
	INT N_p=0;
	get_real_vector(p, N_p);

	DREAL* q=NULL;
	INT N_q=0;
	get_real_vector(q, N_q);

	DREAL* cmd_trans=NULL;
	INT M_cmd_trans=0;
	INT N_cmd_trans=0;
	get_real_matrix(cmd_trans, M_cmd_trans, N_cmd_trans);

	DREAL* seq=NULL;
	INT M_seq=0;
	INT N_seq=0;
	get_real_matrix(seq, M_seq, N_seq);

	if (N_q!=N_p || N_cmd_trans!=3 || M_seq!=N_p)
		SG_ERROR("Model matrices not matching in size.\n");

	INT nbest=get_int();
	if (nbest<1)
		SG_ERROR("nbest < 1.\n");

	CDynProg* h=new CDynProg();
	ASSERT(h);
	h->set_N(N_p);
	h->set_p_vector(p, N_p);
	h->set_q_vector(q, N_p);
	h->set_a_trans_matrix(cmd_trans, M_cmd_trans, 3);

	INT* path=new INT[N_seq*nbest];
	ASSERT(path);
	memset(path, -1, N_seq*nbest*sizeof(INT));
	DREAL* prob=new DREAL[nbest];
	ASSERT(prob);

	h->best_path_trans_simple(seq, N_seq, nbest, prob, path);
	delete h;

	set_real_vector(prob, nbest);
	delete[] prob;

	set_int_matrix(path, nbest, N_seq);
	delete[] path;

	return true;
}


bool CSGInterface::cmd_best_path_no_b_trans()
{
	if (m_nrhs!=6 || !create_return_values(2))
		return false;

	DREAL* p=NULL;
	INT N_p=0;
	get_real_vector(p, N_p);

	DREAL* q=NULL;
	INT N_q=0;
	get_real_vector(q, N_q);

	DREAL* cmd_trans=NULL;
	INT M_cmd_trans=0;
	INT N_cmd_trans=0;
	get_real_matrix(cmd_trans, M_cmd_trans, N_cmd_trans);

	if (N_q!=N_p || N_cmd_trans!=3)
		SG_ERROR("Model matrices not matching in size.\n");

	INT max_iter=get_int();
	if (max_iter<1)
		SG_ERROR("max_iter < 1.\n");

	INT nbest=get_int();
	if (nbest<1)
		SG_ERROR("nbest < 1.\n");

	CDynProg* h=new CDynProg();
	ASSERT(h);
	h->set_N(N_p);
	h->set_p_vector(p, N_p);
	h->set_q_vector(q, N_p);
	h->set_a_trans_matrix(cmd_trans, M_cmd_trans, 3);

	INT* path=new INT[(max_iter+1)*nbest];
	ASSERT(path);
	memset(path, -1, (max_iter+1)*nbest*sizeof(INT));
	INT max_best_iter=0;
	DREAL* prob=new DREAL[nbest];
	ASSERT(prob);

	h->best_path_no_b_trans(max_iter, max_best_iter, nbest, prob, path);
	delete h;

	set_real_vector(prob, nbest);
	delete[] prob;

	set_int_matrix(path, nbest, max_best_iter+1);
	delete[] path;

	return true;
}


bool CSGInterface::cmd_crc()
{
	if (m_nrhs!=2 || !create_return_values(1))
		return false;

	INT slen=0;
	CHAR* string=get_string(slen);
	ASSERT(string);
	BYTE* bstring=new BYTE[slen];
	ASSERT(bstring);

	for (INT i=0; i<slen; i++)
		bstring[i]=string[i];
	delete[] string;

	INT val=CMath::crc32(bstring, slen);
	delete[] bstring;
	set_int(val);

	return true;
}

bool CSGInterface::cmd_system()
{
	return send_command(N_SYSTEM);
}

bool CSGInterface::cmd_exit()
{
	exit(0);
}

bool CSGInterface::cmd_exec()
{
	return send_command(N_EXEC);
}

bool CSGInterface::cmd_set_output()
{
	return send_command(N_SET_OUTPUT);
}

bool CSGInterface::cmd_set_threshold()
{
	return send_command(N_SET_THRESHOLD);
}

bool CSGInterface::cmd_set_num_threads()
{
	return send_command(N_THREADS);
}

bool CSGInterface::cmd_translate_string()
{
	if (m_nrhs!=4 || !create_return_values(1))
		return false;

	DREAL* string=NULL;
	INT len;
	get_real_vector(string, len);

	INT order=get_int();
	INT start=get_int();

	const INT max_val=2; /* DNA->2bits */
	INT i,j;

	WORD* obs=new WORD[len];
	ASSERT(obs);

	for (i=0; i<len; i++)
	{
		switch ((CHAR) string[i])
		{
			case 'A': obs[i]=0; break;
			case 'C': obs[i]=1; break;
			case 'G': obs[i]=2; break;
			case 'T': obs[i]=3; break;
			case 'a': obs[i]=0; break;
			case 'c': obs[i]=1; break;
			case 'g': obs[i]=2; break;
			case 't': obs[i]=3; break;
			default: SG_ERROR("Wrong letter in string.\n");
		}
	}

	//convert interval of size T
	for (i=len-1; i>=order-1; i--)
	{
		WORD value=0;
		for (j=i; j>=i-order+1; j--)
			value=(value>>max_val) | ((obs[j])<<(max_val*(order-1)));
		
		obs[i]=(WORD) value;
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

	DREAL* real_obs=new DREAL[len];
	ASSERT(real_obs);

	for (i=start; i<len; i++)
		real_obs[i-start]=(DREAL) obs[i];
	delete[] obs;

	set_real_vector(real_obs, len);
	delete[] real_obs;

	return true;
}

bool CSGInterface::cmd_clear()
{
	return send_command(N_CLEAR);
}

bool CSGInterface::cmd_tic()
{
	gui->guitime.start();
	return true;
}

bool CSGInterface::cmd_toc()
{
	gui->guitime.stop();
	return true;
}

bool CSGInterface::cmd_echo()
{
	return send_command(N_ECHO);
}

bool CSGInterface::cmd_loglevel()
{
	return send_command(N_LOGLEVEL);
}

bool CSGInterface::cmd_get_version()
{
	if (m_nrhs!=1 || !create_return_values(1))
		return false;

	set_int(version.get_version_revision());

	return true;
}

bool CSGInterface::cmd_help()
{
	if ((m_nrhs!=1 && m_nrhs!=2) || !create_return_values(0))
		return false;

	INT i=0;

	SG_PRINT("\n");
	if (m_nrhs==1) // all commands' help
	{
		SG_PRINT("Listing of all commands:\n");
		while (sg_methods[i].command)
		{
			bool is_group_item=false;
			if (!sg_methods[i].method && !sg_methods[i].usage)
				is_group_item=true;

			if (is_group_item)
				SG_PRINT("\n");

			SG_PRINT("%s\n", sg_methods[i].command);

			if (is_group_item)
			{
				for (UINT j=0; j<strlen(sg_methods[i].command); j++)
					SG_PRINT("-");
				SG_PRINT("\n");
			}

			i++;
		}
		SG_PRINT("\n\nUse sg('help', 'command') to see the usage pattern of a single command, e.g.\n\n\tsg('help', 'classify')\n\nto see the usage pattern of the command 'classify'.\n");
	}
	else // m_nrhs == 2 -> single command or group help
	{
		bool found=false;
		bool in_group=false;
		INT clen=0;
		CHAR* command=get_string(clen);

		while (sg_methods[i].command)
		{
			if (in_group)
			{
				if (sg_methods[i].usage) // display group item
					SG_PRINT("\t%s\n", sg_methods[i].command);
					//SG_PRINT("\t%s: %s\n", sg_methods[i].command, sg_methods[i].usage);
				else // next group -> end
					break;
			}
			else
			{
				found=strmatch(sg_methods[i].command, clen, command);
				if (found)
				{
					if (sg_methods[i].usage)
					{
						SG_PRINT("Usage for %s\n\n\t%s\n",
							sg_methods[i].command, sg_methods[i].usage);
						break;
					}
					else // group item
					{
						SG_PRINT("Commands in group %s\n\n", sg_methods[i].command);
						in_group=true;
					}
				}
			}

			i++;
		}

		if (!found)
			SG_PRINT("Could not find help for command %s.\n", command);

		delete[] command;
	}


	SG_PRINT("\n");

	return true;
}

bool CSGInterface::send_command(const CHAR* cmd)
{
	INT len_args=0;
	CHAR* args=interface->get_string(len_args);
	INT len_line=strlen(cmd)+len_args+2;
	CHAR* line=new CHAR[len_line];

	*line='\0';
	strncat(line, cmd, len_line);
	strncat(line, " \0", len_line);
	strncat(line, args, len_line);

	bool success=gui->parse_line(line);

	delete[] args;
	delete[] line;

	return success;
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
		SG_ERROR("GUI could not be initialized.\n");

	CHAR* command=NULL;
	try
	{
		command=interface->get_command(len);
	}
	catch (ShogunException e)
	{
		SG_ERROR("String expected as first argument: %s\n", e.get_exception_string());
	}

	SG_DEBUG("command: %s, nrhs %d\n", command, m_nrhs);
	INT i=0;
	while (sg_methods[i].command)
	{
		if (strmatch(command, len, sg_methods[i].command))
		{
			SG_DEBUG("found method %s\n", sg_methods[i].command);
			if (!(interface->*(sg_methods[i].method))())
				SG_ERROR("Usage: %s\n", sg_methods[i].usage);
			else
			{
				success=true;
				break;
			}
		}
		i++;
	}

	// FIXME: invoke old interface
	if (!success && strmatch(command, len, N_SEND_COMMAND))
	{
		CHAR* cmd=interface->get_string(len);
		gui->parse_line(cmd);

		delete[] cmd;
		success=true;
	}

#ifndef WIN32
	CSignal::unset_handler();
#endif

	delete[] command;
	return success;
}

#endif // !HAVE_SWIG
