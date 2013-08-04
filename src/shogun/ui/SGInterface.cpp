#include <shogun/ui/GUICommands.h>
#include <shogun/ui/SGInterface.h>
#include <shogun/ui/SyntaxHighLight.h>

#include <shogun/lib/config.h>
#include <shogun/lib/DataType.h>
#include <shogun/lib/SGNDArray.h>
#include <shogun/lib/common.h>
#include <shogun/lib/ShogunException.h>
#include <shogun/mathematics/Math.h>
#include <shogun/lib/Hash.h>
#include <shogun/lib/Map.h>
#include <shogun/lib/Signal.h>

#include <shogun/classifier/svm/SVM.h>
#include <shogun/lib/external/pr_loqo.h>
#include <shogun/machine/LinearMachine.h>
#include <shogun/classifier/mkl/MKL.h>
#include <shogun/kernel/string/WeightedDegreePositionStringKernel.h>
#include <shogun/kernel/string/WeightedDegreeStringKernel.h>
#include <shogun/kernel/string/CommWordStringKernel.h>
#include <shogun/kernel/string/WeightedCommWordStringKernel.h>
#include <shogun/kernel/LinearKernel.h>
#include <shogun/kernel/CombinedKernel.h>
#include <shogun/kernel/CustomKernel.h>
#include <shogun/kernel/string/SalzbergWordStringKernel.h>
#include <shogun/kernel/WaveletKernel.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/PolyFeatures.h>
#include <shogun/preprocessor/SortWordString.h>

#include <shogun/labels/BinaryLabels.h>

#include <shogun/structure/Plif.h>
#include <shogun/structure/PlifArray.h>
#include <shogun/structure/PlifBase.h>
#include <shogun/structure/DynProg.h>
#include <shogun/structure/IntronList.h>
#include <shogun/structure/SegmentLoss.h>

#include <ctype.h>

using namespace shogun;

CSGInterface* interface=NULL;
CSyntaxHighLight hilight;

#if defined(HAVE_CMDLINE)
#define USAGE(method) "", ""
#define USAGE_I(method, in) "", " " in ""
#define USAGE_O(method, out) "" out " = ", ""
#define USAGE_IO(method, in, out) "" out " = ", " " in ""
#define USAGE_COMMA " "
#define USAGE_STR ""
#elif defined(HAVE_R)
#define USAGE(method) "sg('", "')"
#define USAGE_I(method, in) "sg('", "', " in ")"
#define USAGE_O(method, out) "[" out "] <- sg('", "')"
#define USAGE_IO(method, in, out) "[" out "] <- sg('", "', " in ")"
#define USAGE_COMMA ", "
#define USAGE_STR "'"
#else
#define USAGE(method) "sg('", "')"
#define USAGE_I(method, in) "sg('", "', " in ")"
#define USAGE_O(method, out) "[" out "]=sg('", "')"
#define USAGE_IO(method, in, out) "[" out "]=sg('", "', " in ")"
#define USAGE_COMMA ", "
#define USAGE_STR "'"
#endif

CSGInterfaceMethod sg_methods[]=
{
	{ "Features", NULL, NULL, NULL },
	{
		N_PR_LOQO,
		(&CSGInterface::cmd_pr_loqo),
		USAGE_IO(N_PR_LOQO,
			"'Var1', Var1, 'Var2', Var2", "results")
	},
	{
		N_LOAD_FEATURES,
		(&CSGInterface::cmd_load_features),
		USAGE_I(N_LOAD_FEATURES,
			"filename" USAGE_COMMA "feature_class" USAGE_COMMA "type" USAGE_COMMA "target[" USAGE_COMMA "size[" USAGE_COMMA "comp_features]]")
	},
	{
		N_SAVE_FEATURES,
		(&CSGInterface::cmd_save_features),
		USAGE_I(N_SAVE_FEATURES, "filename" USAGE_COMMA "type" USAGE_COMMA "target")
	},
	{
		N_CLEAN_FEATURES,
		(&CSGInterface::cmd_clean_features),
		USAGE_I(N_CLEAN_FEATURES, USAGE_STR "TRAIN|TEST" USAGE_STR )
	},
	{
		N_GET_FEATURES,
		(&CSGInterface::cmd_get_features),
		USAGE_IO(N_GET_FEATURES, USAGE_STR "TRAIN|TEST" USAGE_STR, "features")
	},
	{
		N_ADD_FEATURES,
		(&CSGInterface::cmd_add_features),
		USAGE_I(N_ADD_FEATURES,
			USAGE_STR "TRAIN|TEST" USAGE_STR USAGE_COMMA "features[" USAGE_COMMA "DNABINFILE|<ALPHABET>]")
	},
	{
		N_ADD_MULTIPLE_FEATURES,
		(&CSGInterface::cmd_add_multiple_features),
		USAGE_I(N_ADD_MULTIPLE_FEATURES,
			USAGE_STR "TRAIN|TEST" USAGE_STR USAGE_COMMA "repetitions" USAGE_COMMA "features[" USAGE_COMMA "DNABINFILE|<ALPHABET>]")
	},
	{
		N_ADD_DOTFEATURES,
		(&CSGInterface::cmd_add_dotfeatures),
		USAGE_I(N_ADD_DOTFEATURES,
			USAGE_STR "TRAIN|TEST" USAGE_STR USAGE_COMMA "features[" USAGE_COMMA "DNABINFILE|<ALPHABET>]")
	},
	{
		N_SET_FEATURES,
		(&CSGInterface::cmd_set_features),
		USAGE_I(N_SET_FEATURES,
			USAGE_STR "TRAIN|TEST" USAGE_STR
			USAGE_COMMA "features["
			USAGE_COMMA "DNABINFILE|<ALPHABET>]["
			USAGE_COMMA "[from_position_list|slide_window]"
			USAGE_COMMA "window size"
			USAGE_COMMA "[position_list|shift]"
			USAGE_COMMA "skip")
	},
	{
		N_SET_REF_FEAT,
		(&CSGInterface::cmd_set_reference_features),
		USAGE_I(N_SET_REF_FEAT, USAGE_STR "TRAIN|TEST" USAGE_STR)
	},
	{
		N_DEL_LAST_FEATURES,
		(&CSGInterface::cmd_del_last_features),
		USAGE_I(N_DEL_LAST_FEATURES, USAGE_STR "TRAIN|TEST" USAGE_STR )
	},
	{
		N_CONVERT,
		(&CSGInterface::cmd_convert),
		USAGE_I(N_CONVERT, USAGE_STR "TRAIN|TEST" USAGE_STR
				USAGE_COMMA "from_class"
				USAGE_COMMA "from_type"
				USAGE_COMMA "to_class"
				USAGE_COMMA "to_type["
				USAGE_COMMA "order"
				USAGE_COMMA "start"
				USAGE_COMMA "gap"
				USAGE_COMMA "reversed]")
	},
	{
		N_RESHAPE,
		(&CSGInterface::cmd_reshape),
		USAGE_I(N_RESHAPE, USAGE_STR "TRAIN|TEST"
				USAGE_COMMA "num_feat"
				USAGE_COMMA "num_vec")
	},
	{
		N_LOAD_LABELS,
		(&CSGInterface::cmd_load_labels),
		USAGE_I(N_LOAD_LABELS, "filename"
				USAGE_COMMA USAGE_STR "TRAIN|TARGET" USAGE_STR)
	},
	{
		N_SET_LABELS,
		(&CSGInterface::cmd_set_labels),
		USAGE_I(N_SET_LABELS, USAGE_STR "TRAIN|TEST" USAGE_STR
				USAGE_COMMA "labels")
	},
	{
		N_GET_LABELS,
		(&CSGInterface::cmd_get_labels),
		USAGE_IO(N_GET_LABELS, USAGE_STR "TRAIN|TEST" USAGE_STR, "labels")
	},


	{ "Kernel", NULL, NULL },
	{
		N_SET_KERNEL_NORMALIZATION,
		(&CSGInterface::cmd_set_kernel_normalization),
		USAGE_I(N_SET_KERNEL_NORMALIZATION, "IDENTITY|AVGDIAG|SQRTDIAG|FIRSTELEMENT|VARIANCE|ZEROMEANCENTER"
				USAGE_COMMA "size[" USAGE_COMMA "kernel-specific parameters]")
	},
	{
		N_SET_KERNEL,
		(&CSGInterface::cmd_set_kernel),
		USAGE_I(N_SET_KERNEL, "type" USAGE_COMMA "size[" USAGE_COMMA "kernel-specific parameters]")
	},
	{
		N_ADD_KERNEL,
		(&CSGInterface::cmd_add_kernel),
		USAGE_I(N_ADD_KERNEL, "weight" USAGE_COMMA "kernel-specific parameters")
	},
	{
		N_DEL_LAST_KERNEL,
		(&CSGInterface::cmd_del_last_kernel),
		USAGE(N_DEL_LAST_KERNEL)
	},
	{
		N_INIT_KERNEL,
		(&CSGInterface::cmd_init_kernel),
		USAGE_I(N_INIT_KERNEL, USAGE_STR "TRAIN|TEST" USAGE_STR)
	},
	{
		N_CLEAN_KERNEL,
		(&CSGInterface::cmd_clean_kernel),
		USAGE(N_CLEAN_KERNEL)
	},
	{
		N_SAVE_KERNEL,
		(&CSGInterface::cmd_save_kernel),
		USAGE_I(N_SAVE_KERNEL, "filename" USAGE_COMMA USAGE_STR "TRAIN|TEST" USAGE_STR)
	},
	{
		N_GET_KERNEL_MATRIX,
		(&CSGInterface::cmd_get_kernel_matrix),
		USAGE_IO(N_GET_KERNEL_MATRIX, "[" USAGE_STR "TRAIN|TEST" USAGE_STR, "K]")
	},
	{
		N_SET_WD_POS_WEIGHTS,
		(&CSGInterface::cmd_set_WD_position_weights),
		USAGE_I(N_SET_WD_POS_WEIGHTS, "W[" USAGE_COMMA USAGE_STR "TRAIN|TEST" USAGE_STR "]")
	},
	{
		N_GET_SUBKERNEL_WEIGHTS,
		(&CSGInterface::cmd_get_subkernel_weights),
		USAGE_O(N_GET_SUBKERNEL_WEIGHTS, "W")
	},
	{
		N_SET_SUBKERNEL_WEIGHTS,
		(&CSGInterface::cmd_set_subkernel_weights),
		USAGE_I(N_SET_SUBKERNEL_WEIGHTS, "W")
	},
	{
		N_SET_SUBKERNEL_WEIGHTS_COMBINED,
		(&CSGInterface::cmd_set_subkernel_weights_combined),
		USAGE_I(N_SET_SUBKERNEL_WEIGHTS_COMBINED, "W" USAGE_COMMA "idx")
	},
	{
		N_GET_DOTFEATURE_WEIGHTS_COMBINED,
		(&CSGInterface::cmd_get_dotfeature_weights_combined),
		USAGE_IO(N_GET_DOTFEATURE_WEIGHTS_COMBINED,  USAGE_STR "TRAIN|TEST" USAGE_STR, "W")
	},
	{
		N_SET_DOTFEATURE_WEIGHTS_COMBINED,
		(&CSGInterface::cmd_set_dotfeature_weights_combined),
		USAGE_I(N_SET_DOTFEATURE_WEIGHTS_COMBINED, "W" USAGE_COMMA "idx")
	},
	{
		N_SET_LAST_SUBKERNEL_WEIGHTS,
		(&CSGInterface::cmd_set_last_subkernel_weights),
		USAGE_I(N_SET_LAST_SUBKERNEL_WEIGHTS, "W")
	},
	{
		N_GET_WD_POS_WEIGHTS,
		(&CSGInterface::cmd_get_WD_position_weights),
		USAGE_O(N_GET_WD_POS_WEIGHTS, "W")
	},
	{
		N_GET_LAST_SUBKERNEL_WEIGHTS,
		(&CSGInterface::cmd_get_last_subkernel_weights),
		USAGE_O(N_GET_LAST_SUBKERNEL_WEIGHTS, "W")
	},
	{
		N_COMPUTE_BY_SUBKERNELS,
		(&CSGInterface::cmd_compute_by_subkernels),
		USAGE_O(N_COMPUTE_BY_SUBKERNELS, "W")
	},
	{
		N_INIT_KERNEL_OPTIMIZATION,
		(&CSGInterface::cmd_init_kernel_optimization),
		USAGE(N_INIT_KERNEL_OPTIMIZATION)
	},
	{
		N_GET_KERNEL_OPTIMIZATION,
		(&CSGInterface::cmd_get_kernel_optimization),
		USAGE_O(N_GET_KERNEL_OPTIMIZATION, "W")
	},
	{
		N_DELETE_KERNEL_OPTIMIZATION,
		(&CSGInterface::cmd_delete_kernel_optimization),
		USAGE(N_DELETE_KERNEL_OPTIMIZATION)
	},
	{
		N_USE_DIAGONAL_SPEEDUP,
		(&CSGInterface::cmd_use_diagonal_speedup),
		USAGE_I(N_USE_DIAGONAL_SPEEDUP, USAGE_STR "0|1" USAGE_STR)
	},
	{
		N_SET_KERNEL_OPTIMIZATION_TYPE,
		(&CSGInterface::cmd_set_kernel_optimization_type),
		USAGE_I(N_SET_KERNEL_OPTIMIZATION_TYPE, USAGE_STR "FASTBUTMEMHUNGRY|SLOWBUTMEMEFFICIENT" USAGE_STR)
	},
	{
		N_SET_SOLVER,
		(&CSGInterface::cmd_set_solver),
		USAGE_I(N_SET_SOLVER, USAGE_STR "AUTO|CPLEX|GLPK|INTERNAL" USAGE_STR)
	},
	{
		N_SET_CONSTRAINT_GENERATOR,
		(&CSGInterface::cmd_set_constraint_generator),
		USAGE_I(N_SET_CONSTRAINT_GENERATOR, USAGE_STR "LIBSVM_ONECLASS|LIBSVM_MULTICLASS|LIBSVM"
				"|SVMLIGHT|LIGHT|SVMLIGHT_ONECLASS|GPBTSVM|MPDSVM|GNPPSVM|GMNPSVM"
				USAGE_STR)
	},
	{
		N_SET_PRIOR_PROBS,
		(&CSGInterface::cmd_set_prior_probs),
		USAGE_I(N_SET_PRIOR_PROBS, USAGE_STR "pos probs, neg_probs" USAGE_STR)
	},
	{
		N_SET_PRIOR_PROBS_FROM_LABELS,
		(&CSGInterface::cmd_set_prior_probs_from_labels),
		USAGE_I(N_SET_PRIOR_PROBS_FROM_LABELS, USAGE_STR "labels" USAGE_STR)
	},
#ifdef USE_SVMLIGHT
	{
		N_RESIZE_KERNEL_CACHE,
		(&CSGInterface::cmd_resize_kernel_cache),
		USAGE_I(N_RESIZE_KERNEL_CACHE, "size")
	},
#endif //USE_SVMLIGHT


	{ "Distance", NULL, NULL },
	{
		N_SET_DISTANCE,
		(&CSGInterface::cmd_set_distance),
		USAGE_I(N_SET_DISTANCE, "type" USAGE_COMMA "data type[" USAGE_COMMA "distance-specific parameters]")
	},
	{
		N_INIT_DISTANCE,
		(&CSGInterface::cmd_init_distance),
		USAGE_I(N_INIT_DISTANCE, USAGE_STR "TRAIN|TEST" USAGE_STR)
	},
	{
		N_GET_DISTANCE_MATRIX,
		(&CSGInterface::cmd_get_distance_matrix),
		USAGE_O(N_GET_DISTANCE_MATRIX, "D")
	},


	{ "Classifier", NULL, NULL },
	{
		N_CLASSIFY,
		(&CSGInterface::cmd_classify),
		USAGE_O(N_CLASSIFY, "result")
	},
	{
		N_SVM_CLASSIFY,
		(&CSGInterface::cmd_classify),
		USAGE_O(N_SVM_CLASSIFY, "result")
	},
	{
		N_CLASSIFY_EXAMPLE,
		(&CSGInterface::cmd_classify_example),
		USAGE_IO(N_CLASSIFY_EXAMPLE, "feature_vector_index", "result")
	},
	{
		N_SVM_CLASSIFY_EXAMPLE,
		(&CSGInterface::cmd_classify_example),
		USAGE_IO(N_SVM_CLASSIFY_EXAMPLE, "feature_vector_index", "result")
	},
	{
		N_GET_CLASSIFIER,
		(&CSGInterface::cmd_get_classifier),
		USAGE_IO(N_GET_CLASSIFIER, "[index in case of MultiClassSVM]", "bias" USAGE_COMMA "weights")
	},
	{
		N_GET_CLUSTERING,
		(&CSGInterface::cmd_get_classifier),
		USAGE_O(N_GET_CLUSTERING, "radi" USAGE_COMMA "centers|merge_distances" USAGE_COMMA "pairs")
	},
	{
		N_NEW_SVM,
		(&CSGInterface::cmd_new_classifier),
		USAGE_I(N_NEW_SVM, USAGE_STR "LIBSVM_ONECLASS|LIBSVM_MULTICLASS|LIBSVM"
				"|SVMLIGHT|LIGHT|LIGHT_ONECLASS|SVMLIN|GPBTSVM|MPDSVM|GNPPSVM|GMNPSVM"
				"|WDSVMOCAS|SVMOCAS|SVMSGD|SVMBMRM|SVMPERF"
				"|KERNELPERCEPTRON|PERCEPTRON|LIBLINEAR_LR|LIBLINEAR_L2|LDA"
				"|LPM|LPBOOST|KNN" USAGE_STR)
	},
	{
		N_NEW_CLASSIFIER,
		(&CSGInterface::cmd_new_classifier),
		USAGE_I(N_NEW_CLASSIFIER, USAGE_STR "LIBSVM_ONECLASS|LIBSVM_MULTICLASS"
				"|LIBSVM|SVMLIGHT|LIGHT|LIGHT_ONECLASS|SVMLIN|GPBTSVM|MPDSVM|GNPPSVM|GMNPSVM"
				"|WDSVMOCAS|SVMOCAS|SVMSGD|SVMBMRM|SVMPERF"
				"|KERNELPERCEPTRON|PERCEPTRON|LIBLINEAR_LR|LIBLINEAR_L2|LDA"
				"|LPM|LPBOOST|KNN" USAGE_STR)
	},
	{
		N_NEW_REGRESSION,
		(&CSGInterface::cmd_new_classifier),
		USAGE_I(N_NEW_REGRESSION, USAGE_STR "SVRLIGHT|LIBSVR|KRR" USAGE_STR)
	},
	{
		N_NEW_CLUSTERING,
		(&CSGInterface::cmd_new_classifier),
		USAGE_I(N_NEW_CLUSTERING, USAGE_STR "KMEANS|HIERARCHICAL" USAGE_STR)
	},
	{
		N_LOAD_CLASSIFIER,
		(&CSGInterface::cmd_load_classifier),
		USAGE_O(N_LOAD_CLASSIFIER, "filename" USAGE_COMMA "type")
	},
	{
		N_SAVE_CLASSIFIER,
		(&CSGInterface::cmd_save_classifier),
		USAGE_I(N_SAVE_CLASSIFIER, "filename")
	},
	{
		N_GET_NUM_SVMS,
		(&CSGInterface::cmd_get_num_svms),
		USAGE_O(N_GET_NUM_SVMS, "number of SVMs in MultiClassSVM")
	},
	{
		N_GET_SVM,
		(&CSGInterface::cmd_get_svm),
		USAGE_IO(N_GET_SVM, "[index in case of MultiClassSVM]", "bias" USAGE_COMMA "alphas")
	},
	{
		N_SET_SVM,
		(&CSGInterface::cmd_set_svm),
		USAGE_I(N_SET_SVM, "bias" USAGE_COMMA "alphas")
	},
	{
		N_SET_LINEAR_CLASSIFIER,
		(&CSGInterface::cmd_set_linear_classifier),
		USAGE_I(N_SET_LINEAR_CLASSIFIER, "bias" USAGE_COMMA "w")
	},
	{
		N_GET_SVM_OBJECTIVE,
		(&CSGInterface::cmd_get_svm_objective),
		USAGE_O(N_GET_SVM_OBJECTIVE, "objective")
	},
	{
		N_COMPUTE_SVM_PRIMAL_OBJECTIVE,
		(&CSGInterface::cmd_compute_svm_primal_objective),
		USAGE_O(N_COMPUTE_SVM_PRIMAL_OBJECTIVE, "objective")
	},
	{
		N_COMPUTE_SVM_DUAL_OBJECTIVE,
		(&CSGInterface::cmd_compute_svm_dual_objective),
		USAGE_O(N_COMPUTE_SVM_DUAL_OBJECTIVE, "objective")
	},
	{
		N_COMPUTE_MKL_PRIMAL_OBJECTIVE,
		(&CSGInterface::cmd_compute_svm_primal_objective),
		USAGE_O(N_COMPUTE_MKL_PRIMAL_OBJECTIVE, "objective")
	},
	{
		N_COMPUTE_MKL_DUAL_OBJECTIVE,
		(&CSGInterface::cmd_compute_mkl_dual_objective),
		USAGE_O(N_COMPUTE_MKL_DUAL_OBJECTIVE, "objective")
	},
	{
		N_COMPUTE_RELATIVE_MKL_DUALITY_GAP,
		(&CSGInterface::cmd_compute_relative_mkl_duality_gap),
		USAGE_O(N_COMPUTE_RELATIVE_MKL_DUALITY_GAP, "gap")
	},
	{
		N_COMPUTE_ABSOLUTE_MKL_DUALITY_GAP,
		(&CSGInterface::cmd_compute_absolute_mkl_duality_gap),
		USAGE_O(N_COMPUTE_ABSOLUTE_MKL_DUALITY_GAP, "gap")
	},
	{
		N_DO_AUC_MAXIMIZATION,
		(&CSGInterface::cmd_do_auc_maximization),
		USAGE_I(N_DO_AUC_MAXIMIZATION, USAGE_STR "auc" USAGE_STR)
	},
	{
		N_SET_PERCEPTRON_PARAMETERS,
		(&CSGInterface::cmd_set_perceptron_parameters),
		USAGE_I(N_SET_PERCEPTRON_PARAMETERS, "learnrate" USAGE_COMMA "maxiter")
	},
	{
		N_TRAIN_CLASSIFIER,
		(&CSGInterface::cmd_train_classifier),
		USAGE_I(N_TRAIN_CLASSIFIER, "[classifier-specific parameters]")
	},
	{
		N_TRAIN_REGRESSION,
		(&CSGInterface::cmd_train_classifier),
		USAGE(N_TRAIN_REGRESSION)
	},
	{
		N_TRAIN_CLUSTERING,
		(&CSGInterface::cmd_train_classifier),
		USAGE(N_TRAIN_CLUSTERING)
	},
	{
		N_SVM_TRAIN,
		(&CSGInterface::cmd_train_classifier),
		USAGE_I(N_SVM_TRAIN, "[classifier-specific parameters]")
	},
	{
		N_SVMQPSIZE,
		(&CSGInterface::cmd_set_svm_qpsize),
		USAGE_I(N_SVMQPSIZE, "size")
	},
	{
		N_SVMMAXQPSIZE,
		(&CSGInterface::cmd_set_svm_max_qpsize),
		USAGE_I(N_SVMMAXQPSIZE, "size")
	},
	{
		N_SVMBUFSIZE,
		(&CSGInterface::cmd_set_svm_bufsize),
		USAGE_I(N_SVMBUFSIZE, "size")
	},
	{
		N_C,
		(&CSGInterface::cmd_set_svm_C),
		USAGE_I(N_C, "C1[" USAGE_COMMA "C2]")
	},
	{
		N_SVM_EPSILON,
		(&CSGInterface::cmd_set_svm_epsilon),
		USAGE_I(N_SVM_EPSILON, "epsilon")
	},
	{
		N_SVR_TUBE_EPSILON,
		(&CSGInterface::cmd_set_svr_tube_epsilon),
		USAGE_I(N_SVR_TUBE_EPSILON, "tube_epsilon")
	},
	{
		N_SVM_NU,
		(&CSGInterface::cmd_set_svm_nu),
		USAGE_I(N_SVM_NU, "nu")
	},
	{
		N_MKL_PARAMETERS,
		(&CSGInterface::cmd_set_svm_mkl_parameters),
		USAGE_I(N_MKL_PARAMETERS, "weight_epsilon" USAGE_COMMA "C_MKL [" USAGE_COMMA "mkl_norm ]")
	},
	{
	  N_ENT_LAMBDA,
	  (&CSGInterface::cmd_set_elasticnet_lambda),
	  USAGE_I(N_ENT_LAMBDA, "ent_lambda")
	},
	{
	  N_MKL_BLOCK_NORM,
	  (&CSGInterface::cmd_set_mkl_block_norm),
	  USAGE_I(N_MKL_BLOCK_NORM, "mkl_block_norm")
	},
	{
		N_SVM_MAX_TRAIN_TIME,
		(&CSGInterface::cmd_set_max_train_time),
		USAGE_I(N_SVM_MAX_TRAIN_TIME, "max_train_time")
	},
	{
		N_USE_SHRINKING,
		(&CSGInterface::cmd_set_svm_shrinking_enabled),
		USAGE_I(N_USE_SHRINKING, "enable_shrinking")
	},
	{
		N_USE_BATCH_COMPUTATION,
		(&CSGInterface::cmd_set_svm_batch_computation_enabled),
		USAGE_I(N_USE_BATCH_COMPUTATION, "enable_batch_computation")
	},
	{
		N_USE_LINADD,
		(&CSGInterface::cmd_set_svm_linadd_enabled),
		USAGE_I(N_USE_LINADD, "enable_linadd")
	},
	{
		N_SVM_USE_BIAS,
		(&CSGInterface::cmd_set_svm_bias_enabled),
		USAGE_I(N_SVM_USE_BIAS, "enable_bias")
	},
	{
		N_MKL_USE_INTERLEAVED_OPTIMIZATION,
		(&CSGInterface::cmd_set_mkl_interleaved_enabled),
		USAGE_I(N_MKL_USE_INTERLEAVED_OPTIMIZATION, "enable_interleaved_optimization")
	},
	{
		N_KRR_TAU,
		(&CSGInterface::cmd_set_krr_tau),
		USAGE_I(N_KRR_TAU, "tau")
	},


	{ "Preprocessors", NULL, NULL },
	{
		N_ADD_PREPROC,
		(&CSGInterface::cmd_add_preproc),
		USAGE_I(N_ADD_PREPROC, "preproc[, preproc-specific parameters]")
	},
	{
		N_DEL_PREPROC,
		(&CSGInterface::cmd_del_preproc),
		USAGE(N_DEL_PREPROC)
	},
	{
		N_ATTACH_PREPROC,
		(&CSGInterface::cmd_attach_preproc),
		USAGE_I(N_ATTACH_PREPROC, USAGE_STR "TRAIN|TEST" USAGE_STR USAGE_COMMA "force")
	},
	{
		N_CLEAN_PREPROC,
		(&CSGInterface::cmd_clean_preproc),
		USAGE(N_CLEAN_PREPROC)
	},

	{ "Converters", NULL, NULL },
	{
		N_SET_CONVERTER,
		(&CSGInterface::cmd_set_converter),
		USAGE(N_SET_CONVERTER)
	},
	{
		N_APPLY_CONVERTER,
		(&CSGInterface::cmd_apply_converter),
		USAGE_O(N_APPLY_CONVERTER, "conv_features")
	},
	{
		N_EMBED,
		(&CSGInterface::cmd_embed),
		USAGE_IO(N_EMBED,"target dim","embedding")
	},


	{ "HMM", NULL, NULL },
	{
		N_NEW_HMM,
		(&CSGInterface::cmd_new_hmm),
		USAGE_I(N_NEW_HMM, "N" USAGE_COMMA "M")
	},
	{
		N_LOAD_HMM,
		(&CSGInterface::cmd_load_hmm),
		USAGE_I(N_LOAD_HMM, "filename")
	},
	{
		N_SAVE_HMM,
		(&CSGInterface::cmd_save_hmm),
		USAGE_I(N_SAVE_HMM, "filename[" USAGE_COMMA "save_binary]")
	},
	{
		N_GET_HMM,
		(&CSGInterface::cmd_get_hmm),
		USAGE_O(N_GET_HMM, "p" USAGE_COMMA "q" USAGE_COMMA "a" USAGE_COMMA "b")
	},
	{
		N_APPEND_HMM,
		(&CSGInterface::cmd_append_hmm),
		USAGE_I(N_APPEND_HMM, "p" USAGE_COMMA "q" USAGE_COMMA "a" USAGE_COMMA "b")
	},
	{
		N_APPEND_MODEL,
		(&CSGInterface::cmd_append_model),
		USAGE_I(N_APPEND_MODEL, USAGE_STR "filename" USAGE_STR "[" USAGE_COMMA "base1" USAGE_COMMA "base2]")
	},
	{
		N_SET_HMM,
		(&CSGInterface::cmd_set_hmm),
		USAGE_I(N_SET_HMM, "p" USAGE_COMMA "q" USAGE_COMMA "a" USAGE_COMMA "b")
	},
	{
		N_SET_HMM_AS,
		(&CSGInterface::cmd_set_hmm_as),
		USAGE_I(N_SET_HMM_AS, "POS|NEG|TEST")
	},
	{
		N_CHOP,
		(&CSGInterface::cmd_set_chop),
		USAGE_I(N_CHOP, "chop")
	},
	{
		N_PSEUDO,
		(&CSGInterface::cmd_set_pseudo),
		USAGE_I(N_PSEUDO, "pseudo")
	},
	{
		N_LOAD_DEFINITIONS,
		(&CSGInterface::cmd_load_definitions),
		USAGE_I(N_LOAD_DEFINITIONS, "filename" USAGE_COMMA "init")
	},
	{
		N_HMM_CLASSIFY,
		(&CSGInterface::cmd_hmm_classify),
		USAGE_O(N_HMM_CLASSIFY, "result")
	},
	{
		N_ONE_CLASS_LINEAR_HMM_CLASSIFY,
		(&CSGInterface::cmd_one_class_linear_hmm_classify),
		USAGE_O(N_ONE_CLASS_LINEAR_HMM_CLASSIFY, "result")
	},
	{
		N_ONE_CLASS_HMM_CLASSIFY,
		(&CSGInterface::cmd_one_class_hmm_classify),
		USAGE_O(N_ONE_CLASS_HMM_CLASSIFY, "result")
	},
	{
		N_ONE_CLASS_HMM_CLASSIFY_EXAMPLE,
		(&CSGInterface::cmd_one_class_hmm_classify_example),
		USAGE_IO(N_ONE_CLASS_HMM_CLASSIFY_EXAMPLE, "feature_vector_index", "result")
	},
	{
		N_HMM_CLASSIFY_EXAMPLE,
		(&CSGInterface::cmd_hmm_classify_example),
		USAGE_IO(N_HMM_CLASSIFY_EXAMPLE, "feature_vector_index", "result")
	},
	{
		N_OUTPUT_HMM,
		(&CSGInterface::cmd_output_hmm),
		USAGE(N_OUTPUT_HMM)
	},
	{
		N_OUTPUT_HMM_DEFINED,
		(&CSGInterface::cmd_output_hmm_defined),
		USAGE(N_OUTPUT_HMM_DEFINED)
	},
	{
		N_HMM_LIKELIHOOD,
		(&CSGInterface::cmd_hmm_likelihood),
		USAGE_O(N_HMM_LIKELIHOOD, "likelihood")
	},
	{
		N_LIKELIHOOD,
		(&CSGInterface::cmd_likelihood),
		USAGE(N_LIKELIHOOD)
	},
	{
		N_SAVE_LIKELIHOOD,
		(&CSGInterface::cmd_save_likelihood),
		USAGE_I(N_SAVE_LIKELIHOOD, "filename[" USAGE_COMMA "save_binary]")
	},
	{
		N_GET_VITERBI_PATH,
		(&CSGInterface::cmd_get_viterbi_path),
		USAGE_IO(N_GET_VITERBI_PATH, "dim", "path" USAGE_COMMA "likelihood")
	},
	{
		N_VITERBI_TRAIN_DEFINED,
		(&CSGInterface::cmd_viterbi_train_defined),
		USAGE(N_VITERBI_TRAIN_DEFINED)
	},
	{
		N_VITERBI_TRAIN,
		(&CSGInterface::cmd_viterbi_train),
		USAGE(N_VITERBI_TRAIN)
	},
	{
		N_BAUM_WELCH_TRAIN,
		(&CSGInterface::cmd_baum_welch_train),
		USAGE(N_BAUM_WELCH_TRAIN)
	},
	{
		N_BAUM_WELCH_TRAIN_DEFINED,
		(&CSGInterface::cmd_baum_welch_train_defined),
		USAGE(N_BAUM_WELCH_TRAIN_DEFINED)
	},
	{
		N_BAUM_WELCH_TRANS_TRAIN,
		(&CSGInterface::cmd_baum_welch_trans_train),
		USAGE(N_BAUM_WELCH_TRANS_TRAIN)
	},
	{
		N_LINEAR_TRAIN,
		(&CSGInterface::cmd_linear_train),
		USAGE(N_LINEAR_TRAIN)
	},
	{
		N_SAVE_PATH,
		(&CSGInterface::cmd_save_path),
		USAGE_I(N_SAVE_PATH, "filename[" USAGE_COMMA "save_binary]")
	},
	{
		N_CONVERGENCE_CRITERIA,
		(&CSGInterface::cmd_convergence_criteria),
		USAGE_I(N_CONVERGENCE_CRITERIA, "num_iterations" USAGE_COMMA "epsilon")
	},
	{
		N_NORMALIZE,
		(&CSGInterface::cmd_normalize),
		USAGE_I(N_NORMALIZE, "[keep_dead_states]")
	},
	{
		N_ADD_STATES,
		(&CSGInterface::cmd_add_states),
		USAGE_I(N_ADD_STATES, "states" USAGE_COMMA "value")
	},
	{
		N_PERMUTATION_ENTROPY,
		(&CSGInterface::cmd_permutation_entropy),
		USAGE_I(N_PERMUTATION_ENTROPY, "width" USAGE_COMMA "seqnum")
	},
	{
		N_RELATIVE_ENTROPY,
		(&CSGInterface::cmd_relative_entropy),
		USAGE_O(N_RELATIVE_ENTROPY, "result")
	},
	{
		N_ENTROPY,
		(&CSGInterface::cmd_entropy),
		USAGE_O(N_ENTROPY, "result")
	},
	{
		(char*) N_SET_FEATURE_MATRIX,
		(&CSGInterface::cmd_set_feature_matrix),
		(char*) USAGE_I(N_SET_FEATURE_MATRIX, "features")
	},
	{
		(char*) N_SET_FEATURE_MATRIX_SPARSE,
		(&CSGInterface::cmd_set_feature_matrix_sparse),
		(char*) USAGE_I(N_SET_FEATURE_MATRIX_SPARSE, "sp1" USAGE_COMMA "sp2" )
	},
	{
		N_NEW_PLUGIN_ESTIMATOR,
		(&CSGInterface::cmd_new_plugin_estimator),
		USAGE_I(N_NEW_PLUGIN_ESTIMATOR, "pos_pseudo" USAGE_COMMA "neg_pseudo")
	},
	{
		N_TRAIN_ESTIMATOR,
		(&CSGInterface::cmd_train_estimator),
		USAGE(N_TRAIN_ESTIMATOR)
	},
	{
		N_PLUGIN_ESTIMATE_CLASSIFY_EXAMPLE,
		(&CSGInterface::cmd_plugin_estimate_classify_example),
		USAGE_IO(N_PLUGIN_ESTIMATE_CLASSIFY_EXAMPLE, "feature_vector_index", "result")
	},
	{
		N_PLUGIN_ESTIMATE_CLASSIFY,
		(&CSGInterface::cmd_plugin_estimate_classify),
		USAGE_O(N_PLUGIN_ESTIMATE_CLASSIFY, "result")
	},
	{
		N_SET_PLUGIN_ESTIMATE,
		(&CSGInterface::cmd_set_plugin_estimate),
		USAGE_I(N_SET_PLUGIN_ESTIMATE, "emission_probs" USAGE_COMMA "model_sizes")
	},
	{
		N_GET_PLUGIN_ESTIMATE,
		(&CSGInterface::cmd_get_plugin_estimate),
		USAGE_O(N_GET_PLUGIN_ESTIMATE, "emission_probs" USAGE_COMMA "model_sizes")
	},
	{ "Signals", NULL, NULL },
	{
		N_SIGNALS_SET_MODEL,
		(&CSGInterface::cmd_signals_set_model),
		USAGE_I(N_SIGNALS_SET_MODEL, "arg1")
	},
	{
		N_SIGNALS_SET_POSITIONS,
		(&CSGInterface::cmd_signals_set_positions),
		USAGE_I(N_SIGNALS_SET_POSITIONS, "positions")
	},
	{
		N_SIGNALS_SET_LABELS,
		(&CSGInterface::cmd_signals_set_labels),
		USAGE_I(N_SIGNALS_SET_LABELS, "labels")
	},
	{
		N_SIGNALS_SET_SPLIT,
		(&CSGInterface::cmd_signals_set_split),
		USAGE_I(N_SIGNALS_SET_SPLIT, "split")
	},
	{
		N_SIGNALS_SET_TRAIN_MASK,
		(&CSGInterface::cmd_signals_set_train_mask),
		USAGE_I(N_SIGNALS_SET_TRAIN_MASK, "")
	},
	{
		N_SIGNALS_ADD_FEATURE,
		(&CSGInterface::cmd_signals_add_feature),
		USAGE_I(N_SIGNALS_ADD_FEATURE, "feature")
	},
	{
		N_SIGNALS_ADD_KERNEL,
		(&CSGInterface::cmd_signals_add_kernel),
		USAGE_I(N_SIGNALS_ADD_KERNEL, "kernelparam")
	},
	{
		N_SIGNALS_RUN,
		(&CSGInterface::cmd_signals_run),
		USAGE_I(N_SIGNALS_RUN, "arg1")
	},
	{ "Structure", NULL, NULL },
	{
		N_BEST_PATH,
		(&CSGInterface::cmd_best_path),
		USAGE_I(N_BEST_PATH, "from" USAGE_COMMA "to")
	},
	{
		N_BEST_PATH_2STRUCT,
		(&CSGInterface::cmd_best_path_2struct),
		USAGE_IO(N_BEST_PATH_2STRUCT, "p"
				USAGE_COMMA "q"
				USAGE_COMMA "cmd_trans"
				USAGE_COMMA "seq"
				USAGE_COMMA "pos"
				USAGE_COMMA "genestr"
				USAGE_COMMA "penalties"
				USAGE_COMMA "penalty_info"
				USAGE_COMMA "nbest"
				USAGE_COMMA "content_weights"
				USAGE_COMMA "segment_sum_weights",
			"prob" USAGE_COMMA "path" USAGE_COMMA "pos")
	},
	{
		(char*) N_SET_PLIF_STRUCT,
		(&CSGInterface::cmd_set_plif_struct),
		(char*) USAGE_I(N_SET_PLIF_STRUCT, "id"
				USAGE_COMMA "name"
				USAGE_COMMA "limits"
				USAGE_COMMA "penalties"
				USAGE_COMMA "transform"
				USAGE_COMMA "min_value"
				USAGE_COMMA "max_value"
				USAGE_COMMA "use_cache"
				USAGE_COMMA "use_svm")
	},
	{
		(char*) N_GET_PLIF_STRUCT,
		(&CSGInterface::cmd_get_plif_struct),
		(char*) USAGE_O(N_GET_PLIF_STRUCT, "id"
				USAGE_COMMA "name"
				USAGE_COMMA "limits"
				USAGE_COMMA "penalties"
				USAGE_COMMA "transform"
				USAGE_COMMA "min_value"
				USAGE_COMMA "max_value"
				USAGE_COMMA "use_cache"
				USAGE_COMMA "use_svm")
	},
	{
		(char*) N_PRECOMPUTE_SUBKERNELS,
		(&CSGInterface::cmd_precompute_subkernels),
		(char*) USAGE(N_PRECOMPUTE_SUBKERNELS)
	},
	{
		(char*) N_PRECOMPUTE_CONTENT_SVMS,
		(&CSGInterface::cmd_precompute_content_svms),
		(char*) USAGE_I(N_PRECOMPUTE_CONTENT_SVMS, "sequence"
				USAGE_COMMA "position_list"
				USAGE_COMMA "weights")
	},
	{
		(char*) N_GET_LIN_FEAT,
		(&CSGInterface::cmd_get_lin_feat),
		(char*) USAGE_O(N_GET_LIN_FEAT, "lin_feat")
	},
	{
		(char*) N_SET_LIN_FEAT,
		(&CSGInterface::cmd_set_lin_feat),
		(char*) USAGE_I(N_SET_LIN_FEAT, "lin_feat")
	},
	{
		(char*) N_INIT_DYN_PROG,
		(&CSGInterface::cmd_init_dyn_prog),
		(char*) USAGE_I(N_INIT_DYN_PROG, "num_svms")
	},
	{
		(char*) N_CLEAN_UP_DYN_PROG,
		(&CSGInterface::cmd_clean_up_dyn_prog),
		(char*) USAGE(N_CLEAN_UP_DYN_PROG)
	},
	{
		(char*) N_INIT_INTRON_LIST,
		(&CSGInterface::cmd_init_intron_list),
		(char*) USAGE_I(N_INIT_INTRON_LIST, "start_positions"
				USAGE_COMMA "end_positions"
				USAGE_COMMA "quality")
	},
	{
		(char*) N_PRECOMPUTE_TILING_FEATURES,
		(&CSGInterface::cmd_precompute_tiling_features),
		(char*) USAGE_I(N_PRECOMPUTE_TILING_FEATURES, "intensities"
				USAGE_COMMA "probe_pos"
				USAGE_COMMA "tiling_plif_ids")
	},
	{
		(char*) N_LONG_TRANSITION_SETTINGS,
		(&CSGInterface::cmd_long_transition_settings),
		(char*) USAGE_I(N_LONG_TRANSITION_SETTINGS, "use_long_transitions"
				USAGE_COMMA "threshold"
				USAGE_COMMA "max_len")
	},

	{
		(char*) N_SET_MODEL,
		(&CSGInterface::cmd_set_model),
		(char*) USAGE_I(N_SET_MODEL, "content_weights"
				USAGE_COMMA "transition_pointers"
				USAGE_COMMA "use_orf"
				USAGE_COMMA "mod_words")
	},

	{
		(char*) N_BEST_PATH_TRANS,
		(&CSGInterface::cmd_best_path_trans),
		USAGE_IO(N_BEST_PATH_TRANS, "p"
				USAGE_COMMA "q"
				USAGE_COMMA "nbest"
				USAGE_COMMA "seq_path"
				USAGE_COMMA "a_trans"
				USAGE_COMMA "segment_loss",
			"prob" USAGE_COMMA "path" USAGE_COMMA "pos")
	},
	{
		N_BEST_PATH_TRANS_DERIV,
		(&CSGInterface::cmd_best_path_trans_deriv),
		USAGE_IO(N_BEST_PATH_TRANS_DERIV,
			USAGE_COMMA "my_path"
			USAGE_COMMA "my_pos"
			USAGE_COMMA "p"
			USAGE_COMMA "q"
			USAGE_COMMA "cmd_trans"
			USAGE_COMMA "seq"
			USAGE_COMMA "pos"
			USAGE_COMMA "genestr"
			USAGE_COMMA "penalties"
			USAGE_COMMA "state_signals"
			USAGE_COMMA "penalty_info"
			USAGE_COMMA "dict_weights"
			USAGE_COMMA "mod_words ["
			USAGE_COMMA "segment_loss"
			USAGE_COMMA "segmend_ids_mask]", "p_deriv"
			USAGE_COMMA "q_deriv"
			USAGE_COMMA "cmd_deriv"
			USAGE_COMMA "penalties_deriv"
			USAGE_COMMA "my_scores"
			USAGE_COMMA "my_loss")
	},

	{ "POIM", NULL, NULL },
	{
		N_COMPUTE_POIM_WD,
		(&CSGInterface::cmd_compute_POIM_WD),
		USAGE_IO(N_COMPUTE_POIM_WD, "max_order" USAGE_COMMA "distribution", "W")
	},
	{
		N_GET_SPEC_CONSENSUS,
		(&CSGInterface::cmd_get_SPEC_consensus),
		USAGE_O(N_GET_SPEC_CONSENSUS, "W")
	},
	{
		N_GET_SPEC_SCORING,
		(&CSGInterface::cmd_get_SPEC_scoring),
		USAGE_IO(N_GET_SPEC_SCORING, "max_order", "W")
	},
	{
		N_GET_WD_CONSENSUS,
		(&CSGInterface::cmd_get_WD_consensus),
		USAGE_O(N_GET_WD_CONSENSUS, "W")
	},
	{
		N_GET_WD_SCORING,
		(&CSGInterface::cmd_get_WD_scoring),
		USAGE_IO(N_GET_WD_SCORING, "max_order", "W")
	},


	{ "Utility", NULL, NULL },
	{
		N_CRC,
		(&CSGInterface::cmd_crc),
		USAGE_IO(N_CRC, "string", "crc32")
	},
	{
		N_SYSTEM,
		(&CSGInterface::cmd_system),
		USAGE_I(N_SYSTEM, "system_command")
	},
	{
		N_EXIT,
		(&CSGInterface::cmd_exit),
		USAGE(N_EXIT)
	},
	{
		N_QUIT,
		(&CSGInterface::cmd_exit),
		USAGE(N_QUIT)
	},
	{
		N_EXEC,
		(&CSGInterface::cmd_exec),
		USAGE_I(N_EXEC, "filename")
	},
	{
		N_SET_OUTPUT,
		(&CSGInterface::cmd_set_output),
		USAGE_I(N_SET_OUTPUT, USAGE_STR "STDERR|STDOUT|filename" USAGE_STR)
	},
	{
		N_SET_THRESHOLD,
		(&CSGInterface::cmd_set_threshold),
		USAGE_I(N_SET_THRESHOLD, "threshold")
	},
	{
		N_INIT_RANDOM,
		(&CSGInterface::cmd_init_random),
		USAGE_I(N_INIT_RANDOM, "value_to_initialize_RNG_with")
	},
	{
		N_THREADS,
		(&CSGInterface::cmd_set_num_threads),
		USAGE_I(N_THREADS, "num_threads")
	},
	{
		N_TRANSLATE_STRING,
		(&CSGInterface::cmd_translate_string),
		USAGE_IO(N_TRANSLATE_STRING,
			"string, order, start", "translation")
	},
	{
		N_CLEAR,
		(&CSGInterface::cmd_clear),
		USAGE(N_CLEAR)
	},
	{
		N_TIC,
		(&CSGInterface::cmd_tic),
		USAGE(N_TIC)
	},
	{
		N_TOC,
		(&CSGInterface::cmd_toc),
		USAGE(N_TOC)
	},
	{
		N_PRINT,
		(&CSGInterface::cmd_print),
		USAGE_I(N_PRINT, "msg")
	},
	{
		N_ECHO,
		(&CSGInterface::cmd_echo),
		USAGE_I(N_ECHO, "level")
	},
	{
		N_LOGLEVEL,
		(&CSGInterface::cmd_loglevel),
		USAGE_I(N_LOGLEVEL, USAGE_STR "ALL|DEBUG|INFO|NOTICE|WARN|ERROR|CRITICAL|ALERT|EMERGENCY" USAGE_STR)
	},
	{
		N_SYNTAX_HIGHLIGHT,
		(&CSGInterface::cmd_syntax_highlight),
		USAGE_I(N_SYNTAX_HIGHLIGHT, USAGE_STR "ON|OFF" USAGE_STR)
	},
	{
		N_PROGRESS,
		(&CSGInterface::cmd_progress),
		USAGE_I(N_PROGRESS, USAGE_STR "ON|OFF" USAGE_STR)
	},
	{
		N_GET_VERSION,
		(&CSGInterface::cmd_get_version),
		USAGE_O(N_GET_VERSION, "version")
	},
	{
		N_HELP,
		(&CSGInterface::cmd_help),
		USAGE(N_HELP)
	},
	{
		N_WHOS,
		(&CSGInterface::cmd_whos),
		USAGE(N_WHOS)
	},
	{
		N_SEND_COMMAND,
		(&CSGInterface::cmd_send_command),
		NULL
	},
	{
		N_RUN_PYTHON,
		(&CSGInterface::cmd_run_python),
		USAGE_IO(N_RUN_PYTHON,
			"'Var1', Var1, 'Var2', Var2,..., python_function", "results")
	},
	{
		N_RUN_OCTAVE,
		(&CSGInterface::cmd_run_octave),
		USAGE_IO(N_RUN_OCTAVE,
			"'Var1', Var1, 'Var2', Var2,..., octave_function", "results")
	},
	{
		N_RUN_R,
		(&CSGInterface::cmd_run_r),
		USAGE_IO(N_RUN_R,
			"'Var1', Var1, 'Var2', Var2,..., r_function", "results")
	},
	{NULL, NULL, NULL}        /* Sentinel */
};


CSGInterface::CSGInterface(bool print_copyright)
: CSGObject(),
	ui_classifier(new CGUIClassifier(this)),
	ui_distance(new CGUIDistance(this)),
	ui_features(new CGUIFeatures(this)),
	ui_hmm(new CGUIHMM(this)),
	ui_kernel(new CGUIKernel(this)),
	ui_labels(new CGUILabels(this)),
	ui_math(new CGUIMath(this)),
	ui_pluginestimate(new CGUIPluginEstimate(this)),
	ui_preproc(new CGUIPreprocessor(this)),
	ui_time(new CGUITime(this)),
	ui_structure(new CGUIStructure(this)),
	ui_converter(new CGUIConverter(this))/*,
/	ui_signals(new CGUISignals(this))*/
{
	if (print_copyright)
	{
		version->print_version();
		SG_PRINT("( seeding random number generator with %u (seed size %d))\n",
				CMath::get_seed(), RNG_SEED_SIZE);
#ifdef USE_LOGCACHE
		SG_PRINT("initializing log-table (size=%i*%i*%i=%2.1fMB) ... ) ",
				CMath::get_log_range(),CMath::get_log_accuracy(),sizeof(float64_t),
				CMath::get_log_range()*CMath::get_log_accuracy()*sizeof(float64_t)/(1024.0*1024.0));
#else
		SG_PRINT("determined range for x in log(1+exp(-x)) is:%d )\n", CMath::get_log_range())
#endif
	}

	reset();
}

CSGInterface::~CSGInterface()
{
	delete ui_classifier;
	delete ui_hmm;
	delete ui_pluginestimate;
	delete ui_kernel;
	delete ui_preproc;
	delete ui_features;
	delete ui_labels;
	delete ui_math;
	delete ui_structure;
	//delete ui_signals;
	delete ui_time;
	delete ui_distance;
	delete ui_converter;

	if (file_out)
		fclose(file_out);
}

void CSGInterface::reset()
{
	m_lhs_counter=0;
	m_rhs_counter=0;
	m_nlhs=0;
	m_nrhs=0;
	m_legacy_strptr=NULL;
	file_out=NULL;
	echo=true;
}

void CSGInterface::translate_arg(CSGInterface* source, CSGInterface* target)
{
	switch (source->get_argument_type())
	{
		case SCALAR_INT:
			target->set_int(source->get_int());
			break;
		case SCALAR_REAL:
			target->set_real(source->get_real());
			break;
		case SCALAR_BOOL:
			target->set_bool(source->get_bool());
			break;
		case VECTOR_BOOL:
			{
				bool* v=NULL;
				int32_t len=0;
				source->get_vector(v, len);
				target->set_vector(v, len);
				SG_FREE(v);
				break;
			}
		case VECTOR_BYTE:
			{
				uint8_t* v=NULL;
				int32_t len=0;
				source->get_vector(v, len);
				target->set_vector(v, len);
				SG_FREE(v);
				break;
			}
		case VECTOR_CHAR:
			{
				char* v=NULL;
				int32_t len=0;
				source->get_vector(v, len);
				target->set_vector(v, len);
				SG_FREE(v);
				break;
			}
		case VECTOR_INT:
			{
				int32_t* v=NULL;
				int32_t len=0;
				source->get_vector(v, len);
				target->set_vector(v, len);
				SG_FREE(v);
				break;
			}
		case VECTOR_REAL:
			{
				float64_t* v=NULL;
				int32_t len=0;
				source->get_vector(v, len);
				target->set_vector(v, len);
				SG_FREE(v);
				break;
			}
		case VECTOR_SHORTREAL:
			{
				float32_t* v=NULL;
				int32_t len=0;
				source->get_vector(v, len);
				target->set_vector(v, len);
				SG_FREE(v);
				break;
			}
		case VECTOR_SHORT:
			{
				int16_t* v=NULL;
				int32_t len=0;
				source->get_vector(v, len);
				target->set_vector(v, len);
				SG_FREE(v);
				break;
			}
		case VECTOR_WORD:
			{
				uint16_t* v=NULL;
				int32_t len=0;
				source->get_vector(v, len);
				target->set_vector(v, len);
				SG_FREE(v);
				break;
			}

		case STRING_BYTE:
			{
				int32_t num_str=0;
				int32_t max_str_len=0;
				SGString<uint8_t>* strs=NULL;
				source->get_string_list(strs, num_str, max_str_len);
				target->set_string_list(strs, num_str);
				SG_FREE(strs);
				break;
			}
		case STRING_CHAR:
			{
				int32_t num_str=0;
				int32_t max_str_len=0;
				SGString<char>* strs;
				source->get_string_list(strs, num_str,max_str_len);
				target->set_string_list(strs, num_str);
				SG_FREE(strs);
				break;
			}
		case STRING_INT:
			{
				int32_t num_str=0;
				int32_t max_str_len=0;
				SGString<int32_t>* strs;
				source->get_string_list(strs, num_str,max_str_len);
				target->set_string_list(strs, num_str);
				SG_FREE(strs);
				break;
			}
		case STRING_SHORT:
			{
				int32_t num_str=0;
				int32_t max_str_len=0;
				SGString<int16_t>* strs=NULL;
				source->get_string_list(strs, num_str, max_str_len);
				target->set_string_list(strs, num_str);
				SG_FREE(strs);
				break;
			}
		case STRING_WORD:
			{
				int32_t num_str=0;
				int32_t max_str_len=0;
				SGString<uint16_t>* strs=NULL;
				source->get_string_list(strs, num_str, max_str_len);
				target->set_string_list(strs, num_str);
				SG_FREE(strs);
				break;
			}
		case DENSE_INT:
			{
				int32_t num_feat=0;
				int32_t num_vec=0;
				int32_t* fmatrix=NULL;
				source->get_matrix(fmatrix, num_feat, num_vec);
				target->set_matrix(fmatrix, num_feat, num_vec);
				SG_FREE(fmatrix);
				break;
			}
		case DENSE_REAL:
			{
				int32_t num_feat=0;
				int32_t num_vec=0;
				float64_t* fmatrix=NULL;
				source->get_matrix(fmatrix, num_feat, num_vec);
				target->set_matrix(fmatrix, num_feat, num_vec);
				SG_FREE(fmatrix);
				break;
			}
		case DENSE_SHORT:
			{
				int32_t num_feat=0;
				int32_t num_vec=0;
				int16_t* fmatrix=NULL;
				source->get_matrix(fmatrix, num_feat, num_vec);
				target->set_matrix(fmatrix, num_feat, num_vec);
				SG_FREE(fmatrix);
				break;
			}
		case DENSE_SHORTREAL:
			{
				int32_t num_feat=0;
				int32_t num_vec=0;
				float32_t* fmatrix=NULL;
				source->get_matrix(fmatrix, num_feat, num_vec);
				target->set_matrix(fmatrix, num_feat, num_vec);
				SG_FREE(fmatrix);
				break;
			}
		case DENSE_WORD:
			{
				int32_t num_feat=0;
				int32_t num_vec=0;
				uint16_t* fmatrix=NULL;
				source->get_matrix(fmatrix, num_feat, num_vec);
				target->set_matrix(fmatrix, num_feat, num_vec);
				SG_FREE(fmatrix);
				break;
			}
			/*
		case NDARRAY_BYTE:
			{
				uint8_t* a=NULL;
				int32_t* dims=NULL;
				int32_t num_dims=0;
				source->get_ndarray(a, dims, num_dims);
				target->set_ndarray(a, dims, num_dims);
				SG_FREE(a);
				SG_FREE(dims);
				break;
			}
		case NDARRAY_CHAR:
			{
				char* a=NULL;
				int32_t* dims=NULL;
				int32_t num_dims=0;
				source->get_ndarray(a, dims, num_dims);
				target->set_ndarray(a, dims, num_dims);
				SG_FREE(a);
				SG_FREE(dims);
				break;
			}
		case NDARRAY_INT:
			{
				int32_t* a=NULL;
				int32_t* dims=NULL;
				int32_t num_dims=0;
				source->get_ndarray(a, dims, num_dims);
				target->set_ndarray(a, dims, num_dims);
				SG_FREE(a);
				SG_FREE(dims);
				break;
			}
		case NDARRAY_REAL:
			{
				float64_t* a=NULL;
				int32_t* dims=NULL;
				int32_t num_dims=0;
				source->get_ndarray(a, dims, num_dims);
				target->set_ndarray(a, dims, num_dims);
				SG_FREE(a);
				SG_FREE(dims);
				break;
			}
		case NDARRAY_SHORTREAL:
			{
				float32_t* a=NULL;
				int32_t* dims=NULL;
				int32_t num_dims=0;
				source->get_ndarray(a, dims, num_dims);
				target->set_ndarray(a, dims, num_dims);
				SG_FREE(a);
				SG_FREE(dims);
				break;
			}
		case NDARRAY_SHORT:
			{
				int16_t* a=NULL;
				int32_t* dims=NULL;
				int32_t num_dims=0;
				source->get_ndarray(a, dims, num_dims);
				target->set_ndarray(a, dims, num_dims);
				SG_FREE(a);
				SG_FREE(dims);
				break;
			}
		case NDARRAY_WORD:
			{
				uint16_t* a=NULL;
				int32_t* dims=NULL;
				int32_t num_dims=0;
				source->get_ndarray(a, dims, num_dims);
				target->set_ndarray(a, dims, num_dims);
				SG_FREE(a);
				SG_FREE(dims);
				break;
			}*/
		case SPARSE_REAL:
			{
				int32_t num_feat=0;
				int32_t num_vec=0;
				SGSparseVector<float64_t>* fmatrix=NULL;
				source->get_sparse_matrix(fmatrix, num_feat, num_vec);
				int64_t nnz=0;
				for (int32_t i=0; i<num_vec; i++)
					nnz+=fmatrix[i].num_feat_entries;
				target->set_sparse_matrix(fmatrix, num_feat, num_vec, nnz);
				SG_FREE(fmatrix);
				break;
			}

		default:
			SG_ERROR("unknown return type")
			break;
	}
}

////////////////////////////////////////////////////////////////////////////
// commands
////////////////////////////////////////////////////////////////////////////

/* Features */

bool CSGInterface::cmd_load_features()
{
	if (m_nrhs<8 || !create_return_values(0))
		return false;

	int32_t len=0;
	char* filename=get_str_from_str_or_direct(len);
	char* fclass=get_str_from_str_or_direct(len);
	char* type=get_str_from_str_or_direct(len);
	char* target=get_str_from_str_or_direct(len);
	int32_t size=get_int_from_int_or_str();
	int32_t comp_features=get_int_from_int_or_str();

	bool success=ui_features->load(
		filename, fclass, type, target, size, comp_features);

	SG_FREE(filename);
	SG_FREE(fclass);
	SG_FREE(type);
	SG_FREE(target);
	return success;
}

bool CSGInterface::cmd_save_features()
{
	if (m_nrhs<5 || !create_return_values(0))
		return false;

	int32_t len=0;
	char* filename=get_str_from_str_or_direct(len);
	char* type=get_str_from_str_or_direct(len);
	char* target=get_str_from_str_or_direct(len);

	bool success=ui_features->save(filename, type, target);

	SG_FREE(filename);
	SG_FREE(type);
	SG_FREE(target);
	return success;
}

bool CSGInterface::cmd_clean_features()
{
	if (m_nrhs<2 || !create_return_values(0))
		return false;

	int32_t len=0;
	char* target=get_str_from_str_or_direct(len);

	bool success=ui_features->clean(target);

	SG_FREE(target);
	return success;
}

bool CSGInterface::cmd_get_features()
{
	if (m_nrhs!=2 || !create_return_values(1))
		return false;

	int32_t tlen=0;
	char* target=get_string(tlen);
	CFeatures* feat=NULL;

	if (strmatch(target, "TRAIN"))
		feat=ui_features->get_train_features();
	else if (strmatch(target, "TEST"))
		feat=ui_features->get_test_features();
	else
	{
		SG_FREE(target);
		SG_ERROR("Unknown target, neither TRAIN nor TEST.\n")
	}
	SG_FREE(target);

	ASSERT(feat)

	switch (feat->get_feature_class())
	{
		case C_DENSE:
		{
			int32_t num_feat=0;
			int32_t num_vec=0;

			switch (feat->get_feature_type())
			{
				case F_BYTE:
				{
					uint8_t* fmatrix=((CDenseFeatures<uint8_t> *) feat)->get_feature_matrix(num_feat, num_vec);
					set_matrix(fmatrix, num_feat, num_vec);
					break;
				}

				case F_CHAR:
				{
					char* fmatrix=((CDenseFeatures<char> *) feat)->get_feature_matrix(num_feat, num_vec);
					set_matrix(fmatrix, num_feat, num_vec);
					break;
				}

				case F_DREAL:
				{
					float64_t* fmatrix=((CDenseFeatures<float64_t> *) feat)->get_feature_matrix(num_feat, num_vec);
					set_matrix(fmatrix, num_feat, num_vec);
					break;
				}

				case F_INT:
				{
					int32_t* fmatrix=((CDenseFeatures<int32_t> *) feat)->get_feature_matrix(num_feat, num_vec);
					set_matrix(fmatrix, num_feat, num_vec);
					break;
				}

				case F_SHORT:
				{
					int16_t* fmatrix=((CDenseFeatures<int16_t> *) feat)->get_feature_matrix(num_feat, num_vec);
					set_matrix(fmatrix, num_feat, num_vec);
					break;
				}

				case F_SHORTREAL:
				{
					float32_t* fmatrix=((CDenseFeatures<float32_t> *) feat)->get_feature_matrix(num_feat, num_vec);
					set_matrix(fmatrix, num_feat, num_vec);
					break;
				}

				case F_WORD:
				{
					uint16_t* fmatrix=((CDenseFeatures<uint16_t> *) feat)->get_feature_matrix(num_feat, num_vec);
					set_matrix(fmatrix, num_feat, num_vec);
					break;
				}

				default:
					SG_NOTIMPLEMENTED
			}
			break;
		}

		case C_SPARSE:
		{
			switch (feat->get_feature_type())
			{
				case F_DREAL:
				{
					int64_t nnz=((CSparseFeatures<float64_t>*) feat)->
						get_num_nonzero_entries();
					SGSparseMatrix<float64_t> fmatrix = ((CSparseFeatures<float64_t>*) feat)->get_sparse_feature_matrix();
					SG_INFO("sparse matrix has %d feats, %d vecs and %d nnz elemements\n", fmatrix.num_features, fmatrix.num_vectors, nnz)

					set_sparse_matrix(fmatrix.sparse_matrix, fmatrix.num_features, fmatrix.num_vectors, nnz);
					break;
				}

				default:
					SG_NOTIMPLEMENTED
			}
			break;
		}

		case C_STRING:
		{
			int32_t num_str=0;
			int32_t max_str_len=0;
			switch (feat->get_feature_type())
			{
				case F_BYTE:
				{
					SGString<uint8_t>* fmatrix=((CStringFeatures<uint8_t>*) feat)->get_features(num_str, max_str_len);
					set_string_list(fmatrix, num_str);
					break;
				}

				case F_CHAR:
				{
					SGString<char>* fmatrix=((CStringFeatures<char>*) feat)->get_features(num_str, max_str_len);
					set_string_list(fmatrix, num_str);
					break;
				}

				case F_WORD:
				{
					SGString<uint16_t>* fmatrix=((CStringFeatures<uint16_t>*) feat)->get_features(num_str, max_str_len);
					set_string_list(fmatrix, num_str);
					break;
				}

				default:
					SG_NOTIMPLEMENTED
			}
			break;
		}

		case C_WD:
		case C_WEIGHTEDSPEC:
		case C_SPEC:
		case C_COMBINED_DOT:
		case C_POLY:
		{

			SGMatrix<float64_t> fmatrix = ((CDotFeatures*) feat)->get_computed_dot_feature_matrix();
			set_matrix(fmatrix.matrix, fmatrix.num_cols, fmatrix.num_rows);
			break;
		}

		default:
			SG_NOTIMPLEMENTED
	}

	return true;
}

bool CSGInterface::cmd_add_features()
{
	if (m_nrhs<3 || !create_return_values(0))
		return false;

	return do_set_features(true, false);
}

bool CSGInterface::cmd_add_multiple_features()
{
	if ((m_nrhs!=4 && m_nrhs<5) || !create_return_values(0))
		return false;

	int32_t repetitions=get_int();

	ASSERT(repetitions>=1)

	return do_set_features(true, false, repetitions);
}

bool CSGInterface::cmd_add_dotfeatures()
{
	if (m_nrhs<3 || !create_return_values(0))
		return false;

	return do_set_features(true, true);
}

bool CSGInterface::cmd_set_features()
{
	if (m_nrhs<3 || !create_return_values(0))
		return false;

	return do_set_features(false, false);
}

bool CSGInterface::do_set_features(bool add, bool check_dot, int32_t repetitions)
{
	int32_t tlen=0;
	char* target=get_string(tlen);
	if (!strmatch(target, "TRAIN") && !strmatch(target, "TEST"))
	{
		SG_FREE(target);
		SG_ERROR("Unknown target, neither TRAIN nor TEST.\n")
	}

	CFeatures* feat=NULL;
	int32_t num_feat=0;
	int32_t num_vec=0;

	switch (get_argument_type())
	{
		case SPARSE_REAL:
		{
			SGSparseVector<float64_t>* fmatrix=NULL;
			get_sparse_matrix(fmatrix, num_feat, num_vec);

			feat=new CSparseFeatures<float64_t>(SGSparseMatrix<float64_t>(fmatrix, num_feat, num_vec));
			break;
		}

		case DENSE_REAL:
		{
			float64_t* fmatrix=NULL;
			get_matrix(fmatrix, num_feat, num_vec);

			feat=new CDenseFeatures<float64_t>(0);
			((CDenseFeatures<float64_t>*) feat)->
				set_feature_matrix(SGMatrix<float64_t>(fmatrix, num_feat, num_vec));

			if (m_nrhs==6)
				feat = create_custom_real_features((CDenseFeatures<float64_t>*) feat);

			break;
		}

		case DENSE_INT:
		{
			int32_t* fmatrix=NULL;
			get_matrix(fmatrix, num_feat, num_vec);

			feat=new CDenseFeatures<int32_t>(0);
			((CDenseFeatures<int32_t>*) feat)->
				set_feature_matrix(SGMatrix<int32_t>(fmatrix, num_feat, num_vec));
			break;
		}

		case DENSE_SHORT:
		{
			int16_t* fmatrix=NULL;
			get_matrix(fmatrix, num_feat, num_vec);

			feat=new CDenseFeatures<int16_t>(0);
			((CDenseFeatures<int16_t>*) feat)->
				set_feature_matrix(SGMatrix<int16_t>(fmatrix, num_feat, num_vec));
			break;
		}

		case DENSE_WORD:
		{
			uint16_t* fmatrix=NULL;
			get_matrix(fmatrix, num_feat, num_vec);

			feat=new CDenseFeatures<uint16_t>(0);
			((CDenseFeatures<uint16_t>*) feat)->
				set_feature_matrix(SGMatrix<uint16_t>(fmatrix, num_feat, num_vec));
			break;
		}

		case DENSE_SHORTREAL:
		{
			float32_t* fmatrix=NULL;
			get_matrix(fmatrix, num_feat, num_vec);

			feat=new CDenseFeatures<float32_t>(0);
			((CDenseFeatures<float32_t>*) feat)->
				set_feature_matrix(SGMatrix<float32_t>(fmatrix, num_feat, num_vec));
			break;
		}

		case STRING_CHAR:
		{
			if (m_nrhs<4)
				SG_ERROR("Please specify alphabet!\n")

			int32_t num_str=0;
			int32_t max_str_len=0;
			SGString<char>* fmatrix=NULL;
			get_string_list(fmatrix, num_str, max_str_len);

			int32_t alphabet_len=0;
			char* alphabet_str=get_string(alphabet_len);
			ASSERT(alphabet_str)

			if (strmatch(alphabet_str, "DNABINFILE"))
			{
				SG_FREE(alphabet_str);

				ASSERT(fmatrix[0].string)
				feat=new CStringFeatures<uint8_t>(DNA);

				try
				{
					((CStringFeatures<uint8_t>*) feat)->load_ascii_file(fmatrix[0].string);
				}
				catch (...)
				{
					SG_UNREF(feat);
					SG_ERROR("Couldn't load DNA features from file.\n")
				}
				feat=create_custom_string_features((CStringFeatures<uint8_t>*) feat);
				break;
			}
			else
			{
				bool convert_to_word=false;
				bool convert_to_ulong=false;
				CAlphabet* alphabet=NULL;
				if (strmatch(alphabet_str, "DNAWORD"))
				{
					alphabet=new CAlphabet(DNA);
					convert_to_word=true;
				}
				else if (strmatch(alphabet_str, "DNAULONG"))
				{
					alphabet=new CAlphabet(DNA);
					convert_to_ulong=true;
				}
				else
					alphabet=new CAlphabet(alphabet_str, alphabet_len);

				SG_REF(alphabet);
				SG_FREE(alphabet_str);

				feat=new CStringFeatures<char>(alphabet);

				if (!((CStringFeatures<char>*) feat)->set_features(fmatrix, num_str, max_str_len))
				{
					SG_UNREF(alphabet);
					SG_UNREF(feat);
					SG_ERROR("Couldnt set byte string features.\n")
				}

				SG_UNREF(alphabet);

				if (convert_to_word || convert_to_ulong)
					convert_to_bitembedding(feat, convert_to_word, convert_to_ulong);
			}

			obtain_from_single_string(feat);
			break;
		}

		case STRING_BYTE:
		{
			if (m_nrhs<4)
				SG_ERROR("Please specify alphabet!\n")

			int32_t num_str=0;
			int32_t max_str_len=0;
			SGString<uint8_t>* fmatrix=NULL;
			get_string_list(fmatrix, num_str, max_str_len);

			int32_t alphabet_len=0;
			char* alphabet_str=get_string(alphabet_len);
			ASSERT(alphabet_str)
			CAlphabet* alphabet=NULL;
			alphabet=new CAlphabet(alphabet_str, alphabet_len);
			SG_FREE(alphabet_str);

			feat=new CStringFeatures<uint8_t>(alphabet);
			if (!((CStringFeatures<uint8_t>*) feat)->set_features(fmatrix, num_str, max_str_len))
			{
				SG_UNREF(alphabet);
				SG_UNREF(feat);
				SG_ERROR("Couldnt set byte string features.\n")
			}
			feat=create_custom_string_features((CStringFeatures<uint8_t>*) feat);
			break;
		}

		default:
			SG_ERROR("Wrong argument type %d.\n", get_argument_type())
	}

	if (check_dot && !feat->has_property(FP_DOT))
	{
		SG_UNREF(feat);
		SG_ERROR("Feature type not supported by DOT Features\n")
	}

	if (strmatch(target, "TRAIN"))
	{
		if (!add)
			ui_features->set_train_features(feat);
		else if (check_dot)
		{
			for (int32_t i=0; i<repetitions; i++)
			ui_features->add_train_dotfeatures((CDotFeatures*) feat);
		}
		else
		{
			for (int32_t i=0; i<repetitions; i++)
			ui_features->add_train_features(feat);
		}
	}
	else
	{
		if (!add)
			ui_features->set_test_features(feat);
		else if (check_dot)
		{
			for (int32_t i=0; i<repetitions; i++)
				ui_features->add_test_dotfeatures((CDotFeatures*) feat);
		}
		else
		{
			for (int32_t i=0; i<repetitions; i++)
				ui_features->add_test_features(feat);
		}
	}

	SG_FREE(target);

	return true;
}

bool CSGInterface::cmd_set_reference_features()
{
	if (m_nrhs<3 || !create_return_values(0))
		return false;

	int32_t len=0;
	char* target=get_str_from_str_or_direct(len);

	bool success=ui_features->set_reference_features(target);

	SG_FREE(target);
	return success;
}

bool CSGInterface::cmd_del_last_features()
{
	if (m_nrhs<2 || !create_return_values(0))
		return false;

	int32_t len=0;
	char* target=get_str_from_str_or_direct(len);
	bool success=ui_features->del_last_feature_obj(target);

	SG_FREE(target);
	return success;
}

bool CSGInterface::cmd_convert()
{
	if (m_nrhs<5 || !create_return_values(0))
		return false;

	int32_t len=0;
	char* target=get_str_from_str_or_direct(len);
	CFeatures* features=ui_features->get_convert_features(target);
	if (!features)
	{
		SG_FREE(target);
		SG_ERROR("No \"%s\" features available.\n", target)
	}

	char* from_class=get_str_from_str_or_direct(len);
	char* from_type=get_str_from_str_or_direct(len);
	char* to_class=get_str_from_str_or_direct(len);
	char* to_type=get_str_from_str_or_direct(len);

	CFeatures* result=NULL;
	if (strmatch(from_class, "SIMPLE"))
	{
		if (strmatch(from_type, "REAL"))
		{
			if (strmatch(to_class, "SPARSE") &&
				strmatch(to_type, "REAL"))
			{
				result=ui_features->convert_simple_real_to_sparse_real(
					((CDenseFeatures<float64_t>*) features));
			}
			else
				SG_NOTIMPLEMENTED
		} // from_type REAL

		else if (strmatch(from_type, "CHAR"))
		{
			if (strmatch(to_class, "STRING") &&
				strmatch(to_type, "CHAR"))
			{
				result=ui_features->convert_simple_char_to_string_char(
					((CDenseFeatures<char>*) features));
			}
			else if (strmatch(to_class, "SIMPLE"))
			{
				if (strmatch(to_type, "ALIGN") && m_nrhs==8)
				{
					float64_t gap_cost=get_real_from_real_or_str();
					result=ui_features->convert_simple_char_to_simple_align(
						(CDenseFeatures<char>*) features, gap_cost);
				}
				else
					SG_NOTIMPLEMENTED
			}
			else
				SG_NOTIMPLEMENTED
		} // from_type CHAR

		else if (strmatch(from_type, "WORD"))
		{
			if (strmatch(to_class, "SIMPLE") &&
				strmatch(to_type, "SALZBERG"))
			{
				result=ui_features->convert_simple_word_to_simple_salzberg(
					(CDenseFeatures<uint16_t>*) features);
			}
			else
				SG_NOTIMPLEMENTED
		} // from_type WORD

		else
			SG_NOTIMPLEMENTED
	} // from_class SIMPLE

	else if (strmatch(from_class, "SPARSE"))
	{
		if (strmatch(from_type, "REAL"))
		{
			if (strmatch(to_class, "SIMPLE") &&
				strmatch(to_type, "REAL"))
			{
				result=ui_features->convert_sparse_real_to_simple_real(
					(CSparseFeatures<float64_t>*) features);
			}
			else
				SG_NOTIMPLEMENTED
		} // from_type REAL
		else
			SG_NOTIMPLEMENTED
	} // from_class SPARSE

	else if (strmatch(from_class, "STRING"))
	{
		if (strmatch(from_type, "CHAR"))
		{
			if (strmatch(to_class, "STRING"))
			{
				int32_t order=1;
				int32_t start=0;
				int32_t gap=0;
				char rev='f';

				if (m_nrhs>6)
				{
					order=get_int_from_int_or_str();

					if (m_nrhs>7)
					{
						start=get_int_from_int_or_str();

						if (m_nrhs>8)
						{
							gap=get_int_from_int_or_str();

							if (m_nrhs>9)
							{
								char* rev_str=get_str_from_str_or_direct(len);
								if (rev_str)
									rev=rev_str[0];

								SG_FREE(rev_str);
							}
						}
					}
				}

				if (strmatch(to_type, "BYTE"))
				{
						result=ui_features->convert_string_char_to_string_generic<char,uint8_t>(
						(CStringFeatures<char>*) features, order, start,
						gap, rev);
				}
				else if (strmatch(to_type, "WORD"))
				{
						result=ui_features->convert_string_char_to_string_generic<char,uint16_t>(
						(CStringFeatures<char>*) features, order, start,
						gap, rev);
				}
				else if (strmatch(to_type, "ULONG"))
				{
					result=ui_features->convert_string_char_to_string_generic<char,uint64_t>(
					(CStringFeatures<char>*) features, order, start,
						gap, rev);
				}
				else
					SG_NOTIMPLEMENTED
			}
			else
				SG_NOTIMPLEMENTED
		} // from_type CHAR

		else if (strmatch(from_type, "BYTE"))
		{
			if (strmatch(to_class, "STRING"))
			{
				int32_t order=1;
				int32_t start=0;
				int32_t gap=0;
				char rev='f';

				if (m_nrhs>6)
				{
					order=get_int_from_int_or_str();

					if (m_nrhs>7)
					{
						start=get_int_from_int_or_str();

						if (m_nrhs>8)
						{
							gap=get_int_from_int_or_str();

							if (m_nrhs>9)
							{
								char* rev_str=get_str_from_str_or_direct(len);
								if (rev_str)
									rev=rev_str[0];

								SG_FREE(rev_str);
							}
						}
					}
				}

				if (strmatch(to_type, "WORD"))
				{
					result=ui_features->convert_string_char_to_string_generic<uint8_t,uint16_t>(
						(CStringFeatures<uint8_t>*) features, order, start,
						gap, rev);
				}
				else if (strmatch(to_type, "ULONG"))
				{
					result=ui_features->convert_string_char_to_string_generic<uint8_t,uint64_t>(
						(CStringFeatures<uint8_t>*) features, order, start,
						gap, rev);
				}
				else
					SG_NOTIMPLEMENTED
			}
			else
				SG_NOTIMPLEMENTED
		} // from_type uint8_t

		else if (strmatch(from_type, "WORD"))
		{
			if (strmatch(to_class, "SIMPLE") && strmatch(to_type, "TOP"))
			{
				result=ui_features->convert_string_word_to_simple_top(
					(CStringFeatures<uint16_t>*) features);
			}
			else if (strmatch(to_class, "SPEC") && strmatch(to_type, "WORD") && m_nrhs==7)
			{
				bool use_norm=get_bool();
				result=ui_features->convert_string_byte_to_spec_word((CStringFeatures<uint16_t>*) features, use_norm);

			}
			else
				SG_NOTIMPLEMENTED
		} // from_type WORD

		else if (strmatch(to_class, "SIMPLE") && strmatch(to_type, "FK"))
		{
			result=ui_features->convert_string_word_to_simple_fk(
				(CStringFeatures<uint16_t>*) features);
		} // to_type FK

		else
			SG_NOTIMPLEMENTED

	} // from_class STRING

	if (result && ui_features->set_convert_features(result, target))
		SG_INFO("Conversion was successful.\n")
	else
		SG_ERROR("Conversion failed.\n")

	SG_FREE(target);
	SG_FREE(from_class);
	SG_FREE(from_type);
	SG_FREE(to_class);
	SG_FREE(to_type);
	return (result!=NULL);
}

void CSGInterface::convert_to_bitembedding(CFeatures* &features, bool convert_to_word, bool convert_to_ulong)
{
	int32_t order=1;
	int32_t start=0;
	int32_t gap=0;
	char rev='f';

	if (m_nrhs<5)
		return;

	order=get_int();
	// remove arg, for parameters to come
	m_nrhs--;

	if (convert_to_word)
	{
		SG_INFO("Converting into word-bitembedding\n")
		features=ui_features->convert_string_char_to_string_generic<char,uint16_t>(
				(CStringFeatures<char>*) features, order, start, gap, rev);
	}

	if (convert_to_ulong)
	{
		SG_INFO("Converting into ulong-bitembedding\n")
		features=ui_features->convert_string_char_to_string_generic<char,uint64_t>(
				(CStringFeatures<char>*) features, order, start, gap, rev);
	}
}

void CSGInterface::obtain_from_single_string(CFeatures* features)
{
	if (m_nrhs<5)
		return;

	int32_t len=0;
	char* str=get_string(len);
	ASSERT(str)

	if (strmatch(str, "from_position_list"))
	{
		obtain_from_position_list(features);
	}
	else if (strmatch(str, "slide_window"))
	{
		obtain_by_sliding_window(features);
	}
	else
		SG_SERROR("Unknown conversion\n")
}

bool CSGInterface::obtain_from_position_list(CFeatures* features)
{
	int32_t winsize=get_int();

	int32_t* shifts=NULL;
	int32_t num_shift=0;
	get_vector(shifts, num_shift);

	int32_t skip=0;
	if (m_nrhs==8)
		skip=get_int();

	SG_DEBUG("winsize: %d num_shifts: %d skip: %d\n", winsize, num_shift, skip)

	CDynamicArray<int32_t> positions(num_shift+1);

	for (int32_t i=0; i<num_shift; i++)
		positions.set_element(shifts[i], i);

	if (features->get_feature_class()!=C_STRING)
		SG_ERROR("No string features.\n")

	bool success=false;
	switch (features->get_feature_type())
	{
		case F_CHAR:
		{
			success=(((CStringFeatures<char>*) features)->
				obtain_by_position_list(winsize, &positions, skip)>0);
			break;
		}
		case F_BYTE:
		{
			success=(((CStringFeatures<uint8_t>*) features)->
				obtain_by_position_list(winsize, &positions, skip)>0);
			break;
		}
		case F_WORD:
		{
			success=(((CStringFeatures<uint16_t>*) features)->
				obtain_by_position_list(winsize, &positions, skip)>0);
			break;
		}
		case F_ULONG:
		{
			success=(((CStringFeatures<uint64_t>*) features)->
				obtain_by_position_list(winsize, &positions, skip)>0);
			break;
		}
		default:
			SG_ERROR("Unsupported string features type.\n")
	}

	return success;
}

bool CSGInterface::obtain_by_sliding_window(CFeatures* features)
{
	int32_t winsize=get_int();
	int32_t shift=get_int();
	int32_t skip=0;

	if (m_nrhs==8)
		skip=get_int();

	bool success=false;

	ASSERT(features)
	ASSERT(((CFeatures*) features)->get_feature_class()==C_STRING)

	switch (features->get_feature_type())
	{
		case F_CHAR:
			return ( ((CStringFeatures<char>*) features)->obtain_by_sliding_window(winsize, shift, skip)>0);
		case F_BYTE:
			return ( ((CStringFeatures<uint8_t>*) features)->obtain_by_sliding_window(winsize, shift, skip)>0);
		case F_WORD:
			return ( ((CStringFeatures<uint16_t>*) features)->obtain_by_sliding_window(winsize, shift, skip)>0);
		case F_ULONG:
			return ( ((CStringFeatures<uint64_t>*) features)->obtain_by_sliding_window(winsize, shift, skip)>0);
		default:
			SG_SERROR("Unsupported string features type.\n")
			return false;
	}

	return success;
}

bool CSGInterface::cmd_reshape()
{
	if (m_nrhs<4 || !create_return_values(0))
		return false;

	int32_t len=0;
	char* target=get_str_from_str_or_direct(len);
	int32_t num_feat=get_int_from_int_or_str();
	int32_t num_vec=get_int_from_int_or_str();

	bool success=ui_features->reshape(target, num_feat, num_vec);

	SG_FREE(target);
	return success;
}

bool CSGInterface::cmd_load_labels()
{
	if (m_nrhs<4 || !create_return_values(0))
		return false;

	int32_t len=0;
	char* filename=get_str_from_str_or_direct(len);
	char* target=get_str_from_str_or_direct(len);

	bool success=ui_labels->load(filename, target);

	SG_FREE(filename);
	SG_FREE(target);
	return success;
}

bool CSGInterface::cmd_set_labels()
{
	if (m_nrhs!=3 || !create_return_values(0))
		return false;

	int32_t tlen=0;
	char* target=get_string(tlen);
	if (!strmatch(target, "TRAIN") && !strmatch(target, "TEST"))
	{
		SG_FREE(target);
		SG_ERROR("Unknown target, neither TRAIN nor TEST.\n")
	}

	float64_t* lab=NULL;
	int32_t len=0;
	get_vector(lab, len);

	CLabels* labels=ui_labels->infer_labels(lab, len);

	SG_INFO("num labels: %d\n", labels->get_num_labels())

	if (strmatch(target, "TRAIN"))
		ui_labels->set_train_labels(labels);
	else if (strmatch(target, "TEST"))
		ui_labels->set_test_labels(labels);
	else
	{
		SG_FREE(target);
		SG_ERROR("Unknown target, neither TRAIN nor TEST.\n")
	}
	SG_FREE(target);

	return true;
}

bool CSGInterface::cmd_get_labels()
{
	if (m_nrhs!=2 || !create_return_values(1))
		return false;

	int32_t tlen=0;
	char* target=get_string(tlen);
	CLabels* labels=NULL;

	if (strmatch(target, "TRAIN"))
		labels=ui_labels->get_train_labels();
	else if (strmatch(target, "TEST"))
		labels=ui_labels->get_test_labels();
	else
	{
		SG_FREE(target);
		SG_ERROR("Unknown target, neither TRAIN nor TEST.\n")
	}
	SG_FREE(target);

	if (!labels)
		SG_ERROR("No labels.\n")

	//FIXME
	SGVector<float64_t> lab=((CBinaryLabels*) labels)->get_labels();

	set_vector(lab.vector, lab.vlen);
	return true;
}


/** Kernel */

bool CSGInterface::cmd_set_kernel_normalization()
{
	if (m_nrhs<2 || !create_return_values(0))
		return false;

	int32_t len=0;
	char* normalization=get_string(len);

	float64_t c=0;
	float64_t r=0;

	if (m_nrhs>=3)
		c=get_real();
	if (m_nrhs>=4)
		r=get_real();

	bool success=ui_kernel->set_normalization(normalization, c, r);

	SG_FREE(normalization);
	return success;
}

bool CSGInterface::cmd_set_kernel()
{
	if (m_nrhs<2 || !create_return_values(0))
		return false;

	SG_DEBUG("SGInterface: set_kernel\n")
	CKernel* kernel=create_kernel();
	return ui_kernel->set_kernel(kernel);
}

bool CSGInterface::cmd_add_kernel()
{
	if (m_nrhs<3 || !create_return_values(0))
		return false;

	float64_t weight=get_real_from_real_or_str();
	// adjust m_nrhs to play well with checks in create_kernel
	m_nrhs--;
	CKernel* kernel=create_kernel();

	SG_DEBUG("SGInterface: add_kernel\n")
	return ui_kernel->add_kernel(kernel, weight);
}

bool CSGInterface::cmd_del_last_kernel()
{
	if (m_nrhs<1 || !create_return_values(0))
		return false;

	return ui_kernel->del_last_kernel();
}

CKernel* CSGInterface::create_kernel()
{
	CKernel* kernel=NULL;
	int32_t len=0;
	char* type=get_str_from_str_or_direct(len);

	SG_DEBUG("set_kernel with type: %s\n", type)

	if (strmatch(type, "COMBINED"))
	{
		if (m_nrhs<3)
			return NULL;

		int32_t size=get_int_from_int_or_str();
		bool append_subkernel_weights=false;
		if (m_nrhs>3)
			append_subkernel_weights=get_bool_from_bool_or_str();

		kernel=ui_kernel->create_combined(size, append_subkernel_weights);
	}
	else if (strmatch(type, "DISTANCE"))
	{
		if (m_nrhs<3)
			return NULL;

		int32_t size=get_int_from_int_or_str();
		float64_t width=1;
		if (m_nrhs>3)
			width=get_real_from_real_or_str();

		kernel=ui_kernel->create_distance(size, width);
	}
	else if (strmatch(type, "WAVELET"))
	{

                if (m_nrhs<4)
                        return NULL;

                char* dtype=get_str_from_str_or_direct(len);
                if (strmatch(dtype, "REAL"))
                {
                        int32_t size=get_int_from_int_or_str();
                        float64_t Wdilation=5.0;
                        float64_t Wtranslation=2.0;

                        if (m_nrhs>4)
                        {
                                Wdilation=get_real_from_real_or_str();

                                if (m_nrhs>5)
                                        Wtranslation=get_real_from_real_or_str();
                        }

                        kernel=ui_kernel->create_sigmoid(size, Wdilation, Wtranslation);
                }

                SG_FREE(dtype);
        }
	else if (strmatch(type, "LINEAR"))
	{
		if (m_nrhs<4)
			return NULL;
		if (m_nrhs>5)
			return NULL;

		char* dtype=get_str_from_str_or_direct(len);
		int32_t size=get_int_from_int_or_str();
		float64_t scale=-1;
		if (m_nrhs==5)
			scale=get_real_from_real_or_str();

		if (strmatch(dtype, "BYTE"))
			kernel=ui_kernel->create_linearbyte(size, scale);
		else if (strmatch(dtype, "WORD"))
			kernel=ui_kernel->create_linearword(size, scale);
		else if (strmatch(dtype, "CHAR"))
			kernel=ui_kernel->create_linearstring(size, scale);
		else if (strmatch(dtype, "REAL"))
			kernel=ui_kernel->create_linear(size, scale);
		else if (strmatch(dtype, "SPARSEREAL"))
			kernel=ui_kernel->create_sparselinear(size, scale);

		SG_FREE(dtype);
	}
	else if (strmatch(type, "HISTOGRAM"))
	{
		if (m_nrhs<4)
			return NULL;

		char* dtype=get_str_from_str_or_direct(len);
		if (strmatch(dtype, "WORD"))
		{
			int32_t size=get_int_from_int_or_str();
			kernel=ui_kernel->create_histogramword(size);
		}

		SG_FREE(dtype);
	}
	else if (strmatch(type, "SALZBERG"))
	{
		if (m_nrhs<4)
			return NULL;

		char* dtype=get_str_from_str_or_direct(len);
		if (strmatch(dtype, "WORD"))
		{
			int32_t size=get_int_from_int_or_str();
			kernel=ui_kernel->create_salzbergword(size);
		}

		SG_FREE(dtype);
	}
	else if (strmatch(type, "POLYMATCH"))
	{
		if (m_nrhs<4)
			return NULL;

		char* dtype=get_str_from_str_or_direct(len);
		int32_t size=get_int_from_int_or_str();
		int32_t degree=3;
		bool inhomogene=false;
		bool normalize=true;

		if (m_nrhs>4)
		{
			degree=get_int_from_int_or_str();
			if (m_nrhs>5)
			{
				inhomogene=get_bool_from_bool_or_str();
				if (m_nrhs>6)
					normalize=get_bool_from_bool_or_str();
			}
		}

		if (strmatch(dtype, "CHAR"))
		{
			kernel=ui_kernel->create_polymatchstring(
				size, degree, inhomogene, normalize);
		}
		else if (strmatch(dtype, "WORD"))
		{
			kernel=ui_kernel->create_polymatchwordstring(
				size, degree, inhomogene, normalize);
		}

		SG_FREE(dtype);
	}
	else if (strmatch(type, "MATCH"))
	{
		if (m_nrhs<4)
			return NULL;

		char* dtype=get_str_from_str_or_direct(len);
		if (strmatch(dtype, "WORD"))
		{
			int32_t size=get_int_from_int_or_str();
			int32_t d=3;
			bool normalize=true;

			if (m_nrhs>4)
				d=get_int_from_int_or_str();
			if (m_nrhs>5)
				normalize=get_bool_from_bool_or_str();

			kernel=ui_kernel->create_matchwordstring(size, d, normalize);
		}

		SG_FREE(dtype);
	}
	else if (strmatch(type, "WEIGHTEDCOMMSTRING") || strmatch(type, "COMMSTRING"))
	{
		char* dtype=get_str_from_str_or_direct(len);
		int32_t size=get_int_from_int_or_str();
		bool use_sign=false;
		char* norm_str=NULL;

		if (m_nrhs>4)
		{
			 use_sign=get_bool_from_bool_or_str();

			 if (m_nrhs>5)
				norm_str=get_str_from_str_or_direct(len);
		}

		if (strmatch(dtype, "WORD"))
		{
			if (strmatch(type, "WEIGHTEDCOMMSTRING"))
			{
				kernel=ui_kernel->create_commstring(
					size, use_sign, norm_str, K_WEIGHTEDCOMMWORDSTRING);
			}
			else if (strmatch(type, "COMMSTRING"))
			{
				kernel=ui_kernel->create_commstring(
					size, use_sign, norm_str, K_COMMWORDSTRING);
			}
		}
		else if (strmatch(dtype, "ULONG"))
		{
			kernel=ui_kernel->create_commstring(
				size, use_sign, norm_str, K_COMMULONGSTRING);
		}

		SG_FREE(dtype);
		SG_FREE(norm_str);
	}
	else if (strmatch(type, "CHI2"))
	{
		if (m_nrhs<4)
			return NULL;

		char* dtype=get_str_from_str_or_direct(len);
		if (strmatch(dtype, "REAL"))
		{
			int32_t size=get_int_from_int_or_str();
			float64_t width=1;

			if (m_nrhs>4)
				width=get_real_from_real_or_str();

			kernel=ui_kernel->create_chi2(size, width);
		}

		SG_FREE(dtype);
	}
	else if (strmatch(type, "FIXEDDEGREE"))
	{
		if (m_nrhs<4)
			return NULL;

		char* dtype=get_str_from_str_or_direct(len);
		if (strmatch(dtype, "CHAR"))
		{
			int32_t size=get_int_from_int_or_str();
			int32_t d=3;
			if (m_nrhs>4)
				d=get_int_from_int_or_str();

			kernel=ui_kernel->create_fixeddegreestring(size, d);
		}

		SG_FREE(dtype);
	}
	else if (strmatch(type, "LOCALALIGNMENT"))
	{
		if (m_nrhs<4)
			return NULL;

		char* dtype=get_str_from_str_or_direct(len);
		if (strmatch(dtype, "CHAR"))
		{
			int32_t size=get_int_from_int_or_str();

			kernel=ui_kernel->create_localalignmentstring(size);
		}

		SG_FREE(dtype);
	}
	else if (strmatch(type, "OLIGO"))
	{
		if (m_nrhs<6)
			return NULL;

		char* dtype=get_str_from_str_or_direct(len);
		if (strmatch(dtype, "CHAR"))
		{
			int32_t size=get_int_from_int_or_str();
			int32_t k=get_int_from_int_or_str();
			float64_t w=get_real_from_real_or_str();

			kernel=ui_kernel->create_oligo(size, k, w);
		}

		SG_FREE(dtype);
	}
	else if (strmatch(type, "WEIGHTEDDEGREEPOS2") ||
		strmatch(type, "WEIGHTEDDEGREEPOS2_NONORM"))
	{
		if (m_nrhs<7)
			return NULL;

		char* dtype=get_str_from_str_or_direct(len);
		if (strmatch(dtype, "CHAR") || strmatch(dtype, "STRING"))
		{
			int32_t size=get_int_from_int_or_str();
			int32_t order=get_int_from_int_or_str();
			int32_t max_mismatch=get_int_from_int_or_str();
			int32_t length=get_int_from_int_or_str();
			int32_t* shifts=NULL;
			int32_t l=0;
			get_vector_from_int_vector_or_str(shifts, l);

			ASSERT(l==length)

			bool use_normalization=true;
			if (strmatch(type, "WEIGHTEDDEGREEPOS2_NONORM"))
				use_normalization=false;

			kernel=ui_kernel->create_weighteddegreepositionstring2(
				size, order, max_mismatch, shifts, length,
				use_normalization);

			SG_FREE(shifts);
		}

		SG_FREE(dtype);
	}
	else if (strmatch(type, "WEIGHTEDDEGREEPOS3"))
	{
		if (m_nrhs<7)
			return NULL;

		char* dtype=get_str_from_str_or_direct(len);
		if (strmatch(dtype, "CHAR") || strmatch(dtype, "STRING"))
		{
			int32_t size=get_int_from_int_or_str();
			int32_t order=get_int_from_int_or_str();
			int32_t max_mismatch=get_int_from_int_or_str();
			int32_t length=get_int_from_int_or_str();
			int32_t mkl_stepsize=get_int_from_int_or_str();
			int32_t* shifts=NULL;
			int32_t l=0;
			get_vector_from_int_vector_or_str(shifts, l);
			ASSERT(l==length)

			float64_t* position_weights=NULL;
			if (m_nrhs>9+length)
			{
				get_vector_from_real_vector_or_str(
					position_weights, length);
			}

			kernel=ui_kernel->create_weighteddegreepositionstring3(
				size, order, max_mismatch, shifts, length,
				mkl_stepsize, position_weights);

			SG_FREE(position_weights);
			SG_FREE(shifts);
		}

		SG_FREE(dtype);
	}
	else if (strmatch(type, "WEIGHTEDDEGREEPOS"))
	{
		if (m_nrhs<4)
			return NULL;

		char* dtype=get_str_from_str_or_direct(len);
		if (strmatch(dtype, "CHAR") || strmatch(dtype, "STRING"))
		{
			int32_t size=get_int_from_int_or_str();
			int32_t order=3;
			int32_t max_mismatch=0;
			int32_t length=0;
			int32_t center=0;
			float64_t step=1;

			if (m_nrhs>4)
			{
				order=get_int_from_int_or_str();

				if (m_nrhs>5)
				{
					max_mismatch=get_int_from_int_or_str();

					if (m_nrhs>6)
					{
						length=get_int_from_int_or_str();

						if (m_nrhs>7)
						{
							center=get_int_from_int_or_str();

							if (m_nrhs>8)
								step=get_real_from_real_or_str();
						}
					}
				}
			}

			kernel=ui_kernel->create_weighteddegreepositionstring(
				size, order, max_mismatch, length, center, step);
		}

		SG_FREE(dtype);
	}
	else if (strmatch(type, "WEIGHTEDDEGREE"))
	{
		if (m_nrhs<4)
			return NULL;

		char* dtype=get_str_from_str_or_direct(len);
		if (strmatch(dtype, "CHAR") || strmatch(dtype, "STRING"))
		{
			int32_t size=get_int_from_int_or_str();
			int32_t order=3;
			int32_t max_mismatch=0;
			bool use_normalization=true;
			int32_t mkl_stepsize=1;
			bool block_computation=true;
			int32_t single_degree=-1;

			if (m_nrhs>4)
			{
				order=get_int_from_int_or_str();

				if (m_nrhs>5)
				{
					max_mismatch=get_int_from_int_or_str();

					if (m_nrhs>6)
					{
						use_normalization=get_bool_from_bool_or_str();

						if (m_nrhs>7)
						{
							mkl_stepsize=get_int_from_int_or_str();

							if (m_nrhs>8)
							{
								block_computation=get_int_from_int_or_str();

								if (m_nrhs>9)
									single_degree=get_int_from_int_or_str();
							}
						}
					}
				}
			}

			kernel=ui_kernel->create_weighteddegreestring(
				size, order, max_mismatch, use_normalization,
				mkl_stepsize, block_computation, single_degree);
		}

		SG_FREE(dtype);
	}
	else if (strmatch(type, "WEIGHTEDDEGREERBF"))
	{
		if (m_nrhs<5)
			return NULL;

		char* dtype=get_str_from_str_or_direct(len);
		int32_t size=get_int_from_int_or_str();
		int32_t nof_properties=get_int_from_int_or_str();
		int32_t degree=1;
		float64_t width=1;
		if (m_nrhs>5)
		{
			degree=get_int_from_int_or_str();
			if (m_nrhs>6)
			{
				width=get_real_from_real_or_str();
			}

		}
		//if (strmatch(dtype, "REAL"))

		kernel=ui_kernel->create_weighteddegreerbf(size, degree, nof_properties, width);

		SG_FREE(dtype);

	}
	else if (strmatch(type, "SPECTRUMMISMATCHRBF"))
	{
		if (m_nrhs<7)
			return NULL;

		char* dtype=get_str_from_str_or_direct(len);
		if (strmatch(dtype, "CHAR") || strmatch(dtype, "STRING"))
		{
			int32_t size=get_int_from_int_or_str();
			int32_t degree=get_int_from_int_or_str();
			int32_t max_mismatch=get_int_from_int_or_str();
			float64_t width=get_real_from_real_or_str();
			float64_t* AA_matrix = NULL;

			//int32_t length=128*128;
			//get_vector_from_real_vector_or_str(AA_matrix, length);
			float64_t* helper_matrix=NULL;
			int32_t N=0;
			int32_t M=0;
			get_matrix(helper_matrix, N, M);

			if (N == 128 && M == 128)
			{
				AA_matrix=SG_MALLOC(float64_t, N*M);
				memcpy(AA_matrix, helper_matrix, N*M*sizeof(float64_t)) ;
				kernel=ui_kernel->create_spectrummismatchrbf(size, AA_matrix, 128, 128, max_mismatch, degree, width);
			}
			else
			{
				SG_ERROR("Matrix size %d %d\n", N, M)
			}
		}
		SG_FREE(dtype);

	}

	else if (strmatch(type, "SLIK") || strmatch(type, "LIK"))
	{
		if (m_nrhs<4)
			return NULL;

		char* dtype=get_str_from_str_or_direct(len);
		if (strmatch(dtype, "CHAR"))
		{
			int32_t size=get_int_from_int_or_str();
			int32_t length=3;
			int32_t inner_degree=3;
			int32_t outer_degree=1;

			if (m_nrhs>4)
			{
				length=get_int_from_int_or_str();

				if (m_nrhs>5)
				{
					inner_degree=get_int_from_int_or_str();

					if (m_nrhs>6)
						outer_degree=get_int_from_int_or_str();
				}
			}

			if (strmatch(type, "SLIK"))
			{
				kernel=ui_kernel->create_localityimprovedstring(
					size, length, inner_degree, outer_degree,
					K_SIMPLELOCALITYIMPROVED);
			}
			else
			{
				kernel=ui_kernel->create_localityimprovedstring(
					size, length, inner_degree, outer_degree,
					K_LOCALITYIMPROVED);
			}
		}

		SG_FREE(dtype);
	}
	else if (strmatch(type, "POLY"))
	{
		if (m_nrhs<4)
			return NULL;

		char* dtype=get_str_from_str_or_direct(len);
		int32_t size=get_int_from_int_or_str();
		int32_t degree=2;
		bool inhomogene=false;
		bool normalize=true;

		if (m_nrhs>4)
		{
			degree=get_int_from_int_or_str();

			if (m_nrhs>5)
			{
				inhomogene=get_bool_from_bool_or_str();

				if (m_nrhs>6)
					normalize=get_bool_from_bool_or_str();
			}
		}

		if (strmatch(dtype, "REAL"))
		{
			kernel=ui_kernel->create_poly(
				size, degree, inhomogene, normalize);
		}
		else if (strmatch(dtype, "SPARSEREAL"))
		{
			kernel=ui_kernel->create_sparsepoly(
				size, degree, inhomogene, normalize);
		}

		SG_FREE(dtype);
	}
	else if (strmatch(type, "SIGMOID"))
	{
		if (m_nrhs<4)
			return NULL;

		char* dtype=get_str_from_str_or_direct(len);
		if (strmatch(dtype, "REAL"))
		{
			int32_t size=get_int_from_int_or_str();
			float64_t gamma=0.01;
			float64_t coef0=0;

			if (m_nrhs>4)
			{
				gamma=get_real_from_real_or_str();

				if (m_nrhs>5)
					coef0=get_real_from_real_or_str();
			}

			kernel=ui_kernel->create_sigmoid(size, gamma, coef0);
		}

		SG_FREE(dtype);
	}
	else if (strmatch(type, "GAUSSIAN")) // RBF
	{
		if (m_nrhs<4)
			return NULL;

		char* dtype=get_str_from_str_or_direct(len);
		int32_t size=get_int_from_int_or_str();
		float64_t width=1;
		if (m_nrhs>4)
			width=get_real_from_real_or_str();

		if (strmatch(dtype, "REAL"))
			kernel=ui_kernel->create_gaussian(size, width);
		else if (strmatch(dtype, "SPARSEREAL"))
			kernel=ui_kernel->create_sparsegaussian(size, width);

		SG_FREE(dtype);
	}
	else if (strmatch(type, "GAUSSIANSHIFT")) // RBF
	{
		if (m_nrhs<7)
			return NULL;

		char* dtype=get_str_from_str_or_direct(len);
		if (strmatch(dtype, "REAL"))
		{
			int32_t size=get_int_from_int_or_str();
			float64_t width=get_real_from_real_or_str();
			int32_t max_shift=get_int_from_int_or_str();
			int32_t shift_step=get_int_from_int_or_str();

			kernel=ui_kernel->create_gaussianshift(
				size, width, max_shift, shift_step);
		}

		SG_FREE(dtype);
	}
	else if (strmatch(type, "CUSTOM"))
	{
		if (m_nrhs!=4 || !create_return_values(0))
			return NULL;

		float64_t* kmatrix=NULL;
		int32_t num_feat=0;
		int32_t num_vec=0;
		get_matrix(kmatrix, num_feat, num_vec);

		int32_t tlen=0;
		char* ktype=get_string(tlen);

		if (!strmatch(ktype, "DIAG") &&
				!strmatch(ktype, "FULL") &&
				!strmatch(ktype, "FULL2DIAG"))
		{
			SG_FREE(ktype);
			SG_ERROR("Undefined type, not DIAG, FULL or FULL2DIAG.\n")
		}

		bool source_is_diag=false;
		bool dest_is_diag=false;

		if (strmatch(ktype, "FULL2DIAG"))
			dest_is_diag=true;
		else if (strmatch(ktype, "DIAG"))
		{
			source_is_diag=true;
			dest_is_diag=true;
		}

		kernel=ui_kernel->create_custom(kmatrix, num_feat, num_vec,
				source_is_diag, dest_is_diag);
	}
	else if (strmatch(type, "CONST"))
	{
		if (m_nrhs<4)
			return NULL;

		char* dtype=get_str_from_str_or_direct(len);
		if (strmatch(dtype, "REAL"))
		{
			int32_t size=get_int_from_int_or_str();
			float64_t c=1;
			if (m_nrhs>4)
				c=get_real_from_real_or_str();

			kernel=ui_kernel->create_const(size, c);
		}

		SG_FREE(dtype);
	}
	else if (strmatch(type, "DIAG"))
	{
		if (m_nrhs<4)
			return NULL;

		char* dtype=get_str_from_str_or_direct(len);
		if (strmatch(dtype, "REAL"))
		{
			int32_t size=get_int_from_int_or_str();
			float64_t diag=1;
			if (m_nrhs>4)
				diag=get_real_from_real_or_str();

			kernel=ui_kernel->create_diag(size, diag);
		}

		SG_FREE(dtype);
	}

	else if (strmatch(type, "TPPK"))
	{
		if (m_nrhs!=5)
			return NULL;

		char* dtype=get_str_from_str_or_direct(len);
		if (strmatch(dtype, "INT"))
		{
			int32_t size=get_int_from_int_or_str();
			float64_t* km=NULL;
			int32_t rows=0;
			int32_t cols=0;
			get_matrix(km, rows, cols);
			kernel=ui_kernel->create_tppk(size, km, rows, cols);
		}

		SG_FREE(dtype);
	}
	else
		SG_NOTIMPLEMENTED

	SG_FREE(type);
	SG_DEBUG("created kernel: %p\n", kernel)
	return kernel;
}


CFeatures* CSGInterface::create_custom_string_features(CStringFeatures<uint8_t>* orig_feat)
{
	CFeatures* feat=orig_feat;

	if (m_nrhs>4)
	{
		int32_t start=-1;
		int32_t order=0;
		int32_t from_order=0;
		bool normalize=true;

		int32_t feature_class_len=0;
		char* feature_class_str=get_string(feature_class_len);
		ASSERT(feature_class_str)
		CAlphabet* alphabet=NULL;
		if (strmatch(feature_class_str, "WD"))
		{
			if (m_nrhs!=7)
				SG_ERROR("Please specify alphabet, WD, order, from_order\n")

			alphabet=new CAlphabet(RAWDNA);
			order=get_int();
			from_order=get_int();
			feat = new CWDFeatures((CStringFeatures<uint8_t>*) feat, order, from_order);
		}
		else if (strmatch(feature_class_str, "WSPEC"))
		{
			if (m_nrhs!=8)
				SG_ERROR("Please specify alphabet, order, WSPEC, start, normalize\n")

			alphabet=new CAlphabet(RAWDNA);
			order=get_int();
			start=get_int();
			normalize=get_bool();
			CStringFeatures<uint16_t>* sf=new CStringFeatures<uint16_t>(RAWDNA);
			sf->obtain_from_char_features((CStringFeatures<uint8_t>*) feat, start, order, 0, normalize);
			sf->add_preprocessor(new CSortWordString());
			sf->apply_preprocessor();
			SG_UNREF(feat);
			feat = new CImplicitWeightedSpecFeatures(sf, normalize);
		}
		SG_FREE(feature_class_str);

		SG_UNREF(alphabet);
	}

	return feat;
}

CFeatures* CSGInterface::create_custom_real_features(CDenseFeatures<float64_t>* orig_feat)
{
	CFeatures* feat=orig_feat;

	if (m_nrhs==6)
	{
		int32_t degree=0;
		int32_t feature_class_len=0;
		bool normalize;
		char* feature_class_str=get_string(feature_class_len);
		ASSERT(feature_class_str)
		if (strmatch(feature_class_str, "POLY"))
		{
			//if (m_nrhs!=7)
			//	SG_ERROR("Please specify POLY, degree\n")

			degree=get_int();
			normalize = get_bool();
			feat = new CPolyFeatures((CDenseFeatures<float64_t>*) feat, degree, normalize);

		}
		else
			SG_ERROR("Unknown feature class: %s\n", feature_class_str)

		SG_FREE(feature_class_str);
	}

	return feat;
}

bool CSGInterface::cmd_init_kernel()
{
	SG_DEPRECATED
	return true;
}

bool CSGInterface::cmd_clean_kernel()
{
	if (m_nrhs<1 || !create_return_values(0))
		return false;

	return ui_kernel->clean_kernel();
}

bool CSGInterface::cmd_save_kernel()
{
	if (m_nrhs<2 || !create_return_values(0))
		return false;

	int32_t len=0;
	char* filename=get_str_from_str_or_direct(len);

	bool success=ui_kernel->save_kernel(filename);

	SG_FREE(filename);
	return success;
}

bool CSGInterface::cmd_get_kernel_matrix()
{
	if (m_nrhs>2 || !create_return_values(1))
		return false;

	int32_t len=0;
	char* target=NULL;

	if (m_nrhs==2)
		target=get_string(len);
	bool success=ui_kernel->init_kernel(target);

	if (success)
	{
		CKernel* kernel=ui_kernel->get_kernel();
		if (!kernel || !kernel->has_features())
			SG_ERROR("No kernel defined or not initialized.\n")

		SGMatrix<float64_t> km=kernel->get_kernel_matrix<float64_t>();
		set_matrix(km.matrix, km.num_rows, km.num_cols);
	}

	SG_FREE(target);

	return success;
}

bool CSGInterface::cmd_set_WD_position_weights()
{
	if (m_nrhs<2 || m_nrhs>3 || !create_return_values(0))
		return false;

	CKernel* kernel=ui_kernel->get_kernel();
	if (!kernel)
		SG_ERROR("No kernel.\n")

	if (kernel->get_kernel_type()==K_COMBINED)
	{
		kernel=((CCombinedKernel*) kernel)->get_last_kernel();
		if (!kernel)
			SG_ERROR("No last kernel.\n")

		EKernelType ktype=kernel->get_kernel_type();
		if (ktype!=K_WEIGHTEDDEGREE && ktype!=K_WEIGHTEDDEGREEPOS)
			SG_ERROR("Unsupported kernel.\n")
	}

	bool success=false;
	float64_t* weights=NULL;
	int32_t dim=0;
	int32_t len=0;
	get_matrix(weights, dim, len);

	if (kernel->get_kernel_type()==K_WEIGHTEDDEGREE)
	{
		CWeightedDegreeStringKernel* k=
			(CWeightedDegreeStringKernel*) kernel;

		if (dim!=1 && len>0)
			SG_ERROR("Dimension mismatch (should be 1 x seq_length or 0x0\n")

		ui_kernel->init_kernel("TRAIN");
		success=k->set_position_weights(weights, len);
	}
	else
	{
		CWeightedDegreePositionStringKernel* k=
			(CWeightedDegreePositionStringKernel*) kernel;
		char* target=NULL;
		bool is_train=true;

		if (m_nrhs==3)
		{
			int32_t tlen=0;
			target=get_string(tlen);
			if (!target)
			{
				SG_FREE(weights);
				SG_ERROR("Couldn't find second argument to method.\n")
			}

			if (!strmatch(target, "TRAIN") && !strmatch(target, "TEST"))
			{
				SG_FREE(target);
				SG_ERROR("Second argument none of TRAIN or TEST.\n")
			}

			if (strmatch(target, "TEST"))
				is_train=false;
		}

		if (dim!=1 && len>0)
		{
			SG_FREE(target);
			SG_ERROR("Dimension mismatch (should be 1 x seq_length or 0x0\n")
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
			{
				ui_kernel->init_kernel("TRAIN");
				k->set_position_weights(SGVector<float64_t>(weights, len));
				success=true;
			}
		}

		SG_FREE(target);
	}

	return success;
}

bool CSGInterface::cmd_get_subkernel_weights()
{
	if (m_nrhs!=1 || !create_return_values(1))
		return false;

	CKernel *kernel=ui_kernel->get_kernel();
	if (!kernel)
		SG_ERROR("Invalid kernel.\n")

	EKernelType ktype=kernel->get_kernel_type();
	const float64_t* weights=NULL;

	if (ktype==K_COMBINED)
	{
		int32_t num_weights=-1;
		weights=((CCombinedKernel *) kernel)->get_subkernel_weights(num_weights);

		// matrices of shape 1 x num_weight are returned
		set_matrix(weights, 1, num_weights);
		return true;
	}

	int32_t degree=-1;
	int32_t length=-1;

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
		SG_ERROR("Setting subkernel weights not supported on this kernel.\n")

	if (length==0)
		length=1;

	set_matrix(weights, degree, length);
	return true;
}

bool CSGInterface::cmd_set_subkernel_weights()
{
	if (m_nrhs!=2 || !create_return_values(0))
		return false;

	CKernel* kernel=ui_kernel->get_kernel();
	if (!kernel)
		SG_ERROR("No kernel.\n")

	bool success=false;
	float64_t* weights=NULL;
	int32_t dim=0;
	int32_t len=0;
	get_matrix(weights, dim, len);

	EKernelType ktype=kernel->get_kernel_type();
	if (ktype==K_WEIGHTEDDEGREE)
	{
		CWeightedDegreeStringKernel* k=
			(CWeightedDegreeStringKernel*) kernel;
		int32_t degree=k->get_degree();
		if (dim!=degree || len<1)
			SG_ERROR("WD: Dimension mismatch (should be (seq_length | 1) x degree) got (%d x %d)\n", len, degree)

		if (len==1)
			len=0;

		success=k->set_weights(SGMatrix<float64_t>(weights, dim, len));
	}
	else if (ktype==K_WEIGHTEDDEGREEPOS)
	{
		CWeightedDegreePositionStringKernel* k=
			(CWeightedDegreePositionStringKernel*) kernel;
		int32_t degree=k->get_degree();
		if (dim!=degree || len<1)
			SG_ERROR("WDPos: Dimension mismatch (should be (seq_length | 1) x degree) got (%d x %d)\n", len, degree)

		if (len==1)
			len=0;

		success=k->set_weights(SGMatrix<float64_t>(weights, dim, len));
	}
	else // all other kernels
	{
		int32_t num_subkernels=kernel->get_num_subkernels();
		if (dim!=1 || len!=num_subkernels)
			SG_ERROR("All: Dimension mismatch (should be 1 x num_subkernels)\n")

		kernel->set_subkernel_weights(SGVector<float64_t>(weights, len));
		success=true;
	}

	return success;
}

bool CSGInterface::cmd_set_subkernel_weights_combined()
{
	if (m_nrhs!=3 || !create_return_values(0))
		return false;

	CKernel* kernel=ui_kernel->get_kernel();
	if (!kernel)
		SG_ERROR("No kernel.\n")
	if (kernel->get_kernel_type()!=K_COMBINED)
		SG_ERROR("Only works for combined kernels.\n")

	bool success=false;
	float64_t* weights=NULL;
	int32_t dim=0;
	int32_t len=0;
	get_matrix(weights, dim, len);

	int32_t idx=get_int();
	SG_DEBUG("using kernel_idx=%i\n", idx)

	kernel=((CCombinedKernel*) kernel)->get_kernel(idx);
	if (!kernel)
		SG_ERROR("No subkernel at idx %d.\n", idx)

	EKernelType ktype=kernel->get_kernel_type();
	if (ktype==K_WEIGHTEDDEGREE)
	{
		CWeightedDegreeStringKernel* k=
			(CWeightedDegreeStringKernel*) kernel;
		int32_t degree=k->get_degree();
		if (dim!=degree || len<1)
			SG_ERROR("Dimension mismatch (should be de(seq_length | 1) x degree)\n")

		if (len==1)
			len=0;

		success=k->set_weights(SGMatrix<float64_t>(weights, dim, len));
	}
	else if (ktype==K_WEIGHTEDDEGREEPOS)
	{
		CWeightedDegreePositionStringKernel* k=
			(CWeightedDegreePositionStringKernel*) kernel;
		int32_t degree=k->get_degree();
		if (dim!=degree || len<1)
			SG_ERROR("Dimension mismatch (should be de(seq_length | 1) x degree)\n")

		if (len==1)
			len=0;

		success=k->set_weights(SGMatrix<float64_t>(weights, dim, len));
	}
	else // all other kernels
	{
		int32_t num_subkernels=kernel->get_num_subkernels();
		if (dim!=1 || len!=num_subkernels)
			SG_ERROR("Dimension mismatch (should be 1 x num_subkernels)\n")

		kernel->set_subkernel_weights(SGVector<float64_t>(weights, len));
		success=true;
	}

	return success;
}

bool CSGInterface::cmd_get_dotfeature_weights_combined()
{
	if (m_nrhs!=2 || !create_return_values(1))
		return false;

	int32_t tlen=0;
	char* target=get_string(tlen);
	CFeatures* features=NULL;

	if (strmatch(target, "TRAIN"))
		features=ui_features->get_train_features();
	else if (strmatch(target, "TEST"))
		features=ui_features->get_test_features();
	else
	{
		SG_FREE(target);
		SG_ERROR("Unknown target, neither TRAIN nor TEST.\n")
	}
	SG_FREE(target);

	if (!features)
		SG_ERROR("No features.\n")
	if (features->get_feature_class()!=C_COMBINED_DOT)
		SG_ERROR("Only works for combined dot features.\n")

	SGVector<float64_t> weights = ((CCombinedDotFeatures*) features)->get_subfeature_weights();
	set_vector(weights.vector, weights.vlen);

	return true;
}

bool CSGInterface::cmd_set_dotfeature_weights_combined()
{
	if (m_nrhs!=3 || !create_return_values(0))
		return false;

	int32_t tlen=0;
	char* target=get_string(tlen);
	CFeatures* features=NULL;

	if (strmatch(target, "TRAIN"))
		features=ui_features->get_train_features();
	else if (strmatch(target, "TEST"))
		features=ui_features->get_test_features();
	else
	{
		SG_FREE(target);
		SG_ERROR("Unknown target, neither TRAIN nor TEST.\n")
	}
	SG_FREE(target);

	if (!features)
		SG_ERROR("No features.\n")
	if (features->get_feature_class()!=C_COMBINED_DOT)
		SG_ERROR("Only works for combined dot features.\n")

	float64_t* weights=NULL;
	int32_t dim=0;
	int32_t len=0;
	get_matrix(weights, dim, len);

	((CCombinedDotFeatures*) features)->set_subfeature_weights(SGVector<float64_t>(weights, len));

	return true;
}

bool CSGInterface::cmd_set_last_subkernel_weights()
{
	if (m_nrhs!=2 || !create_return_values(0))
		return false;

	CKernel* kernel=ui_kernel->get_kernel();
	if (!kernel)
		SG_ERROR("No kernel.\n")
	if (kernel->get_kernel_type()!=K_COMBINED)
		SG_ERROR("Only works for Combined kernels.\n")

	kernel=((CCombinedKernel*) kernel)->get_last_kernel();
	if (!kernel)
		SG_ERROR("No last kernel.\n")

	bool success=false;
	float64_t* weights=NULL;
	int32_t dim=0;
	int32_t len=0;
	get_matrix(weights, dim, len);

	EKernelType ktype=kernel->get_kernel_type();
	if (ktype==K_WEIGHTEDDEGREE)
	{
		CWeightedDegreeStringKernel* k=(CWeightedDegreeStringKernel*) kernel;
		if (dim!=k->get_degree() || len<1)
			SG_ERROR("Dimension mismatch (should be de(seq_length | 1) x degree)\n")

		if (len==1)
			len=0;

		success=k->set_weights(SGMatrix<float64_t>(weights, dim, len));
	}
	else if (ktype==K_WEIGHTEDDEGREEPOS)
	{
		CWeightedDegreePositionStringKernel* k=
			(CWeightedDegreePositionStringKernel*) kernel;
		if (dim!=k->get_degree() || len<1)
			SG_ERROR("Dimension mismatch (should be de(seq_length | 1) x degree)\n")

		if (len==1)
			len=0;

		success=k->set_weights(SGMatrix<float64_t>(weights, dim, len));
	}
	else // all other kernels
	{
		int32_t num_subkernels=kernel->get_num_subkernels();
		if (dim!=1 || len!=num_subkernels)
			SG_ERROR("Dimension mismatch (should be 1 x num_subkernels)\n")

		kernel->set_subkernel_weights(SGVector<float64_t>(weights, len));
		success=true;
	}

	return success;
}

bool CSGInterface::cmd_get_WD_position_weights()
{
	if (m_nrhs!=1 || !create_return_values(1))
		return false;

	CKernel* kernel=ui_kernel->get_kernel();
	if (!kernel)
		SG_ERROR("No kernel.\n")

	if (kernel->get_kernel_type()==K_COMBINED)
	{
		kernel=((CCombinedKernel*) kernel)->get_last_kernel();
		if (!kernel)
			SG_ERROR("Couldn't find last kernel.\n")

		EKernelType ktype=kernel->get_kernel_type();
		if (ktype!=K_WEIGHTEDDEGREE && ktype!=K_WEIGHTEDDEGREEPOS)
			SG_ERROR("Wrong subkernel type.\n")
	}

	int32_t len=0;
	const float64_t* position_weights;

	if (kernel->get_kernel_type()==K_WEIGHTEDDEGREE)
		position_weights=((CWeightedDegreeStringKernel*) kernel)->get_position_weights(len);
	else
		position_weights=((CWeightedDegreePositionStringKernel*) kernel)->get_position_weights(len);

	if (position_weights==NULL)
		set_vector(position_weights, 0);
	else
		set_vector(position_weights, len);

	return true;
}

bool CSGInterface::cmd_get_last_subkernel_weights()
{
	if (m_nrhs!=1 || !create_return_values(1))
		return false;

	CKernel* kernel=ui_kernel->get_kernel();
	EKernelType ktype=kernel->get_kernel_type();
	if (!kernel)
		SG_ERROR("No kernel.\n")
	if (ktype!=K_COMBINED)
		SG_ERROR("Only works for Combined kernels.\n")

	kernel=((CCombinedKernel*) kernel)->get_last_kernel();
	if (!kernel)
		SG_ERROR("Couldn't find last kernel.\n")

	int32_t degree=0;
	int32_t len=0;

	if (ktype==K_COMBINED)
	{
		int32_t num_weights=0;
		const float64_t* weights=
			((CCombinedKernel*) kernel)->get_subkernel_weights(num_weights);

		set_vector(weights, num_weights);
		return true;
	}

	float64_t* weights=NULL;
	if (ktype==K_WEIGHTEDDEGREE)
		weights=((CWeightedDegreeStringKernel*) kernel)->
			get_degree_weights(degree, len);
	else if (ktype==K_WEIGHTEDDEGREEPOS)
		weights=((CWeightedDegreePositionStringKernel*) kernel)->
			get_degree_weights(degree, len);
	else
		SG_ERROR("Only works for Weighted Degree (Position) kernels.\n")

	if (len==0)
		len=1;

	set_matrix(weights, degree, len);

	return true;
}

bool CSGInterface::cmd_compute_by_subkernels()
{
	if (m_nrhs!=1 || !create_return_values(1))
		return false;

	CKernel* kernel=ui_kernel->get_kernel();
	if (!kernel)
		SG_ERROR("No kernel.\n")
	if (!kernel->get_rhs())
		SG_ERROR("No rhs.\n")

	int32_t num_vec=kernel->get_rhs()->get_num_vectors();
	int32_t degree=0;
	int32_t len=0;
	EKernelType ktype=kernel->get_kernel_type();

	// it would be nice to have a common base class for the WD kernels
	if (ktype==K_WEIGHTEDDEGREE)
	{
		CWeightedDegreeStringKernel* k=(CWeightedDegreeStringKernel*) kernel;
		k->get_degree_weights(degree, len);
		if (!k->is_tree_initialized())
			SG_ERROR("Kernel optimization not initialized.\n")
	}
	else if (ktype==K_WEIGHTEDDEGREEPOS)
	{
		CWeightedDegreePositionStringKernel* k=
			(CWeightedDegreePositionStringKernel*) kernel;
		k->get_degree_weights(degree, len);
		if (!k->is_tree_initialized())
			SG_ERROR("Kernel optimization not initialized.\n")
	}
	else
		SG_ERROR("Only works for Weighted Degree (Position) kernels.\n")

	if (len==0)
		len=1;

	int32_t num_feat=degree*len;
	int32_t num=num_feat*num_vec;
	float64_t* result=SG_MALLOC(float64_t, num);

	for (int32_t i=0; i<num; i++)
		result[i]=0;

	if (ktype==K_WEIGHTEDDEGREE)
	{
		CWeightedDegreeStringKernel* k=(CWeightedDegreeStringKernel*) kernel;
		for (int32_t i=0; i<num_vec; i++)
			k->compute_by_tree(i, &result[i*num_feat]);
	}
	else
	{
		CWeightedDegreePositionStringKernel* k=
			(CWeightedDegreePositionStringKernel*) kernel;
		for (int32_t i=0; i<num_vec; i++)
			k->compute_by_tree(i, &result[i*num_feat]);
	}

	set_matrix(result, num_feat, num_vec);
	SG_FREE(result);

	return true;
}

bool CSGInterface::cmd_init_kernel_optimization()
{
	if (m_nrhs<1 || !create_return_values(0))
		return false;

	return ui_kernel->init_kernel_optimization();
}

bool CSGInterface::cmd_get_kernel_optimization()
{
	if (m_nrhs<1 || !create_return_values(1))
		return false;

	CKernel* kernel=ui_kernel->get_kernel();
	if (!kernel)
		SG_ERROR("No kernel defined.\n")

	switch (kernel->get_kernel_type())
	{
		case K_WEIGHTEDDEGREEPOS:
		{
			if (m_nrhs!=2)
				SG_ERROR("parameter missing\n")

			int32_t max_order=get_int();
			if ((max_order<1) || (max_order>12))
			{
				SG_WARNING("max_order out of range 1..12 (%d). setting to 1\n", max_order)
				max_order=1;
			}

			CWeightedDegreePositionStringKernel* k=(CWeightedDegreePositionStringKernel*) kernel;
			CSVM* svm=(CSVM*) ui_classifier->get_classifier();
			if (!svm)
				SG_ERROR("No SVM defined.\n")

			int32_t num_suppvec=svm->get_num_support_vectors();
			int32_t* sv_idx=SG_MALLOC(int32_t, num_suppvec);
			float64_t* sv_weight=SG_MALLOC(float64_t, num_suppvec);
			int32_t num_feat=0;
			int32_t num_sym=0;

			for (int32_t i=0; i<num_suppvec; i++)
			{
				sv_idx[i]=svm->get_support_vector(i);
				sv_weight[i]=svm->get_alpha(i);
			}

			float64_t* position_weights=k->extract_w(max_order, num_feat,
				num_sym, NULL, num_suppvec, sv_idx, sv_weight);
			SG_FREE(sv_idx);
			SG_FREE(sv_weight);

			set_matrix(position_weights, num_sym, num_feat);
			SG_FREE(position_weights);

			return true;
		}

		case K_COMMWORDSTRING:
		case K_WEIGHTEDCOMMWORDSTRING:
		{
			CCommWordStringKernel* k=(CCommWordStringKernel*) kernel;
			int32_t len=0;
			float64_t* weights;
			k->get_dictionary(len, weights);

			set_vector(weights, len);
			return true;
		}
		case K_LINEAR:
		{
			CLinearKernel* k=(CLinearKernel*) kernel;
			SGVector<float64_t> weights=k->get_w();

			set_vector(weights.vector, weights.size());
			return true;
		}
		default:
			SG_ERROR("Unsupported kernel %s.\n", kernel->get_name())
	}

	return true;
}

bool CSGInterface::cmd_delete_kernel_optimization()
{
	if (m_nrhs<1 || !create_return_values(0))
		return false;

	return ui_kernel->delete_kernel_optimization();
}

bool CSGInterface::cmd_use_diagonal_speedup()
{
	if (m_nrhs<2 || !create_return_values(0))
		return false;

	bool speedup=get_bool();

	CKernel* kernel=ui_kernel->get_kernel();
	if (!kernel)
		SG_ERROR("No kernel defined.\n")

	if (kernel->get_kernel_type()==K_COMBINED)
	{
		SG_DEBUG("Identified combined kernel.\n")
		kernel=((CCombinedKernel*) kernel)->get_last_kernel();
		if (!kernel)
			SG_ERROR("No last kernel defined.\n")
	}

	if (kernel->get_kernel_type()!=K_COMMWORDSTRING)
		SG_ERROR("Currently only commwordstring kernel supports diagonal speedup\n")

	((CCommWordStringKernel*) kernel)->set_use_dict_diagonal_optimization(speedup);

	SG_INFO("Diagonal speedup %s.\n", speedup ? "enabled" : "disabled")

	return true;
}

bool CSGInterface::cmd_set_kernel_optimization_type()
{
	if (m_nrhs<2 || !create_return_values(0))
		return false;

	int32_t len=0;
	char* opt_type=get_str_from_str_or_direct(len);

	bool success=ui_kernel->set_optimization_type(opt_type);

	SG_FREE(opt_type);
	return success;
}

bool CSGInterface::cmd_set_solver()
{
	if (m_nrhs<2 || !create_return_values(0))
		return false;

	int32_t len=0;
	char* solver=get_str_from_str_or_direct(len);

	bool success=ui_classifier->set_solver(solver);

	SG_FREE(solver);
	return success;
}

bool CSGInterface::cmd_set_constraint_generator()
{
	if (m_nrhs<2 || !create_return_values(0))
		return false;

	int32_t len=0;
	char* cg=get_str_from_str_or_direct(len);

	bool success=ui_classifier->set_constraint_generator(cg);

	SG_FREE(cg);
	return success;
}

bool CSGInterface::cmd_set_prior_probs()
{
	if (m_nrhs<3 || !create_return_values(0))
		return false;

	CSalzbergWordStringKernel* kernel=
		(CSalzbergWordStringKernel*) ui_kernel->get_kernel();
	if (kernel->get_kernel_type()!=K_SALZBERG)
		SG_ERROR("SalzbergWordStringKernel required for setting prior probs!\n")

	float64_t pos_probs=get_real_from_real_or_str();
	float64_t neg_probs=get_real_from_real_or_str();

	kernel->set_prior_probs(pos_probs, neg_probs);

	return true;
}

bool CSGInterface::cmd_set_prior_probs_from_labels()
{
	if (m_nrhs<2 || !create_return_values(0))
		return false;

	CSalzbergWordStringKernel* kernel=
		(CSalzbergWordStringKernel*) ui_kernel->get_kernel();
	if (kernel->get_kernel_type()!=K_SALZBERG)
	SG_ERROR("SalzbergWordStringKernel required for setting prior probs!\n")

	float64_t* lab=NULL;
	int32_t len=0;
	get_vector(lab, len);

	CBinaryLabels* labels=new CBinaryLabels(len);
	for (int32_t i=0; i<len; i++)
	{
		if (!labels->set_label(i, lab[i]))
			SG_ERROR("Couldn't set label %d (of %d): %f.\n", i, len, lab[i])
	}
	SG_FREE(lab);

	kernel->set_prior_probs_from_labels(labels);

	SG_UNREF(labels);
	return true;
}

#ifdef USE_SVMLIGHT
bool CSGInterface::cmd_resize_kernel_cache()
{
	if (m_nrhs<2 || !create_return_values(0))
		return false;

	int32_t size=get_int_from_int_or_str();
	return ui_kernel->resize_kernel_cache(size);
}
#endif //USE_SVMLIGHT


/** Distance */

bool CSGInterface::cmd_set_distance()
{
	if (m_nrhs<3 || !create_return_values(0))
		return false;

	CDistance* distance=NULL;
	int32_t len=0;
	char* type=get_str_from_str_or_direct(len);
	char* dtype=get_str_from_str_or_direct(len);

	if (strmatch(type, "MINKOWSKI") && m_nrhs==4)
	{
		float64_t k=get_real_from_real_or_str();
		distance=ui_distance->create_minkowski(k);
	}
	else if (strmatch(type, "MANHATTAN"))
	{
		if (strmatch(dtype, "REAL"))
			distance=ui_distance->create_generic(D_MANHATTAN);
		else if (strmatch(dtype, "WORD"))
			distance=ui_distance->create_generic(D_MANHATTANWORD);
	}
	else if (strmatch(type, "HAMMING") && strmatch(dtype, "WORD"))
	{
		bool use_sign=false;
		if (m_nrhs==4)
			use_sign=get_bool_from_bool_or_str(); // optional

		distance=ui_distance->create_hammingword(use_sign);
	}
	else if (strmatch(type, "CANBERRA"))
	{
		if (strmatch(dtype, "REAL"))
			distance=ui_distance->create_generic(D_CANBERRA);
		else if (strmatch(dtype, "WORD"))
			distance=ui_distance->create_generic(D_CANBERRAWORD);
	}
	else if (strmatch(type, "CHEBYSHEW") && strmatch(dtype, "REAL"))
	{
		distance=ui_distance->create_generic(D_CHEBYSHEW);
	}
	else if (strmatch(type, "GEODESIC") && strmatch(dtype, "REAL"))
	{
		distance=ui_distance->create_generic(D_GEODESIC);
	}
	else if (strmatch(type, "JENSEN") && strmatch(dtype, "REAL"))
	{
		distance=ui_distance->create_generic(D_JENSEN);
	}
	else if (strmatch(type, "CHISQUARE") && strmatch(dtype, "REAL"))
	{
		distance=ui_distance->create_generic(D_CHISQUARE);
	}
	else if (strmatch(type, "TANIMOTO") && strmatch(dtype, "REAL"))
	{
		distance=ui_distance->create_generic(D_TANIMOTO);
	}
	else if (strmatch(type, "COSINE") && strmatch(dtype, "REAL"))
	{
		distance=ui_distance->create_generic(D_COSINE);
	}
	else if (strmatch(type, "BRAYCURTIS") && strmatch(dtype, "REAL"))
	{
		distance=ui_distance->create_generic(D_BRAYCURTIS);
	}
	else if (strmatch(type, "EUCLIDEAN"))
	{
		if (strmatch(dtype, "REAL"))
			distance=ui_distance->create_generic(D_EUCLIDEAN);
		else if (strmatch(dtype, "SPARSEREAL"))
			distance=ui_distance->create_generic(D_SPARSEEUCLIDEAN);
	}
	else
		SG_NOTIMPLEMENTED

	SG_FREE(type);
	SG_FREE(dtype);
	return ui_distance->set_distance(distance);
}

bool CSGInterface::cmd_init_distance()
{
	SG_DEPRECATED
	return true;
}

bool CSGInterface::cmd_get_distance_matrix()
{
	if (m_nrhs!=2 || !create_return_values(1))
		return false;

	int32_t len=0;
	char* target=get_string(len);

	bool success=ui_distance->init_distance(target);

	if (success)
	{
		CDistance* distance=ui_distance->get_distance();
		if (!distance || !distance->has_features())
			SG_ERROR("No distance defined or not initialized.\n")

		SGMatrix<float64_t> dmatrix=distance->get_distance_matrix();
		set_matrix(dmatrix.matrix, dmatrix.num_rows, dmatrix.num_cols);
	}

	return success;
}


/* POIM */

bool CSGInterface::cmd_get_SPEC_consensus()
{
	if (m_nrhs!=1 || !create_return_values(1))
		return false;

	CKernel* kernel=ui_kernel->get_kernel();
	if (!kernel)
		SG_ERROR("No kernel.\n")
	if (kernel->get_kernel_type()!=K_COMMWORDSTRING)
		SG_ERROR("Only works for CommWordString kernels.\n")

	CSVM* svm=(CSVM*) ui_classifier->get_classifier();
	ASSERT(svm)
	int32_t num_suppvec=svm->get_num_support_vectors();
	int32_t* sv_idx=SG_MALLOC(int32_t, num_suppvec);
	float64_t* sv_weight=SG_MALLOC(float64_t, num_suppvec);
	int32_t num_feat=0;

	for (int32_t i=0; i<num_suppvec; i++)
	{
		sv_idx[i]=svm->get_support_vector(i);
		sv_weight[i]=svm->get_alpha(i);
	}

	char* consensus=((CCommWordStringKernel*) kernel)->compute_consensus(
		num_feat, num_suppvec, sv_idx, sv_weight);
	SG_FREE(sv_idx);
	SG_FREE(sv_weight);

	set_vector(consensus, num_feat);
	SG_FREE(consensus);

	return true;
}

bool CSGInterface::cmd_get_SPEC_scoring()
{
	if (m_nrhs!=2 || !create_return_values(1))
		return false;

	int32_t max_order=get_int();
	CKernel* kernel=ui_kernel->get_kernel();
	if (!kernel)
		SG_ERROR("No kernel.\n")

	EKernelType ktype=kernel->get_kernel_type();
	if (ktype!=K_COMMWORDSTRING && ktype!=K_WEIGHTEDCOMMWORDSTRING)
		SG_ERROR("Only works for (Weighted) CommWordString kernels.\n")

	CSVM* svm=(CSVM*) ui_classifier->get_classifier();
	ASSERT(svm)
	int32_t num_suppvec=svm->get_num_support_vectors();
	int32_t* sv_idx=SG_MALLOC(int32_t, num_suppvec);
	float64_t* sv_weight=SG_MALLOC(float64_t, num_suppvec);
	int32_t num_feat=0;
	int32_t num_sym=0;

	for (int32_t i=0; i<num_suppvec; i++)
	{
		sv_idx[i]=svm->get_support_vector(i);
		sv_weight[i]=svm->get_alpha(i);
	}

	if ((max_order<1) || (max_order>8))
	{
		SG_WARNING("max_order out of range 1..8 (%d). setting to 1\n", max_order)
		max_order=1;
	}

	float64_t* position_weights=NULL;
	if (ktype==K_COMMWORDSTRING)
		position_weights=((CCommWordStringKernel*) kernel)->compute_scoring(
			max_order, num_feat, num_sym, NULL,
			num_suppvec, sv_idx, sv_weight);
	else
		position_weights=((CWeightedCommWordStringKernel*) kernel)->compute_scoring(
			max_order, num_feat, num_sym, NULL,
			num_suppvec, sv_idx, sv_weight);
	SG_FREE(sv_idx);
	SG_FREE(sv_weight);

	set_matrix(position_weights, num_sym, num_feat);
	SG_FREE(position_weights);

	return true;
}

bool CSGInterface::cmd_get_WD_consensus()
{
	if (m_nrhs!=1 || !create_return_values(1))
		return false;

	CKernel* kernel=ui_kernel->get_kernel();
	if (!kernel)
		SG_ERROR("No kernel.\n")
	if (kernel->get_kernel_type()!=K_WEIGHTEDDEGREEPOS)
		SG_ERROR("Only works for Weighted Degree Position kernels.\n")

	CSVM* svm=(CSVM*) ui_classifier->get_classifier();
	ASSERT(svm)
	int32_t num_suppvec=svm->get_num_support_vectors();
	int32_t* sv_idx=SG_MALLOC(int32_t, num_suppvec);
	float64_t* sv_weight=SG_MALLOC(float64_t, num_suppvec);
	int32_t num_feat=0;

	for (int32_t i=0; i<num_suppvec; i++)
	{
		sv_idx[i]=svm->get_support_vector(i);
		sv_weight[i]=svm->get_alpha(i);
	}

	char* consensus=((CWeightedDegreePositionStringKernel*) kernel)->compute_consensus(
			num_feat, num_suppvec, sv_idx, sv_weight);
	SG_FREE(sv_idx);
	SG_FREE(sv_weight);

	set_vector(consensus, num_feat);
	SG_FREE(consensus);

	return true;
}

bool CSGInterface::cmd_compute_POIM_WD()
{
	if (m_nrhs!=3 || !create_return_values(1))
		return false;

	int32_t max_order=get_int();
	float64_t* distribution=NULL;
	int32_t num_dfeat=0;
	int32_t num_dvec=0;
	get_matrix(distribution, num_dfeat, num_dvec);

	if (!distribution)
		SG_ERROR("Wrong distribution.\n")

	CKernel* kernel=ui_kernel->get_kernel();
	if (!kernel)
		SG_ERROR("No Kernel.\n")
	if (kernel->get_kernel_type()!=K_WEIGHTEDDEGREEPOS)
		SG_ERROR("Only works for Weighted Degree Position kernels.\n")

	int32_t seqlen=0;
	int32_t num_sym=0;
	CStringFeatures<char>* sfeat=(CStringFeatures<char>*)
		(((CWeightedDegreePositionStringKernel*) kernel)->get_lhs());
	ASSERT(sfeat)
	seqlen=sfeat->get_max_vector_length();
	num_sym=(int32_t) sfeat->get_num_symbols();

	if (num_dvec!=seqlen || num_dfeat!=num_sym)
	{
		SG_ERROR("distribution should have (seqlen x num_sym) elements"
				"(seqlen: %d vs. %d symbols: %d vs. %d)\n", seqlen,
				num_dvec, num_sym, num_dfeat);
	}

		CSVM* svm=(CSVM*) ui_classifier->get_classifier();
		ASSERT(svm)
		int32_t num_suppvec=svm->get_num_support_vectors();
		int32_t* sv_idx=SG_MALLOC(int32_t, num_suppvec);
		float64_t* sv_weight=SG_MALLOC(float64_t, num_suppvec);

		for (int32_t i=0; i<num_suppvec; i++)
		{
			sv_idx[i]=svm->get_support_vector(i);
			sv_weight[i]=svm->get_alpha(i);
		}

		/*
		if ((max_order < 1) || (max_order > 12))
		{
			SG_WARNING("max_order out of range 1..12 (%d). setting to 1.\n", max_order)
			max_order=1;
		}
		*/

		float64_t* position_weights;
		position_weights=((CWeightedDegreePositionStringKernel*) kernel)->compute_POIM(
				max_order, seqlen, num_sym, NULL,
				num_suppvec, sv_idx, sv_weight, distribution);
		SG_FREE(sv_idx);
		SG_FREE(sv_weight);

		set_matrix(position_weights, num_sym, seqlen);
		SG_FREE(position_weights);

		return true;
	}

	bool CSGInterface::cmd_get_WD_scoring()
	{
		if (m_nrhs!=2 || !create_return_values(1))
			return false;

		int32_t max_order=get_int();

		CKernel* kernel=ui_kernel->get_kernel();
		if (!kernel)
			SG_ERROR("No kernel.\n")
		if (kernel->get_kernel_type()!=K_WEIGHTEDDEGREEPOS)
			SG_ERROR("Only works for Weighted Degree Position kernels.\n")

	CSVM* svm=(CSVM*) ui_classifier->get_classifier();
	ASSERT(svm)
	int32_t num_suppvec=svm->get_num_support_vectors();
	int32_t* sv_idx=SG_MALLOC(int32_t, num_suppvec);
	float64_t* sv_weight=SG_MALLOC(float64_t, num_suppvec);
	int32_t num_feat=0;
	int32_t num_sym=0;

	for (int32_t i=0; i<num_suppvec; i++)
	{
		sv_idx[i]=svm->get_support_vector(i);
		sv_weight[i]=svm->get_alpha(i);
	}

	if ((max_order<1) || (max_order>12))
	{
		SG_WARNING("max_order out of range 1..12 (%d). setting to 1\n", max_order)
		max_order=1;
	}

	float64_t* position_weights=
		((CWeightedDegreePositionStringKernel*) kernel)->compute_scoring(
			max_order, num_feat, num_sym, NULL, num_suppvec, sv_idx, sv_weight);
	SG_FREE(sv_idx);
	SG_FREE(sv_weight);

	set_matrix(position_weights, num_sym, num_feat);
	SG_FREE(position_weights);

	return true;
}


/* Classifier */

bool CSGInterface::cmd_classify()
{
	if (m_nrhs!=1 || !create_return_values(1))
		return false;

	if (!ui_kernel->get_kernel() ||
		ui_kernel->get_kernel()->get_kernel_type()!=K_CUSTOM)
	{
		CFeatures* feat=ui_features->get_test_features();
		if (!feat)
			SG_ERROR("No features found.\n")
	}

	CLabels* labels=ui_classifier->classify();
	if (!labels)
		SG_ERROR("Classify failed\n")

	int32_t num_vec=labels->get_num_labels();
	float64_t* result=SG_MALLOC(float64_t, num_vec);
	for (int32_t i=0; i<num_vec; i++)
	{
		float64_t value = 0;
		switch (labels->get_label_type())
		{
			case LT_REGRESSION:
				value = ((CRegressionLabels*) labels)->get_label(i);
				break;
			case LT_BINARY:
				value = ((CBinaryLabels*) labels)->get_value(i);
				break;
			case LT_MULTICLASS:
				value = ((CMulticlassLabels*) labels)->get_label(i);
				break;
			default:
				SG_NOTIMPLEMENTED;
				break;
		}
		result[i]=value;
	}
	SG_UNREF(labels);

	set_vector(result, num_vec);
	SG_FREE(result);

	return true;
}

bool CSGInterface::cmd_classify_example()
{
	if (m_nrhs!=2 || !create_return_values(1))
		return false;

	int32_t idx=get_int();
	float64_t result=0;

	if (!ui_classifier->classify_example(idx, result))
		SG_ERROR("Classify_example failed.\n")

	set_real(result);

	return true;
}

bool CSGInterface::cmd_get_classifier()
{
	if (m_nrhs<1 || m_nrhs>2 || !create_return_values(2))
		return false;

	int32_t idx=-1;
	if (m_nrhs==2)
		idx=get_int();

	float64_t* bias=NULL;
	float64_t* weights=NULL;
	int32_t rows=0;
	int32_t cols=0;
	int32_t brows=0;
	int32_t bcols=0;

	if (!ui_classifier->get_trained_classifier(
		weights, rows, cols, bias, brows, bcols, idx))
		return false;

	//SG_PRINT("brows %d, bcols %d\n", brows, bcols)
	//CMath::display_matrix(bias, brows, bcols);
	set_matrix(bias, brows, bcols);
	SG_FREE(bias);

	//SG_PRINT("rows %d, cols %d\n", rows, cols)
	//CMath::display_matrix(weights, rows, cols);
	set_matrix(weights, rows, cols);
	SG_FREE(weights);

	return true;
}

bool CSGInterface::cmd_new_classifier()
{
	if (m_nrhs<2 || !create_return_values(0))
		return false;

	int32_t len=0;
	char* name=get_str_from_str_or_direct(len);
	int32_t d=6;
	int32_t from_d=40;

	if (m_nrhs>2)
	{
		d=get_int_from_int_or_str();

		if (m_nrhs>3)
			from_d=get_int_from_int_or_str();
	}

	bool success=ui_classifier->new_classifier(name, d, from_d);

	SG_FREE(name);
	return success;
}

bool CSGInterface::cmd_save_classifier()
{
	if (m_nrhs<2 || !create_return_values(0))
		return false;

	int32_t len=0;
	char* filename=get_str_from_str_or_direct(len);

	bool success=ui_classifier->save(filename);

	SG_FREE(filename);
	return success;
}

bool CSGInterface::cmd_load_classifier()
{
	if (m_nrhs<3 || !create_return_values(0))
		return false;

	int32_t len=0;
	char* filename=get_str_from_str_or_direct(len);
	char* type=get_str_from_str_or_direct(len);

	bool success=ui_classifier->load(filename, type);

	if (dynamic_cast<CKernelMachine*>(ui_classifier->get_classifier()))
	{
		CKernelMachine* kernel_machine = dynamic_cast<CKernelMachine*>(ui_classifier->get_classifier());
		ui_features->set_train_features(kernel_machine->get_kernel()->get_lhs());
		ui_features->set_test_features(kernel_machine->get_kernel()->get_rhs());
		ui_kernel->set_kernel(kernel_machine->get_kernel());
	}

	SG_FREE(filename);
	SG_FREE(type);
	return success;
}

bool CSGInterface::cmd_get_num_svms()
{
	if (m_nrhs!=1 || !create_return_values(1))
		return false;

	set_int(ui_classifier->get_num_svms());

	return true;
}


bool CSGInterface::cmd_get_svm()
{
	return cmd_get_classifier();
}

bool CSGInterface::cmd_set_svm()
{
	if (m_nrhs!=3 || !create_return_values(0))
		return false;

	float64_t bias=get_real();

	float64_t* alphas=NULL;
	int32_t num_feat_alphas=0;
	int32_t num_vec_alphas=0;
	get_matrix(alphas, num_feat_alphas, num_vec_alphas);

	if (!alphas)
		SG_ERROR("No proper alphas given.\n")
	if (num_vec_alphas!=2)
		SG_ERROR("Not 2 vectors in alphas.\n")

	CSVM* svm=(CSVM*) ui_classifier->get_classifier();
	if (!svm)
		SG_ERROR("No SVM object available.\n")

	svm->create_new_model(num_feat_alphas);
	svm->set_bias(bias);

	int32_t num_support_vectors=svm->get_num_support_vectors();
	for (int32_t i=0; i<num_support_vectors; i++)
	{
		svm->set_alpha(i, alphas[i]);
		svm->set_support_vector(i, (int32_t) alphas[i+num_support_vectors]);
	}
	SG_FREE(alphas);

	return true;
}

bool CSGInterface::cmd_set_linear_classifier()
{
	if (m_nrhs!=3 || !create_return_values(0))
		return false;

	float64_t bias=get_real();

	float64_t* w=NULL;
	int32_t len=0;
	get_vector(w, len);

	if (!len)
		SG_ERROR("No proper weight vector given.\n")

	CLinearMachine* c=(CLinearMachine*) ui_classifier->get_classifier();
	if (!c)
		SG_ERROR("No Linear Classifier object available.\n")

	c->set_w(SGVector<float64_t>(w, len));
	c->set_bias(bias);
	return true;
}

bool CSGInterface::cmd_get_svm_objective()
{
	if (m_nrhs!=1 || !create_return_values(1))
		return false;

	CSVM* svm=(CSVM*) ui_classifier->get_classifier();
	if (!svm)
		SG_ERROR("No SVM set.\n")

	set_real(svm->get_objective());

	return true;
}

bool CSGInterface::cmd_compute_svm_primal_objective()
{
	return do_compute_objective(SVM_PRIMAL);
}

bool CSGInterface::cmd_compute_svm_dual_objective()
{
	return do_compute_objective(SVM_DUAL);
}

bool CSGInterface::cmd_compute_mkl_dual_objective()
{
	return do_compute_objective(MKL_DUAL);
}

bool CSGInterface::cmd_compute_relative_mkl_duality_gap()
{
	return do_compute_objective(MKL_RELATIVE_DUALITY_GAP);
}

bool CSGInterface::cmd_compute_absolute_mkl_duality_gap()
{
	return do_compute_objective(MKL_ABSOLUTE_DUALITY_GAP);
}

bool CSGInterface::do_compute_objective(E_WHICH_OBJ obj)
{
	if (m_nrhs!=1 || !create_return_values(1))
		return false;

	float64_t result=23.5;

	CSVM* svm=(CSVM*) ui_classifier->get_classifier();
	if (!svm)
		SG_ERROR("No SVM set.\n")

	CLabels* trainlabels=NULL;
	trainlabels=ui_labels->get_train_labels();

	if (!trainlabels)
		SG_ERROR("No trainlabels available.\n")

	CKernel* kernel=ui_kernel->get_kernel();
	if (!kernel)
		SG_ERROR("No kernel available.\n")

	if (!ui_kernel->is_initialized() || !kernel->has_features())
		SG_ERROR("Kernel not initialized.\n")

	((CKernelMachine*) svm)->set_labels(trainlabels);
	((CKernelMachine*) svm)->set_kernel(kernel);


	switch (obj)
	{
		case  SVM_PRIMAL:
			result=svm->compute_svm_primal_objective();
			break;
		case  SVM_DUAL:
			result=svm->compute_svm_dual_objective();
			break;
		case  MKL_PRIMAL:
			ASSERT( svm->get_classifier_type() == CT_MKLCLASSIFICATION )
			result=((CMKL*) svm)->compute_mkl_primal_objective();
			break;
		case  MKL_DUAL:
			ASSERT( svm->get_classifier_type() == CT_MKLCLASSIFICATION )
			result=((CMKL*) svm)->compute_mkl_dual_objective();
			break;
		case  MKL_RELATIVE_DUALITY_GAP:
			{
				ASSERT( svm->get_classifier_type() == CT_MKLCLASSIFICATION )
				float64_t primal=((CMKL*) svm)->compute_mkl_dual_objective();
				float64_t dual=((CMKL*) svm)->compute_mkl_primal_objective();
				result=(primal-dual)/dual;
			}
			break;
		case  MKL_ABSOLUTE_DUALITY_GAP:
			{
				ASSERT( svm->get_classifier_type() == CT_MKLCLASSIFICATION )
				float64_t primal=((CMKL*) svm)->compute_mkl_dual_objective();
				float64_t dual=((CMKL*) svm)->compute_mkl_primal_objective();
				result=dual-primal;
			}
			break;
		default:
			SG_SERROR("Error calling do_compute_objective\n")
			return false;
	};

	set_real(result);
	return true;
}

bool CSGInterface::cmd_train_classifier()
{
	if (m_nrhs<1 || !create_return_values(0))
		return false;

	CMachine* classifier=ui_classifier->get_classifier();
	if (!classifier)
		SG_ERROR("No classifier available.\n")

	EMachineType type=classifier->get_classifier_type();
	switch (type)
	{
		case CT_LIGHT:
		case CT_LIGHTONECLASS:
		case CT_LIBSVM:
		case CT_SCATTERSVM:
		case CT_MPD:
		case CT_GPBT:
		case CT_CPLEXSVM:
		case CT_GMNPSVM:
		case CT_GNPPSVM:
		case CT_KERNELPERCEPTRON:
		case CT_LIBSVR:
		case CT_LIBSVMMULTICLASS:
		case CT_LIBSVMONECLASS:
		case CT_SVRLIGHT:
		case CT_LARANK:
			return ui_classifier->train_svm();
		case CT_MKLMULTICLASS:
			return ui_classifier->train_mkl_multiclass();
		case CT_MKLCLASSIFICATION:
		case CT_MKLREGRESSION:
		case CT_MKLONECLASS:
			return ui_classifier->train_mkl();

		case CT_KERNELRIDGEREGRESSION:
			return ui_classifier->train_krr();

		case CT_KNN:
		{
			if (m_nrhs<2)
				return false;

			int32_t k=get_int_from_int_or_str();

			return ui_classifier->train_knn(k);
		}

		case CT_KMEANS:
		{
			if (m_nrhs<3)
				return false;

			int32_t k=get_int_from_int_or_str();
			int32_t max_iter=get_int_from_int_or_str();

			return ui_classifier->train_clustering(k, max_iter);
		}

		case CT_HIERARCHICAL:
		{
			if (m_nrhs<2)
				return false;

			int32_t merges=get_int_from_int_or_str();

			return ui_classifier->train_clustering(merges);
		}

		case CT_LDA:
		{
			float64_t gamma=0;
			if (m_nrhs==2)
				gamma=get_real_from_real_or_str();

			return ui_classifier->train_linear(gamma);
		}

		case CT_PERCEPTRON:
		case CT_SVMLIN:
		case CT_SVMPERF:
		case CT_SVMOCAS:
		case CT_SVMSGD:
		case CT_LPM:
		case CT_LPBOOST:
		case CT_LIBLINEAR:
			return ui_classifier->train_linear();

		case CT_WDSVMOCAS:
			return ui_classifier->train_wdocas();

		default:
			SG_ERROR("Unknown classifier type %d.\n", type)
	}

	return false;
}

bool CSGInterface::cmd_do_auc_maximization()
{
	if (m_nrhs!=2 || !create_return_values(0))
		return false;

	bool do_auc=get_bool_from_bool_or_str();

	return ui_classifier->set_do_auc_maximization(do_auc);
}

bool CSGInterface::cmd_set_perceptron_parameters()
{
	if (m_nrhs!=3 || !create_return_values(0))
		return false;

	float64_t lernrate=get_real_from_real_or_str();
	int32_t maxiter=get_int_from_int_or_str();

	return ui_classifier->set_perceptron_parameters(lernrate, maxiter);
}

bool CSGInterface::cmd_set_svm_qpsize()
{
	if (m_nrhs!=2 || !create_return_values(0))
		return false;

	int32_t qpsize=get_int_from_int_or_str();

	return ui_classifier->set_svm_qpsize(qpsize);
}

bool CSGInterface::cmd_set_svm_max_qpsize()
{
	if (m_nrhs!=2 || !create_return_values(0))
		return false;

	int32_t max_qpsize=get_int_from_int_or_str();

	return ui_classifier->set_svm_max_qpsize(max_qpsize);
}

bool CSGInterface::cmd_set_svm_bufsize()
{
	if (m_nrhs!=2 || !create_return_values(0))
		return false;

	int32_t bufsize=get_int_from_int_or_str();

	return ui_classifier->set_svm_bufsize(bufsize);
}

bool CSGInterface::cmd_set_svm_C()
{
	if (m_nrhs<2 || !create_return_values(0))
		return false;

	float64_t C1=get_real_from_real_or_str();
	float64_t C2=C1;

	if (m_nrhs==3)
		C2=get_real_from_real_or_str();

	return ui_classifier->set_svm_C(C1, C2);
}

bool CSGInterface::cmd_set_svm_epsilon()
{
	if (m_nrhs!=2 || !create_return_values(0))
		return false;

	float64_t epsilon=get_real_from_real_or_str();

	return ui_classifier->set_svm_epsilon(epsilon);
}

bool CSGInterface::cmd_set_svr_tube_epsilon()
{
	if (m_nrhs!=2 || !create_return_values(0))
		return false;

	float64_t tube_epsilon=get_real_from_real_or_str();

	return ui_classifier->set_svr_tube_epsilon(tube_epsilon);
}

bool CSGInterface::cmd_set_svm_nu()
{
	if (m_nrhs!=2 || !create_return_values(0))
		return false;

	float64_t nu=get_real_from_real_or_str();

	return ui_classifier->set_svm_nu(nu);
}

bool CSGInterface::cmd_set_svm_mkl_parameters()
{
	if (m_nrhs<3 || m_nrhs>4 || !create_return_values(0))
		return false;

	float64_t weight_epsilon=get_real_from_real_or_str();
	float64_t C_mkl=get_real_from_real_or_str();
	float64_t mkl_norm=1.0;

	if (m_nrhs==4)
		mkl_norm=get_real_from_real_or_str();

	return ui_classifier->set_svm_mkl_parameters(weight_epsilon, C_mkl, mkl_norm);
}

bool CSGInterface::cmd_set_elasticnet_lambda()
{
	if (m_nrhs!=2 || !create_return_values(0))
		return false;
	float64_t lambda=get_real_from_real_or_str();
	return ui_classifier->set_elasticnet_lambda(lambda);
}

bool CSGInterface::cmd_set_mkl_block_norm()
{
	if (m_nrhs!=2 || !create_return_values(0))
		return false;
	float64_t bnorm=get_real_from_real_or_str();
	return ui_classifier->set_mkl_block_norm(bnorm);
}


bool CSGInterface::cmd_set_max_train_time()
{
	if (m_nrhs!=2 || !create_return_values(0))
		return false;

	float64_t max_train_time=get_real_from_real_or_str();

	return ui_classifier->set_max_train_time(max_train_time);
}

bool CSGInterface::cmd_set_svm_shrinking_enabled()
{
	if (m_nrhs!=2 || !create_return_values(0))
		return false;

	bool shrinking_enabled=get_bool_from_bool_or_str();

	return ui_classifier->set_svm_shrinking_enabled(shrinking_enabled);
}

bool CSGInterface::cmd_set_svm_batch_computation_enabled()
{
	if (m_nrhs!=2 || !create_return_values(0))
		return false;

	bool batch_computation_enabled=get_bool_from_bool_or_str();

	return ui_classifier->set_svm_batch_computation_enabled(
		batch_computation_enabled);
}

bool CSGInterface::cmd_set_svm_linadd_enabled()
{
	if (m_nrhs!=2 || !create_return_values(0))
		return false;

	bool linadd_enabled=get_bool_from_bool_or_str();

	return ui_classifier->set_svm_linadd_enabled(linadd_enabled);
}

bool CSGInterface::cmd_set_svm_bias_enabled()
{
	if (m_nrhs!=2 || !create_return_values(0))
		return false;

	bool bias_enabled=get_bool_from_bool_or_str();

	return ui_classifier->set_svm_bias_enabled(bias_enabled);
}

bool CSGInterface::cmd_set_mkl_interleaved_enabled()
{
	if (m_nrhs!=2 || !create_return_values(0))
		return false;

	bool interleaved_enabled=get_bool_from_bool_or_str();

	return ui_classifier->set_mkl_interleaved_enabled(interleaved_enabled);
}

bool CSGInterface::cmd_set_krr_tau()
{
	if (m_nrhs!=2 || !create_return_values(0))
		return false;

	float64_t tau=get_real_from_real_or_str();

	return ui_classifier->set_krr_tau(tau);
}


/* Preproc */

bool CSGInterface::cmd_add_preproc()
{
	if (m_nrhs<2 || !create_return_values(0))
		return false;

	int32_t len=0;
	char* type=get_str_from_str_or_direct(len);
	CPreprocessor* preproc=NULL;

	if (strmatch(type, "NORMONE"))
		preproc=ui_preproc->create_generic(P_NORMONE);
	else if (strmatch(type, "LOGPLUSONE"))
		preproc=ui_preproc->create_generic(P_LOGPLUSONE);
	else if (strmatch(type, "SORTWORDSTRING"))
		preproc=ui_preproc->create_generic(P_SORTWORDSTRING);
	else if (strmatch(type, "SORTULONGSTRING"))
		preproc=ui_preproc->create_generic(P_SORTULONGSTRING);
	else if (strmatch(type, "DECOMPRESSCHARSTRING"))
		preproc=ui_preproc->create_generic(P_DECOMPRESSCHARSTRING);
	else if (strmatch(type, "SORTWORD"))
		preproc=ui_preproc->create_generic(P_SORTWORD);

	else if (strmatch(type, "PRUNEVARSUBMEAN"))
	{
		bool divide_by_std=false;
		if (m_nrhs==3)
			divide_by_std=get_bool_from_bool_or_str();

		preproc=ui_preproc->create_prunevarsubmean(divide_by_std);
	}

#ifdef HAVE_LAPACK
	else if (strmatch(type, "PCA") && m_nrhs==4)
	{
		bool do_whitening=get_bool_from_bool_or_str();
		float64_t threshold=get_real_from_real_or_str();

		preproc=ui_preproc->create_pca(do_whitening, threshold);
	}
#endif

	else
		SG_NOTIMPLEMENTED

	SG_FREE(type);
	return ui_preproc->add_preproc(preproc);
}

bool CSGInterface::cmd_del_preproc()
{
	if (m_nrhs!=1 || !create_return_values(0))
		return false;

	return ui_preproc->del_preproc();
}

bool CSGInterface::cmd_attach_preproc()
{
	if (m_nrhs<2 || !create_return_values(0))
		return false;

	int32_t len=0;
	char* target=get_str_from_str_or_direct(len);

	bool do_force=false;
	if (m_nrhs==3)
		do_force=get_bool_from_bool_or_str();

	bool success=ui_preproc->attach_preproc(target, do_force);

	SG_FREE(target);
	return success;
}

bool CSGInterface::cmd_clean_preproc()
{
	if (m_nrhs!=1 || !create_return_values(0))
		return false;

	return ui_preproc->clean_preproc();
}

/* Converter */

bool CSGInterface::cmd_set_converter()
{
	int32_t len=0;
	char* type=get_str_from_str_or_direct(len);

	if (strmatch(type, "lle"))
	{
		int32_t k = get_int_from_int_or_str();
		ui_converter->create_locallylinearembedding(k);
		return true;
	}
	if (strmatch(type, "npe"))
	{
		int32_t k = get_int_from_int_or_str();
		ui_converter->create_neighborhoodpreservingembedding(k);
		return true;
	}
	if (strmatch(type, "ltsa"))
	{
		int32_t k = get_int_from_int_or_str();
		ui_converter->create_localtangentspacealignment(k);
		return true;
	}
	if (strmatch(type, "lltsa"))
	{
		int32_t k = get_int_from_int_or_str();
		ui_converter->create_linearlocaltangentspacealignment(k);
		return true;
	}
	if (strmatch(type, "hlle"))
	{
		int32_t k = get_int_from_int_or_str();
		ui_converter->create_hessianlocallylinearembedding(k);
		return true;
	}
	if (strmatch(type, "laplacian_eigenmaps"))
	{
		int32_t k = get_int_from_int_or_str();
		int32_t width = get_real_from_real_or_str();
		ui_converter->create_laplacianeigenmaps(k,width);
		return true;
	}
	if (strmatch(type, "lpp"))
	{
		int32_t k = get_int_from_int_or_str();
		int32_t width = get_real_from_real_or_str();
		ui_converter->create_localitypreservingprojections(k,width);
		return true;
	}
	if (strmatch(type, "diffusion_maps"))
	{
		int32_t t = get_int_from_int_or_str();
		int32_t width = get_real_from_real_or_str();
		ui_converter->create_diffusionmaps(t,width);
		return true;
	}
	if (strmatch(type, "isomap"))
	{
		int32_t k = get_int_from_int_or_str();
		ui_converter->create_isomap(k);
		return true;
	}
	if (strmatch(type, "mds"))
	{
		ui_converter->create_multidimensionalscaling();
		return true;
	}
	if (strmatch(type, "jade"))
	{
		ui_converter->create_jade();
		return true;
	}
	return false;
}

bool CSGInterface::cmd_apply_converter()
{
	if (m_nrhs!=1 || !create_return_values(1))
		return false;
		
	CDenseFeatures<float64_t>* conv_features = ui_converter->apply();
	SGMatrix<float64_t> converted_mat = conv_features->get_feature_matrix();
	set_matrix(converted_mat.matrix,converted_mat.num_rows,converted_mat.num_cols);
	return true;
}

bool CSGInterface::cmd_embed()
{
	int32_t target_dim = get_int_from_int_or_str();
	
	if (m_nrhs!=1 || !create_return_values(1))
		return false;
	
	CDenseFeatures<float64_t>* embedding = ui_converter->embed(target_dim);
	SGMatrix<float64_t> embedding_matrix = embedding->get_feature_matrix();
	set_matrix(embedding_matrix.matrix,embedding_matrix.num_cols,embedding_matrix.num_rows);
	return true;
}

/* HMM */

bool CSGInterface::cmd_new_plugin_estimator()
{
	if (m_nrhs<2 || !create_return_values(0))
		return false;

	float64_t pos_pseudo=get_real_from_real_or_str();
	float64_t neg_pseudo=get_real_from_real_or_str();

	return ui_pluginestimate->new_estimator(pos_pseudo, neg_pseudo);
}

bool CSGInterface::cmd_train_estimator()
{
	if (m_nrhs!=1 || !create_return_values(0))
		return false;

	return ui_pluginestimate->train();
}

bool CSGInterface::cmd_plugin_estimate_classify_example()
{
	if (m_nrhs!=2 || !create_return_values(1))
		return false;

	int32_t idx=get_int();
	float64_t result=ui_pluginestimate->apply_one(idx);

	set_vector(&result, 1);
	return true;
}

bool CSGInterface::cmd_plugin_estimate_classify()
{
	if (m_nrhs!=1 || !create_return_values(1))
		return false;

	CFeatures* feat=ui_features->get_test_features();
	if (!feat)
		SG_ERROR("No features found.\n")

	int32_t num_vec=feat->get_num_vectors();
	float64_t* result=SG_MALLOC(float64_t, num_vec);
	CLabels* labels=ui_pluginestimate->apply();
	for (int32_t i=0; i<num_vec; i++)
		result[i]=((CRegressionLabels*) labels)->get_label(i);
	SG_UNREF(labels);

	set_vector(result, num_vec);
	SG_FREE(result);

	return true;
}

bool CSGInterface::cmd_set_plugin_estimate()
{
	if (m_nrhs!=3 || !create_return_values(0))
		return false;

	float64_t* emission_probs=NULL;
	int32_t num_probs=0;
	int32_t num_vec=0;
	get_matrix(emission_probs, num_probs, num_vec);

	if (num_vec!=2)
		SG_ERROR("Need at least 1 set of positive and 1 set of negative params.\n")

	float64_t* pos_params=emission_probs;
	float64_t* neg_params=&(emission_probs[num_probs]);

	float64_t* model_sizes=NULL;
	int32_t len=0;
	get_vector(model_sizes, len);

	int32_t seq_length=(int32_t) model_sizes[0];
	int32_t num_symbols=(int32_t) model_sizes[1];
	if (num_probs!=seq_length*num_symbols)
		SG_ERROR("Mismatch in number of emission probs and sequence length * number of symbols.\n")

	ui_pluginestimate->get_estimator()->set_model_params(
		pos_params, neg_params, seq_length, num_symbols);

	return true;
}

bool CSGInterface::cmd_get_plugin_estimate()
{
	if (m_nrhs!=1 || !create_return_values(2))
		return false;

	float64_t* pos_params=NULL;
	float64_t* neg_params=NULL;
	int32_t num_params=0;
	int32_t seq_length=0;
	int32_t num_symbols=0;

	if (!ui_pluginestimate->get_estimator()->get_model_params(
		pos_params, neg_params, seq_length, num_symbols))
		return false;

	num_params=seq_length*num_symbols;

	float64_t* result=SG_MALLOC(float64_t, num_params*2);
	for (int32_t i=0; i<num_params; i++)
		result[i]=pos_params[i];
	for (int32_t i=0; i<num_params; i++)
		result[i+num_params]=neg_params[i];

	set_matrix(result, num_params, 2);
	SG_FREE(result);

	float64_t model_sizes[2];
	model_sizes[0]=(float64_t) seq_length;
	model_sizes[1]=(float64_t) num_symbols;
	set_vector(model_sizes, 2);

	return true;
}

bool CSGInterface::cmd_convergence_criteria()
{
	if (m_nrhs<3 || !create_return_values(0))
		return false;

	int32_t num_iterations=get_int_from_int_or_str();
	float64_t epsilon=get_real_from_real_or_str();

	return ui_hmm->convergence_criteria(num_iterations, epsilon);
}

bool CSGInterface::cmd_normalize()
{
	if (m_nrhs<2 || !create_return_values(0))
		return false;

	bool keep_dead_states=get_bool_from_bool_or_str();

	return ui_hmm->normalize(keep_dead_states);
}

bool CSGInterface::cmd_add_states()
{
	if (m_nrhs<3 || !create_return_values(0))
		return false;

	int32_t num_states=get_int_from_int_or_str();
	float64_t value=get_real_from_real_or_str();

	return ui_hmm->add_states(num_states, value);
}

bool CSGInterface::cmd_permutation_entropy()
{
	if (m_nrhs<3 || !create_return_values(0))
		return false;

	int32_t width=get_int_from_int_or_str();
	int32_t seq_num=get_int_from_int_or_str();

	return ui_hmm->permutation_entropy(width, seq_num);
}

bool CSGInterface::cmd_relative_entropy()
{
	if (m_nrhs!=1 || !create_return_values(1))
		return false;

	float64_t* entropy=NULL;
	int32_t len=0;
	bool success=ui_hmm->relative_entropy(entropy, len);
	if (!success)
		return false;

	set_vector(entropy, len);

	SG_FREE(entropy);
	return true;
}

bool CSGInterface::cmd_entropy()
{
	if (m_nrhs!=1 || !create_return_values(1))
		return false;

	float64_t* entropy=NULL;
	int32_t len=0;
	bool success=ui_hmm->entropy(entropy, len);
	if (!success)
		return false;

	set_vector(entropy, len);

	SG_FREE(entropy);
	return true;
}

bool CSGInterface::cmd_hmm_classify()
{
	return do_hmm_classify(false, false);
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

	CFeatures* feat=ui_features->get_test_features();
	if (!feat)
		return false;

	int32_t num_vec=feat->get_num_vectors();
	CRegressionLabels* labels=NULL;

	if (linear) // must be one_class as well
	{
		labels=ui_hmm->linear_one_class_classify();
	}
	else
	{
		if (one_class)
			labels=ui_hmm->one_class_classify();
		else
			labels=ui_hmm->classify();
	}
	if (!labels)
		return false;

	float64_t* result=SG_MALLOC(float64_t, num_vec);
	for (int32_t i=0; i<num_vec; i++)
		result[i]=labels->get_label(i);
	SG_UNREF(labels);

	set_vector(result, num_vec);
	SG_FREE(result);

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

	int32_t idx=get_int();
	float64_t result=0;

	if (one_class)
		result=ui_hmm->one_class_classify_example(idx);
	else
		result=ui_hmm->classify_example(idx);

	set_real(result);

	return true;
}

bool CSGInterface::cmd_output_hmm()
{
	if (m_nrhs!=1 || !create_return_values(0))
		return false;

	return ui_hmm->output_hmm();
}

bool CSGInterface::cmd_output_hmm_defined()
{
	if (m_nrhs!=1 || !create_return_values(0))
		return false;

	return ui_hmm->output_hmm_defined();
}

bool CSGInterface::cmd_hmm_likelihood()
{
	if (m_nrhs!=1 || !create_return_values(1))
		return false;

	CHMM* h=ui_hmm->get_current();
	if (!h)
		SG_ERROR("No HMM.\n")

	float64_t likelihood=h->model_probability();
	set_real(likelihood);

	return true;
}

bool CSGInterface::cmd_likelihood()
{
	if (m_nrhs!=1 || !create_return_values(0))
		return false;

	return ui_hmm->likelihood();
}

bool CSGInterface::cmd_save_likelihood()
{
	if (m_nrhs<2 || !create_return_values(0))
		return false;

	int32_t len=0;
	char* filename=get_str_from_str_or_direct(len);

	bool is_binary=false;
	if (m_nrhs==3)
		is_binary=get_bool_from_bool_or_str();

	bool success=ui_hmm->save_likelihood(filename, is_binary);

	SG_FREE(filename);
	return success;
}

bool CSGInterface::cmd_get_viterbi_path()
{
	if (m_nrhs!=2 || !create_return_values(2))
		return false;

	int32_t dim=get_int();
	SG_DEBUG("dim: %f\n", dim)

	CHMM* h=ui_hmm->get_current();
	if (!h)
		return false;

	CFeatures* feat=ui_features->get_test_features();
	if (!feat || (feat->get_feature_class()!=C_STRING) ||
			(feat->get_feature_type()!=F_WORD))
		return false;

	h->set_observations((CStringFeatures<uint16_t>*) feat);

	int32_t num_feat=0;
	bool free_vec;
	uint16_t* vec=((CStringFeatures<uint16_t>*) feat)->get_feature_vector(dim, num_feat, free_vec);
	if (!vec || num_feat<=0)
	{
		((CStringFeatures<uint16_t>*) feat)->free_feature_vector(vec, dim, free_vec);
		return false;
	}
	((CStringFeatures<uint16_t>*) feat)->free_feature_vector(vec, dim, free_vec);

	SG_DEBUG("computing viterbi path for vector %d (length %d)\n", dim, num_feat)
	float64_t likelihood=0;
	T_STATES* path=h->get_path(dim, likelihood);

	set_vector(path, num_feat);
	SG_FREE(path);
	set_real(likelihood);

	return true;
}

bool CSGInterface::cmd_viterbi_train()
{
	if (m_nrhs!=1 || !create_return_values(0))
		return false;

	return ui_hmm->viterbi_train();
}

bool CSGInterface::cmd_viterbi_train_defined()
{
	if (m_nrhs!=1 || !create_return_values(0))
		return false;

	return ui_hmm->viterbi_train_defined();
}

bool CSGInterface::cmd_baum_welch_train()
{
	if (m_nrhs!=1 || !create_return_values(0))
		return false;

	return ui_hmm->baum_welch_train();
}

bool CSGInterface::cmd_baum_welch_train_defined()
{
	if (m_nrhs!=1 || !create_return_values(0))
		return false;

	return ui_hmm->baum_welch_train_defined();
}


bool CSGInterface::cmd_baum_welch_trans_train()
{
	if (m_nrhs!=1 || !create_return_values(0))
		return false;

	return ui_hmm->baum_welch_trans_train();
}

bool CSGInterface::cmd_linear_train()
{
	if (m_nrhs<1 || !create_return_values(0))
		return false;

	if (m_nrhs==2)
	{
		int32_t len=0;
		char* align=get_str_from_str_or_direct(len);

		bool success=ui_hmm->linear_train(align[0]);

		SG_FREE(align);
		return success;
	}
	else
		return ui_hmm->linear_train();
}

bool CSGInterface::cmd_save_path()
{
	if (m_nrhs<2 || !create_return_values(0))
		return false;

	int32_t len=0;
	char* filename=get_str_from_str_or_direct(len);

	bool is_binary=false;
	if (m_nrhs==3)
		is_binary=get_bool_from_bool_or_str();

	bool success=ui_hmm->save_path(filename, is_binary);

	SG_FREE(filename);
	return success;
}

bool CSGInterface::cmd_append_hmm()
{
	if (m_nrhs!=5 || !create_return_values(0))
		return false;

	CHMM* old_h=ui_hmm->get_current();
	if (!old_h)
		SG_ERROR("No current HMM set.\n")

	float64_t* p=NULL;
	int32_t N_p=0;
	get_vector(p, N_p);

	float64_t* q=NULL;
	int32_t N_q=0;
	get_vector(q, N_q);

	float64_t* a=NULL;
	int32_t M_a=0;
	int32_t N_a=0;
	get_matrix(a, M_a, N_a);
	int32_t N=N_a;

	float64_t* b=NULL;
	int32_t M_b=0;
	int32_t N_b=0;
	get_matrix(b, M_b, N_b);
	int32_t M=N_b;

	if (N_p!=N || N_q!=N || N_a!=N || M_a!=N || N_b!=M || M_b!=N)
	{
		SG_ERROR("Model matrices not matching in size.\n"
				"p:(%d) q:(%d) a:(%d,%d) b(%d,%d)\n",
				N_p, N_q, N_a, M_a, N_b, M_b);
	}

	CHMM* h=new CHMM(N, M, NULL, ui_hmm->get_pseudo());
	int32_t i,j;

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
	SG_UNREF(h);

	return true;
}

bool CSGInterface::cmd_append_model()
{
	if (m_nrhs<2 || !create_return_values(0))
		return false;
	if (m_nrhs>2 && m_nrhs!=4)
		return false;

	int32_t len=0;
	char* filename=get_str_from_str_or_direct(len);
	int32_t base1=-1;
	int32_t base2=-1;
	if (m_nrhs>2)
	{
		base1=get_int_from_int_or_str();
		base2=get_int_from_int_or_str();
	}

	bool success=ui_hmm->append_model(filename, base1, base2);

	SG_FREE(filename);
	return success;
}

bool CSGInterface::cmd_new_hmm()
{
	if (m_nrhs!=3 || !create_return_values(0))
		return false;

	int32_t n=get_int_from_int_or_str();
	int32_t m=get_int_from_int_or_str();

	return ui_hmm->new_hmm(n, m);
}

bool CSGInterface::cmd_load_hmm()
{
	if (m_nrhs!=2 || !create_return_values(0))
		return false;

	int32_t len=0;
	char* filename=get_str_from_str_or_direct(len);

	bool success=ui_hmm->load(filename);

	SG_FREE(filename);
	return success;
}

bool CSGInterface::cmd_save_hmm()
{
	if (m_nrhs<2 || !create_return_values(0))
		return false;

	int32_t len=0;
	char* filename=get_str_from_str_or_direct(len);

	bool is_binary=false;
	if (m_nrhs==3)
		is_binary=get_bool_from_bool_or_str();

	bool success=ui_hmm->save(filename, is_binary);

	SG_FREE(filename);
	return success;
}

bool CSGInterface::cmd_set_hmm()
{
	if (m_nrhs!=5 || !create_return_values(0))
		return false;

	float64_t* p=NULL;
	int32_t N_p=0;
	get_vector(p, N_p);

	float64_t* q=NULL;
	int32_t N_q=0;
	get_vector(q, N_q);

	float64_t* a=NULL;
	int32_t M_a=0;
	int32_t N_a=0;
	get_matrix(a, M_a, N_a);
	int32_t N=N_a;

	float64_t* b=NULL;
	int32_t M_b=0;
	int32_t N_b=0;
	get_matrix(b, M_b, N_b);
	int32_t M=N_b;

	if (N_p!=N || N_q!=N || N_a!=N || M_a!=N || N_b!=M || M_b!=N)
	{
		SG_ERROR("Model matrices not matching in size.\n"
				"p:(%d) q:(%d) a:(%d,%d) b(%d,%d)\n",
				N_p, N_q, N_a, M_a, N_b, M_b);
	}

	CHMM* current=ui_hmm->get_current();
	if (!current)
		SG_ERROR("Need a previously created HMM.\n")

	int32_t i,j;

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

	CStringFeatures<uint16_t>* sf = ((CStringFeatures<uint16_t>*) (ui_features->get_train_features()));
	current->set_observations(sf);

	return true;
}

bool CSGInterface::cmd_set_hmm_as()
{
	if (m_nrhs!=2 || !create_return_values(0))
		return false;

	int32_t len=0;
	char* target=get_str_from_str_or_direct(len);

	bool success=ui_hmm->set_hmm_as(target);

	SG_FREE(target);
	return success;
}

bool CSGInterface::cmd_set_chop()
{
	if (m_nrhs!=2 || !create_return_values(0))
		return false;

	float64_t value=get_real_from_real_or_str();
	return ui_hmm->chop(value);
}

bool CSGInterface::cmd_set_pseudo()
{
	if (m_nrhs!=2 || !create_return_values(0))
		return false;

	float64_t value=get_real_from_real_or_str();
	return ui_hmm->set_pseudo(value);
}

bool CSGInterface::cmd_load_definitions()
{
	if (m_nrhs<2 || !create_return_values(0))
		return false;

	int32_t len=0;
	char* filename=get_str_from_str_or_direct(len);

	bool do_init=false;
	if (m_nrhs==3)
		do_init=get_bool_from_bool_or_str();

	bool success=ui_hmm->load_definitions(filename, do_init);

	SG_FREE(filename);
	return success;
}

bool CSGInterface::cmd_get_hmm()
{
	if (m_nrhs!=1 || !create_return_values(4))
		return false;

	CHMM* h=ui_hmm->get_current();
	if (!h)
		return false;

	int32_t N=h->get_N();
	int32_t M=h->get_M();
	int32_t i=0;
	int32_t j=0;
	float64_t* p=SG_MALLOC(float64_t, N);
	float64_t* q=SG_MALLOC(float64_t, N);

	for (i=0; i<N; i++)
	{
		p[i]=h->get_p(i);
		q[i]=h->get_q(i);
	}

	set_vector(p, N);
	SG_FREE(p);
	set_vector(q, N);
	SG_FREE(q);

	float64_t* a=SG_MALLOC(float64_t, N*N);
	for (i=0; i<N; i++)
		for (j=0; j<N; j++)
			a[i+j*N]=h->get_a(i, j);
	set_matrix(a, N, N);
	SG_FREE(a);

	float64_t* b=SG_MALLOC(float64_t, N*M);
	for (i=0; i<N; i++)
		for (j=0; j<M; j++)
			b[i+j*N]=h->get_b(i, j);
	set_matrix(b, N, M);
	SG_FREE(b);

	return true;
}

bool CSGInterface::cmd_best_path()
{
	if (m_nrhs!=3 || !create_return_values(0))
		return false;

	int32_t from=get_int_from_int_or_str();
	int32_t to=get_int_from_int_or_str();

	return ui_hmm->best_path(from, to);
}

bool CSGInterface::cmd_best_path_2struct()
{
	if (m_nrhs!=12 || !create_return_values(3))
		return false;

	SG_ERROR("Sorry, this parameter list is awful!\n")

	return true;
}

void CSGInterface::get_vector(bool*& vector, int32_t& len)
{
	int32_t* int_vector;
	get_vector(int_vector, len);

	ASSERT(len>0)
	vector= SG_MALLOC(bool, len);

	for (int32_t i=0; i<len; i++)
		vector[i]= (int_vector[i]!=0);

	SG_FREE(int_vector);
}

void CSGInterface::set_vector(const bool* vector, int32_t len)
{
	int32_t* int_vector = SG_MALLOC(int32_t, len);
	for (int32_t i=0;i<len;i++)
	{
		if (vector[i])
			int_vector[i]=1;
		else
			int_vector[i]=0;
	}
	set_vector(int_vector,len);
	SG_FREE(int_vector);
}

bool CSGInterface::cmd_set_plif_struct()
{
	// ARG 2
	int32_t Nid=0;
	int32_t* ids;
	get_vector(ids,Nid);

	// ARG 3
	int32_t Nname=0;
	int32_t Mname=0;
	SGString<char>* names;
	get_string_list(names, Nname,Mname);

	// ARG 4
	int32_t Nlimits=0;
	int32_t Mlimits=0;
	float64_t* all_limits;
	get_matrix(all_limits, Mlimits, Nlimits);

	// ARG 5
	int32_t Npenalties=0;
	int32_t Mpenalties=0;
	float64_t* all_penalties;
	get_matrix(all_penalties, Mpenalties, Npenalties);

	// ARG 6
	int32_t Ntransform=0;
	int32_t Mtransform=0;
	SGString<char>* all_transform;
	get_string_list(all_transform, Ntransform, Mtransform);

	// ARG 7
	int32_t Nmin=0;
	float64_t* min_values;
	get_vector(min_values,Nmin);

	// ARG 8
	int32_t Nmax=0;
	float64_t* max_values;
	get_vector(max_values,Nmax);

	// ARG 9
	int32_t Ncache=0;
	bool* all_use_cache;
	get_vector(all_use_cache,Ncache);

	// ARG 10
	int32_t Nsvm=0;
	int32_t* all_use_svm;
	get_vector(all_use_svm,Nsvm);

	// ARG 11
	int32_t Ncalc=0;
	bool* all_do_calc;
	get_vector(all_do_calc,Ncalc);

	if (Ncalc!=Nsvm)
		SG_ERROR("Ncalc!=Nsvm, Ncalc:%i, Nsvm:%i\n",Ncalc,Nsvm)
	if (Ncalc!=Ncache)
		SG_ERROR("Ncalc!=Ncache, Ncalc:%i, Ncache:%i\n",Ncalc,Ncache)
	if (Ncalc!=Ntransform)
		SG_ERROR("Ncalc!=Ntransform, Ncalc:%i, Ntransform:%i\n",Ncalc,Ntransform)
	if (Ncalc!=Nmin)
		SG_ERROR("Ncalc!=Nmin, Ncalc:%i, Nmin:%i\n",Ncalc,Nmin)
	if (Ncalc!=Nmax)
		SG_ERROR("Ncalc!=Nmax, Ncalc:%i, Nmax:%i\n",Ncalc,Nmax)
	if (Ncalc!=Npenalties)
		SG_ERROR("Ncalc!=Npenalties, Ncalc:%i, Npenalties:%i\n",Ncalc,Npenalties)
	if (Ncalc!=Nlimits)
		SG_ERROR("Ncalc!=Nlimits, Ncalc:%i, Nlimits:%i\n",Ncalc,Nlimits)
	if (Ncalc!=Nname)
		SG_ERROR("Ncalc!=Nname, Ncalc:%i, Nname:%i\n",Ncalc,Nname)
	if (Ncalc!=Nid)
		SG_ERROR("Ncalc!=Nid, Ncalc:%i, Nid:%i\n",Ncalc,Nid)
	if (Mlimits!=Mpenalties)
		SG_ERROR("Mlimits!=Mpenalties, Mlimits:%i, Mpenalties:%i\n",Mlimits,Mpenalties)

	int32_t N = Ncalc;
	int32_t M = Mlimits;
	CPlifMatrix* pm=ui_structure->get_plif_matrix();
	pm->create_plifs(N, M);
	pm->set_plif_ids(SGVector<int32_t>(ids, N));
	pm->set_plif_min_values(SGVector<float64_t>(min_values, N));
	pm->set_plif_max_values(SGVector<float64_t>(max_values, N));
	pm->set_plif_use_cache(SGVector<bool>(all_use_cache, N));
	pm->set_plif_use_svm(SGVector<int32_t>(all_use_svm, N));
	pm->set_plif_limits(SGMatrix<float64_t>(all_limits, N, M));
	pm->set_plif_penalties(SGMatrix<float64_t>(all_penalties, N, M));
	pm->set_plif_names(names, N);
	pm->set_plif_transform_type(all_transform, N);

	SG_FREE(names);
	SG_FREE(all_transform);
	SG_FREE(all_do_calc);

	return true;
}

bool CSGInterface::cmd_get_plif_struct()
{
	CPlifMatrix* pm=ui_structure->get_plif_matrix();
	CPlif** PEN = pm->get_PEN();
	int32_t N = pm->get_num_plifs();
	int32_t M = pm->get_num_limits();


	int32_t* ids = SG_MALLOC(int32_t, N);
	float64_t* max_values = SG_MALLOC(float64_t, N);
	float64_t* min_values = SG_MALLOC(float64_t, N);
	SGString<char>* names = SG_MALLOC(SGString<char>, N);
	SGString<char>* all_transform = SG_MALLOC(SGString<char>, N);
	float64_t* all_limits = SG_MALLOC(float64_t, N*M);
	float64_t* all_penalties = SG_MALLOC(float64_t, N*M);
	bool* all_use_cache = SG_MALLOC(bool, N);
	int32_t* all_use_svm = SG_MALLOC(int32_t, N);
	bool* all_do_calc = SG_MALLOC(bool, N);
	for (int32_t i=0;i<N;i++)
	{
		ids[i]=PEN[i]->get_id();
		names[i].string = PEN[i]->get_plif_name();
		names[i].slen = strlen(PEN[i]->get_plif_name());
		SGVector<float64_t> limits = PEN[i]->get_plif_limits();
		SGVector<float64_t> penalties = PEN[i]->get_plif_penalties();
		for (int32_t j=0;j<M;j++)
		{
			all_limits[i*M+j]=limits[j];
			all_penalties[i*M+j]=penalties[j];
		}
		all_transform[i].string = (char*) PEN[i]->get_transform_type();
		all_transform[i].slen = strlen(PEN[i]->get_transform_type());
		min_values[i]=PEN[i]->get_min_value();
		max_values[i]=PEN[i]->get_max_value();
		all_use_cache[i]=PEN[i]->get_use_cache();
		all_use_svm[i]=PEN[i]->get_use_svm();
		all_do_calc[i]=PEN[i]->get_do_calc();

	}
	set_vector(ids,N);
	set_string_list(names, N);
	set_matrix(all_limits, M, N);
	set_matrix(all_penalties, M, N);
	set_string_list(all_transform, N);
	set_vector(min_values,N);
	set_vector(max_values,N);
	set_vector(all_use_cache,N);
	set_vector(all_use_svm,N);
	set_vector(all_do_calc,N);

	SG_FREE(ids);
	SG_FREE(max_values);
	SG_FREE(min_values);
	SG_FREE(names);
	SG_FREE(all_transform);
	SG_FREE(all_limits);
	SG_FREE(all_penalties);
	SG_FREE(all_use_cache);
	SG_FREE(all_use_svm);
	SG_FREE(all_do_calc);

	return true;
}
/*bool CSGInterface::cmd_signals_set_model()
{
	// ARG 1
	int32_t len=0;
	char* filename;
	filename = get_string(len);

	CTrainPredMaster* tpm = new CTrainPredMaster(ui_kernel);

	tpm->read_models_from_file(filename);

	return true;
	}*/
bool CSGInterface::cmd_signals_set_positions()
{
	return true;
}
bool CSGInterface::cmd_signals_set_labels()
{
	return true;
}
bool CSGInterface::cmd_signals_set_split()
{
	return true;
}
bool CSGInterface::cmd_signals_set_train_mask()
{
	return true;
}
bool CSGInterface::cmd_signals_add_feature()
{
	return true;
}
bool CSGInterface::cmd_signals_add_kernel()
{
	return true;
}
bool CSGInterface::cmd_signals_run()
{
	return true;
}

bool CSGInterface::cmd_init_dyn_prog()
{
	//ARG 1
	int32_t num_svms=get_int();

	CDynProg* h=new CDynProg(num_svms);
	ui_structure->set_dyn_prog(h);
	return true;
}

bool CSGInterface::cmd_clean_up_dyn_prog()
{
	return ui_structure->cleanup();
}

bool CSGInterface::cmd_set_model()
{

	CPlifMatrix* pm=ui_structure->get_plif_matrix();

	CDynProg* h = ui_structure->get_dyn_prog();
	int32_t num_svms = h->get_num_svms();
	//CDynProg* h=new CDynProg(Nweights/* = num_svms */);

	//ARG 1
	// transition pointers
	// link transitions to length, content, frame (and tiling)
	// plifs (#states x #states x 3 or 4)
	int32_t numDim=0;
	int32_t* Dim=0;
	float64_t* penalties_array=NULL;
	get_ndarray(penalties_array,Dim,numDim);
	ASSERT(numDim==3)
	ASSERT(Dim[0]==Dim[1])

	if (!pm->compute_plif_matrix(SGNDArray<float64_t>(penalties_array, Dim, numDim, false)))
		SG_ERROR("error computing plif  matrix\n")
	ui_structure->set_num_states(Dim[0]);
	SG_FREE(penalties_array);

	// ARG 2
	// bool-> determines if orf information should be used
	bool use_orf = get_bool();
	ui_structure->set_use_orf(use_orf);

	// ARG 3
	// determines for which contents which orf should be used (#contents x 2)
	int32_t Nmod=0;
	int32_t Mmod=0;
	int32_t* mod_words;
	get_matrix(mod_words, Nmod,Mmod);
	if (Nmod != num_svms)
		SG_ERROR("should be equal: Nmod: %i, num_svms: %i\n",Nmod,num_svms)
	ASSERT(Mmod == 2)
	h->init_mod_words_array(SGMatrix<int32_t>(mod_words, Nmod, Mmod));

	// ARG 4
	// links: states -> signal plifs (#states x 2)
	int32_t num_states=0;
	int32_t feat_dim3=0;
	int32_t* state_signals;
	get_matrix(state_signals,num_states,feat_dim3);
	ASSERT(num_states==Dim[0])
	pm->compute_signal_plifs(SGMatrix<int32_t>(state_signals, feat_dim3, num_states));


	// ARG 5
	// ORF info (#states x 2)
	int32_t Norf=0;
	int32_t Morf=0;
	int32_t* orf_info;
	get_matrix(orf_info,Norf,Morf);
	ASSERT(Norf==num_states)
	ASSERT(Morf==2)

	ui_structure->set_orf_info(orf_info, Norf, Morf);
	h->set_orf_info(SGMatrix<int32_t>(orf_info, Norf, Morf));

	h->set_num_states(num_states) ;

	return true;
}

bool CSGInterface::cmd_precompute_content_svms()
{

	// ARG 1
	int32_t seq_len=0;
	char* seq;
	seq = get_string(seq_len);

	// ARG 2
	// all feature positions
	int32_t Npos=0;
	int32_t* all_pos;
	get_vector(all_pos, Npos);

	//ARG 3
	// content svm weights
	int32_t Nweights=0;
	int32_t num_svms=0;
	float64_t* weights;
	get_matrix(weights, Nweights, num_svms);
	if (Nweights!=5440)
	  SG_PRINT("Dimension mismatch: got %i, expect %i\n", Nweights, 5440)
	ui_structure->set_content_svm_weights(weights, Nweights, num_svms);

	CDynProg* h = ui_structure->get_dyn_prog();
	if (!h)
		SG_ERROR("no DynProg object found, use init_dyn_prog first\n")


	//float64_t* weights = ui_structure->get_content_svm_weights();
	//int32_t Mweights = h->get_num_svms();
	//int32_t Nweights = ui_structure->get_num_svm_weights();
	h->set_pos(SGVector<int32_t>(all_pos, Npos));
	h->set_gene_string(SGVector<char>(seq, seq_len));
	h->create_word_string();
	h->precompute_stop_codons();
	h->init_content_svm_value_array(num_svms);
	h->set_dict_weights(SGMatrix<float64_t>(weights, Nweights, num_svms));
	h->precompute_content_values();
	SG_DEBUG("precompute_content_svms done\n")
	return true;
}

bool CSGInterface::cmd_get_lin_feat()
{
	CDynProg* h = ui_structure->get_dyn_prog();
	if (!h)
		SG_ERROR("no DynProg object found, use set_model first\n")


	int32_t dim1, dim2 = 0;
	float64_t* lin_feat = h->get_lin_feat(dim1, dim2);

	set_matrix(lin_feat, dim1, dim2);

	return true;
}
bool CSGInterface::cmd_set_lin_feat()
{
	// ARG 1
	int32_t Nseq=0;
	char* seq;
	seq = get_string(Nseq);

	// ARG 2
	// all feature positions
	int32_t Npos=0;
	int32_t* all_pos;
	get_vector(all_pos, Npos);

	//ARG 3
	//
	int32_t num_svms, seq_len;
	float64_t* lin_feat=NULL;
	get_matrix(lin_feat, num_svms, seq_len);

        if (Npos!=seq_len)
	  {
	    SG_ERROR("Dimension mismatch: got %i positions and (%ix%i) values\n", Npos, num_svms, seq_len)

	    SG_FREE(lin_feat);
	    SG_FREE(seq);
	    SG_FREE(all_pos);

	    return false ;
	  }

	CDynProg* h = ui_structure->get_dyn_prog();
	if (!h)
		SG_ERROR("no DynProg object found, use set_model first\n")

	h->set_pos(SGVector<int32_t>(all_pos, Npos));
	h->set_gene_string(SGVector<char>(seq, Nseq));
	h->precompute_stop_codons();
	h->init_content_svm_value_array(num_svms);
	h->set_lin_feat(lin_feat, num_svms, seq_len);

	SG_FREE(lin_feat);

	return true;
}
bool CSGInterface::cmd_long_transition_settings()
{
	bool use_long_transitions = get_bool();
	int32_t threshold = get_int();
	int32_t max_len = get_int();

	CDynProg* h = ui_structure->get_dyn_prog();
        if (!h)
                SG_ERROR("no DynProg object found, use set_model first\n")

	h->long_transition_settings(use_long_transitions, threshold, max_len);

	return true;
}
bool CSGInterface::cmd_set_feature_matrix()
{
	int32_t num_states = ui_structure->get_num_states();

	//ARG 1
	// feature matrix (#states x #feature_positions x max_num_signals)
	int32_t* Dims=0;
	int32_t numDims=0;
	float64_t* features = NULL;
	get_ndarray(features, Dims, numDims);

	if (numDims!=3)
		SG_ERROR("expected a 3 dimensional array, got %i dimensions\n", numDims)
	if (Dims[0]!=num_states)
		SG_ERROR("number of rows (%i) not equal number of states (%i)\n",Dims[0], num_states)
	ASSERT(ui_structure->set_feature_matrix(features, Dims))

	ASSERT(ui_structure->set_feature_dims(Dims))

	SG_FREE(features);
	SG_FREE(Dims);

	return true;
}
bool CSGInterface::cmd_set_feature_matrix_sparse()
{
	int32_t num_pos = ui_structure->get_num_positions();
	int32_t num_states = ui_structure->get_num_states();

	//ARG 1
	// feature matrix (#states x #feature_positions x max_num_signals)
	int32_t dim11, dim12 ;
	SGSparseVector<float64_t> *features1=NULL ;
	get_sparse_matrix(features1, dim11, dim12);

	int32_t dim21, dim22 ;
	SGSparseVector<float64_t> *features2=NULL ;
	get_sparse_matrix(features2, dim21, dim22);

	ASSERT(dim11==dim21)
	ASSERT(dim12==dim22)

	int32_t *Dims = SG_MALLOC(int32_t, 3);
	Dims[0]=dim11 ;
	Dims[1]=dim12 ;
	Dims[2]=2 ;

	ASSERT(Dims[0]==num_states)
	ASSERT(Dims[1]==num_pos)

	ASSERT(ui_structure->set_feature_matrix_sparse(features1, features2, Dims))
	ASSERT(ui_structure->set_feature_dims(Dims))

	SG_FREE(features1);
	SG_FREE(features2);
	SG_FREE(Dims);

	return true;
}
bool CSGInterface::cmd_init_intron_list()
{
	//ARG1 start_positions
	int32_t Nstart_positions;
	int32_t* start_positions;
	get_vector(start_positions, Nstart_positions);
        //SG_PRINT("Nstart_positions:%i\n",Nstart_positions)

	//ARG2 end_positions
	int32_t Nend_positions;
	int32_t* end_positions;
	get_vector(end_positions, Nend_positions);
        //SG_PRINT("Nend_positions:%i\n",Nend_positions)

	//ARG3 quality
	int32_t Nquality;
        int32_t* quality;
        get_vector(quality, Nquality);
        //SG_PRINT("Nquality:%i\n",Nquality)

	//ARG4 all candidate positions
	int32_t Nall_pos;
        int32_t* all_pos;
        get_vector(all_pos, Nall_pos);
        //SG_PRINT("Nall_pos:%i\n",Nall_pos)

	ASSERT(Nquality==Nend_positions)
	ASSERT(Nend_positions==Nstart_positions)

	CIntronList* intron_list = new CIntronList();

	intron_list->init_list(all_pos, Nall_pos);

	intron_list->read_introns(start_positions, end_positions, quality, Nstart_positions);

	SG_FREE(start_positions);
	SG_FREE(end_positions);
	SG_FREE(quality);
	SG_FREE(all_pos);

	//int32_t test;
	//int32_t testq;
	//intron_list->get_coverage(&test, &testq, 15 ,16);

	//SG_PRINT("coverage: %i, quality: %i\n",test, testq)

	CDynProg* h = ui_structure->get_dyn_prog();
	if (!h)
		SG_ERROR("no DynProg object found, use set_model first\n")

	h->set_intron_list(intron_list, 2);

	return true;
}
bool CSGInterface::cmd_precompute_tiling_features()
{
	CPlifMatrix* pm=ui_structure->get_plif_matrix();
	CPlif** PEN  = pm->get_PEN();
	CDynProg* h  = ui_structure->get_dyn_prog();

	int32_t Nintensities=0;
	float64_t* intensities;
	get_vector(intensities, Nintensities);

	int32_t Nprobe_pos=0;
	int32_t* probe_pos;
	get_vector(probe_pos, Nprobe_pos);
	ASSERT(Nprobe_pos==Nintensities)

	int32_t Ntiling_plif_ids=0;
	int32_t* tiling_plif_ids;
	get_vector(tiling_plif_ids, Ntiling_plif_ids);

	h->init_tiling_data(probe_pos,intensities, Nprobe_pos);
	h->precompute_tiling_plifs(PEN, tiling_plif_ids, Ntiling_plif_ids);
	return true;
}

bool CSGInterface::cmd_best_path_trans()
{
	CDynProg* h = ui_structure->get_dyn_prog();

	CSegmentLoss* seg_loss_obj = h->get_segment_loss_object();

	CPlifMatrix* pm=ui_structure->get_plif_matrix();

	int32_t num_states = h->get_num_states();
	int32_t* feat_dims = ui_structure->get_feature_dims();
	float64_t* features = (ui_structure->get_feature_matrix(false));
	CSparseFeatures<float64_t>* features_sparse1 = (ui_structure->get_feature_matrix_sparse(0));
	CSparseFeatures<float64_t>* features_sparse2 = (ui_structure->get_feature_matrix_sparse(1));
	int32_t* orf_info = ui_structure->get_orf_info();
	bool use_orf = ui_structure->get_use_orf();
	int32_t Nplif = pm->get_num_plifs();

	// ARG 1
	// transitions from initial state (#states x 1)
	int32_t Np=0;
	float64_t* p;
	get_vector(p, Np);
	if (Np!=num_states)
		SG_ERROR("# transitions from initial state (%i) does not match # states (%i)\n", Np, num_states)

	// ARG 2
	// transitions to end state (#states x 1)
	int32_t Nq=0;
	float64_t* q;
	get_vector(q, Nq);
	if (Nq!=num_states)
		SG_ERROR("# transitions to end state (%i) does not match # states (%i)\n", Nq, num_states)

	// ARG 3
	// number of best paths
	int32_t Nnbest=0;
	int32_t* all_nbest;
	get_vector(all_nbest, Nnbest);
	int32_t nbest;
	int32_t nother = 0;
	if (Nnbest==2)
	{
		nbest =all_nbest[0];
		nother=all_nbest[1];
	}
	else
		nbest =all_nbest[0];
	SG_FREE(all_nbest);

	// ARG 4
	// segment path (2 x #feature_positions)
	// masking/weighting of loss for specific
	// regions of the true path
	int32_t Nseg_path=0;
	int32_t Mseg_path=0;
	float64_t* seg_path;
	get_matrix(seg_path, Nseg_path, Mseg_path);

	// ARG 5
	// links for transitions (#transitions x 4)
	int32_t Na_trans=0;
	int32_t num_a_trans=0;
	float64_t* a_trans;
	get_matrix(a_trans, num_a_trans, Na_trans);

	// ARG 6
	// loss matrix (#segment x 2*#segments)
	// one (#segment x #segments)-matrix for segment loss
	// and one for nucleotide loss
	int32_t Nloss=0;
	int32_t Mloss=0;
	float64_t* loss;
	get_matrix(loss, Nloss,Mloss);

	int32_t M = h->get_num_positions();

	/////////////////////////////////////////////////////////////////////////////////
	// check input
	/////////////////////////////////////////////////////////////////////////////////
	ASSERT(num_states==Nq)

	CPlif** PEN=pm->get_PEN();
	ASSERT(PEN)

	h->set_p_vector(SGVector<float64_t>(p, num_states));
	h->set_q_vector(SGVector<float64_t>(q, num_states));

	if (seg_path!=NULL)
	{
		h->set_a_trans_matrix(SGMatrix<float64_t>(a_trans, num_a_trans, Na_trans)) ;
	}
	else
	{
		h->set_a_trans_matrix(SGMatrix<float64_t>(a_trans, num_a_trans, 3)) ; // segment_id = 0
	}

	if (!h->check_svm_arrays())
	{
		SG_ERROR("svm arrays inconsistent\n")
		CPlif::delete_penalty_struct(PEN, Nplif) ;
		return false ;
	}

	SG_DEBUG("best_path_trans: M: %i, Mseg_path: %i\n", M, Mseg_path)

	h->set_observation_matrix(SGNDArray<float64_t>(features, feat_dims, 3, false));

	if (seg_path!=NULL)
	{
		h->best_path_set_segment_loss(SGMatrix<float64_t>(loss, Nloss, Mloss, false)) ;
		seg_loss_obj->set_segment_loss(loss, Nloss, Mloss);
	}
	else
	{
		float64_t zero2[2] = {0.0, 0.0} ;
		h->best_path_set_segment_loss(SGMatrix<float64_t>(zero2, 2, 1)) ;
		seg_loss_obj->set_segment_loss(zero2, 2, 1);
	}
	h->set_content_type_array(SGMatrix<float64_t>(seg_path,Nseg_path,Mseg_path));

	bool segment_loss_non_zero=false;
	for (int32_t i=0; i<Nloss*Mloss; i++)
	{
		if (loss[i]>1e-3)
			segment_loss_non_zero=true;
	}

	SG_FREE(loss);
	loss=NULL;

	h->set_orf_info(SGMatrix<int32_t>(orf_info, num_states, 2));
	h->set_sparse_features(features_sparse1, features_sparse2);
	h->set_plif_matrices(pm);

	if (segment_loss_non_zero)
	{
	        SG_DEBUG("Using version with segment_loss\n")
	        if (nbest==1)
				h->compute_nbest_paths(feat_dims[2], use_orf, 1,true,false);
	        else
				h->compute_nbest_paths(feat_dims[2], use_orf, 2,true,false);
	}
	else
	{
	        SG_DEBUG("Using version without segment_loss\n")
	        if (nbest==1)
				h->compute_nbest_paths(feat_dims[2], use_orf, 1,false,false);
	        else
				h->compute_nbest_paths(feat_dims[2], use_orf, 2,false,false);
	}

	SGVector<float64_t> p_prob=h->get_scores();

	SGMatrix<int32_t> states=h->get_states();

	SGMatrix<int32_t> my_pos=h->get_positions();

	// transcribe result
	float64_t* d_my_path= SG_MALLOC(float64_t, (nbest+nother)*M);
	float64_t* d_my_pos= SG_MALLOC(float64_t, (nbest+nother)*M);

	for (int32_t k=0; k<(nbest+nother); k++)
	{
		for (int32_t i=0; i<M; i++)
		{
			d_my_path[i*(nbest+nother)+k] = states.matrix[i+k*M] ;
			d_my_pos[i*(nbest+nother)+k] = my_pos.matrix[i+k*M] ;
		}
	}
	set_vector(p_prob.vector,nbest+nother);
	set_vector(d_my_path, (nbest+nother)*M);
	set_vector(d_my_pos, (nbest+nother)*M);

	SG_FREE(d_my_path);
	SG_FREE(d_my_pos);

	return true;

}

bool CSGInterface::cmd_best_path_trans_deriv()
{
	int32_t num_states = ui_structure->get_num_states();
	int32_t* feat_dims = ui_structure->get_feature_dims();
	float64_t* features = (ui_structure->get_feature_matrix(false));

	CPlifMatrix* pm=ui_structure->get_plif_matrix();
	int32_t Nplif = pm->get_num_plifs();
	CPlif** PEN = pm->get_PEN();

	// ARG 1
	// transitions from initial state (#states x 1)
	int32_t Np=0;
	float64_t* p=NULL;
	get_vector(p, Np);
	if (Np!=num_states)
		SG_ERROR("Np!=num_states; Np:%i num_states:%i",Np,num_states)

	// ARG 2
	// transitions to end state (#states x 1)
	int32_t Nq=0;
	float64_t* q=NULL;
	get_vector(q, Nq);
	if (Nq!=num_states)
		SG_ERROR("Nq!=num_states; Nq:%i num_states:%i",Nq,num_states)


	// ARG 3
	// segment path (2 x #feature_positions)
	// masking/weighting of loss for specific
	// regions of the true path
	int32_t Nseg_path=0;
	int32_t Mseg_path=0;
	float64_t* seg_path;
	get_matrix(seg_path,Nseg_path,Mseg_path);

	// ARG 4
	// links for transitions (#transitions x 4)
	int32_t Na_trans=0;
	int32_t num_a_trans=0;
	float64_t* a_trans=NULL;
	get_matrix(a_trans, num_a_trans, Na_trans);

	// ARG 5
	// loss matrix (#segment x 2*#segments)
	// one (#segment x #segments)-matrix for segment loss
	// and one for nucleotide loss
	int32_t Nloss=0;
	int32_t Mloss=0;
	float64_t* loss=NULL;
	get_matrix(loss, Nloss,Mloss);

	// ARG 6
	// path to calc derivative for
	int32_t Nmystate_seq=0;
	int32_t* mystate_seq=NULL;
	get_vector(mystate_seq, Nmystate_seq);

	// ARG 7
	// positions of the path
	int32_t Nmypos_seq=0;
	int32_t* mypos_seq=NULL;
	get_vector(mypos_seq, Nmypos_seq);


	//a => a_trans

	int32_t max_plif_id = 0 ;
	int32_t max_plif_len = 1 ;
	for (int32_t i=0; i<Nplif; i++)
	{
		if (i>0 && PEN[i]->get_id()!=i)
			SG_ERROR("PEN[i]->get_id()!=i; PEN[%i]->get_id():%i  ,\n",i, PEN[i]->get_id())
		if (i>max_plif_id)
			max_plif_id=i ;
		if (PEN[i]->get_plif_len()>max_plif_len)
			max_plif_len=PEN[i]->get_plif_len() ;
	} ;


	CDynProg* h = ui_structure->get_dyn_prog();
	CSegmentLoss* seg_loss_obj = h->get_segment_loss_object();
	h->set_num_states(num_states) ;
	h->set_p_vector(SGVector<float64_t>(p, num_states)) ;
	h->set_q_vector(SGVector<float64_t>(q, num_states)) ;

	if (seg_path!=NULL)
		h->set_a_trans_matrix(SGMatrix<float64_t>(a_trans, num_a_trans, Na_trans)) ;
	else
		h->set_a_trans_matrix(SGMatrix<float64_t>(a_trans, num_a_trans, 3)) ;

	if (!h->check_svm_arrays())
		SG_ERROR("svm arrays inconsistent\n")

	int32_t *my_path = SG_MALLOC(int32_t, Nmypos_seq+1);
	memset(my_path, -1, Nmypos_seq*sizeof(int32_t)) ;
	int32_t *my_pos = SG_MALLOC(int32_t, Nmypos_seq+1);
	memset(my_pos, -1, Nmypos_seq*sizeof(int32_t)) ;

	h->set_observation_matrix(SGNDArray<float64_t>(features, feat_dims, 3));
	for (int32_t i=0; i<Nmypos_seq; i++)
	{
		my_path[i] = mystate_seq[i] ;
		my_pos[i]  = mypos_seq[i] ;
	}

	if (seg_path!=NULL)
	{
		h->best_path_set_segment_loss(SGMatrix<float64_t>(loss, Nloss, Mloss)) ;
		seg_loss_obj->set_segment_loss(loss, Nloss, Mloss);
	}
	else
	{
		float64_t zero2[2] = {0.0, 0.0} ;
		h->best_path_set_segment_loss(SGMatrix<float64_t>(zero2, 2, 1, false)) ;
		seg_loss_obj->set_segment_loss(zero2, 2, 1);
	}
	h->set_content_type_array(SGMatrix<float64_t>(seg_path,Nseg_path,Mseg_path));

	float64_t* p_Plif_deriv = SG_MALLOC(float64_t, (max_plif_id+1)*max_plif_len);
	CDynamicArray <float64_t> a_Plif_deriv(p_Plif_deriv, max_plif_id+1, max_plif_len, false, false) ; // 2d

	float64_t* p_A_deriv   = SG_MALLOC(float64_t, num_states*num_states);
	float64_t* p_p_deriv   = SG_MALLOC(float64_t, num_states);
	float64_t* p_q_deriv   = SG_MALLOC(float64_t, num_states);

	h->set_plif_matrices(pm);
	h->best_path_trans_deriv(my_path, my_pos, Nmypos_seq, features, feat_dims[2]);

	float64_t* p_my_scores;
	int32_t n_scores;
	h->get_path_scores(&p_my_scores, &n_scores);

	float64_t* p_my_losses;
	int32_t n_losses;
	h->get_path_losses(&p_my_losses, &n_losses);

	for (int32_t i=0; i<num_states; i++)
	{
		for (int32_t j=0; j<num_states; j++)
			p_A_deriv[i+j*num_states] = h->get_a_deriv(i, j) ;

		p_p_deriv[i]=h->get_p_deriv(i) ;
		p_q_deriv[i]=h->get_q_deriv(i) ;
	}

	for (int32_t id=0; id<=max_plif_id; id++)
	{
		int32_t len=0 ;
		const float64_t * deriv = PEN[id]->get_cum_derivative(len) ;
		ASSERT(len<=max_plif_len)
		for (int32_t j=0; j<max_plif_len; j++)
			a_Plif_deriv.element(id, j)= deriv[j] ;
	}

	set_vector(p_p_deriv, num_states);
	set_vector(p_q_deriv, num_states);
	set_matrix(p_A_deriv, num_states, num_states);
	set_matrix(p_Plif_deriv, (max_plif_id+1), max_plif_len);
	set_vector(p_my_scores, Nmypos_seq);
	set_vector(p_my_losses, Nmypos_seq);

	SG_FREE(p_A_deriv);
	SG_FREE(p_p_deriv);
	SG_FREE(p_q_deriv);
	SG_FREE(p_Plif_deriv);
	free(p_my_scores);
	free(p_my_losses);

	SG_FREE(my_path);
	SG_FREE(my_pos);

	SG_FREE(p);
	SG_FREE(q);
	SG_FREE(a_trans);
	SG_FREE(loss);
	SG_FREE(mystate_seq);
	SG_FREE(mypos_seq);

	return true ;
}

bool CSGInterface::cmd_precompute_subkernels()
{
	if (m_nrhs!=1 || !create_return_values(0))
		return false;

	return ui_kernel->precompute_subkernels();
}
bool CSGInterface::cmd_crc()
{
	if (m_nrhs!=2 || !create_return_values(1))
		return false;

	int32_t slen=0;
	char* string=get_string(slen);
	ASSERT(string)
	uint8_t* bstring=SG_MALLOC(uint8_t, slen);

	for (int32_t i=0; i<slen; i++)
		bstring[i]=string[i];
	SG_FREE(string);

	int32_t val=CHash::crc32(bstring, slen);
	SG_FREE(bstring);
	set_int(val);

	return true;
}

bool CSGInterface::cmd_system()
{
	if (m_nrhs<2 || !create_return_values(0))
		return false;

	int32_t len=0;
	char* command=SG_MALLOC(char, 10000);
	memset(command, 0, sizeof(char)*10000);
	char* cmd=get_str_from_str_or_direct(len);
	strncat(command, cmd, 10000);
	SG_FREE(cmd);

	while (m_rhs_counter<m_nrhs)
	{
		strncat(command, " ", 10000);
		char* arg=get_str_from_str_or_direct(len);
		strncat(command, arg, 10000);
		SG_FREE(arg);
	}

	int32_t success=system(command);

	return (success==0);
}

bool CSGInterface::cmd_exit()
{
	exit(0);
	return 0; //never reached but necessary to keep sun compiler happy
}

bool CSGInterface::cmd_exec()
{
	if (m_nrhs<2 || !create_return_values(0))
		return false;

	int32_t len=0;
	char* filename=get_str_from_str_or_direct(len);
	FILE* file=fopen(filename, "r");
	if (!file)
	{
		SG_FREE(filename);
		SG_ERROR("Error opening file: %s.\n", filename)
	}

	while (!feof(file))
	{
		// FIXME: interpret lines as input
		break;
	}

	fclose(file);
	return true;
}

bool CSGInterface::cmd_set_output()
{
	if (m_nrhs<2 || !create_return_values(0))
		return false;

	int32_t len=0;
	char* filename=get_str_from_str_or_direct(len);

	if (file_out)
		fclose(file_out);
	file_out=NULL;

	SG_INFO("Setting output file to: %s.\n", filename)

	if (strmatch(filename, "STDERR"))
		io->set_target(stderr);
	else if (strmatch(filename, "STDOUT"))
		io->set_target(stdout);
	else
	{
		file_out=fopen(filename, "w");
		if (!file_out)
			SG_ERROR("Error opening output file %s.\n", filename)
		io->set_target(file_out);
	}

	return true;
}

bool CSGInterface::cmd_set_threshold()
{
	if (m_nrhs!=2 || !create_return_values(0))
		return false;

	float64_t value=get_real_from_real_or_str();

	ui_math->set_threshold(value);
	return true;
}

bool CSGInterface::cmd_init_random()
{
	if (m_nrhs!=2 || !create_return_values(0))
		return false;

	uint32_t initseed=(uint32_t) get_int_from_int_or_str();
	ui_math->init_random(initseed);

	return true;
}

bool CSGInterface::cmd_set_num_threads()
{
	if (m_nrhs!=2 || !create_return_values(0))
		return false;

	int32_t num_threads=get_int_from_int_or_str();

	parallel->set_num_threads(num_threads);
	SG_INFO("Set number of threads to %d.\n", num_threads)

	return true;
}

bool CSGInterface::cmd_translate_string()
{
	if (m_nrhs!=4 || !create_return_values(1))
		return false;

	float64_t* string=NULL;
	int32_t len;
	get_vector(string, len);

	int32_t order=get_int();
	int32_t start=get_int();

	const int32_t max_val=2; /* DNA->2bits */
	int32_t i,j;
	uint16_t* obs=SG_MALLOC(uint16_t, len);

	for (i=0; i<len; i++)
	{
		switch ((char) string[i])
		{
			case 'A': obs[i]=0; break;
			case 'C': obs[i]=1; break;
			case 'G': obs[i]=2; break;
			case 'T': obs[i]=3; break;
			case 'a': obs[i]=0; break;
			case 'c': obs[i]=1; break;
			case 'g': obs[i]=2; break;
			case 't': obs[i]=3; break;
			default: SG_ERROR("Wrong letter in string.\n")
		}
	}

	//convert interval of size T
	for (i=len-1; i>=order-1; i--)
	{
		uint16_t value=0;
		for (j=i; j>=i-order+1; j--)
			value=(value>>max_val) | ((obs[j])<<(max_val*(order-1)));

		obs[i]=(uint16_t) value;
	}

	for (i=order-2;i>=0;i--)
	{
		uint16_t value=0;
		for (j=i; j>=i-order+1; j--)
		{
			value= (value >> max_val);
			if (j>=0)
				value|=(obs[j]) << (max_val * (order-1));
		}
		obs[i]=value;
	}

	float64_t* real_obs=SG_MALLOC(float64_t, len);
	for (i=start; i<len; i++)
		real_obs[i-start]=(float64_t) obs[i];
	SG_FREE(obs);

	set_vector(real_obs, len);
	SG_FREE(real_obs);

	return true;
}

bool CSGInterface::cmd_clear()
{
	// reset guilib
	SG_UNREF(ui_classifier);
	ui_classifier=new CGUIClassifier(this);
	SG_UNREF(ui_distance);
	ui_distance=new CGUIDistance(this);
	SG_UNREF(ui_features);
	ui_features=new CGUIFeatures(this);
	SG_UNREF(ui_hmm);
	ui_hmm=new CGUIHMM(this);
	SG_UNREF(ui_kernel);
	ui_kernel=new CGUIKernel(this);
	SG_UNREF(ui_labels);
	ui_labels=new CGUILabels(this);
	SG_UNREF(ui_math);
	ui_math=new CGUIMath(this);
	SG_UNREF(ui_pluginestimate);
	ui_pluginestimate=new CGUIPluginEstimate(this);
	SG_UNREF(ui_preproc);
	ui_preproc=new CGUIPreprocessor(this);
	SG_UNREF(ui_time);
	ui_time=new CGUITime(this);

	return true;
}

bool CSGInterface::cmd_tic()
{
	ui_time->start();
	return true;
}

bool CSGInterface::cmd_toc()
{
	ui_time->stop();
	return true;
}

bool CSGInterface::cmd_print()
{
	if (m_nrhs<2 || !create_return_values(0))
		return false;

	int32_t len=0;
	char* msg=get_str_from_str_or_direct(len);

	SG_PRINT("%s\n", msg)

	SG_FREE(msg);
	return true;
}

bool CSGInterface::cmd_echo()
{
	if (m_nrhs<2 || !create_return_values(0))
		return false;

	int32_t len=0;
	char* level=get_str_from_str_or_direct(len);

	if (strmatch(level, "OFF"))
	{
		echo=false;
		SG_INFO("Echo is off.\n")
	}
	else
	{
		echo=true;
		SG_INFO("Echo is on.\n")
	}

	SG_FREE(level);
	return true;
}

bool CSGInterface::cmd_loglevel()
{
	if (m_nrhs<2 || !create_return_values(0))
		return false;

	int32_t len=0;
	char* level=get_str_from_str_or_direct(len);

	if (strmatch(level, "ALL") || strmatch(level, "GCDEBUG"))
		io->set_loglevel(MSG_GCDEBUG);
	else if (strmatch(level, "DEBUG"))
		io->set_loglevel(MSG_DEBUG);
	else if (strmatch(level, "INFO"))
		io->set_loglevel(MSG_INFO);
	else if (strmatch(level, "NOTICE"))
		io->set_loglevel(MSG_NOTICE);
	else if (strmatch(level, "WARN"))
		io->set_loglevel(MSG_WARN);
	else if (strmatch(level, "ERROR"))
		io->set_loglevel(MSG_ERROR);
	else if (strmatch(level, "CRITICAL"))
		io->set_loglevel(MSG_CRITICAL);
	else if (strmatch(level, "ALERT"))
		io->set_loglevel(MSG_ALERT);
	else if (strmatch(level, "EMERGENCY"))
		io->set_loglevel(MSG_EMERGENCY);
	else
		SG_ERROR("Unknown loglevel '%s'.\n", level)

	SG_INFO("Loglevel set to %s.\n", level)

	SG_FREE(level);
	return true;
}

bool CSGInterface::cmd_syntax_highlight()
{
	if (m_nrhs<2 || !create_return_values(0))
		return false;

	int32_t len=0;
	char* hili=get_str_from_str_or_direct(len);

	if (strmatch(hili, "ON"))
	{
		hilight.set_ansi_syntax_hilighting();
		io->enable_syntax_highlighting();
	}
	else if (strmatch(hili, "OFF"))
	{
		hilight.disable_syntax_hilighting();
		io->disable_syntax_highlighting();
	}
	else
		SG_ERROR("arguments to " N_SYNTAX_HIGHLIGHT " are ON|OFF - found '%s'.\n", hili)

	SG_INFO("Syntax hilighting set to %s.\n", hili)

	SG_FREE(hili);
	return true;
}

bool CSGInterface::cmd_progress()
{
	if (m_nrhs<2 || !create_return_values(0))
		return false;

	int32_t len=0;
	char* progress=get_str_from_str_or_direct(len);

	if (strmatch(progress, "ON"))
		io->enable_progress();
	else if (strmatch(progress, "OFF"))
		io->disable_progress();
	else
		SG_ERROR("arguments to progress are ON|OFF - found '%s'.\n", progress)

	SG_INFO("Progress set to %s.\n", progress)

	SG_FREE(progress);
	return true;
}

bool CSGInterface::cmd_get_version()
{
	if (m_nrhs!=1 || !create_return_values(1))
		return false;

	set_int(version->get_version_revision());

	return true;
}

bool CSGInterface::cmd_help()
{
	if ((m_nrhs!=1 && m_nrhs!=2) || !create_return_values(0))
		return false;

	int32_t i=0;

	SG_PRINT("\n")
	if (m_nrhs==1) // unspecified help
	{
		SG_PRINT("Help is available for the following topics.\n"
				 "-------------------------------------------\n\n");
		while (sg_methods[i].command)
		{
			bool is_group_item=false;
			if (!sg_methods[i].method && !sg_methods[i].usage_prefix)
				is_group_item=true;

			if (is_group_item)
			{
				SG_PRINT("%s%s%s\n",
						hilight.get_command_prefix(),
						sg_methods[i].command,
						hilight.get_command_suffix());
			}

			i++;
		}
		SG_PRINT("\nUse sg('%shelp%s', '%s<topic>%s')"
				" to see the list of commands in this group, e.g.\n\n"
				"\tsg('%shelp%s', '%sFeatures%s')\n\n"
				"to see the list of commands for the 'Features' group.\n"
				"\nOr use sg('%shelp%s', '%sall%s')"
				" to see a brief listing of all commands.\n\nTo disable syntax"
				" highlighting (useful e.g. in the matlab GUI) use\n\n"
				"\tsg('syntax_highlight','OFF')\n",
					hilight.get_command_prefix(), hilight.get_command_suffix(),
					hilight.get_command_prefix(), hilight.get_command_suffix(),
					hilight.get_command_prefix(), hilight.get_command_suffix(),
					hilight.get_command_prefix(), hilight.get_command_suffix(),
					hilight.get_command_prefix(), hilight.get_command_suffix(),
					hilight.get_command_prefix(), hilight.get_command_suffix());
	}
	else // m_nrhs == 2 -> all commands, single command or group help
	{
		bool found=false;
		bool in_group=false;
		int32_t clen=0;
		char* command=get_string(clen);

		if (strmatch("doxygen", command) || strmatch("DOXYGEN", command))
		{
			found=true;
			while (sg_methods[i].command)
			{
				if (sg_methods[i].usage_prefix) // display group item
				{
					SG_PRINT("\\arg \\b %s \\verbatim %s%s%s \\endverbatim\n",
							sg_methods[i].command,
							sg_methods[i].usage_prefix,
							sg_methods[i].command,
							sg_methods[i].usage_suffix);
				}
				else if (!sg_methods[i].method) // display group
				{
					SG_PRINT("\n\\section %s_sec %s\n",
							sg_methods[i].command, sg_methods[i].command);
				}
				i++;
			}
		}
		if (strmatch("all", command) || strmatch("ALL", command))
		{
			found=true;
			while (sg_methods[i].command)
			{
				if (sg_methods[i].usage_prefix) // display group item
				{
					SG_PRINT("\t%s%s%s%s%s\n", sg_methods[i].usage_prefix,
							hilight.get_command_prefix(),
							sg_methods[i].command,
							hilight.get_command_suffix(),
							sg_methods[i].usage_suffix);
				}
				else if (!sg_methods[i].method) // display group
				{
					SG_PRINT("\nCommands in group %s%s%s\n",
							hilight.get_command_prefix(),
							sg_methods[i].command,
							hilight.get_command_suffix());
				}
				i++;
			}
		}
		else
		{
			while (sg_methods[i].command)
			{
				if (in_group)
				{
					if (sg_methods[i].usage_prefix) // display group item
						SG_PRINT("\t%s%s%s\n",
							hilight.get_command_prefix(),
							sg_methods[i].command,
							hilight.get_command_suffix())
					else // next group reached -> end
						break;
				}
				else
				{
					found=strmatch(sg_methods[i].command, command);
					if (found)
					{
						if (sg_methods[i].usage_prefix) // found item
						{
							SG_PRINT("Usage for %s%s%s\n\n\t%s%s%s%s%s\n",
									hilight.get_command_prefix(),
									sg_methods[i].command,
									hilight.get_command_suffix(),
									sg_methods[i].usage_prefix,
									hilight.get_command_prefix(),
									sg_methods[i].command,
									hilight.get_command_suffix(),
									sg_methods[i].usage_suffix)
							break;
						}
						else // found group item
						{
							SG_PRINT("Commands in group %s%s%s\n\n",
									hilight.get_command_prefix(),
									sg_methods[i].command,
									hilight.get_command_suffix())
							in_group=true;
						}
					}
				}

				i++;
			}
		}

		if (!found)
			SG_PRINT("Could not find help for command %s.\n", command)
		else if (in_group)
		{
			SG_PRINT("\n\nUse sg('%shelp%s', '%s<command>%s')"
					" to see the usage pattern of a single command, e.g.\n\n"
					"\tsg('%shelp%s', '%sclassify%s')\n\n"
					" to see the usage pattern of the command 'classify'.\n",
					hilight.get_command_prefix(), hilight.get_command_suffix(),
					hilight.get_command_prefix(), hilight.get_command_suffix(),
					hilight.get_command_prefix(), hilight.get_command_suffix(),
					hilight.get_command_prefix(), hilight.get_command_suffix());
		}

		SG_FREE(command);
	}


	SG_PRINT("\n")

	return true;
}
#ifdef TRACE_MEMORY_ALLOCS
	extern CMap<void*, MemoryBlock>* sg_mallocs;
#endif

bool CSGInterface::cmd_whos()
{
	if ((m_nrhs!=1) || !create_return_values(0))
		return false;

#ifdef TRACE_MEMORY_ALLOCS
	SG_PRINT("Blocks allocated by shogun\n")
	list_memory_allocs();
	SG_PRINT("\n")
	return true;
#else
	SG_PRINT("Requires shogun to be compiled with --enable-trace-mallocs\n")
	return false;
#endif
}

bool CSGInterface::cmd_send_command()
{
	SG_DEPRECATED

	int32_t len=0;
	char* arg=get_string(len);
	//SG_DEBUG("legacy: arg == %s\n", arg)
	m_legacy_strptr=arg;

	char* command=get_str_from_str(len);
	int32_t i=0;
	bool success=false;

	while (sg_methods[i].command)
	{
		if (strmatch(command, sg_methods[i].command))
		{
			SG_DEBUG("legacy: found command %s\n", sg_methods[i].command)
			// fix-up m_nrhs; +1 to include command
			m_nrhs=get_num_args_in_str()+1;

			if (!(interface->*(sg_methods[i].method))())
			{
				SG_ERROR("Usage: %s%s%s\n\n\t%s%s%s%s%s\n",
						hilight.get_command_prefix(),
						sg_methods[i].command,
						hilight.get_command_suffix(),
						sg_methods[i].usage_prefix,
						hilight.get_command_prefix(),
						sg_methods[i].command,
						hilight.get_command_suffix(),
						sg_methods[i].usage_suffix);
			}
			else
			{
				success=true;
				break;
			}
		}

		i++;
	}

	if (!success)
		SG_ERROR("Non-supported legacy command %s.\n", command)

	SG_FREE(command);
	SG_FREE(arg);
	return success;
}

bool CSGInterface::cmd_run_python()
{
	SG_ERROR("Only available in the elwms interface\n")
	return false;
}

bool CSGInterface::cmd_run_octave()
{
	SG_ERROR("Only available in the elwms interface\n")
	return false;
}

bool CSGInterface::cmd_run_r()
{
	SG_ERROR("Only available in the elwms interface\n")
	return false;
}

bool CSGInterface::cmd_pr_loqo()
{
	if (m_nrhs!=7 || !create_return_values(2))
		return false;

	float64_t* c=NULL;
	int32_t lenc=0;
	get_vector(c, lenc);

	int32_t n = lenc;

	float64_t* H=NULL;
	int32_t nH=0;
	int32_t mH=0;
	get_matrix(H, nH, mH);
	ASSERT(nH==n && mH==n)

	float64_t* A=NULL;
	int32_t nA=0;
	int32_t mA=0;
	get_matrix(A, nA, mA);
	ASSERT(mA==n)
	int32_t m=nA;

	float64_t* b=NULL;
	int32_t lenb=0;
	get_vector(b, lenb);
	ASSERT(lenb==m)

	float64_t* l=NULL;
	int32_t lenl=0;
	get_vector(l, lenl);
	ASSERT(lenl==n)

	float64_t* u=NULL;
	int32_t lenu=0;
	get_vector(u, lenu);
	ASSERT(lenu==n)

	float64_t* x=SG_MALLOC(float64_t, 3*n);
	SGVector<float64_t>::fill_vector(x, 3*n, 0.0);

	float64_t* y=SG_MALLOC(float64_t, m+2*n);
	SGVector<float64_t>::fill_vector(y, m+2*n, 0.0);

	pr_loqo(n,m, c, H, A, b, l, u, x, y, 0, 5, 50, 0.05, 100, 0);

	set_vector(x, n);
	set_vector(y, m);

	SG_FREE(c);
	SG_FREE(H);
	SG_FREE(A);
	SG_FREE(b);
	SG_FREE(l);
	SG_FREE(u);
	SG_FREE(x);
	SG_FREE(y);
	return true;
}

void CSGInterface::print_prompt()
{
	SG_PRINT("%sshogun%s >> ",
			hilight.get_prompt_prefix(),
			hilight.get_prompt_suffix());
}

////////////////////////////////////////////////////////////////////////////
// legacy-related methods
////////////////////////////////////////////////////////////////////////////

char* CSGInterface::get_str_from_str_or_direct(int32_t& len)
{
	if (m_legacy_strptr)
		return get_str_from_str(len);
	else
		return get_string(len);
}

int32_t CSGInterface::get_int_from_int_or_str()
{
	if (m_legacy_strptr)
	{
		int32_t len=0;
		char* str=get_str_from_str(len);
		int32_t val=strtol(str, NULL, 10);

		SG_FREE(str);
		return val;
	}
	else
		return get_int();
}

float64_t CSGInterface::get_real_from_real_or_str()
{
	if (m_legacy_strptr)
	{
		int32_t len=0;
		char* str=get_str_from_str(len);
		float64_t val=strtod(str, NULL);

		SG_FREE(str);
		return val;
	}
	else
		return get_real();
}

bool CSGInterface::get_bool_from_bool_or_str()
{
	if (m_legacy_strptr)
	{
		int32_t len=0;
		char* str=get_str_from_str(len);
		bool val=strtol(str, NULL, 10)!=0;

		SG_FREE(str);
		return val;
	}
	else
		return get_bool();
}

void CSGInterface::get_vector_from_int_vector_or_str(int32_t*& vector, int32_t& len)
{
	if (m_legacy_strptr)
	{
		len=get_vector_len_from_str(len);
		if (len==0)
		{
			vector=NULL;
			return;
		}

		vector=SG_MALLOC(int32_t, len);
		char* str=NULL;
		int32_t slen=0;
		for (int32_t i=0; i<len; i++)
		{
			str=get_str_from_str(slen);
			vector[i]=strtol(str, NULL, 10);
			//SG_DEBUG("vec[%d]: %d\n", i, vector[i])
			SG_FREE(str);
		}
	}
	else
		get_vector(vector, len);
}

void CSGInterface::get_vector_from_real_vector_or_str(
	float64_t*& vector, int32_t& len)
{
	if (m_legacy_strptr)
	{
		len=get_vector_len_from_str(len);
		if (len==0)
		{
			vector=NULL;
			return;
		}

		vector=SG_MALLOC(float64_t, len);
		char* str=NULL;
		int32_t slen=0;
		for (int32_t i=0; i<len; i++)
		{
			str=get_str_from_str(slen);
			vector[i]=strtod(str, NULL);
			//SG_DEBUG("vec[%d]: %f\n", i, vector[i])
			SG_FREE(str);
		}
	}
	else
		get_vector(vector, len);
}

int32_t CSGInterface::get_vector_len_from_str(int32_t expected_len)
{
	int32_t num_args=get_num_args_in_str();

	if (expected_len==0 || num_args==expected_len)
		return num_args;
	else if (num_args==2*expected_len)
	{
		// special case for position_weights; a bit shaky...
		return expected_len;
	}
	else
		SG_ERROR("Expected vector length %d does not match actual length %d.\n", expected_len, num_args)

	return 0;
}

char* CSGInterface::get_str_from_str(int32_t& len)
{
	if (!m_legacy_strptr)
		return NULL;

	int32_t i=0;
	while (m_legacy_strptr[i]!='\0' && !isspace(m_legacy_strptr[i]))
		i++;

	len=i;
	char* str=SG_MALLOC(char, len+1);
	for (i=0; i<len; i++)
		str[i]=m_legacy_strptr[i];
	str[len]='\0';

	// move legacy strptr
	if (m_legacy_strptr[len]=='\0')
		m_legacy_strptr=NULL;
	else
	{
		m_legacy_strptr=m_legacy_strptr+len;
		m_legacy_strptr=SGIO::skip_spaces(m_legacy_strptr);
	}

	return str;
}

int32_t CSGInterface::get_num_args_in_str()
{
	if (!m_legacy_strptr)
		return 0;

	int32_t count=0;
	int32_t i=0;
	bool in_arg=false;
	while (m_legacy_strptr[i]!='\0')
	{
		if (!isspace(m_legacy_strptr[i]) && !in_arg)
		{
			count++;
			in_arg=true;
		}
		else if (isspace(m_legacy_strptr[i]) && in_arg)
			in_arg=false;

		i++;
	}

	return count;
}

////////////////////////////////////////////////////////////////////////////
// handler
////////////////////////////////////////////////////////////////////////////

bool CSGInterface::handle()
{
	int32_t len=0;
	bool success=false;

#ifndef WIN32
	CSignal::set_handler();
#endif

	char* command=NULL;
	command=interface->get_command(len);

	SG_DEBUG("command: %s, nrhs %d\n", command, m_nrhs)
	int32_t i=0;
	while (sg_methods[i].command)
	{
		if (strmatch(command, sg_methods[i].command))
		{
			SG_DEBUG("found command %s%s%s\n",
					hilight.get_command_prefix(),
					sg_methods[i].command,
					hilight.get_command_suffix());

			if (!(interface->*(sg_methods[i].method))())
			{
				if (sg_methods[i].usage_prefix)
				{
					SG_ERROR("Usage: %s%s%s\n\n\t%s%s%s%s%s\n",
							hilight.get_command_prefix(),
							sg_methods[i].command,
							hilight.get_command_suffix(),
							sg_methods[i].usage_prefix,
							hilight.get_command_prefix(),
							sg_methods[i].command,
							hilight.get_command_suffix(),
							sg_methods[i].usage_suffix);
				}
				else
					SG_ERROR("Non-supported command %s%s%s.\n",
							hilight.get_command_prefix(),
							sg_methods[i].command,
							hilight.get_command_suffix());
			}
			else
			{
				success=true;
				break;
			}
		}
		i++;
	}

#ifndef WIN32
	CSignal::unset_handler();
#endif

	if (!success)
		SG_ERROR("Unknown command %s%s%s.\n",
				hilight.get_command_prefix(),
				command,
				hilight.get_command_suffix());

	SG_FREE(command);
	return success;
}
