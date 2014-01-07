#ifndef __SGINTERFACE__H_
#define __SGINTERFACE__H_

#include <lib/config.h>
#include <lib/common.h>
#include <base/SGObject.h>
#include <features/Features.h>
#include <features/StringFeatures.h>
#include <features/SparseFeatures.h>
#include <features/AttributeFeatures.h>
#include <kernel/Kernel.h>

#include <ui/GUIClassifier.h>
#include <ui/GUIDistance.h>
#include <ui/GUIFeatures.h>
#include <ui/GUIHMM.h>
#include <ui/GUIKernel.h>
#include <ui/GUILabels.h>
#include <ui/GUIMath.h>
#include <ui/GUIPluginEstimate.h>
#include <ui/GUIPreprocessor.h>
#include <ui/GUITime.h>
#include <ui/GUIStructure.h>
#include <ui/GUIConverter.h>

namespace shogun
{
/** Interface datatypes that shogun supports. Note that some interfaces like e.g.
 * octave/matlab cannot distinguish between scalars and matrices and thus might
 * always return more complex types like matrices.
 */
enum IFType
{
	/// undefined
	UNDEFINED,

	///simple scalar/string types
	SCALAR_INT,
	SCALAR_REAL,
	SCALAR_BOOL,
	STANDARD_STRING,

	///vector type
	VECTOR_BOOL,
	VECTOR_BYTE,
	VECTOR_CHAR,
	VECTOR_INT,
	VECTOR_REAL,
	VECTOR_SHORTREAL,
	VECTOR_SHORT,
	VECTOR_WORD,

	///dense matrices
	DENSE_INT,
	DENSE_REAL,
	DENSE_SHORTREAL,
	DENSE_SHORT,
	DENSE_WORD,

	///dense nd arrays
	NDARRAY_BYTE,
	NDARRAY_CHAR,
	NDARRAY_INT,
	NDARRAY_REAL,
	NDARRAY_SHORTREAL,
	NDARRAY_SHORT,
	NDARRAY_WORD,

	///sparse matrices
	SPARSE_BYTE,
	SPARSE_CHAR,
	SPARSE_INT,
	SPARSE_REAL,
	SPARSE_SHORT,
	SPARSE_SHORTREAL,
	SPARSE_WORD,

	///strings of arbitrary type
	STRING_BYTE,
	STRING_CHAR,
	STRING_INT,
	STRING_SHORT,
	STRING_WORD,

	/// structures
	ATTR_STRUCT
};

/** objective enumerate */
enum E_WHICH_OBJ
{
	/// svm primal task
	SVM_PRIMAL,
	/// svm dual task
	SVM_DUAL,
	/// mkl primal task
	MKL_PRIMAL,
	/// mkl dual task
	MKL_DUAL,
	/// mkl relative duality gap
	MKL_RELATIVE_DUALITY_GAP,
	/// mkl absolute duality gap
	MKL_ABSOLUTE_DUALITY_GAP
};

/** @brief shogun interface */
class CSGInterface : public CSGObject
{
	public:
		/** constructor
		 * @param print_copyrights
		 */
		CSGInterface(bool print_copyrights=true);

		/** destructor */
		~CSGInterface();

		/// reset to clean state
		virtual void reset();

		/// translate matrix from language A to language B
		void translate_arg(CSGInterface* source, CSGInterface* target);

		/* commands */
		/** load features from file */
		bool cmd_load_features();
		/** save features to file */
		bool cmd_save_features();
		/** clear/clean features */
		bool cmd_clean_features();
		/** get features */
		bool cmd_get_features();
		/** add features */
		bool cmd_add_features();
		/** add multiple features */
		bool cmd_add_multiple_features();
		/** add dot features */
		bool cmd_add_dotfeatures();
		/** set features */
		bool cmd_set_features();
		/** set reference features */
		bool cmd_set_reference_features();
		/** del last features from combined features */
		bool cmd_del_last_features();
		/** convert features */
		bool cmd_convert();
		/** reshape features */
		bool cmd_reshape();
		/** load labels from file */
		bool cmd_load_labels();
		/** set labels */
		bool cmd_set_labels();
		/** get labels */
		bool cmd_get_labels();

		/** set kernel normalization */
		bool cmd_set_kernel_normalization();
		/** set kernel */
		bool cmd_set_kernel();
		/** add kernel (to e.g. CombinedKernel) */
		bool cmd_add_kernel();
		/** delete last kernel from combined kernel */
		bool cmd_del_last_kernel();
		/** initialize kernel */
		bool cmd_init_kernel();
		/** clear/clean kernel */
		bool cmd_clean_kernel();
		/** save kernel to file */
		bool cmd_save_kernel();
		/** load kernel init from file */
		bool cmd_load_kernel_init();
		/** save kernel init to file */
		bool cmd_save_kernel_init();
		/** get kernel matrix */
		bool cmd_get_kernel_matrix();
		/** set WD position weights */
		bool cmd_set_WD_position_weights();
		/** get subkernel weights */
		bool cmd_get_subkernel_weights();
		/** set subkernel weights */
		bool cmd_set_subkernel_weights();
		/** set subkernel weights combined */
		bool cmd_set_subkernel_weights_combined();
		/** get dotfeature weights combined */
		bool cmd_get_dotfeature_weights_combined();
		/** set dotfeature weights combined */
		bool cmd_set_dotfeature_weights_combined();
		/** set last subkernel weights */
		bool cmd_set_last_subkernel_weights();
		/** get WD position weights */
		bool cmd_get_WD_position_weights();
		/** get last subkernel weights */
		bool cmd_get_last_subkernel_weights();
		/** compute by subkernels */
		bool cmd_compute_by_subkernels();
		/** initialize kernel optimization */
		bool cmd_init_kernel_optimization();
		/** get kernel optimization */
		bool cmd_get_kernel_optimization();
		/** delete kernel optimization */
		bool cmd_delete_kernel_optimization();
		/** set diagonal speedup */
		bool cmd_use_diagonal_speedup();
		/** set kernel optimization type */
		bool cmd_set_kernel_optimization_type();
		/** set solver type */
		bool cmd_set_solver();
		/** set constraint generator */
		bool cmd_set_constraint_generator();
		/** set Salzberg prior probs */
		bool cmd_set_prior_probs();
		/** set Salzberg prior probs from labels */
		bool cmd_set_prior_probs_from_labels();
#ifdef USE_SVMLIGHT
		/** resize kernel cache */
		bool cmd_resize_kernel_cache();
#endif //USE_SVMLIGHT


		/** set distance */
		bool cmd_set_distance();
		/** init distance */
		bool cmd_init_distance();
		/** get distance matrix */
		bool cmd_get_distance_matrix();

		/** get SPEC consensus */
		bool cmd_get_SPEC_consensus();
		/** get SPEC scoring */
		bool cmd_get_SPEC_scoring();
		/** get WD consensus */
		bool cmd_get_WD_consensus();
		/** compute POIM WD */
		bool cmd_compute_POIM_WD();
		/** get WD scoring */
		bool cmd_get_WD_scoring();

		/** create new SVM/classifier */
		bool cmd_new_classifier();
		/** load SVM/classifier */
		bool cmd_load_classifier();
		/** save SVM/classifier */
		bool cmd_save_classifier();
		/** get SVM */
		bool cmd_get_svm();
		/** get number of SVMs in MultiClass */
		bool cmd_get_num_svms();
		/** set SVM */
		bool cmd_set_svm();
		/** set linear classifier */
		bool cmd_set_linear_classifier();
		/** classify */
		bool cmd_classify();
		/** classify example */
		bool cmd_classify_example();
		/** get classifier */
		bool cmd_get_classifier();
		/** get SVM objective */
		bool cmd_get_svm_objective();
		/** compute SVM objective from scratch*/
		bool cmd_compute_svm_primal_objective();
		/** compute SVM objective from scratch*/
		bool cmd_compute_svm_dual_objective();
		/** compute SVM objective from scratch*/
		bool cmd_compute_mkl_dual_objective();
		/** compute relative mkl duality gap */
		bool cmd_compute_relative_mkl_duality_gap();
		/** compute absolute mkl duality gap */
		bool cmd_compute_absolute_mkl_duality_gap();
		/** train classifier/SVM */
		bool cmd_train_classifier();
		/** do AUC maximization */
		bool cmd_do_auc_maximization();
		/** set perceptron parameters */
		bool cmd_set_perceptron_parameters();
		/** set SVM qpsize */
		bool cmd_set_svm_qpsize();
		/** set SVM max qpsize */
		bool cmd_set_svm_max_qpsize();
		/** set SVM bufsize */
		bool cmd_set_svm_bufsize();
		/** set SVM C */
		bool cmd_set_svm_C();
		/** set svm epsilon */
		bool cmd_set_svm_epsilon();
		/** set SVR tube epsilon */
		bool cmd_set_svr_tube_epsilon();
		/** set SVM OneClass nu */
		bool cmd_set_svm_nu();
		/** set SVM MKL parameters */
		bool cmd_set_svm_mkl_parameters();
		/** set ElasticnetMKL parameter lambda */
		bool cmd_set_elasticnet_lambda();
		/** set block norm parameter for block norm mkl */
		bool cmd_set_mkl_block_norm();
		/** set max train time */
		bool cmd_set_max_train_time();
		/** set SVM MKL enabled */
		bool cmd_set_svm_mkl_enabled();
		/** set SVM shrinking enabled */
		bool cmd_set_svm_shrinking_enabled();
		/** set SVM batch computation enabled */
		bool cmd_set_svm_batch_computation_enabled();
		/** set SVM linadd enabled */
		bool cmd_set_svm_linadd_enabled();
		/** set SVM bias enabled */
		bool cmd_set_svm_bias_enabled();
		/** set MKL intebias enabled */
		bool cmd_set_mkl_interleaved_enabled();
		/** set krr tau */
		bool cmd_set_krr_tau();

		/** add preproc */
		bool cmd_add_preproc();
		/** delete preproc */
		bool cmd_del_preproc();
		/** attach preproc to test/train */
		bool cmd_attach_preproc();
		/** clear/clean preproc */
		bool cmd_clean_preproc();

		/** create converter */
		bool cmd_set_converter();
		/** apply converter */
		bool cmd_apply_converter();
		/** embed features */
		bool cmd_embed();

		/** create new HMM */
		bool cmd_new_hmm();
		/** load HMM from file */
		bool cmd_load_hmm();
		/** save HMM to file */
		bool cmd_save_hmm();
		/** HMM classify */
		bool cmd_hmm_classify();
		/** HMM classify for a single example */
		bool cmd_hmm_classify_example();
		/** LinearHMM classify for 1-class examples */
		bool cmd_one_class_linear_hmm_classify();
		/** HMM classify for 1-class examples */
		bool cmd_one_class_hmm_classify();
		/** HMM classify for a single 1-class example */
		bool cmd_one_class_hmm_classify_example();
		/** output HMM */
		bool cmd_output_hmm();
		/** output HMM defined */
		bool cmd_output_hmm_defined();
		/** get HMM likelihood */
		bool cmd_hmm_likelihood();
		/** likelihood */
		bool cmd_likelihood();
		/** save HMM likelihoods to file */
		bool cmd_save_likelihood();
		/** get HMM's Viterbi Path */
		bool cmd_get_viterbi_path();
		/** train viterbi defined */
		bool cmd_viterbi_train_defined();
		/** train viterbi */
		bool cmd_viterbi_train();
		/** train baum welch */
		bool cmd_baum_welch_train();
		/** train defined baum welch */
		bool cmd_baum_welch_train_defined();
		/** train baum welch trans */
		bool cmd_baum_welch_trans_train();
		/** linear train */
		bool cmd_linear_train();
		/** save path to file */
		bool cmd_save_path();
		/** append HMM */
		bool cmd_append_hmm();
		/** append model (like HMM, but for CmdlineInterface */
		bool cmd_append_model();
		/** set HMM */
		bool cmd_set_hmm();
		/** set HMM as */
		bool cmd_set_hmm_as();
		/** get HMM */
		bool cmd_get_hmm();
		/** set chop value */
		bool cmd_set_chop();
		/** set pseudo value */
		bool cmd_set_pseudo();
		/** load definitions from file */
		bool cmd_load_definitions();
		/** convergence criteria */
		bool cmd_convergence_criteria();
		/** normalize HMM */
		bool cmd_normalize();
		/** add HMM states */
		bool cmd_add_states();
		/** permutation entropy */
		bool cmd_permutation_entropy();
		/** compute HMM relative entropy */
		bool cmd_relative_entropy();
		/** compute HMM entropy */
		bool cmd_entropy();
		/** create new plugin estimator */
		bool cmd_new_plugin_estimator();
		/** train plugin estimator */
		bool cmd_train_estimator();
		/** plugin estimate classify one example */
		bool cmd_plugin_estimate_classify_example();
		/** plugin estimate classify */
		bool cmd_plugin_estimate_classify();
		/** set plugin estimate */
		bool cmd_set_plugin_estimate();
		/** get plugin estimate */
		bool cmd_get_plugin_estimate();
		/** best path */
		bool cmd_best_path();
		/** best path 2struct */
		bool cmd_best_path_2struct();
		/**
		 * -assemble plif struct from a bunch of
		 *  arrays of the same length corresponding
		 *  to the fields of the plif-struct-array
		 */
		bool cmd_set_plif_struct();
		/**
		 * -get plif struct as a bunch
		 *  of arrays of the same length
		 * -each array corresponding to one
		 *  field of the struct
		 */
		bool cmd_get_plif_struct();
		/**
		 * precompute subkernels of a combined kernel
		 */
		bool cmd_precompute_subkernels();
		/** cmd signals model
		 *
		 */
		bool cmd_signals_set_model() { return false; };
		/** cmd signals set position
		 *
		 */
		bool cmd_signals_set_positions();
		/** cmd signals set labels
		 *
		 */
		bool cmd_signals_set_labels();
		/** cmd signals set split
		 *
		 */
		bool cmd_signals_set_split();
		/** cmd signals set train mask
		 *
		 */
		bool cmd_signals_set_train_mask();
		/** cmd signals add feature
		 *
		 */
		bool cmd_signals_add_feature();
		/** cmd signals add kernel
		 *
		 */
		bool cmd_signals_add_kernel();
		/** cmd signals run
		 *
		 */
		bool cmd_signals_run();
		/**
		 * -precompute content svms
		 *  and save the outputs
		 *  in a matrix with dim nof contents times
		 *  nof feature positions
		 *
		 * -the SVM score for a specific segment can be
		 *  calculated by subtraction the
		 *  the start and end position entries
		 *  from the row corresponding to the segment
		 *  type
		 */
		bool cmd_precompute_content_svms();
		/**
		 * -get lin feat
		 */
		bool cmd_get_lin_feat();
		/**
		 * -set lin feat
		 */
		bool cmd_set_lin_feat();

		/**
		 * -init dyn prog
		 */
		bool cmd_init_dyn_prog();

		/**
		 * clean up dyn prog
		 */
		bool cmd_clean_up_dyn_prog();

		/**
		 * initialize list of introns as DynProg features
		 */
		bool cmd_init_intron_list();

		/** settings for long transition approximation*/
		bool cmd_long_transition_settings();

		/**
		 * -precompute tiling features
		 *  and save the outputs (# content types x
		 *  # features positons) in a member variable
		 *  of the DynProg Object
		 * -the tiling intensities are transformed
		 *  and then stored as cumulative scores
		 */
		bool cmd_precompute_tiling_features();
		/**
		 * -compute the matrix that links
		 *  the plif ids to the transitions
		 *
		 * -the matrix has dimensions nof states
		 *  times nof states times nof feature types
		 *
		 * - feature types are for example
		 *   signal features, length features,
		 *   content features and tiling features
		 */
		bool cmd_set_model();
		/**
		 * set sparse feature matrix and
		 * all feature positions
		 */
		bool cmd_set_feature_matrix_sparse();
		/**
		 * set feature matrix and
		 * all feature positions
		 */
		bool cmd_set_feature_matrix();
		/** best path trans */
		bool cmd_best_path_trans();
		/** best path trans deriv */
		bool cmd_best_path_trans_deriv();
		/** best path no b */
		bool cmd_best_path_no_b();
		/** best path no b trans */
		bool cmd_best_path_no_b_trans();

		/** calculate CRC sum */
		bool cmd_crc();
		/** send command to operating system */
		bool cmd_system();
		/** exit/quit shogun/interface */
		bool cmd_exit();
		/** execute script from file */
		bool cmd_exec();
		/** set output target */
		bool cmd_set_output();
		/** set threshold */
		bool cmd_set_threshold();
		/** initialize random number generator */
		bool cmd_init_random();
		/** set number of threads */
		bool cmd_set_num_threads();
		/** translate string */
		bool cmd_translate_string();
		/** clear Shogun */
		bool cmd_clear();
		/** start timer */
		bool cmd_tic();
		/** stop timer */
		bool cmd_toc();
		/** echo */
		bool cmd_echo();
		/** print a message (for e.g. cmdline interface) */
		bool cmd_print();
		/** set loglevel */
		bool cmd_loglevel();
		/** set progress */
		bool cmd_progress();
		/** en/disable syntax hilighting */
		bool cmd_syntax_highlight();
		/** get version */
		bool cmd_get_version();
		/** issue help message */
		bool cmd_help();
		/** list alloc'd memory */
		bool cmd_whos();
		/** wrapper for compatibility send_command */
		bool cmd_send_command();
		/** execute code under python from octave,... */
		virtual bool cmd_run_python();
		/** execute code under octave from python,... */
		virtual bool cmd_run_octave();
		/** execute code under r from python,... */
		virtual bool cmd_run_r();
		/** call pr_loqo solver */
		virtual bool cmd_pr_loqo();

		/** get functions - to pass data from the target interface to shogun */

		/// get type of current argument (does not increment argument counter)
		virtual IFType get_argument_type()=0;

		/** get int */
		virtual int32_t get_int()=0;
		/** get real */
		virtual float64_t get_real()=0;
		/** get bool */
		virtual bool get_bool()=0;

		/** get string
		 * @param len
		 */
		virtual char* get_string(int32_t& len)=0;

		/** get vector
		 * @param vector
		 * @param len
		 */
		virtual void get_vector(bool*& vector, int32_t& len);
		/** get vector
		 * @param vector
		 * @param len
		 */
		virtual void get_vector(uint8_t*& vector, int32_t& len)=0;
		/** get vector
		 * @param vector
		 * @param len
		 */
		virtual void get_vector(char*& vector, int32_t& len)=0;
		/** get vector
		 * @param vector
		 * @param len
		 */
		virtual void get_vector(int32_t*& vector, int32_t& len)=0;
		/** get vector
		 * @param vector
		 * @param len
		 */
		virtual void get_vector(float64_t*& vector, int32_t& len)=0;
		/** get vector
		 * @param vector
		 * @param len
		 */
		virtual void get_vector(float32_t*& vector, int32_t& len)=0;
		/** get vector
		 * @param vector
		 * @param len
		 */
		virtual void get_vector(int16_t*& vector, int32_t& len)=0;
		/** get vector
		 * @param vector
		 * @param len
		 */
		virtual void get_vector(uint16_t*& vector, int32_t& len)=0;

		/** get matrix
		 * @param matrix
		 * @param num_feat
		 * @param num_vec
		 */
		virtual void get_matrix(
			uint8_t*& matrix, int32_t& num_feat, int32_t& num_vec)=0;
		/** get matrix
		 * @param matrix
		 * @param num_feat
		 * @param num_vec
		 */
		virtual void get_matrix(
			char*& matrix, int32_t& num_feat, int32_t& num_vec)=0;
		/** get matrix
		 * @param matrix
		 * @param num_feat
		 * @param num_vec
		 */
		virtual void get_matrix(
			int32_t*& matrix, int32_t& num_feat, int32_t& num_vec)=0;
		/** get matrix
		 * @param matrix
		 * @param num_feat
		 * @param num_vec
		 */
		virtual void get_matrix(
			float32_t*& matrix, int32_t& num_feat, int32_t& num_vec)=0;
		/** get matrix
		 * @param matrix
		 * @param num_feat
		 * @param num_vec
		 */
		virtual void get_matrix(
			float64_t*& matrix, int32_t& num_feat, int32_t& num_vec)=0;
		/** get matrix
		 * @param matrix
		 * @param num_feat
		 * @param num_vec
		 */
		virtual void get_matrix(
			int16_t*& matrix, int32_t& num_feat, int32_t& num_vec)=0;
		/** get matrix
		 * @param matrix
		 * @param num_feat
		 * @param num_vec
		 */
		virtual void get_matrix(
			uint16_t*& matrix, int32_t& num_feat, int32_t& num_vec)=0;

		/** get nd array
		 * @param array
		 * @param dims
		 * @param num_dims
		 */
		virtual void get_ndarray(
			uint8_t*& array, int32_t*& dims, int32_t& num_dims)=0;
		/** get nd array
		 * @param array
		 * @param dims
		 * @param num_dims
		 */
		virtual void get_ndarray(
			char*& array, int32_t*& dims, int32_t& num_dims)=0;
		/** get nd array
		 * @param array
		 * @param dims
		 * @param num_dims
		 */
		virtual void get_ndarray(
			int32_t*& array, int32_t*& dims, int32_t& num_dims)=0;
		/** get nd array
		 * @param array
		 * @param dims
		 * @param num_dims
		 */
		virtual void get_ndarray(
			float32_t*& array, int32_t*& dims, int32_t& num_dims)=0;
		/** get nd array
		 * @param array
		 * @param dims
		 * @param num_dims
		 */
		virtual void get_ndarray(
			float64_t*& array, int32_t*& dims, int32_t& num_dims)=0;
		/** get nd array
		 * @param array
		 * @param dims
		 * @param num_dims
		 */
		virtual void get_ndarray(
			int16_t*& array, int32_t*& dims, int32_t& num_dims)=0;
		/** get nd array
		 * @param array
		 * @param dims
		 * @param num_dims
		 */
		virtual void get_ndarray(
			uint16_t*& array, int32_t*& dims, int32_t& num_dims)=0;

		/** get sparse matrix
		 * @param matrix
		 * @param num_feat
		 * @param num_vec
		 */
		virtual void get_sparse_matrix(
			SGSparseVector<float64_t>*& matrix, int32_t& num_feat, int32_t& num_vec)=0;

		/*  future versions might support types other than float64_t

		virtual void get_sparse_matrix(SGSparseVector<uint8_t>*& matrix, int32_t& num_feat, int32_t& num_vec)=0;
		virtual void get_sparse_matrix(SGSparseVector<char>*& matrix, int32_t& num_feat, int32_t& num_vec)=0;
		virtual void get_sparse_matrix(SGSparseVector<int32_t>*& matrix, int32_t& num_feat, int32_t& num_vec)=0;
		virtual void get_sparse_matrix(SGSparseVector<float32_t>*& matrix, int32_t& num_feat, int32_t& num_vec)=0;
		virtual void get_sparse_matrix(SGSparseVector<int16_t>*& matrix, int32_t& num_feat, int32_t& num_vec)=0;
		virtual void get_sparse_matrix(SGSparseVector<uint16_t>*& matrix, int32_t& num_feat, int32_t& num_vec)=0; */

		/** get string list
		 * @param strings
		 * @param num_str
		 * @param max_string_len
		 */
		virtual void get_string_list(
			SGString<uint8_t>*& strings, int32_t& num_str,
			int32_t& max_string_len)=0;
		/** get string list
		 * @param strings
		 * @param num_str
		 * @param max_string_len
		 */
		virtual void get_string_list(
			SGString<char>*& strings, int32_t& num_str,
			int32_t& max_string_len)=0;
		/** get string list
		 * @param strings
		 * @param num_str
		 * @param max_string_len
		 */
		virtual void get_string_list(
			SGString<int32_t>*& strings, int32_t& num_str,
			int32_t& max_string_len)=0;
		/** get string list
		 * @param strings
		 * @param num_str
		 * @param max_string_len
		 */
		virtual void get_string_list(
			SGString<int16_t>*& strings, int32_t& num_str,
			int32_t& max_string_len)=0;
		/** get string list
		 * @param strings
		 * @param num_str
		 * @param max_string_len
		 */
		virtual void get_string_list(
			SGString<uint16_t>*& strings, int32_t& num_str,
			int32_t& max_string_len)=0;

		/** get attribute struct
		 * @param attrs
		 */
		virtual void get_attribute_struct(
			const CDynamicArray<T_ATTRIBUTE>* &attrs)=0;

		// set functions - to pass data from shogun to the target interface
		/** create return values
		 * @param num_val
		 */
		virtual bool create_return_values(int32_t num_val)=0;

		/** set int
		 * @param scalar
		 */
		virtual void set_int(int32_t scalar)=0;
		/** set real
		 * @param scalar
		 */
		virtual void set_real(float64_t scalar)=0;
		/** set bool
		 * @param scalar
		 */
		virtual void set_bool(bool scalar)=0;

		/** set vector
		 * @param vector
		 * @param len
		 */
		virtual void set_vector(const bool* vector, int32_t len);
		/** set vector
		 * @param vector
		 * @param len
		 */
		virtual void set_vector(const uint8_t* vector, int32_t len)=0;
		/** set vector
		 * @param vector
		 * @param len
		 */
		virtual void set_vector(const char* vector, int32_t len)=0;
		/** set vector
		 * @param vector
		 * @param len
		 */
		virtual void set_vector(const int32_t* vector, int32_t len)=0;
		/** set vector
		 * @param vector
		 * @param len
		 */
		virtual void set_vector(const float32_t* vector, int32_t len)=0;
		/** set vector
		 * @param vector
		 * @param len
		 */
		virtual void set_vector(const float64_t* vector, int32_t len)=0;
		/** set vector
		 * @param vector
		 * @param len
		 */
		virtual void set_vector(const int16_t* vector, int32_t len)=0;
		/** set vector
		 * @param vector
		 * @param len
		 */
		virtual void set_vector(const uint16_t* vector, int32_t len)=0;

		/** set matrix
		 * @param matrix
		 * @param num_feat
		 * @param num_vec
		 */
		virtual void set_matrix(
			const uint8_t* matrix, int32_t num_feat, int32_t num_vec)=0;
		/** set matrix
		 * @param matrix
		 * @param num_feat
		 * @param num_vec
		 */
		virtual void set_matrix(
			const char* matrix, int32_t num_feat, int32_t num_vec)=0;
		/** set matrix
		 * @param matrix
		 * @param num_feat
		 * @param num_vec
		 */
		virtual void set_matrix(
			const int32_t* matrix, int32_t num_feat, int32_t num_vec)=0;
		/** set matrix
		 * @param matrix
		 * @param num_feat
		 * @param num_vec
		 */
		virtual void set_matrix(
			const float32_t* matrix, int32_t num_feat, int32_t num_vec)=0;
		/** set matrix
		 * @param matrix
		 * @param num_feat
		 * @param num_vec
		 */
		virtual void set_matrix(
			const float64_t* matrix, int32_t num_feat, int32_t num_vec)=0;
		/** set matrix
		 * @param matrix
		 * @param num_feat
		 * @param num_vec
		 */
		virtual void set_matrix(
			const int16_t* matrix, int32_t num_feat, int32_t num_vec)=0;
		/** set matrix
		 * @param matrix
		 * @param num_feat
		 * @param num_vec
		 */
		virtual void set_matrix(
			const uint16_t* matrix, int32_t num_feat, int32_t num_vec)=0;

		/** set sparse matrix
		 * @param matrix
		 * @param num_feat
		 * @param num_vec
		 * @param nnz
		 */
		virtual void set_sparse_matrix(
			const SGSparseVector<float64_t>* matrix, int32_t num_feat,
			int32_t num_vec, int64_t nnz)=0;

		/*  future versions might support types other than float64_t

		virtual void set_sparse_matrix(const SGSparseVector<uint8_t>* matrix, int32_t num_feat, int32_t num_vec)=0;
		virtual void set_sparse_matrix(const SGSparseVector<char>* matrix, int32_t num_feat, int32_t num_vec)=0;
		virtual void set_sparse_matrix(const SGSparseVector<int32_t>* matrix, int32_t num_feat, int32_t num_vec)=0;
		virtual void set_sparse_matrix(const SGSparseVector<float32_t>* matrix, int32_t num_feat, int32_t num_vec)=0;
		virtual void set_sparse_matrix(const SGSparseVector<int16_t>* matrix, int32_t num_feat, int32_t num_vec)=0;
		virtual void set_sparse_matrix(const SGSparseVector<uint16_t>* matrix, int32_t num_feat, int32_t num_vec)=0; */

		/** set string list
		 * @param strings
		 * @param num_str
		 */
		virtual void set_string_list(
			const SGString<uint8_t>* strings, int32_t num_str)=0;
		/** set string list
		 * @param strings
		 * @param num_str
		 */
		virtual void set_string_list(
			const SGString<char>* strings, int32_t num_str)=0;
		/** set string list
		 * @param strings
		 * @param num_str
		 */
		virtual void set_string_list(
			const SGString<int32_t>* strings, int32_t num_str)=0;
		/** set string list
		 * @param strings
		 * @param num_str
		 */
		virtual void set_string_list(
			const SGString<int16_t>* strings, int32_t num_str)=0;
		/** set string list
		 * @param strings
		 * @param num_str
		 */
		virtual void set_string_list(
			const SGString<uint16_t>* strings, int32_t num_str)=0;

		/** set attribute struct
		 * @param attrs
		 */
		virtual void set_attribute_struct(
			const CDynamicArray<T_ATTRIBUTE>* attrs)=0;

		/// general interface handler
		bool handle();

		/// print the shogun prompt
		void print_prompt();

		/// return number of lhs args
		int32_t get_nlhs() { return m_nlhs; }

		/// return number of lhs args
		int32_t get_nrhs() { return m_nrhs; }


		// ui lib
		/** ui classifier */
		CGUIClassifier* ui_classifier;
		/** ui distance */
		CGUIDistance* ui_distance;
		/** ui features */
		CGUIFeatures* ui_features;
		/** ui hmm */
		CGUIHMM* ui_hmm;
		/** ui kernel */
		CGUIKernel* ui_kernel;
		/** ui labels */
		CGUILabels* ui_labels;
		/** ui math */
		CGUIMath* ui_math;
		/** ui pluginestimate */
		CGUIPluginEstimate* ui_pluginestimate;
		/** ui preproc */
		CGUIPreprocessor* ui_preproc;
		/** ui time */
		CGUITime* ui_time;
		/** ui structure */
		CGUIStructure* ui_structure;
		/** ui_converter */
		CGUIConverter* ui_converter;

	protected:
		/** return true if str starts with cmd
		 *
		 * @param str string to look in, not necessarily 0-terminated
		 * @param cmd 0-terminated string const
		 * @param len number of char to compare, length of cmd if not given
		 *
		 */
		static bool strmatch(const char* str, const char* cmd, int32_t len=-1)
		{
			if (len==-1)
			{
				len=strlen(cmd);
				if (strlen(str)!=(size_t) len) // match exact length
					return false;
			}

			return (strncmp(str, cmd, len)==0);
		}

		/** ends with
		 * @param str
		 * @param cmd
		 */
		static bool strendswith(const char* str, const char* cmd)
		{
			size_t idx=strlen(str);
			size_t len=strlen(cmd);

			if (strlen(str) < len)
				return false;

			str=&str[idx-len];

			return (strncmp(str, cmd, len)==0);
		}
		/// get command name like 'get_svm', 'new_hmm', etc.
		char* get_command(int32_t &len)
		{
			ASSERT(m_rhs_counter==0)
			if (m_nrhs<=0)
				SG_SERROR("No input arguments supplied.\n")

			return get_string(len);
		}
	private:
		/** helper function for computing objective */
		bool do_compute_objective(E_WHICH_OBJ obj);
		/** helper function for hmm classify */
		bool do_hmm_classify(bool linear=false, bool one_class=false);
		/** helper function for hmm classify on 1 example */
		bool do_hmm_classify_example(bool one_class=false);
		/** helper function for add/set features */
		bool do_set_features(bool add=false, bool check_dot=false, int32_t repetitions=1);

		/** perform bit embedding */
		void convert_to_bitembedding(CFeatures* &features, bool convert_to_word, bool convert_to_ulong);
		/** obtain from single string */
		void obtain_from_single_string(CFeatures* features);
		/** obtain from position list */
		bool obtain_from_position_list(CFeatures* features);
		/** obtain by sliding window */
		bool obtain_by_sliding_window(CFeatures* features);
		/** helper function to create a kernel */
		CKernel* create_kernel();

		/** helper function to create certain string features */
		CFeatures* create_custom_string_features(CStringFeatures<uint8_t>* f);

		CFeatures* create_custom_real_features(CDenseFeatures<float64_t>* orig_feat);
		/** legacy-related stuff - anybody got a better idea? */
		char* get_str_from_str_or_direct(int32_t& len);
		int32_t get_int_from_int_or_str();
		float64_t get_real_from_real_or_str();
		bool get_bool_from_bool_or_str();
		void get_vector_from_int_vector_or_str(
			int32_t*& vector, int32_t& len);
		void get_vector_from_real_vector_or_str(
			float64_t*& vector, int32_t& len);
		int32_t get_vector_len_from_str(int32_t expected_len=0);
		char* get_str_from_str(int32_t& len);
		int32_t get_num_args_in_str();

		/// get line from user/stdin/file input
		/// @return true at EOF
		char* get_line(FILE* infile=stdin, bool show_prompt=true);

	protected:
		/** lhs counter */
		int32_t m_lhs_counter;
		/** rhs counter */
		int32_t m_rhs_counter;
		/** nlhs */
		int32_t m_nlhs;
		/** nrhs */
		int32_t m_nrhs;

		// related to cmd_exec and cmd_echo
		/** file out */
		FILE* file_out;
		/** input */
		char input[10000];
		/** echo */
		bool echo;

		/** legacy strptr */
		char* m_legacy_strptr;
};

/** sg interface ptr typedef */
typedef bool (CSGInterface::*CSGInterfacePtr)();

#ifndef DOXYGEN_SHOULD_SKIP_THIS
/** interface method */
typedef struct {
	/// command
	const char* command;
	/// method
	CSGInterfacePtr method;
	/// usage prefix
	const char* usage_prefix;
	/// usage suffix
	const char* usage_suffix;
} CSGInterfaceMethod;
}
#endif

#endif // __SGINTERFACE__H_
