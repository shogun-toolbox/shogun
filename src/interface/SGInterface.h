#ifndef __SGINTERFACE__H_
#define __SGINTERFACE__H_

#include "lib/config.h"

#if !defined(HAVE_SWIG)

#include "lib/common.h"
#include "base/SGObject.h"
#include "features/StringFeatures.h"
#include "features/SparseFeatures.h"

enum IFType
{
	UNDEFINED,

	DENSE_INT,
	DENSE_REAL,
	DENSE_SHORT,
	DENSE_SHORTREAL,
	DENSE_WORD,

	SPARSE_BYTE,
	SPARSE_CHAR,
	SPARSE_INT,
	SPARSE_REAL,
	SPARSE_SHORT,
	SPARSE_SHORTREAL,
	SPARSE_WORD,

	STRING_BYTE,
	STRING_CHAR,
	STRING_INT,
	STRING_SHORT,
	STRING_WORD,
};

class CSGInterface : public CSGObject
{

	public:
		CSGInterface();
		~CSGInterface();

		/* actions */
		/** load features from file */
		bool a_load_features();
		/** save features to file */
		bool a_save_features();
		/** clear/clean features */
		bool a_clean_features();
		/** get features */
		bool a_get_features();
		/** add features */
		bool a_add_features();
		/** set features */
		bool a_set_features();
		/** set reference features */
		bool a_set_reference_features();
		/** convert features */
		bool a_convert();
		/** obtain from position list */
		bool a_obtain_from_position_list();
		/** obtain by sliding window */
		bool a_obtain_by_sliding_window();
		/** reshape features */
		bool a_reshape();
		/** load labels from file */
		bool a_load_labels();
		/** set labels */
		bool a_set_labels();
		/** get labels */
		bool a_get_labels();

		/** set kernel */
		bool a_set_kernel();
		/** add kernel (to e.g. CombinedKernel) */
		bool a_add_kernel();
		/** initialize kernel */
		bool a_init_kernel();
		/** clear/clean kernel */
		bool a_clean_kernel();
		/** save kernel to file */
		bool a_save_kernel();
		/** load kernel init from file */
		bool a_load_kernel_init();
		/** save kernel init to file */
		bool a_save_kernel_init();
		/** get kernel matrix */
		bool a_get_kernel_matrix();
		/** set custom kernel */
		bool a_set_custom_kernel();
		/** set WD position weights */
		bool a_set_WD_position_weights();
		/** get subkernel weights */
		bool a_get_subkernel_weights();
		/** set subkernel weights */
		bool a_set_subkernel_weights();
		/** set subkernel weights combined */
		bool a_set_subkernel_weights_combined();
		/** set last subkernel weights */
		bool a_set_last_subkernel_weights();
		/** get WD position weights */
		bool a_get_WD_position_weights();
		/** get last subkernel weights */
		bool a_get_last_subkernel_weights();
		/** compute by subkernels */
		bool a_compute_by_subkernels();
		/** initialize kernel optimization */
		bool a_init_kernel_optimization();
		/** get kernel optimization */
		bool a_get_kernel_optimization();
		/** delete kernel optimization */
		bool a_delete_kernel_optimization();
		/** set kernel optimization type */
		bool a_set_kernel_optimization_type();
#ifdef USE_SVMLIGHT
		bool a_resize_kernel_cache();
#endif //USE_SVMLIGHT


		/** set distance */
		bool a_set_distance();
		/** init distance */
		bool a_init_distance();
		/** get distance matrix */
		bool a_get_distance_matrix();

		/** get SPEC consensus */
		bool a_get_SPEC_consensus();
		/** get SPEC scoring */
		bool a_get_SPEC_scoring();
		/** get WD consensus */
		bool a_get_WD_consensus();
		/** compute POIM WD */
		bool a_compute_POIM_WD();
		/** get WD scoring */
		bool a_get_WD_scoring();

		/** create new SVM/classifier */
		bool a_new_classifier();
		/** load SVM/classifier */
		bool a_load_classifier();
		/** get SVM */
		bool a_get_svm();
		/** set SVM */
		bool a_set_svm();
		/** classify */
		bool a_classify();
		/** classify example */
		bool a_classify_example();
		/** get classifier */
		bool a_get_classifier();
		/** get SVM objective */
		bool a_get_svm_objective();
		/** do AUC maximization */
		bool a_do_auc_maximization();
		/** set perceptron parameters */
		bool a_set_perceptron_parameters();
		/** train classifier/SVM */
		bool a_train_classifier();
		/** test SVM */
		bool a_test_svm();
		/** set SVM qpsize */
		bool a_set_svm_qpsize();
		/** set SVM max qpsize */
		bool a_set_svm_max_qpsize();
		/** set SVM bufsize */
		bool a_set_svm_bufsize();
		/** set SVM C */
		bool a_set_svm_C();
		/** set SVM epsilon */
		bool a_set_svm_epsilon();
		/** set SVR tube epsilon */
		bool a_set_svr_tube_epsilon();
		/** set SVM OneClass nu */
		bool a_set_svm_one_class_nu();
		/** set SVM MKL parameters */
		bool a_set_svm_mkl_parameters();
		/** set max train time */
		bool a_set_max_train_time();
		/** set SVM precompute enabled */
		bool a_set_svm_precompute_enabled();
		/** set SVM MKL enabled */
		bool a_set_svm_mkl_enabled();
		/** set SVM shrinking enabled */
		bool a_set_svm_shrinking_enabled();
		/** set SVM batch computation enabled */
		bool a_set_svm_batch_computation_enabled();
		/** set SVM linadd enabled */
		bool a_set_svm_linadd_enabled();
		/** set SVM bias enabled */
		bool a_set_svm_bias_enabled();

		/** add preproc */
		bool a_add_preproc();
		/** delete preproc */
		bool a_del_preproc();
		/** load preproc from file */
		bool a_load_preproc();
		/** save preproc to file */
		bool a_save_preproc();
		/** attach preproc to test/train */
		bool a_attach_preproc();
		/** clear/clean preproc */
		bool a_clean_preproc();

		/** create new HMM */
		bool a_new_hmm();
		/** load HMM from file */
		bool a_load_hmm();
		/** save HMM to file */
		bool a_save_hmm();
		/** HMM classify */
		bool a_hmm_classify();
		/** HMM test */
		bool a_hmm_test();
		/** HMM classify for a single example */
		bool a_hmm_classify_example();
		/** LinearHMM classify for 1-class examples */
		bool a_one_class_linear_hmm_classify();
		/** HMM classify for 1-class examples */
		bool a_one_class_hmm_classify();
		/** One Class HMM test */
		bool a_one_class_hmm_test();
		/** HMM classify for a single 1-class example */
		bool a_one_class_hmm_classify_example();
		/** output HMM */
		bool a_output_hmm();
		/** output HMM defined */
		bool a_output_hmm_defined();
		/** get HMM likelihood */
		bool a_hmm_likelihood();
		/** likelihood */
		bool a_likelihood();
		/** save HMM likelihoods to file */
		bool a_save_likelihood();
		/** get HMM's Viterbi Path */
		bool a_get_viterbi_path();
		/** train viterbi defined */
		bool a_viterbi_train_defined();
		/** train viterbi */
		bool a_viterbi_train();
		/** train baum welch */
		bool a_baum_welch_train();
		/** train baum welch trans */
		bool a_baum_welch_trans_train();
		/** linear train */
		bool a_linear_train();
		/** save path to file */
		bool a_save_path();
		/** append HMM */
		bool a_append_hmm();
		/** set HMM */
		bool a_set_hmm();
		/** set HMM as */
		bool a_set_hmm_as();
		/** get HMM */
		bool a_get_hmm();
		/** set chop value */
		bool a_set_chop();
		/** set pseudo value */
		bool a_set_pseudo();
		/** load definitions from file */
		bool a_load_definitions();
		/** convergence criteria */
		bool a_convergence_criteria();
		/** normalize HMM */
		bool a_normalize();
		/** add HMM states */
		bool a_add_states();
		/** permutation entropy */
		bool a_permutation_entropy();
		/** compute HMM relative entropy */
		bool a_relative_entropy();
		/** compute HMM entropy */
		bool a_entropy();
		/** create new plugin estimator */
		bool a_new_plugin_estimator();
		/** train plugin estimator */
		bool a_train_estimator();
		/** test plugin estimator */
		bool a_test_estimator();
		/** plugin estimate classify one example */
		bool a_plugin_estimate_classify_example();
		/** plugin estimate classify */
		bool a_plugin_estimate_classify();
		/** set plugin estimate */
		bool a_set_plugin_estimate();
		/** get plugin estimate */
		bool a_get_plugin_estimate();
		/** best path */
		bool a_best_path();
		/** best path 2struct */
		bool a_best_path_2struct();
		/** best path trans */
		bool a_best_path_trans();
		/** best path trans deriv */
		bool a_best_path_trans_deriv();
		/** best path no b */
		bool a_best_path_no_b();
		/** best path trans simple */
		bool a_best_path_trans_simple();
		/** best path no b trans */
		bool a_best_path_no_b_trans();

		/** calculate CRC sum */
		bool a_crc();
		/** send command to operating system */
		bool a_system();
		/** exit/quit shogun/interface */
		bool a_exit();
		/** execute script from file */
		bool a_exec();
		/** set output target */
		bool a_set_output();
		/** set threshold */
		bool a_set_threshold();
		/** set number of threads */
		bool a_set_num_threads();
		/** translate string */
		bool a_translate_string();
		/** clear Shogun */
		bool a_clear();
		/** start timer */
		bool a_tic();
		/** stop timer */
		bool a_toc();
		/** echo */
		bool a_echo();
		/** set loglevel */
		bool a_loglevel();
		/** get version */
		bool a_get_version();
		/** issue help message */
		bool a_help();

		/** get functions - to pass data from the target interface to shogun */

		/// get type of current argument (does not increment argument counter)
		virtual IFType get_argument_type()=0;

		virtual INT get_int()=0;
		virtual DREAL get_real()=0;
		virtual bool get_bool()=0;

		virtual CHAR* get_string(INT& len)=0;
		virtual INT get_int_from_string();
		virtual DREAL get_real_from_string();
		virtual bool get_bool_from_string();

		virtual void get_byte_vector(BYTE*& vector, INT& len)=0;
		virtual void get_char_vector(CHAR*& vector, INT& len)=0;
		virtual void get_int_vector(INT*& vector, INT& len)=0;
		virtual void get_real_vector(DREAL*& vector, INT& len)=0;
		virtual void get_shortreal_vector(SHORTREAL*& vector, INT& len)=0;
		virtual void get_short_vector(SHORT*& vector, INT& len)=0;
		virtual void get_word_vector(WORD*& vector, INT& len)=0;


		virtual void get_byte_matrix(BYTE*& matrix, INT& num_feat, INT& num_vec)=0;
		virtual void get_char_matrix(CHAR*& matrix, INT& num_feat, INT& num_vec)=0;
		virtual void get_int_matrix(INT*& matrix, INT& num_feat, INT& num_vec)=0;
		virtual void get_shortreal_matrix(SHORTREAL*& matrix, INT& num_feat, INT& num_vec)=0;
		virtual void get_real_matrix(DREAL*& matrix, INT& num_feat, INT& num_vec)=0;
		virtual void get_short_matrix(SHORT*& matrix, INT& num_feat, INT& num_vec)=0;
		virtual void get_word_matrix(WORD*& matrix, INT& num_feat, INT& num_vec)=0;


		virtual void get_real_sparsematrix(TSparse<DREAL>*& matrix, INT& num_feat, INT& num_vec)=0;

		/*  future versions might support types other than DREAL
		
		virtual void get_byte_sparsematrix(TSparse<BYTE>*& matrix, INT& num_feat, INT& num_vec)=0;
		virtual void get_char_sparsematrix(TSparse<CHAR>*& matrix, INT& num_feat, INT& num_vec)=0;
		virtual void get_int_sparsematrix(TSparse<INT>*& matrix, INT& num_feat, INT& num_vec)=0;
		virtual void get_shortreal_sparsematrix(TSparse<SHORTREAL>*& matrix, INT& num_feat, INT& num_vec)=0;
		virtual void get_short_sparsematrix(TSparse<SHORT>*& matrix, INT& num_feat, INT& num_vec)=0;
		virtual void get_word_sparsematrix(TSparse<WORD>*& matrix, INT& num_feat, INT& num_vec)=0; */

		virtual void get_byte_string_list(T_STRING<BYTE>*& strings, INT& num_str, INT& max_string_len)=0;
		virtual void get_char_string_list(T_STRING<CHAR>*& strings, INT& num_str, INT& max_string_len)=0;
		virtual void get_int_string_list(T_STRING<INT>*& strings, INT& num_str, INT& max_string_len)=0;
		virtual void get_short_string_list(T_STRING<SHORT>*& strings, INT& num_str, INT& max_string_len)=0;
		virtual void get_word_string_list(T_STRING<WORD>*& strings, INT& num_str, INT& max_string_len)=0;


		/** set functions - to pass data from shogun to the target interface */
		virtual bool create_return_values(INT num_val)=0;

		virtual void set_int(INT scalar)=0;
		virtual void set_real(DREAL scalar)=0;
		virtual void set_bool(bool scalar)=0;

		virtual void set_byte_vector(const BYTE* vector, INT len)=0;
		virtual void set_char_vector(const CHAR* vector, INT len)=0;
		virtual void set_int_vector(const INT* vector, INT len)=0;
		virtual void set_shortreal_vector(const SHORTREAL* vector, INT len)=0;
		virtual void set_real_vector(const DREAL* vector, INT len)=0;
		virtual void set_short_vector(const SHORT* vector, INT len)=0;
		virtual void set_word_vector(const WORD* vector, INT len)=0;


		virtual void set_byte_matrix(const BYTE* matrix, INT num_feat, INT num_vec)=0;
		virtual void set_char_matrix(const CHAR* matrix, INT num_feat, INT num_vec)=0;
		virtual void set_int_matrix(const INT* matrix, INT num_feat, INT num_vec)=0;
		virtual void set_shortreal_matrix(const SHORTREAL* matrix, INT num_feat, INT num_vec)=0;
		virtual void set_real_matrix(const DREAL* matrix, INT num_feat, INT num_vec)=0;
		virtual void set_short_matrix(const SHORT* matrix, INT num_feat, INT num_vec)=0;
		virtual void set_word_matrix(const WORD* matrix, INT num_feat, INT num_vec)=0;

		virtual void set_real_sparsematrix(const TSparse<DREAL>* matrix, INT num_feat, INT num_vec, LONG nnz)=0;

		/*  future versions might support types other than DREAL
		
		virtual void set_byte_sparsematrix(const TSparse<BYTE>* matrix, INT num_feat, INT num_vec)=0;
		virtual void set_char_sparsematrix(const TSparse<CHAR>* matrix, INT num_feat, INT num_vec)=0;
		virtual void set_int_sparsematrix(const TSparse<INT>* matrix, INT num_feat, INT num_vec)=0;
		virtual void set_shortreal_sparsematrix(const TSparse<SHORTREAL>* matrix, INT num_feat, INT num_vec)=0;
		virtual void set_short_sparsematrix(const TSparse<SHORT>* matrix, INT num_feat, INT num_vec)=0;
		virtual void set_word_sparsematrix(const TSparse<WORD>* matrix, INT num_feat, INT num_vec)=0; */


		virtual void set_byte_string_list(const T_STRING<BYTE>* strings, INT num_str)=0;
		virtual void set_char_string_list(const T_STRING<CHAR>* strings, INT num_str)=0;
		virtual void set_int_string_list(const T_STRING<INT>* strings, INT num_str)=0;
		virtual void set_short_string_list(const T_STRING<SHORT>* strings, INT num_str)=0;
		virtual void set_word_string_list(const T_STRING<WORD>* strings, INT num_str)=0;

		/// general interface handler
		bool handle();

	protected:
		/// return true if str starts with cmd
		/// cmd is a 0 terminated string const
		/// str is a string of length len (not 0 terminated)
		static bool strmatch(CHAR* str, UINT len, const CHAR* cmd)
		{
			return (len==strlen(cmd)
					&& !strncmp(str, cmd, strlen(cmd)));
		}

		/// get action name like 'get_svm', 'new_hmm'
		CHAR* get_action(INT &len)
		{
			ASSERT(m_rhs_counter==0);
			if (m_nrhs<=0)
				SG_SERROR("No input arguments supplied.");

			return get_string(len);
		}

	private:
		/** helper function for hmm classify */
		bool do_hmm_classify(bool linear=false, bool one_class=false);
		/** helper function for hmm classify on 1 example */
		bool do_hmm_classify_example(bool one_class=false);
		/** helper function for add/set features */
		bool do_set_features(bool add=false);
		/** temp command to invoke send_command in old interface */
		bool send_command(const CHAR* cmd);

	protected:
		INT m_lhs_counter;
		INT m_rhs_counter;
		INT m_nlhs;
		INT m_nrhs;

};

typedef bool (CSGInterface::*CSGInterfacePtr)();

typedef struct {
	CHAR* action;
	CSGInterfacePtr method;
	CHAR* usage;
} CSGInterfaceMethod;

#endif // !HAVE_SWIG
#endif // __SGINTERFACE__H_
