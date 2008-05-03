#ifndef __SGINTERFACE__H_
#define __SGINTERFACE__H_

#include "lib/config.h"

#if !defined(HAVE_SWIG)

#include "lib/common.h"
#include "base/SGObject.h"
#include "features/StringFeatures.h"
#include "features/SparseFeatures.h"
#include "kernel/Kernel.h"

#include "guilib/GUIClassifier.h"
#include "guilib/GUIDistance.h"
#include "guilib/GUIFeatures.h"
#include "guilib/GUIHMM.h"
#include "guilib/GUIKNN.h"
#include "guilib/GUIKernel.h"
#include "guilib/GUILabels.h"
#include "guilib/GUIMath.h"
#include "guilib/GUIPluginEstimate.h"
#include "guilib/GUIPreProc.h"
#include "guilib/GUITime.h"

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

		/// reset to clean state
		virtual void reset();

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
		/** set features */
		bool cmd_set_features();
		/** set reference features */
		bool cmd_set_reference_features();
		/** convert features */
		bool cmd_convert();
		/** obtain from position list */
		bool cmd_obtain_from_position_list();
		/** obtain by sliding window */
		bool cmd_obtain_by_sliding_window();
		/** reshape features */
		bool cmd_reshape();
		/** load labels from file */
		bool cmd_load_labels();
		/** set labels */
		bool cmd_set_labels();
		/** get labels */
		bool cmd_get_labels();

		/** set kernel */
		bool cmd_set_kernel();
		/** add kernel (to e.g. CombinedKernel) */
		bool cmd_add_kernel();
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
		/** set custom kernel */
		bool cmd_set_custom_kernel();
		/** set WD position weights */
		bool cmd_set_WD_position_weights();
		/** get subkernel weights */
		bool cmd_get_subkernel_weights();
		/** set subkernel weights */
		bool cmd_set_subkernel_weights();
		/** set subkernel weights combined */
		bool cmd_set_subkernel_weights_combined();
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
		/** set kernel optimization type */
		bool cmd_set_kernel_optimization_type();
#ifdef USE_SVMLIGHT
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
		/** get SVM */
		bool cmd_get_svm();
		/** set SVM */
		bool cmd_set_svm();
		/** classify */
		bool cmd_classify();
		/** classify example */
		bool cmd_classify_example();
		/** get classifier */
		bool cmd_get_classifier();
		/** get SVM objective */
		bool cmd_get_svm_objective();
		/** train classifier/SVM */
		bool cmd_train_classifier();
		/** test SVM */
		bool cmd_test_svm();
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
		/** set SVM epsilon */
		bool cmd_set_svm_epsilon();
		/** set SVR tube epsilon */
		bool cmd_set_svr_tube_epsilon();
		/** set SVM OneClass nu */
		bool cmd_set_svm_one_class_nu();
		/** set SVM MKL parameters */
		bool cmd_set_svm_mkl_parameters();
		/** set max train time */
		bool cmd_set_max_train_time();
		/** set SVM precompute enabled */
		bool cmd_set_svm_precompute_enabled();
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

		/** add preproc */
		bool cmd_add_preproc();
		/** delete preproc */
		bool cmd_del_preproc();
		/** load preproc from file */
		bool cmd_load_preproc();
		/** save preproc to file */
		bool cmd_save_preproc();
		/** attach preproc to test/train */
		bool cmd_attach_preproc();
		/** clear/clean preproc */
		bool cmd_clean_preproc();

		/** create new HMM */
		bool cmd_new_hmm();
		/** load HMM from file */
		bool cmd_load_hmm();
		/** save HMM to file */
		bool cmd_save_hmm();
		/** HMM classify */
		bool cmd_hmm_classify();
		/** HMM test */
		bool cmd_hmm_test();
		/** HMM classify for a single example */
		bool cmd_hmm_classify_example();
		/** LinearHMM classify for 1-class examples */
		bool cmd_one_class_linear_hmm_classify();
		/** HMM classify for 1-class examples */
		bool cmd_one_class_hmm_classify();
		/** One Class HMM test */
		bool cmd_one_class_hmm_test();
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
		/** train baum welch trans */
		bool cmd_baum_welch_trans_train();
		/** linear train */
		bool cmd_linear_train();
		/** save path to file */
		bool cmd_save_path();
		/** append HMM */
		bool cmd_append_hmm();
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
		/** test plugin estimator */
		bool cmd_test_estimator();
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
		/** best path trans */
		bool cmd_best_path_trans();
		/** best path trans deriv */
		bool cmd_best_path_trans_deriv();
		/** best path no b */
		bool cmd_best_path_no_b();
		/** best path trans simple */
		bool cmd_best_path_trans_simple();
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
		/** set loglevel */
		bool cmd_loglevel();
		/** get version */
		bool cmd_get_version();
		/** issue help message */
		bool cmd_help();
		/** wrapper for compatibility send_command */
		bool cmd_send_command();

		/** get functions - to pass data from the target interface to shogun */

		/// get type of current argument (does not increment argument counter)
		virtual IFType get_argument_type()=0;

		virtual INT get_int()=0;
		virtual DREAL get_real()=0;
		virtual bool get_bool()=0;

		virtual CHAR* get_string(INT& len)=0;

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

		/// print the shogun prompt
		void print_prompt();

		/** ui lib */
		CGUIClassifier* ui_classifier;
		CGUIDistance* ui_distance;
		CGUIFeatures* ui_features;
		CGUIHMM* ui_hmm;
		CGUIKernel* ui_kernel;
		CGUIKNN* ui_knn;
		CGUILabels* ui_labels;
		CGUIMath* ui_math;
		CGUIPluginEstimate* ui_pluginestimate;
		CGUIPreProc* ui_preproc;
		CGUITime* ui_time;

	protected:
		/* return true if str starts with cmd
		 *
		 * @param str string to look in, not necessarily 0-terminated
		 * @param cmd 0-terminated string const
		 * @param len number of CHAR to compare, length of cmd if not given
		 *
		 */
		static bool strmatch(CHAR* str, const CHAR* cmd, INT len=-1)
		{
			if (len==-1)
			{
				len=strlen(cmd);
				if (strlen(str)!=(size_t) len) // match exact length
					return false;
			}

			return (strncmp(str, cmd, len)==0);
		}

		/// get command name like 'get_svm', 'new_hmm', etc.
		CHAR* get_command(INT &len)
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
		/** helper function to create a kernel */
		CKernel* create_kernel();

		/** legacy-related stuff - anybody got a better idea? */
		CHAR* get_str_from_str_or_direct(INT& len);
		INT get_int_from_int_or_str();
		DREAL get_real_from_real_or_str();
		bool get_bool_from_bool_or_str();
		void get_int_vector_from_int_vector_or_str(INT*& vector, INT& len);
		void get_real_vector_from_real_vector_or_str(DREAL*& vector, INT& len);
		INT get_vector_len_from_str(INT expected_len=0);
		CHAR* get_str_from_str(INT& len);
		INT get_num_args_in_str();

		/// get line from user/stdin/file input
		/// @return true at EOF
		CHAR* get_line(FILE* infile=stdin, bool show_prompt=true);

	protected:
		INT m_lhs_counter;
		INT m_rhs_counter;
		INT m_nlhs;
		INT m_nrhs;

		// related to cmd_exec and cmd_echo
		FILE* file_out;
		CHAR input[10000];
		bool echo;

		CHAR* m_legacy_strptr;
};

typedef bool (CSGInterface::*CSGInterfacePtr)();

typedef struct {
	CHAR* command;
	CSGInterfacePtr method;
	CHAR* usage;
} CSGInterfaceMethod;

#endif // !HAVE_SWIG
#endif // __SGINTERFACE__H_
